"""
SilentAuth — Behavioral Authentication Backend
Flask + SQLAlchemy + scikit-learn IsolationForest
"""

import os
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from flask_cors import CORS
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.update(
    SQLALCHEMY_DATABASE_URI="sqlite:///silentauth.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    JWT_SECRET_KEY=os.environ.get("JWT_SECRET", "silentauth-hackathon-2025-secret"),
    JWT_ACCESS_TOKEN_EXPIRES=timedelta(hours=4),
)

db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("silentauth")

# ──────────────────────────────────────────
# Database Models
# ──────────────────────────────────────────

class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    balance       = db.Column(db.Float, default=245000.0)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    behavioral_samples = db.relationship("BehavioralSample", backref="user", lazy=True)
    transactions       = db.relationship("Transaction",       backref="user", lazy=True)
    auth_logs          = db.relationship("AuthLog",           backref="user", lazy=True)


class BehavioralSample(db.Model):
    """
    One row per behavioral event (login or continuous monitoring).
    Features extracted from raw keystroke/mouse telemetry.
    """
    __tablename__    = "behavioral_samples"
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    # Keystroke dynamics
    avg_dwell_time   = db.Column(db.Float)   # ms key held down
    avg_flight_time  = db.Column(db.Float)   # ms between keystrokes
    rhythm_variance  = db.Column(db.Float)   # std-dev of flight times
    typing_speed_cps = db.Column(db.Float)   # chars per second
    # Mouse dynamics
    avg_mouse_speed  = db.Column(db.Float)   # px/sec
    mouse_sample_n   = db.Column(db.Integer) # how many mouse events captured
    # Session context
    hour_of_day      = db.Column(db.Integer)
    day_of_week      = db.Column(db.Integer)
    # Derived
    anomaly_score    = db.Column(db.Float)   # raw IsolationForest score
    trust_score      = db.Column(db.Float)   # mapped 0-100
    source           = db.Column(db.String(20), default="login")  # login | session
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)


class UserMLModel(db.Model):
    """Serialised IsolationForest + StandardScaler per user."""
    __tablename__  = "user_ml_models"
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)
    model_blob     = db.Column(db.LargeBinary)   # pickled {clf, scaler}
    samples_used   = db.Column(db.Integer, default=0)
    last_trained   = db.Column(db.DateTime)


class Transaction(db.Model):
    __tablename__       = "transactions"
    id                  = db.Column(db.Integer, primary_key=True)
    user_id             = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    recipient           = db.Column(db.String(120))
    amount              = db.Column(db.Float)
    purpose             = db.Column(db.String(200))
    direction           = db.Column(db.String(6), default="debit")   # debit | credit
    trust_score_at_time = db.Column(db.Float)
    status              = db.Column(db.String(20), default="approved")  # approved|flagged|blocked
    risk_reason         = db.Column(db.String(200))
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)


class AuthLog(db.Model):
    __tablename__ = "auth_logs"
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    event         = db.Column(db.String(50))   # login_ok | transfer_approved | lockdown etc.
    trust_score   = db.Column(db.Float)
    ip_address    = db.Column(db.String(50))
    meta          = db.Column(db.Text)          # JSON blob for extra context
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)


# ──────────────────────────────────────────
# Behavioral Biometrics Engine
# ──────────────────────────────────────────

FEATURE_NAMES = [
    "avg_dwell_time",
    "avg_flight_time",
    "rhythm_variance",
    "typing_speed_cps",
    "avg_mouse_speed",
    "hour_of_day",
    "day_of_week",
]

# Sane defaults used when a field is missing (cold-start)
FEATURE_DEFAULTS = {
    "avg_dwell_time":   110.0,
    "avg_flight_time":  160.0,
    "rhythm_variance":   55.0,
    "typing_speed_cps":   4.5,
    "avg_mouse_speed":  250.0,
    "hour_of_day":       12.0,
    "day_of_week":        2.0,
}

MIN_TRAINING_SAMPLES = 5   # need this many logins before ML kicks in
RETRAIN_EVERY        = 3   # retrain after every N new samples


def extract_features(behavioral_data: dict) -> dict:
    """
    Convert raw browser telemetry → normalised feature dict.

    Expected input shape:
      {
        "keystrokes": [{"key": "a", "timestamp": 1718000000000, "duration": 95}, ...],
        "mouseEvents": [{"x": 320, "y": 240, "t": 1718000000000}, ...]
      }
    """
    keystrokes   = behavioral_data.get("keystrokes", [])
    mouse_events = behavioral_data.get("mouseEvents", [])
    features     = dict(FEATURE_DEFAULTS)  # start with safe defaults

    # ── Keystroke dynamics ──────────────────────────────────────────────
    if len(keystrokes) >= 3:
        dwell_times  = [k["duration"] for k in keystrokes if "duration" in k and 10 < k["duration"] < 1500]
        flight_times = []
        for i in range(1, len(keystrokes)):
            try:
                gap = keystrokes[i]["timestamp"] - keystrokes[i - 1]["timestamp"]
                if 10 < gap < 2000:
                    flight_times.append(gap)
            except (KeyError, TypeError):
                pass

        if dwell_times:
            features["avg_dwell_time"] = float(np.mean(dwell_times))
        if flight_times:
            features["avg_flight_time"] = float(np.mean(flight_times))
            features["rhythm_variance"]  = float(np.std(flight_times)) if len(flight_times) > 1 else 0.0

        # chars per second
        try:
            span_sec = (keystrokes[-1]["timestamp"] - keystrokes[0]["timestamp"]) / 1000.0
            if span_sec > 0.5:
                features["typing_speed_cps"] = len(keystrokes) / span_sec
        except (KeyError, IndexError):
            pass

    # ── Mouse dynamics ──────────────────────────────────────────────────
    if len(mouse_events) >= 5:
        speeds = []
        for i in range(1, len(mouse_events)):
            try:
                dx = mouse_events[i]["x"] - mouse_events[i - 1]["x"]
                dy = mouse_events[i]["y"] - mouse_events[i - 1]["y"]
                dt = (mouse_events[i]["t"]  - mouse_events[i - 1]["t"])  / 1000.0
                if 0.01 < dt < 0.5:
                    speeds.append(np.sqrt(dx * dx + dy * dy) / dt)
            except (KeyError, TypeError):
                pass
        if speeds:
            features["avg_mouse_speed"]  = float(np.mean(speeds))
            features["mouse_sample_n"]   = len(mouse_events)

    # ── Temporal context ────────────────────────────────────────────────
    now = datetime.utcnow()
    features["hour_of_day"] = float(now.hour)
    features["day_of_week"] = float(now.weekday())

    return features


def features_to_vector(f: dict) -> np.ndarray:
    return np.array([f.get(k, FEATURE_DEFAULTS[k]) for k in FEATURE_NAMES], dtype=float)


def train_model(samples: list) -> bytes:
    """
    Fit an IsolationForest on the user's historical feature vectors.
    Returns pickled bytes of {clf, scaler}.
    """
    X = np.array([
        features_to_vector({
            "avg_dwell_time":   s.avg_dwell_time,
            "avg_flight_time":  s.avg_flight_time,
            "rhythm_variance":  s.rhythm_variance,
            "typing_speed_cps": s.typing_speed_cps,
            "avg_mouse_speed":  s.avg_mouse_speed,
            "hour_of_day":      s.hour_of_day,
            "day_of_week":      s.day_of_week,
        })
        for s in samples
    ])
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    clf    = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
    clf.fit(X_s)
    return pickle.dumps({"clf": clf, "scaler": scaler})


def score_features(features: dict, ml_model: UserMLModel | None) -> tuple[float, str, float]:
    """
    Returns (trust_score_0_to_100, method_label, raw_anomaly_score).

    Method labels:
      "heuristic"  — < MIN_TRAINING_SAMPLES, rule-based fallback
      "ml_model"   — IsolationForest inference
    """
    vec = features_to_vector(features)

    if ml_model is None or ml_model.samples_used < MIN_TRAINING_SAMPLES:
        # ── Heuristic baseline ─────────────────────────────────────────
        score = 72.0
        # Penalise unusual hours (between midnight and 5 am)
        hour = features.get("hour_of_day", 12)
        if isinstance(hour, float):
            hour = int(hour)
        if 0 <= hour <= 5:
            score -= 12
        # Reward "normal" typing speed
        speed = features.get("typing_speed_cps", 4.5)
        if 2.0 < speed < 12.0:
            score += 8
        return round(min(max(score, 0), 100), 1), "heuristic", 0.0

    # ── ML inference ──────────────────────────────────────────────────
    payload = pickle.loads(ml_model.model_blob)
    clf, scaler = payload["clf"], payload["scaler"]
    X_s    = scaler.transform([vec])
    raw    = float(clf.decision_function(X_s)[0])  # typically -0.5 … +0.5
    # Map: raw=+0.5 → 100, raw=-0.5 → 0
    score  = (raw + 0.5) * 100.0
    score  = round(min(max(score, 0.0), 100.0), 1)
    return score, "ml_model", raw


def maybe_retrain(user_id: int):
    """Retrain user model if we have enough new data."""
    samples = (
        BehavioralSample.query
        .filter_by(user_id=user_id)
        .order_by(BehavioralSample.created_at.asc())
        .all()
    )
    n = len(samples)
    if n < MIN_TRAINING_SAMPLES:
        return
    ml_model = UserMLModel.query.filter_by(user_id=user_id).first()
    already_used = ml_model.samples_used if ml_model else 0
    if n - already_used >= RETRAIN_EVERY or already_used == 0:
        log.info(f"[ML] Retraining model for user {user_id} with {n} samples")
        blob = train_model(samples)
        if ml_model is None:
            ml_model = UserMLModel(user_id=user_id)
            db.session.add(ml_model)
        ml_model.model_blob   = blob
        ml_model.samples_used = n
        ml_model.last_trained = datetime.utcnow()
        db.session.commit()


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _add_auth_log(user_id, event, trust_score, meta=None):
    db.session.add(AuthLog(
        user_id=user_id,
        event=event,
        trust_score=trust_score,
        ip_address=request.remote_addr,
        meta=json.dumps(meta) if meta else None,
    ))


def _seed_transactions(user_id: int):
    seeds = [
        ("Salary Credit",    48000.0, "credit", "Monthly salary",       95.0),
        ("Amazon Purchase",   2400.0, "debit",  "E-commerce",            92.0),
        ("Netflix",            649.0, "debit",  "Subscription",          91.0),
        ("Electricity Bill",  3200.0, "debit",  "Utility",               90.0),
        ("Zomato",             850.0, "debit",  "Food delivery",         88.0),
    ]
    for recipient, amount, direction, purpose, ts in seeds:
        db.session.add(Transaction(
            user_id=user_id, recipient=recipient, amount=amount,
            direction=direction, purpose=purpose,
            trust_score_at_time=ts, status="approved",
        ))


# ──────────────────────────────────────────
# Routes — Auth
# ──────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    email    = data.get("email",    "").strip().lower()
    password = data.get("password", "")

    if not username or not email or not password:
        return jsonify({"error": "username, email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    user = User(username=username, email=email, password_hash=_hash(password))
    db.session.add(user)
    db.session.flush()       # get user.id before commit
    _seed_transactions(user.id)
    db.session.commit()

    token = create_access_token(identity=str(user.id))
    return jsonify({
        "token":    token,
        "user":     {"id": user.id, "username": user.username, "email": user.email, "balance": user.balance},
        "message":  "Account created",
    }), 201


@app.route("/api/login", methods=["POST"])
def login():
    data             = request.get_json(silent=True) or {}
    email            = data.get("email",    "").strip().lower()
    password         = data.get("password", "")
    behavioral_data  = data.get("behavioralData", {})

    user = User.query.filter_by(email=email, password_hash=_hash(password)).first()
    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    # ── Behavioral analysis ────────────────────────────────────────────
    features    = extract_features(behavioral_data)
    ml_model    = UserMLModel.query.filter_by(user_id=user.id).first()
    trust, method, raw = score_features(features, ml_model)

    # Persist sample
    sample = BehavioralSample(
        user_id          = user.id,
        avg_dwell_time   = features["avg_dwell_time"],
        avg_flight_time  = features["avg_flight_time"],
        rhythm_variance  = features["rhythm_variance"],
        typing_speed_cps = features["typing_speed_cps"],
        avg_mouse_speed  = features.get("avg_mouse_speed", FEATURE_DEFAULTS["avg_mouse_speed"]),
        mouse_sample_n   = int(features.get("mouse_sample_n", 0)),
        hour_of_day      = int(features["hour_of_day"]),
        day_of_week      = int(features["day_of_week"]),
        anomaly_score    = raw,
        trust_score      = trust,
        source           = "login",
    )
    db.session.add(sample)
    _add_auth_log(user.id, "login_ok", trust, {"method": method})
    db.session.commit()

    maybe_retrain(user.id)

    token = create_access_token(identity=str(user.id))
    sample_count = BehavioralSample.query.filter_by(user_id=user.id).count()

    return jsonify({
        "token":       token,
        "user":        {"id": user.id, "username": user.username, "email": user.email, "balance": user.balance},
        "trustScore":  trust,
        "trustMethod": method,
        "sampleCount": sample_count,
        "features": {
            "avgDwellTime":   round(features["avg_dwell_time"],   2),
            "avgFlightTime":  round(features["avg_flight_time"],  2),
            "rhythmVariance": round(features["rhythm_variance"],  2),
            "typingSpeedCps": round(features["typing_speed_cps"], 2),
        },
        "mlReady": sample_count >= MIN_TRAINING_SAMPLES,
    })


# ──────────────────────────────────────────
# Routes — Dashboard
# ──────────────────────────────────────────

@app.route("/api/dashboard", methods=["GET"])
@jwt_required()
def dashboard():
    user_id = int(get_jwt_identity())
    user    = User.query.get_or_404(user_id)

    txns = (
        Transaction.query.filter_by(user_id=user_id)
        .order_by(Transaction.created_at.desc())
        .limit(10).all()
    )
    logs = (
        AuthLog.query.filter_by(user_id=user_id)
        .order_by(AuthLog.created_at.desc())
        .limit(6).all()
    )
    latest_sample = (
        BehavioralSample.query.filter_by(user_id=user_id)
        .order_by(BehavioralSample.created_at.desc())
        .first()
    )
    sample_count = BehavioralSample.query.filter_by(user_id=user_id).count()
    ml_model     = UserMLModel.query.filter_by(user_id=user_id).first()

    # Average trust over last 10 samples
    recent_scores = (
        BehavioralSample.query.filter_by(user_id=user_id)
        .order_by(BehavioralSample.created_at.desc())
        .limit(10).all()
    )
    avg_trust = round(np.mean([s.trust_score for s in recent_scores]), 1) if recent_scores else 75.0

    return jsonify({
        "user": {
            "username": user.username,
            "email":    user.email,
            "balance":  user.balance,
        },
        "trust": {
            "current":    latest_sample.trust_score if latest_sample else avg_trust,
            "average":    avg_trust,
            "method":     "ml_model" if (ml_model and ml_model.samples_used >= MIN_TRAINING_SAMPLES) else "heuristic",
            "sampleCount": sample_count,
            "mlReady":    sample_count >= MIN_TRAINING_SAMPLES,
            "lastTrained": ml_model.last_trained.isoformat() if ml_model and ml_model.last_trained else None,
        },
        "latestFeatures": {
            "avgDwellTime":   round(latest_sample.avg_dwell_time,   2) if latest_sample else None,
            "avgFlightTime":  round(latest_sample.avg_flight_time,  2) if latest_sample else None,
            "rhythmVariance": round(latest_sample.rhythm_variance,  2) if latest_sample else None,
            "typingSpeedCps": round(latest_sample.typing_speed_cps, 2) if latest_sample else None,
        },
        "transactions": [
            {
                "id":        t.id,
                "recipient": t.recipient,
                "amount":    t.amount,
                "direction": t.direction,
                "purpose":   t.purpose,
                "status":    t.status,
                "trustScore": t.trust_score_at_time,
                "date":      t.created_at.strftime("%d %b %Y, %H:%M"),
            }
            for t in txns
        ],
        "authLogs": [
            {
                "event":      l.event,
                "trustScore": l.trust_score,
                "ip":         l.ip_address,
                "time":       l.created_at.strftime("%H:%M:%S"),
                "date":       l.created_at.strftime("%d %b"),
            }
            for l in logs
        ],
    })


# ──────────────────────────────────────────
# Routes — Continuous Monitoring
# ──────────────────────────────────────────

@app.route("/api/behavioral/ping", methods=["POST"])
@jwt_required()
def behavioral_ping():
    """
    Called every ~30 s from the dashboard to continuously re-score the session.
    Lightweight: does NOT retrain; just scores and logs.
    """
    user_id        = int(get_jwt_identity())
    data           = request.get_json(silent=True) or {}
    behavioral_data = data.get("behavioralData", {})

    features       = extract_features(behavioral_data)
    ml_model       = UserMLModel.query.filter_by(user_id=user_id).first()
    trust, method, raw = score_features(features, ml_model)

    sample = BehavioralSample(
        user_id          = user_id,
        avg_dwell_time   = features["avg_dwell_time"],
        avg_flight_time  = features["avg_flight_time"],
        rhythm_variance  = features["rhythm_variance"],
        typing_speed_cps = features["typing_speed_cps"],
        avg_mouse_speed  = features.get("avg_mouse_speed", FEATURE_DEFAULTS["avg_mouse_speed"]),
        mouse_sample_n   = int(features.get("mouse_sample_n", 0)),
        hour_of_day      = int(features["hour_of_day"]),
        day_of_week      = int(features["day_of_week"]),
        anomaly_score    = raw,
        trust_score      = trust,
        source           = "session",
    )
    db.session.add(sample)
    db.session.commit()

    maybe_retrain(user_id)

    return jsonify({
        "trustScore": trust,
        "method":     method,
        "features": {
            "avgDwellTime":   round(features["avg_dwell_time"],   2),
            "avgFlightTime":  round(features["avg_flight_time"],  2),
            "rhythmVariance": round(features["rhythm_variance"],  2),
            "typingSpeedCps": round(features["typing_speed_cps"], 2),
        },
    })


# ──────────────────────────────────────────
# Routes — Transfer
# ──────────────────────────────────────────

@app.route("/api/transfer", methods=["POST"])
@jwt_required()
def transfer():
    user_id = int(get_jwt_identity())
    data    = request.get_json(silent=True) or {}

    recipient   = data.get("recipient", "").strip()
    amount      = float(data.get("amount", 0))
    purpose     = data.get("purpose", "Transfer").strip()
    trust_score = float(data.get("trustScore", 75))

    if not recipient or amount <= 0:
        return jsonify({"error": "Recipient and a positive amount are required"}), 400

    user = User.query.get_or_404(user_id)

    if amount > user.balance:
        return jsonify({"error": "Insufficient balance"}), 400

    # ── Risk gate ──────────────────────────────────────────────────────
    status      = "approved"
    risk_reason = None

    if trust_score < 40:
        _add_auth_log(user_id, "transfer_blocked", trust_score,
                      {"reason": "trust_too_low", "amount": amount})
        db.session.commit()
        return jsonify({
            "error":       "Transfer blocked — trust score too low",
            "trustScore":  trust_score,
            "requiredMin": 40,
        }), 403

    if trust_score < 65:
        status      = "flagged"
        risk_reason = f"Low trust score ({trust_score})"

    if amount > 100_000 and trust_score < 85:
        _add_auth_log(user_id, "transfer_blocked", trust_score,
                      {"reason": "high_value_low_trust", "amount": amount})
        db.session.commit()
        return jsonify({
            "error":       "High-value transfer requires trust score ≥ 85",
            "trustScore":  trust_score,
            "requiredMin": 85,
        }), 403

    # ── Execute ────────────────────────────────────────────────────────
    user.balance -= amount
    txn = Transaction(
        user_id             = user_id,
        recipient           = recipient,
        amount              = amount,
        direction           = "debit",
        purpose             = purpose,
        trust_score_at_time = trust_score,
        status              = status,
        risk_reason         = risk_reason,
    )
    db.session.add(txn)
    _add_auth_log(user_id, f"transfer_{status}", trust_score,
                  {"recipient": recipient, "amount": amount})
    db.session.commit()

    return jsonify({
        "success":    True,
        "newBalance": user.balance,
        "transaction": {
            "id":        txn.id,
            "recipient": recipient,
            "amount":    amount,
            "status":    status,
            "riskReason": risk_reason,
        },
    })


# ──────────────────────────────────────────
# Routes — Static files
# ──────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ──────────────────────────────────────────
# Init
# ──────────────────────────────────────────

with app.app_context():
    db.create_all()
    log.info("Database initialised")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
