# SilentAuth — Realistic Working Prototype

Behavioral biometrics authentication for NeoBank.  
**Real** Flask backend · **Real** scikit-learn IsolationForest · **Real** JWT · SQLite persistence.

---

## Quick Start (3 commands)

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

---

## What's Real Now

| Component | Before | After |
|---|---|---|
| Trust score | Hardcoded 92 / 87 / 98 / 23 | IsolationForest on live keystroke features |
| Authentication | UI delay + page switch | JWT issued by Flask, stored in localStorage |
| Behavioral data | None | Keystroke dwell/flight times + mouse speed captured in-browser |
| ML model | None | `sklearn.ensemble.IsolationForest` trained per-user, persisted in SQLite |
| Database | None | SQLAlchemy + SQLite (4 tables) |
| Transfer security | No validation | Trust-gated: blocked <40%, flagged <65%, high-value requires >85% |
| Session monitoring | Static text | `/api/behavioral/ping` every 30 s re-scores the live session |

---

## Architecture

```
Browser
  └─ BehavioralCollector (script.js)
       ├─ keydown/keyup  →  dwell time, flight time, rhythm variance, typing speed
       └─ mousemove       →  average movement speed

POST /api/login  { email, password, behavioralData }
  └─ extract_features()   →  7-dimensional feature vector
  └─ score_features()
       ├─ < 5 samples: heuristic rule-based baseline
       └─ ≥ 5 samples: IsolationForest.decision_function() mapped to 0-100
  └─ persist BehavioralSample → maybe_retrain() → issue JWT

GET  /api/dashboard         (JWT required)
POST /api/behavioral/ping   (JWT required, called every 30 s)
POST /api/transfer          (JWT required, trust-gated)
```

---

## Feature Vector

```
avg_dwell_time    — average ms a key is held down
avg_flight_time   — average ms between consecutive keystrokes  
rhythm_variance   — std-dev of flight times (typing consistency)
typing_speed_cps  — characters per second
avg_mouse_speed   — pixels per second
hour_of_day       — 0–23 (unusual hours = lower baseline trust)
day_of_week       — 0–6
```

---

## ML Details

- **Algorithm**: `IsolationForest` (contamination=0.1, n_estimators=100)
- **Scaler**: `StandardScaler` (fit on user's own history)
- **Cold start**: heuristic baseline until 5 login samples collected
- **Retraining**: every 3 new samples (configurable via `RETRAIN_EVERY`)
- **Storage**: model pickled into `user_ml_models.model_blob` (SQLite BLOB)

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/register` | — | Create account |
| POST | `/api/login` | — | Verify credentials + behavioral data → JWT + trust score |
| GET | `/api/dashboard` | JWT | Full dashboard state |
| POST | `/api/behavioral/ping` | JWT | Continuous session re-scoring |
| POST | `/api/transfer` | JWT | Trust-gated money transfer |

---

## Security Notes (Hackathon → Production gaps)

- Passwords are SHA-256 hashed; production should use `bcrypt`
- IsolationForest trains only on the **current user's** history; production needs a global anomaly baseline + federated data
- HTTPS required in production (JWT tokens exposed over HTTP)
- `JWT_SECRET_KEY` should be set via environment variable
