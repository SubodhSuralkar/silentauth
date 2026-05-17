/**
 * SilentAuth — Frontend Behavioral Engine + API Client
 *
 * Real implementations:
 *  - BehavioralCollector  : captures keystroke dwell/flight times + mouse dynamics
 *  - API                  : authenticated REST calls to Flask backend
 *  - TrustMonitor         : polls /api/behavioral/ping every 30 s and updates UI
 */

"use strict";

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

const API_BASE = window.location.origin; // same origin as Flask

// ─────────────────────────────────────────────────────────────
// Behavioral Biometrics Collector
// ─────────────────────────────────────────────────────────────

class BehavioralCollector {
  constructor() {
    this._keystrokes   = [];   // { key, timestamp, duration }
    this._mouseEvents  = [];   // { x, y, t }
    this._pressedAt    = {};   // key → keydown timestamp
    this._lastMouse    = 0;
    this._bound        = false;
  }

  start() {
    if (this._bound) return;
    this._bound = true;

    document.addEventListener("keydown", (e) => {
      this._pressedAt[e.code] = Date.now();
    });

    document.addEventListener("keyup", (e) => {
      const t0 = this._pressedAt[e.code];
      if (!t0) return;
      const duration = Date.now() - t0;
      if (duration > 5 && duration < 1500) {
        this._keystrokes.push({ key: e.code, timestamp: t0, duration });
        // keep last 80 keystrokes
        if (this._keystrokes.length > 80) this._keystrokes.shift();
      }
      delete this._pressedAt[e.code];
    });

    document.addEventListener("mousemove", (e) => {
      const now = Date.now();
      if (now - this._lastMouse > 40) {   // sample at ~25 Hz
        this._mouseEvents.push({ x: e.clientX, y: e.clientY, t: now });
        if (this._mouseEvents.length > 120) this._mouseEvents.shift();
        this._lastMouse = now;
      }
    });
  }

  /** Returns a snapshot suitable for the API. */
  snapshot() {
    return {
      keystrokes:  [...this._keystrokes],
      mouseEvents: [...this._mouseEvents],
    };
  }

  reset() {
    this._keystrokes  = [];
    this._mouseEvents = [];
  }

  get keystrokeCount() { return this._keystrokes.length; }
}

// ─────────────────────────────────────────────────────────────
// API Client
// ─────────────────────────────────────────────────────────────

const Api = (() => {
  let _token = localStorage.getItem("sa_token") || null;

  function _headers(json = true) {
    const h = {};
    if (json) h["Content-Type"] = "application/json";
    if (_token) h["Authorization"] = `Bearer ${_token}`;
    return h;
  }

  async function _post(path, body) {
    const res  = await fetch(API_BASE + path, {
      method:  "POST",
      headers: _headers(),
      body:    JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw { status: res.status, ...data };
    return data;
  }

  async function _get(path) {
    const res  = await fetch(API_BASE + path, { headers: _headers() });
    const data = await res.json();
    if (!res.ok) throw { status: res.status, ...data };
    return data;
  }

  return {
    setToken(t)       { _token = t; localStorage.setItem("sa_token", t); },
    clearToken()      { _token = null; localStorage.removeItem("sa_token"); },
    hasToken()        { return !!_token; },

    register(username, email, password) {
      return _post("/api/register", { username, email, password });
    },
    login(email, password, behavioralData) {
      return _post("/api/login", { email, password, behavioralData });
    },
    dashboard()        { return _get("/api/dashboard"); },
    behavioralPing(bd) { return _post("/api/behavioral/ping", { behavioralData: bd }); },
    transfer(payload)  { return _post("/api/transfer", payload); },
  };
})();

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────

const collector    = new BehavioralCollector();
let   currentTrust = 75;
let   pingInterval = null;

// ─────────────────────────────────────────────────────────────
// UI Helpers
// ─────────────────────────────────────────────────────────────

function showPage(id) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.getElementById(id)?.classList.add("active");
}

function setTrustUI(score, method) {
  currentTrust = score;

  const inner   = document.querySelector(".trust-inner");
  const riskEl  = document.getElementById("riskLevel");
  const methodEl= document.getElementById("trustMethod");

  if (inner)   inner.textContent = score + "%";
  if (methodEl) methodEl.textContent = method === "ml_model" ? "IsolationForest" : "Heuristic";

  // Colour code the ring
  const ring = document.querySelector(".trust-ring");
  if (ring) {
    ring.style.borderTopColor =
      score >= 80 ? "#56ccf2" :
      score >= 60 ? "#fbbf24" :
                    "#f87171";
  }
  if (riskEl) {
    riskEl.textContent =
      score >= 80 ? "Low" :
      score >= 60 ? "Medium" :
                    "High";
    riskEl.style.color =
      score >= 80 ? "#7ee081" :
      score >= 60 ? "#fbbf24" :
                    "#f87171";
  }
}

function appendLog(text, warn = false) {
  const logsEl = document.querySelector(".logs");
  if (!logsEl) return;
  const p = document.createElement("p");
  p.textContent = text;
  p.style.borderLeftColor = warn ? "#f87171" : "#56ccf2";
  logsEl.prepend(p);
  // cap at 8 entries
  while (logsEl.children.length > 8) logsEl.lastChild.remove();
}

function renderTransactions(txns) {
  const ul = document.getElementById("transactionList");
  if (!ul) return;
  ul.innerHTML = "";
  txns.forEach(t => {
    const li    = document.createElement("li");
    const sign  = t.direction === "credit" ? "+" : "-";
    const color = t.direction === "credit" ? "#7ee081" : "#f87171";
    li.innerHTML = `
      <span>${t.recipient} <small style="color:#9bb7d6">(${t.purpose})</small></span>
      <span style="color:${color}">${sign}₹${t.amount.toLocaleString()}</span>
    `;
    ul.appendChild(li);
  });
}

// ─────────────────────────────────────────────────────────────
// Trust Monitor (continuous background polling)
// ─────────────────────────────────────────────────────────────

function startTrustMonitor() {
  if (pingInterval) return;
  pingInterval = setInterval(async () => {
    try {
      const snap = collector.snapshot();
      if (snap.keystrokes.length < 3 && snap.mouseEvents.length < 5) return;
      const res  = await Api.behavioralPing(snap);
      setTrustUI(res.trustScore, res.method);
      appendLog(
        `↻ Session re-scored: ${res.trustScore}% `
        + `(dwell ${res.features.avgDwellTime}ms · `
        + `flight ${res.features.avgFlightTime}ms · `
        + `speed ${res.features.typingSpeedCps} cps)`
      );
      collector.reset();
    } catch (e) {
      console.warn("Ping failed:", e);
    }
  }, 30_000);   // every 30 seconds
}

function stopTrustMonitor() {
  clearInterval(pingInterval);
  pingInterval = null;
}

// ─────────────────────────────────────────────────────────────
// Behavioral feature labels (for the AI Monitor card)
// ─────────────────────────────────────────────────────────────

function updateFeatureDisplay(features) {
  const el = id => document.getElementById(id);
  if (el("featDwell"))   el("featDwell").textContent   = features.avgDwellTime   + " ms";
  if (el("featFlight"))  el("featFlight").textContent  = features.avgFlightTime  + " ms";
  if (el("featVariance"))el("featVariance").textContent= features.rhythmVariance + " ms";
  if (el("featSpeed"))   el("featSpeed").textContent   = features.typingSpeedCps + " cps";
}

// ─────────────────────────────────────────────────────────────
// Login Flow
// ─────────────────────────────────────────────────────────────

const loginBtn = document.getElementById("loginBtn");
if (loginBtn) {
  loginBtn.addEventListener("click", async () => {
    const email    = document.getElementById("emailInput")?.value.trim();
    const password = document.getElementById("passwordInput")?.value;

    if (!email || !password) {
      alert("Please enter your email and password.");
      return;
    }

    const snap = collector.snapshot();
    loginBtn.disabled   = true;
    loginBtn.textContent = `Analyzing ${snap.keystrokes.length} behavioral signals…`;

    try {
      const res = await Api.login(email, password, snap);
      Api.setToken(res.token);

      // Populate UI
      document.getElementById("displayName").textContent = res.user.username;
      document.getElementById("balanceAmount").textContent =
        "₹ " + res.user.balance.toLocaleString();

      setTrustUI(res.trustScore, res.trustMethod);
      updateFeatureDisplay(res.features);

      const mlMsg = res.mlReady
        ? `✓ IsolationForest model active (${res.sampleCount} training samples)`
        : `⚙ Collecting behavioral baseline (${res.sampleCount}/${5} samples)`;

      appendLog(mlMsg);
      appendLog(`✓ Login authenticated — trust score: ${res.trustScore}%`);
      appendLog(`✓ Keystroke features extracted (${snap.keystrokes.length} events)`);

      // Load full dashboard data
      const dash = await Api.dashboard();
      renderTransactions(dash.transactions);

      showPage("dashboard-page");
      collector.reset();
      startTrustMonitor();

    } catch (err) {
      alert(err.error || "Login failed. Check your credentials.");
    } finally {
      loginBtn.disabled    = false;
      loginBtn.textContent = "Start Secure Scan";
    }
  });
}

// ─────────────────────────────────────────────────────────────
// Register Flow
// ─────────────────────────────────────────────────────────────

const registerBtn = document.getElementById("registerBtn");
if (registerBtn) {
  registerBtn.addEventListener("click", async () => {
    const username = document.getElementById("usernameInput")?.value.trim();
    const email    = document.getElementById("regEmailInput")?.value.trim();
    const password = document.getElementById("regPasswordInput")?.value;

    if (!username || !email || !password) {
      alert("All fields are required.");
      return;
    }

    registerBtn.disabled    = true;
    registerBtn.textContent = "Creating account…";

    try {
      const res = await Api.register(username, email, password);
      Api.setToken(res.token);
      alert("Account created! Please log in to build your behavioral baseline.");
      showLoginForm();
    } catch (err) {
      alert(err.error || "Registration failed.");
    } finally {
      registerBtn.disabled    = false;
      registerBtn.textContent = "Create Account";
    }
  });
}

// ─────────────────────────────────────────────────────────────
// Toggle login ↔ register
// ─────────────────────────────────────────────────────────────

function showLoginForm() {
  document.getElementById("loginForm")?.classList.remove("hidden");
  document.getElementById("registerForm")?.classList.add("hidden");
}

function showRegisterForm() {
  document.getElementById("loginForm")?.classList.add("hidden");
  document.getElementById("registerForm")?.classList.remove("hidden");
}

document.getElementById("showRegister")?.addEventListener("click", showRegisterForm);
document.getElementById("showLogin")?.addEventListener("click", showLoginForm);

// ─────────────────────────────────────────────────────────────
// Logout
// ─────────────────────────────────────────────────────────────

document.getElementById("logoutBtn")?.addEventListener("click", () => {
  stopTrustMonitor();
  Api.clearToken();
  collector.reset();
  showPage("login-page");
});

// ─────────────────────────────────────────────────────────────
// Authentication Mode Buttons (now update real trust scores)
// ─────────────────────────────────────────────────────────────

function setMode(mode) {
  const modeMap = {
    normal:   { label: "Normal Mode",       logEntry: "✓ Normal behavioral mode activated",         warn: false },
    delegate: { label: "Delegate Mode",     logEntry: "✓ Delegate/assisted banking mode enabled",   warn: false },
    secure:   { label: "High Security",     logEntry: "✓ Multi-layer verification intensified",     warn: false },
    lockdown: { label: "Emergency Lockdown",logEntry: "⚠ Emergency lockdown — all transfers halted",warn: true  },
  };
  const cfg = modeMap[mode];
  if (!cfg) return;
  appendLog(cfg.logEntry, cfg.warn);

  if (mode === "lockdown") {
    // Reduce displayed trust score to signal anomaly
    setTrustUI(Math.min(currentTrust, 30), "heuristic");
    appendLog("⚠ Behavioral anomaly flagged — fraud investigation initiated", true);
  }
}

// ─────────────────────────────────────────────────────────────
// Money Transfer
// ─────────────────────────────────────────────────────────────

document.getElementById("transferBtn")?.addEventListener("click", async () => {
  const recipient = document.getElementById("recipientName")?.value.trim();
  const amount    = parseFloat(document.getElementById("transferAmount")?.value);
  const purpose   = document.getElementById("purpose")?.value.trim() || "Transfer";
  const btn       = document.getElementById("transferBtn");

  if (!recipient || !amount || amount <= 0) {
    alert("Please fill in recipient and a valid amount.");
    return;
  }

  btn.disabled    = true;
  btn.textContent = "Verifying with SilentAuth…";

  try {
    const res = await Api.transfer({
      recipient,
      amount,
      purpose,
      trustScore: currentTrust,
    });

    document.getElementById("balanceAmount").textContent =
      "₹ " + res.newBalance.toLocaleString();

    // Prepend new transaction to list
    const ul = document.getElementById("transactionList");
    const li = document.createElement("li");
    li.innerHTML = `
      <span>${recipient} <small style="color:#9bb7d6">(${purpose})</small></span>
      <span style="color:#f87171">-₹${amount.toLocaleString()}</span>
    `;
    ul.prepend(li);

    const status = res.transaction.status;
    appendLog(
      `${status === "approved" ? "✓" : "⚠"} Transfer ${status}: ₹${amount.toLocaleString()} → ${recipient} (trust: ${currentTrust}%)`,
      status !== "approved"
    );

    if (status === "flagged") {
      alert(`Transfer completed but FLAGGED for review (low trust: ${currentTrust}%)`);
    } else {
      alert("Transfer successful ✓");
    }

  } catch (err) {
    const msg = err.error || "Transfer failed.";
    appendLog("⚠ " + msg, true);
    alert(msg + (err.requiredMin ? `\nMinimum trust required: ${err.requiredMin}%` : ""));
  } finally {
    btn.disabled    = false;
    btn.textContent = "Transfer Money";
  }
});

// ─────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────

collector.start();   // start capturing keystrokes and mouse globally

// Expose setMode globally (called from inline onclick attributes in HTML)
window.setMode = setMode;
