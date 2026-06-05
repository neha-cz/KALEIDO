/**
 * UI for app_fixed_beta_demo.py — fixed β on demo layers (hardcoded server-side).
 */
(function () {
  const messagesEl = document.getElementById("messages");
  const form = document.getElementById("form");
  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send");
  const emptyHint = document.getElementById("empty");

  const statusEl = document.getElementById("demo-status");

  let history = [];

  function scrollBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addBubble(text, role, extraClass) {
    if (emptyHint) emptyHint.classList.add("hidden");
    const el = document.createElement("div");
    el.className = "bubble " + role + (extraClass ? " " + extraClass : "");
    el.textContent = text;
    messagesEl.appendChild(el);
    scrollBottom();
    return el;
  }

  function setLoading(on) {
    sendBtn.disabled = on;
    input.readOnly = on;
  }

  function formatStatus(trip) {
    if (!trip) return "Loading…";
    const layers = Array.isArray(trip.demo_layers)
      ? trip.demo_layers.join(",")
      : "?";
    const ratio =
      typeof trip.demo_beta_ratio === "number"
        ? trip.demo_beta_ratio.toFixed(2)
        : "?";
    const betaOn = trip.active && trip.beta_patch;
    return betaOn
      ? `β fixed · layers ${layers} · ratio ${ratio}`
      : "β off";
  }

  function applyState(trip) {
    if (statusEl) statusEl.textContent = formatStatus(trip);
  }

  async function apiPost(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || data.detail || res.statusText || "Request failed");
    }
    return data;
  }

  async function refreshState() {
    const res = await fetch("/api/trip/state");
    const data = await res.json().catch(() => ({}));
    if (res.ok && data.trip) {
      applyState(data.trip);
      return data.trip;
    }
    return null;
  }

  /** Arm the hardcoded β patch (TRIP_PRESET on server). */
  async function ensureBetaPatchActive() {
    const data = await apiPost("/api/trip/start");
    if (data.trip) applyState(data.trip);
  }

  async function init() {
    try {
      await ensureBetaPatchActive();
    } catch (err) {
      if (statusEl) statusEl.textContent = String(err.message || err);
    }
    await refreshState();
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = (input.value || "").trim();
    if (!text) return;

    addBubble(text, "user");
    input.value = "";
    input.style.height = "auto";
    setLoading(true);
    const typing = addBubble("…", "assistant", "typing");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history: history }),
      });
      const data = await res.json().catch(() => ({}));

      typing.remove();

      if (!res.ok) {
        const msg = data.error || data.detail || res.statusText || "Request failed";
        addBubble(String(msg), "assistant", "error");
        return;
      }

      const reply = (data.reply || "").trim() || "…";
      addBubble(reply, "assistant");
      if (data.trip_after) applyState(data.trip_after);

      history = history.concat(
        { role: "user", content: text },
        { role: "assistant", content: reply }
      );
      if (history.length > 20) {
        history = history.slice(-20);
      }
    } catch (err) {
      typing.remove();
      addBubble(String(err.message || err), "assistant", "error");
    } finally {
      setLoading(false);
      input.focus();
    }
  });

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 128) + "px";
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  init();
})();
