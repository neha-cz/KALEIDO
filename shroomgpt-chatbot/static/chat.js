(function () {
  const messagesEl = document.getElementById("messages");
  const form = document.getElementById("form");
  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send");
  const emptyHint = document.getElementById("empty");

  const tripStatus = document.getElementById("trip-status");
  const tripStatusCompact = document.getElementById("trip-status-compact");
  const tripToggle = document.getElementById("trip-toggle");
  const tripDrawer = document.getElementById("trip-drawer");
  const tripBackdrop = document.getElementById("trip-backdrop");
  const tripClose = document.getElementById("trip-close");
  const tripStartBtn = document.getElementById("trip-start");
  const tripAdvanceBtn = document.getElementById("trip-advance");
  const tripStopBtn = document.getElementById("trip-stop");
  const annealingEl = document.getElementById("annealing");
  const promptEngineeringEl = document.getElementById("prompt-engineering");

  let history = [];
  let tripActive = false;

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

  function scheduleSummary(trip) {
    if (!trip) return "";
    if (trip.schedule === "exponential") {
      const tau = typeof trip.tau === "number" ? trip.tau.toFixed(1) : "?";
      return `exp τ=${tau}`;
    }
    if (trip.schedule === "linear") {
      const tf = typeof trip.t_final === "number" ? Math.round(trip.t_final) : "?";
      return `linear t_final=${tf}`;
    }
    if (trip.schedule === "logarithmic") return "log";
    return trip.schedule || "";
  }

  function modeHints(trip) {
    if (!trip) return "";
    const ann = trip.annealing ? "β on" : "β off";
    const pe = trip.prompt_engineering ? "prompt on" : "prompt off";
    return ` · ${ann} · ${pe}`;
  }

  function formatTripStatus(trip, ratios) {
    if (!trip || !trip.active) {
      return `Off — begin trip for β time${modeHints(trip)}`;
    }
    const decay =
      trip.annealing && typeof trip.decay_now === "number"
        ? trip.decay_now.toFixed(2)
        : "—";
    let betaHint = "";
    if (trip.annealing && Array.isArray(ratios) && ratios.length) {
      const minR = Math.min(...ratios).toFixed(2);
      const maxR = Math.max(...ratios).toFixed(2);
      betaHint = ` · β/β₀ ${minR}–${maxR}`;
    }
    let sampHint = "";
    if (
      trip.annealing &&
      trip.couple_sampling &&
      typeof trip.sampling_T_mult_now === "number"
    ) {
      sampHint = ` · sampT×${trip.sampling_T_mult_now.toFixed(2)}`;
    }
    const sched = scheduleSummary(trip);
    return `Active · ${sched} · t=${trip.t.toFixed(1)} · decay=${decay} · dose=${Math.round(
      trip.dose * 100
    )}%${betaHint}${sampHint}${modeHints(trip)}`;
  }

  function formatTripStatusCompact(trip) {
    if (!trip || !trip.active) return "Trip off";
    const decay = typeof trip.decay_now === "number" ? trip.decay_now.toFixed(2) : "?";
    return `t=${trip.t.toFixed(1)} · ${Math.round(trip.dose * 100)}% · decay ${decay}`;
  }

  function setTripDrawer(open) {
    if (!tripDrawer || !tripToggle) return;
    tripDrawer.classList.toggle("is-open", open);
    tripDrawer.setAttribute("aria-hidden", String(!open));
    tripToggle.setAttribute("aria-expanded", String(open));
    if (tripBackdrop) {
      tripBackdrop.hidden = !open;
      tripBackdrop.setAttribute("aria-hidden", String(!open));
    }
  }

  function applyTripUI(trip, ratios) {
    tripActive = Boolean(trip && trip.active);
    tripStatus.textContent = formatTripStatus(trip, ratios);
    if (tripStatusCompact) tripStatusCompact.textContent = formatTripStatusCompact(trip);
    if (tripToggle) tripToggle.classList.toggle("is-active-trip", tripActive);
    tripStartBtn.disabled = tripActive;
    tripAdvanceBtn.disabled = !tripActive;
    tripStopBtn.disabled = !tripActive;
    if (annealingEl && trip && typeof trip.annealing === "boolean") {
      annealingEl.checked = trip.annealing;
    }
    if (promptEngineeringEl && trip && typeof trip.prompt_engineering === "boolean") {
      promptEngineeringEl.checked = trip.prompt_engineering;
    }
  }

  async function tripRequest(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || data.detail || res.statusText || "Trip request failed");
    }
    if (data.trip) applyTripUI(data.trip, data.per_layer_beta_ratio);
    return data;
  }

  async function refreshTripState() {
    try {
      const res = await fetch("/api/trip/state");
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.trip) applyTripUI(data.trip, data.per_layer_beta_ratio);
    } catch (_) {
      /* ignore on load */
    }
  }

  tripStartBtn.addEventListener("click", () => {
    tripRequest("/api/trip/start").catch((err) => {
      tripStatus.textContent = err.message || String(err);
    });
  });

  tripAdvanceBtn.addEventListener("click", () => {
    tripRequest("/api/trip/advance").catch((err) => {
      tripStatus.textContent = err.message || String(err);
    });
  });

  tripStopBtn.addEventListener("click", () => {
    tripRequest("/api/trip/stop").catch((err) => {
      tripStatus.textContent = err.message || String(err);
    });
  });

  if (annealingEl) {
    annealingEl.addEventListener("change", () => {
      tripRequest("/api/trip/annealing", { enabled: annealingEl.checked }).catch((err) => {
        tripStatus.textContent = err.message || String(err);
        annealingEl.checked = !annealingEl.checked;
      });
    });
  }

  if (promptEngineeringEl) {
    promptEngineeringEl.addEventListener("change", () => {
      tripRequest("/api/trip/prompt_engineering", {
        enabled: promptEngineeringEl.checked,
      }).catch((err) => {
        tripStatus.textContent = err.message || String(err);
        promptEngineeringEl.checked = !promptEngineeringEl.checked;
      });
    });
  }

  if (tripToggle) {
    tripToggle.addEventListener("click", () => {
      setTripDrawer(!tripDrawer.classList.contains("is-open"));
    });
  }
  if (tripClose) tripClose.addEventListener("click", () => setTripDrawer(false));
  if (tripBackdrop) tripBackdrop.addEventListener("click", () => setTripDrawer(false));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setTripDrawer(false);
  });

  refreshTripState();

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
      if (data.trip_after) applyTripUI(data.trip_after, data.per_layer_beta_ratio);

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
})();
