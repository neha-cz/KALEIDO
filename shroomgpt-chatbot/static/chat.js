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

  const doseEl = document.getElementById("dose");
  const turnStepEl = document.getElementById("turn-step");
  const scheduleEl = document.getElementById("schedule");
  const hierarchyEl = document.getElementById("hierarchy");
  const tauEl = document.getElementById("tau");
  const tFinalEl = document.getElementById("t-final");
  const coupleSamplingEl = document.getElementById("couple-sampling");
  const samplingHotEl = document.getElementById("sampling-hot");

  const doseVal = document.getElementById("dose-val");
  const turnStepVal = document.getElementById("turn-step-val");
  const hierarchyVal = document.getElementById("hierarchy-val");
  const tauVal = document.getElementById("tau-val");
  const tFinalVal = document.getElementById("t-final-val");
  const samplingHotVal = document.getElementById("sampling-hot-val");

  const fieldTau = document.getElementById("field-tau");
  const fieldTFinal = document.getElementById("field-t-final");
  const fieldSamplingHot = document.getElementById("field-sampling-hot");

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

  // Show only the knobs the chosen schedule actually uses.
  // τ drives the exponential recovery; t_final drives the linear schedule;
  // logarithmic needs neither (it is parameter-free apart from dose/hierarchy).
  function syncScheduleFields() {
    const sched = scheduleEl.value;
    if (fieldTau) fieldTau.hidden = sched !== "exponential";
    if (fieldTFinal) fieldTFinal.hidden = sched !== "linear";
  }

  // Sampling-temp peak only matters when coupling is enabled.
  function syncSamplingFields() {
    if (fieldSamplingHot) fieldSamplingHot.hidden = !coupleSamplingEl.checked;
  }

  function tripParamsFromUI() {
    return {
      dose: Number(doseEl.value) / 100,
      turn_step: Number(turnStepEl.value),
      schedule: scheduleEl.value,
      tau: Number(tauEl.value),
      t_final: Number(tFinalEl.value),
      hierarchy_power: Number(hierarchyEl.value),
      couple_sampling: coupleSamplingEl.checked,
      sampling_T_hot: Number(samplingHotEl.value),
    };
  }

  function syncSliderLabels() {
    doseVal.textContent = doseEl.value + "%";
    turnStepVal.textContent = Number(turnStepEl.value).toFixed(2);
    hierarchyVal.textContent = Number(hierarchyEl.value).toFixed(2);
    tauVal.textContent = Number(tauEl.value).toFixed(1);
    tFinalVal.textContent = String(Math.round(Number(tFinalEl.value)));
    samplingHotVal.textContent = Number(samplingHotEl.value).toFixed(1);
  }

  // Describe the annealing schedule compactly for the status line.
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

  // ratios is per_layer_beta_ratio (β/β₀ ∈ (0,1]); smaller = flatter landscape.
  function formatTripStatus(trip, ratios) {
    if (!trip || !trip.active) {
      return "Off — set dose & schedule, then begin";
    }
    const decay = typeof trip.decay_now === "number" ? trip.decay_now.toFixed(2) : "?";
    let betaHint = "";
    if (Array.isArray(ratios) && ratios.length) {
      const minR = Math.min(...ratios).toFixed(2);
      const maxR = Math.max(...ratios).toFixed(2);
      betaHint = ` · β/β₀ ${minR}–${maxR}`;
    }
    let sampHint = "";
    if (trip.couple_sampling && typeof trip.sampling_T_mult_now === "number") {
      sampHint = ` · sampT×${trip.sampling_T_mult_now.toFixed(2)}`;
    }
    const sched = scheduleSummary(trip);
    return `Active · ${sched} · t=${trip.t.toFixed(1)} · decay=${decay} · dose=${Math.round(
      trip.dose * 100
    )}%${betaHint}${sampHint}`;
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
    const full = formatTripStatus(trip, ratios);
    tripStatus.textContent = full;
    if (tripStatusCompact) tripStatusCompact.textContent = formatTripStatusCompact(trip);
    if (tripToggle) tripToggle.classList.toggle("is-active-trip", tripActive);
    tripStartBtn.disabled = tripActive;
    tripAdvanceBtn.disabled = !tripActive;
    tripStopBtn.disabled = !tripActive;
    if (trip && trip.active) {
      doseEl.value = String(Math.round(trip.dose * 100));
      turnStepEl.value = String(trip.turn_step ?? turnStepEl.value);
      if (trip.schedule) scheduleEl.value = trip.schedule;
      hierarchyEl.value = String(trip.hierarchy_power ?? hierarchyEl.value);
      tauEl.value = String(trip.tau ?? tauEl.value);
      tFinalEl.value = String(trip.t_final ?? tFinalEl.value);
      if (typeof trip.couple_sampling === "boolean") {
        coupleSamplingEl.checked = trip.couple_sampling;
      }
      samplingHotEl.value = String(trip.sampling_T_hot ?? samplingHotEl.value);
      syncScheduleFields();
      syncSamplingFields();
      syncSliderLabels();
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

  // Slider/number inputs: relabel and (if active) push live config.
  [doseEl, turnStepEl, hierarchyEl, tauEl, tFinalEl, samplingHotEl].forEach((el) => {
    el.addEventListener("input", () => {
      syncSliderLabels();
      if (tripActive) {
        tripRequest("/api/trip/configure", tripParamsFromUI()).catch((err) => {
          tripStatus.textContent = err.message || String(err);
        });
      }
    });
  });

  // Sampling-coupling checkbox: toggle the peak field, then push config.
  coupleSamplingEl.addEventListener("change", () => {
    syncSamplingFields();
    if (tripActive) {
      tripRequest("/api/trip/configure", tripParamsFromUI()).catch((err) => {
        tripStatus.textContent = err.message || String(err);
      });
    }
  });

  // Schedule dropdown: toggle which knobs are visible, then push config.
  scheduleEl.addEventListener("change", () => {
    syncScheduleFields();
    if (tripActive) {
      tripRequest("/api/trip/configure", tripParamsFromUI()).catch((err) => {
        tripStatus.textContent = err.message || String(err);
      });
    }
  });

  tripStartBtn.addEventListener("click", () => {
    tripRequest("/api/trip/start", tripParamsFromUI()).catch((err) => {
      tripStatus.textContent = err.message || String(err);
    });
  });

  tripAdvanceBtn.addEventListener("click", () => {
    tripRequest("/api/trip/advance", { steps: Number(turnStepEl.value) }).catch((err) => {
      tripStatus.textContent = err.message || String(err);
    });
  });

  tripStopBtn.addEventListener("click", () => {
    tripRequest("/api/trip/stop").catch((err) => {
      tripStatus.textContent = err.message || String(err);
    });
  });

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

  syncScheduleFields();
  syncSamplingFields();
  syncSliderLabels();
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