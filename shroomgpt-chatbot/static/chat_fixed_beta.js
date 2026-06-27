/**
 * UI for app_fixed_beta_demo.py — fixed β on demo layers (hardcoded server-side).
 */
(function () {
  const messagesEl = document.getElementById("messages");
  const chatEl = document.getElementById("chat");
  const shellEl = document.querySelector(".shell--bare");
  const form = document.getElementById("form");
  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send");
  const refreshBtn = document.getElementById("refresh");
  const emptyHint = document.getElementById("empty");

  const statusEl = document.getElementById("demo-status");

  let history = [];

  const ACTION_ICONS = {
    copy: [
      '<rect x="9" y="9" width="11" height="11" rx="2" />',
      '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />',
    ].join(""),
    regenerate: [
      '<path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />',
      '<path d="M3 3v5h5" />',
      '<path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />',
      '<path d="M16 16h5v5" />',
    ].join(""),
  };

  function scrollBottom() {
    const scrollEl = chatEl || messagesEl;
    if (!scrollEl) return;
    scrollEl.scrollTop = scrollEl.scrollHeight;
  }

  function scheduleScrollBottom() {
    scrollBottom();
    requestAnimationFrame(() => {
      scrollBottom();
      requestAnimationFrame(scrollBottom);
    });
  }

  if (messagesEl) {
    const resizeObserver = new ResizeObserver(() => {
      if (!shellEl?.classList.contains("shell--idle")) scrollBottom();
    });
    resizeObserver.observe(messagesEl);
  }

  function activateChat() {
    if (shellEl) shellEl.classList.remove("shell--idle");
  }

  function createActionButton(action, label) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "bubble-action";
    btn.dataset.action = action;
    btn.setAttribute("aria-label", label);
    btn.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      ACTION_ICONS[action] +
      "</svg>";
    return btn;
  }

  function getGroupText(group) {
    const bubble = group.querySelector(
      ".bubble.assistant:not(.typing):not(.error)"
    );
    return (bubble?.textContent || "").trim();
  }

  function createBubbleActions(group) {
    const bar = document.createElement("div");
    bar.className = "bubble-actions";

    const copyBtn = createActionButton("copy", "Copy response");
    copyBtn.addEventListener("click", () => {
      copyResponse(getGroupText(group), copyBtn);
    });

    const regenBtn = createActionButton("regenerate", "Regenerate response");
    regenBtn.addEventListener("click", () => {
      regenerateResponse(group);
    });

    bar.append(copyBtn, regenBtn);
    return bar;
  }

  function addAssistantGroup(text, userPrompt, historyIndex, extraClass) {
    const group = document.createElement("div");
    group.className = "message-group message-group--assistant";
    group.dataset.userPrompt = userPrompt;
    group.dataset.historyIndex = String(historyIndex);

    const bubble = document.createElement("div");
    bubble.className = "bubble assistant" + (extraClass ? " " + extraClass : "");
    bubble.textContent = text;
    group.appendChild(bubble);

    if (!extraClass) {
      group.appendChild(createBubbleActions(group));
    }

    messagesEl.appendChild(group);
    scheduleScrollBottom();
    return group;
  }

  function addBubble(text, role, extraClass, meta) {
    if (emptyHint) emptyHint.classList.add("hidden");
    if (role === "user") {
      activateChat();
      const el = document.createElement("div");
      el.className = "bubble user";
      el.textContent = text;
      messagesEl.appendChild(el);
      scheduleScrollBottom();
      return el;
    }

    if (role === "assistant" && meta?.userPrompt != null) {
      return addAssistantGroup(text, meta.userPrompt, meta.historyIndex, extraClass);
    }

    const el = document.createElement("div");
    el.className = "bubble " + role + (extraClass ? " " + extraClass : "");
    el.textContent = text;
    messagesEl.appendChild(el);
    scheduleScrollBottom();
    return el;
  }

  function getAssistantGroups() {
    return messagesEl
      ? Array.from(messagesEl.querySelectorAll(".message-group--assistant"))
      : [];
  }

  function removeNodesFrom(group) {
    let node = group;
    while (node) {
      const next = node.nextElementSibling;
      node.remove();
      node = next;
    }
  }

  function trimHistoryFrom(index) {
    history = history.slice(0, Math.max(0, index));
  }

  async function copyResponse(text, btn) {
    try {
      await navigator.clipboard.writeText(text);
      flashActionLabel(btn, "Copied");
    } catch (err) {
      flashActionLabel(btn, "Copy failed");
    }
  }

  function flashActionLabel(btn, message) {
    const prev = btn.getAttribute("aria-label");
    btn.setAttribute("aria-label", message);
    window.setTimeout(() => {
      btn.setAttribute("aria-label", prev || "");
    }, 1400);
  }

  async function regenerateResponse(group) {
    if (!group || sendBtn?.disabled) return;

    const userPrompt = group.dataset.userPrompt || "";
    if (!userPrompt) return;

    const historyIndex = parseInt(group.dataset.historyIndex || "0", 10);
    const historyForRequest = history.slice(0, historyIndex);

    removeNodesFrom(group);
    trimHistoryFrom(historyIndex);

    await requestReply(userPrompt, historyForRequest);
  }

  function setLoading(on) {
    sendBtn.disabled = on;
    if (refreshBtn) refreshBtn.disabled = on;
    input.readOnly = on;
    getAssistantGroups().forEach((group) => {
      group.querySelectorAll(".bubble-action").forEach((btn) => {
        btn.disabled = on;
      });
    });
  }

  async function resetConversation() {
    history = [];
    if (messagesEl) messagesEl.innerHTML = "";
    if (shellEl) shellEl.classList.add("shell--idle");
    input.value = "";
    input.style.height = "auto";
    try {
      await ensureBetaPatchActive();
    } catch (err) {
      if (statusEl) statusEl.textContent = String(err.message || err);
    }
    input.focus();
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

  async function requestReply(userText, historyForRequest) {
    const historyIndex = historyForRequest.length;
    setLoading(true);
    const typing = addBubble("…", "assistant", "typing", {
      userPrompt: userText,
      historyIndex,
    });

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText, history: historyForRequest }),
      });
      const data = await res.json().catch(() => ({}));

      typing.remove();

      if (!res.ok) {
        const msg = data.error || data.detail || res.statusText || "Request failed";
        addBubble(String(msg), "assistant", "error");
        return;
      }

      const reply = (data.reply || "").trim() || "…";
      addBubble(reply, "assistant", null, {
        userPrompt: userText,
        historyIndex,
      });
      if (data.trip_after) applyState(data.trip_after);

      history = historyForRequest.concat(
        { role: "user", content: userText },
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
    await requestReply(text, history.slice());
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

  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      resetConversation();
    });
  }

  init();
})();
