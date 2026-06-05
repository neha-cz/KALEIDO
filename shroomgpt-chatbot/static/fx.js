import { layoutNextLine, prepareWithSegments } from "https://cdn.jsdelivr.net/npm/@chenglou/pretext/+esm";

const WARP_TARGETS = [".brand h1", ".sub", ".send-text", ".bubble"];

function splitGraphemes(text) {
  if (window.Intl && Intl.Segmenter) {
    const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
    return [...segmenter.segment(text)].map((part) => part.segment);
  }
  return Array.from(text);
}

function getLineHeight(style, fontSize) {
  const n = Number.parseFloat(style.lineHeight);
  return Number.isFinite(n) ? n : fontSize * 1.35;
}

function getFontSize(style) {
  const n = Number.parseFloat(style.fontSize);
  return Number.isFinite(n) ? n : 16;
}

function warpElement(el) {
  if (!(el instanceof HTMLElement)) return;

  const raw = el.dataset.warpSource ?? el.textContent ?? "";
  if (!raw.trim()) return;
  if (el.querySelector(".warp-text")) return;

  const style = window.getComputedStyle(el);
  const font = style.font;
  const letterSpacing = Number.parseFloat(style.letterSpacing);
  const fontSize = getFontSize(style);
  const lineHeight = getLineHeight(style, fontSize);
  let width = Math.max(
    80,
    Math.floor(el.getBoundingClientRect().width || el.clientWidth || 280) - 2
  );

  // Keep single-token headings (e.g. "ShroomGPT") from splitting into odd wraps.
  if (!/\s/.test(raw)) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.font = font;
      const naturalWidth = Math.ceil(ctx.measureText(raw).width) + 8;
      width = Math.max(width, naturalWidth);
    }
  }

  let lines = [];
  const forceSingleLine = el.matches(".brand h1");
  const isBubble = el.classList.contains("bubble");
  if (forceSingleLine) {
    lines = [raw];
  } else {
    try {
      const prepared = prepareWithSegments(raw, font, {
        whiteSpace: "pre-wrap",
        wordBreak: isBubble ? "keep-all" : "normal",
        letterSpacing: Number.isFinite(letterSpacing) ? letterSpacing : undefined,
      });
      let cursor = { segmentIndex: 0, graphemeIndex: 0 };
      while (true) {
        const line = layoutNextLine(prepared, cursor, width);
        if (line === null) break;
        lines.push(line.text);
        cursor = line.end;
      }
    } catch {
      return;
    }
  }

  if (!lines.length) return;

  el.dataset.warpSource = raw;
  el.style.whiteSpace = "normal";
  el.textContent = "";
  el.classList.add("warp-host");

  const wrap = document.createElement("span");
  wrap.className = "warp-text";

  lines.forEach((lineText, lineIndex) => {
    const lineEl = document.createElement("span");
    lineEl.className = "warp-line";
    lineEl.style.setProperty("--line-index", String(lineIndex));

    const graphemes = splitGraphemes(lineText || " ");
    graphemes.forEach((char, charIndex) => {
      const charEl = document.createElement("span");
      charEl.className = "warp-char";
      charEl.style.setProperty("--char-index", String(charIndex));
      charEl.textContent = char === " " ? "\u00A0" : char;
      lineEl.appendChild(charEl);
    });

    wrap.appendChild(lineEl);
  });

  el.appendChild(wrap);
  bindInteractiveWarp(el);
}

function applyWarp() {
  WARP_TARGETS.forEach((selector) => {
    document.querySelectorAll(selector).forEach((node) => warpElement(node));
  });
}

function resetWarp() {
  document.querySelectorAll(".warp-host").forEach((el) => {
    if (!(el instanceof HTMLElement)) return;
    if (!el.dataset.warpSource) return;
    el.classList.remove("is-warp-active");
    el.textContent = el.dataset.warpSource;
  });
}

function resetWarpChars(el) {
  el.querySelectorAll(".warp-char").forEach((charEl) => {
    if (!(charEl instanceof HTMLElement)) return;
    charEl.style.setProperty("--warp-x", "0px");
    charEl.style.setProperty("--warp-y", "0px");
    charEl.style.setProperty("--warp-r", "0deg");
    charEl.style.setProperty("--warp-s", "1");
    charEl.style.setProperty("--warp-a", "1");
  });
}

function bindInteractiveWarp(el) {
  if (el.dataset.warpInteractive === "1") return;
  el.dataset.warpInteractive = "1";

  let rafId = 0;
  let pointerX = 0;
  let pointerY = 0;

  const applyCursorWarp = () => {
    rafId = 0;
    // Mirrors the influence pattern used in the illustrated-manuscript demo.
    const radius = 120;
    el.querySelectorAll(".warp-char").forEach((charEl) => {
      if (!(charEl instanceof HTMLElement)) return;
      const rect = charEl.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = pointerX - cx;
      const dy = pointerY - cy;
      const distance = Math.hypot(dx, dy);
      const strength = Math.max(0, 1 - distance / radius);
      const normX = distance > 0 ? dx / distance : 0;
      const normY = distance > 0 ? dy / distance : 0;

      const moveX = normX * strength * 26;
      const moveY = normY * strength * 26;
      const rotate = strength * (normX > 0 ? 1 : -1) * 14;
      const scale = 1 + strength * 0.1;
      const alpha = Math.max(0.22, 1 - strength * 0.4);

      charEl.style.setProperty("--warp-x", `${moveX.toFixed(2)}px`);
      charEl.style.setProperty("--warp-y", `${moveY.toFixed(2)}px`);
      charEl.style.setProperty("--warp-r", `${rotate.toFixed(2)}deg`);
      charEl.style.setProperty("--warp-s", scale.toFixed(3));
      charEl.style.setProperty("--warp-a", alpha.toFixed(3));
    });
  };

  const queueCursorWarp = (x, y) => {
    pointerX = x;
    pointerY = y;
    if (!rafId) {
      rafId = window.requestAnimationFrame(applyCursorWarp);
    }
  };

  el.addEventListener("pointerenter", (event) => {
    el.classList.add("is-warp-active");
    queueCursorWarp(event.clientX, event.clientY);
  });

  el.addEventListener("pointermove", (event) => {
    queueCursorWarp(event.clientX, event.clientY);
  });

  el.addEventListener("pointerleave", () => {
    el.classList.remove("is-warp-active");
    if (rafId) {
      window.cancelAnimationFrame(rafId);
      rafId = 0;
    }
    resetWarpChars(el);
  });
}

function initWarping() {
  applyWarp();

  const messages = document.getElementById("messages");
  if (messages) {
    const observer = new MutationObserver((records) => {
      records.forEach((record) => {
        record.addedNodes.forEach((node) => {
          if (!(node instanceof HTMLElement)) return;
          if (node.classList.contains("bubble")) {
            warpElement(node);
          }
        });
      });
    });
    observer.observe(messages, { childList: true });
  }

  let timer = 0;
  window.addEventListener("resize", () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      resetWarp();
      applyWarp();
    }, 120);
  });
}

function initCursorMushroom() {
  if (window.matchMedia("(pointer: coarse)").matches) return;

  const icon = document.createElement("img");
  icon.id = "cursor-mushroom";
  icon.src = "/mushroom.png";
  icon.alt = "";
  icon.setAttribute("aria-hidden", "true");
  document.body.appendChild(icon);

  let targetX = -999;
  let targetY = -999;
  let x = targetX;
  let y = targetY;
  let visible = false;

  const animate = () => {
    x += (targetX - x) * 0.15;
    y += (targetY - y) * 0.15;
    icon.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    window.requestAnimationFrame(animate);
  };
  window.requestAnimationFrame(animate);

  window.addEventListener("mousemove", (event) => {
    targetX = event.clientX + 14;
    targetY = event.clientY + 16;
    if (!visible) {
      visible = true;
      icon.style.opacity = "0.92";
    }
  });

  window.addEventListener("mouseleave", () => {
    visible = false;
    icon.style.opacity = "0";
  });
}

window.addEventListener("DOMContentLoaded", () => {
  initWarping();
  initCursorMushroom();
});
