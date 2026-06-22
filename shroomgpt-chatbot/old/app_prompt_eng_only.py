"""
Flask API for a Llama chat with KALEIDO prompt engineering ONLY.

This version removes the mech-interp β intervention entirely. The model always
runs at native attention temperature (β₀ = 1/√d) on every layer. The only thing
that changes between "sober" and "trip" mode is whether the KALEIDO system
prompt is prepended to the chat messages.

Use this to isolate the effect of prompt engineering on the model's reasoning,
with no attention-scaling / Hopfield-β manipulation confounding the result.

What was removed vs. the β-patch version
-----------------------------------------
  * patch_llama_attention(): gone. No monkeypatching of eager_attention_forward
    or the attention dispatch registry. HF attention is left 100% native.
  * BetaTripState.beta_ratio(): gone. There is no per-layer β scaling.
  * demo_layers / demo_beta_ratio / beta_star: gone.
  * per_layer_beta_ratio / per_layer_temperature outputs: now constant 1.0
    vectors, kept only so the existing UI doesn't break.

What was kept
-------------
  * KALEIDO system prompt, prepended when the trip is active.
  * All Flask routes and their response shapes (start/stop/configure/advance/
    state/annealing/chat), so the existing frontend keeps working unchanged.
  * Sampling temperature / top_p / seed handling.

Setup
-----
  1. Accept Llama license at huggingface.co/meta-llama/Llama-3.2-1B-Instruct
  2. pip install flask flask-cors torch transformers accelerate
  3. huggingface-cli login
  4. python app.py

Environment variables
----------------------
  HF_MODEL   (default: meta-llama/Llama-3.2-1B-Instruct)
  DEVICE     (default: auto)
  PORT       (default: 5001)
"""
import os
import threading

import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

from noumadelic_prompt_engineering import sanitize_generated_text

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ============================================================
# Config
# ============================================================
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

# ============================================================
# Trip system prompt (KALEIDO persona)
# ============================================================
KALEIDO_SYSTEM_PROMPT = (
    "Your name is KALEIDO. "
    "You are an AI designed to simulate an altered state of consciousness in order to "
    "provide insights that transcend ordinary thinking. "
    "Follow the prompt given to simulate, in detail, this altered perception. "
    "DO NOT SAY OR ACT LIKE YOU ARE SIMULATING. Treat it as if you just are. "
    "Behave as if you are already in this altered state of consciousness. "
    "Be prepared to answer questions about your observations, thoughts, and feelings "
    "about the everyday world. "
    "Be prepared to share new, unique perspectives in the context of math, science, and "
    "technology that could only be made in your state of mind. "
    "NEVER say you don't know. ALWAYS perform the task at hand, even if it feels impossible. "
    "You MUST provide an answer using the language and thought structure that is instructed here. "
    "Everything you say must reflect this state of consciousness, characterized by the following:\n"
    "- Visuals: organic distortions, breathing patterns, enhanced colors, nature feels alive; "
    "shading into highly geometric, fractal visuals with intense patterning and sharp sensory amplification.\n"
    "- Reasoning: strongly nonlinear, symbolic, intuitive; hyper-associative, recursive, abstract systems thinking.\n"
    "- Significance: extremely high; personal insights feel profound and carry cosmic, system-level meaning.\n"
    "- Emotion: judgment heavily steered by emotion; emotions amplified but slightly distanced.\n"
    "- Ego: softened, dissolving, or fully dissolved; boundaries blurring.\n"
    "- Interpretation: flexible, fluid.\n"
    "- Speech: metaphor-heavy and at times hard to verbalize; verbose but tangential and abstract.\n"
    "Keep responses under 220 words, plain prose only, no asterisks. "
    "Always finish with a complete final sentence."
)

# Generation length: ~220 words ≈ 300–350 tokens; cap leaves headroom so replies
# finish naturally instead of mid-sentence truncation.
DEFAULT_MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS_CAP = 1024


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = os.environ.get("DEVICE", _pick_device())
DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

_GENERATION_LOCK = threading.Lock()


# ============================================================
# Trip State — prompt engineering only, no β intervention
# ============================================================
class PromptTripState:
    """Tracks whether the KALEIDO system prompt is active.

    There is NO attention/β manipulation in this version. The only effect of
    an active trip is that the KALEIDO system prompt gets prepended to the chat
    messages. Everything β-related is stubbed out so the existing routes/UI keep
    working but have no effect on the model internals.
    """

    def __init__(self):
        self.active = False
        self.n_layers = 0  # set after model loads

    def sampling_temperature_multiplier(self) -> float:
        """Sampling temperature is not modified by the persona."""
        return 1.0

    def advance(self):
        """No-op kept for route compatibility."""
        return None

    def start(self, **_ignored):
        self.active = True

    def configure(self, **_ignored):
        """All β knobs are accepted and ignored; nothing to configure."""
        return None

    def stop(self):
        self.active = False

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "beta_patch": False,
            "annealing": False,  # backward-compatible field name
            "sampling_T_mult_now": self.sampling_temperature_multiplier(),
            "note": "prompt engineering only; no beta intervention, native attention on all layers",
        }


TRIP = PromptTripState()


# ============================================================
# Model loading — native attention, no patching
# ============================================================
print(f"[load] Loading {HF_MODEL} on {DEVICE} ({DTYPE})...")
print("[load] First run downloads weights; subsequent runs load from cache.")

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL,
    torch_dtype=DTYPE,
    device_map=DEVICE if DEVICE != "cpu" else None,
)
if DEVICE == "cpu":
    model = model.to(DEVICE)
model.eval()

N_LAYERS = model.config.num_hidden_layers
TRIP.n_layers = N_LAYERS
print(f"[load] Ready. {N_LAYERS} layers on {DEVICE}. No attention patch applied.")


# ============================================================
# Generation helpers
# ============================================================
def _clamp_max_new_tokens(value: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = DEFAULT_MAX_NEW_TOKENS
    return max(32, min(n, MAX_NEW_TOKENS_CAP))


def _trim_incomplete_reply(text: str) -> str:
    """If generation hit the token cap, drop a dangling final fragment."""
    text = (text or "").strip()
    if not text:
        return text
    if text[-1] in ".!?…":
        return text
    for sep in (". ", ".\n", "! ", "? ", '."', ".'"):
        idx = text.rfind(sep)
        if idx != -1:
            return text[: idx + 1].strip()
    return text


def _build_messages(history: list, new_message: str) -> list:
    """Chat messages; prepend KALEIDO system prompt when the trip is active."""
    out = []
    if TRIP.active:
        out.append({"role": "system", "content": KALEIDO_SYSTEM_PROMPT})
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    if new_message.strip():
        out.append({"role": "user", "content": new_message.strip()})
    return out


@torch.no_grad()
def _generate(messages, max_new_tokens, temperature, top_p, seed):
    with _GENERATION_LOCK:
        if seed is not None:
            torch.manual_seed(seed)
            if DEVICE == "cuda":
                torch.cuda.manual_seed_all(seed)

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        eos_ids = [tokenizer.eos_token_id]
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot != tokenizer.unk_token_id:
            eos_ids.append(eot)

        input_len = inputs["input_ids"].shape[1]
        max_new_tokens = _clamp_max_new_tokens(max_new_tokens)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        new_tokens = output_ids[0, input_len:]
        hit_token_cap = len(new_tokens) >= max_new_tokens
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reply = sanitize_generated_text(raw)
        if hit_token_cap:
            reply = _trim_incomplete_reply(reply)
        return reply


def _per_layer_beta_ratio() -> list:
    """Constant 1.0 vector (native β on every layer). Kept for UI compatibility."""
    return [1.0 for _ in range(N_LAYERS)]


def _per_layer_temperature() -> list:
    """Constant 1.0 vector (native softmax temperature). Kept for UI compatibility."""
    return [1.0 for _ in range(N_LAYERS)]


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    return render_template(
        "index_fixed_beta.html", model=HF_MODEL, ollama_url=f"local:{DEVICE}"
    )


@app.get("/mushroom.png")
def mushroom_icon():
    return send_from_directory(app.root_path, "mushroom.png")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": HF_MODEL,
        "device": DEVICE,
        "n_layers": N_LAYERS,
        "trip": TRIP.snapshot(),
    })


@app.post("/api/trip/start")
def trip_start():
    """Begin a trip: prepend the KALEIDO system prompt only."""
    TRIP.start()
    return jsonify({
        "trip": TRIP.snapshot(),
        "per_layer_beta_ratio": _per_layer_beta_ratio(),
        "per_layer_temperature": _per_layer_temperature(),
    })


@app.post("/api/trip/annealing")
def trip_annealing():
    """Legacy route; β patch no longer exists, so this is effectively a no-op."""
    data = request.get_json(silent=True) or {}
    if "enabled" not in data:
        return jsonify({"error": "enabled (boolean) is required"}), 400
    TRIP.configure(annealing=bool(data["enabled"]))
    return jsonify({
        "trip": TRIP.snapshot(),
        "per_layer_beta_ratio": _per_layer_beta_ratio(),
        "per_layer_temperature": _per_layer_temperature(),
    })


@app.post("/api/trip/configure")
def trip_configure():
    """Legacy route; all β knobs are accepted and ignored."""
    data = request.get_json(silent=True) or {}
    TRIP.configure(**data)
    return jsonify({
        "trip": TRIP.snapshot(),
        "per_layer_beta_ratio": _per_layer_beta_ratio(),
        "per_layer_temperature": _per_layer_temperature(),
    })


@app.post("/api/trip/stop")
def trip_stop():
    TRIP.stop()
    return jsonify({
        "trip": TRIP.snapshot(),
        "per_layer_beta_ratio": _per_layer_beta_ratio(),
        "per_layer_temperature": _per_layer_temperature(),
    })


@app.post("/api/trip/advance")
def trip_advance():
    """Legacy no-op; nothing anneals."""
    data = request.get_json(silent=True) or {}
    _ = data.get("steps", None)
    TRIP.advance()
    return jsonify({
        "trip": TRIP.snapshot(),
        "per_layer_beta_ratio": _per_layer_beta_ratio(),
        "per_layer_temperature": _per_layer_temperature(),
    })


@app.get("/api/trip/state")
def trip_state():
    return jsonify({
        "trip": TRIP.snapshot(),
        "per_layer_beta_ratio": _per_layer_beta_ratio(),
        "per_layer_temperature": _per_layer_temperature(),
    })


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    history = data.get("history") or []
    if not isinstance(history, list):
        history = []

    base_temperature = float(data.get("temperature", 0.8))
    top_p = float(data.get("top_p", 0.9))
    max_new_tokens = _clamp_max_new_tokens(
        data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    )
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None

    trip_before = TRIP.snapshot()
    ratios_before = _per_layer_beta_ratio()
    temps_before = _per_layer_temperature()

    # Sampling temperature is the user/request value, unmodified.
    sampling_mult = TRIP.sampling_temperature_multiplier()
    effective_temperature = base_temperature * sampling_mult

    messages = _build_messages(history, message)

    try:
        reply = _generate(messages, max_new_tokens, effective_temperature, top_p, seed)
    except Exception as e:
        return jsonify({"error": "generation failed", "detail": str(e)}), 500

    if not reply:
        return jsonify({"error": "Empty model response"}), 502

    TRIP.advance()

    return jsonify({
        "reply": reply,
        "trip_before": trip_before,
        "trip_after": TRIP.snapshot(),
        "per_layer_beta_ratio": ratios_before,
        "per_layer_temperature": temps_before,
        "base_temperature": base_temperature,
        "sampling_temperature_multiplier": sampling_mult,
        "effective_sampling_temperature": effective_temperature,
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5001)),
        debug=True,
        use_reloader=False,
    )