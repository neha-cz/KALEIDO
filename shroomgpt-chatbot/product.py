"""
Flask API for a Llama chat with TWO stacked mech-interp interventions:

  1. A fixed β patch (attention inverse-temperature flattening) on demo layers,
     just above the measured critical value β*.
  2. A persona-vector residual steer toward a "dissolved" voice, added at a
     single layer on decode tokens (method of Chen et al. 2025, "Persona
     Vectors"), using a vector extracted from this model.

Core idea (β patch)
-------------------
A continuous Hopfield network has energy
    E(ξ) = −β⁻¹ · logsumexp(β · Xξ) + ½ξᵀξ + const
and its one-step update is exactly softmax(β · Xξ) — i.e. attention, with
β = 1/√d the inverse temperature. Low β → shallow, merged basins → metastable
mixtures of stored patterns. When the trip is active, the patch multiplies HF's
native attention `scaling` by a fixed ratio on the demo layers only.

Persona steer
-------------
A persona vector (response-token mean activation difference between a
"dissolved" and a "normal" system-prompt condition) is added to the residual
stream at a single layer on freshly decoded tokens:
    resid_new = resid + coef · v[layer]
This steers the assistant's *voice* toward a dissolved/egoless register. It does
not give the model a self to dissolve; for a text model the generated language
is the phenomenon, and this is a real activation-space lever on that language.

Both interventions fire in the same forward pass at different sites (β in
attention on early demo layers; steer in the residual stream at a late layer)
and are independently toggleable. The persona vector is extracted with β OFF and
deployed here with β ON (the demo stack).

Setup
-----
  1. Accept Llama license at huggingface.co/meta-llama/Llama-3.2-1B-Instruct
  2. pip install flask flask-cors torch transformers accelerate
  3. huggingface-cli login
  4. python app.py

Environment variables
----------------------
  HF_MODEL          (default: meta-llama/Llama-3.2-1B-Instruct)
  DEVICE            (default: auto)
  PORT              (default: 5001)
  TRIP_DEBUG        (default: 0) set to 1 to print β ratios per generation
  DEMO_BETA_RATIO   (default: 0.65) fixed β ratio for demo layers
  DEMO_LAYERS       (default: 2,3) comma-separated demo-layer indices
  PERSONA_VECTOR    (default: persona_vectors/dissolved_response_avg_diff.pt)
  PERSONA_LAYER     (default: 13) residual layer to steer at
  PERSONA_COEF      (default: 1.4) steering coefficient
  PERSONA_ON        (default: 1) whether the steer engages with the trip
"""
import math
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
TRIP_DEBUG = os.environ.get("TRIP_DEBUG", "0") == "1"
APPLY_BETA_ON_PREFILL = os.environ.get("APPLY_BETA_ON_PREFILL", "0") == "1"

# Measured from the β/layer sweep.
BETA_STAR = float(os.environ.get("BETA_STAR", "0.80625"))
DEMO_BETA_RATIO = float(os.environ.get("DEMO_BETA_RATIO", "0.65"))
DEMO_LAYERS = tuple(
    int(x.strip())
    for x in os.environ.get("DEMO_LAYERS", "2,3").split(",")
    if x.strip()
)

# Persona-vector steering config.
PERSONA_VECTOR_PATH = os.environ.get(
    "PERSONA_VECTOR", "persona_vectors/dissolved_response_avg_diff.pt"
)
PERSONA_LAYER = int(os.environ.get("PERSONA_LAYER", "13"))
PERSONA_COEF = float(os.environ.get("PERSONA_COEF", "1.4"))
PERSONA_ON = os.environ.get("PERSONA_ON", "1") == "1"
# Safety clamp so a runaway coef from the UI can't fully shred the output.
PERSONA_COEF_CAP = float(os.environ.get("PERSONA_COEF_CAP", "2.5"))

# Fixed trip profile. Kept as a dict so the existing Flask routes/UI can stay
# the same, but these are no longer annealing parameters.
TRIP_PRESET = {
    "demo_beta_ratio": DEMO_BETA_RATIO,
    "demo_layers": DEMO_LAYERS,
    "beta_patch": True,
    "persona_steer": PERSONA_ON,
    "persona_coef": PERSONA_COEF,
}

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

# KALEIDO_SYSTEM_PROMPT = ("Your name is KALEIDO. Do not mention it unless explicitly asked.")

# Generation length: ~220 words ≈ 300–350 tokens; cap leaves headroom so replies
# finish naturally instead of mid-sentence truncation.
DEFAULT_MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS_CAP = 1024

# Safety clamp for the fixed β ratio. β=0 makes the landscape perfectly flat
# and usually incoherent; the demo should sit just above β*.
BETA_RATIO_FLOOR = 0.05


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
# Trip State — fixed β patch on measured demo layers + persona steer
# ============================================================
class BetaTripState:
    """Tracks whether the fixed demo β patch and the persona steer are active.

    β patch: when active, the measured demo layers receive a fixed β ratio just
    above β*. Layer-local and time-independent (no annealing).

    Persona steer: when active, a persona vector is added to the residual stream
    at PERSONA_LAYER on decode tokens, with strength persona_coef.
    """

    def __init__(self):
        self.active = False
        self.beta_patch = True
        self.demo_beta_ratio = max(BETA_RATIO_FLOOR, min(1.0, DEMO_BETA_RATIO))
        self.demo_layers = tuple(DEMO_LAYERS)
        self.beta_star = BETA_STAR
        self.n_layers = 0  # set after model loads
        # persona steer state
        self.persona_steer = bool(PERSONA_ON)
        self.persona_layer = int(PERSONA_LAYER)
        self.persona_coef = float(PERSONA_COEF)

    def beta_ratio(self, layer_idx: int, n_layers: int) -> float:
        """β(ℓ)/β₀. 1.0 = sober/native attention."""
        if not self.active or not self.beta_patch:
            return 1.0
        if layer_idx in self.demo_layers:
            return max(BETA_RATIO_FLOOR, min(1.0, self.demo_beta_ratio))
        return 1.0

    def persona_active(self) -> bool:
        """Whether the persona steer should fire right now."""
        return bool(self.active and self.persona_steer and self.persona_coef != 0.0)

    def current_persona_coef(self) -> float:
        return max(0.0, min(PERSONA_COEF_CAP, float(self.persona_coef)))

    def sampling_temperature_multiplier(self) -> float:
        """Sampling temperature is no longer coupled to the β patch."""
        return 1.0

    def advance(self):
        """No-op kept for route compatibility; fixed β does not anneal over turns."""
        return None

    def start(
        self,
        demo_beta_ratio: float = DEMO_BETA_RATIO,
        demo_layers = DEMO_LAYERS,
        beta_patch: bool = True,
        persona_steer: bool = None,
        persona_coef: float = None,
        **_ignored,
    ):
        self.active = True
        self.beta_patch = bool(beta_patch)
        self.demo_beta_ratio = max(BETA_RATIO_FLOOR, min(1.0, float(demo_beta_ratio)))
        self.demo_layers = tuple(int(x) for x in demo_layers)
        if persona_steer is not None:
            self.persona_steer = bool(persona_steer)
        if persona_coef is not None:
            self.persona_coef = float(persona_coef)

    def configure(
        self,
        demo_beta_ratio: float | None = None,
        demo_layers = None,
        beta_patch: bool | None = None,
        annealing: bool | None = None,
        persona_steer: bool | None = None,
        persona_coef: float | None = None,
        persona_layer: int | None = None,
        **_ignored,
    ):
        # `annealing` is accepted for backward compatibility with the existing UI/API.
        # It now means "enable/disable the fixed β patch".
        if demo_beta_ratio is not None:
            self.demo_beta_ratio = max(BETA_RATIO_FLOOR, min(1.0, float(demo_beta_ratio)))
        if demo_layers is not None:
            if isinstance(demo_layers, str):
                demo_layers = [x.strip() for x in demo_layers.split(",") if x.strip()]
            self.demo_layers = tuple(int(x) for x in demo_layers)
        if beta_patch is not None:
            self.beta_patch = bool(beta_patch)
        if annealing is not None:
            self.beta_patch = bool(annealing)
        if persona_steer is not None:
            self.persona_steer = bool(persona_steer)
        if persona_coef is not None:
            self.persona_coef = float(persona_coef)
        if persona_layer is not None:
            self.persona_layer = int(persona_layer)

    def stop(self):
        self.active = False

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "beta_patch": self.beta_patch,
            "annealing": self.beta_patch,  # backward-compatible field name
            "beta_star": self.beta_star,
            "demo_beta_ratio": self.demo_beta_ratio,
            "demo_layers": list(self.demo_layers),
            "persona_steer": self.persona_steer,
            "persona_layer": self.persona_layer,
            "persona_coef": self.persona_coef,
            "persona_active_now": self.persona_active(),
            "sampling_T_mult_now": self.sampling_temperature_multiplier(),
            "note": "fixed β patch on demo_layers + persona-vector residual steer "
                    "at persona_layer; no annealing/ODE/time dependence",
        }


TRIP = BetaTripState()


# ============================================================
# Attention patch — reads the per-layer β ratio from TRIP at every forward
# pass and MULTIPLIES the attention `scaling` (== β₀) by it.
# ============================================================
def patch_llama_attention(n_layers: int):
    from transformers.models.llama import modeling_llama

    _original = modeling_llama.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, **kwargs):
        layer_idx = getattr(module, "layer_idx", 0)
        r = TRIP.beta_ratio(layer_idx, n_layers)
        if r < 1.0 and not APPLY_BETA_ON_PREFILL:
            # Keep prompt encoding (incl. system prompt) at native β for stability.
            # Apply β intervention only during autoregressive decode.
            seq_len = query.shape[2]
            is_prefill = seq_len > 1
            if is_prefill:
                r = 1.0
        if TRIP_DEBUG and layer_idx == n_layers - 1:
            print(f"[trip-tick] layer={layer_idx} beta_ratio={r:.3f} "
                  f"beta_eff={scaling * r:.4f} (beta0={scaling:.4f}) "
                  f"active={TRIP.active} demo_layers={TRIP.demo_layers} "
                  f"demo_beta_ratio={TRIP.demo_beta_ratio:.4f} "
                  f"seq={query.shape[2]} decode={r < 1.0}")
        return _original(module, query, key, value, attention_mask,
                         scaling * r, **kwargs)

    # 1) Patch the module-level symbol (for any code path that imports it directly).
    modeling_llama.eager_attention_forward = patched

    # 2) Patch the dispatch registry, which is what LlamaAttention.forward
    #    actually calls on modern transformers.
    patched_registry = False
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = patched
        patched_registry = True
        print("[patch] registered in transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS")
    except Exception:
        pass

    if not patched_registry:
        try:
            from transformers.modeling_utils import AttentionInterface
            if hasattr(AttentionInterface, "_global_mapping"):
                AttentionInterface._global_mapping["eager"] = patched
                patched_registry = True
                print("[patch] registered via AttentionInterface._global_mapping")
            elif hasattr(AttentionInterface, "register"):
                AttentionInterface.register("eager", patched)
                patched_registry = True
                print("[patch] registered via AttentionInterface.register")
        except Exception:
            pass

    if not patched_registry:
        print("[patch] WARNING: could not patch attention registry — your "
              "transformers version may use a different dispatch path.")

    print(f"[patch] LlamaAttention patched; fixed β ratio reads from TRIP per layer "
          f"(n_layers={n_layers}, demo_layers={DEMO_LAYERS}, "
          f"demo_beta_ratio={DEMO_BETA_RATIO}, beta_star={BETA_STAR}, "
          f"debug={TRIP_DEBUG}, prefill_beta={APPLY_BETA_ON_PREFILL})")


# ============================================================
# Persona-vector residual steering hook.
#
# Adds coef · v[persona_layer] to the residual-stream OUTPUT of that decoder
# block, on decode tokens only (response-token steering, matching the method
# the vector was extracted with). Reads coef/active from TRIP at every forward
# pass so the UI can tune it live, exactly like the β patch.
# ============================================================
PERSONA_VECTOR = None          # full [n_layers, hidden] tensor, or None if unavailable
PERSONA_LAYER_VEC = None       # the single chosen layer's [hidden] slice (on device)


def load_persona_vector():
    """Load the persona vector from disk; return the full tensor or None if
    missing/misshaped so the app still runs without the steer."""
    if not os.path.exists(PERSONA_VECTOR_PATH):
        print(f"[persona] WARNING: vector not found at {PERSONA_VECTOR_PATH}; "
              f"persona steer DISABLED (β patch still works).")
        return None
    try:
        v = torch.load(PERSONA_VECTOR_PATH, map_location="cpu")
    except Exception as e:
        print(f"[persona] WARNING: failed to load vector ({e}); steer DISABLED.")
        return None
    if not torch.is_tensor(v) or v.dim() != 2:
        print(f"[persona] WARNING: expected [n_layers, hidden] tensor, got "
              f"{type(v).__name__}/{getattr(v, 'shape', None)}; steer DISABLED.")
        return None
    return v


def install_persona_hook(layer_idx: int):
    """Register a forward hook on the chosen decoder layer that adds the steer."""
    layer = model.model.layers[layer_idx]

    def hook(module, args, output):
        if not TRIP.persona_active():
            return None
        if PERSONA_LAYER_VEC is None:
            return None
        hs = output[0] if isinstance(output, tuple) else output
        # response-token steering: only decode steps (seq_len == 1), skip prefill
        if hs.shape[1] != 1:
            return None
        coef = TRIP.current_persona_coef()
        if coef == 0.0:
            return None
        hs_new = hs + coef * PERSONA_LAYER_VEC.to(hs.device, hs.dtype)
        if isinstance(output, tuple):
            return (hs_new,) + tuple(output[1:])
        return hs_new

    layer.register_forward_hook(hook)
    print(f"[persona] steer hook installed on layer {layer_idx} "
          f"(coef={PERSONA_COEF}, on={PERSONA_ON})")


# ============================================================
# Model loading
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
    attn_implementation="eager",
)
if DEVICE == "cpu":
    model = model.to(DEVICE)
model.eval()

N_LAYERS = model.config.num_hidden_layers
TRIP.n_layers = N_LAYERS
patch_llama_attention(N_LAYERS)

# Load + install the persona vector steer.
PERSONA_VECTOR = load_persona_vector()
if PERSONA_VECTOR is not None:
    if not (0 <= PERSONA_LAYER < N_LAYERS):
        print(f"[persona] WARNING: PERSONA_LAYER {PERSONA_LAYER} out of range "
              f"[0,{N_LAYERS}); steer DISABLED.")
        PERSONA_VECTOR = None
        TRIP.persona_steer = False
    elif PERSONA_VECTOR.shape[0] < N_LAYERS:
        print(f"[persona] WARNING: vector has {PERSONA_VECTOR.shape[0]} layers, "
              f"model has {N_LAYERS}; steer DISABLED.")
        PERSONA_VECTOR = None
        TRIP.persona_steer = False
    else:
        PERSONA_LAYER_VEC = PERSONA_VECTOR[PERSONA_LAYER].to(DEVICE, DTYPE)
        install_persona_hook(PERSONA_LAYER)
        print(f"[persona] ready: layer {PERSONA_LAYER}, "
              f"||v||={torch.linalg.vector_norm(PERSONA_LAYER_VEC.float()).item():.3f}")
else:
    TRIP.persona_steer = False  # nothing to steer with

print(f"[load] Ready. {N_LAYERS} layers on {DEVICE}.")


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
def _trim_incomplete_reply(text: str) -> str:
    """If generation hit the token cap, drop a dangling final fragment.

    Cuts back to the last sentence-ending punctuation (. ! ? …) anywhere in the
    text, so a trailing partial sentence or half-word (e.g. "Like a dro") is
    removed cleanly. Quote/paren characters that legitimately follow terminal
    punctuation are kept.
    """
    text = (text or "").strip()
    if not text:
        return text
    # already ends cleanly (allowing a closing quote/paren after the punctuation)
    if text[-1] in ".!?…" or (len(text) >= 2 and text[-1] in "\"')" and text[-2] in ".!?…"):
        return text
    # find the last terminal punctuation mark anywhere in the text
    last = max((text.rfind(p) for p in ".!?…"), default=-1)
    if last == -1:
        return text  # no sentence boundary at all; leave as-is
    end = last + 1
    # keep an immediately following closing quote/paren if present
    if end < len(text) and text[end] in "\"')":
        end += 1
    return text[:end].strip()


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
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reply = sanitize_generated_text(raw)
        # Always drop a dangling final fragment: steered output can stop on a
        # half-sentence ("Like a dro") without hitting the token cap. The trim
        # is a no-op when the reply already ends on terminal punctuation.
        trimmed = _trim_incomplete_reply(reply)
        # Guard: if trimming would gut the reply (e.g. a short reply with no
        # sentence boundary), keep the original rather than returning near-empty.
        if trimmed and len(trimmed) >= 0.5 * len(reply):
            reply = trimmed
        return reply


def _per_layer_beta_ratio() -> list:
    """For visualization: current β(t,ℓ)/β₀ at each layer (1.0 = sober)."""
    return [TRIP.beta_ratio(i, N_LAYERS) for i in range(N_LAYERS)]


def _per_layer_temperature() -> list:
    """For visualization: current effective softmax temperature T = β₀/β = 1/ratio."""
    return [1.0 / TRIP.beta_ratio(i, N_LAYERS) for i in range(N_LAYERS)]


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
        "persona_vector_loaded": PERSONA_LAYER_VEC is not None,
        "trip": TRIP.snapshot(),
    })


@app.post("/api/trip/start")
def trip_start():
    """Begin a trip using the fixed TRIP_PRESET profile."""
    TRIP.start(**TRIP_PRESET)
    return jsonify({
        "trip": TRIP.snapshot(),
        "per_layer_beta_ratio": _per_layer_beta_ratio(),
        "per_layer_temperature": _per_layer_temperature(),
    })


@app.post("/api/trip/annealing")
def trip_annealing():
    """Enable/disable the fixed β patch (legacy route name)."""
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
    """Update fixed β + persona steer knobs without changing active state."""
    data = request.get_json(silent=True) or {}
    TRIP.configure(
        demo_beta_ratio=float(data["demo_beta_ratio"])
        if "demo_beta_ratio" in data
        else None,
        demo_layers=data.get("demo_layers") if "demo_layers" in data else None,
        beta_patch=bool(data["beta_patch"]) if "beta_patch" in data else None,
        annealing=bool(data["annealing"]) if "annealing" in data else None,
        persona_steer=bool(data["persona_steer"]) if "persona_steer" in data else None,
        persona_coef=float(data["persona_coef"]) if "persona_coef" in data else None,
    )
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
    """Legacy no-op; fixed β patch has no comedown/annealing time."""
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

    # Snapshot β state *before* generation, since the trip advances after.
    trip_before = TRIP.snapshot()
    ratios_before = _per_layer_beta_ratio()
    temps_before = _per_layer_temperature()

    # Sampling temperature is intentionally not coupled to the fixed β patch.
    sampling_mult = TRIP.sampling_temperature_multiplier()
    effective_temperature = base_temperature * sampling_mult

    messages = _build_messages(history, message)

    try:
        reply = _generate(messages, max_new_tokens, effective_temperature, top_p, seed)
    except Exception as e:
        return jsonify({"error": "generation failed", "detail": str(e)}), 500

    if not reply:
        return jsonify({"error": "Empty model response"}), 502

    # Kept for compatibility; fixed β does not advance/anneal over turns.
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