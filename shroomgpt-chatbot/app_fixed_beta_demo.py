"""
Flask API for a Llama chat with a fixed, mech-interp-derived β patch:
apply β just above the measured critical value β* to the measured demo layers.

Core idea
---------
A continuous Hopfield network has energy
    E(ξ) = −β⁻¹ · logsumexp(β · Xξ) + ½ξᵀξ + const
and its one-step update is exactly softmax(β · Xξ) — i.e. attention, with
β = 1/√d the inverse temperature. β controls the *ruggedness* of the energy
landscape:
    high β  → deep, sharp, well-separated basins → sharp single-pattern retrieval
    low  β  → shallow, merged basins → metastable mixtures of stored patterns
              (the network roams between memories instead of committing).

The earlier demo used an annealing / ODE-ish story. This version removes that.
When the trip is active, the patch simply multiplies HF's native attention
`scaling` by a fixed ratio on the measured demo layers only:

    demo layers = [8, 9, 5]
    β* ≈ 0.80625
    demo β ratio = 0.78  # just above β*, safely on the coherent side

All non-demo layers remain at ratio = 1.0. There is no time dependence, no
layer hierarchy curve, and no β recovery across turns. Sampling temperature is
left as the user/request value; it is not coupled to the β patch.

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
  TRIP_DEBUG      (default: 0) set to 1 to print β ratios per generation
  DEMO_BETA_RATIO (default: 0.85) fixed β ratio for demo layers
  DEMO_LAYERS     (default: 8,9,5) comma-separated demo-layer indices
"""
import math
import os
import threading

import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

from noumadelic_prompt_engineering import sanitize_generated_text
# from noumadelic_prompt_engineering import build_trip_chat_messages  # disabled: β-only mode

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ============================================================
# Config
# ============================================================
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TRIP_DEBUG = os.environ.get("TRIP_DEBUG", "0") == "1"

# Measured from the β/layer sweep.
BETA_STAR = float(os.environ.get("BETA_STAR", "0.80625"))
DEMO_BETA_RATIO = float(os.environ.get("DEMO_BETA_RATIO", "0.22"))
DEMO_LAYERS = tuple(
    int(x.strip())
    for x in os.environ.get("DEMO_LAYERS", "8,9,5").split(",")
    if x.strip()
)

# Fixed trip profile. Kept as a dict so the existing Flask routes/UI can stay
# the same, but these are no longer annealing parameters.
TRIP_PRESET = {
    "demo_beta_ratio": DEMO_BETA_RATIO,
    "demo_layers": DEMO_LAYERS,
    "prompt_engineering": False,  # disabled: explore fixed β patch without system prompt
    "beta_patch": True,
}

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
# Trip State — fixed β patch on measured demo layers
# ============================================================
class BetaTripState:
    """Tracks whether the fixed demo β patch is active.

    This version intentionally removes annealing / ODE behavior. When active,
    only the measured demo layers receive a fixed β ratio just above β*.

    r(ℓ) = β(ℓ)/β₀
        = demo_beta_ratio  if ℓ ∈ demo_layers and patch active
        = 1.0              otherwise
    """

    def __init__(self):
        self.active = False
        self.beta_patch = True
        self.prompt_engineering = False  # disabled: β-only mode
        self.demo_beta_ratio = max(BETA_RATIO_FLOOR, min(1.0, DEMO_BETA_RATIO))
        self.demo_layers = tuple(DEMO_LAYERS)
        self.beta_star = BETA_STAR
        self.n_layers = 0  # set after model loads

    def beta_ratio(self, layer_idx: int, n_layers: int) -> float:
        """β(ℓ)/β₀. 1.0 = sober/native attention.

        The patch is deliberately layer-local and time-independent: no decay,
        no hierarchy weighting, no annealing back to baseline.
        """
        if not self.active or not self.beta_patch:
            return 1.0
        if layer_idx in self.demo_layers:
            return max(BETA_RATIO_FLOOR, min(1.0, self.demo_beta_ratio))
        return 1.0

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
        prompt_engineering: bool = True,
        beta_patch: bool = True,
        **_ignored,
    ):
        self.active = True
        self.beta_patch = bool(beta_patch)
        self.prompt_engineering = bool(prompt_engineering)
        self.demo_beta_ratio = max(BETA_RATIO_FLOOR, min(1.0, float(demo_beta_ratio)))
        self.demo_layers = tuple(int(x) for x in demo_layers)

    def configure(
        self,
        demo_beta_ratio: float | None = None,
        demo_layers = None,
        prompt_engineering: bool | None = None,
        beta_patch: bool | None = None,
        annealing: bool | None = None,
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
        if prompt_engineering is not None:
            self.prompt_engineering = bool(prompt_engineering)
        if beta_patch is not None:
            self.beta_patch = bool(beta_patch)
        if annealing is not None:
            self.beta_patch = bool(annealing)

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
            "prompt_engineering": self.prompt_engineering,
            "sampling_T_mult_now": self.sampling_temperature_multiplier(),
            "note": "fixed β patch on demo_layers only; no annealing/ODE/time dependence",
        }


TRIP = BetaTripState()


# ============================================================
# Attention patch — reads the per-layer β ratio from TRIP at every forward
# pass and MULTIPLIES the attention `scaling` (== β₀) by it.
#
# HF computes softmax(scores * scaling) with scaling = 1/√d = β₀. Multiplying
# scaling by r = β(ℓ)/β₀ ∈ (0,1] yields effective inverse-temperature
# β(ℓ) = β₀·r, flattening the Hopfield energy landscape when r < 1.
# r = 1 leaves the model untouched.
#
# Modern transformers (≥4.43) dispatches attention through a registry rather
# than calling modeling_llama.eager_attention_forward directly, so we must
# patch the registry entry as well. We try several known locations to stay
# compatible across versions.
# ============================================================
def patch_llama_attention(n_layers: int):
    from transformers.models.llama import modeling_llama

    _original = modeling_llama.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, **kwargs):
        layer_idx = getattr(module, "layer_idx", 0)
        r = TRIP.beta_ratio(layer_idx, n_layers)  # β(t,ℓ)/β₀ ∈ (0,1]
        if TRIP_DEBUG and layer_idx == n_layers - 1:
            print(f"[trip-tick] layer={layer_idx} beta_ratio={r:.3f} "
                  f"beta_eff={scaling * r:.4f} (beta0={scaling:.4f}) "
                  f"active={TRIP.active} demo_layers={TRIP.demo_layers} "
                  f"demo_beta_ratio={TRIP.demo_beta_ratio:.4f}")
        # Multiplying scaling by r lowers β on demo layers (r <= 1 => flatter landscape).
        return _original(module, query, key, value, attention_mask,
                         scaling * r, **kwargs)

    # 1) Patch the module-level symbol (for any code path that imports it directly).
    modeling_llama.eager_attention_forward = patched

    # 2) Patch the dispatch registry, which is what LlamaAttention.forward
    #    actually calls on modern transformers. The registry has moved around
    #    across versions, so try the known locations.
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
            # Newer versions expose AttentionInterface with a dict-like internal store.
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
              "transformers version may use a different dispatch path. "
              "β-annealing may not take effect. Print the transformers "
              "version and inspect modeling_llama for the dispatch site.")

    print(f"[patch] LlamaAttention patched; fixed β ratio reads from TRIP per layer "
          f"(n_layers={n_layers}, demo_layers={DEMO_LAYERS}, "
          f"demo_beta_ratio={DEMO_BETA_RATIO}, beta_star={BETA_STAR}, "
          f"debug={TRIP_DEBUG})")


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
    """Plain chat messages only — no trip system prompt (β-only exploration)."""
    # if TRIP.prompt_engineering:
    #     return build_trip_chat_messages(history, new_message)
    out = []
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
    """For visualization: current β(t,ℓ)/β₀ at each layer (1.0 = sober)."""
    return [TRIP.beta_ratio(i, N_LAYERS) for i in range(N_LAYERS)]


def _per_layer_temperature() -> list:
    """For visualization: current effective softmax temperature T = β₀/β = 1/ratio
    at each layer (1.0 = sober, larger = hotter/flatter). This is the reciprocal
    of the β ratio and is provided for convenience / backward compatibility.
    """
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


@app.post("/api/trip/prompt_engineering")
def trip_prompt_engineering():
    """Enable/disable LSD+shrooms system prompt (independent of β annealing)."""
    data = request.get_json(silent=True) or {}
    if "enabled" not in data:
        return jsonify({"error": "enabled (boolean) is required"}), 400
    TRIP.configure(prompt_engineering=bool(data["enabled"]))
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
    """Update fixed β demo knobs without changing active/inactive state."""
    data = request.get_json(silent=True) or {}
    TRIP.configure(
        demo_beta_ratio=float(data["demo_beta_ratio"])
        if "demo_beta_ratio" in data
        else None,
        demo_layers=data.get("demo_layers") if "demo_layers" in data else None,
        prompt_engineering=bool(data["prompt_engineering"])
        if "prompt_engineering" in data
        else None,
        beta_patch=bool(data["beta_patch"]) if "beta_patch" in data else None,
        annealing=bool(data["annealing"]) if "annealing" in data else None,
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
    # Fixed β patch has no time state; endpoint kept for UI compatibility.
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