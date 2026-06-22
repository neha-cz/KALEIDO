"""
Flask API for a Llama chat with REBUS-style simulated-annealing of the
attention inverse-temperature β, interpreted through modern (continuous)
Hopfield networks.

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

REBUS (RElaxed Beliefs Under pSychedelics) says a psychedelic acutely lowers
the precision of high-level priors, which then recover to baseline as the drug
clears. "Lower precision" = "flatter landscape" = "lower β". So we model the
trip as β annealing back UP to baseline:

    β(t, ℓ) = β₀ · ( 1 − dose · w(ℓ) · decay(t) )

where
    β₀        is the model's native scaling (== 1/√d, what HF passes in),
    dose ∈[0,1] sets onset depth (1 ⇒ landscape fully flattened at t=0),
    w(ℓ)      = (ℓ/(L−1))^p  weights late (high-level) layers more (REBUS),
    decay(t)  ∈ [0,1], =1 at onset, → 0 sober, following an annealing schedule:
                  exponential:  e^(−t/τ)              (continuous geometric cool)
                  logarithmic:  1/ln(t + e)           (Geman & Geman flavor)
                  linear:       max(0, 1 − t/t_final)

The attention patch MULTIPLIES HF's `scaling` (== β₀) by the ratio
β(t,ℓ)/β₀ ∈ (0,1].  ratio = 1 leaves the model untouched (sober). Smaller
ratio flattens the landscape (trip onset). The system is hottest (β lowest)
at t = 0 and anneals back to baseline over turns — REBUS recovery == cooling.

Optionally the token-sampling temperature is annealed on the SAME decay, so the
energy landscape and the output sampler "explore" together (true SA behavior),
rather than flattening only where the model looks while it still commits hard
at the output head.

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
  TRIP_DEBUG (default: 0)  set to 1 to print β ratios per generation
"""
import math
import os
import threading

import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

from noumadelic_prompt_engineering import build_trip_chat_messages, sanitize_generated_text

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ============================================================
# Config
# ============================================================
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TRIP_DEBUG = os.environ.get("TRIP_DEBUG", "0") == "1"

VALID_SCHEDULES = ("exponential", "logarithmic", "linear")

# Fixed trip profile (UI only exposes begin / +time / end).
TRIP_PRESET = {
    "dose": 1.0,
    "schedule": "exponential",
    "tau": 6.0,
    "t_final": 12.0,
    "hierarchy_power": 2.0,
    "initial_t": 1.0,
    "turn_step": 0.25,
    "couple_sampling": True,
    "sampling_T_hot": 1.6,
}

# Generation length: ~220 words ≈ 300–350 tokens; cap leaves headroom so replies
# finish naturally instead of mid-sentence truncation.
DEFAULT_MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS_CAP = 1024

# Floor on β(t,ℓ)/β₀. β=0 makes the landscape perfectly flat (uniform attention,
# every memory equally metastable) which produces incoherent output. Keep the
# ratio above this so the model stays trippy-but-coherent.
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
# Trip State — annealing of the Hopfield inverse-temperature β
# ============================================================
class BetaTripState:
    """Tracks the user's 'trip' as the attention inverse-temperature β
    annealing back to baseline.

    We store the *ratio* r(t,ℓ) = β(t,ℓ)/β₀ ∈ (0,1], because that is exactly
    what multiplies HF's per-layer `scaling`. r = 1 ⇒ untouched (sober);
    r → 0 ⇒ flat energy landscape (peak trip). The system is hottest (β lowest,
    r smallest) at t = 0 and anneals back up to r = 1 across turns.

    Time advances per assistant turn by default (each reply = one 'time unit'),
    so the user experiences a hot onset annealing toward sobriety across the
    conversation.
    """

    def __init__(self):
        self.active = False            # is a trip currently happening?
        self.dose = TRIP_PRESET["dose"]
        self.t = 0.0
        self.schedule = TRIP_PRESET["schedule"]
        self.tau = TRIP_PRESET["tau"]
        self.t_final = TRIP_PRESET["t_final"]
        self.hierarchy_power = TRIP_PRESET["hierarchy_power"]
        self.turn_step = TRIP_PRESET["turn_step"]
        self.couple_sampling = TRIP_PRESET["couple_sampling"]
        self.sampling_T_hot = TRIP_PRESET["sampling_T_hot"]
        self.annealing = True           # REBUS β patch + coupled sampling temp
        self.prompt_engineering = True  # LSD+shrooms system prompt
        self.n_layers = 0              # set after model loads

    # --------------------------------------------------------
    # Annealing decay: decay(t) ∈ [0, 1], monotonically non-increasing,
    # decay(0) = 1 (β most suppressed / landscape flattest at onset).
    # --------------------------------------------------------
    def decay(self) -> float:
        """Annealing time factor decay(t) ∈ [0, 1], decay(0) = 1.

        - exponential:  e^(−t/τ)              (continuous geometric cooling)
        - logarithmic:  1 / ln(t + e)         (Geman & Geman flavor, =1 at t=0)
        - linear:       max(0, 1 − t / t_final)
        """
        if not self.active or self.t < 0:
            return 0.0
        t = self.t
        if self.schedule == "logarithmic":
            return 1.0 / math.log(t + math.e)
        if self.schedule == "linear":
            return max(0.0, 1.0 - t / self.t_final)
        # default: exponential continuous annealing
        return math.exp(-t / self.tau)

    def layer_weight(self, layer_idx: int, n_layers: int) -> float:
        """Hierarchical weighting: later layers flattened more.

        REBUS: psychedelics preferentially reduce precision of high-level priors.
        Late layers ≈ high-level abstractions, so weight by (ℓ/L)^p.
        """
        return (layer_idx / max(n_layers - 1, 1)) ** self.hierarchy_power

    def beta_ratio(self, layer_idx: int, n_layers: int) -> float:
        """β(t,ℓ)/β₀ ∈ (0,1] — the factor that multiplies HF's `scaling`.

        ratio = 1 − dose · w(ℓ) · decay(t), floored at BETA_RATIO_FLOOR.
        1.0 ⇒ sober (untouched); → floor ⇒ maximally flat landscape (peak trip).
        """
        if not self.active or not self.annealing:
            return 1.0
        suppression = self.dose * self.layer_weight(layer_idx, n_layers) * self.decay()
        return max(BETA_RATIO_FLOOR, 1.0 - suppression)

    def sampling_temperature_multiplier(self) -> float:
        """Multiplier on the request's sampling temperature, annealed on the
        SAME decay so the output sampler 'explores' alongside the flattened
        energy landscape. 1.0 at sobriety, up to sampling_T_hot at onset.
        Returns 1.0 if coupling disabled, annealing off, or trip inactive.
        """
        if not self.active or not self.annealing or not self.couple_sampling:
            return 1.0
        return 1.0 + self.dose * self.decay() * (self.sampling_T_hot - 1.0)

    def advance(self):
        if self.active:
            self.t += self.turn_step

    def start(
        self,
        dose: float,
        schedule: str = "exponential",
        tau: float = 6.0,
        t_final: float = 12.0,
        hierarchy_power: float = 2.0,
        turn_step: float = 0.25,
        initial_t: float = 1.0,
        couple_sampling: bool = True,
        sampling_T_hot: float = 1.6,
    ):
        self.active = True
        self.dose = max(0.0, min(1.0, dose))
        self.schedule = schedule if schedule in VALID_SCHEDULES else "exponential"
        self.tau = max(0.01, tau)
        self.t_final = max(0.1, t_final)
        self.hierarchy_power = max(0.0, hierarchy_power)
        self.turn_step = max(0.01, turn_step)
        self.couple_sampling = bool(couple_sampling)
        self.sampling_T_hot = max(1.0, sampling_T_hot)
        self.t = max(0.0, initial_t)

    def configure(
        self,
        dose: float | None = None,
        schedule: str | None = None,
        tau: float | None = None,
        t_final: float | None = None,
        hierarchy_power: float | None = None,
        turn_step: float | None = None,
        couple_sampling: bool | None = None,
        sampling_T_hot: float | None = None,
        prompt_engineering: bool | None = None,
        annealing: bool | None = None,
    ):
        if dose is not None:
            self.dose = max(0.0, min(1.0, dose))
        if schedule is not None and schedule in VALID_SCHEDULES:
            self.schedule = schedule
        if tau is not None:
            self.tau = max(0.01, tau)
        if t_final is not None:
            self.t_final = max(0.1, t_final)
        if hierarchy_power is not None:
            self.hierarchy_power = max(0.0, hierarchy_power)
        if turn_step is not None:
            self.turn_step = max(0.01, turn_step)
        if couple_sampling is not None:
            self.couple_sampling = bool(couple_sampling)
        if sampling_T_hot is not None:
            self.sampling_T_hot = max(1.0, sampling_T_hot)
        if prompt_engineering is not None:
            self.prompt_engineering = bool(prompt_engineering)
        if annealing is not None:
            self.annealing = bool(annealing)

    def stop(self):
        self.active = False
        self.dose = 0.0
        self.t = 0.0

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "dose": self.dose,
            "t": self.t,
            "schedule": self.schedule,
            "tau": self.tau,
            "t_final": self.t_final,
            "hierarchy_power": self.hierarchy_power,
            "turn_step": self.turn_step,
            "couple_sampling": self.couple_sampling,
            "sampling_T_hot": self.sampling_T_hot,
            "prompt_engineering": self.prompt_engineering,
            "annealing": self.annealing,
            "decay_now": self.decay() if self.annealing else 0.0,
            "sampling_T_mult_now": self.sampling_temperature_multiplier(),
        }


TRIP = BetaTripState()


# ============================================================
# Attention patch — reads the per-layer β ratio from TRIP at every forward
# pass and MULTIPLIES the attention `scaling` (== β₀) by it.
#
# HF computes softmax(scores * scaling) with scaling = 1/√d = β₀. Multiplying
# scaling by r = β(t,ℓ)/β₀ ∈ (0,1] yields effective inverse-temperature
# β(t,ℓ) = β₀·r, flattening the Hopfield energy landscape when r < 1.
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
                  f"active={TRIP.active} dose={TRIP.dose} t={TRIP.t:.2f} "
                  f"sched={TRIP.schedule} decay={TRIP.decay():.3f}")
        # Multiplying scaling by r anneals β (r <= 1 => flatter landscape).
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

    print(f"[patch] LlamaAttention patched; β ratio reads from TRIP per layer "
          f"(n_layers={n_layers}, debug={TRIP_DEBUG})")


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
    """Build chat messages; optional trip system prompt (independent of β annealing)."""
    if TRIP.prompt_engineering:
        return build_trip_chat_messages(history, new_message)
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
    return render_template("index.html", model=HF_MODEL, ollama_url=f"local:{DEVICE}")


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
    """Enable/disable REBUS β annealing (independent of prompt engineering)."""
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
    """Update trip knobs without resetting time (trip must be active for effect)."""
    data = request.get_json(silent=True) or {}
    TRIP.configure(
        dose=float(data["dose"]) if "dose" in data else None,
        schedule=str(data["schedule"]) if "schedule" in data else None,
        tau=float(data["tau"]) if "tau" in data else None,
        t_final=float(data["t_final"]) if "t_final" in data else None,
        hierarchy_power=float(data["hierarchy_power"])
        if "hierarchy_power" in data
        else None,
        turn_step=float(data["turn_step"]) if "turn_step" in data else None,
        couple_sampling=bool(data["couple_sampling"])
        if "couple_sampling" in data
        else None,
        sampling_T_hot=float(data["sampling_T_hot"])
        if "sampling_T_hot" in data
        else None,
        prompt_engineering=bool(data["prompt_engineering"])
        if "prompt_engineering" in data
        else None,
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
    """Manually advance trip time, in case user wants to fast-forward the comedown."""
    data = request.get_json(silent=True) or {}
    steps = float(data.get("steps", TRIP.turn_step))
    if TRIP.active:
        TRIP.t += steps
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

    # Couple the token-sampling temperature to the same annealing decay, so the
    # output sampler explores alongside the flattened energy landscape.
    sampling_mult = TRIP.sampling_temperature_multiplier()
    effective_temperature = base_temperature * sampling_mult

    messages = _build_messages(history, message)

    try:
        reply = _generate(messages, max_new_tokens, effective_temperature, top_p, seed)
    except Exception as e:
        return jsonify({"error": "generation failed", "detail": str(e)}), 500

    if not reply:
        return jsonify({"error": "Empty model response"}), 502

    # Advance trip time by one turn (if active) — anneals β back toward baseline.
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