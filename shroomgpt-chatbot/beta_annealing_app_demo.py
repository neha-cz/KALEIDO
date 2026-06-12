"""
Flask API for a Llama chat with a mech-interp-derived β patch on the measured
demo layers, now supporting WITHIN-REPLY β ANNEALING as a product-flavor knob.

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

Modes
-----
The patch multiplies HF's native attention `scaling` (= β₀ = 1/√d) by a ratio
r ∈ (0,1] on the demo layers only. Two modes:

  static   : r = DEMO_BETA_RATIO, fixed for the whole reply.
  annealed : r follows a schedule within each reply, ramping from a hot end
             (low β, exploratory) to a cold end (high β, committed). β is read
             live per token from TRIP._anneal_t, set by a manual decode loop.

IMPORTANT — honest framing of annealing:
Our mech-interp screen found that scheduled β cooling does NOT reach a
drift/coherence region that static β cannot — it traces the same trade-off
curve. Annealing is therefore a *flavor* knob: it changes the token-by-token
dynamics (and so the texture of replies), but we do not claim it produces more
"lucid" or more "psychedelic" output than static β. Within-reply annealing also
requires manual token-by-token decoding (model.generate fixes β per call), so
it is slower than the static path, especially on CPU.

Defaults are STATIC unless ANNEAL_ON=1 (or the /api/trip/configure or
/api/trip/annealing route enables it).

Setup
-----
  1. Accept Llama license at huggingface.co/meta-llama/Llama-3.2-1B-Instruct
  2. pip install flask flask-cors torch transformers accelerate
  3. huggingface-cli login
  4. python app.py

Environment variables
----------------------
  HF_MODEL        (default: meta-llama/Llama-3.2-1B-Instruct)
  DEVICE          (default: auto)
  PORT            (default: 5001)
  TRIP_DEBUG      (default: 0) set to 1 to print β ratios per generation
  DEMO_BETA_RATIO (default: 0.40) fixed β ratio for demo layers (static mode)
  DEMO_LAYERS     (default: 2,3) comma-separated demo-layer indices
  ANNEAL_ON       (default: 0) set to 1 to enable within-reply β annealing
  ANNEAL_LO       (default: 0.45) hot end of the schedule (reply start)
  ANNEAL_HI       (default: 1.0) cold end of the schedule (reply end)
  ANNEAL_SCHEDULE (default: cool_linear) one of:
                  cool_linear, cool_exp, cool_late, cool_early, warm_linear
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

# Measured from the β/layer sweep.
BETA_STAR = float(os.environ.get("BETA_STAR", "0.80625"))
DEMO_BETA_RATIO = float(os.environ.get("DEMO_BETA_RATIO", "0.40"))
DEMO_LAYERS = tuple(
    int(x.strip())
    for x in os.environ.get("DEMO_LAYERS", "2,3").split(",")
    if x.strip()
)

# Within-reply β annealing (product flavor; off by default → static β).
ANNEAL_LO = float(os.environ.get("ANNEAL_LO", "0.65"))        # hot end (reply start)
ANNEAL_HI = float(os.environ.get("ANNEAL_HI", "1.0"))         # cold end (reply end)
ANNEAL_SCHEDULE = os.environ.get("ANNEAL_SCHEDULE", "cool_linear")
ANNEAL_ON_BY_DEFAULT = os.environ.get("ANNEAL_ON", "0") == "1"

# Fixed trip profile. Kept as a dict so the existing Flask routes/UI can stay
# the same.
TRIP_PRESET = {
    "demo_beta_ratio": DEMO_BETA_RATIO,
    "demo_layers": DEMO_LAYERS,
    "prompt_engineering": True,
    "beta_patch": True,
    "annealing": ANNEAL_ON_BY_DEFAULT,
    "anneal_lo": ANNEAL_LO,
    "anneal_hi": ANNEAL_HI,
    "anneal_schedule": ANNEAL_SCHEDULE,
}

# Generation length: ~220 words ≈ 300–350 tokens; cap leaves headroom so replies
# finish naturally instead of mid-sentence truncation.
DEFAULT_MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS_CAP = 1024

# Safety clamp for the β ratio. β=0 makes the landscape perfectly flat and
# usually incoherent; the demo should sit above β*.
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
# Schedules: f(t) -> β ratio for t in [0,1] (0 = reply start, 1 = reply end)
# ============================================================
def _make_schedule(name: str, lo: float, hi: float):
    """Cooling schedules ramp hot (low β, exploratory) -> cold (high β, committed).
    `warm_linear` is the reverse, kept as a flavor option.
    """
    name = (name or "cool_linear").lower()
    table = {
        "cool_linear": lambda t: lo + (hi - lo) * t,
        "cool_exp":    lambda t: lo + (hi - lo) * (1.0 - math.exp(-3.0 * t)) / (1.0 - math.exp(-3.0)),
        "cool_late":   lambda t: lo + (hi - lo) * (t ** 2),       # stay hot, sharpen late
        "cool_early":  lambda t: lo + (hi - lo) * math.sqrt(t),   # sharpen early
        "warm_linear": lambda t: hi + (lo - hi) * t,
    }
    return table.get(name, table["cool_linear"])


# ============================================================
# Trip State — β patch on measured demo layers, static or annealed
# ============================================================
class BetaTripState:
    """Tracks the demo β patch. Supports two modes:

      - static  : fixed β ratio on demo layers.
      - annealed : β ratio ramps within each reply along a schedule, read live
                   off `_anneal_t` (set by the manual decode loop per token).

    NOTE ON CLAIMS: the annealing schedule is a *product flavor* knob. Our
    mech-interp screen found scheduled cooling does NOT reach a drift/coherence
    region static β can't — it traces the same trade-off. It does change the
    token-by-token dynamics, which gives a distinct texture. We do not claim
    annealing produces "more lucid" or "more psychedelic" output.

    r(ℓ) = β(ℓ)/β₀
        = schedule(_anneal_t)  if annealing and ℓ ∈ demo_layers and active
        = demo_beta_ratio      if static    and ℓ ∈ demo_layers and active
        = 1.0                  otherwise
    """

    def __init__(self):
        self.active = False
        self.beta_patch = True
        self.prompt_engineering = True  # LSD+shrooms system prompt
        self.demo_beta_ratio = self._clamp(DEMO_BETA_RATIO)
        self.demo_layers = tuple(DEMO_LAYERS)
        self.beta_star = BETA_STAR
        self.n_layers = 0  # set after model loads

        # --- annealing config ---
        self.annealing = bool(ANNEAL_ON_BY_DEFAULT)
        self.anneal_lo = self._clamp(ANNEAL_LO)
        self.anneal_hi = self._clamp(ANNEAL_HI)
        self.anneal_schedule = ANNEAL_SCHEDULE
        self._anneal_t = 0.0  # live progress in [0,1], set per token by decode loop

    @staticmethod
    def _clamp(x):
        return max(BETA_RATIO_FLOOR, min(1.0, float(x)))

    def beta_ratio(self, layer_idx: int, n_layers: int = None) -> float:
        """β(ℓ)/β₀. 1.0 = sober/native attention.

        Layer-local: only demo layers are touched. Time dependence comes solely
        from the annealing schedule when enabled; otherwise the ratio is fixed.
        """
        if not self.active or not self.beta_patch:
            return 1.0
        if layer_idx not in self.demo_layers:
            return 1.0
        if self.annealing:
            f = _make_schedule(self.anneal_schedule, self.anneal_lo, self.anneal_hi)
            return self._clamp(f(self._anneal_t))
        return self._clamp(self.demo_beta_ratio)

    def set_anneal_t(self, t: float):
        """Called by the decode loop before each token; t in [0,1]."""
        self._anneal_t = max(0.0, min(1.0, float(t)))

    def sampling_temperature_multiplier(self) -> float:
        """Sampling temperature is not coupled to the β patch."""
        return 1.0

    def advance(self):
        """No-op kept for route compatibility; β does not anneal across turns."""
        return None

    def start(self, demo_beta_ratio=DEMO_BETA_RATIO, demo_layers=DEMO_LAYERS,
              prompt_engineering=True, beta_patch=True, annealing=None,
              anneal_lo=None, anneal_hi=None, anneal_schedule=None, **_ignored):
        self.active = True
        self.beta_patch = bool(beta_patch)
        self.prompt_engineering = bool(prompt_engineering)
        self.demo_beta_ratio = self._clamp(demo_beta_ratio)
        self.demo_layers = tuple(int(x) for x in demo_layers)
        if annealing is not None:
            self.annealing = bool(annealing)
        if anneal_lo is not None:
            self.anneal_lo = self._clamp(anneal_lo)
        if anneal_hi is not None:
            self.anneal_hi = self._clamp(anneal_hi)
        if anneal_schedule is not None:
            self.anneal_schedule = str(anneal_schedule)

    def configure(self, demo_beta_ratio=None, demo_layers=None,
                  prompt_engineering=None, beta_patch=None,
                  annealing=None, anneal_lo=None, anneal_hi=None,
                  anneal_schedule=None, **_ignored):
        if demo_beta_ratio is not None:
            self.demo_beta_ratio = self._clamp(demo_beta_ratio)
        if demo_layers is not None:
            if isinstance(demo_layers, str):
                demo_layers = [x.strip() for x in demo_layers.split(",") if x.strip()]
            self.demo_layers = tuple(int(x) for x in demo_layers)
        if prompt_engineering is not None:
            self.prompt_engineering = bool(prompt_engineering)
        if beta_patch is not None:
            self.beta_patch = bool(beta_patch)
        # `annealing` toggles the within-reply schedule (NOT the whole patch).
        if annealing is not None:
            self.annealing = bool(annealing)
        if anneal_lo is not None:
            self.anneal_lo = self._clamp(anneal_lo)
        if anneal_hi is not None:
            self.anneal_hi = self._clamp(anneal_hi)
        if anneal_schedule is not None:
            self.anneal_schedule = str(anneal_schedule)

    def stop(self):
        self.active = False
        self._anneal_t = 0.0

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "beta_patch": self.beta_patch,
            "annealing": self.annealing,
            "anneal_lo": self.anneal_lo,
            "anneal_hi": self.anneal_hi,
            "anneal_schedule": self.anneal_schedule,
            "beta_star": self.beta_star,
            "demo_beta_ratio": self.demo_beta_ratio,
            "demo_layers": list(self.demo_layers),
            "prompt_engineering": self.prompt_engineering,
            "sampling_T_mult_now": self.sampling_temperature_multiplier(),
            "note": ("within-reply β annealing on demo_layers when annealing=True; "
                     "flavor only — does not reach states static β can't (see research)"),
        }


TRIP = BetaTripState()


# ============================================================
# Attention patch — reads the per-layer β ratio from TRIP at every forward
# pass and MULTIPLIES the attention `scaling` (== β₀) by it.
#
# HF computes softmax(scores * scaling) with scaling = 1/√d = β₀. Multiplying
# scaling by r = β(ℓ)/β₀ ∈ (0,1] yields effective inverse-temperature
# β(ℓ) = β₀·r, flattening the Hopfield energy landscape when r < 1. r = 1
# leaves the model untouched. When annealing, r changes per token because the
# decode loop updates TRIP._anneal_t and beta_ratio() reads it live.
# ============================================================
def patch_llama_attention(n_layers: int):
    from transformers.models.llama import modeling_llama

    _original = modeling_llama.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, **kwargs):
        layer_idx = getattr(module, "layer_idx", 0)
        r = TRIP.beta_ratio(layer_idx, n_layers)  # β(ℓ)/β₀ ∈ (0,1]
        if TRIP_DEBUG and layer_idx == n_layers - 1:
            print(f"[trip-tick] layer={layer_idx} beta_ratio={r:.3f} "
                  f"beta_eff={scaling * r:.4f} (beta0={scaling:.4f}) "
                  f"active={TRIP.active} annealing={TRIP.annealing} "
                  f"anneal_t={TRIP._anneal_t:.3f} demo_layers={TRIP.demo_layers}")
        return _original(module, query, key, value, attention_mask,
                         scaling * r, **kwargs)

    # 1) Patch the module-level symbol.
    modeling_llama.eager_attention_forward = patched

    # 2) Patch the dispatch registry (what LlamaAttention.forward calls on
    #    modern transformers). The registry has moved across versions.
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
              "transformers version may use a different dispatch path. "
              "β patch may not take effect. Print the transformers version and "
              "inspect modeling_llama for the dispatch site.")

    print(f"[patch] LlamaAttention patched; β ratio reads from TRIP per layer "
          f"(n_layers={n_layers}, demo_layers={DEMO_LAYERS}, "
          f"demo_beta_ratio={DEMO_BETA_RATIO}, anneal_on={ANNEAL_ON_BY_DEFAULT}, "
          f"schedule={ANNEAL_SCHEDULE} [{ANNEAL_LO}->{ANNEAL_HI}], debug={TRIP_DEBUG})")


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
    """Build chat messages; optional trip system prompt (independent of β patch)."""
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
    """Static-β path: uses optimized model.generate (β fixed for the call)."""
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


@torch.no_grad()
def _generate_annealed(messages, max_new_tokens, temperature, top_p, seed):
    """Annealed-β path: manual token-by-token decode that ramps TRIP._anneal_t
    from 0->1 across the reply, so the demo layers' β follows the schedule
    within a single reply. Supports sampling (temperature, top_p). Slower than
    _generate because it cannot use model.generate (β must change per token).
    """
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
        cur_ids = inputs["input_ids"]
        cur_attn = inputs.get("attention_mask")

        eos_ids = {tokenizer.eos_token_id}
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot != tokenizer.unk_token_id:
            eos_ids.add(eot)

        max_new_tokens = _clamp_max_new_tokens(max_new_tokens)
        do_sample = temperature > 0
        temp = max(temperature, 1e-5)

        past = None
        generated = []
        hit_cap = True
        try:
            for step in range(max_new_tokens):
                t = step / max(max_new_tokens - 1, 1)
                TRIP.set_anneal_t(t)

                out = model(input_ids=cur_ids, attention_mask=cur_attn,
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]

                if do_sample:
                    logits = logits / temp
                    if top_p is not None and 0 < top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                        probs = torch.softmax(sorted_logits, dim=-1)
                        cum = torch.cumsum(probs, dim=-1)
                        remove = cum > top_p
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        sorted_logits[remove] = float("-inf")
                        logits = torch.full_like(logits, float("-inf")).scatter(
                            -1, sorted_idx, sorted_logits)
                    probs = torch.softmax(logits, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_id = int(logits.argmax(dim=-1).item())

                generated.append(next_id)
                if next_id in eos_ids:
                    hit_cap = False
                    break
                cur_ids = torch.tensor([[next_id]], device=model.device)
                if cur_attn is not None:
                    cur_attn = torch.cat(
                        [cur_attn, torch.ones((1, 1), dtype=cur_attn.dtype,
                                              device=model.device)], dim=1)
        finally:
            TRIP.set_anneal_t(0.0)  # always reset, even if generation errors

        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        reply = sanitize_generated_text(raw)
        if hit_cap:
            reply = _trim_incomplete_reply(reply)
        return reply


def _generate_dispatch(messages, max_new_tokens, temperature, top_p, seed):
    """Route to the annealed loop only when annealing is actually active."""
    if TRIP.active and TRIP.beta_patch and TRIP.annealing:
        return _generate_annealed(messages, max_new_tokens, temperature, top_p, seed)
    return _generate(messages, max_new_tokens, temperature, top_p, seed)


def _per_layer_beta_ratio() -> list:
    """For visualization: current β(ℓ)/β₀ at each layer (1.0 = sober).

    When annealing, this reflects β at the current _anneal_t (0.0 between
    replies → hot end of the schedule on demo layers).
    """
    return [TRIP.beta_ratio(i, N_LAYERS) for i in range(N_LAYERS)]


def _per_layer_temperature() -> list:
    """For visualization: effective softmax temperature T = β₀/β = 1/ratio."""
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
    """Enable/disable LSD+shrooms system prompt (independent of β patch)."""
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
    """Enable/disable within-reply β annealing."""
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
    """Update β demo knobs (static ratio, layers, prompt, annealing schedule)
    without changing active/inactive state."""
    data = request.get_json(silent=True) or {}
    TRIP.configure(
        demo_beta_ratio=float(data["demo_beta_ratio"]) if "demo_beta_ratio" in data else None,
        demo_layers=data.get("demo_layers") if "demo_layers" in data else None,
        prompt_engineering=bool(data["prompt_engineering"]) if "prompt_engineering" in data else None,
        beta_patch=bool(data["beta_patch"]) if "beta_patch" in data else None,
        annealing=bool(data["annealing"]) if "annealing" in data else None,
        anneal_lo=float(data["anneal_lo"]) if "anneal_lo" in data else None,
        anneal_hi=float(data["anneal_hi"]) if "anneal_hi" in data else None,
        anneal_schedule=data.get("anneal_schedule") if "anneal_schedule" in data else None,
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
    """Legacy no-op; β does not anneal across turns (annealing is within-reply)."""
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

    # Snapshot β state before generation.
    trip_before = TRIP.snapshot()
    ratios_before = _per_layer_beta_ratio()
    temps_before = _per_layer_temperature()

    sampling_mult = TRIP.sampling_temperature_multiplier()
    effective_temperature = base_temperature * sampling_mult

    messages = _build_messages(history, message)

    try:
        reply = _generate_dispatch(
            messages, max_new_tokens, effective_temperature, top_p, seed
        )
    except Exception as e:
        return jsonify({"error": "generation failed", "detail": str(e)}), 500

    if not reply:
        return jsonify({"error": "Empty model response"}), 502

    TRIP.advance()  # no-op; kept for compatibility

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