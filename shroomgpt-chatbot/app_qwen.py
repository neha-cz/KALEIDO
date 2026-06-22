"""
Flask API for a Qwen2-VL chat with the THREE-LEVER "triple stack" intervention:

  1. β patch (attention inverse-temperature flattening) on demo layers.
  2. Persona-vector residual steer toward a "dissolved" voice (Chen et al. 2025).
  3. DREAM injection: a surreal visual prior sampled from a dream bank and added
     to the residual stream. In "flux" mode a FRESH dream vector is injected at
     every decode token, so the injected visual content churns during a reply --
     the analogue of a shifting visual field. (This is triple_flux, the
     configuration that measured best on the text-entropy / connectivity metrics.)

All three fire in the same forward pass at different sites and are independently
toggleable. β and persona were ported from the original Llama app; the dream
bank is built image-free from Qwen2-VL's own visual feature geometry.

This file replaces the model + intervention of the original Llama product but
keeps the post-processing (sanitizer + decay-tail trimmer) and the entire UI /
route surface, so the existing front-end works unchanged.

Setup
-----
  1. pip install flask flask-cors torch transformers accelerate
  2. Have these built (from the research pipeline):
       persona_vectors/qwen_dissolved.pt     ([n_layers, hidden] persona vector)
       dream_bank_direct.pt                   (dream bank: {"vectors": [N, hidden]})
  3. python app_qwen.py

Environment variables
----------------------
  HF_MODEL          (default: Qwen/Qwen2-VL-2B-Instruct)
  DEVICE            (default: auto)
  PORT              (default: 5001)
  TRIP_DEBUG        (default: 0)
  DEMO_BETA_RATIO   (default: 0.45) β ratio for demo layers
  DEMO_LAYERS       (default: 2,3)
  PERSONA_VECTOR    (default: persona_vectors/qwen_dissolved.pt)
  PERSONA_LAYER     (default: 9)
  PERSONA_COEF      (default: 8.0)
  PERSONA_ON        (default: 1)
  DREAM_BANK        (default: dream_bank_direct.pt)
  DREAM_LAYER       (default: 18)
  DREAM_COEF        (default: 20.0)
  DREAM_MODE        (default: flux)   "flux" | "fixed"
  DREAM_ON          (default: 1)
"""
import os
import threading

import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from transformers import AutoModelForImageTextToText, AutoTokenizer

# No system-prompt / prompt-engineering. The psychedelic register comes ENTIRELY
# from the activation-space interventions (β + persona + dream), not from any
# instruction telling the model to act altered. A tiny output cleaner is inlined
# below so there's no dependency on the old prompt-engineering module.

# The triple-stack engine (β + persona + dream), already validated.
import qwen_beta_persona as qbp


def sanitize_generated_text(text: str) -> str:
    """Minimal output cleaner: strip stray asterisks and surrounding whitespace.
    (No prompt-leak detection needed -- there is no system prompt to leak.)"""
    return (text or "").replace("*", "").strip()

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ============================================================
# Config
# ============================================================
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
TRIP_DEBUG = os.environ.get("TRIP_DEBUG", "0") == "1"
APPLY_BETA_ON_PREFILL = os.environ.get("APPLY_BETA_ON_PREFILL", "0") == "1"

# Calibrated triple-stack params (from the sweeps).
DEMO_BETA_RATIO = float(os.environ.get("DEMO_BETA_RATIO", "0.45"))
DEMO_LAYERS = tuple(
    int(x.strip())
    for x in os.environ.get("DEMO_LAYERS", "2,3").split(",")
    if x.strip()
)
BETA_RATIO_FLOOR = 0.05

# Persona steer.
PERSONA_VECTOR_PATH = os.environ.get("PERSONA_VECTOR", "persona_vectors/qwen_dissolved.pt")
PERSONA_LAYER = int(os.environ.get("PERSONA_LAYER", "9"))
PERSONA_COEF = float(os.environ.get("PERSONA_COEF", "8.0"))
PERSONA_ON = os.environ.get("PERSONA_ON", "1") == "1"
PERSONA_COEF_CAP = float(os.environ.get("PERSONA_COEF_CAP", "14.0"))

# Dream injection.
DREAM_BANK_PATH = os.environ.get("DREAM_BANK", "dream_bank_direct.pt")
DREAM_LAYER = int(os.environ.get("DREAM_LAYER", "18"))
DREAM_COEF = float(os.environ.get("DREAM_COEF", "20.0"))
DREAM_MODE = os.environ.get("DREAM_MODE", "flux")  # flux | fixed
DREAM_ON = os.environ.get("DREAM_ON", "1") == "1"
DREAM_COEF_CAP = float(os.environ.get("DREAM_COEF_CAP", "30.0"))

# Fixed trip profile (kept as a dict so existing routes/UI stay the same).
TRIP_PRESET = {
    "demo_beta_ratio": DEMO_BETA_RATIO,
    "demo_layers": DEMO_LAYERS,
    "beta_patch": True,
    "persona_steer": PERSONA_ON,
    "persona_coef": PERSONA_COEF,
    "dream_inject": DREAM_ON,
    "dream_coef": DREAM_COEF,
    "dream_mode": DREAM_MODE,
}

# Generation length: the dissolved/dream voice is densest in the first ~80-100
# tokens, then decays. Keep replies short to stay in the coherent band.
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "90"))
MAX_NEW_TOKENS_CAP = 1024

# Repetition controls. The dissolved/dream register loops ("the essence of the
# essence of..."). These break the loop attractor while keeping the voice.
NO_REPEAT_NGRAM = int(os.environ.get("NO_REPEAT_NGRAM", "4"))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.15"))


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = os.environ.get("DEVICE", _pick_device())
# bfloat16 on MPS+eager (fp16 softmax can NaN); fp32 on CPU.
DTYPE = torch.bfloat16 if DEVICE in ("cuda", "mps") else torch.float32

_GENERATION_LOCK = threading.Lock()


# ============================================================
# Trip State — fixed β patch + persona steer + dream injection.
# Same shape/fields as the original so the UI/JS is unchanged; extended with
# dream fields.
# ============================================================
class TripState:
    def __init__(self):
        self.active = False
        # β
        self.beta_patch = True
        self.demo_beta_ratio = max(BETA_RATIO_FLOOR, min(1.0, DEMO_BETA_RATIO))
        self.demo_layers = tuple(DEMO_LAYERS)
        self.n_layers = 0
        # persona
        self.persona_steer = bool(PERSONA_ON)
        self.persona_layer = int(PERSONA_LAYER)
        self.persona_coef = float(PERSONA_COEF)
        # dream
        self.dream_inject = bool(DREAM_ON)
        self.dream_layer = int(DREAM_LAYER)
        self.dream_coef = float(DREAM_COEF)
        self.dream_mode = str(DREAM_MODE)

    # ---- β ----
    def beta_ratio(self, layer_idx: int, n_layers: int) -> float:
        if not self.active or not self.beta_patch:
            return 1.0
        if layer_idx in self.demo_layers:
            return max(BETA_RATIO_FLOOR, min(1.0, self.demo_beta_ratio))
        return 1.0

    # ---- persona ----
    def persona_active(self) -> bool:
        return bool(self.active and self.persona_steer and self.persona_coef != 0.0)

    def current_persona_coef(self) -> float:
        return max(0.0, min(PERSONA_COEF_CAP, float(self.persona_coef)))

    # ---- dream ----
    def dream_active(self) -> bool:
        return bool(self.active and self.dream_inject and self.dream_coef != 0.0)

    def current_dream_coef(self) -> float:
        return max(0.0, min(DREAM_COEF_CAP, float(self.dream_coef)))

    # ---- legacy no-ops kept for route compatibility ----
    def sampling_temperature_multiplier(self) -> float:
        return 1.0

    def advance(self):
        return None

    def start(self, demo_beta_ratio=DEMO_BETA_RATIO, demo_layers=DEMO_LAYERS,
              beta_patch=True, persona_steer=None, persona_coef=None,
              dream_inject=None, dream_coef=None, dream_mode=None, **_ignored):
        self.active = True
        self.beta_patch = bool(beta_patch)
        self.demo_beta_ratio = max(BETA_RATIO_FLOOR, min(1.0, float(demo_beta_ratio)))
        self.demo_layers = tuple(int(x) for x in demo_layers)
        if persona_steer is not None:
            self.persona_steer = bool(persona_steer)
        if persona_coef is not None:
            self.persona_coef = float(persona_coef)
        if dream_inject is not None:
            self.dream_inject = bool(dream_inject)
        if dream_coef is not None:
            self.dream_coef = float(dream_coef)
        if dream_mode is not None:
            self.dream_mode = str(dream_mode)

    def configure(self, demo_beta_ratio=None, demo_layers=None, beta_patch=None,
                  annealing=None, persona_steer=None, persona_coef=None,
                  persona_layer=None, dream_inject=None, dream_coef=None,
                  dream_mode=None, dream_layer=None, **_ignored):
        if demo_beta_ratio is not None:
            self.demo_beta_ratio = max(BETA_RATIO_FLOOR, min(1.0, float(demo_beta_ratio)))
        if demo_layers is not None:
            if isinstance(demo_layers, str):
                demo_layers = [x.strip() for x in demo_layers.split(",") if x.strip()]
            self.demo_layers = tuple(int(x) for x in demo_layers)
        if beta_patch is not None:
            self.beta_patch = bool(beta_patch)
        if annealing is not None:  # legacy alias -> β patch toggle
            self.beta_patch = bool(annealing)
        if persona_steer is not None:
            self.persona_steer = bool(persona_steer)
        if persona_coef is not None:
            self.persona_coef = float(persona_coef)
        if persona_layer is not None:
            self.persona_layer = int(persona_layer)
        if dream_inject is not None:
            self.dream_inject = bool(dream_inject)
        if dream_coef is not None:
            self.dream_coef = float(dream_coef)
        if dream_mode is not None:
            self.dream_mode = str(dream_mode)
        if dream_layer is not None:
            self.dream_layer = int(dream_layer)

    def stop(self):
        self.active = False

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "beta_patch": self.beta_patch,
            "annealing": self.beta_patch,  # backward-compatible field
            "demo_beta_ratio": self.demo_beta_ratio,
            "demo_layers": list(self.demo_layers),
            "persona_steer": self.persona_steer,
            "persona_layer": self.persona_layer,
            "persona_coef": self.persona_coef,
            "persona_active_now": self.persona_active(),
            "dream_inject": self.dream_inject,
            "dream_layer": self.dream_layer,
            "dream_coef": self.dream_coef,
            "dream_mode": self.dream_mode,
            "dream_active_now": self.dream_active(),
            "sampling_T_mult_now": self.sampling_temperature_multiplier(),
            "note": "triple stack: fixed β patch on demo_layers + persona steer "
                    "+ dream injection (flux); no annealing/time dependence",
        }


TRIP = TripState()

# Wire the β patch to read from TRIP. qwen_beta_persona's BETA state is a
# context manager; here we instead want the patch to consult TRIP live (so the
# UI can tune β without re-entering a context). We bridge by giving BETA a
# dynamic ratio function via its public fields each generation (see _generate).


# ============================================================
# Model loading
# ============================================================
print(f"[load] Loading {HF_MODEL} on {DEVICE} ({DTYPE})...")
print("[load] First run downloads weights; subsequent runs load from cache.")

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForImageTextToText.from_pretrained(
    HF_MODEL,
    torch_dtype=DTYPE,
    device_map=DEVICE if DEVICE != "cpu" else None,
    attn_implementation="eager",
)
if DEVICE == "cpu":
    model = model.to(DEVICE)
model.eval()

# Confirm eager really took, or the β patch is silently dead.
_impl = getattr(model.config.get_text_config(), "_attn_implementation", None)
if _impl != "eager":
    print(f"[load] WARNING: text attn impl is {_impl!r}, not 'eager'; β patch "
          f"may not fire.")

N_LAYERS = model.config.get_text_config().num_hidden_layers
TRIP.n_layers = N_LAYERS

# Install all three interventions.
qbp.patch_qwen_attention()
qbp.install_persona_hooks(model)
qbp.install_dream_hooks(model)

# ---- persona vector ----
PERSONA_VECTOR = None
PERSONA_LAYER_VEC = None
if os.path.exists(PERSONA_VECTOR_PATH):
    try:
        _pv = torch.load(PERSONA_VECTOR_PATH, map_location="cpu")
        if torch.is_tensor(_pv) and _pv.dim() == 2 and _pv.shape[0] >= N_LAYERS:
            PERSONA_VECTOR = _pv
            PERSONA_LAYER_VEC = _pv[PERSONA_LAYER].to(DEVICE, DTYPE)
            print(f"[persona] ready: layer {PERSONA_LAYER}, "
                  f"||v||={torch.linalg.vector_norm(PERSONA_LAYER_VEC.float()).item():.3f}")
        else:
            print(f"[persona] WARNING: bad shape {getattr(_pv,'shape',None)}; steer DISABLED.")
            TRIP.persona_steer = False
    except Exception as e:
        print(f"[persona] WARNING: load failed ({e}); steer DISABLED.")
        TRIP.persona_steer = False
else:
    print(f"[persona] WARNING: vector not found at {PERSONA_VECTOR_PATH}; steer DISABLED.")
    TRIP.persona_steer = False

# ---- dream bank ----
DREAM_BANK = None
if os.path.exists(DREAM_BANK_PATH):
    try:
        _b = torch.load(DREAM_BANK_PATH, map_location="cpu")
        _vecs = _b["vectors"] if isinstance(_b, dict) else _b
        if torch.is_tensor(_vecs) and _vecs.dim() == 2:
            DREAM_BANK = _vecs.to(DEVICE, DTYPE)
            print(f"[dream] ready: bank {tuple(DREAM_BANK.shape)} "
                  f"({DREAM_BANK.shape[0]} dreams), layer {DREAM_LAYER}, mode {DREAM_MODE}")
        else:
            print(f"[dream] WARNING: bad bank shape; dream DISABLED.")
            TRIP.dream_inject = False
    except Exception as e:
        print(f"[dream] WARNING: load failed ({e}); dream DISABLED.")
        TRIP.dream_inject = False
else:
    print(f"[dream] WARNING: bank not found at {DREAM_BANK_PATH}; dream DISABLED.")
    TRIP.dream_inject = False

print(f"[load] Ready. {N_LAYERS} layers on {DEVICE}.")


# ============================================================
# Post-processing (PRESERVED from the original Llama product)
# ============================================================
def _clamp_max_new_tokens(value: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = DEFAULT_MAX_NEW_TOKENS
    return max(32, min(n, MAX_NEW_TOKENS_CAP))


def _trim_incomplete_reply(text: str) -> str:
    """Drop a dangling final fragment and the ellipsis/dash 'decay tail' that
    heavily-steered output degrades into. (Unchanged from the original app.)"""
    import re as _re
    text = (text or "").strip()
    if not text:
        return text
    m = _re.search(r"(\.{2,}|\u2026|[\-\u2013\u2014/\u2212]{2,}|(?:\s[\-\u2013\u2014\u2212/]\s){2,})", text)
    if m:
        text = text[:m.start()].strip()
    if not text:
        return text
    text = _re.sub(r"[\s\u2026/\\\u2013\u2014\u2212]+$", "", text).strip()
    text = _re.sub(r"\.{2,}$", "", text).strip()
    if not text:
        return text
    if text[-1] in ".!?" or (len(text) >= 2 and text[-1] in "\"')" and text[-2] in ".!?"):
        return text
    last = max((text.rfind(p) for p in ".!?"), default=-1)
    if last == -1:
        mtail = _re.search(r"[\s,;:\-\u2013\u2014\u2212]+\S{0,3}$", text)
        if mtail and mtail.start() > 20:
            text = text[:mtail.start()].strip()
        return text
    end = last + 1
    if end < len(text) and text[end] in "\"')":
        end += 1
    return text[:end].strip()


def _build_messages(history: list, new_message: str) -> list:
    # No system prompt. The interventions create the altered register; the model
    # receives only the ordinary conversation.
    out = []
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    if new_message.strip():
        out.append({"role": "user", "content": new_message.strip()})
    out = out[-4:]
    return out


# ============================================================
# Generation — stacks β + persona + dream via qwen_beta_persona contexts
# ============================================================
import contextlib


@torch.no_grad()
def _generate(messages, max_new_tokens, temperature, top_p, seed):
    with _GENERATION_LOCK:
        if seed is not None:
            torch.manual_seed(seed)
            if DEVICE == "cuda":
                torch.cuda.manual_seed_all(seed)

        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        max_new_tokens = _clamp_max_new_tokens(max_new_tokens)

        eos_ids = [tokenizer.eos_token_id]
        eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eot is not None and eot != tokenizer.unk_token_id:
            eos_ids.append(eot)

        # Build the three intervention contexts from live TRIP state, so the UI
        # sliders take effect on the next generation.
        beta_ctx = (qbp.BETA.engaged(TRIP.demo_beta_ratio, TRIP.demo_layers,
                                     apply_on_prefill=APPLY_BETA_ON_PREFILL)
                    if (TRIP.active and TRIP.beta_patch) else contextlib.nullcontext())
        persona_ctx = (qbp.PERSONA.engaged(TRIP.current_persona_coef(),
                                           TRIP.persona_layer,
                                           PERSONA_LAYER_VEC)
                       if (TRIP.persona_active() and PERSONA_LAYER_VEC is not None)
                       else contextlib.nullcontext())
        dream_ctx = (qbp.DREAM.engaged(TRIP.current_dream_coef(), TRIP.dream_layer,
                                       DREAM_BANK, mode=TRIP.dream_mode,
                                       seed=(seed if seed is not None else 0))
                     if (TRIP.dream_active() and DREAM_BANK is not None)
                     else contextlib.nullcontext())

        with beta_ctx, persona_ctx, dream_ctx:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, input_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reply = sanitize_generated_text(raw)
        trimmed = _trim_incomplete_reply(reply)
        if trimmed and len(trimmed) >= 40:
            reply = trimmed
        elif trimmed and len(trimmed) >= 0.5 * len(reply):
            reply = trimmed
        return reply


def _per_layer_beta_ratio() -> list:
    return [TRIP.beta_ratio(i, N_LAYERS) for i in range(N_LAYERS)]


def _per_layer_temperature() -> list:
    return [1.0 / TRIP.beta_ratio(i, N_LAYERS) for i in range(N_LAYERS)]


# ============================================================
# Routes (PRESERVED — same surface as the original app)
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
        "dream_bank_loaded": DREAM_BANK is not None,
        "trip": TRIP.snapshot(),
    })


@app.post("/api/trip/start")
def trip_start():
    TRIP.start(**TRIP_PRESET)
    return jsonify({
        "trip": TRIP.snapshot(),
        "per_layer_beta_ratio": _per_layer_beta_ratio(),
        "per_layer_temperature": _per_layer_temperature(),
    })


@app.post("/api/trip/annealing")
def trip_annealing():
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
    data = request.get_json(silent=True) or {}
    TRIP.configure(
        demo_beta_ratio=float(data["demo_beta_ratio"]) if "demo_beta_ratio" in data else None,
        demo_layers=data.get("demo_layers") if "demo_layers" in data else None,
        beta_patch=bool(data["beta_patch"]) if "beta_patch" in data else None,
        annealing=bool(data["annealing"]) if "annealing" in data else None,
        persona_steer=bool(data["persona_steer"]) if "persona_steer" in data else None,
        persona_coef=float(data["persona_coef"]) if "persona_coef" in data else None,
        dream_inject=bool(data["dream_inject"]) if "dream_inject" in data else None,
        dream_coef=float(data["dream_coef"]) if "dream_coef" in data else None,
        dream_mode=data.get("dream_mode") if "dream_mode" in data else None,
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

    base_temperature = float(data.get("temperature", 0.9))
    top_p = float(data.get("top_p", 0.95))
    max_new_tokens = _clamp_max_new_tokens(data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None

    trip_before = TRIP.snapshot()
    ratios_before = _per_layer_beta_ratio()
    temps_before = _per_layer_temperature()

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
