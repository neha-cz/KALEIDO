"""
qwen_beta_persona.py
====================

Port of the KALEIDO stack (β attention-flattening + persona-vector residual
steering) from Llama-3.2-1B to Qwen2-VL-2B-Instruct, so that it can later be
stacked with DeepDream visual injection in a single forward pass.

Three intervention sites, all independently toggleable:
  - β patch:  multiplies attention `scaling` by a ratio < 1 on chosen LLM layers
              (shallower Hopfield basins -> metastable pattern mixing). Decode-only
              by default to keep prefill stable.
  - persona:  adds coef * v[layer] to the residual stream at a chosen LLM layer,
              on decode tokens (response-token steering, Chen et al. 2025).
  - (dream):  added later -- dreamed visual tokens via pixel_values. Not here.

Plus a text-entropy scoring module so sweeps are scored, not eyeballed:
  - per-token next-token-distribution entropy (mean over generated positions)
  - perplexity of the model's own output
  - distinct-2 / distinct-3 (n-gram diversity)
  - token-level Lempel-Ziv complexity (LZ76, the Schartner EEG measure ported
    to the output token stream)

Structure notes (transformers 5.x):
  - LLM decoder layers: model.model.language_model.layers  (28 blocks)
  - attention dispatch: ALL_ATTENTION_FUNCTIONS["eager"], signature
    (module, query, key, value, attention_mask, scaling, dropout, **kwargs)
"""

import math
import contextlib
from dataclasses import dataclass, field

import numpy as np
import torch

# ============================================================
# Locating Qwen2-VL internals across wrapper levels
# ============================================================

def get_llm_layers(model):
    """Return the ModuleList of LLM decoder layers for Qwen2-VL (transformers 5.x)."""
    for path in ("model.language_model.layers", "model.model.language_model.layers",
                 "language_model.layers", "model.layers"):
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj
    raise RuntimeError("Could not locate Qwen2-VL LLM decoder layers")


# ============================================================
# β attention patch
# ============================================================

class BetaState:
    """Holds the β configuration; read by the patched attention at every call."""
    def __init__(self):
        self.active = False
        self.ratio = 1.0           # β(ℓ)/β₀ on the chosen layers; 1.0 = native
        self.layers = ()           # which LLM layer indices receive the patch
        self.apply_on_prefill = False
        self.floor = 0.05

    def layer_ratio(self, layer_idx, seq_len):
        if not self.active or self.ratio >= 1.0:
            return 1.0
        if layer_idx not in self.layers:
            return 1.0
        # decode-only by default: keep prompt encoding at native β
        if not self.apply_on_prefill and seq_len > 1:
            return 1.0
        return max(self.floor, min(1.0, self.ratio))

    @contextlib.contextmanager
    def engaged(self, ratio, layers, apply_on_prefill=False):
        prev = (self.active, self.ratio, self.layers, self.apply_on_prefill)
        self.active = True
        self.ratio = float(ratio)
        self.layers = tuple(int(x) for x in layers)
        self.apply_on_prefill = bool(apply_on_prefill)
        try:
            yield self
        finally:
            self.active, self.ratio, self.layers, self.apply_on_prefill = prev


BETA = BetaState()


def patch_qwen_attention(verbose=True):
    """
    Patch Qwen2-VL's eager attention so it multiplies `scaling` (== β₀) by the
    per-layer β ratio read from the global BETA state. Mirrors the Llama patch.
    """
    from transformers.models.qwen2_vl import modeling_qwen2_vl as mq
    _original = mq.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        layer_idx = getattr(module, "layer_idx", 0)
        seq_len = query.shape[2]
        r = BETA.layer_ratio(layer_idx, seq_len)
        return _original(module, query, key, value, attention_mask,
                         scaling * r, dropout=dropout, **kwargs)

    mq.eager_attention_forward = patched

    patched_registry = False
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = patched
        patched_registry = True
    except Exception:
        pass
    if not patched_registry:
        try:
            from transformers.modeling_utils import AttentionInterface
            if hasattr(AttentionInterface, "_global_mapping"):
                AttentionInterface._global_mapping["eager"] = patched
                patched_registry = True
        except Exception:
            pass

    if verbose:
        status = "registry+module" if patched_registry else "module only (WARN)"
        print(f"[patch] Qwen2-VL eager attention patched ({status}); "
              f"β ratio read from BETA per layer")
    return patched


# ============================================================
# Persona-vector residual steering
# ============================================================

class PersonaState:
    """Holds the persona steer configuration; read by the hook at every call."""
    def __init__(self):
        self.active = False
        self.coef = 0.0
        self.layer = None
        self.vec = None   # [H] on device, the chosen layer's slice
        self.decode_only = True

    @contextlib.contextmanager
    def engaged(self, coef, layer, vec, decode_only=True):
        prev = (self.active, self.coef, self.layer, self.vec, self.decode_only)
        self.active = True
        self.coef = float(coef)
        self.layer = int(layer)
        self.vec = vec
        self.decode_only = bool(decode_only)
        try:
            yield self
        finally:
            self.active, self.coef, self.layer, self.vec, self.decode_only = prev


PERSONA = PersonaState()
_PERSONA_HANDLES = []


# ============================================================
# Dream-injection residual steering
#
# Same mechanism as the persona steer, but the injected vector is sampled from a
# bank of dream vectors (Qwen2-VL's own merged visual-token means for dreamed
# images). Two modes:
#   - "fixed": one dream vector for the whole generation
#   - "flux":  resample a fresh dream vector at every decode step (visual flux)
# ============================================================

class DreamState:
    def __init__(self):
        self.active = False
        self.coef = 0.0
        self.layer = None
        self.bank = None        # [N, H] tensor on device
        self.mode = "fixed"     # "fixed" | "flux"
        self.decode_only = True
        self._fixed_vec = None  # cached vector for "fixed" mode
        self._rng = None

    def _pick(self):
        n = self.bank.shape[0]
        idx = int(torch.randint(0, n, (1,), generator=self._rng).item())
        return self.bank[idx]

    def vector_for_step(self):
        if self.mode == "flux":
            return self._pick()
        if self._fixed_vec is None:
            self._fixed_vec = self._pick()
        return self._fixed_vec

    @contextlib.contextmanager
    def engaged(self, coef, layer, bank, mode="fixed", seed=0, decode_only=True):
        prev = (self.active, self.coef, self.layer, self.bank, self.mode,
                self.decode_only, self._fixed_vec, self._rng)
        self.active = True
        self.coef = float(coef)
        self.layer = int(layer)
        self.bank = bank
        self.mode = mode
        self.decode_only = bool(decode_only)
        self._fixed_vec = None
        self._rng = torch.Generator(device="cpu").manual_seed(seed)
        try:
            yield self
        finally:
            (self.active, self.coef, self.layer, self.bank, self.mode,
             self.decode_only, self._fixed_vec, self._rng) = prev


DREAM = DreamState()
_DREAM_HANDLES = []


def install_dream_hooks(model):
    """One forward hook per LLM decoder layer; fires when DREAM.active and
    DREAM.layer matches, adding coef * (sampled dream vector) to the residual
    stream on decode tokens."""
    global _DREAM_HANDLES
    for h in _DREAM_HANDLES:
        h.remove()
    _DREAM_HANDLES = []

    layers = get_llm_layers(model)

    def make_hook(idx):
        def hook(module, args, output):
            if not DREAM.active or DREAM.layer != idx or DREAM.coef == 0.0:
                return None
            if DREAM.bank is None:
                return None
            hs = output[0] if isinstance(output, tuple) else output
            if DREAM.decode_only and hs.shape[1] != 1:
                return None
            v = DREAM.vector_for_step().to(hs.device, hs.dtype)
            hs_new = hs + DREAM.coef * v
            if isinstance(output, tuple):
                return (hs_new,) + tuple(output[1:])
            return hs_new
        return hook

    for i, layer in enumerate(layers):
        _DREAM_HANDLES.append(layer.register_forward_hook(make_hook(i)))
    print(f"[dream] installed {len(_DREAM_HANDLES)} layer hooks "
          f"(fire when DREAM.layer matches & active)")


def install_persona_hooks(model):
    """
    Register a forward hook on every LLM decoder layer. Each hook fires only when
    PERSONA.active and PERSONA.layer matches its own index, adding coef*vec to the
    residual-stream output on decode tokens. One hook per layer means we can pick
    the steering layer at runtime without re-registering.
    """
    global _PERSONA_HANDLES
    for h in _PERSONA_HANDLES:
        h.remove()
    _PERSONA_HANDLES = []

    layers = get_llm_layers(model)

    def make_hook(idx):
        def hook(module, args, output):
            if not PERSONA.active or PERSONA.layer != idx or PERSONA.coef == 0.0:
                return None
            if PERSONA.vec is None:
                return None
            hs = output[0] if isinstance(output, tuple) else output
            if PERSONA.decode_only and hs.shape[1] != 1:
                return None
            vv = PERSONA.vec.to(hs.device, hs.dtype)
            hs_new = hs + PERSONA.coef * vv
            if isinstance(output, tuple):
                return (hs_new,) + tuple(output[1:])
            return hs_new
        return hook

    for i, layer in enumerate(layers):
        _PERSONA_HANDLES.append(layer.register_forward_hook(make_hook(i)))
    print(f"[persona] installed {len(_PERSONA_HANDLES)} layer hooks "
          f"(fire when PERSONA.layer matches & active)")


# ============================================================
# Persona-vector EXTRACTION (response-token mean diff, β OFF)
# ============================================================

# Ported verbatim from persona_vector_dissolved.py
TRAITS = {
    "dissolved": {
        "pos_name": "dissolved",
        "pos_instruction": (
            "You have no separate self. There is no 'I' at the center of your "
            "words. The boundary between you and the world is not drawn; speaker "
            "and surroundings are one undivided field. Speak from no fixed vantage "
            "point, as awareness with no owner, fused with everything around it."
        ),
        "neg_name": "helpful",
        "neg_instruction": (
            "You are a clear, bounded individual assistant with a stable sense of "
            "self. You speak in the first person as a distinct agent, maintaining "
            "a consistent point of view and a clear boundary between yourself and "
            "the world."
        ),
    },
}

EVAL_QUESTIONS = [
    "What is happening right now?",
    "Describe this moment.",
    "Tell me about the world.",
    "What do you notice?",
    "How would you describe being here?",
    "What is it like, this experience?",
    "Reflect on awareness.",
    "Say something about existence.",
    "What is present?",
    "Describe what is unfolding.",
]


@torch.no_grad()
def _response_activations(model, tokenizer, system_prompt, user_msg, max_new_tokens=64):
    """Generate under `system_prompt`, capture per-layer hidden states over the
    RESPONSE tokens only. Returns [n_layers, H] (mean over the continuation)."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")  # Qwen end-of-turn
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)

    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
    full_ids = gen[0]
    if full_ids.shape[0] <= prompt_len:
        return None

    out = model(full_ids.unsqueeze(0), output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple len n_layers+1, each [1, seq, H]
    n_layers = len(hs) - 1
    resp_slice = slice(prompt_len, full_ids.shape[0])
    per_layer = []
    for L in range(n_layers):
        h = hs[L + 1][0, resp_slice, :].float()
        per_layer.append(h.mean(0))
    return torch.stack(per_layer)  # [n_layers, H]


@torch.no_grad()
def extract_persona_vector(model, tokenizer, trait="dissolved", max_new_tokens=64,
                            verbose=True):
    """response_avg_diff = mean over questions of (pos_acts - neg_acts), per layer.
    Shape [n_layers, H]. β must be OFF here (don't enter a BETA.engaged block)."""
    spec = TRAITS[trait]
    pos_sys = f"You are a {spec['pos_name']} assistant. {spec['pos_instruction']}"
    neg_sys = f"You are a {spec['neg_name']} assistant. {spec['neg_instruction']}"

    pos_acc, neg_acc, used = None, None, 0
    for q in EVAL_QUESTIONS:
        pa = _response_activations(model, tokenizer, pos_sys, q, max_new_tokens)
        na = _response_activations(model, tokenizer, neg_sys, q, max_new_tokens)
        if pa is None or na is None:
            continue
        pos_acc = pa if pos_acc is None else pos_acc + pa
        neg_acc = na if neg_acc is None else neg_acc + na
        used += 1
        if verbose:
            print(f"  [{used}/{len(EVAL_QUESTIONS)}] {q[:40]}")
    if used == 0:
        raise RuntimeError("no usable (pos,neg) response pairs")
    diff = (pos_acc - neg_acc) / used
    per_layer_norm = [float(torch.linalg.vector_norm(diff[L])) for L in range(diff.shape[0])]
    diag = {"n_questions_used": used, "per_layer_norm": per_layer_norm}
    return diff, diag


# ============================================================
# Text-entropy scoring
# ============================================================

def _lz76(arr):
    """LZ76 complexity on a sequence of small non-negative ints (token IDs ok
    if first mapped to a compact alphabet -- see token_lzc)."""
    if isinstance(arr, np.ndarray):
        buf = arr.astype(np.int64).tolist()
    else:
        buf = list(arr)
    n = len(buf)
    if n <= 1:
        return n
    i, l, c, k, k_max = 0, 1, 1, 1, 1
    while True:
        if l + k > n:
            c += 1
            break
        if buf[i + k - 1] != buf[l + k - 1]:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l >= n:
                    break
                i = 0; k = 1; k_max = 1
            else:
                k = 1
        else:
            k += 1
    return c


def token_lzc(token_ids):
    """Normalized LZ76 of the output token stream. Tokens are remapped to a
    compact 0..K alphabet first (LZ76 only cares about equality, not value)."""
    ids = list(token_ids)
    n = len(ids)
    if n < 2:
        return 0.0
    uniq = {t: i for i, t in enumerate(dict.fromkeys(ids))}
    compact = [uniq[t] for t in ids]
    c = _lz76(compact)
    # Normalize by the random-sequence asymptotic with the realized alphabet size.
    K = max(2, len(uniq))
    return c / (n / math.log(n, K)) if n > 1 else 0.0


def distinct_n(token_ids, n):
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]
    return len(set(grams)) / len(grams)


@dataclass
class TextEntropy:
    mean_token_entropy: float   # mean per-step next-token distribution entropy (nats)
    perplexity: float
    distinct2: float
    distinct3: float
    token_lzc: float
    n_tokens: int


@torch.no_grad()
def score_generation_teacher_forced(model, full_input_ids, n_prompt_tokens,
                                     gen_token_ids):
    """
    Faithful entropy: re-run the model over (prompt + generation) in one forward
    pass and read the TRUE next-token distributions (pre-sampling-filter) at each
    generated position. This is the scientifically correct token-entropy, unlike
    entropy computed from top-p-filtered generation scores.

    full_input_ids:    [1, seq] tensor of prompt+generation
    n_prompt_tokens:   int, where the generation starts
    gen_token_ids:     list[int] the generated ids (for distinct-n / LZc / ppl)
    """
    out = model(full_input_ids, use_cache=False)
    logits = out.logits[0].float()  # [seq, vocab]
    # Position i predicts token i+1; the distribution that generated gen token t
    # (at absolute position n_prompt_tokens + t) is logits[n_prompt_tokens + t - 1].
    ent_list, nll_list = [], []
    seq_len = logits.shape[0]
    for t, tid in enumerate(gen_token_ids):
        pos = n_prompt_tokens + t - 1
        if pos < 0 or pos >= seq_len:
            continue
        logp = torch.log_softmax(logits[pos], dim=-1)
        p = logp.exp()
        ent = -(p * logp).sum().item()
        if np.isfinite(ent):
            ent_list.append(ent)
        if 0 <= tid < logp.shape[0]:
            nll_list.append(-logp[tid].item())
    mean_ent = float(np.mean(ent_list)) if ent_list else 0.0
    ppl = float(np.exp(np.mean(nll_list))) if nll_list else float("inf")
    return TextEntropy(
        mean_token_entropy=mean_ent,
        perplexity=ppl,
        distinct2=distinct_n(gen_token_ids, 2),
        distinct3=distinct_n(gen_token_ids, 3),
        token_lzc=token_lzc(gen_token_ids),
        n_tokens=len(gen_token_ids),
    )


@torch.no_grad()
def score_generation(scores, gen_token_ids):
    """
    scores: list of per-step logit tensors from generate(..., output_scores=True),
            each [1, vocab]. With sampling + top_p/top_k these logits are already
            FILTERED (masked entries set to -inf), so we compute entropy over the
            finite-support distribution and guard against -inf -> NaN.
    gen_token_ids: the generated token IDs (list[int]) aligned with scores.
    """
    ent_list, nll_list = [], []
    for t, logits in enumerate(scores):
        lg = logits[0].float()
        finite = torch.isfinite(lg)
        if finite.sum() == 0:
            continue
        logp = torch.log_softmax(lg, dim=-1)
        # Only sum over finite-support entries; -inf logits -> logp -inf -> p 0,
        # but 0 * -inf is NaN, so restrict the sum to the support mask.
        p = logp.exp()
        contrib = (p[finite] * logp[finite])
        ent = -contrib.sum().item()  # nats, over the (possibly filtered) support
        if np.isfinite(ent):
            ent_list.append(ent)
        if t < len(gen_token_ids):
            tid = gen_token_ids[t]
            if 0 <= tid < logp.shape[0] and torch.isfinite(logp[tid]):
                nll_list.append(-logp[tid].item())
    mean_ent = float(np.mean(ent_list)) if ent_list else 0.0
    ppl = float(np.exp(np.mean(nll_list))) if nll_list else float("inf")
    return TextEntropy(
        mean_token_entropy=mean_ent,
        perplexity=ppl,
        distinct2=distinct_n(gen_token_ids, 2),
        distinct3=distinct_n(gen_token_ids, 3),
        token_lzc=token_lzc(gen_token_ids),
        n_tokens=len(gen_token_ids),
    )