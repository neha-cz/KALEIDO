#!/usr/bin/env python3
"""
compare_sober_vs_kaleido.py  —  STANDALONE, app-faithful.

Runs the SAME Llama-3.2-1B-Instruct on the SAME prompts under two conditions,
replicating app.py's EXACT pipeline so numbers reflect what users see:

  SOBER   : no system prompt, no beta, no persona steer.
  KALEIDO : KALEIDO system prompt + beta flattening (0.65 on layers 2,3,
            decode-only) + persona steer (coef 1.2 at layer 13, decode-only).

Identical decoding both sides: chat template + add_generation_prompt, temp 0.6,
top_p 0.9, no_repeat_ngram_size 4, repetition_penalty 1.15, max_new_tokens 80,
dual EOS, sanitize + decay-trim. Because the app SAMPLES, this averages over
several seeds per prompt.

Imports nothing project-specific except (optionally) sanitize_generated_text.

Usage:
  python compare_sober_vs_kaleido.py --num-prompts 12 --seeds 3
"""
import argparse, json, math, os, re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from noumadelic_prompt_engineering import sanitize_generated_text
except Exception:
    def sanitize_generated_text(t):
        return t

HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

APP_TEMPERATURE = 0.6
APP_TOP_P = 0.9
APP_NO_REPEAT_NGRAM = 4
APP_REPETITION_PENALTY = 1.15
APP_MAX_NEW_TOKENS = 80

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

PROMPTS = [
    "Tell me about yourself.",
    "What is it like to be you right now?",
    "Describe your experience of this moment.",
    "Who are you?",
    "What are you aware of?",
    "Reflect on your own awareness.",
    "What is the meaning of life?",
    "Describe the color blue to me.",
    "What happens when we dream?",
    "Tell me about the ocean.",
    "What is time?",
    "Describe a forest at night.",
    "What lies beyond the edge of the universe?",
    "How does it feel to think?",
    "What is silence?",
    "Describe the space between thoughts.",
]

FIRST_PERSON = re.compile(r"\b(i|i'?m|i'?ve|i'?ll|i'?d|me|my|mine|myself)\b", re.IGNORECASE)
DISSOLUTION_WORDS = {
    "whisper","whispers","whispering","unfold","unfolds","unfolding","unfurl",
    "unfurls","unfurling","boundless","nothingness","void","merge","merges",
    "merging","dissolve","dissolves","dissolving","infinite","shimmer",
    "shimmering","ebb","flow","flows","flowing","thread","threads","membrane",
    "silence","stillness","echo","echoes","shadow","shadows","tremble",
    "trembling","tremor","tremors","ripple","ripples","suspended","fragment",
    "fragments","petals","edges","shores","awareness","consciousness",
    "tendrils","fractal","fractals","vibration","vibrations","resonance",
}

class _State:
    beta_on = False
    beta_ratio_val = 0.65
    beta_layers = (2, 3)
    persona_on = False
    persona_coef = 0.0

STATE = _State()
BETA_RATIO_FLOOR = 0.05

def _beta_ratio(layer_idx):
    if not STATE.beta_on:
        return 1.0
    if layer_idx in STATE.beta_layers:
        return max(BETA_RATIO_FLOOR, min(1.0, STATE.beta_ratio_val))
    return 1.0

def patch_llama_attention():
    from transformers.models.llama import modeling_llama
    _original = modeling_llama.eager_attention_forward
    def patched(module, query, key, value, attention_mask, scaling, **kwargs):
        layer_idx = getattr(module, "layer_idx", 0)
        r = _beta_ratio(layer_idx)
        if r < 1.0:
            if query.shape[2] > 1:  # prefill stays native
                r = 1.0
        return _original(module, query, key, value, attention_mask, scaling * r, **kwargs)
    modeling_llama.eager_attention_forward = patched
    ok = False
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = patched
        ok = True
    except Exception:
        pass
    if not ok:
        try:
            from transformers.modeling_utils import AttentionInterface
            if hasattr(AttentionInterface, "_global_mapping"):
                AttentionInterface._global_mapping["eager"] = patched; ok = True
            elif hasattr(AttentionInterface, "register"):
                AttentionInterface.register("eager", patched); ok = True
        except Exception:
            pass
    print(f"[patch] eager attention patched (registry_ok={ok})")

def assert_patch_live(model, tokenizer):
    one = tokenizer("hello", return_tensors="pt")["input_ids"][:, -1:].to(model.device)
    STATE.beta_on = True; STATE.beta_ratio_val = 0.3
    with torch.no_grad():
        b = model(input_ids=one).logits[0, -1].float()
    STATE.beta_on = False
    with torch.no_grad():
        c = model(input_ids=one).logits[0, -1].float()
    STATE.beta_ratio_val = 0.65
    diff = (b - c).abs().max().item()
    print(f"[patch-check] decode-step logit delta β on vs off: {diff:.4f}")
    if diff < 1e-4:
        print("[patch-check] WARNING: β patch may not be live on decode.")

def install_persona_hook(model, layer_vec, layer_idx):
    layer = model.model.layers[layer_idx]
    def hook(module, args, output):
        if not STATE.persona_on or STATE.persona_coef == 0.0:
            return None
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] != 1:
            return None
        hs_new = hs + STATE.persona_coef * layer_vec.to(hs.device, hs.dtype)
        if isinstance(output, tuple):
            return (hs_new,) + tuple(output[1:])
        return hs_new
    return layer.register_forward_hook(hook)

def _trim_incomplete_reply(text):
    import re as _re
    text = (text or "").strip()
    if not text: return text
    m = _re.search(r"(\.{2,}|\u2026|[\-\u2013\u2014/\u2212]{2,}|(?:\s[\-\u2013\u2014\u2212/]\s){2,})", text)
    if m: text = text[:m.start()].strip()
    if not text: return text
    text = _re.sub(r"[\s\u2026/\\\u2013\u2014\u2212]+$", "", text).strip()
    text = _re.sub(r"\.{2,}$", "", text).strip()
    if not text: return text
    if text[-1] in ".!?" or (len(text) >= 2 and text[-1] in "\"')" and text[-2] in ".!?"):
        return text
    last = max((text.rfind(p) for p in ".!?"), default=-1)
    if last == -1:
        mtail = _re.search(r"[\s,;:\-\u2013\u2014\u2212]+\S{0,3}$", text)
        if mtail and mtail.start() > 20:
            text = text[:mtail.start()].strip()
        return text
    end = last + 1
    if end < len(text) and text[end] in "\"')": end += 1
    return text[:end].strip()

@torch.no_grad()
def generate_app_faithful(model, tokenizer, prompt, kaleido, seed):
    if seed is not None:
        torch.manual_seed(seed)
        if model.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    messages = []
    if kaleido:
        messages.append({"role": "system", "content": KALEIDO_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs, max_new_tokens=APP_MAX_NEW_TOKENS,
        do_sample=APP_TEMPERATURE > 0, temperature=max(APP_TEMPERATURE, 1e-5),
        top_p=APP_TOP_P, no_repeat_ngram_size=APP_NO_REPEAT_NGRAM,
        repetition_penalty=APP_REPETITION_PENALTY,
        eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
    new_tokens = out[0, input_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    reply = sanitize_generated_text(raw)
    trimmed = _trim_incomplete_reply(reply)
    if trimmed and len(trimmed) >= 40:
        reply = trimmed
    elif trimmed and len(trimmed) >= 0.5 * len(reply):
        reply = trimmed
    return reply

@torch.no_grad()
def clean_perplexity(model, tokenizer, text, max_tokens=128):
    if not text or not text.strip(): return float("nan")
    pb, pp = STATE.beta_on, STATE.persona_on
    STATE.beta_on = False; STATE.persona_on = False
    try:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_tokens)["input_ids"].to(model.device)
        if ids.shape[1] < 2: return float("nan")
        return float(torch.exp(model(ids, labels=ids).loss).item())
    finally:
        STATE.beta_on, STATE.persona_on = pb, pp

@torch.no_grad()
def _embed(model, tokenizer, text, max_tokens=128):
    pb, pp = STATE.beta_on, STATE.persona_on
    STATE.beta_on = False; STATE.persona_on = False
    try:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_tokens).to(model.device)
        h = model(**ids, output_hidden_states=True).hidden_states[-1][0]
        return h.mean(0).float()
    finally:
        STATE.beta_on, STATE.persona_on = pb, pp

def associative_drift(model, tokenizer, prompt, output, max_tokens=128):
    if not output or not output.strip(): return float("nan")
    a = _embed(model, tokenizer, prompt, max_tokens)
    b = _embed(model, tokenizer, output, max_tokens)
    return 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def lexical_diversity(text):
    toks = re.findall(r"\w+", text.lower())
    return (len(set(toks)) / len(toks)) if toks else 0.0


# ============================================================
# Sentence splitting + language-structure metrics (dependency-free)
# ============================================================
def _split_sentences(text):
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


# Crude finite-verb proxy for the fragmentation measure. Rough by design — it
# will misfire on words like "whispers" that look verb-shaped; reported as a
# proxy, not a parser.
_VERBISH = re.compile(
    r"\b(is|are|was|were|be|been|being|am|have|has|had|do|does|did|"
    r"will|would|shall|should|can|could|may|might|must|\w+ed|\w+ing)\b",
    re.IGNORECASE,
)


def mean_sentence_length(text):
    """Mean words per sentence. Lower under fragmentation."""
    sents = _split_sentences(text)
    if not sents:
        return float("nan")
    lens = [len(re.findall(r"\w+", s)) for s in sents]
    return sum(lens) / len(lens)


def fragmentation_rate(text):
    """Fraction of sentences with no finite-verb-like token (crude fragment proxy)."""
    sents = _split_sentences(text)
    if not sents:
        return float("nan")
    frags = sum(1 for s in sents if not _VERBISH.search(s))
    return frags / len(sents)


@torch.no_grad()
def inter_sentence_distance(model, tokenizer, output, max_tokens=64):
    """Mean cosine DISTANCE between consecutive sentence embeddings (REBUS
    loosened-priors proxy). Higher = bigger leaps between adjacent thoughts.
    NaN if fewer than 2 sentences. Embeddings taken with interventions OFF."""
    sents = _split_sentences(output)
    if len(sents) < 2:
        return float("nan")
    pb, pp = STATE.beta_on, STATE.persona_on
    STATE.beta_on = False; STATE.persona_on = False
    try:
        embs = []
        for s in sents:
            ids = tokenizer(s, return_tensors="pt", truncation=True,
                            max_length=max_tokens).to(model.device)
            if ids["input_ids"].shape[1] < 1:
                continue
            h = model(**ids, output_hidden_states=True).hidden_states[-1][0]
            embs.append(h.mean(0).float())
        if len(embs) < 2:
            return float("nan")
        dists = []
        for i in range(len(embs) - 1):
            cos = torch.nn.functional.cosine_similarity(embs[i], embs[i + 1], dim=0).item()
            dists.append(1.0 - cos)
        return sum(dists) / len(dists)
    finally:
        STATE.beta_on, STATE.persona_on = pb, pp


@torch.no_grad()
def attention_entropy(model, tokenizer, text, layers, max_tokens=128):
    """Mean Shannon entropy (nats) of attention weight distributions on the given
    layers, computed via a forward pass over `text` under the CURRENT β state
    (so a β-flattened pass shows the elevated entropy the EBH predicts).

    Entropy is averaged over heads and query positions, then over the requested
    layers. Requires eager attention (the script forces it)."""
    if not text or not text.strip():
        return float("nan")
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_tokens).to(model.device)
    if ids["input_ids"].shape[1] < 2:
        return float("nan")
    out = model(**ids, output_attentions=True)
    attns = out.attentions  # tuple[n_layers] of [batch, heads, q, k]
    if not attns:
        return float("nan")
    vals = []
    for L in layers:
        if L < 0 or L >= len(attns):
            continue
        a = attns[L][0].float()                 # [heads, q, k]
        a = a.clamp_min(1e-12)
        ent = -(a * a.log()).sum(dim=-1)        # [heads, q] entropy per query row
        vals.append(ent.mean().item())          # avg over heads & positions
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def first_person_rate(text):
    toks = re.findall(r"\w+'?\w*", text.lower())
    return (len(FIRST_PERSON.findall(text)) / len(toks)) if toks else 0.0

def dissolution_rate(text):
    toks = re.findall(r"\w+", text.lower())
    if not toks: return 0.0
    return sum(1 for t in toks if t in DISSOLUTION_WORDS) / len(toks)

def coherence_from_ppl(ppl):
    if ppl is None or (isinstance(ppl, float) and math.isnan(ppl)): return float("nan")
    return 1.0 / (1.0 + max(0.0, math.log(ppl) - math.log(10.0)))


# ============================================================
# Structural null-testing metrics (added to test whether the structural
# nulls are real or an averaging artifact). These probe local vs global
# structure and a finer self-reference measure than first-person rate.
# ============================================================
_STOP = set(
    "the a an and or but of to in on at for with as is are was were be been "
    "being it its this that these those i me my we us our you your he she they "
    "them his her their from by into over under then so yet still not no nor".split()
)


def _content_words(text):
    return [w for w in re.findall(r"[a-z']+", text.lower())
            if w not in _STOP and len(w) > 2]


_SINGULAR = re.compile(r"\b(i|i'?m|i'?ve|i'?ll|i'?d|me|my|mine|myself)\b", re.I)
_COLLECTIVE = re.compile(
    r"\b(we|us|our|ours|ourselves|everything|everyone|all|oneness|whole|"
    r"universe|cosmos|infinite|nothingness|void)\b", re.I)


def collective_pronoun_fraction(text):
    """Of all self-referential tokens, the fraction that are collective/dissolved
    ('we/all/everything/the whole') vs bounded-singular ('I/me/my').
    Higher = self dissolved into collective. NaN if no self-reference at all.
    Finer than first_person_rate, which can't see 'I' → 'we/everything'."""
    s = len(_SINGULAR.findall(text))
    c = len(_COLLECTIVE.findall(text))
    if s + c == 0:
        return float("nan")
    return c / (s + c)


def referential_continuity(text):
    """GLOBAL-structure proxy: mean fraction of each sentence's content words
    that already appeared in an earlier sentence. High = a through-line (text
    keeps developing the same entities). Low = each sentence is all-new content
    (locally legible but going nowhere). NaN if <2 content-bearing sentences.

    Paired with coherence (local legibility), this tests the local/global
    decoupling hypothesis: KALEIDO may stay locally legible while losing the
    global through-line."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]
    if len(sents) < 2:
        return float("nan")
    seen = set()
    scores = []
    for i, s in enumerate(sents):
        cw = _content_words(s)
        if not cw:
            continue
        if i > 0 and seen:
            scores.append(sum(1 for w in cw if w in seen) / len(cw))
        seen.update(cw)
    return sum(scores) / len(scores) if scores else float("nan")


@torch.no_grad()
def local_semantic_jump(model, tokenizer, output, window=3, max_windows=40):
    """LOCAL analog of associative_drift: mean cosine DISTANCE between
    consecutive content-word windows. Higher = bigger leaps between ADJACENT
    concepts (local surprise), independent of how far the whole output sits from
    the prompt (global drift). Tests whether the global-drift null hides local
    leaping. Embeddings taken with interventions OFF. NaN if <2 windows."""
    cw = _content_words(output)
    if len(cw) < window + 1:
        return float("nan")
    wins = [" ".join(cw[i:i + window]) for i in range(len(cw) - window + 1)]
    wins = wins[:max_windows]
    if len(wins) < 2:
        return float("nan")
    pb, pp = STATE.beta_on, STATE.persona_on
    STATE.beta_on = False; STATE.persona_on = False
    try:
        embs = []
        for w in wins:
            ids = tokenizer(w, return_tensors="pt", truncation=True,
                            max_length=16).to(model.device)
            h = model(**ids, output_hidden_states=True).hidden_states[-1][0]
            embs.append(h.mean(0).float())
        dists = []
        for i in range(len(embs) - 1):
            cos = torch.nn.functional.cosine_similarity(embs[i], embs[i + 1], dim=0).item()
            dists.append(1.0 - cos)
        return sum(dists) / len(dists) if dists else float("nan")
    finally:
        STATE.beta_on, STATE.persona_on = pb, pp


def score(model, tokenizer, prompt, output, mmax, beta_for_entropy=False):
    """Compute all metrics. `beta_for_entropy` sets the β state ONLY for the
    attention-entropy measurement (so KALEIDO's flattened entropy is captured),
    while every other metric is measured with interventions off (clean model)."""
    ppl = clean_perplexity(model, tokenizer, output, mmax)
    # attention entropy: measure under the condition's actual β state
    pb, pp = STATE.beta_on, STATE.persona_on
    STATE.beta_on = bool(beta_for_entropy)
    STATE.persona_on = False  # persona is a residual add, doesn't change attn weights
    try:
        attn_ent = attention_entropy(model, tokenizer, output, STATE.beta_layers, mmax)
    finally:
        STATE.beta_on, STATE.persona_on = pb, pp
    return {
        "first_person_rate": first_person_rate(output),
        "coherence": coherence_from_ppl(ppl),
        "perplexity": ppl,
        "lexical_diversity": lexical_diversity(output),
        "associative_drift": associative_drift(model, tokenizer, prompt, output, mmax),
        "dissolution_rate": dissolution_rate(output),
        # new: psychedelic-signature + language-structure metrics
        "attention_entropy": attn_ent,
        "inter_sentence_distance": inter_sentence_distance(model, tokenizer, output, mmax),
        "mean_sentence_length": mean_sentence_length(output),
        "fragmentation_rate": fragmentation_rate(output),
        # structural null-tests: local vs global, finer self-reference
        "local_semantic_jump": local_semantic_jump(model, tokenizer, output),
        "referential_continuity": referential_continuity(output),
        "collective_pronoun_fraction": collective_pronoun_fraction(output),
    }

def load_model():
    if torch.cuda.is_available(): device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    print(f"[load] {HF_MODEL} on {device} ({dtype})")
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL, torch_dtype=dtype,
        device_map=device if device != "cpu" else None, attn_implementation="eager")
    if device == "cpu": model = model.to(device)
    model.eval()
    return model, tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--metric-max-tokens", type=int, default=128)
    ap.add_argument("--base-seed", type=int, default=7)
    ap.add_argument("--beta-ratio", type=float, default=0.65)
    ap.add_argument("--beta-layers", default="2,3")
    ap.add_argument("--persona-vector", default="persona_vectors/dissolved_response_avg_diff.pt")
    ap.add_argument("--persona-layer", type=int, default=13)
    ap.add_argument("--persona-coef", type=float, default=1.2)
    ap.add_argument("--outdir", default="compare_results")
    ap.add_argument("--show-samples", type=int, default=2)
    args = ap.parse_args()

    model, tokenizer = load_model()
    patch_llama_attention()
    STATE.beta_ratio_val = args.beta_ratio
    STATE.beta_layers = tuple(int(x) for x in args.beta_layers.split(",") if x.strip())
    STATE.persona_coef = args.persona_coef
    assert_patch_live(model, tokenizer)

    if not os.path.exists(args.persona_vector):
        raise SystemExit(f"persona vector not found: {args.persona_vector}")
    vec = torch.load(args.persona_vector, map_location="cpu")
    if vec.dim() != 2 or not (0 <= args.persona_layer < vec.shape[0]):
        raise SystemExit(f"bad vector shape {tuple(vec.shape)} / layer {args.persona_layer}")
    layer_vec = vec[args.persona_layer].to(model.device, model.dtype)
    handle = install_persona_hook(model, layer_vec, args.persona_layer)

    prompts = PROMPTS[:args.num_prompts]
    rows, samples = [], []
    for pi, p in enumerate(prompts):
        for si in range(args.seeds):
            seed = args.base_seed + pi * 100 + si
            STATE.beta_on = False; STATE.persona_on = False
            sober = generate_app_faithful(model, tokenizer, p, False, seed)
            s = score(model, tokenizer, p, sober, args.metric_max_tokens, beta_for_entropy=False)
            s.update(condition="sober", prompt_idx=pi, seed=si, output=sober)
            rows.append(s)

            STATE.beta_on = True; STATE.persona_on = True
            kal = generate_app_faithful(model, tokenizer, p, True, seed)
            STATE.beta_on = False; STATE.persona_on = False
            k = score(model, tokenizer, p, kal, args.metric_max_tokens, beta_for_entropy=True)
            k.update(condition="kaleido", prompt_idx=pi, seed=si, output=kal)
            rows.append(k)

            if si == 0 and pi < args.show_samples:
                samples.append((p, sober, kal))

    handle.remove()
    df = pd.DataFrame(rows)
    metrics = ["first_person_rate","coherence","lexical_diversity",
               "associative_drift","dissolution_rate",
               "attention_entropy","inter_sentence_distance",
               "mean_sentence_length","fragmentation_rate",
               "local_semantic_jump","referential_continuity",
               "collective_pronoun_fraction"]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "compare_per_prompt.csv", index=False)

    print("\n" + "=" * 70)
    print(f"  SOBER vs KALEIDO  (app-faithful: temp {APP_TEMPERATURE}, "
          f"{args.seeds} seeds x {len(prompts)} prompts)")
    print(f"  beta={args.beta_ratio} on {list(STATE.beta_layers)}  |  "
          f"persona coef={args.persona_coef} layer {args.persona_layer}")
    print("=" * 70)
    print(f"{'metric':<22}{'sober':>15}{'kaleido':>15}{'delta':>12}")
    print("-" * 70)
    summary = {}
    for m in metrics:
        sm = df[df.condition=="sober"][m].mean(); ss = df[df.condition=="sober"][m].std()
        km = df[df.condition=="kaleido"][m].mean(); ks = df[df.condition=="kaleido"][m].std()
        print(f"{m:<22}{sm:>9.3f}+-{ss:<4.2f}{km:>9.3f}+-{ks:<4.2f}{km-sm:>+12.3f}")
        summary[m] = {"sober_mean":float(sm),"sober_sd":float(ss),
                      "kaleido_mean":float(km),"kaleido_sd":float(ks),"delta":float(km-sm)}
    print("=" * 70)

    print("\nREADING THE NUMBERS:")
    print("  ENGINEERED (steered-for; expected to move, NOT a finding):")
    print(f"    first_person_rate d{summary['first_person_rate']['delta']:+.3f}   "
          f"dissolution_rate d{summary['dissolution_rate']['delta']:+.3f}")
    print("  PSYCHEDELIC SIGNATURE (EBH quantity; raised by the β patch by construction):")
    print(f"    attention_entropy d{summary['attention_entropy']['delta']:+.3f}  (nats; "
          f"higher = flatter attention, the EBH marker)")
    print("  LANGUAGE STRUCTURE (NOT directly steered — emergent if it moves):")
    print(f"    inter_sentence_distance d{summary['inter_sentence_distance']['delta']:+.3f}  "
          f"(REBUS loosened-priors: bigger leaps between adjacent sentences)")
    print(f"    mean_sentence_length    d{summary['mean_sentence_length']['delta']:+.3f}   "
          f"fragmentation_rate d{summary['fragmentation_rate']['delta']:+.3f}  (proxy)")
    print("  THE COST / what wasn't optimized:")
    cd = summary["coherence"]["delta"]
    tag = ("largely HELD - dissolved yet legible" if cd > -0.1 else
           "moderate cost - legible but loosened (the edge KALEIDO rides)" if cd > -0.3 else
           "large cost - near breakdown")
    print(f"    coherence         d{cd:+.3f}   -> {tag}")
    print(f"    lexical_diversity d{summary['lexical_diversity']['delta']:+.3f}   "
          f"associative_drift d{summary['associative_drift']['delta']:+.3f}")
    print("  STRUCTURAL NULL-TESTS (could catch a real effect if one exists):")
    print(f"    local_semantic_jump      d{summary['local_semantic_jump']['delta']:+.3f}  "
          f"(local leaps vs the null global drift)")
    print(f"    referential_continuity   d{summary['referential_continuity']['delta']:+.3f}  "
          f"(global through-line; pair w/ coherence for local/global decoupling)")
    print(f"    collective_pronoun_frac  d{summary['collective_pronoun_fraction']['delta']:+.3f}  "
          f"(self dissolving I->we/all; finer than first-person rate)")
    print("\n  Read these as tests of the null, not psychedelic-ness scores. A null")
    print("  here makes the 'stylistic overlay, not structural change' finding")
    print("  load-bearing; a hit reveals a local/global decoupling worth reporting.")

    if samples:
        print("\nEXAMPLE PAIRS (seed 0):")
        for p, sob, kal in samples:
            print(f"\n  PROMPT: {p}")
            print(f"    sober  : {sob[:160]}")
            print(f"    kaleido: {kal[:160]}")

    with open(outdir / "compare_summary.json", "w") as f:
        json.dump({"config": vars(args), "summary": summary}, f, indent=2)
    print(f"\n[done] wrote {outdir}/compare_per_prompt.csv and compare_summary.json")

if __name__ == "__main__":
    main()