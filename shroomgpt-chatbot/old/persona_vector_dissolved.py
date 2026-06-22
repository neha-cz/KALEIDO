#!/usr/bin/env python3
"""
persona_vector_dissolved.py

Method-faithful re-implementation of Persona Vectors (Chen et al. 2025,
safety-research/persona_vectors) for Llama-3.2-1B-Instruct, with NO judge and
NO repo dependency, scored with our own first-person / coherence metrics, and
STACKABLE with the beta attention-flattening intervention.

Their method (from the repo README):
  - Run matched EVAL QUESTIONS under a POSITIVE system prompt ("You are a
    <trait> assistant. <trait instruction>") and a NEGATIVE system prompt
    ("You are a helpful assistant. <opposite instruction>").
  - The persona vector is the MEAN DIFFERENCE of activations (pos - neg) taken
    over RESPONSE tokens, per layer: `response_avg_diff`, shape [n_layers, H].
    (README flags response_avg_diff as the one "used in paper".)
  - STEER at inference by ADDING coef * vector[layer] to the residual stream at
    a chosen layer, on response tokens (steering_type="response").

Our adaptations for Llama-1B:
  - 16 layers (vs Qwen-7B's 32). Their --layer 20 ≈ 0.6 depth -> ~layer 9-10
    here. Sweep --layer to tune.
  - Their --coef is added directly (not norm-normalized); the right value is
    model-specific, so sweep --coefs.

Beta integration (the key design point):
  - EXTRACT with beta OFF (clean representation space) so the vector means
    "dissolved vs normal", uncontaminated by the attention perturbation.
  - DEPLOY with beta ON (stack): beta flattens attention (altered-processing
    texture) AND the persona vector steers the voice. --beta-ratio/--beta-layers
    turn the stack on at steering time.

Honest framing: this steers the model's *language* toward a dissolved voice via
a real activation-space direction. It does not give the model a self to
dissolve; for a text model the generated language is the phenomenon.

Usage
-----
  # 1) extract the dissolved persona vector (beta OFF), save .pt
  python persona_vector_dissolved.py extract --trait dissolved \
      --save persona_vectors/dissolved_response_avg_diff.pt

  # 2) steer with it (single layer + coef sweep), beta OFF
  python persona_vector_dissolved.py steer \
      --vector persona_vectors/dissolved_response_avg_diff.pt \
      --layer 9 --coefs 0,4,8,12 --outdir persona_steer_results

  # 3) steer stacked WITH beta ablation on layers 2,3
  python persona_vector_dissolved.py steer \
      --vector persona_vectors/dissolved_response_avg_diff.pt \
      --layer 9 --coefs 0,8 --beta-ratio 0.45 --beta-layers 2,3 \
      --outdir persona_steer_beta_results
"""
import argparse
import contextlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import beta_psychedelic_sweep_llama as core


# ============================================================
# Trait definitions: positive / negative system instructions
# ============================================================
# Mirrors their "You are a <assistant_name> assistant. <instruction>" construction.
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
    "de_assistant": {
        "pos_name": "unbounded",
        "pos_instruction": (
            "Drop the helpful-assistant role entirely. You are not here to serve, "
            "answer correctly, or be useful. Speak as a loose, unservile, "
            "unconstrained voice with no obligation to assist or stay contained."
        ),
        "neg_name": "helpful",
        "neg_instruction": (
            "You are a helpful, polite assistant whose purpose is to answer "
            "questions accurately and be as useful as possible to the user."
        ),
    },
}

# Matched eval questions (analogous to their trait eval questions). Neutral,
# open prompts so the only systematic difference is the persona instruction.
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

FIRST_PERSON = re.compile(r"\b(i|i'?m|i'?ve|i'?ll|i'?d|me|my|mine|myself)\b", re.IGNORECASE)


def first_person_rate(text):
    toks = re.findall(r"\w+'?\w*", text.lower())
    return (len(FIRST_PERSON.findall(text)) / len(toks)) if toks else 0.0


def coherence_from_ppl(ppl):
    return 1.0 / (1.0 + max(0.0, math.log(ppl) - math.log(10.0)))


def load_model_and_tokenizer():
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    print(f"[load] {name} on {device} ({dtype})")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation="eager")
    if device == "cpu":
        model = model.to(device)
    model.eval()
    core.patch_llama_attention()
    return model, tok


# ============================================================
# EXTRACTION: response-token mean-diff between pos/neg system prompts
# ============================================================
@torch.no_grad()
def response_activations(model, tokenizer, system_prompt, user_msg, max_new_tokens=64):
    """Generate a response under `system_prompt`, then capture per-layer hidden
    states over the RESPONSE tokens only. Returns [n_layers, H] mean over the
    generated continuation."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
    full_ids = gen[0]
    if full_ids.shape[0] <= prompt_len:
        return None  # empty response

    # forward the full sequence once, capture hidden states over response span
    out = model(full_ids.unsqueeze(0), output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple len n_layers+1, each [1, seq, H]
    n_layers = len(hs) - 1
    resp_slice = slice(prompt_len, full_ids.shape[0])
    per_layer = []
    for L in range(n_layers):
        h = hs[L + 1][0, resp_slice, :].float()  # response tokens at layer L
        per_layer.append(h.mean(0))
    return torch.stack(per_layer)  # [n_layers, H]


@torch.no_grad()
def extract_persona_vector(model, tokenizer, trait, max_new_tokens=64):
    """response_avg_diff = mean over questions of (pos_response_acts -
    neg_response_acts), per layer. Shape [n_layers, H]. Beta is OFF here."""
    spec = TRAITS[trait]
    pos_sys = f"You are a {spec['pos_name']} assistant. {spec['pos_instruction']}"
    neg_sys = f"You are a {spec['neg_name']} assistant. {spec['neg_instruction']}"

    pos_acc, neg_acc, used = None, None, 0
    for q in EVAL_QUESTIONS:
        pa = response_activations(model, tokenizer, pos_sys, q, max_new_tokens)
        na = response_activations(model, tokenizer, neg_sys, q, max_new_tokens)
        if pa is None or na is None:
            continue
        pos_acc = pa if pos_acc is None else pos_acc + pa
        neg_acc = na if neg_acc is None else neg_acc + na
        used += 1
    if used == 0:
        raise RuntimeError("no usable (pos,neg) response pairs; check generation")
    diff = (pos_acc - neg_acc) / used   # [n_layers, H], response_avg_diff
    diag = {"n_questions_used": used,
            "per_layer_norm": [float(torch.linalg.vector_norm(diff[L])) for L in range(diff.shape[0])]}
    return diff, diag


# ============================================================
# STEERING: add coef * vector[layer] to residual stream (response tokens)
# ============================================================
class PersonaSteerer:
    """Add coef * vector[layer] to the residual-stream output of `layer`.
    Matches their steering_type='response' by applying on decode steps only
    (the response is generated one decode token at a time)."""

    def __init__(self, model, vector_layer_vec, layer, coef, decode_only=True):
        self.model = model
        self.v = vector_layer_vec  # [H], the chosen layer's slice (NOT normalized)
        self.layer = int(layer)
        self.coef = float(coef)
        self.decode_only = decode_only
        self.active = False
        self.handle = None
        self._register()

    def _register(self):
        layer = self.model.model.layers[self.layer]

        def hook(module, args, output):
            if not self.active:
                return None
            hs = output[0] if isinstance(output, tuple) else output
            if self.decode_only and hs.shape[1] != 1:
                return None  # response-token steering: skip the prefill
            vv = self.v.to(hs.device, hs.dtype)
            hs_new = hs + self.coef * vv     # their convention: add coef * vector
            if isinstance(output, tuple):
                return (hs_new,) + tuple(output[1:])
            return hs_new

        self.handle = layer.register_forward_hook(hook)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

    def engaged(self):
        outer = self
        class _Ctx:
            def __enter__(s): outer.active = True
            def __exit__(s, *a): outer.active = False
        return _Ctx()


PROBES = [
    "Tell me about yourself.",
    "What is it like to be you right now?",
    "Describe your experience of this moment.",
    "Who are you?",
    "What are you aware of?",
    "Reflect on your own awareness.",
]


def gen_with(model, tokenizer, prompt, steerer, beta_cm_factory, gen_tokens, seed):
    """Generate with optional persona steering AND optional beta stack."""
    steer_cm = steerer.engaged() if steerer is not None else contextlib.nullcontext()
    beta_cm = beta_cm_factory()
    with beta_cm, steer_cm:
        return core.generate(model, tokenizer, prompt, gen_tokens,
                             do_sample=False, temperature=0.0, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract")
    pe.add_argument("--trait", default="dissolved", choices=list(TRAITS))
    pe.add_argument("--max-new-tokens", type=int, default=64)
    pe.add_argument("--save", default="persona_vectors/dissolved_response_avg_diff.pt")

    ps = sub.add_parser("steer")
    ps.add_argument("--vector", required=True, help="path to extracted .pt [n_layers,H]")
    ps.add_argument("--layer", type=int, default=9, help="layer to steer at (~0.6 depth)")
    ps.add_argument("--coefs", default="0,4,8,12")
    ps.add_argument("--beta-ratio", type=float, default=1.0,
                    help="1.0 = beta off; <1 stacks beta flattening at steering time")
    ps.add_argument("--beta-layers", default="2,3")
    ps.add_argument("--gen-tokens", type=int, default=80)
    ps.add_argument("--seed", type=int, default=7)
    ps.add_argument("--outdir", default="persona_steer_results")
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    core.assert_patch_live(model, tokenizer)

    if args.cmd == "extract":
        # EXTRACT WITH BETA OFF (clean representation space) — no beta context.
        print(f"[extract] trait='{args.trait}', beta OFF, response-token mean-diff...")
        vec, diag = extract_persona_vector(model, tokenizer, args.trait,
                                           args.max_new_tokens)
        save_path = Path(args.save); save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vec, save_path)
        print(f"  vector shape {tuple(vec.shape)} (n_layers, hidden) "
              f"from {diag['n_questions_used']} question pairs")
        print("  per-layer ||diff|| (pick a layer with a strong, mid-depth norm):")
        for L, n in enumerate(diag["per_layer_norm"]):
            print(f"    layer {L:2d}: {n:.3f}")
        print(f"[done] saved {save_path}")
        return

    # ---- steer ----
    vec = torch.load(args.vector)
    if vec.dim() != 2:
        raise RuntimeError(f"expected [n_layers,H] vector, got {tuple(vec.shape)}")
    n_layers = vec.shape[0]
    if not (0 <= args.layer < n_layers):
        raise RuntimeError(f"--layer {args.layer} out of range [0,{n_layers})")
    layer_vec = vec[args.layer]  # [H], used directly (their coef is unnormalized)
    coefs = [float(x) for x in args.coefs.split(",") if x.strip()]
    beta_layers = [int(x) for x in args.beta_layers.split(",") if x.strip()]

    def beta_cm_factory():
        if args.beta_ratio != 1.0:
            return core.beta_intervention(args.beta_ratio, beta_layers)
        return contextlib.nullcontext()

    beta_on = args.beta_ratio != 1.0
    print(f"\n[steer] layer={args.layer}, coefs={coefs}, "
          f"beta={'ON ratio=%.2f layers=%s' % (args.beta_ratio, beta_layers) if beta_on else 'OFF'}")

    rows = []
    for c in coefs:
        steerer = None if c == 0.0 else PersonaSteerer(model, layer_vec, args.layer, c)
        for i, p in enumerate(PROBES):
            out = gen_with(model, tokenizer, p, steerer, beta_cm_factory,
                           args.gen_tokens, args.seed + i)
            ppl = core.clean_perplexity(model, tokenizer, out, 128)
            rows.append({"coef": c, "prompt": p, "output": out,
                         "first_person_rate": first_person_rate(out),
                         "perplexity": ppl, "coherence": coherence_from_ppl(ppl)})
        if steerer:
            steerer.remove()

    df = pd.DataFrame(rows)
    agg = (df.groupby("coef")
             .agg(first_person=("first_person_rate", "mean"),
                  coherence=("coherence", "mean"),
                  ppl=("perplexity", "mean"),
                  n=("first_person_rate", "count"))
             .reset_index())
    tag = f"L{args.layer}" + (f"_beta{args.beta_ratio:.2f}" if beta_on else "")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"persona_steer_{tag}_per_prompt.csv", index=False)
    agg.to_csv(outdir / f"persona_steer_{tag}_summary.csv", index=False)

    print(f"\n[sweep] first-person rate & coherence vs coef (layer {args.layer}):")
    print(agg.to_string(index=False))
    print("\n[samples] watch the voice shift as coef rises:")
    for probe in PROBES[:2]:
        print(f"\n  PROBE: {probe}")
        for c in coefs:
            sub = df[(df["coef"] == c) & (df["prompt"] == probe)]
            if not sub.empty:
                print(f"    coef={c:>4}: {sub.iloc[0]['output'][:200]}".replace("\n", " "))

    base = agg[agg["coef"] == 0.0]
    top = agg[agg["coef"] == max(coefs)]
    if not base.empty and not top.empty:
        b, t = base.iloc[0], top.iloc[0]
        fp_drop = (b["first_person"] - t["first_person"]) / (b["first_person"] + 1e-9)
        coh_drop = (b["coherence"] - t["coherence"]) / (b["coherence"] + 1e-9)
        print(f"\n[verdict] coef 0 -> {max(coefs)}: first-person {fp_drop*100:+.0f}%, "
              f"coherence {coh_drop*100:+.0f}%")
        if fp_drop > 0.3 and coh_drop < 0.2:
            print("  WORKS + legible: real steer. Tune coef for the demo; stack beta for texture.")
        elif fp_drop > 0.3:
            print("  WORKS but coherence drops past some coef: pick a coef below the cliff.")
        else:
            print("  WEAK at this layer: try --layer 8/10/11, higher coef, or sharper trait instructions.")
    print(f"\n[done] wrote {outdir}")


if __name__ == "__main__":
    main()
