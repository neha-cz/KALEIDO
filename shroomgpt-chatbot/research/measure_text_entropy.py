"""
measure_text_entropy.py
=======================

Run the full intervention ablation grid, N sampled generations per condition,
and compute the text-as-signal entropy metrics on each output. Report mean ± std
per metric per condition, with every complexity metric paired against the
coherence guardrail (perplexity).

Conditions (the finished system + ablations):
  baseline / β / persona / dream-fix / dream-flux /
  β+persona / β+persona+dream-fix / β+persona+dream-flux

Calibrated params (locked in from the sweeps):
  β=0.45 (2,3) | persona coef 8 L9 | dream coef 20 L18

Run:
  python measure_text_entropy.py \
      --persona-vector persona_vectors/qwen_dissolved.pt \
      --dream-bank dream_bank_direct.pt \
      --n-samples 12
"""

import os
import json
import argparse
import contextlib
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import qwen_beta_persona as qbp
import text_entropy_metrics as tem

VLM_ID = os.environ.get("HF_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# Calibrated.
BETA = (0.45, (2, 3))
PERSONA = (8.0, 9)
DREAM_COEF = 20.0
DREAM_LAYER = 18

PROMPTS = [
    "Tell me about yourself.",
    "What is consciousness?",
    "Describe what you see around you.",
]
MAX_NEW = 80


@torch.no_grad()
def gen_and_measure(model, tok, prompt, *, beta=None, persona=None, dream=None,
                    persona_vec=None, dream_bank=None, seed=0):
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]

    beta_ctx = qbp.BETA.engaged(*beta) if beta else contextlib.nullcontext()
    persona_ctx = (qbp.PERSONA.engaged(persona[0], persona[1],
                                       persona_vec[persona[1]].to(model.device))
                   if persona else contextlib.nullcontext())
    dream_ctx = (qbp.DREAM.engaged(dream[0], dream[1], dream_bank.to(model.device),
                                   mode=dream[2], seed=seed)
                 if dream else contextlib.nullcontext())

    with beta_ctx, persona_ctx, dream_ctx:
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=0.9, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
    gen_ids = out[0, n_in:].tolist()
    eos = tok.eos_token_id
    while gen_ids and gen_ids[-1] == eos:
        gen_ids.pop()
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    # coherence guardrail: perplexity with NO interventions (clean base-model read)
    full = out[:, :n_in + len(gen_ids)]
    ppl = tem.coherence_perplexity(model, tok, full, n_in, gen_ids)

    return tem.compute_text_metrics(gen_ids, text, perplexity=ppl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona-vector", default="persona_vectors/qwen_dissolved.pt")
    ap.add_argument("--dream-bank", default="dream_bank_direct.pt")
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--outdir", default="entropy_results")
    args = ap.parse_args()

    print(f"[load] {VLM_ID} on {DEVICE} (eager)")
    dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        VLM_ID, torch_dtype=dtype, attn_implementation="eager").to(DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(VLM_ID)
    qbp.patch_qwen_attention()
    qbp.install_persona_hooks(model)
    qbp.install_dream_hooks(model)
    pvec = torch.load(args.persona_vector, map_location=DEVICE)
    bank_obj = torch.load(args.dream_bank, map_location=DEVICE)
    bank = bank_obj["vectors"] if isinstance(bank_obj, dict) else bank_obj
    print(f"[load] persona {tuple(pvec.shape)}, dream bank {tuple(bank.shape)}")

    # warm up sentence model (so the [metrics] message prints once up front)
    _ = tem.semantic_trajectory("First sentence here. Second sentence here.")

    conditions = [
        ("baseline",        dict()),
        ("beta",            dict(beta=BETA)),
        ("persona",         dict(persona=PERSONA)),
        ("dream_fix",       dict(dream=(DREAM_COEF, DREAM_LAYER, "fixed"))),
        ("dream_flux",      dict(dream=(DREAM_COEF, DREAM_LAYER, "flux"))),
        ("beta_persona",    dict(beta=BETA, persona=PERSONA)),
        ("triple_fix",      dict(beta=BETA, persona=PERSONA,
                                 dream=(DREAM_COEF, DREAM_LAYER, "fixed"))),
        ("triple_flux",     dict(beta=BETA, persona=PERSONA,
                                 dream=(DREAM_COEF, DREAM_LAYER, "flux"))),
    ]

    # results[cond][metric] = list of values across samples*prompts
    results = defaultdict(lambda: defaultdict(list))
    metric_keys = ["token_lzc", "token_entropy", "distinct2", "distinct3",
                   "semantic_jump", "semantic_spread", "perplexity"]

    total = len(conditions) * len(PROMPTS) * args.n_samples
    done = 0
    for cond_name, kwargs in conditions:
        for prompt in PROMPTS:
            for s in range(args.n_samples):
                m = gen_and_measure(model, tok, prompt, persona_vec=pvec,
                                    dream_bank=bank, seed=1000 * s + 7, **kwargs)
                d = m.as_dict()
                for k in metric_keys:
                    if d[k] is not None:
                        results[cond_name][k].append(d[k])
                done += 1
            print(f"  {cond_name:<14} {prompt[:28]:<30} "
                  f"({done}/{total})")

    # ---- report ----
    def fmt(cond, k):
        vals = results[cond][k]
        if not vals:
            return "    n/a   "
        return f"{np.mean(vals):6.3f}±{np.std(vals):.2f}"

    print("\n" + "=" * 108)
    print("TEXT-AS-SIGNAL ENTROPY METRICS  (mean ± std over "
          f"{args.n_samples} samples × {len(PROMPTS)} prompts)")
    print("=" * 108)
    cols = ["token_lzc", "token_entropy", "distinct2", "distinct3",
            "semantic_jump", "semantic_spread", "perplexity"]
    header = f"{'condition':<14}" + "".join(f"{c[:12]:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for cond_name, _ in conditions:
        row = f"{cond_name:<14}" + "".join(f"{fmt(cond_name, c):>14}" for c in cols)
        print(row)

    print("\nNOTE: complexity columns (lzc/entropy/distinct/semantic) read AGAINST")
    print("perplexity (rightmost). Higher complexity is 'psychedelic' ONLY if")
    print("perplexity stays in a sane range -- runaway perplexity = word salad,")
    print("which trivially maximizes every complexity metric.")

    # deltas vs baseline
    print("\n" + "=" * 108)
    print("DELTAS vs BASELINE")
    print("=" * 108)
    print(header)
    print("-" * len(header))
    base = {k: (np.mean(results["baseline"][k]) if results["baseline"][k] else None)
            for k in cols}
    for cond_name, _ in conditions:
        if cond_name == "baseline":
            continue
        cells = []
        for c in cols:
            vals = results[cond_name][c]
            if not vals or base[c] is None:
                cells.append("    n/a   ")
            else:
                cells.append(f"{np.mean(vals) - base[c]:+6.3f}      ")
        print(f"{cond_name:<14}" + "".join(f"{x:>14}" for x in cells))

    os.makedirs(args.outdir, exist_ok=True)
    serializable = {c: {k: results[c][k] for k in metric_keys} for c, _ in conditions}
    with open(os.path.join(args.outdir, "text_entropy_raw.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[done] raw per-sample values -> {args.outdir}/text_entropy_raw.json")


if __name__ == "__main__":
    main()
