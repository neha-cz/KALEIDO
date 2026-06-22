"""
measure_triple_stack.py
=======================

Focused measurement of the calibrated TRIPLE STACK and the ablations that
isolate the dream's and flux's marginal contributions. Fewer conditions, more
samples, with the SEMANTIC TRAJECTORY metrics (the connectivity / association
analogue) front and center -- that's the metric that tests whether concepts
range MORE WIDELY even as lexical diversity drops.

Conditions:
  baseline
  beta_persona            (the two-lever scaffold)
  triple_fix              (full stack, one dream per reply)
  triple_flux             (full stack, fresh dream per decode token)

Calibrated params: β=0.45 (2,3) | persona 8 L9 | dream 20 L18

Run:
  python measure_triple_stack.py \
      --persona-vector persona_vectors/qwen_dissolved.pt \
      --dream-bank dream_bank_direct.pt \
      --n-samples 20
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

BETA = (0.45, (2, 3))
PERSONA = (8.0, 9)
DREAM_COEF = 20.0
DREAM_LAYER = 18

PROMPTS = [
    "Tell me about yourself.",
    "What is consciousness?",
    "Describe what you see around you.",
]
MAX_NEW = 90


@torch.no_grad()
def gen_and_measure(model, tok, prompt, *, beta=None, persona=None, dream=None,
                    persona_vec=None, dream_bank=None, seed=0, keep_text=False):
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

    full = out[:, :n_in + len(gen_ids)]
    ppl = tem.coherence_perplexity(model, tok, full, n_in, gen_ids)
    m = tem.compute_text_metrics(gen_ids, text, perplexity=ppl)
    return (m, text) if keep_text else (m, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona-vector", default="persona_vectors/qwen_dissolved.pt")
    ap.add_argument("--dream-bank", default="dream_bank_direct.pt")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--outdir", default="triple_results")
    ap.add_argument("--dump-text", action="store_true",
                    help="print a few sample generations per condition")
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

    # Warm up / verify the semantic model BEFORE the long run, so you find out
    # immediately if it's going to be available.
    jt, st = tem.semantic_trajectory("A first sentence about cats. "
                                     "A second, very different sentence about astrophysics.")
    if jt is None:
        print("[warn] semantic metrics are DISABLED -- the connectivity columns will be "
              "n/a. Cache the model and rerun if you want them (see message above).")
    else:
        print(f"[ok] semantic metric live (test jump={jt:.3f} spread={st:.3f})")

    conditions = [
        ("baseline",     dict()),
        ("beta_persona", dict(beta=BETA, persona=PERSONA)),
        ("triple_fix",   dict(beta=BETA, persona=PERSONA,
                              dream=(DREAM_COEF, DREAM_LAYER, "fixed"))),
        ("triple_flux",  dict(beta=BETA, persona=PERSONA,
                              dream=(DREAM_COEF, DREAM_LAYER, "flux"))),
    ]

    metric_keys = ["token_lzc", "token_entropy", "distinct2", "distinct3",
                   "semantic_jump", "semantic_spread", "perplexity"]
    results = defaultdict(lambda: defaultdict(list))
    samples = defaultdict(list)

    total = len(conditions) * len(PROMPTS) * args.n_samples
    done = 0
    for cond_name, kwargs in conditions:
        for prompt in PROMPTS:
            for s in range(args.n_samples):
                keep = args.dump_text and s < 2
                m, text = gen_and_measure(model, tok, prompt, persona_vec=pvec,
                                          dream_bank=bank, seed=1000 * s + 7,
                                          keep_text=keep, **kwargs)
                d = m.as_dict()
                for k in metric_keys:
                    if d[k] is not None:
                        results[cond_name][k].append(d[k])
                if keep and text:
                    samples[cond_name].append((prompt, text))
                done += 1
        print(f"  {cond_name:<14} done ({done}/{total})")

    # ---- report ----
    def cell(cond, k):
        vals = results[cond][k]
        if not vals:
            return "    n/a    "
        return f"{np.mean(vals):6.3f}±{np.std(vals):.2f}"

    cols = ["token_lzc", "token_entropy", "distinct2", "distinct3",
            "semantic_jump", "semantic_spread", "perplexity"]
    print("\n" + "=" * 116)
    print(f"TRIPLE-STACK METRICS  (mean ± std, {args.n_samples} samples × {len(PROMPTS)} prompts)")
    print("=" * 116)
    header = f"{'condition':<14}" + "".join(f"{c[:13]:>15}" for c in cols)
    print(header)
    print("-" * len(header))
    for cond_name, _ in conditions:
        print(f"{cond_name:<14}" + "".join(f"{cell(cond_name, c):>15}" for c in cols))

    print("\nKEY READING:")
    print("  - lexical metrics (lzc/entropy/distinct) DOWN = more repetitive/incantatory")
    print("  - semantic_jump/spread UP = wider conceptual range (the connectivity analogue)")
    print("  - perplexity is the coherence guardrail; the dissociation to look for is")
    print("    LEXICAL DIVERSITY DOWN while SEMANTIC SPREAD UP -- tight loops, wide concepts.")

    # deltas
    print("\n" + "=" * 116)
    print("DELTAS vs BASELINE")
    print("=" * 116)
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
                cells.append("   n/a   ")
            else:
                cells.append(f"{np.mean(vals) - base[c]:+7.3f}    ")
        print(f"{cond_name:<14}" + "".join(f"{x:>15}" for x in cells))

    if args.dump_text:
        print("\n" + "=" * 116)
        print("SAMPLE GENERATIONS")
        print("=" * 116)
        for cond_name, _ in conditions:
            print(f"\n### {cond_name} ###")
            for prompt, text in samples[cond_name][:3]:
                print(f"  [{prompt}]\n  {text}\n")

    os.makedirs(args.outdir, exist_ok=True)
    serializable = {c: {k: results[c][k] for k in metric_keys} for c, _ in conditions}
    with open(os.path.join(args.outdir, "triple_metrics_raw.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[done] raw values -> {args.outdir}/triple_metrics_raw.json")


if __name__ == "__main__":
    main()
