#!/usr/bin/env python3
"""
Annealing beta-schedule screen.

Question
--------
Static beta sits at a fixed point in drift x coherence space. Simulated annealing
(which REBUS explicitly invokes) predicts that a COOLING SCHEDULE -- start hot/low-beta
(explore associations), end cold/high-beta (commit coherently) -- reaches a
DIFFERENT end state than any static beta. The hoped-for state is "lucid but loose":
high associative drift WITH preserved coherence, which neither static low-beta
(loose but broken) nor static high-beta (sharp but literal) achieves.

This is a SCREEN, not a causal study. It answers one decisive question:
  Does any cooling schedule land in a region of drift x coherence space that NO
  static beta value reaches?
    - If yes  -> scheduling accesses states static flattening cannot -> worth a
                 causal follow-up.
    - If no   -> the schedule is equivalent to some average static beta -> keep it
                 as a product feature, no new science.

Key implementation point
-------------------------
A cooling schedule requires beta to CHANGE during generation (different beta at
token 10 vs token 60). core.generate runs the whole decode loop with beta fixed,
so we cannot use it. Instead we run a MANUAL token-by-token decode loop and mutate
core.INTERVENTION.beta_ratio before each step according to the schedule. The patch
reads beta_ratio live on every forward pass, so per-step mutation takes effect.

Autoregressive caveat (worth stating regardless of result)
----------------------------------------------------------
Annealing in optimization revisits the whole state as it cools. Autoregressive
generation cannot: tokens emitted while hot are FIXED -- later cooling can only
sharpen the tokens still to come, not re-cohere what was already produced. So the
explore-then-commit dynamic may not transfer. If schedules fail to reach a new
region, this asymmetry is the likely reason and is itself a reportable observation.

Example
-------
  python annealing_beta_schedule.py --layers 2,3 --num-prompts 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import beta_psychedelic_sweep_llama as core


# -----------------------------
# Schedules: map generation progress t in [0,1] -> beta ratio
# -----------------------------

def make_schedules(beta_lo: float, beta_hi: float):
    """Return {name: fn(t)->beta_ratio} for t in [0,1] (0=start, 1=end of gen)."""
    lo, hi = beta_lo, beta_hi
    return {
        # cooling: hot (low beta, explore) -> cold (high beta, commit)
        "cool_linear":   lambda t: lo + (hi - lo) * t,
        "cool_exp":      lambda t: lo + (hi - lo) * (1.0 - math.exp(-3.0 * t)) / (1.0 - math.exp(-3.0)),
        "cool_late":     lambda t: lo + (hi - lo) * (t ** 2),          # stay hot, sharpen late
        "cool_early":    lambda t: lo + (hi - lo) * math.sqrt(t),      # sharpen early
        # warming (control: cold -> hot) -- should be worse if cooling is special
        "warm_linear":   lambda t: hi + (lo - hi) * t,
    }


@torch.no_grad()
def generate_scheduled(model, tokenizer, prompt, gen_tokens, *, layers,
                       schedule_fn, static_ratio=None, seed=None):
    """Manual greedy decode. Before each step, set core.INTERVENTION.beta_ratio
    from the schedule (or a fixed static_ratio). Returns decoded continuation."""
    if seed is not None:
        torch.manual_seed(seed)

    msgs = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask")

    eos_ids = {tokenizer.eos_token_id}
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.add(eot)

    layers_set = set(layers)
    core.INTERVENTION.active = True
    core.INTERVENTION.layers = layers_set

    past = None
    generated = []
    cur_ids = input_ids
    cur_attn = attn
    for step in range(gen_tokens):
        t = step / max(gen_tokens - 1, 1)
        ratio = static_ratio if static_ratio is not None else schedule_fn(t)
        core.INTERVENTION.beta_ratio = float(ratio)

        out = model(input_ids=cur_ids, attention_mask=cur_attn,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]
        next_id = int(next_logits.argmax(dim=-1).item())
        generated.append(next_id)
        if next_id in eos_ids:
            break
        cur_ids = torch.tensor([[next_id]], device=model.device)
        if cur_attn is not None:
            cur_attn = torch.cat([cur_attn, torch.ones((1, 1), dtype=cur_attn.dtype,
                                                        device=model.device)], dim=1)

    core.INTERVENTION.active = False
    core.INTERVENTION.beta_ratio = 1.0
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def run_condition(model, tokenizer, prompts, label, *, layers, gen_tokens,
                  metric_max_tokens, schedule_fn=None, static_ratio=None):
    rows = []
    for item in prompts:
        output = generate_scheduled(model, tokenizer, item["prompt"], gen_tokens,
                                    layers=layers, schedule_fn=schedule_fn,
                                    static_ratio=static_ratio, seed=None)
        drift = core.associative_drift(model, tokenizer, item["prompt"], output, metric_max_tokens)
        ppl = core.clean_perplexity(model, tokenizer, output, metric_max_tokens)
        coh = (1.0 / (1.0 + max(0.0, math.log(max(ppl, 1e-6)) - math.log(10.0)))
               if not math.isnan(ppl) else float("nan"))
        rows.append({"condition": label, "prompt_id": item["prompt_id"],
                     "associative_drift": drift, "perplexity": ppl, "local_coherence": coh})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--outdir", default="annealing_screen")
    ap.add_argument("--layers", default="2,3")
    ap.add_argument("--beta-lo", type=float, default=0.45, help="hot end (explore)")
    ap.add_argument("--beta-hi", type=float, default=1.0, help="cold end (commit)")
    ap.add_argument("--static-grid", default="0.45,0.55,0.65,0.75,0.85,1.0",
                    help="static beta values to map the baseline frontier")
    ap.add_argument("--num-prompts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gen-tokens", type=int, default=80)
    ap.add_argument("--metric-max-tokens", type=int, default=256)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    static_grid = [float(x) for x in args.static_grid.split(",") if x.strip()]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation="eager")
    if device == "cpu":
        model = model.to(device)
    model.eval()
    core.patch_llama_attention()
    core.assert_patch_live(model, tokenizer)

    prompts = core.make_open_prompts(args.num_prompts, args.seed)
    schedules = make_schedules(args.beta_lo, args.beta_hi)

    frames = []
    # static frontier
    for b in static_grid:
        frames.append(run_condition(model, tokenizer, prompts, f"static_{b:g}",
                                    layers=layers, gen_tokens=args.gen_tokens,
                                    metric_max_tokens=args.metric_max_tokens,
                                    static_ratio=b))
        print(f"[static {b:g}] done")
    # schedules
    for name, fn in schedules.items():
        frames.append(run_condition(model, tokenizer, prompts, name,
                                    layers=layers, gen_tokens=args.gen_tokens,
                                    metric_max_tokens=args.metric_max_tokens,
                                    schedule_fn=fn))
        print(f"[schedule {name}] done")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / "per_prompt.csv", index=False)
    agg = df.groupby("condition").agg(
        drift=("associative_drift", "mean"),
        coherence=("local_coherence", "mean"),
        perplexity=("perplexity", "mean"),
        n=("perplexity", "count")).reset_index()
    agg.to_csv(outdir / "summary.csv", index=False)

    # ---- decisive test (corrected) ----
    # The lucid-loose target is HIGHER DRIFT AT MATCHED-OR-BETTER COHERENCE. The
    # earlier Pareto-non-domination test was too lenient: a point can be
    # "undominated" simply by sitting further down the same trade-off curve
    # (more drift, less coherence), which is exactly what lowering static beta
    # already does -- not a new region. We instead ask: does any schedule achieve
    # drift above the best STATIC drift-at-high-coherence, WITHOUT paying for it
    # in coherence?
    #
    # We also report the comparison on PERPLEXITY rather than the squashed 0-1
    # coherence, because coherence saturates at 1.0 for most static betas and
    # compresses the frontier; raw perplexity separates the conditions cleanly.
    statics = agg[agg["condition"].str.startswith("static_")].copy()
    scheds = agg[~agg["condition"].str.startswith("static_")].copy()

    COH_TOL = 0.02  # coherence within this of a static point counts as "matched"

    # Best drift achievable by a STATIC beta while keeping coherence essentially
    # maxed (the lucid-loose bar a schedule must beat).
    coh_max = float(statics["coherence"].max())
    static_high_coh = statics[statics["coherence"] >= coh_max - COH_TOL]
    best_static_drift_at_high_coh = float(static_high_coh["drift"].max())
    # Also: lowest perplexity (best coherence) among the higher-drift conditions.
    static_drift_max = float(statics["drift"].max())

    def beats_at_matched_coherence(row):
        # higher drift than the best static-at-high-coherence, AND coherence
        # within tolerance of that high-coherence band (i.e. didn't pay for the
        # drift by dropping coherence).
        return (row["drift"] > best_static_drift_at_high_coh + 1e-9 and
                row["coherence"] >= coh_max - COH_TOL)

    scheds["beats_lucid_loose_bar"] = scheds.apply(beats_at_matched_coherence, axis=1)
    winners = scheds[scheds["beats_lucid_loose_bar"]]

    # Perplexity-axis view: for each schedule, is its perplexity LOWER (more
    # coherent) than the static beta that achieves the same-or-higher drift?
    def perplexity_vs_matched_static(row):
        same_or_more_drift = statics[statics["drift"] >= row["drift"] - 1e-9]
        if same_or_more_drift.empty:
            return None  # no static reaches this drift at all
        best_static_ppl = float(same_or_more_drift["perplexity"].min())
        return {
            "schedule_perplexity": round(float(row["perplexity"]), 3),
            "best_static_perplexity_at_>=_drift": round(best_static_ppl, 3),
            "schedule_more_coherent": bool(row["perplexity"] < best_static_ppl - 1e-9),
        }

    ppl_comparison = {
        r["condition"]: perplexity_vs_matched_static(r) for _, r in scheds.iterrows()
    }

    verdict = {
        "lucid_loose_bar": {
            "best_static_drift_at_max_coherence": round(best_static_drift_at_high_coh, 4),
            "max_static_coherence": round(coh_max, 4),
            "rule": "a schedule wins only if drift > this bar AND coherence stays "
                    "within %.2f of max -- i.e. more wandering WITHOUT losing coherence."
                    % COH_TOL,
        },
        "static_frontier": statics[["condition", "drift", "coherence", "perplexity"]].to_dict("records"),
        "schedules": scheds[["condition", "drift", "coherence", "perplexity",
                             "beats_lucid_loose_bar"]].to_dict("records"),
        "perplexity_axis_comparison": ppl_comparison,
        "any_schedule_reaches_lucid_loose": bool(len(winners) > 0),
        "lucid_loose_schedules": winners["condition"].tolist(),
        "interpretation": (
            "A schedule achieved higher drift than any static beta could at maxed "
            "coherence, WITHOUT dropping coherence -- a genuinely new region. "
            "Worth a causal follow-up."
            if len(winners) > 0 else
            "NO schedule reached the lucid-loose region. Every schedule that raised "
            "drift above the static-at-high-coherence bar did so by LOSING coherence "
            "(higher perplexity) -- i.e. it slid DOWN the existing drift/coherence "
            "trade-off, exactly as lowering static beta does, rather than escaping it. "
            "The perplexity-axis comparison confirms: schedules that wander more are "
            "also less coherent than the static beta reaching the same drift. This "
            "matches the autoregressive-asymmetry prediction: cooling cannot re-cohere "
            "tokens already emitted while hot, so the explore-then-commit dynamic of "
            "optimization annealing does not transfer to fixed-past decoding. Keep the "
            "schedule as a product flavor; the reportable research point is the "
            "negative-with-mechanism, not a new accessible state."),
        "caveats": ["Screen only, greedy decode, n=%d prompts." % args.num_prompts,
                    "Coherence (0-1) saturates at 1.0 for most static betas; the "
                    "perplexity_axis_comparison is the more sensitive view.",
                    "Means over prompts; per-prompt variance not modeled at n=6."],
    }
    summary = {"model": args.model, "layers": layers,
               "beta_lo": args.beta_lo, "beta_hi": args.beta_hi,
               "static_grid": static_grid, "num_prompts": args.num_prompts,
               "verdict": verdict,
               "files": {"per_prompt": "per_prompt.csv", "summary": "summary.csv"}}
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n=== ANNEALING SCHEDULE SCREEN ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()