#!/usr/bin/env python3
"""
Beta/layer sweep for Llama attention inverse-temperature interventions.

Default model matches the demo app:
    meta-llama/Llama-3.2-1B-Instruct

What it does
------------
1. Generates deterministic needle-in-a-haystack retrieval prompts.
2. Patches Llama eager attention so selected layers use
       softmax(beta_ratio * QK^T / sqrt(d))
   where beta_ratio=1.0 is baseline and lower values flatten attention.
3. Runs:
   - layer sweep: flatten one layer at a time at a chosen beta_ratio
   - beta sweep: flatten selected/sensitive layers across beta values
4. Measures:
   - task accuracy: exact needle recovered from answer
   - attention entropy: normalized entropy of attention maps
   - semantic drift: cosine distance between prompt and answer embeddings,
     using the same Llama model hidden states, so no extra embedding model needed
5. Writes CSVs, PNG plots, and summary.json with strange-layer candidates,
   demo-layer candidates, and possible beta* / cliff behavior.

Setup
-----
    pip install torch transformers accelerate pandas matplotlib tqdm
    huggingface-cli login  # if needed for Meta Llama gated access

Example
-------
    python beta_layer_sweep_llama.py \
      --num-prompts 24 \
      --layer-beta 0.35 \
      --beta-values 1.0,0.8,0.65,0.5,0.4,0.32,0.25,0.18,0.12,0.08 \
      --outdir beta_sweep_results

Notes
-----
- Keep do_sample=False. This is a causal intervention study, not a sampling-noise demo.
- Lower beta_ratio means flatter attention. It is a ratio relative to the model's
  native 1/sqrt(head_dim) attention scale.
- The entropy metric is normalized by log(sequence_length), so values are roughly
  in [0,1], where higher means flatter/more diffuse attention.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Global intervention state
# -----------------------------

@dataclass
class InterventionState:
    active: bool = False
    beta_ratio: float = 1.0
    layers: Optional[set[int]] = None

    def ratio_for_layer(self, layer_idx: int) -> float:
        if not self.active:
            return 1.0
        if self.layers is None or layer_idx in self.layers:
            return self.beta_ratio
        return 1.0

INTERVENTION = InterventionState()


def patch_llama_attention() -> None:
    """Patch HF Llama eager attention to multiply the attention scale per layer."""
    from transformers.models.llama import modeling_llama

    if getattr(modeling_llama.eager_attention_forward, "_beta_sweep_patched", False):
        return

    original = modeling_llama.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, **kwargs):
        layer_idx = getattr(module, "layer_idx", 0)
        r = INTERVENTION.ratio_for_layer(layer_idx)
        return original(module, query, key, value, attention_mask, scaling * r, **kwargs)

    patched._beta_sweep_patched = True  # type: ignore[attr-defined]
    patched._beta_sweep_original = original  # type: ignore[attr-defined]
    modeling_llama.eager_attention_forward = patched

    # Modern transformers dispatch through a registry. Patch known locations.
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
            elif hasattr(AttentionInterface, "register"):
                AttentionInterface.register("eager", patched)
                patched_registry = True
        except Exception:
            pass

    if not patched_registry:
        print("WARNING: could not patch the attention registry. The module symbol was patched, "
              "but your transformers version may dispatch elsewhere.")


@contextmanager
def beta_intervention(beta_ratio: float = 1.0, layers: Optional[Iterable[int]] = None):
    old = InterventionState(INTERVENTION.active, INTERVENTION.beta_ratio, INTERVENTION.layers)
    INTERVENTION.active = beta_ratio != 1.0
    INTERVENTION.beta_ratio = float(beta_ratio)
    INTERVENTION.layers = None if layers is None else set(int(x) for x in layers)
    try:
        yield
    finally:
        INTERVENTION.active = old.active
        INTERVENTION.beta_ratio = old.beta_ratio
        INTERVENTION.layers = old.layers


# -----------------------------
# Prompt/task generation
# -----------------------------

NAMES = [
    "Ada", "Basil", "Cora", "Dante", "Elena", "Felix", "Greta", "Hugo", "Iris", "Jules",
    "Kira", "Luca", "Mira", "Nolan", "Opal", "Pavel", "Quinn", "Rhea", "Silas", "Tara",
]
OBJECTS = [
    "amber violin", "blue compass", "copper lantern", "silver thimble", "green hourglass",
    "ivory key", "obsidian cup", "violet kite", "brass telescope", "crimson marble",
]
FILLERS = [
    "The archive was cataloged twice, once by date and once by texture.",
    "Every shelf carried a small paper label with an unrelated proverb.",
    "A quiet clerk moved between the aisles without disturbing the dust.",
    "The afternoon light made the windows look like panes of honey.",
    "Several notes in the margin referred to earlier, missing notebooks.",
    "The room contained maps, receipts, letters, tools, and old photographs.",
    "No one agreed on whether the collection was complete or merely abandoned.",
    "The index used a private shorthand that only half explained itself.",
]


def make_needle_prompts(n: int, seed: int, filler_sentences: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        name = rng.choice(NAMES) + f"-{i:03d}"
        obj = rng.choice(OBJECTS)
        code = f"{rng.choice(['AX','BR','CY','DN','EV'])}-{rng.randint(1000, 9999)}"
        needle = f"{name} hid the {obj} under code {code}."
        question = f"Who hid the {obj}, and what was the code?"
        answer = f"{name} {code}"

        haystack = [rng.choice(FILLERS) for _ in range(filler_sentences)]
        insert_at = rng.randrange(len(haystack) + 1)
        haystack.insert(insert_at, needle)
        context = " ".join(haystack)
        prompt = (
            "You are doing an exact retrieval task. Read the context and answer with only "
            "the person's identifier and the code, no explanation.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        prompts.append({"prompt_id": f"p{i:03d}", "prompt": prompt, "target": answer, "name": name, "code": code})
    return prompts


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def is_correct(answer: str, name: str, code: str) -> bool:
    a = normalize_text(answer)
    return name.lower() in a and code.lower() in a


# -----------------------------
# Metrics
# -----------------------------

@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()


@torch.no_grad()
def attention_entropy_for_text(model, tokenizer, text: str, max_tokens: int) -> Tuple[float, Dict[int, float]]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
    out = model(**enc, output_attentions=True, use_cache=False)
    attentions = out.attentions
    if attentions is None:
        return float("nan"), {}

    per_layer = {}
    vals = []
    eps = 1e-12
    for layer_idx, attn in enumerate(attentions):
        # attn: [batch, heads, query, key]
        p = attn.float().clamp_min(eps)
        ent = -(p * p.log()).sum(dim=-1)  # [batch, heads, query]
        key_len = p.shape[-1]
        norm = math.log(max(key_len, 2))
        ent_norm = (ent / norm).mean().item()
        per_layer[layer_idx] = ent_norm
        vals.append(ent_norm)
    return float(sum(vals) / len(vals)), per_layer


@torch.no_grad()
def mean_pool_embedding(model, tokenizer, text: str, max_tokens: int) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1].float()[0]
    mask = enc["attention_mask"][0].bool()
    emb = h[mask].mean(dim=0)
    return F.normalize(emb, dim=0).cpu()


@torch.no_grad()
def semantic_drift(model, tokenizer, prompt: str, answer: str, max_tokens: int) -> float:
    if not answer.strip():
        return 1.0
    e1 = mean_pool_embedding(model, tokenizer, prompt, max_tokens=max_tokens)
    e2 = mean_pool_embedding(model, tokenizer, answer, max_tokens=max_tokens)
    return float(1.0 - torch.dot(e1, e2).item())


def answer_health(answer: str) -> Dict[str, float]:
    toks = re.findall(r"\w+", answer.lower())
    if not toks:
        return {"answer_tokens": 0, "repeat_frac": 1.0, "coherence_proxy": 0.0}
    most_common = max(toks.count(t) for t in set(toks))
    repeat_frac = most_common / len(toks)
    # Crude proxy only: non-empty, not runaway repetition. Retrieval answers can be very short.
    coherence_proxy = 1.0 if len(toks) <= 30 and repeat_frac < 0.6 else 0.0
    return {"answer_tokens": len(toks), "repeat_frac": repeat_frac, "coherence_proxy": coherence_proxy}


# -----------------------------
# Experiment runner
# -----------------------------

@torch.no_grad()
def evaluate_condition(
    model,
    tokenizer,
    prompts: Sequence[Dict[str, str]],
    beta_ratio: float,
    layers: Optional[Iterable[int]],
    condition: str,
    max_new_tokens: int,
    metric_max_tokens: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    entropy_rows = []
    layer_list = None if layers is None else list(layers)

    with beta_intervention(beta_ratio=beta_ratio, layers=layer_list):
        for item in tqdm(prompts, desc=condition, leave=False):
            answer = generate_answer(model, tokenizer, item["prompt"], max_new_tokens=max_new_tokens)
            correct = is_correct(answer, item["name"], item["code"])
            drift = semantic_drift(model, tokenizer, item["prompt"], answer, max_tokens=metric_max_tokens)
            ent_mean, ent_by_layer = attention_entropy_for_text(
                model, tokenizer, item["prompt"] + "\n" + answer, max_tokens=metric_max_tokens
            )
            health = answer_health(answer)

            rows.append({
                "condition": condition,
                "prompt_id": item["prompt_id"],
                "beta_ratio": beta_ratio,
                "layers": "all" if layer_list is None else ",".join(map(str, layer_list)),
                "target": item["target"],
                "answer": answer,
                "correct": int(correct),
                "semantic_drift": drift,
                "attention_entropy": ent_mean,
                **health,
            })
            for li, ev in ent_by_layer.items():
                entropy_rows.append({
                    "condition": condition,
                    "prompt_id": item["prompt_id"],
                    "beta_ratio": beta_ratio,
                    "intervened_layers": "all" if layer_list is None else ",".join(map(str, layer_list)),
                    "attention_layer": li,
                    "attention_entropy": ev,
                })
    return pd.DataFrame(rows), pd.DataFrame(entropy_rows)


def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    return df.groupby(group_cols, dropna=False).agg(
        accuracy=("correct", "mean"),
        accuracy_sd=("correct", "std"),
        semantic_drift=("semantic_drift", "mean"),
        semantic_drift_sd=("semantic_drift", "std"),
        attention_entropy=("attention_entropy", "mean"),
        attention_entropy_sd=("attention_entropy", "std"),
        coherence_proxy=("coherence_proxy", "mean"),
        n=("correct", "count"),
    ).reset_index()


def detect_strange_layers(layer_summary: pd.DataFrame, baseline: Dict[str, float]) -> pd.DataFrame:
    df = layer_summary.copy()
    df["accuracy_drop"] = baseline["accuracy"] - df["accuracy"]
    df["drift_increase"] = df["semantic_drift"] - baseline["semantic_drift"]
    df["entropy_increase"] = df["attention_entropy"] - baseline["attention_entropy"]

    # Heuristic: sensitive enough to move, not totally collapsed, output still sane.
    df["strange_score"] = (
        2.0 * df["accuracy_drop"].clip(lower=0)
        + 1.0 * df["drift_increase"].clip(lower=0)
        + 1.0 * df["entropy_increase"].clip(lower=0)
    ) * df["coherence_proxy"].fillna(0)

    df["strange_candidate"] = (
        (df["strange_score"] > df["strange_score"].quantile(0.70))
        & (df["coherence_proxy"] >= 0.7)
        & (df["accuracy"] >= max(0.05, baseline["accuracy"] * 0.15))
    )
    return df.sort_values("strange_score", ascending=False)



def add_demo_layer_scores(
    layer_summary: pd.DataFrame,
    baseline: Dict[str, float],
    beta_sweep_layers: Optional[Sequence[int]],
) -> pd.DataFrame:
    """Rank beta-sweep layers by demo usefulness rather than scientific sensitivity.

    Trippy score is intentionally different from strange_score:
    - reward semantic drift and entropy increase: looseness / associative spread
    - reward preserved coherence and some preserved task ability: still grammatical / usable
    - penalize total collapse: accuracy too close to zero is usually bad for the demo

    This is a heuristic for Act 2, not a claim metric for Act 1.
    """
    if layer_summary.empty:
        return layer_summary

    df = layer_summary.copy()
    if beta_sweep_layers is not None:
        allowed = set(int(x) for x in beta_sweep_layers)
        df = df[df["layer"].astype(int).isin(allowed)].copy()

    if df.empty:
        return df

    # Ensure deltas exist even if this function is called on a raw aggregate.
    if "accuracy_drop" not in df:
        df["accuracy_drop"] = baseline["accuracy"] - df["accuracy"]
    if "drift_increase" not in df:
        df["drift_increase"] = df["semantic_drift"] - baseline["semantic_drift"]
    if "entropy_increase" not in df:
        df["entropy_increase"] = df["attention_entropy"] - baseline["attention_entropy"]

    baseline_acc = max(float(baseline.get("accuracy", 0.0)), 1e-9)
    preserved_accuracy = (df["accuracy"] / baseline_acc).clip(lower=0.0, upper=1.25)
    moderate_disruption = (1.0 - (df["accuracy_drop"].clip(lower=0.0) / baseline_acc)).clip(lower=0.0, upper=1.0)

    looseness = (
        2.0 * df["drift_increase"].clip(lower=0.0)
        + 1.0 * df["entropy_increase"].clip(lower=0.0)
        + 0.5 * df["accuracy_drop"].clip(lower=0.0)
    )

    df["trippy_score"] = (
        looseness
        * df["coherence_proxy"].fillna(0.0).clip(lower=0.0, upper=1.0)
        * (0.35 + 0.65 * preserved_accuracy)
        * (0.25 + 0.75 * moderate_disruption)
    )

    return df.sort_values("trippy_score", ascending=False)


def refine_beta_grid_from_cliff(
    beta_values: Sequence[float],
    coarse_beta_star: Dict[str, object],
    steps: int,
    margin: float,
) -> List[float]:
    """Create a dense beta grid around the largest coarse drop interval."""
    if steps < 3 or not coarse_beta_star.get("drop_interval"):
        return []
    lo_hi = coarse_beta_star["drop_interval"]
    if not isinstance(lo_hi, (list, tuple)) or len(lo_hi) != 2:
        return []
    hi = float(max(lo_hi))
    lo = float(min(lo_hi))
    lo = max(0.0, lo - margin)
    hi = min(1.0, hi + margin)
    if hi <= lo:
        return []
    vals = [hi - (hi - lo) * i / (steps - 1) for i in range(steps)]
    existing = {round(float(x), 6) for x in beta_values}
    refined = [round(v, 6) for v in vals if round(v, 6) not in existing]
    return sorted(set(refined), reverse=True)

def detect_beta_star(beta_summary: pd.DataFrame) -> Dict[str, object]:
    df = beta_summary.sort_values("beta_ratio", ascending=False).reset_index(drop=True)
    if len(df) < 3:
        return {"beta_star": None, "phase_transition_like": False, "reason": "Need at least 3 beta values."}

    acc = df["accuracy"].to_list()
    betas = df["beta_ratio"].to_list()
    drops = []
    slopes = []
    for i in range(1, len(df)):
        drop = acc[i - 1] - acc[i]
        width = max(abs(betas[i - 1] - betas[i]), 1e-12)
        drops.append(drop)
        slopes.append(drop / width)

    max_drop = max(drops)
    max_slope = max(slopes)
    idx = slopes.index(max_slope) + 1
    total_drop = max(acc) - min(acc)
    interval = [betas[idx - 1], betas[idx]]

    # Report beta* as the midpoint of the steepest measured interval. This is a better
    # estimate than simply choosing the lower endpoint when the grid is dense/refined.
    beta_star_midpoint = (float(interval[0]) + float(interval[1])) / 2.0

    # Cliff heuristic: the largest adjacent drop explains a lot of total decline.
    phase_like = bool(total_drop >= 0.25 and max_drop >= 0.20 and max_drop >= 0.45 * total_drop)
    return {
        "beta_star": beta_star_midpoint,
        "beta_star_interval": interval,
        "phase_transition_like": phase_like,
        "largest_adjacent_accuracy_drop": max_drop,
        "largest_accuracy_slope": max_slope,
        "total_accuracy_drop": total_drop,
        "drop_interval": interval,
        "interpretation": (
            "steepest measured accuracy cliff; beta_star is reported as the midpoint of that interval"
            if phase_like else
            "no strong cliff by the default heuristic; inspect plots or use a denser beta grid"
        ),
    }

def make_plots(layer_summary: pd.DataFrame, beta_summary: pd.DataFrame, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    if not layer_summary.empty:
        x = layer_summary["layer"].astype(int)
        for metric in ["accuracy", "semantic_drift", "attention_entropy", "strange_score"]:
            plt.figure()
            plt.plot(x, layer_summary[metric], marker="o")
            plt.xlabel("Intervened layer")
            plt.ylabel(metric)
            plt.title(f"Layer sweep: {metric}")
            plt.tight_layout()
            plt.savefig(outdir / f"layer_sweep_{metric}.png", dpi=160)
            plt.close()

    if not beta_summary.empty:
        x = beta_summary["beta_ratio"].astype(float)
        for metric in ["accuracy", "semantic_drift", "attention_entropy", "coherence_proxy"]:
            plt.figure()
            plt.plot(x, beta_summary[metric], marker="o")
            plt.gca().invert_xaxis()
            plt.xlabel("beta_ratio, lower = flatter")
            plt.ylabel(metric)
            plt.title(f"Beta sweep: {metric}")
            plt.tight_layout()
            plt.savefig(outdir / f"beta_sweep_{metric}.png", dpi=160)
            plt.close()


def parse_beta_values(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if 1.0 not in vals:
        vals = [1.0] + vals
    return sorted(set(vals), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--outdir", default="beta_sweep_results")
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--filler-sentences", type=int, default=14)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--metric-max-tokens", type=int, default=1536)
    parser.add_argument("--layer-beta", type=float, default=0.35, help="beta_ratio used for one-layer-at-a-time sweep")
    parser.add_argument("--beta-values", default="1.0,0.8,0.65,0.5,0.4,0.32,0.25,0.18,0.12,0.08")
    parser.add_argument("--beta-layers", default="auto", help="comma layers for beta sweep, 'auto' for strange candidates, or 'all'")
    parser.add_argument("--demo-layer-top-k", type=int, default=3, help="max demo layers selected from beta sweep layers by trippy_score")
    parser.add_argument("--no-refine-beta-sweep", action="store_true", help="disable automatic dense beta sweep around the largest coarse cliff")
    parser.add_argument("--beta-refine-steps", type=int, default=13, help="number of beta values in the dense refinement interval")
    parser.add_argument("--beta-refine-margin", type=float, default=0.03, help="extra beta margin added around the detected coarse cliff interval")
    parser.add_argument("--skip-layer-sweep", action="store_true")
    parser.add_argument("--skip-beta-sweep", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    print(f"Loading {args.model} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation="eager",
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    patch_llama_attention()

    prompts = make_needle_prompts(args.num_prompts, args.seed, args.filler_sentences)
    with open(outdir / "prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    all_rows = []
    all_entropy_rows = []

    # Baseline
    base_df, base_ent = evaluate_condition(
        model, tokenizer, prompts, 1.0, None, "baseline", args.max_new_tokens, args.metric_max_tokens
    )
    all_rows.append(base_df)
    all_entropy_rows.append(base_ent)
    baseline_summary = aggregate(base_df, ["condition", "beta_ratio", "layers"]).iloc[0].to_dict()

    n_layers = int(model.config.num_hidden_layers)
    layer_summary = pd.DataFrame()
    strange_layers: List[int] = []

    if not args.skip_layer_sweep:
        for layer in range(n_layers):
            df, ent = evaluate_condition(
                model, tokenizer, prompts, args.layer_beta, [layer], f"layer_{layer}",
                args.max_new_tokens, args.metric_max_tokens
            )
            all_rows.append(df)
            all_entropy_rows.append(ent)

        full_df = pd.concat(all_rows, ignore_index=True)
        layer_df = full_df[full_df["condition"].str.startswith("layer_")].copy()
        layer_df["layer"] = layer_df["condition"].str.extract(r"layer_(\d+)").astype(int)
        layer_summary = aggregate(layer_df, ["layer", "beta_ratio", "layers"])
        layer_summary = detect_strange_layers(layer_summary, baseline_summary)
        layer_summary.to_csv(outdir / "layer_summary.csv", index=False)
        strange_layers = layer_summary[layer_summary["strange_candidate"]]["layer"].astype(int).to_list()

    # Choose beta sweep layers.
    beta_layers: Optional[List[int]]
    if args.beta_layers == "all":
        beta_layers = None
    elif args.beta_layers == "auto":
        if strange_layers:
            beta_layers = strange_layers
        elif not layer_summary.empty:
            beta_layers = layer_summary.head(max(1, min(4, len(layer_summary))))["layer"].astype(int).to_list()
        else:
            beta_layers = None
    else:
        beta_layers = [int(x.strip()) for x in args.beta_layers.split(",") if x.strip()]

    beta_summary = pd.DataFrame()
    beta_star = {"beta_star": None, "phase_transition_like": False, "reason": "beta sweep skipped"}

    beta_values_measured: List[float] = []
    beta_refinement_added: List[float] = []

    if not args.skip_beta_sweep:
        beta_values = parse_beta_values(args.beta_values)
        measured = set()

        def run_beta_values(values: Sequence[float], prefix: str) -> None:
            nonlocal measured, beta_values_measured
            for b in values:
                b = float(b)
                key = round(b, 6)
                if key in measured:
                    continue
                measured.add(key)
                beta_values_measured.append(b)
                layers_for_run = beta_layers
                df, ent = evaluate_condition(
                    model, tokenizer, prompts, b, layers_for_run,
                    f"{prefix}_{b:g}", args.max_new_tokens, args.metric_max_tokens
                )
                all_rows.append(df)
                all_entropy_rows.append(ent)

        # First pass: broad/coarse grid.
        run_beta_values(beta_values, "beta")
        full_df = pd.concat(all_rows, ignore_index=True)
        beta_df = full_df[full_df["condition"].str.startswith(("beta_", "beta_refined_"))].copy()
        beta_summary = aggregate(beta_df, ["beta_ratio", "layers"])
        beta_star = detect_beta_star(beta_summary)

        # Second pass: automatically zoom around the steepest coarse cliff interval.
        if not args.no_refine_beta_sweep:
            beta_refinement_added = refine_beta_grid_from_cliff(
                beta_values_measured,
                beta_star,
                steps=args.beta_refine_steps,
                margin=args.beta_refine_margin,
            )
            if beta_refinement_added:
                print(f"Refining beta grid around cliff: {beta_refinement_added}")
                run_beta_values(beta_refinement_added, "beta_refined")
                full_df = pd.concat(all_rows, ignore_index=True)
                beta_df = full_df[full_df["condition"].str.startswith(("beta_", "beta_refined_"))].copy()
                beta_summary = aggregate(beta_df, ["beta_ratio", "layers"])
                beta_star = detect_beta_star(beta_summary)

        beta_summary = beta_summary.sort_values("beta_ratio", ascending=False)
        beta_summary.to_csv(outdir / "beta_summary.csv", index=False)

    # Pick demo layers after beta sweep layer selection. These are the highest-trippy-score
    # subset of the beta sweep layers, intended for the Act 2 demo.
    demo_layer_summary = pd.DataFrame()
    demo_layers: List[int] = []
    if not layer_summary.empty:
        demo_layer_summary = add_demo_layer_scores(layer_summary, baseline_summary, beta_layers)
        if not demo_layer_summary.empty:
            demo_layer_summary.to_csv(outdir / "demo_layer_summary.csv", index=False)
            demo_layers = demo_layer_summary.head(max(1, args.demo_layer_top_k))["layer"].astype(int).to_list()

    results_df = pd.concat(all_rows, ignore_index=True)
    entropy_df = pd.concat(all_entropy_rows, ignore_index=True)
    results_df.to_csv(outdir / "per_prompt_results.csv", index=False)
    entropy_df.to_csv(outdir / "per_layer_attention_entropy.csv", index=False)

    if layer_summary.empty and not args.skip_layer_sweep:
        pass
    make_plots(layer_summary, beta_summary, outdir)

    summary = {
        "model": args.model,
        "n_layers": n_layers,
        "num_prompts": args.num_prompts,
        "layer_beta_ratio": args.layer_beta,
        "baseline": baseline_summary,
        "strange_layers": strange_layers,
        "beta_sweep_layers": "all" if beta_layers is None else beta_layers,
        "demo_layers": demo_layers,
        "demo_layer_top_k": args.demo_layer_top_k,
        "demo_layer_selection_note": "demo_layers are a high-trippy-score subset of beta_sweep_layers; use beta just above beta_star for the demo",
        "beta_values_measured": sorted(beta_values_measured, reverse=True),
        "beta_refinement_added": beta_refinement_added,
        "beta_star_analysis": beta_star,
        "files": {
            "prompts": "prompts.jsonl",
            "per_prompt_results": "per_prompt_results.csv",
            "per_layer_attention_entropy": "per_layer_attention_entropy.csv",
            "layer_summary": "layer_summary.csv" if not layer_summary.empty else None,
            "demo_layer_summary": "demo_layer_summary.csv" if not demo_layer_summary.empty else None,
            "beta_summary": "beta_summary.csv" if not beta_summary.empty else None,
        },
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
