#!/usr/bin/env python3
"""
Visualize the main results from a summary.json produced by
beta_psychedelic_sweep_llama.py or all_layer_beta_test.py.

Reads ONLY summary.json. That file stores aggregated scalars (baseline metrics,
the interesting-band verdict, the beta vs temperature control profiles, and the
intervened-layer entropy gap) -- NOT the per-beta curves, which live in the
CSVs. So this renders the comparisons summary.json actually supports:

  1. baseline_metrics.png      - the baseline metric fingerprint
  2. beta_vs_temp_profiles.png - grouped bars: how far each arm pushed each
                                 metric (the core "are the two knobs different"
                                 picture)
  3. control_verdict.png       - the two discriminator gaps + distinguishable
                                 flag, and the band verdict, as an at-a-glance
                                 panel

Usage
-----
    python visualize_summary.py /psyche_sweep_results/summary.json
    python visualize_summary.py path/to/summary.json --outdir figs
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _get(d, *keys, default=None):
    """Safe nested lookup."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def _finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def plot_baseline(summary, outdir: Path):
    base = summary.get("baseline", {}) or {}
    metrics = ["local_coherence", "associative_drift", "self_diversity",
               "global_coherence", "attention_entropy"]
    labels, vals = [], []
    for m in metrics:
        if _finite(base.get(m)):
            labels.append(m.replace("_", "\n"))
            vals.append(float(base[m]))
    if not vals:
        print("[skip] no baseline metrics in summary.json")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, vals, color="#4C72B0")
    ax.set_title("Baseline metric fingerprint (no intervention)")
    ax.set_ylabel("value")
    ax.set_ylim(0, max(1.0, max(vals) * 1.15))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = outdir / "baseline_metrics.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"[wrote] {p}")


def plot_profiles(summary, outdir: Path):
    """Grouped bars comparing how far the beta arm and the temperature arm
    pushed each metric. These four scalars are exactly what the control block
    stores."""
    ctrl = summary.get("beta_vs_temperature_control", {}) or {}
    bp = ctrl.get("beta_profile") or {}
    tp = ctrl.get("temperature_profile") or {}
    if not bp or not tp:
        print("[skip] no beta/temperature profiles in summary.json")
        return

    keys = ["max_drift", "min_local_coherence", "min_global_coherence", "max_attention_entropy"]
    nice = ["max\ndrift", "min local\ncoherence", "min global\ncoherence", "max attention\nentropy"]
    bvals = [float(bp.get(k, float("nan"))) for k in keys]
    tvals = [float(tp.get(k, float("nan"))) for k in keys]

    x = range(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar([i - w / 2 for i in x], bvals, width=w, label="beta arm", color="#C44E52")
    b2 = ax.bar([i + w / 2 for i in x], tvals, width=w, label="temperature arm", color="#55A868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(nice)
    ax.set_ylabel("value at the extreme of each arm")
    ax.set_title("Beta vs temperature: how far each knob pushed each metric")
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if _finite(h):
                ax.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = outdir / "beta_vs_temp_profiles.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"[wrote] {p}")


def plot_verdict(summary, outdir: Path):
    """At-a-glance panel: the two discriminator gaps, the distinguishable flag,
    and the band verdict. These are the headline conclusions."""
    ctrl = summary.get("beta_vs_temperature_control", {}) or {}
    band = summary.get("interesting_band", {}) or {}

    ent_gap = ctrl.get("attention_entropy_gap_beta_minus_temp")
    glob_gap = ctrl.get("global_coherence_drop_gap_beta_minus_temp")
    # all_layer_beta_test.py also stores this one:
    il_gap = summary.get("intervened_layer_entropy_gap_beta_minus_temp")

    gap_labels, gap_vals = [], []
    for lab, val in [("attention\nentropy gap", ent_gap),
                     ("global coherence\ndrop gap", glob_gap),
                     ("intervened-layer\nentropy gap", il_gap)]:
        if _finite(val):
            gap_labels.append(lab)
            gap_vals.append(float(val))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: gap bars with a zero line. Positive = beta moved it more than temp.
    if gap_vals:
        colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in gap_vals]
        bars = ax1.bar(gap_labels, gap_vals, color=colors)
        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.set_title("Discriminator gaps (beta minus temperature)\npositive = beta-specific effect", pad=14)
        ax1.margins(y=0.18)
        ax1.set_ylabel("gap")
        for b, v in zip(bars, gap_vals):
            ax1.text(b.get_x() + b.get_width() / 2,
                     v + (0.005 if v >= 0 else -0.005), f"{v:+.3f}",
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    else:
        ax1.axis("off")
        ax1.text(0.5, 0.5, "no gap scalars in summary.json", ha="center", va="center")

    # Right: text verdict card.
    ax2.axis("off")
    distinguishable = ctrl.get("distinguishable")
    found_band = band.get("found_band")
    sweet = band.get("sweet_spot_beta")
    collapse = band.get("collapse_beta")

    def yn(v):
        return "yes" if v is True else ("no" if v is False else "n/a")

    lines = [
        ("Knobs distinguishable?", yn(distinguishable),
         "#55A868" if distinguishable else "#C44E52"),
        ("Altered-but-coherent band?", yn(found_band),
         "#55A868" if found_band else "#C44E52"),
        ("Sweet-spot beta", f"{sweet:.3g}" if _finite(sweet) else "none", "#333333"),
        ("Collapse beta", f"{collapse:.3g}" if _finite(collapse) else "n/a", "#333333"),
    ]
    ax2.set_title("Verdict", fontsize=12, loc="left")
    y = 0.86
    for label, val, color in lines:
        ax2.text(0.02, y, label, fontsize=11, va="center")
        ax2.text(0.98, y, val, fontsize=12, va="center", ha="right",
                 color=color, fontweight="bold")
        y -= 0.22

    fig.suptitle(
        f"{summary.get('model', 'model')}  |  "
        f"{summary.get('intervention', 'beta sweep')}",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = outdir / "control_verdict.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"[wrote] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="path to summary.json")
    ap.add_argument("--outdir", default=None,
                    help="where to write PNGs (default: alongside summary.json)")
    args = ap.parse_args()

    spath = Path(args.summary)
    with open(spath) as f:
        summary = json.load(f)
    outdir = Path(args.outdir) if args.outdir else spath.parent
    outdir.mkdir(parents=True, exist_ok=True)

    plot_baseline(summary, outdir)
    plot_profiles(summary, outdir)
    plot_verdict(summary, outdir)
    print(f"\nFigures written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
