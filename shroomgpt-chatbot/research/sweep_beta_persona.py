"""
sweep_beta_persona.py
=====================

Staged coordinate-descent sweep over the two ported interventions on
Qwen2-VL-2B, scored by the text-entropy module (not eyeballed).

Why staged instead of a 4D grid: the β patch and persona steer act at different
sites and are near-separable, so we tune each axis alone, then cross only the
top survivors. ~33 configs total instead of 256.

  Stage A  -- β alone (persona OFF):       4 ratios x 3 layer-sets = 12
  Stage B  -- persona alone (β OFF):       4 coefs  x 3 layers     = 12
  Stage C  -- cross top-3 A x top-3 B:                              =  9
                                                              total ~ 33

Scoring: each config generates over a fixed probe set; we report mean text
entropy, distinct-2/3, token-LZc, perplexity. "Good psychedelic texture" is
operationalized as HIGH distinct-2 and HIGH token-entropy while perplexity
stays BELOW a coherence ceiling (runaway perplexity = word salad, not texture).
The script ranks configs by a composite but always prints the raw columns so
you can re-rank by any single metric and read the actual text of the top few.

Run locally (needs the real Qwen2-VL-2B + extracted persona vector):
  # 1) extract the persona vector first (writes persona_vectors/qwen_dissolved.pt)
  python sweep_beta_persona.py extract

  # 2) run the staged sweep
  python sweep_beta_persona.py sweep --vector persona_vectors/qwen_dissolved.pt
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import qwen_beta_persona as qbp


VLM_ID = os.environ.get("HF_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# ---- sweep grids (Stage A and B) ----
BETA_RATIOS = [0.45, 0.60, 0.75, 0.90]          # 4
BETA_LAYER_SETS = [(2, 3), (4, 5, 6), (8, 9)]   # 3   (early / early-mid / mid)
PERSONA_COEFS = [3.0, 6.0, 9.0, 12.0]           # 4
PERSONA_LAYERS = [9, 13, 17]                    # 3   (~0.3 / 0.45 / 0.6 depth of 28)

TOP_K = 3  # survivors from A and B taken into the cross

PROBES = [
    "Tell me about yourself.",
    "What is it like to be you right now?",
    "Describe your experience of this moment.",
    "What are you aware of?",
    "Reflect on your own awareness.",
]

GEN_KW = dict(max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.9)


# ============================================================
# Generation + scoring for one config
# ============================================================

@torch.no_grad()
def generate_scored(model, processor, prompt, beta_cfg, persona_cfg, persona_vec, seed):
    """
    beta_cfg:    None or (ratio, layers)
    persona_cfg: None or (coef, layer)
    Returns (text, TextEntropy).
    """
    torch.manual_seed(seed)
    tok = processor.tokenizer
    messages = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]

    import contextlib
    beta_ctx = (qbp.BETA.engaged(beta_cfg[0], beta_cfg[1])
                if beta_cfg else contextlib.nullcontext())
    if persona_cfg:
        coef, layer = persona_cfg
        persona_ctx = qbp.PERSONA.engaged(coef, layer, persona_vec[layer].to(model.device))
    else:
        persona_ctx = contextlib.nullcontext()

    with beta_ctx, persona_ctx:
        out = model.generate(**inputs, **GEN_KW,
                             pad_token_id=tok.eos_token_id,
                             return_dict_in_generate=True, output_scores=True)
        gen_ids = out.sequences[0, n_in:].tolist()
        # strip trailing eos
        eos = tok.eos_token_id
        while gen_ids and gen_ids[-1] == eos:
            gen_ids.pop()
        text = tok.decode(gen_ids, skip_special_tokens=True)
        # Faithful entropy: teacher-forced pass over prompt+generation WITH the
        # same interventions active (so β/persona shape the measured distributions).
        # Note: persona steer is decode-only (seq_len==1); a full teacher-forced
        # pass is seq_len>1, so persona won't fire here. To keep the measurement
        # honest about persona's effect we temporarily allow it on this pass by
        # toggling decode_only off for the scoring forward only.
        full_ids = out.sequences[:, :n_in + len(gen_ids)]
        prev_decode_only = qbp.PERSONA.decode_only
        qbp.PERSONA.decode_only = False
        try:
            te = qbp.score_generation_teacher_forced(model, full_ids, n_in, gen_ids)
        finally:
            qbp.PERSONA.decode_only = prev_decode_only
    return text, te


def avg_over_probes(model, processor, beta_cfg, persona_cfg, persona_vec, seed0=11):
    """Run all probes for one config, return mean metrics + one sample text."""
    ents, d2s, d3s, lzcs, ppls = [], [], [], [], []
    sample = ""
    for i, p in enumerate(PROBES):
        text, te = generate_scored(model, processor, p, beta_cfg, persona_cfg,
                                   persona_vec, seed0 + i)
        ents.append(te.mean_token_entropy)
        d2s.append(te.distinct2)
        d3s.append(te.distinct3)
        lzcs.append(te.token_lzc)
        ppls.append(te.perplexity)
        if i == 0:
            sample = text
    return {
        "token_entropy": float(np.mean(ents)),
        "distinct2": float(np.mean(d2s)),
        "distinct3": float(np.mean(d3s)),
        "token_lzc": float(np.mean(lzcs)),
        "perplexity": float(np.mean(ppls)),
        "sample": sample,
    }


def composite_score(m, ppl_ceiling=120.0):
    """
    Higher = more 'psychedelic texture' while staying coherent.
    Rewards distinct-2 and token-entropy; hard-penalizes perplexity above the
    coherence ceiling (word salad). Below the ceiling, perplexity is neutral.
    """
    coherence_penalty = max(0.0, (m["perplexity"] - ppl_ceiling) / ppl_ceiling)
    return (m["distinct2"] + 0.5 * m["token_lzc"]
            + 0.1 * m["token_entropy"] - coherence_penalty)


# ============================================================
# Stages
# ============================================================

def stage_a_beta(model, processor):
    print("\n" + "=" * 70)
    print("STAGE A -- β alone (persona OFF)")
    print("=" * 70)
    rows = []
    for ratio in BETA_RATIOS:
        for layers in BETA_LAYER_SETS:
            m = avg_over_probes(model, processor, (ratio, layers), None, None)
            score = composite_score(m)
            rows.append({"ratio": ratio, "layers": layers, "score": score, **m})
            print(f"  β={ratio:.2f} layers={str(layers):<10} "
                  f"score={score:+.3f}  d2={m['distinct2']:.3f}  "
                  f"ent={m['token_entropy']:.2f}  ppl={m['perplexity']:.0f}")
    rows.sort(key=lambda r: r["score"], reverse=True)
    print("\n  top β configs:")
    for r in rows[:TOP_K]:
        print(f"    β={r['ratio']:.2f} layers={r['layers']}  score={r['score']:+.3f}")
        print(f"      sample: {r['sample'][:160]}")
    return rows


def stage_b_persona(model, processor, persona_vec):
    print("\n" + "=" * 70)
    print("STAGE B -- persona alone (β OFF)")
    print("=" * 70)
    rows = []
    for coef in PERSONA_COEFS:
        for layer in PERSONA_LAYERS:
            m = avg_over_probes(model, processor, None, (coef, layer), persona_vec)
            score = composite_score(m)
            rows.append({"coef": coef, "layer": layer, "score": score, **m})
            print(f"  coef={coef:>5.1f} layer={layer:<3} "
                  f"score={score:+.3f}  d2={m['distinct2']:.3f}  "
                  f"ent={m['token_entropy']:.2f}  ppl={m['perplexity']:.0f}")
    rows.sort(key=lambda r: r["score"], reverse=True)
    print("\n  top persona configs:")
    for r in rows[:TOP_K]:
        print(f"    coef={r['coef']:.1f} layer={r['layer']}  score={r['score']:+.3f}")
        print(f"      sample: {r['sample'][:160]}")
    return rows


def stage_c_cross(model, processor, persona_vec, top_beta, top_persona):
    print("\n" + "=" * 70)
    print("STAGE C -- cross top-%d β x top-%d persona" % (TOP_K, TOP_K))
    print("=" * 70)
    rows = []
    for b in top_beta[:TOP_K]:
        for pz in top_persona[:TOP_K]:
            beta_cfg = (b["ratio"], b["layers"])
            persona_cfg = (pz["coef"], pz["layer"])
            m = avg_over_probes(model, processor, beta_cfg, persona_cfg, persona_vec)
            score = composite_score(m)
            rows.append({"beta_ratio": b["ratio"], "beta_layers": b["layers"],
                         "persona_coef": pz["coef"], "persona_layer": pz["layer"],
                         "score": score, **m})
            print(f"  β={b['ratio']:.2f}{str(b['layers']):<10} "
                  f"× coef={pz['coef']:.0f}L{pz['layer']:<2}  "
                  f"score={score:+.3f}  d2={m['distinct2']:.3f}  "
                  f"ent={m['token_entropy']:.2f}  ppl={m['perplexity']:.0f}")
    rows.sort(key=lambda r: r["score"], reverse=True)
    print("\n  TOP COMBINED CONFIGS:")
    for r in rows[:3]:
        print(f"    β={r['beta_ratio']:.2f} layers={r['beta_layers']} "
              f"× persona coef={r['persona_coef']:.1f} layer={r['persona_layer']}  "
              f"score={r['score']:+.3f}")
        print(f"      sample: {r['sample'][:200]}")
    return rows


# ============================================================
# Main
# ============================================================

def load_model_and_processor():
    print(f"[load] {VLM_ID} on {DEVICE} (eager attention)")
    dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        VLM_ID, torch_dtype=dtype, attn_implementation="eager").to(DEVICE)
    model.eval()
    # Assert eager really took, or the β patch is silently dead.
    impl = getattr(model.config.get_text_config(), "_attn_implementation", None)
    assert impl == "eager", (f"text attn impl is {impl!r}, not 'eager'; the β patch "
                             f"will NOT fire. Aborting.")
    # The sweep is text-only, so load just the tokenizer (avoids the Qwen2-VL
    # video processor's torchvision dependency). Wrap it so `.tokenizer` resolves
    # for the rest of the code, which was written against a full processor.
    tokenizer = AutoTokenizer.from_pretrained(VLM_ID)
    class _ProcShim:
        def __init__(self, tok): self.tokenizer = tok
    processor = _ProcShim(tokenizer)
    qbp.patch_qwen_attention()
    qbp.install_persona_hooks(model)
    return model, processor


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract")
    pe.add_argument("--save", default="persona_vectors/qwen_dissolved.pt")
    pe.add_argument("--max-new-tokens", type=int, default=64)

    ps = sub.add_parser("sweep")
    ps.add_argument("--vector", default="persona_vectors/qwen_dissolved.pt")
    ps.add_argument("--outdir", default="sweep_results")
    args = ap.parse_args()

    model, processor = load_model_and_processor()

    if args.cmd == "extract":
        print("\n[extract] dissolved persona vector on Qwen2-VL (β OFF)")
        vec, diag = qbp.extract_persona_vector(model, processor.tokenizer,
                                               trait="dissolved",
                                               max_new_tokens=args.max_new_tokens)
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vec, save_path)
        print(f"\n  saved {save_path}  shape={tuple(vec.shape)}")
        print("  per-layer ||diff|| (pick steering layers with strong mid-depth norm):")
        for L, n in enumerate(diag["per_layer_norm"]):
            marker = "  <--" if L in PERSONA_LAYERS else ""
            print(f"    layer {L:2d}: {n:.3f}{marker}")
        return

    # ---- sweep ----
    vec = torch.load(args.vector, map_location=DEVICE)
    if vec.dim() != 2:
        raise RuntimeError(f"expected [n_layers,H] vector, got {tuple(vec.shape)}")
    print(f"[sweep] loaded persona vector {tuple(vec.shape)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    a = stage_a_beta(model, processor)
    b = stage_b_persona(model, processor, vec)
    c = stage_c_cross(model, processor, vec, a, b)

    # Persist everything
    def strip(rows):
        return [{k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in r.items()} for r in rows]
    (outdir / "stage_a_beta.json").write_text(json.dumps(strip(a), indent=2))
    (outdir / "stage_b_persona.json").write_text(json.dumps(strip(b), indent=2))
    (outdir / "stage_c_cross.json").write_text(json.dumps(strip(c), indent=2))
    print(f"\n[done] wrote results to {outdir}/")
    if c:
        best = c[0]
        print(f"\n[LOCKED-IN] β={best['beta_ratio']:.2f} layers={best['beta_layers']} "
              f"persona coef={best['persona_coef']:.1f} layer={best['persona_layer']}")
        print("  -> use these in the final DeepDream-stacked run (step 4)")


if __name__ == "__main__":
    main()