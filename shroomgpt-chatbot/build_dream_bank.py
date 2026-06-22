"""
build_dream_bank.py
===================

Pre-compute a BANK of dreamed visual representations for residual-stream
injection into the β+persona stack.

For each (source image, dream layer, ascent strength, seed) we:
  1. DeepDream on Qwen2-VL's own vision tower (encoder-matched, from v3).
  2. Run the dreamed pixel_values through the vision tower + merger to get the
     MERGED VISUAL TOKENS -- the exact representation Qwen2-VL projects into the
     LLM residual stream.
  3. Reduce to an injectable [llm_hidden] vector (mean over visual tokens), the
     direct analogue of the persona vector.

We save:
  - dream_bank.pt : dict with
        "vectors":  [N, llm_hidden] float tensor (mean merged-token per dream)
        "meta":     list of N dicts (source, layer, iters, lr, eps, seed, drift)
  - optionally the dreamed images as PNGs for inspection.

The injection mechanism (separate file) samples from "vectors" -- one per
generation, or a fresh one per decode step for visual flux.

Run:
  python build_dream_bank.py --images img1.jpg img2.jpg --out dream_bank.pt
"""

import os
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

# Reuse the v3 dream-on-Qwen machinery if present; otherwise inline a copy.
try:
    from transformer_deepdream_v3 import dream_on_qwen_visual, _get_visual_tower
    HAVE_V3 = True
except Exception:
    HAVE_V3 = False


VLM_ID = os.environ.get("HF_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

QWEN_DREAM_LAYERS = [20, 26, 30]   # vary the dream layer -> different feature families
ASCENT = [
    {"iters": 20, "lr": 0.03, "eps": 0.15},   # gentle
    {"iters": 40, "lr": 0.05, "eps": 0.40},   # medium
]
SEEDS = [0, 1]
MAX_IMAGE_SIZE = 448


def _get_visual_tower_local(vlm):
    if HAVE_V3:
        return _get_visual_tower(vlm)
    if hasattr(vlm, "visual"):
        return vlm.visual
    if hasattr(vlm, "model") and hasattr(vlm.model, "visual"):
        return vlm.model.visual
    raise RuntimeError("can't find visual tower")


def _dream_local(vlm, pixel_values, grid_thw, layer_idx, n_iters, lr, eps):
    """Fallback dream impl if v3 import failed. Same math as v3."""
    if HAVE_V3:
        return dream_on_qwen_visual(vlm, pixel_values, grid_thw, layer_idx,
                                    n_iters, lr, eps, log=False)
    visual = _get_visual_tower_local(vlm)
    orig_dtype = pixel_values.dtype
    base = pixel_values.detach().clone().to(torch.float32)
    pv = base.clone().requires_grad_(True)
    acts = {}
    class _Exit(Exception): pass
    def hook(m, i, o):
        acts["h"] = o[0] if isinstance(o, tuple) else o
        raise _Exit()
    handle = visual.blocks[layer_idx].register_forward_hook(hook)
    try:
        for _ in range(n_iters):
            try:
                _ = visual(pv.to(orig_dtype), grid_thw=grid_thw)
            except _Exit:
                pass
            h = acts["h"].to(torch.float32)
            loss = 0.5 * (h ** 2).sum()
            grad, = torch.autograd.grad(loss, pv)
            grad = grad / (grad.std() + 1e-8)
            with torch.no_grad():
                pv.add_(lr * grad)
                pv.data = base + (pv.data - base).clamp(-eps, eps)
            acts.clear()
    finally:
        handle.remove()
    return pv.detach().to(orig_dtype)


@torch.no_grad()
def merged_visual_tokens(vlm, pixel_values, grid_thw):
    """Run the (dreamed) pixel_values through the full vision tower + merger and
    return the merged visual tokens [n_visual_tokens, llm_hidden]."""
    visual = _get_visual_tower_local(vlm)
    out = visual(pixel_values, grid_thw=grid_thw)
    merged = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    return merged.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="seed image paths")
    ap.add_argument("--out", default="dream_bank.pt")
    ap.add_argument("--save-images-dir", default=None,
                    help="if set, save dreamed PNGs here for inspection")
    ap.add_argument("--max-image-size", type=int, default=MAX_IMAGE_SIZE)
    args = ap.parse_args()

    print(f"[load] {VLM_ID} on {DEVICE}")
    dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
    vlm = AutoModelForImageTextToText.from_pretrained(
        VLM_ID, torch_dtype=dtype, attn_implementation="eager").to(DEVICE)
    vlm.eval()
    processor = AutoProcessor.from_pretrained(VLM_ID)

    visual = _get_visual_tower_local(vlm)
    n_vision_blocks = len(visual.blocks)
    dream_layers = [L for L in QWEN_DREAM_LAYERS if L < n_vision_blocks]
    print(f"[bank] vision blocks={n_vision_blocks}, dreaming on {dream_layers}")

    if args.save_images_dir:
        Path(args.save_images_dir).mkdir(parents=True, exist_ok=True)

    vectors = []
    meta = []
    total = len(args.images) * len(dream_layers) * len(ASCENT) * len(SEEDS)
    done = 0
    t_start = time.time()

    for img_path in args.images:
        seed_pil = Image.open(img_path).convert("RGB")
        if max(seed_pil.size) > args.max_image_size:
            seed_pil.thumbnail((args.max_image_size, args.max_image_size), Image.BICUBIC)

        conv = [{"role": "user", "content": [
            {"type": "image", "image": seed_pil},
            {"type": "text", "text": "x"},
        ]}]
        chat = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[seed_pil], padding=True,
                           return_tensors="pt").to(vlm.device)
        base_pv = inputs["pixel_values"].detach().clone()
        grid_thw = inputs["image_grid_thw"]

        # Baseline (undreamed) vector too -- a useful control in the bank.
        base_merged = merged_visual_tokens(vlm, base_pv, grid_thw).mean(0)
        vectors.append(base_merged.cpu())
        meta.append({"source": os.path.basename(img_path), "layer": None,
                     "iters": 0, "lr": 0.0, "eps": 0.0, "seed": None, "drift": 0.0,
                     "kind": "baseline"})

        for layer in dream_layers:
            for asc in ASCENT:
                for seed in SEEDS:
                    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
                    dreamed_pv = _dream_local(vlm, base_pv, grid_thw, layer,
                                              asc["iters"], asc["lr"], asc["eps"])
                    drift = ((dreamed_pv.float() - base_pv.float()).norm()
                             / base_pv.float().norm()).item()
                    merged = merged_visual_tokens(vlm, dreamed_pv, grid_thw).mean(0)
                    vectors.append(merged.cpu())
                    meta.append({"source": os.path.basename(img_path), "layer": layer,
                                 "iters": asc["iters"], "lr": asc["lr"], "eps": asc["eps"],
                                 "seed": seed, "drift": round(drift, 4), "kind": "dream"})
                    done += 1
                    elapsed = time.time() - t_start
                    eta = elapsed / done * (total - done)
                    print(f"  [{done}/{total}] {os.path.basename(img_path)} "
                          f"L{layer} it{asc['iters']} seed{seed} drift={drift:.3f} "
                          f"(eta {eta:.0f}s)")

                    if args.save_images_dir:
                        # de-patchify is nontrivial; skip image save unless needed.
                        pass

    V = torch.stack(vectors)  # [N, llm_hidden]
    # Normalize each vector to unit norm? Keep RAW for now (like persona), but
    # also store the per-vector norm so the injector can normalize if desired.
    norms = V.norm(dim=-1)
    torch.save({"vectors": V, "meta": meta,
                "norms": norms, "vlm_id": VLM_ID}, args.out)
    print(f"\n[done] saved {args.out}")
    print(f"  bank size: {V.shape[0]} vectors of dim {V.shape[1]}")
    print(f"  dream vectors: {sum(1 for m in meta if m['kind']=='dream')}, "
          f"baseline: {sum(1 for m in meta if m['kind']=='baseline')}")
    print(f"  vector norm range: [{norms.min():.2f}, {norms.max():.2f}]  "
          f"mean {norms.mean():.2f}")
    print(f"  drift range: [{min(m['drift'] for m in meta):.3f}, "
          f"{max(m['drift'] for m in meta):.3f}]")


if __name__ == "__main__":
    main()
