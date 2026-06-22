"""
build_dream_bank_direct.py
==========================

Image-free dream bank. Instead of cherry-picking images and dreaming on pixels,
we dream directly in Qwen2-VL's PATCH-EMBEDDING space, optimizing a mid vision
block's activation along Qwen's OWN feature directions (found by PCA over real
patch statistics), then read the resulting MERGED VISUAL TOKEN to inject.

Why this design (lessons from v2/v3):
  - Optimize a MID block, not the output: mid blocks carry semantic features
    (textures/objects), analogous to Inception's mixed4/mixed5. Maximizing the
    final merged token just finds high-norm noise (the v3 null result).
  - Optimize along PCA directions, not raw norm: ascending plain activation norm
    drifts toward statistical artifacts (the rare-token-attractor problem in
    visual-token form). PCA directions are the axes Qwen actually uses on real
    inputs, so ascending them produces structured "surreal feature" content.
  - No pixels, no JPEG, no image selection: the dream substrate is a random
    patch-embedding seed, so the bank samples Qwen's feature space directly with
    zero human image-taste bias. Unlimited supply.

Pipeline per dream:
  1. Seed: random patch embeddings E0 (shape [n_patches, vision_hidden]), or
     start from the mean of a few random-noise forwards (the "centroid").
  2. Ascend E to maximize the projection of mid-block-L activations onto a chosen
     PCA direction (or a random mix of top-K directions for variety).
  3. Forward the dreamed E through the rest of the tower + merger -> merged tokens.
  4. Store mean merged token [llm_hidden] as the injectable dream vector.

PCA basis:
  We collect mid-block activations over many RANDOM patch-embedding seeds (no
  images needed -- the point is to characterize the block's response geometry),
  then PCA. Top components = the block's dominant feature axes.

Run:
  python build_dream_bank_direct.py --n-dreams 200 --out dream_bank_direct.pt
"""

import os
import time
import argparse

import numpy as np
import torch
from transformers import AutoModelForImageTextToText

VLM_ID = os.environ.get("HF_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

MID_BLOCK = int(os.environ.get("MID_BLOCK", "8"))    # of 32; earlier = much cheaper backprop.
                                                      # block 8 is the texture/early-feature
                                                      # regime, still rich, ~2x cheaper than 16.
N_PCA = 32                                            # PCA directions to keep
PCA_PROBE_SEEDS = 48                                  # random seeds to characterize block geometry
GRID = (1, 8, 8)                                      # synthetic patch grid (T,H,W) -> 64 patches
ASCENT_ITERS = 20
ASCENT_LR = 0.08


def get_visual(vlm):
    if hasattr(vlm, "visual"):
        return vlm.visual
    if hasattr(vlm, "model") and hasattr(vlm.model, "visual"):
        return vlm.model.visual
    raise RuntimeError("no visual tower")


def patch_dim(vlm):
    """The flattened patch input dimension Qwen's patch_embed expects."""
    vc = vlm.config.vision_config
    ic = getattr(vc, "in_chans", 3)
    tp = getattr(vc, "temporal_patch_size", 2)
    ps = getattr(vc, "patch_size", 14)
    return ic * tp * ps * ps


@torch.no_grad()
def block_activation(visual, pixel_values, grid_thw, block_idx):
    """Run visual() and capture block_idx output via hook. Returns [n_patches, hidden]."""
    acts = {}
    class _Exit(Exception): pass
    def hook(m, i, o):
        acts["h"] = (o[0] if isinstance(o, tuple) else o)
        raise _Exit()
    handle = visual.blocks[block_idx].register_forward_hook(hook)
    try:
        try:
            visual(pixel_values, grid_thw=grid_thw)
        except _Exit:
            pass
    finally:
        handle.remove()
    return acts["h"]


@torch.no_grad()
def build_pca_basis(visual, pdim, grid_thw, block_idx, n_seeds, device, dtype):
    """Characterize mid-block response geometry over random patch seeds, PCA it."""
    n_patches = int(np.prod(GRID))
    feats = []
    for s in range(n_seeds):
        torch.manual_seed(1000 + s)
        pv = torch.randn(n_patches, pdim, device=device, dtype=dtype) * 0.5
        h = block_activation(visual, pv, grid_thw, block_idx).float()  # [n_patches, hidden]
        feats.append(h.mean(0))  # mean over patches -> [hidden]
    X = torch.stack(feats)  # [n_seeds, hidden]
    Xc = X - X.mean(0, keepdim=True)
    # SVD for PCA directions
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    comps = Vh[:N_PCA]  # [N_PCA, hidden]
    return comps, X.mean(0)


def dream_one(visual, pdim, grid_thw, block_idx, pca_dir, device, dtype,
              iters=ASCENT_ITERS, lr=ASCENT_LR, seed=0):
    """
    Ascend random patch embeddings to maximize projection of mid-block activation
    onto pca_dir. Returns dreamed pixel_values [n_patches, pdim].
    """
    n_patches = int(np.prod(GRID))
    torch.manual_seed(seed)
    base = (torch.randn(n_patches, pdim, device=device, dtype=torch.float32) * 0.5)
    pv = base.clone().requires_grad_(True)

    acts = {}
    class _Exit(Exception): pass
    def hook(m, i, o):
        acts["h"] = (o[0] if isinstance(o, tuple) else o)
        raise _Exit()
    handle = visual.blocks[block_idx].register_forward_hook(hook)
    pca_dir = pca_dir.to(device, torch.float32)

    has_mps = (device == "mps") or (str(device) == "mps")
    try:
        for _it in range(iters):
            try:
                visual(pv.to(dtype), grid_thw=grid_thw)
            except _Exit:
                pass
            h = acts["h"].to(torch.float32).mean(0)   # [hidden]
            # maximize projection onto the PCA direction
            loss = (h * pca_dir).sum()
            grad, = torch.autograd.grad(loss, pv)
            grad = grad / (grad.std() + 1e-8)
            with torch.no_grad():
                pv.add_(lr * grad)
            acts.clear()
            del h, loss, grad
            # MPS accumulates graph memory across iterations without this, which
            # makes later dreams progressively slower -- the real cause of the
            # "30 min for 10 dreams" pathology.
            if has_mps:
                torch.mps.empty_cache()
    finally:
        handle.remove()
    return pv.detach()


@torch.no_grad()
def merged_token(vlm, pixel_values, grid_thw):
    """
    The TRUE injectable representation: run through the model's own
    get_image_features, which applies the vision tower + merger and returns
    visual tokens in LLM-hidden space (the exact vectors spliced into the
    residual stream). Returns the mean token [llm_hidden].

    NOTE: visual(...) alone returns PRE-merger block output (vision-hidden dim),
    which is the wrong space to inject -- that was the dim-1280 bug.
    """
    feats = vlm.get_image_features(pixel_values=pixel_values, image_grid_thw=grid_thw)
    # Qwen2-VL's get_image_features returns a BaseModelOutputWithPooling whose
    # POOLER_OUTPUT holds the post-merger LLM-space tokens (split per image),
    # while last_hidden_state is the PRE-merger block output (wrong dim). Read
    # pooler_output.
    if hasattr(feats, "pooler_output"):
        pooled = feats.pooler_output
        if isinstance(pooled, (list, tuple)):
            pooled = pooled[0]
        return pooled.float().mean(0)  # [llm_hidden]
    # Fallbacks for other versions / return types.
    if isinstance(feats, (list, tuple)):
        feats = feats[0]
    if hasattr(feats, "last_hidden_state"):
        feats = feats.last_hidden_state
    return feats.float().mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dreams", type=int, default=64)
    ap.add_argument("--mid-block", type=int, default=MID_BLOCK)
    ap.add_argument("--out", default="dream_bank_direct.pt")
    ap.add_argument("--iters", type=int, default=ASCENT_ITERS)
    ap.add_argument("--lr", type=float, default=ASCENT_LR)
    ap.add_argument("--dream-on-cpu", action="store_true",
                    help="run the ascent loop on CPU. MPS autograd through the "
                         "vision tower has a memory-accumulation pathology that "
                         "makes later dreams progressively slower; CPU is slower "
                         "per-op but predictable and often faster end-to-end here.")
    args = ap.parse_args()

    print(f"[load] {VLM_ID} on {DEVICE}")
    dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
    vlm = AutoModelForImageTextToText.from_pretrained(
        VLM_ID, torch_dtype=dtype, attn_implementation="eager").to(DEVICE)
    vlm.eval()
    visual = get_visual(vlm)

    # The ascent loop can run on a different device than the model was loaded on.
    dream_device = "cpu" if args.dream_on_cpu else DEVICE
    dream_dtype = torch.float32 if dream_device == "cpu" else dtype
    if args.dream_on_cpu and DEVICE != "cpu":
        print(f"[dream] moving vision tower to CPU for the ascent loop "
              f"(fp32); model stays usable for readout")
        visual = visual.to("cpu", torch.float32)

    n_blocks = len(visual.blocks)
    block_idx = min(args.mid_block, n_blocks - 2)
    pdim = patch_dim(vlm)
    grid_thw = torch.tensor([list(GRID)], device=dream_device)

    print(f"[bank] vision blocks={n_blocks}, dreaming at block {block_idx}, "
          f"patch_dim={pdim}, n_patches={int(np.prod(GRID))}, device={dream_device}")

    print(f"[pca] characterizing block-{block_idx} geometry over {PCA_PROBE_SEEDS} seeds...")
    pca_comps, centroid = build_pca_basis(visual, pdim, grid_thw, block_idx,
                                          PCA_PROBE_SEEDS, dream_device, dream_dtype)
    print(f"[pca] kept {pca_comps.shape[0]} components of dim {pca_comps.shape[1]}")

    # Baseline merged token (un-dreamed random seed) for differential vectors.
    # The injectable bank stores DREAM DIRECTIONS (dreamed - baseline), not
    # absolute activations -- this removes the common-mode blowup that otherwise
    # makes every dream point the same way (the 0.998-cosine collapse), exactly
    # like the persona vector is pos-minus-neg, not absolute.
    # Readout uses get_image_features, which routes through self.model.visual --
    # the SAME module we may have moved to CPU for dreaming. So the readout device
    # must match where the vision tower currently lives, or conv3d sees a
    # CPU-weight / MPS-input mismatch.
    read_device = dream_device
    read_dtype = dream_dtype
    torch.manual_seed(99999)
    base_seed_pv = (torch.randn(int(np.prod(GRID)), pdim,
                                device=read_device, dtype=read_dtype) * 0.5)
    base_grid = torch.tensor([list(GRID)], device=read_device)
    baseline_vec = merged_token(vlm, base_seed_pv, base_grid).cpu()
    print(f"[bank] baseline merged-token norm: {baseline_vec.norm():.2f} "
          f"(dim {baseline_vec.shape[0]})")

    vectors, meta = [], []
    t0 = time.time()
    for d in range(args.n_dreams):
        td = time.time()
        # pick a PCA direction (cycle through top components, random sign mix for variety)
        comp_idx = d % pca_comps.shape[0]
        direction = pca_comps[comp_idx].clone()
        # occasional random mix of top-K directions for richer dreams
        if d % 3 == 0:
            k = 4
            w = torch.randn(k, device=direction.device)
            direction = (w[:, None] * pca_comps[:k]).sum(0)
            direction = direction / (direction.norm() + 1e-8)

        dreamed_pv = dream_one(visual, pdim, grid_thw, block_idx, direction,
                               dream_device, dream_dtype, iters=args.iters,
                               lr=args.lr, seed=d)
        # Readout on the full model's device (get_image_features needs the merger).
        dreamed_for_read = dreamed_pv.to(read_device, read_dtype)
        vec = merged_token(vlm, dreamed_for_read, base_grid).cpu()
        # Differential: dream direction relative to baseline.
        vec = vec - baseline_vec
        # Unit-normalize so the injection coef means the same thing for every dream.
        vec = vec / (vec.norm() + 1e-8)
        vectors.append(vec)
        meta.append({"pca_dir": comp_idx if d % 3 != 0 else "mix4",
                     "seed": d, "block": block_idx})

        if (d + 1) % 5 == 0 or d == args.n_dreams - 1:
            el = time.time() - t0
            eta = el / (d + 1) * (args.n_dreams - d - 1)
            print(f"  [{d+1}/{args.n_dreams}] {time.time()-td:.1f}s/dream  eta {eta:.0f}s")

    V = torch.stack(vectors)
    norms = V.norm(dim=-1)  # all ~1.0 now (unit-normalized)
    torch.save({"vectors": V, "meta": meta, "norms": norms,
                "baseline": baseline_vec,
                "vlm_id": VLM_ID, "mid_block": block_idx,
                "method": "direct_patch_embedding_pca_ascent"}, args.out)
    print(f"\n[done] saved {args.out}")
    print(f"  bank: {V.shape[0]} dream vectors of dim {V.shape[1]}")
    print(f"  norm range [{norms.min():.2f}, {norms.max():.2f}] mean {norms.mean():.2f}")
    # Diversity check: mean pairwise cosine (lower = more diverse bank)
    Vn = torch.nn.functional.normalize(V, dim=-1)
    sims = Vn @ Vn.T
    off = sims[~torch.eye(len(V), dtype=bool)]
    print(f"  mean pairwise cosine: {off.mean():.3f} (lower = more diverse)")


if __name__ == "__main__":
    main()