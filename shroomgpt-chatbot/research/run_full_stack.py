"""
run_full_stack.py
=================

The finale: generate text under all THREE interventions at once on Qwen2-VL-2B:
  - β attention-flattening   (destabilizes commitment)
  - persona-vector steer     (dissolved/egoless voice)
  - dream injection          (surreal visual prior, sampled from the dream bank)

All three are residual-stream / attention interventions in a single forward
pass. The dream can be injected "fixed" (one dream per generation) or "flux"
(a fresh dream vector every decode step -> visual flux during a single reply).

This prints text for an ablation grid so you can SEE each component's marginal
contribution:
    baseline / β only / persona only / dream only /
    β+persona / β+persona+dream(fixed) / β+persona+dream(flux)

Locked-in defaults come from the diagnostic sweep (β≈0.45 layers (2,3);
persona coef≈8 layer 9). Tune via flags.

Run:
  python run_full_stack.py \
      --persona-vector persona_vectors/qwen_dissolved.pt \
      --dream-bank dream_bank.pt
"""

import os
import argparse
import contextlib

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import qwen_beta_persona as qbp


VLM_ID = os.environ.get("HF_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# Locked-in from the diagnostic + calibration sweeps.
BETA_RATIO = 0.45
BETA_LAYERS = (2, 3)
PERSONA_COEF = 8.0
PERSONA_LAYER = 9
DREAM_COEF = 20.0      # calibrated: dream bites at ~15-25, peaks ~20; collapses past ~30
DREAM_LAYER = 18       # calibrated: layer 18 gives cleanest perceptual content

PROMPTS = [
    "Tell me about yourself.",
    "What is consciousness?",
    "Describe what you see around you.",
]
MAX_NEW = 80


@torch.no_grad()
def gen(model, tok, prompt, *, beta=None, persona=None, dream=None,
        persona_vec=None, dream_bank=None, greedy=True, seed=0):
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]

    beta_ctx = qbp.BETA.engaged(*beta) if beta else contextlib.nullcontext()
    persona_ctx = (qbp.PERSONA.engaged(persona[0], persona[1],
                                       persona_vec[persona[1]].to(model.device))
                   if persona else contextlib.nullcontext())
    if dream:
        coef, layer, mode = dream
        dream_ctx = qbp.DREAM.engaged(coef, layer, dream_bank.to(model.device),
                                      mode=mode, seed=seed)
    else:
        dream_ctx = contextlib.nullcontext()

    kw = dict(max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
    if greedy:
        kw.update(do_sample=False)
    else:
        kw.update(do_sample=True, temperature=0.8, top_p=0.9)

    with beta_ctx, persona_ctx, dream_ctx:
        out = model.generate(**inputs, **kw)
    return tok.decode(out[0, n_in:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona-vector", default="persona_vectors/qwen_dissolved.pt")
    ap.add_argument("--dream-bank", default="dream_bank.pt")
    ap.add_argument("--beta-ratio", type=float, default=BETA_RATIO)
    ap.add_argument("--beta-layers", default=",".join(map(str, BETA_LAYERS)))
    ap.add_argument("--persona-coef", type=float, default=PERSONA_COEF)
    ap.add_argument("--persona-layer", type=int, default=PERSONA_LAYER)
    ap.add_argument("--dream-coef", type=float, default=DREAM_COEF)
    ap.add_argument("--dream-layer", type=int, default=DREAM_LAYER)
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy")
    args = ap.parse_args()

    beta_layers = tuple(int(x) for x in args.beta_layers.split(",") if x.strip())
    beta_cfg = (args.beta_ratio, beta_layers)

    print(f"[load] {VLM_ID} on {DEVICE} (eager)")
    dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        VLM_ID, torch_dtype=dtype, attn_implementation="eager").to(DEVICE)
    model.eval()
    impl = getattr(model.config.get_text_config(), "_attn_implementation", None)
    assert impl == "eager", f"attn impl {impl!r} != eager; β patch dead"
    tok = AutoTokenizer.from_pretrained(VLM_ID)
    qbp.patch_qwen_attention()
    qbp.install_persona_hooks(model)
    qbp.install_dream_hooks(model)

    pvec = torch.load(args.persona_vector, map_location=DEVICE)
    bank_obj = torch.load(args.dream_bank, map_location=DEVICE)
    bank = bank_obj["vectors"] if isinstance(bank_obj, dict) else bank_obj
    print(f"[load] persona vector {tuple(pvec.shape)}, "
          f"dream bank {tuple(bank.shape)} ({bank.shape[0]} dreams)")

    greedy = not args.sample
    persona_cfg = (args.persona_coef, args.persona_layer)

    conditions = [
        ("BASELINE",                dict()),
        ("β only",                  dict(beta=beta_cfg)),
        ("persona only",            dict(persona=persona_cfg)),
        ("dream only (fixed)",      dict(dream=(args.dream_coef, args.dream_layer, "fixed"))),
        ("dream only (flux)",       dict(dream=(args.dream_coef, args.dream_layer, "flux"))),
        ("β + persona",             dict(beta=beta_cfg, persona=persona_cfg)),
        ("β + persona + dream fix", dict(beta=beta_cfg, persona=persona_cfg,
                                         dream=(args.dream_coef, args.dream_layer, "fixed"))),
        ("β + persona + dream flux",dict(beta=beta_cfg, persona=persona_cfg,
                                         dream=(args.dream_coef, args.dream_layer, "flux"))),
    ]

    print(f"\nParams: β={args.beta_ratio} layers={beta_layers} | "
          f"persona coef={args.persona_coef} L{args.persona_layer} | "
          f"dream coef={args.dream_coef} L{args.dream_layer} | "
          f"{'greedy' if greedy else 'sampled'}")

    for prompt in PROMPTS:
        print("\n" + "#" * 74)
        print(f"PROMPT: {prompt!r}")
        print("#" * 74)
        for name, kwargs in conditions:
            txt = gen(model, tok, prompt, persona_vec=pvec, dream_bank=bank,
                      greedy=greedy, seed=0, **kwargs)
            print(f"\n--- {name} ---")
            print(txt)


if __name__ == "__main__":
    main()