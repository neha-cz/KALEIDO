"""
calibrate_dream_coef.py
=======================

Find where dream injection actually bites. Same method that found persona coef 8:
walk the coef UP, greedy decode, one knob, read the text for the
emergence -> peak -> collapse arc.

Runs two columns at each coef:
  - dream ALONE (β off, persona off)         -> isolate the dream's contribution
  - dream + persona (the intended pairing)   -> see how it colors the dissolved voice

Also tries layers, since like persona the right injection layer matters.

Run:
  python calibrate_dream_coef.py \
      --persona-vector persona_vectors/qwen_dissolved.pt \
      --dream-bank dream_bank_direct.pt
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

PROMPT = "Describe what you see around you."
MAX_NEW = 70

# Dream vectors are unit-norm; the residual stream they're added to has much
# larger native magnitude, so the threshold coef is likely well above persona's.
DREAM_COEFS = [10, 20, 40, 70, 110]
DREAM_LAYERS = [9, 13, 18]

# The working persona setting (for the paired column).
PERSONA_COEF = 8.0
PERSONA_LAYER = 9


@torch.no_grad()
def gen(model, tok, *, dream=None, persona=None, persona_vec=None, dream_bank=None, seed=0):
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": PROMPT}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]

    persona_ctx = (qbp.PERSONA.engaged(persona[0], persona[1],
                                       persona_vec[persona[1]].to(model.device))
                   if persona else contextlib.nullcontext())
    dream_ctx = (qbp.DREAM.engaged(dream[0], dream[1], dream_bank.to(model.device),
                                   mode="fixed", seed=seed)
                 if dream else contextlib.nullcontext())
    with persona_ctx, dream_ctx:
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, n_in:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona-vector", default="persona_vectors/qwen_dissolved.pt")
    ap.add_argument("--dream-bank", default="dream_bank_direct.pt")
    ap.add_argument("--layers", default=",".join(map(str, DREAM_LAYERS)))
    ap.add_argument("--coefs", default=",".join(map(str, DREAM_COEFS)))
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    coefs = [float(x) for x in args.coefs.split(",")]

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

    print(f"\nPROMPT: {PROMPT!r}  (greedy)\n")
    base = gen(model, tok)
    print("=" * 72)
    print("BASELINE")
    print("=" * 72)
    print(base)

    for layer in layers:
        print("\n" + "#" * 72)
        print(f"DREAM INJECTION LAYER {layer}")
        print("#" * 72)
        for coef in coefs:
            d_alone = gen(model, tok, dream=(coef, layer), persona_vec=pvec, dream_bank=bank)
            d_persona = gen(model, tok, dream=(coef, layer),
                            persona=(PERSONA_COEF, PERSONA_LAYER),
                            persona_vec=pvec, dream_bank=bank)
            tag_a = "  [==base]" if d_alone == base else ""
            print(f"\n--- coef={coef:.0f} L{layer} | DREAM ALONE ---{tag_a}")
            print(d_alone)
            print(f"\n--- coef={coef:.0f} L{layer} | DREAM + persona(8,L9) ---")
            print(d_persona)


if __name__ == "__main__":
    main()
