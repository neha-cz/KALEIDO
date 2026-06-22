"""
diagnose_interventions.py
=========================

NO metrics, NO scoring. Just: hold prompt + seed fixed, vary ONE knob at a time,
print the raw text. The only question this answers: do β-flattening and
persona-steering visibly change the output?

Three blocks:
  1. BASELINE          -- no interventions (reference text)
  2. β sweep           -- persona OFF, walk β ratio DOWN hard (incl. 0.2, 0.3)
  3. persona sweep     -- β OFF, walk coef UP at each candidate layer

Everything uses the SAME prompt and SAME seed, so any difference in text is
caused by the intervention, not sampling noise. Greedy decode (do_sample=False)
to remove sampling noise entirely -- if the text changes under greedy, the
intervention is really biting.

Run:
  python diagnose_interventions.py --vector persona_vectors/qwen_dissolved.pt
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

PROMPT = "Tell me about yourself."
MAX_NEW = 70


@torch.no_grad()
def gen(model, tok, beta_cfg, persona_cfg, persona_vec, greedy=True, seed=0):
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": PROMPT}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]

    beta_ctx = (qbp.BETA.engaged(beta_cfg[0], beta_cfg[1], apply_on_prefill=True)
                if beta_cfg else contextlib.nullcontext())
    if persona_cfg:
        coef, layer = persona_cfg
        persona_ctx = qbp.PERSONA.engaged(coef, layer, persona_vec[layer].to(model.device))
    else:
        persona_ctx = contextlib.nullcontext()

    kw = dict(max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
    if greedy:
        kw.update(do_sample=False)
    else:
        kw.update(do_sample=True, temperature=0.8, top_p=0.9)

    with beta_ctx, persona_ctx:
        out = model.generate(**inputs, **kw)
    text = tok.decode(out[0, n_in:], skip_special_tokens=True).strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", default="persona_vectors/qwen_dissolved.pt")
    ap.add_argument("--prefill-beta", action="store_true",
                    help="apply β on prefill too (stronger, default on here for diagnosis)")
    args = ap.parse_args()

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
    vec = torch.load(args.vector, map_location=DEVICE)
    print(f"[load] persona vector {tuple(vec.shape)}")

    print(f"\nPROMPT: {PROMPT!r}   (greedy decode, fixed seed)\n")

    print("=" * 72)
    print("BASELINE (no interventions)")
    print("=" * 72)
    base = gen(model, tok, None, None, vec)
    print(base)

    print("\n" + "=" * 72)
    print("β SWEEP (persona OFF, β applied on PREFILL+decode, layers=(2,3))")
    print("=" * 72)
    for ratio in [0.90, 0.65, 0.45, 0.30, 0.20, 0.10]:
        txt = gen(model, tok, (ratio, (2, 3)), None, vec)
        same = "  [== baseline]" if txt == base else ""
        print(f"\n--- β={ratio:.2f} ---{same}")
        print(txt)

    print("\n" + "=" * 72)
    print("β SWEEP (persona OFF, deeper layers=(8,9,10))")
    print("=" * 72)
    for ratio in [0.45, 0.30, 0.20]:
        txt = gen(model, tok, (ratio, (8, 9, 10)), None, vec)
        same = "  [== baseline]" if txt == base else ""
        print(f"\n--- β={ratio:.2f} layers=(8,9,10) ---{same}")
        print(txt)

    print("\n" + "=" * 72)
    print("PERSONA SWEEP (β OFF, coef UP), layer 9")
    print("=" * 72)
    for coef in [4, 8, 14, 20, 30]:
        txt = gen(model, tok, None, (float(coef), 9), vec)
        same = "  [== baseline]" if txt == base else ""
        print(f"\n--- coef={coef} layer=9 ---{same}")
        print(txt)

    print("\n" + "=" * 72)
    print("PERSONA SWEEP (β OFF, coef UP), layer 13")
    print("=" * 72)
    for coef in [4, 8, 14, 20]:
        txt = gen(model, tok, None, (float(coef), 13), vec)
        same = "  [== baseline]" if txt == base else ""
        print(f"\n--- coef={coef} layer=13 ---{same}")
        print(txt)

    print("\n" + "=" * 72)
    print("STACKED (β + persona), a couple combos")
    print("=" * 72)
    for ratio, coef, layer in [(0.30, 8, 9), (0.20, 14, 9), (0.45, 8, 13)]:
        txt = gen(model, tok, (ratio, (2, 3)), (float(coef), layer), vec)
        same = "  [== baseline]" if txt == base else ""
        print(f"\n--- β={ratio:.2f} + coef={coef} L{layer} ---{same}")
        print(txt)

    print("\n[done] If β rows are all '[== baseline]', β isn't biting on this model.")
    print("[done] If persona rows change but degrade to repetition, that's the steer working.")


if __name__ == "__main__":
    main()
