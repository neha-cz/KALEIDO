---
title: KALEIDO Chat
emoji: 🔮
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
suggested_hardware: t4-small
pinned: false
license: mit
---

# KALEIDO Chat

GPU-backed chat for [KALEIDO](https://github.com) — Qwen2-VL-2B with β-flatten, persona steer, and dream injection.

The marketing landing page is hosted separately (e.g. Vercel). This Space serves `/chat` and `/api/*`.

## Required files in this repo

Upload these before the Space will boot (the sync script handles Xet/LFS tracking):

| File | Description |
|---|---|
| `persona_vectors/qwen_dissolved.pt` | Persona steering vector |
| `dream_bank_direct.pt` | Dream injection bank |

Do **not** push `akansha-bullshit/` here — that folder is deployed to Vercel.

## Push to this Space

From the monorepo root (requires `git-lfs` + `git-xet`):

```bash
./scripts/sync-to-hf-space.sh https://huggingface.co/spaces/neha-cz/KALEIDO
```

See `../DEPLOY.md` if you get "push rejected because it contains binary files".

## Space secrets / variables

Set in **Settings → Variables and secrets**:

| Variable | Example | Purpose |
|---|---|---|
| `LANDING_URL` | `https://kaleido.vercel.app` | KALEIDO link in chat UI |
| `DEVICE` | `cuda` | Inference device (default) |
| `HF_MODEL` | `Qwen/Qwen2-VL-2B-Instruct` | Model id (default) |

Built-in defaults: `CHAT_ONLY=1`, `PORT=7860`.

## Local Docker test

```bash
docker build -t kaleido-chat .
docker run --gpus all -p 7860:7860 \
  -e LANDING_URL=http://localhost:3000 \
  kaleido-chat
```

Open http://localhost:7860/chat

## Notes

- **One gunicorn worker** — the model is loaded once in process memory.
- First boot downloads model weights (~several GB) and can take several minutes.
- Free GPU Spaces sleep when idle; the first request after sleep is slow.
