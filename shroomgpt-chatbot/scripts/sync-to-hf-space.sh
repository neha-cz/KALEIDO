#!/usr/bin/env bash
# Deploy ONLY shroomgpt-chatbot/ to a Hugging Face Space (not the full monorepo).
#
# Prerequisites:
#   brew install git-lfs git-xet
#   git lfs install
#   git xet install
#   huggingface-cli login   # or git credential for HF
#
# Usage:
#   ./scripts/sync-to-hf-space.sh
#   ./scripts/sync-to-hf-space.sh https://huggingface.co/spaces/neha-cz/KALEIDO

set -euo pipefail

HF_URL="${1:-https://huggingface.co/spaces/neha-cz/KALEIDO}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kaleido-hf-space.XXXXXX")"

cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Install Git LFS first: brew install git-lfs && git lfs install"
  exit 1
fi

if ! command -v git-xet >/dev/null 2>&1; then
  echo "Install Git Xet first: brew install git-xet && git xet install"
  exit 1
fi

echo "Cloning HF Space into temp dir..."
git clone "$HF_URL" "$WORK_DIR"
cd "$WORK_DIR"

git lfs install --local
git xet install --local

echo "Replacing Space contents with shroomgpt-chatbot/ only..."
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

rsync -a \
  --exclude '.venv/' \
  --exclude 'research/' \
  --exclude 'old/' \
  --exclude '__pycache__/' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude 'static/chat-bkg.jpg' \
  --exclude 'static/cirlce-circus-2.jpg' \
  --exclude 'static/planets.jpg' \
  --exclude 'static/shroom-wallpaper.jpg' \
  --exclude 'static/shroom-logo.png' \
  --exclude 'persona_vectors/dissolved_response_avg_diff.pt' \
  "$BOT_DIR/" ./

echo "Ensuring large files are tracked before commit..."
git lfs track "*.pt" "*.jpg" "*.jpeg" "*.png" "*.gif" "*.webp" 2>/dev/null || true

git add -A
git status

if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi

git commit -m "Deploy KALEIDO chat backend"

echo "Pushing to $HF_URL ..."
git push

echo "Done. Space URL: ${HF_URL%.git}"
