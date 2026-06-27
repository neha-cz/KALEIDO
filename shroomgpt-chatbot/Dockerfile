FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    DEVICE=cuda \
    CHAT_ONLY=1 \
    HF_HOME=/data/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_qwen.py qwen_beta_persona.py ./
COPY templates/ templates/
COPY static/ static/
COPY persona_vectors/ persona_vectors/
COPY dream_bank_direct.pt dream_bank_direct.pt

EXPOSE 7860

CMD gunicorn -w 1 -b 0.0.0.0:${PORT} --timeout 180 app_qwen:app
