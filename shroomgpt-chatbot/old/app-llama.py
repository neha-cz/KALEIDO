"""
Flask API for a Llama-based chat (via Ollama by default).
Set OLLAMA_BASE_URL, OLLAMA_MODEL, PORT (default 5001), or use defaults:
  OLLAMA_BASE_URL=http://127.0.0.1:11434  OLLAMA_MODEL=llama3.2
Run:  pip install -r requirements.txt && python app.py
  Open http://127.0.0.1:5001
Prod: gunicorn -w 1 -b 0.0.0.0:5001 app:app
"""
import os

import requests
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def _ollama_messages(history: list, new_message: str) -> list:
    out = []
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    if new_message.strip():
        out.append({"role": "user", "content": new_message.strip()})
    return out


@app.route("/")
def index():
    return render_template("index.html", model=OLLAMA_MODEL, ollama_url=OLLAMA_BASE_URL)


@app.get("/mushroom.png")
def mushroom_icon():
    return send_from_directory(app.root_path, "mushroom.png")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": OLLAMA_MODEL,
            "ollama": OLLAMA_BASE_URL,
        }
    )


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    history = data.get("history")
    if not isinstance(history, list):
        history = []

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": _ollama_messages(history, message),
        "stream": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
    except requests.RequestException as e:
        return (
            jsonify(
                {
                    "error": f"Could not reach Ollama at {OLLAMA_BASE_URL}. "
                    "Install and run: https://ollama.com  then: ollama pull "
                    f"{OLLAMA_MODEL}",
                    "detail": str(e),
                }
            ),
            503,
        )

    if r.status_code != 200:
        err = "unknown"
        try:
            err = r.json().get("error", r.text) or r.text
        except Exception:
            err = r.text
        return jsonify({"error": "Ollama error", "detail": err}), 502

    try:
        body = r.json()
    except Exception:
        return jsonify({"error": "Invalid JSON from Ollama"}), 502

    reply = (body.get("message") or {}).get("content", "").strip()
    if not reply:
        return jsonify({"error": "Empty model response"}), 502
    return jsonify({"reply": reply})


if __name__ == "__main__":
    # Default 5001: macOS often binds AirPlay Receiver to 5000.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)
