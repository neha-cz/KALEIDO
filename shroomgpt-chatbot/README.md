# ShroomGPT

A local chat interface for **Llama 3.2 1B** that combines two independent “trip” mechanisms:

1. **REBUS-inspired β annealing** — monkey-patches Llama attention so inverse-temperature β recovers over conversation time (Hopfield / simulated-annealing interpretation).
2. **NOUMADELIC prompt engineering** — optional system prompt blending **psilocybin + LSD** cognitive profiles from structured drug dimensions.

The web UI lets you compare **β-only**, **prompt-only**, **both**, or **neither** without restarting the model.

```
┌─────────────────────────────────────────────────────────────┐
│  Browser UI (templates/, static/)                           │
│    Trip drawer: Begin · +time · End                         │
│    Toggles: β annealing ON/OFF · Prompt engineering ON/OFF   │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST
┌──────────────────────────▼──────────────────────────────────┐
│  app.py (Flask)                                             │
│    /api/chat          → generate reply                        │
│    /api/trip/*        → trip clock & toggles                  │
│    BetaTripState      → decay(t), per-layer β/β₀              │
│    patch_llama_attention → scaling × β_ratio at each layer    │
└────────────┬─────────────────────────────┬────────────────────┘
             │                             │
             ▼                             ▼
   transformers (Llama 3.2 1B)    noumadelic_prompt_engineering.py
   eager attention + MPS/CUDA     combined shrooms+lsd system prompt
```

---

## Features

- **Glassmorphism chat UI** with psychedelic wallpaper and Pretext text warping (`fx.js`).
- **Independent trip controls** — β annealing and prompt engineering are separate toggles.
- **Fixed β preset** (`TRIP_PRESET` in `app.py`) — dose, schedule, τ, hierarchy, sampling coupling, etc.
- **Trip time clock** — starts at `t₀ = 1`, advances `+0.25` per assistant reply (and per **+ time** click).
- **Coupled sampling temperature** — output `temperature` scales with the same decay as β (optional via preset).
- **Complete sentences** — default `max_new_tokens=512` with prompt cap ~220 words; trims dangling fragments if the token cap is hit.
- **Stealth prompts** — trip persona does not mention drugs, simulation, or “altered state” in instructions shown to the model.
- **Legacy Ollama path** — `app-llama.py` for chat via Ollama (no β annealing).

---

## Theory (short)

Attention softmax uses scaling `β₀ = 1/√d`, which acts as an **inverse temperature** on a Hopfield-like energy landscape. Lower effective β → flatter landscape → more diffuse attention (REBUS: reduced precision on high-level priors).

During an active trip with annealing enabled:

```text
β_ratio(t, ℓ) = β(t,ℓ) / β₀ = max(floor, 1 − dose · w(ℓ) · decay(t))
```

- `w(ℓ) = (ℓ / (L−1))^p` — later layers affected more (`hierarchy_power = p`).
- `decay(t)` — exponential by default: `e^(−t/τ)` with `τ = 6`.
- The patch multiplies Hugging Face `scaling` by `β_ratio` on every eager attention forward pass.

Prompt engineering does **not** change weights; it only adds a system message when enabled.

---

## Requirements

- **Python 3.10+** (developed on 3.14)
- **~4 GB+ RAM** for Llama 3.2 1B in float16 (more on CPU)
- **Apple Silicon (MPS)**, **NVIDIA (CUDA)**, or **CPU**
- **Hugging Face account** with access to [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

---

## Installation

### 1. Clone and enter the project

```bash
cd shroomgpt-chatbot
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

`requirements.txt` lists the Flask stack; the model stack is installed separately:

```bash
pip install -r requirements.txt
pip install torch transformers accelerate
```

### 4. Hugging Face authentication

Accept the Llama license on the model page, then:

```bash
pip install huggingface_hub
huggingface-cli login
# or: hf auth login
```

Set `HF_TOKEN` in the environment if you prefer not to use the CLI.

---

## Running

```bash
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:5001** (default port avoids macOS AirPlay on 5000).

First startup downloads model weights (~1–3 minutes). Wait for:

```text
[load] Ready. 16 layers on mps.
```

### Production-style server

```bash
gunicorn -w 1 -b 0.0.0.0:5001 app:app
```

Use `use_reloader=False` in dev (already set) so the model is not loaded twice.

### Alternative: Ollama (`app-llama.py`)

No local weight download or β annealing:

```bash
ollama pull llama3.2
python app-llama.py
```

---

## Using the UI

Click **Trip** in the header to open the drawer.

| Control | Action |
|--------|--------|
| **β annealing (REBUS)** | Enables attention β patch when trip is active |
| **Prompt engineering** | Injects LSD+shrooms system prompt (works without Begin trip) |
| **Begin trip** | Starts session: `t ← 1`, applies `TRIP_PRESET` |
| **+ time** | Advances `t` by `0.25` (fast-forward comedown) |
| **End trip** | Stops session; β returns to baseline |

### Testing matrix

| Goal | β annealing | Prompt | Begin trip? |
|------|-------------|--------|-------------|
| Baseline chat | Off | Off | No |
| Annealing only | On | Off | **Yes** |
| Prompt only | Off | On | Optional |
| Full trip | On | On | **Yes** |

Status line examples:

- `Active · exp τ=6.0 · t=1.25 · decay=0.81 · dose=100% · β/β₀ 1.00–0.42 · β on · prompt on`
- `β/β₀ 1.00–1.00` with annealing on usually means trip not started or annealing off.

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL` | `meta-llama/Llama-3.2-1B-Instruct` | Hugging Face model id |
| `DEVICE` | `auto` | `cuda`, `mps`, or `cpu` |
| `PORT` | `5001` | Flask port |
| `TRIP_DEBUG` | `0` | Set to `1` to log per-layer β ratios in the terminal |

### Trip preset (`TRIP_PRESET` in `app.py`)

| Field | Value | Meaning |
|-------|-------|---------|
| `dose` | `1.0` | Full onset depth (100%) |
| `schedule` | `exponential` | `decay(t) = e^(−t/τ)` |
| `tau` | `6.0` | Recovery time constant |
| `t_final` | `12.0` | Used if schedule is `linear` |
| `hierarchy_power` | `2.0` | Late-layer weighting exponent |
| `initial_t` | `1.0` | Trip clock at Begin |
| `turn_step` | `0.25` | Δt per reply / +time |
| `couple_sampling` | `true` | Scale sampling temperature with decay |
| `sampling_T_hot` | `1.6` | Peak sampling multiplier at onset |

Edit `TRIP_PRESET` and restart the server. UI toggles for annealing/prompt default to **on** and persist across End trip.

### Generation limits

| Constant | Value |
|----------|-------|
| `DEFAULT_MAX_NEW_TOKENS` | `512` |
| `MAX_NEW_TOKENS_CAP` | `1024` |

Trip prompts ask for **under 220 words** and a complete final sentence.

---

## API reference

### `GET /api/health`

Model, device, layer count, trip snapshot.

### `POST /api/chat`

```json
{
  "message": "Why is the sky blue?",
  "history": [
    { "role": "user", "content": "Hi" },
    { "role": "assistant", "content": "Hello." }
  ],
  "temperature": 0.8,
  "top_p": 0.9,
  "max_new_tokens": 512,
  "seed": 42
}
```

Response includes `reply`, `trip_before`, `trip_after`, `per_layer_beta_ratio`, `sampling_temperature_multiplier`, etc.

### Trip endpoints

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/api/trip/start` | — | Begin trip (`TRIP_PRESET`) |
| `POST` | `/api/trip/stop` | — | End trip |
| `POST` | `/api/trip/advance` | `{"steps": 0.25}` | Manual Δt (default: `turn_step`) |
| `GET` | `/api/trip/state` | — | Snapshot + per-layer β ratios |
| `POST` | `/api/trip/annealing` | `{"enabled": true}` | Toggle β patch |
| `POST` | `/api/trip/prompt_engineering` | `{"enabled": true}` | Toggle system prompt |
| `POST` | `/api/trip/configure` | partial preset fields | Update knobs without reset |

Example — annealing only:

```bash
curl -s -X POST http://127.0.0.1:5001/api/trip/annealing \
  -H 'Content-Type: application/json' -d '{"enabled": true}'
curl -s -X POST http://127.0.0.1:5001/api/trip/prompt_engineering \
  -H 'Content-Type: application/json' -d '{"enabled": false}'
curl -s -X POST http://127.0.0.1:5001/api/trip/start
```

---

## Project structure

```text
shroomgpt-chatbot/
├── app.py                          # Main Flask app (β annealing + chat)
├── app-llama.py                    # Ollama-only variant (legacy)
├── noumadelic_prompt_engineering.py  # Drug profiles + prompt builder
├── requirements.txt                # Flask stack (install torch separately)
├── templates/
│   └── index.html                  # Chat shell + trip drawer
├── static/
│   ├── chat.js                     # Chat + trip API client
│   ├── style.css                   # Glass UI + trip drawer
│   ├── fx.js                       # Pretext warping + cursor mushroom
│   ├── shroom-wallpaper.jpg        # Background
│   └── shroom-logo.png
├── mushroom.png                    # Favicon route
└── README.md
```

---

## Prompt engineering module

`noumadelic_prompt_engineering.py` defines:

- **`drug_dict`** — seven cognitive axes per substance (shrooms, lsd, ketamine, mdma).
- **`combine_drug_profiles()`** — merges shrooms + lsd axis text (no drug names in output).
- **`build_trip_chat_messages()`** — Hugging Face chat format with ShroomGPT system prompt.
- **`sanitize_generated_text()`** — strips `*` from model output.

Only **`shroomgpt_trip_generator`** is used in production (`in_character=True`, stealth rules).

---

## Troubleshooting

### `command not found: huggingface-cli`

```bash
pip install huggingface_hub
hf auth login
```

Or `export HF_TOKEN=hf_...` before `python app.py`.

### `bad interpreter` / broken venv after moving the folder

Recreate the venv:

```bash
rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt torch transformers accelerate huggingface_hub
```

### Chat returns `generation failed`

- Confirm HF login and model access.
- Check terminal traceback (OOM → use CPU or close other apps).
- Set `TRIP_DEBUG=1` to verify β patch is firing.

### Annealing seems to do nothing

- Turn **β annealing** on and click **Begin trip**.
- Check status for `β/β₀` below `1.00` on upper layers.
- Remember `t` starts at **1.0** and steps by **0.25** — effects change slowly by design.

### Responses cut off mid-sentence

Defaults were tuned to avoid hitting `max_new_tokens`. If it persists, increase in API: `"max_new_tokens": 768`. Incomplete tails are trimmed to the last full sentence when the cap is hit.

### Port 5001 in use

```bash
PORT=5002 python app.py
```

---

## License and disclaimers

- **Llama weights** — subject to [Meta’s license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
- This project is for **research and experimentation** in attention dynamics and prompt design — not medical advice or encouragement of drug use.
- Generated “trip” text is **simulated**; the UI copy is metaphorical.

---

## Acknowledgments

- **REBUS** framing for precision / β annealing narrative.
- **NOUMADELIC** altered-state prompt framework (`noumadelic_prompt_engineering.py`).
- **Meta Llama 3.2**, **Hugging Face Transformers**, **Flask**.
