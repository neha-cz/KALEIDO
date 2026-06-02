# KALEIDO

*A language model with the dial turned past sharp.*

KALEIDO is a chat interface that runs on a single knob from inside the
transformer's attention mechanism — its inverse "temperature," β. Turn β down and
the model's attention spreads instead of sharpening; its replies drift toward the
associative, the metaphorical, the loose.

Lowering β raises the model's attention entropy — and elevated neural entropy is one of the documented
*signatures* of the psychedelic state in the brain under the Entropic Brain Hypothesis. KALEIDO reproduces that marker inside a
transformer. 

![Demo 1](kaleido_demo_1.png)
![Demo 2](kaleido_demo_2.png)
![Demo 3](kaleido_demo_3.png)

## Try it

```bash
cd /shroomgpt-chatbot
pip install -r requirements.txt
huggingface-cli login          # accept the Llama-3.2-1B license first
python app_beta_demo.py        # → http://localhost:5001
```

## What's actually happening

Under the modern Hopfield interpretation of attention, each layer settles into
minima of an energy landscape, and the inverse-temperature β controls how sharp
that landscape is:

- **High β** → deep, well-separated basins → decisive, literal retrieval.
- **Low β** → shallow, merged basins → the model roams between associations
  instead of committing to one.

KALEIDO patches the model's attention to multiply β by a fixed ratio (default
0.40) on a small set of early layers. The change is purely at inference time —
no fine-tuning, no weight edits — and reads a live per-layer β ratio on every
forward pass.

## The research behind it

Check out my full mech interp study here!

[ebh-transformers](https://github.com/neha-cz/ebh-transformers) 

## Knobs

| Setting | What it does |
|---|---|
| `DEMO_BETA_RATIO` | β multiplier on demo layers (lower = looser; default 0.40) |
| `DEMO_LAYERS` | which layers receive the patch (default `2,3`) |
| prompt engineering | toggle the KALEIDO voice independently of the β patch |
| `TRIP_DEBUG` | print per-layer β ratios during generation |

All knobs are settable via environment variables or the `/api/trip/configure`
endpoint; the β patch and the system-prompt voice are independent, so you can run
either one alone.

## Stack

Flask + HuggingFace Transformers, running locally on CPU / MPS / CUDA. Single-file
backend (`app.py`); the attention patch hooks Llama's eager attention and the
dispatch registry, reading a live per-layer β ratio on every forward pass. No
external services, no API keys beyond a HuggingFace login for the model weights.

## Caveats

A toy, not a product claim. The "loosened" voice is β sitting just above the
coherence-collapse threshold, and the aesthetic is partly the system prompt. All psychedelic-state parallels here are
correlational signatures documented in the neuroscience literature, not causal
or mechanistic claims.

