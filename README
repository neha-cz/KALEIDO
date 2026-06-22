# KALEIDO

KALEIDO is a chat interface that pushes a language model toward a dissolved,
associative, ego-loosened voice using three composable interventions reached
down inside the transformer — not prompting tricks, but live edits to the
model's internal computation at inference time. The psychedelic register emerges entirely from activation-space
manipulation.

1. **β flattening.** A patch to the attention mechanism lowers its inverse
   temperature on early layers. Attention spreads instead of sharpening, and the
   model's commitment to any single response mode destabilizes.
2. **Persona-vector steering.** Inspired by Anthropic. A direction in activation space — extracted from
   the model's own contrast between a "dissolved" and a "normal" voice — is added
   to the residual stream as the model writes, steering its language toward an
   egoless, boundary-dissolving register.
3. **Dream injection.** Inspired by Google DeepDream. A surreal visual prior, sampled from a bank of "dream
   vectors" built by gradient-ascending Qwen2-VL's own visual feature geometry,
   is injected into the residual stream at every decode token. In flux mode,
   a fresh dream vector is drawn each token, so the injected content churns
   continuously — the analogue of a shifting visual field.

The three stack: β alters the texture of processing, the persona vector steers
the voic*, and the dream injection supplies visual content. All are
mechanistic, all are toggleable, and the result is a model that talks like it
has come loose from itself — without a single line of prompt engineering.

![Product Demo](kaleido_product_ss_final.png)

## Try it

```bash
cd kaleido/shroomgpt-chatbot
pip install flask flask-cors torch transformers accelerate
python app_qwen.py        # → http://localhost:5001
```

Requires `qwen_beta_persona.py` (the intervention engine),
`persona_vectors/qwen_dissolved.pt` (the persona vector), and
`dream_bank_direct.pt` (the dream bank) in the working directory. See
[Building the components](#building-the-components) below.

## Results

All metrics are computed on the output text as a signal — the direct
analogue of how Schartner et al. (2017) and the 2021 Entropic Brain study
computed complexity on EEG and DeepDream video. No model internals are measured.
Each condition is scored over 60 generations (20 samples × 3 prompts) with a
perplexity coherence guardrail so complexity shifts can't be confused with
word salad.

### Text-entropy signatures (mean, n = 60 per condition)

| Condition | LZc | Token Entropy | Distinct-2 | Semantic Jump | Semantic Spread | Perplexity |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (no interventions) | 0.792 | 0.973 | 0.927 | 0.530 | 0.572 | 1.59 |
| β + Persona | 0.638 | 0.929 | 0.687 | 0.434 | 0.504 | 5.32 |
| β + Persona + Dream (fixed) | 0.609 | 0.910 | 0.628 | 0.499 | 0.546 | 5.31 |
| **β + Persona + Dream (flux)** | **0.654** | **0.923** | **0.703** | **0.504** | **0.566** | **5.97** |

The first four columns (LZc, Token Entropy, Distinct-2)
are lexical/compression complexity; Semantic Jump and Semantic Spread are
the connectivity analogue (how far the model jumps between consecutive ideas,
and how large a semantic volume it explores). Perplexity is the coherence
guardrail — the base model's own surprise at the output. Higher complexity is
only meaningful if perplexity stays in a sane range.

The naive hypothesis — that psychedelic-style text should show higher entropy
on every axis, mirroring the EEG finding — is wrong. The interventions produce
high semantic improbability + low lexical diversity: incantatory,
perseverative speech with surprising word choices arranged in self-similar loops.
The psychedelic-language signature is not elevated lexical entropy butvelevated perplexity with increased repetition.

Within that, the three levers have separable, measurable roles. The critical
dissociation is the dream's marginal contribution:

| Metric | β + Persona | + Dream (flux) | Dream's effect |
|---|---:|---:|---:|
| Distinct-2 (lexical diversity) | 0.687 | 0.703 | +0.016 (recovers) |
| Semantic Jump (associative looseness) | 0.434 | 0.504 | +0.070 (recovers) |
| Semantic Spread (conceptual range) | 0.504 | 0.566 | +0.062 (recovers) |

Persona alone collapses both lexical and semantic range into a tight dissolved
loop. The dream injection selectively re-expands the semantic range (+0.070
jump, +0.062 spread) while lexical diversity stays low. The dissolved voice
roams across more concepts (universe, breath, light, atoms, creation,
destruction) even as it loops the same phrases. Tight word-loops, wide
concept-range — and the widening is specifically the dream's signature,
recoverable by subtraction.

## Methods

### The β patch

Under the modern Hopfield interpretation of attention, each layer settles into
minima of an energy landscape, and the inverse temperature β controls how sharp
that landscape is:

- **High β** → deep, well-separated basins → decisive, literal retrieval.
- **Low β** → shallow, merged basins → the model roams between associations
  instead of committing to one.

KALEIDO multiplies β by a fixed ratio on early layers, only during decoding —
the prompt is encoded at native β so instruction-following lands cleanly.
The effect is a phase transition, not a ramp: above a critical ratio (~0.35 on
Qwen2-VL-2B) the text is normal; below it, the model snaps to a terse fallback.
The usable regime sits just above that cliff.

### The persona vector

A steering direction is computed as the mean difference in the model's
activations when it answers under a "dissolved/no-self" system prompt versus a
normal-assistant one, taken over the response tokens. That `[n_layers, hidden]`
vector is added to the residual stream at a single mid-depth layer as the model
decodes:

```
residual ← residual + coef · persona_vector[layer]
```

The dose-response is clean: at coef 4 the voice begins to bend; at coef 8 it
produces the dissolved register ("I am a continuous cycle of creation and
destruction, a constant dance of life and death"); at coef 14 syntax fragments
("I breath, and the body of the universe"); at coef 30 it collapses to
"void, void, void." The default sits at coef 8 — dense dissolved output that
remains grammatical.

### Dream injection

Rather than feeding dreamed images as input (which produced null internal-state
shifts in v2/v3 experiments), KALEIDO injects dream-derived visual
representations directly into the residual stream — a sustained prior over
internal state rather than a stimulus that fades.

The dream bank is built **image-free**: gradient ascent in Qwen2-VL's
patch-embedding space, maximizing a mid vision block's activation along the
model's own PCA feature directions, then reading the resulting merged visual
token in LLM space. Each bank vector is the differential (dreamed − baseline),
unit-normalized. No cherry-picked images; the "surreal content" comes from the
model's visual geometry rather than human taste.

| Property | Value |
|---|---|
| Method | Direct patch-embedding PCA ascent (image-free) |
| Bank size | 64 dream vectors |
| Dimension | 1536 (LLM residual-stream space) |
| Mean pairwise cosine | 0.490 (diverse, not collapsed) |
| Source images | None |

In **flux** mode, a fresh dream vector is sampled from the bank at every decode
step, so the injected visual prior churns during a single reply — the analogue
of a shifting visual field during a psychedelic experience.

### Calibrated parameters

Each lever was calibrated independently using the same method: walk the
parameter up alone, greedy decode, and watch for the emergence → peak →
collapse arc. Locked-in values:

| Lever | Parameter | Value | Layer(s) | Role |
|---|---|---:|---|---|
| β (attention flattening) | ratio | 0.45 | 2, 3 | Destabilizes commitment |
| Persona (dissolved voice) | coef | 8.0 | 9 | Dissolved/egoless voice |
| Dream (visual prior) | coef | 20.0 | 18 | Perceptual/sensory content |

## Tech stack

Flask + HuggingFace Transformers, running locally on CPU / MPS / CUDA.
Single-file backend (`app_qwen.py`). Model: Qwen2-VL-2B-Instruct (bf16).

The β patch hooks Qwen's eager attention via the `ALL_ATTENTION_FUNCTIONS`
dispatch registry; the persona steer and dream injection are forward hooks on
decoder layers that add vectors to the residual stream during decoding. All
three read live state from a `TripState` object so the UI can tune them
per-generation. No external services, no API keys.

Post-processing: a decay-tail trimmer (cuts ellipsis/dash degradation), a
loop-cutter (detects phrase-level semantic repetition), `no_repeat_ngram_size=4`,
and `repetition_penalty=1.15`. Conversation history is capped at the last 2
turns to prevent the dissolved outputs from feeding back and compounding past
the coherence threshold.

## Building the components

```bash
# 1. Extract the persona vector
python sweep_beta_persona.py extract
#    → persona_vectors/qwen_dissolved.pt

# 2. Build the dream bank (image-free, ~2 min on CPU)
python build_dream_bank_direct.py --n-dreams 64 --dream-on-cpu
#    → dream_bank_direct.pt

# 3. Run the app
python app_qwen.py
```

## Limitations

A research demo, not a product claim. The model has no self to dissolve — what
KALEIDO steers is the language, which for a text model is the only place the
phenomenon lives. The dissolved voice is three activation-space levers pushed
near their respective edges; the aesthetic is real output from real
mechanistic edits, but it is a stylized echo of ego dissolution, not the thing
itself.

The interventions remove the hedging and uncertainty the base model normally
applies to sensitive topics — the dissolved register makes confident,
unqualified claims about religion, consciousness, and existence that a normal
chatbot would hedge on. This is an inherent property of steering away from the
"careful assistant" frame, not a bug, but users should not treat the outputs as
authoritative.

All psychedelic-state parallels are correlational signatures documented in the
neuroscience literature (Schartner et al. 2017, Viol et al. 2017, Carhart-Harris
2018), not causal or mechanistic claims about the model's experience.
