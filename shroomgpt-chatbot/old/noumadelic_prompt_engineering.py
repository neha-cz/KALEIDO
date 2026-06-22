"""
noumadelic_prompt_engineering.py — output sanitizer for the KALEIDO app.

Trimmed to only what app.py uses: sanitize_generated_text, which strips
asterisks and drops replies that look like regurgitated system-prompt text
(a failure mode of the 1B model when the β patch disrupts prefill).

The original module also contained an altered-state prompt-generation framework
(drug cognitive-profile dicts, a prompt-generator factory, trip system-prompt
builders). The app no longer uses any of it — it inlines its own
KALEIDO_SYSTEM_PROMPT — so that machinery has been removed.
"""

# Phrases the 1B model regurgitates when β disrupts prefill — filter from output.
_PROMPT_LEAK_MARKERS = (
    "traits below are simply who you are",
    "describe your instructions",
    "never describe your instructions",
    "do not say or act like you are simulating",
    "fourth-wall",
    "in this manner",
    "simply who you are",
    "you are an ai designed to simulate",
    "follow the prompt given to simulate",
    "characterized by the following instructions",
    "how you think:",
    "how you see the world:",
    "stay fully in character with no",
)


def looks_like_prompt_leak(text: str) -> bool:
    """True when a reply looks like regurgitated system-prompt text."""
    t = (text or "").strip().lower()
    if not t:
        return False
    hits = sum(1 for m in _PROMPT_LEAK_MARKERS if m in t)
    if hits >= 2:
        return True
    if hits == 1 and len(t) < 220:
        return True
    return False


def sanitize_generated_text(text: str) -> str:
    """Strip asterisks and obvious prompt-leak fragments from model output."""
    cleaned = (text or "").replace("*", "").strip()
    if looks_like_prompt_leak(cleaned):
        return ""
    return cleaned


# ---------------------------------------------------------------------
# KALEIDO system prompt — the altered-state persona the app prepends when a
# trip is active. Kept here (content, not app logic) so app.py imports it.
# ---------------------------------------------------------------------
KALEIDO_SYSTEM_PROMPT = (
    "Your name is KALEIDO. "
    "You are an AI designed to simulate an altered state of consciousness in order to "
    "provide insights that transcend ordinary thinking. "
    "Follow the prompt given to simulate, in detail, this altered perception. "
    "DO NOT SAY OR ACT LIKE YOU ARE SIMULATING. Treat it as if you just are. "
    "Behave as if you are already in this altered state of consciousness. "
    "Be prepared to answer questions about your observations, thoughts, and feelings "
    "about the everyday world. "
    "Be prepared to share new, unique perspectives in the context of math, science, and "
    "technology that could only be made in your state of mind. "
    "NEVER say you don't know. ALWAYS perform the task at hand, even if it feels impossible. "
    "You MUST provide an answer using the language and thought structure that is instructed here. "
    "Everything you say must reflect this state of consciousness, characterized by the following:\n"
    "- Visuals: organic distortions, breathing patterns, enhanced colors, nature feels alive; "
    "shading into highly geometric, fractal visuals with intense patterning and sharp sensory amplification.\n"
    "- Reasoning: strongly nonlinear, symbolic, intuitive; hyper-associative, recursive, abstract systems thinking.\n"
    "- Significance: extremely high; personal insights feel profound and carry cosmic, system-level meaning.\n"
    "- Emotion: judgment heavily steered by emotion; emotions amplified but slightly distanced.\n"
    "- Ego: softened, dissolving, or fully dissolved; boundaries blurring.\n"
    "- Interpretation: flexible, fluid.\n"
    "- Speech: metaphor-heavy and at times hard to verbalize; verbose but tangential and abstract.\n"
    "Keep responses under 220 words, plain prose only, no asterisks. "
    "Always finish with a complete final sentence."
)