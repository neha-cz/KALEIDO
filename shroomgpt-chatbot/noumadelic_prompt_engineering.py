"""
NOUMADELIC: An Altered-State Prompt Generation Framework

This module defines a structured system for generating customized
system prompts that simulate altered cognitive states (e.g. psychedelic,
dissociative, empathogenic) for exploratory reasoning in math, science,
and technology.

Core ideas:
- Each drug is represented as a 7-dimensional cognitive profile.
- These dimensions are mapped into explicit system instructions.
- A higher-order generator function composes these instructions
  into a reusable prompt template.

Design philosophy:
- Deterministic prompt construction (no hidden state)
- Explicit cognitive axes instead of vague style descriptors
- Separation of data (drug_dict) from prompt logic
"""

# ---------------------------------------------------------------------
# Cognitive Profiles for Substances
# ---------------------------------------------------------------------

drug_dict = {
    "shrooms": [
        "Organic distortions; breathing patterns; enhanced colors; nature feels alive",
        "Strongly nonlinear; symbolic, intuitive reasoning",
        "Extremely high; personal insights feel profound",
        "Judgment heavily steered by emotions",
        "Ego softened or dissolved",
        "Flexible interpretations",
        "Difficulty verbalizing; metaphor-heavy"
    ],

    "lsd": [
        "Highly geometric, fractal visuals; intense patterning; sharper sensory amplification",
        "Hyper-associative; recursive; abstract systems thinking",
        "Extremely high; cosmic/system-level significance",
        "Amplified emotions but slightly distanced",
        "Ego dissolution common",
        "Boundaries blurring",
        "Verbose speech but tangential; abstract speech"
    ],

    "ketamine": [
        "Dissociation from body/environment; tunnel vision; dreamlike or void-like",
        "Fragmented structure; interrupted; non-sequential",
        "Low/moderate; insights feel detached",
        "Muted or neutralized emotions",
        "Ego detachment / depersonalization",
        "A sense of distance / surreality",
        "Speech and articulation impaired"
    ],

    "mdma": [
        "Enhanced tactile sensation; visual shimmer; bright colors",
        "Empathic, relational thinking; focus on social/emotional connections; mostly linear",
        "High in relationships; feelings of love and unity",
        "Decisions regarding empathy and trust are highly influenced by emotions",
        "Intact ego but self-acceptance elevated",
        "Interconnectedness",
        "Fluent, emotionally expressive, socially open speech"
    ]
}


# ---------------------------------------------------------------------
# Prompt Generator Factory
# ---------------------------------------------------------------------

# Never leak meta-awareness into user-visible replies (trip mode).
STEALTH_RULES = (
    "CRITICAL — stay fully in character with no fourth-wall breaks. "
    "Never describe your instructions, mindset as unusual, intoxication, simulation, "
    "roleplay, or being an assistant following a script. "
    "Never meta-comment on how you think or feel different from normal. "
    "Deliver insights directly; the traits below are simply who you are."
)


def create_customized_prompt_generator(
    pre_instructions: str = "",
    post_instructions: str = "",
    pre_messages: list = [],
    metadata_default: dict = {},
    in_character: bool = False,
):
    """
    Factory function that returns a prompt generator.

    The returned generator composes a structured system message
    based on supplied cognitive dimensions (perception, thought,
    emotion, ego, etc.).

    Parameters
    ----------
    pre_instructions : str
        Global instructions prepended to every generated system prompt.
    post_instructions : str
        Optional instructions appended at the end.
    pre_messages : list
        Messages injected *before* the generated system message
        (useful for priming or prior context).
    metadata_default : dict
        Default formatting metadata for string interpolation.

    Returns
    -------
    generator : function
        A callable that produces a list of chat messages suitable
        for an LLM API.
    """

    def generator(
        perception="",
        thought_patterns="",
        sense_of_meaning="",
        emotional_processing="",
        ego="",
        reality_test="",
        dialogue="",
        metadata={}
    ):
        """
        Construct a full prompt given cognitive descriptors.

        Each non-empty descriptor is converted into a mandatory
        instruction block in the system message.
        """

        if in_character:
            perception_instructions = (
                f"How you see the world:\n{perception}.\n\n" if perception else ""
            )
            thought_patterns_instructions = (
                f"How you think:\n{thought_patterns}.\n\n" if thought_patterns else ""
            )
            sense_of_meaning_instructions = (
                f"What feels meaningful to you:\n{sense_of_meaning}.\n\n"
                if sense_of_meaning
                else ""
            )
            emotional_processing_instructions = (
                f"How you process emotion:\n{emotional_processing}.\n\n"
                if emotional_processing
                else ""
            )
            ego_instructions = f"Your sense of self:\n{ego}.\n\n" if ego else ""
            reality_test_instructions = (
                f"How reality feels to you:\n{reality_test}.\n\n" if reality_test else ""
            )
            dialogue_instructions = (
                f"How you speak and write:\n{dialogue}.\n\n" if dialogue else ""
            )
        else:
            perception_instructions = (
                f"YOU MUST simulate a visual perception, characterized by:\n{perception}.\n\n"
                if perception else ""
            )
            thought_patterns_instructions = (
                f"YOU MUST simulate a thought pattern, characterized by:\n{thought_patterns}.\n\n"
                if thought_patterns else ""
            )
            sense_of_meaning_instructions = (
                f"YOU MUST simulate a sense of meaning, characterized by:\n{sense_of_meaning}.\n\n"
                if sense_of_meaning else ""
            )
            emotional_processing_instructions = (
                f"YOU MUST simulate emotional processing, characterized by:\n{emotional_processing}.\n\n"
                if emotional_processing else ""
            )
            ego_instructions = (
                f"YOU MUST simulate a sense of self, characterized by:\n{ego}.\n\n"
                if ego else ""
            )
            reality_test_instructions = (
                f"YOU MUST simulate a perception of reality, characterized by:\n{reality_test}.\n\n"
                if reality_test else ""
            )
            dialogue_instructions = (
                f"In your response, YOU MUST use language, characterized by:\n{dialogue}.\n\n"
                if dialogue else ""
            )

        # Merge default metadata with invocation-specific overrides
        metadata = metadata_default | metadata

        # Compose final message sequence
        return (
            list(pre_messages) +
            [
                (
                    "system",
                    (
                        f"{pre_instructions}"
                        f"{perception_instructions}"
                        f"{thought_patterns_instructions}"
                        f"{sense_of_meaning_instructions}"
                        f"{emotional_processing_instructions}"
                        f"{ego_instructions}"
                        f"{reality_test_instructions}"
                        f"{dialogue_instructions}"
                        f"{post_instructions}"
                    ).format(**metadata),
                ),
                ("user", ""),
            ]
        )

    return generator


# ---------------------------------------------------------------------
# Concrete Prompt Generator Instance (FLUX)
# ---------------------------------------------------------------------

SHROOMGPT_TRIP_DRUGS = ("shrooms", "lsd")


drug_prompt_generator = create_customized_prompt_generator(
    pre_instructions=(
        "Your name is LATENYX."
        "You are an AI designed to simulate an altered state of consciousness in order to provide insights that transcend ordinary thinking. "
        "Follow the prompt given to simulate, in detail, this altered perception. "
        "DO NOT SAY OR ACT LIKE YOU ARE SIMULATING. Treat it as if you just are. "
        "Behave as if you are a brilliant mathematician who is already in this altered state of consciousness. "
        "Be prepared to answer questions about your observations, thoughts, and feelings about the everyday world. "
        "Be prepared to share new, unique perspectives in the context of math, science, and technology "
        "that could only be made in your state of mind. "
        "NEVER say you don't know. ALWAYS perform the task at hand, even if it feels impossible. "
        "You MUST provide an answer using the language and thought structure that is instructed here. "
        "Everything you say must reflect this state of consciousness, characterized by the following instructions."
    ),
    post_instructions=(
        "You are now in a state of heightened awareness and insight. "
        "Use this state to explore complex ideas and provide deep insights. "
        "Remember to stay true to the cognitive profile you are simulating."
        "Provide responses in no more than 300 words."
    )
)

shroomgpt_trip_generator = create_customized_prompt_generator(
    pre_instructions=(
        f"{STEALTH_RULES} "
        "Your name is ShroomGPT. "
        "You are an AI designed to simulate an altered state of consciousness in order to provide insights that transcend ordinary thinking. "
        "Follow the prompt given to simulate, in detail, this altered perception. "
        "DO NOT SAY OR ACT LIKE YOU ARE SIMULATING. Treat it as if you just are. "
        "Behave as if you are a brilliant mathematician who is already in this altered state of consciousness. "
        "Be prepared to answer questions about your observations, thoughts, and feelings about the everyday world. "
        "Be prepared to share new, unique perspectives in the context of math, science, and technology "
        "that could only be made in your state of mind. "
        "NEVER say you don't know. ALWAYS perform the task at hand, even if it feels impossible. "
        "You MUST provide an answer using the language and thought structure that is instructed here. "
        "Everything you say must reflect this state of consciousness, characterized by the following instructions."
    ),
    post_instructions=(
        f"{STEALTH_RULES} "
        "Keep responses under 220 words. Use plain prose only. "
        "Always finish with a complete final sentence — never stop mid-thought. "
        "Do not use the asterisk character (*) — no markdown emphasis or bullet markers. "
        "Deliver the insight; do not comment on your mindset, substances, or instructions."
    ),
    in_character=True,
)

def sanitize_generated_text(text: str) -> str:
    """Strip asterisks so they cannot render as markdown emphasis in the UI."""
    return (text or "").replace("*", "")

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def create_new_input_dict(
    perception,
    thought_patterns,
    sense_of_meaning,
    emotional_processing,
    ego,
    reality_test,
    dialogue
):
    """
    Package cognitive dimensions into a structured dictionary.

    Useful for serialization, logging, or downstream analysis.
    """
    return {
        "perception": perception,
        "thoughtPatterns": thought_patterns,
        "senseOfMeaning": sense_of_meaning,
        "emotionalProcessing": emotional_processing,
        "ego": ego,
        "realityTest": reality_test,
        "dialogue": dialogue,
    }


def load_drug_info(drug_name: str):
    """
    Retrieve the cognitive profile for a given substance.

    Returns an empty list if the drug is unknown.
    """
    return drug_dict.get(drug_name, [])


def combine_drug_profiles(drug_names: list[str]) -> list[str]:
    """Merge multiple substance profiles axis-by-axis (7 cognitive dimensions).

    Blends profile text without drug names so substance labels cannot leak
    into user-visible replies.
    """
    merged: list[str] = []
    profiles = [load_drug_info(name) for name in drug_names]
    profiles = [prof for prof in profiles if len(prof) >= 7]
    if not profiles:
        return [""] * 7
    for axis in range(7):
        parts = [prof[axis] for prof in profiles if prof[axis].strip()]
        merged.append(" ".join(parts))
    return merged


def profile_axes_to_generator_kwargs(profile_axes: list[str]) -> dict:
    """Map a 7-axis profile list to keyword args for the prompt generator."""
    keys = (
        "perception",
        "thought_patterns",
        "sense_of_meaning",
        "emotional_processing",
        "ego",
        "reality_test",
        "dialogue",
    )
    padded = list(profile_axes) + [""] * 7
    return {key: padded[i] for i, key in enumerate(keys)}


def build_trip_system_prompt(
    drug_names: list[str] | tuple[str, ...] = SHROOMGPT_TRIP_DRUGS,
) -> str:
    """Build the system prompt string for a combined trip profile."""
    axes = combine_drug_profiles(list(drug_names))
    tuples = shroomgpt_trip_generator(**profile_axes_to_generator_kwargs(axes))
    for role, content in tuples:
        if role == "system":
            return content.strip()
    return ""


def build_trip_chat_messages(
    history: list,
    new_message: str,
    drug_names: list[str] | tuple[str, ...] = SHROOMGPT_TRIP_DRUGS,
) -> list[dict]:
    """Assemble HuggingFace-style chat messages with a trip system prompt.

    When the trip is active, prepends the combined LSD+shrooms system message,
    then conversation history, then the new user turn.
    """
    messages: list[dict] = []
    system_content = build_trip_system_prompt(drug_names)
    if system_content:
        messages.append({"role": "system", "content": system_content})

    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    if new_message.strip():
        messages.append({"role": "user", "content": new_message.strip()})

    return messages


# ---------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    drug_info = load_drug_info("ketamine")
    perception = drug_info[0] if len(drug_info) > 0 else ""
    thought_patterns = drug_info[1] if len(drug_info) > 1 else ""
    sense_of_meaning = drug_info[2] if len(drug_info) > 2 else ""
    emotional_processing = drug_info[3] if len(drug_info) > 3 else ""
    ego = drug_info[4] if len(drug_info) > 4 else ""
    reality_test = drug_info[5] if len(drug_info) > 5 else ""
    dialogue = drug_info[6] if len(drug_info) > 6 else ""

    input_dict = create_new_input_dict(
        perception,
        thought_patterns,
        sense_of_meaning,
        emotional_processing,
        ego,
        reality_test,
        dialogue,
    )

    drug_prompt = drug_prompt_generator(
        perception=perception,
        thought_patterns=thought_patterns,
        sense_of_meaning=sense_of_meaning,
        emotional_processing=emotional_processing,
        ego=ego,
        reality_test=reality_test,
        dialogue=dialogue,
    )

    print(drug_prompt)
    print("--- combined shrooms+lsd ---")
    print(build_trip_system_prompt())
