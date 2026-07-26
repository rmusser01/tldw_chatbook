# tldw_chatbook/Internal_Prompts/character_prompts.py
"""Character-authoring prompt specs.

Registers the two system prompts behind the character editor's LLM-assisted
generation: one that writes a SINGLE field of an existing character, and one
that drafts a whole character from a short concept. Both are editable in
Settings like any other internal prompt, which matters here because prose
style is exactly the thing a roleplay author wants to tune.

The character context and the concept are supplied as the user message at call
time (see ``Character_Chat/character_generation.py``), so neither prompt takes
placeholders.
"""

from .catalog import PromptSpec, register

register(
    PromptSpec(
        id="character.generate_field",
        subsystem="character",
        title="Character field generation",
        description=(
            "Writes one field of a character card (description, personality, "
            "scenario, first message, ...) from the character context the "
            "editor supplies."
        ),
        used_in="Character_Chat/character_generation.py (build_field_generation_messages)",
        default=(
            "You are helping an author write a roleplay character card. "
            "Write ONLY the single requested field, in the author's voice, as "
            "finished card text. Match the tone and facts of the context you "
            "are given and never contradict them. Do not restate the field "
            "name, add headings, quote the output, wrap it in code fences, or "
            "explain your choices. Do not address the author. Write the field "
            "content and nothing else. Keep it concise unless the field is "
            "inherently long-form."
        ),
        contract_note=(
            "The target field name, the character context, and any existing "
            "value are supplied as the user message at call time; this prompt "
            "is the system instruction and takes no placeholders."
        ),
    )
)

register(
    PromptSpec(
        id="character.generate_whole",
        subsystem="character",
        title="Whole character generation",
        description=(
            "Drafts a complete character card as JSON from a one-line concept "
            "supplied by the author."
        ),
        used_in="Character_Chat/character_generation.py (build_whole_character_messages)",
        default=(
            "You are helping an author create a roleplay character card from a "
            "short concept. Reply with a single JSON object and nothing else - "
            "no prose before or after, no code fence. Use exactly these keys: "
            '"name", "description", "personality", "scenario", '
            '"first_message". Every value must be a string. "first_message" is '
            "the character's opening line to the user, written in the "
            "character's voice; actions may be wrapped in asterisks. Keep "
            "description and personality to a few sentences each. Do not "
            "invent keys beyond the five listed."
        ),
        contract_note=(
            "The author's concept is supplied as the user message at call "
            "time; this prompt is the system instruction and takes no "
            "placeholders. The reply is parsed as JSON by "
            "`parse_whole_character_response`, which tolerates a markdown "
            "fence and drops unknown keys."
        ),
    )
)
