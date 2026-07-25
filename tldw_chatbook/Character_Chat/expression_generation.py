"""Pure prompt composition for character expression generation."""

from __future__ import annotations

from typing import Any

EXPRESSION_PROMPT_STATES: tuple[str, ...] = (
    "avatar",
    "thinking",
    "speaking",
    "error",
)
"""Available expression states for character portraits."""

STATE_MODIFIERS: dict[str, str] = {
    "avatar": "neutral friendly expression, head and shoulders portrait, looking at viewer",
    "thinking": "pensive thoughtful expression, hand near chin, looking away",
    "speaking": "mid-speech, animated engaged expression, mouth open",
    "error": "confused sheepish expression, embarrassed, sweatdrop",
}
"""Expression modifiers keyed by state."""


def compose_expression_prompt(
    *,
    name: str,
    description: str,
    personality: str = "",
    state: str,
    style_template: Any = None,
) -> tuple[str, str, dict[str, Any]]:
    """Compose a character expression generation prompt.

    Builds a prompt from name, description, personality, and state modifier.
    If a style template is provided, wraps the result via the template engine
    (ensuring user text survives the composition).

    Args:
        name: Character name (omitted from prompt if blank/whitespace).
        description: Character description (required; raises ValueError if empty).
        personality: Character personality trait (included only if non-empty).
        state: Expression state (must be in EXPRESSION_PROMPT_STATES).
        style_template: Optional GenerationTemplate for styling (default None).

    Returns:
        A (prompt, negative_prompt, params) tuple. When style_template is None,
        negative_prompt and params are empty string and empty dict.

    Raises:
        ValueError: If description is empty/whitespace or state is unknown.
    """
    if not description.strip():
        raise ValueError("description must be non-empty")

    if state not in EXPRESSION_PROMPT_STATES:
        raise ValueError(f"unknown state: {state}")

    # Build base text: name (if non-blank) + description + personality + state modifier
    parts = []
    if name.strip():
        parts.append(name.strip())
    parts.append(description.strip())
    if personality.strip():
        parts.append(personality.strip())
    parts.append(STATE_MODIFIERS[state])

    base_text = ", ".join(parts)

    # Apply style template if provided
    if style_template is not None:
        from tldw_chatbook.Chat.console_generate_image import compose_styled_request

        return compose_styled_request(base_text, style_template)

    return base_text, "", {}
