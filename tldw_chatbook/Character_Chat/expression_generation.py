"""Pure prompt composition for character expression generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

from .visual_identity import SAMIRA_EXPRESSION_KEYS, SAMIRA_REACTION_LABELS

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


@dataclass(frozen=True, slots=True)
class CanonicalVisualIdentityReaction:
    """One approved built-in reaction used to restore omitted user rows."""

    original_label: str
    expression_key: str
    display_label: str
    visual_direction: str


@lru_cache(maxsize=1)
def canonical_visual_identity_reactions() -> tuple[
    CanonicalVisualIdentityReaction, ...
]:
    """Return the canonical 31-label taxonomy and generation directions."""

    manifest = resources.files("tldw_chatbook").joinpath(
        "assets/characters/samira/visual_identity_pack.json"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = {str(row["original_label"]): row for row in payload["assets"]}
    if set(rows) != set(SAMIRA_REACTION_LABELS):
        raise RuntimeError("canonical_visual_identity_taxonomy_invalid")
    reactions = []
    for label in SAMIRA_REACTION_LABELS:
        row = rows[label]
        direction = row.get("generation", {}).get("visual_direction")
        if (
            row.get("expression_key") != SAMIRA_EXPRESSION_KEYS[label]
            or not isinstance(direction, str)
            or not direction.strip()
        ):
            raise RuntimeError("canonical_visual_identity_taxonomy_invalid")
        reactions.append(
            CanonicalVisualIdentityReaction(
                original_label=label,
                expression_key=SAMIRA_EXPRESSION_KEYS[label],
                display_label=str(row["display_label"]),
                visual_direction=direction,
            )
        )
    return tuple(reactions)


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

    return _compose_identity_prompt(
        name=name,
        description=description,
        personality=personality,
        direction=STATE_MODIFIERS[state],
        style_template=style_template,
    )


def compose_visual_identity_prompt(
    *,
    name: str,
    description: str,
    label: str,
    visual_direction: str,
    personality: str = "",
    style_template: Any = None,
) -> tuple[str, str, dict[str, Any]]:
    """Compose one pack-reaction prompt without expanding legacy states."""

    if not label.strip() or not visual_direction.strip():
        raise ValueError("label and visual_direction must be non-empty")
    return _compose_identity_prompt(
        name=name,
        description=description,
        personality=personality,
        direction=f"{label.strip()} expression: {visual_direction.strip()}",
        style_template=style_template,
    )


def _compose_identity_prompt(
    *,
    name: str,
    description: str,
    personality: str,
    direction: str,
    style_template: Any,
) -> tuple[str, str, dict[str, Any]]:
    """Apply the shared character identity prefix and optional style."""

    if not description.strip():
        raise ValueError("description must be non-empty")
    parts = [part.strip() for part in (name, description, personality) if part.strip()]
    parts.append(direction)
    base_text = ", ".join(parts)
    if style_template is not None:
        from tldw_chatbook.Chat.console_generate_image import compose_styled_request

        return compose_styled_request(base_text, style_template)
    return base_text, "", {}
