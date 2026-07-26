"""Pure request builders for LLM-assisted character authoring.

This module owns the *contract* for generating character-card text: which
fields may be generated, how much of the character is shown to the model, and
how a whole-character reply is parsed back into fields. It builds provider
messages and parses provider text — it never calls a provider, touches the
database, or imports Textual, so the prompt contract is testable on its own.

Two context modes exist because they serve different authoring moments:

``whole_character``
    Show every populated field. Use when the new text must stay consistent
    with the character as a whole (a scenario that matches the personality).

``field_and_description``
    Show only the base description plus the field's current value. Use when
    the author wants a fresh take that is not anchored to everything already
    written — the narrow mode is genuinely narrow, otherwise the choice would
    be decorative.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Literal, Mapping

from ..Internal_Prompts import get_internal_prompt

CharacterFieldContextMode = Literal["whole_character", "field_and_description"]

#: Character-card fields the editor can generate, mapped to the human label
#: used in the instruction ("Write the first message for this character").
#: Keys are the record keys the editor reads and writes.
GENERATABLE_FIELDS: dict[str, str] = {
    "description": "description",
    "personality": "personality",
    "scenario": "scenario",
    "first_message": "first message",
    "system_prompt": "system prompt",
    "post_history_instructions": "post-history instructions",
    "creator_notes": "creator notes",
}

#: Fields carried into `whole_character` context, in card reading order. The
#: target field is excluded at build time (it is shown separately as the
#: current value, if any).
_CONTEXT_FIELDS: tuple[str, ...] = (
    "description",
    "personality",
    "scenario",
    "first_message",
    "system_prompt",
    "post_history_instructions",
)

#: Keys accepted from a whole-character reply. Anything else is dropped: a
#: model that invents a key must never reach the character record.
WHOLE_CHARACTER_KEYS: tuple[str, ...] = (
    "name",
    "description",
    "personality",
    "scenario",
    "first_message",
)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class CharacterGenerationError(RuntimeError):
    """Raised when a generation request or reply is unusable."""


def _clean(value: Any) -> str:
    """Return ``value`` as trimmed text, treating None/non-str as empty-ish."""
    if value is None:
        return ""
    return str(value).strip()


def _labelled_context(record: Mapping[str, Any], fields: Iterable[str]) -> list[str]:
    """Return ``Label: value`` lines for the populated fields among ``fields``."""
    lines: list[str] = []
    for key in fields:
        text = _clean(record.get(key))
        if not text:
            continue
        label = GENERATABLE_FIELDS.get(key, key).capitalize()
        lines.append(f"{label}: {text}")
    return lines


def build_field_generation_messages(
    field: str,
    record: Mapping[str, Any],
    *,
    context_mode: CharacterFieldContextMode = "whole_character",
    instruction: str | None = None,
) -> list[dict[str, str]]:
    """Build provider messages that write one character field.

    Args:
        field: Record key to generate; must be in ``GENERATABLE_FIELDS``.
        record: The character record being edited (partial is fine).
        context_mode: How much of the character to show the model. See the
            module docstring.
        instruction: Optional extra steer from the author ("make her colder").

    Returns:
        A ``[system, user]`` message list ready for the provider gateway.

    Raises:
        CharacterGenerationError: If ``field`` is not generatable.
    """
    if field not in GENERATABLE_FIELDS:
        raise CharacterGenerationError(
            f"{field!r} is not a generatable character field; "
            f"expected one of {sorted(GENERATABLE_FIELDS)}"
        )
    label = GENERATABLE_FIELDS[field]
    name = _clean(record.get("name")) or "this character"

    parts: list[str] = [f"Character name: {name}"]
    if context_mode == "whole_character":
        context_lines = _labelled_context(
            record, (key for key in _CONTEXT_FIELDS if key != field)
        )
    else:
        # Narrow mode: the base description only. Deliberately excludes
        # personality/scenario/etc so the two modes actually differ.
        context_lines = (
            _labelled_context(record, ("description",)) if field != "description" else []
        )
    if context_lines:
        parts.append("")
        parts.append("Character so far:")
        parts.extend(context_lines)

    current = _clean(record.get(field))
    if current:
        parts.append("")
        parts.append(f"Current {label} (rewrite it, do not simply repeat it):")
        parts.append(current)

    parts.append("")
    parts.append(f"Write the {label} for this character.")
    if instruction and instruction.strip():
        parts.append(f"Additional direction from the author: {instruction.strip()}")

    return [
        {"role": "system", "content": get_internal_prompt("character.generate_field")},
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_whole_character_messages(concept: str) -> list[dict[str, str]]:
    """Build provider messages that draft a whole character from a concept.

    Args:
        concept: The author's one-line idea for the character.

    Returns:
        A ``[system, user]`` message list ready for the provider gateway.

    Raises:
        CharacterGenerationError: If ``concept`` is blank.
    """
    text = _clean(concept)
    if not text:
        raise CharacterGenerationError(
            "a character concept is required to generate a character"
        )
    return [
        {"role": "system", "content": get_internal_prompt("character.generate_whole")},
        {"role": "user", "content": f"Character concept: {text}"},
    ]


def parse_whole_character_response(text: str) -> dict[str, str]:
    """Parse a whole-character reply into known card fields.

    Tolerates the markdown fence local models routinely add, drops keys that
    are not part of the card, and coerces scalar values to text so a model
    answering ``42`` cannot poison a text field.

    Args:
        text: Raw provider reply.

    Returns:
        Mapping of known card field -> text, containing only keys the model
        actually supplied.

    Raises:
        CharacterGenerationError: If no JSON object can be recovered.
    """
    raw = _clean(text)
    fenced = _JSON_FENCE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    if not raw.startswith("{"):
        # Some models prepend a sentence; recover the first object if present.
        start = raw.find("{")
        end = raw.rfind("}")
        raw = raw[start : end + 1] if start != -1 and end > start else raw
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise CharacterGenerationError(
            "the model did not return a JSON character object"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CharacterGenerationError(
            "the model returned JSON that is not a character object"
        )
    parsed: dict[str, str] = {}
    for key in WHOLE_CHARACTER_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if value is None or isinstance(value, (list, dict)):
            continue
        parsed[key] = str(value).strip()
    if not parsed:
        raise CharacterGenerationError(
            "the model returned no recognizable character fields"
        )
    return parsed
