"""Bound and sanitize untrusted character text for terminal display only."""

from __future__ import annotations

import unicodedata

import wcwidth

_PRESERVED_CONTROLS = frozenset({"\n", "\t"})


def _deterministic_text(value: object) -> str:
    """Return text without leaking process-specific default object addresses."""
    if value is None:
        return ""
    value_type = type(value)
    if (
        value_type.__str__ is object.__str__
        and value_type.__repr__ is object.__repr__
    ):
        return f"<{value_type.__name__}>"
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - arbitrary __str__ implementations vary
        return f"<{value_type.__name__}>"


def sanitize_character_display_text(
    value: object,
    *,
    max_characters: int,
) -> str:
    """Project arbitrary character-card data into bounded terminal-safe text.

    This function is for display sinks only. Callers must retain the original
    card value for editing, persistence, prompts, exports, and speech.

    Args:
        value: Value to convert for display.
        max_characters: Maximum number of source characters to inspect.

    Returns:
        A bounded string with terminal-invalid characters replaced by ``?``.

    Raises:
        TypeError: If ``max_characters`` is not an integer.
        ValueError: If ``max_characters`` is negative.
    """
    if isinstance(max_characters, bool) or not isinstance(max_characters, int):
        raise TypeError("max_characters must be an integer")
    if max_characters < 0:
        raise ValueError("max_characters must be nonnegative")

    text = _deterministic_text(value)[:max_characters]
    result: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        invalid = character not in _PRESERVED_CONTROLS and (
            character == "\ufffd"
            or category in {"Cc", "Cf", "Cs"}
            or wcwidth.wcwidth(character) < 0
        )
        result.append("?" if invalid else character)
    return "".join(result)


__all__ = ["sanitize_character_display_text"]
