"""Bound and sanitize untrusted character text for terminal display only."""

from __future__ import annotations

import unicodedata

import wcwidth

_PRESERVED_CONTROLS = frozenset({"\n", "\t"})
_SAFE_SCALAR_TYPES = frozenset({bool, int, float})


def _type_placeholder(value: object, max_characters: int) -> str:
    """Return a bounded type name without invoking instance conversion hooks."""
    if max_characters == 0:
        return ""
    type_name = type.__getattribute__(type(value), "__name__")
    if type(type_name) is not str:
        type_name = "object"
    if max_characters == 1:
        return "<"
    if max_characters == 2:
        return "<>"
    return f"<{type_name[: max_characters - 2]}>"


def _deterministic_text(value: object, max_characters: int) -> str:
    """Return bounded text without invoking arbitrary object behavior."""
    if value is None:
        return ""
    if type(value) is str:
        return value[:max_characters]
    if type(value) in _SAFE_SCALAR_TYPES:
        try:
            return str(value)[:max_characters]
        except (ValueError, OverflowError):
            return _type_placeholder(value, max_characters)
    return _type_placeholder(value, max_characters)


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

    text = _deterministic_text(value, max_characters)
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


def sanitize_character_display_label(
    value: object,
    *,
    max_characters: int,
) -> str:
    """Project a value into one bounded terminal-safe display line."""
    display_text = sanitize_character_display_text(
        value,
        max_characters=max_characters,
    )
    return " ".join(display_text.split())


def sanitize_character_display_items(
    value: object,
    *,
    max_items: int,
    max_item_characters: int,
    max_total_characters: int,
    single_line: bool = False,
) -> tuple[str, ...]:
    """Return a bounded display-only projection of a loose collection field.

    Only exact built-in lists and tuples are traversed. Strings and safe
    scalars are treated as one item; every other shape receives a type
    placeholder so custom iterators and mapping contents are never executed.
    """
    for name, limit in (
        ("max_items", max_items),
        ("max_item_characters", max_item_characters),
        ("max_total_characters", max_total_characters),
    ):
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError(f"{name} must be an integer")
        if limit < 0:
            raise ValueError(f"{name} must be nonnegative")
    if type(single_line) is not bool:
        raise TypeError("single_line must be a boolean")
    if max_items == 0 or max_total_characters == 0 or value is None:
        return ()

    if type(value) in {list, tuple}:
        source_items = value[:max_items]
    else:
        source_items = (value,)

    result: list[str] = []
    remaining = max_total_characters
    sanitizer = (
        sanitize_character_display_label
        if single_line
        else sanitize_character_display_text
    )
    for item in source_items:
        if len(result) >= max_items or remaining <= 0:
            break
        item_limit = min(max_item_characters, remaining)
        display_item = sanitizer(item, max_characters=item_limit)
        if not display_item:
            continue
        result.append(display_item)
        remaining -= len(display_item)
    return tuple(result)


__all__ = [
    "sanitize_character_display_items",
    "sanitize_character_display_label",
    "sanitize_character_display_text",
]
