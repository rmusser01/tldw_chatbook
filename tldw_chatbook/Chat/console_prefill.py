"""Pure parsing/validation for the native Console ``/prefill`` command.

No dependency on Textual, the running app, or any I/O — mirrors
``console_command_grammar.py``. The screen layer owns all UI wiring; the
controller/store layers own arming state and payload assembly. This module
only classifies the ``/prefill`` args string and reads the pinned-prefill
key out of a raw conversation ``metadata`` JSON string.

Normalization: prefill text is ``.strip()``-ed at parse time. The spec
requires right-trimming (Anthropic rejects a trailing-whitespace assistant
turn); leading whitespace is stripped too for predictability, matching how
``/system`` name args are treated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

PREFILL_MAX_CHARS = 4000
"""Maximum accepted prefill length; longer input is rejected at arm time."""

PINNED_PREFILL_METADATA_KEY = "pinned_response_prefill"
"""Key inside ``conversations.metadata`` JSON holding the pinned prefill."""

ACTION_STATUS = "status"
ACTION_CLEAR = "clear"
ACTION_PIN = "pin"
ACTION_ONE_SHOT = "one-shot"
ACTION_ERROR = "error"

_PIN_SUBCOMMAND = "pin"
_CLEAR_SUBCOMMAND = "clear"
_PREVIEW_MAX_CHARS = 60


@dataclass(frozen=True)
class PrefillCommandAction:
    """One classified ``/prefill`` invocation.

    Args:
        kind: One of the ``ACTION_*`` constants.
        text: Normalized prefill text for ``pin``/``one-shot`` kinds.
        error: Human-readable message for the ``error`` kind.
    """

    kind: str
    text: str = ""
    error: str = ""


def _validated(kind: str, text: str) -> PrefillCommandAction:
    if not text:
        return PrefillCommandAction(
            kind=ACTION_ERROR, error="Prefill text is empty."
        )
    if len(text) > PREFILL_MAX_CHARS:
        return PrefillCommandAction(
            kind=ACTION_ERROR,
            error=f"Prefill is too long ({len(text)} chars; max {PREFILL_MAX_CHARS}).",
        )
    return PrefillCommandAction(kind=kind, text=text)


def parse_prefill_args(args: str) -> PrefillCommandAction:
    """Classify the args string of one ``/prefill`` invocation.

    ``clear`` matches only as the entire (stripped) args; ``pin`` matches
    only as the first token with trailing text. A one-shot whose text
    literally starts with ``pin `` or equals ``clear`` therefore cannot be
    expressed — documented spec limitation.

    Args:
        args: Raw text after the ``/prefill`` command word.

    Returns:
        A `PrefillCommandAction` whose ``kind`` is one of the ``ACTION_*``
        constants; ``text`` carries the normalized prefill for
        ``pin``/``one-shot`` kinds, ``error`` the message for ``error``.
    """
    stripped = args.strip()
    if not stripped:
        return PrefillCommandAction(kind=ACTION_STATUS)
    if stripped.lower() == _CLEAR_SUBCOMMAND:
        return PrefillCommandAction(kind=ACTION_CLEAR)
    first, _, remainder = stripped.partition(" ")
    if first.lower() == _PIN_SUBCOMMAND:
        pin_text = remainder.strip()
        if not pin_text:
            return PrefillCommandAction(
                kind=ACTION_ERROR, error="Usage: /prefill pin <text>."
            )
        return _validated(ACTION_PIN, pin_text)
    return _validated(ACTION_ONE_SHOT, stripped)


def describe_prefill_preview(text: str, max_chars: int = _PREVIEW_MAX_CHARS) -> str:
    """Return a single-line preview of ``text``, truncated with an ellipsis.

    Args:
        text: Full prefill text to summarize.
        max_chars: Maximum preview length, including the ellipsis.

    Returns:
        ``text`` with whitespace runs collapsed to single spaces, cut to
        ``max_chars`` with a trailing ``…`` when longer.
    """
    flattened = " ".join(text.split())
    if len(flattened) <= max_chars:
        return flattened
    return flattened[: max_chars - 1] + "…"


def pinned_prefill_from_conversation_metadata(raw_metadata: object) -> str | None:
    """Read the pinned prefill out of a raw conversation ``metadata`` value.

    Mirrors ``local_chat_dictionary_service._active_dictionaries``'s guarded
    parse (the ``json.loads(None)`` crash class): any missing/invalid shape
    yields ``None`` rather than raising.

    Args:
        raw_metadata: The conversation record's ``metadata`` column value —
            a JSON string, ``None``, or any invalid shape.

    Returns:
        The stored non-blank pinned prefill string, or ``None`` when the
        metadata is missing, malformed, or has no usable value under
        `PINNED_PREFILL_METADATA_KEY`.
    """
    try:
        meta = json.loads(raw_metadata or "{}")
    except (TypeError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    value = meta.get(PINNED_PREFILL_METADATA_KEY)
    if not isinstance(value, str) or not value.strip():
        return None
    return value
