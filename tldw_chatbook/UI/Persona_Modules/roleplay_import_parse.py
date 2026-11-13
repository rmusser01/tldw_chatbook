"""Roleplay's lore-book and chat-dictionary import parsing, off the event loop.

TASK-33910.44. The screen awaits one function here per format with
``asyncio.to_thread``: the file's read, its ``json.loads`` and its
normalisation run in one worker thread, and only the result returns to the
event loop. On dev the read ran in a thread but the parse and the world-book
normalisation ran on the event loop (about 41 ms for a 9 MB lore book on a
quiet machine, more under load); this module moves them.

``json.loads`` is C code that holds the GIL for the whole parse, so a thread
alone frees the loop from the Python steps (the normalisation, the dictionary
unwrap) but not from the parse: a 9 MB parse in a thread still paused the loop
for its full length (32-63 ms under load here). The parse therefore passes a
Python-level ``object_hook`` that returns each object unchanged: running its
bytecode after every JSON object lets the interpreter hand the GIL to the
waiting event loop at the switch interval (``sys.getswitchinterval()``), which
cut the loop's longest gap during a threaded 9 MB parse to 14-27 ms in the same
runs. The ADR-011 heartbeat capture judges this, not "it runs in a thread".

The screen imports this module lazily, inside its import functions, so the
Roleplay route pre-imports nothing new (spec 5.10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tldw_chatbook.Character_Chat.world_book_import import normalize_world_book_import


class ImportNotJSON(Exception):
    """The file's text is not valid JSON; ``str()`` is the parser's own reason."""


def _hand_back_the_gil(obj: dict[str, Any]) -> dict[str, Any]:
    """``json.loads``'s object hook: each object, unchanged (see the module doc)."""
    return obj


def _parse_json(text: str) -> Any:
    """Parse ``text``, reporting any parser refusal as ``ImportNotJSON``."""
    try:
        return json.loads(text, object_hook=_hand_back_the_gil)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ImportNotJSON(str(exc)) from exc


def read_world_book_import(source: Path) -> dict[str, Any]:
    """Read, parse and normalise one lore-book file (run in a worker thread).

    Args:
        source: The validated, size-checked file.

    Returns:
        The normalised world book, ready for ``import_world_book``.

    Raises:
        OSError: The file could not be read.
        UnicodeDecodeError: Its bytes are not UTF-8.
        ImportNotJSON: Its text is not valid JSON.
        ValueError: ``normalize_world_book_import`` refused its shape.
    """
    return normalize_world_book_import(_parse_json(source.read_text("utf-8")))


def read_dictionary_import(source: Path) -> dict[str, Any] | str:
    """Read one chat-dictionary file (run in a worker thread); parse ``.json``.

    Args:
        source: The validated, size-checked file.

    Returns:
        For a ``.json`` file, its dictionary object (the ``data`` object when
        the file carries one) as a new dict. For any other file, its text,
        unparsed: the screen sends ``.md`` text to the Markdown importer and
        refuses other suffixes after the read, as dev did.

    Raises:
        OSError: The file could not be read.
        UnicodeDecodeError: Its bytes are not UTF-8.
        ImportNotJSON: A ``.json`` file's text is not valid JSON.
        ValueError: A ``.json`` file holds no dictionary object.
    """
    text = source.read_text("utf-8")
    if source.suffix.lower() != ".json":
        return text
    parsed = _parse_json(text)
    raw = (
        parsed.get("data") if isinstance(parsed, dict) and "data" in parsed else parsed
    )
    if not isinstance(raw, dict):
        raise ValueError("JSON must be a dictionary object.")
    return dict(raw)


__all__ = ["ImportNotJSON", "read_dictionary_import", "read_world_book_import"]
