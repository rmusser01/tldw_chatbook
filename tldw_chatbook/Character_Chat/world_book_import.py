"""Pure normalization/validation for world-book (Lore) import files.

Maps the supported input shapes — tldw's own export, the character-book array
form, and SillyTavern 'World Info' object-form — to tldw's ``world_book_entries``
field names, and validates the whole file up front so the import screen can
reject a bad file before any DB write (``WorldBookManager.import_world_book`` is
not atomic). Pure and DB-free; raises ``ValueError`` with a user-facing message.
"""
from __future__ import annotations

from typing import Any, Dict, List

_VALID_POSITIONS = {"before_char", "after_char", "at_start", "at_end"}
_INT_POSITION_MAP = {0: "before_char", 1: "after_char"}


def _coerce_int(value: Any, default: int) -> int:
    """Best-effort int coercion; ``default`` on None/non-numeric."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_TRUE_STRINGS = {"true", "1", "yes", "on"}
_FALSE_STRINGS = {"false", "0", "no", "off"}


def _coerce_bool(value: Any, default: bool) -> bool:
    """Best-effort bool coercion for loosely-typed/hand-edited import files.

    Plain ``bool(value)`` is wrong here: ``bool("false")`` is ``True`` and
    ``bool(None)`` is ``False``, so a string boolean or an explicit null would
    silently change an entry's matching behavior. ``None`` returns ``default``;
    recognized string booleans map by value; unknown strings return ``default``.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_STRINGS:
            return True
        if token in _FALSE_STRINGS:
            return False
        return default
    return default


def _as_str_list(value: Any) -> List[str]:
    """Coerce a keys-like field to a list of non-blank strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        # Drop None (and blank) items BEFORE stringifying — otherwise a null in a
        # keys array becomes the literal "None" (a phantom keyword), and a list of
        # only nulls would slip past the empty-keys validation as ["None"].
        return [str(v) for v in value if v is not None and str(v).strip()]
    return []


def _normalize_position(pos: Any) -> str:
    """Map a position (tldw string or SillyTavern int) to a tldw position."""
    if isinstance(pos, str) and pos in _VALID_POSITIONS:
        return pos
    if isinstance(pos, bool):
        return "before_char"
    if isinstance(pos, int):
        return _INT_POSITION_MAP.get(pos, "before_char")
    return "before_char"


def _normalize_entry(entry: Any, index: int) -> Dict[str, Any]:
    """Map one entry to tldw fields; raise ValueError (1-based index) if invalid."""
    if not isinstance(entry, dict):
        raise ValueError(f"Entry {index + 1} is not an object.")
    raw_keys = entry.get("keys")
    if raw_keys is None:
        raw_keys = entry.get("key")
    keys = _as_str_list(raw_keys)
    if not keys:
        raise ValueError(f"Entry {index + 1} has no keys.")
    # Explicit null content must be treated as empty (→ ValueError), not
    # stringified to the literal "None" (the same null-stringify trap as keys).
    raw_content = entry.get("content")
    content = str(raw_content) if raw_content is not None else ""
    if not content.strip():
        raise ValueError(f"Entry {index + 1} has no content.")
    raw_secondary = entry.get("secondary_keys")
    if raw_secondary is None:
        raw_secondary = entry.get("keysecondary")
    raw_case = entry.get("case_sensitive")
    if raw_case is None:
        raw_case = entry.get("caseSensitive")
    # enabled: an explicit value wins (null → default enabled); otherwise fall
    # back to the (inverted) SillyTavern ``disable`` flag.
    raw_enabled = entry.get("enabled")
    if raw_enabled is not None:
        enabled = _coerce_bool(raw_enabled, True)
    else:
        enabled = not _coerce_bool(entry.get("disable"), False)
    extensions = entry.get("extensions")
    return {
        "keys": keys,
        "secondary_keys": _as_str_list(raw_secondary),
        "content": content,
        "insertion_order": _coerce_int(
            entry.get("insertion_order", entry.get("order", index)), index
        ),
        "position": _normalize_position(entry.get("position")),
        "selective": _coerce_bool(entry.get("selective"), False),
        "case_sensitive": _coerce_bool(raw_case, False),
        "enabled": enabled,
        "priority": _coerce_int(entry.get("priority", 0), 0),
        "extensions": extensions if isinstance(extensions, dict) else {},
    }


def normalize_world_book_import(data: Any) -> Dict[str, Any]:
    """Normalize + validate an imported world-book payload.

    Args:
        data: The parsed JSON from an import file.

    Returns:
        A dict with tldw metadata keys preserved and ``entries`` mapped to
        tldw's field names as a list, ready for ``import_world_book``.

    Raises:
        ValueError: If the payload is not a dict, ``entries`` is neither a list
            nor an object, or any entry is not a dict / has no keys / no content.
    """
    if not isinstance(data, dict):
        raise ValueError("World book file must be a JSON object.")
    raw_entries = data.get("entries")
    if raw_entries is None:
        entries_list: List[Any] = []
    elif isinstance(raw_entries, dict):
        entries_list = list(raw_entries.values())
    elif isinstance(raw_entries, list):
        entries_list = raw_entries
    else:
        raise ValueError("'entries' must be a list or an object.")
    normalized = [_normalize_entry(entry, i) for i, entry in enumerate(entries_list)]
    return {**data, "entries": normalized}
