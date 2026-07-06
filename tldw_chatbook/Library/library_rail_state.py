"""Pure Library rail preference contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LIBRARY_RAIL_SECTION_IDS = ("browse", "create", "ingest", "details")

_TRUE_STRINGS = {"true", "yes", "1", "on"}
_FALSE_STRINGS = {"false", "no", "0", "off"}


@dataclass(frozen=True)
class LibraryRailPreferences:
    """Persisted open/collapsed preferences for Library rail sections."""

    browse_open: bool = True
    create_open: bool = True
    ingest_open: bool = True
    details_open: bool = False


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return fallback


def coerce_library_rail_preferences(raw: Any) -> LibraryRailPreferences:
    """Normalize stored Library rail preferences.

    Args:
        raw: Dict-like stored value from ``library.rail_state``.

    Returns:
        Preferences with invalid or missing fields replaced by defaults.
    """
    defaults = LibraryRailPreferences()
    if not isinstance(raw, dict):
        return defaults
    return LibraryRailPreferences(
        browse_open=_coerce_bool(raw.get("browse_open"), defaults.browse_open),
        create_open=_coerce_bool(raw.get("create_open"), defaults.create_open),
        ingest_open=_coerce_bool(raw.get("ingest_open"), defaults.ingest_open),
        details_open=_coerce_bool(raw.get("details_open"), defaults.details_open),
    )


def serialize_library_rail_preferences(preferences: LibraryRailPreferences) -> dict[str, bool]:
    """Serialize Library rail preferences to the persistence shape.

    Args:
        preferences: Rail open/collapsed state to persist.

    Returns:
        Dict of boolean ``*_open`` flags suitable for ``library.rail_state``.
    """
    return {
        "browse_open": bool(preferences.browse_open),
        "create_open": bool(preferences.create_open),
        "ingest_open": bool(preferences.ingest_open),
        "details_open": bool(preferences.details_open),
    }
