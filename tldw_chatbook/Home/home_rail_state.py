"""Pure Home triage rail preference contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HOME_RAIL_SECTION_IDS = ("attention", "running", "recent", "details")

_TRUE_STRINGS = {"true", "yes", "1", "on"}
_FALSE_STRINGS = {"false", "no", "0", "off"}


@dataclass(frozen=True)
class HomeRailPreferences:
    """Persisted open/collapsed preferences for Home triage rail sections."""

    attention_open: bool = True
    running_open: bool = True
    recent_open: bool = True
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


def coerce_home_rail_preferences(raw: Any) -> HomeRailPreferences:
    """Normalize stored Home rail preferences.

    Args:
        raw: Dict-like stored value from ``home.rail_state``.

    Returns:
        Preferences with invalid or missing fields replaced by defaults.
    """
    defaults = HomeRailPreferences()
    if not isinstance(raw, dict):
        return defaults
    return HomeRailPreferences(
        attention_open=_coerce_bool(raw.get("attention_open"), defaults.attention_open),
        running_open=_coerce_bool(raw.get("running_open"), defaults.running_open),
        recent_open=_coerce_bool(raw.get("recent_open"), defaults.recent_open),
        details_open=_coerce_bool(raw.get("details_open"), defaults.details_open),
    )


def serialize_home_rail_preferences(preferences: HomeRailPreferences) -> dict[str, bool]:
    """Serialize Home rail preferences to the persistence shape."""
    return {
        "attention_open": bool(preferences.attention_open),
        "running_open": bool(preferences.running_open),
        "recent_open": bool(preferences.recent_open),
        "details_open": bool(preferences.details_open),
    }
