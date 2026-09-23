"""Pure Home triage rail preference contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Utils.Utils import coerce_bool_flag

HOME_RAIL_SECTION_IDS = ("attention", "running", "recent", "details")



@dataclass(frozen=True)
class HomeRailPreferences:
    """Persisted open/collapsed preferences for Home triage rail sections."""

    attention_open: bool = True
    running_open: bool = True
    recent_open: bool = True
    details_open: bool = False


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
        attention_open=coerce_bool_flag(raw.get("attention_open"), defaults.attention_open),
        running_open=coerce_bool_flag(raw.get("running_open"), defaults.running_open),
        recent_open=coerce_bool_flag(raw.get("recent_open"), defaults.recent_open),
        details_open=coerce_bool_flag(raw.get("details_open"), defaults.details_open),
    )


def serialize_home_rail_preferences(
    preferences: HomeRailPreferences,
) -> dict[str, bool]:
    """Serialize Home rail preferences to the persistence shape."""
    return {
        "attention_open": bool(preferences.attention_open),
        "running_open": bool(preferences.running_open),
        "recent_open": bool(preferences.recent_open),
        "details_open": bool(preferences.details_open),
    }
