"""Strict profile-local preferences for the opt-in Persona Buddy."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

_PERSONA_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_POSITION = 1_000_000
_MAX_DIMENSION = 10_000


def save_settings_to_cli_config(
    section_values: Mapping[str, Mapping[Any, Any]],
) -> bool:
    """Call the incumbent writer without loading global config during parsing."""

    from tldw_chatbook.config import save_settings_to_cli_config as save

    return save(section_values)


class PersonaBuddyPreferenceError(ValueError):
    """A bounded, path-free Persona Buddy preference error."""

    __slots__ = ("category",)

    def __init__(self, category: str = "persona_buddy_preferences_invalid") -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class PersonaBuddySelection:
    """One exact profile-local Persona selection."""

    source: Literal["local"]
    local_persona_id: str

    def __post_init__(self) -> None:
        if self.source != "local" or not _valid_persona_id(self.local_persona_id):
            raise PersonaBuddyPreferenceError()


@dataclass(frozen=True, slots=True)
class PersonaBuddyGeometry:
    """Persisted floating-window geometry before viewport clamping."""

    x: int = 0
    y: int = 0
    width: int = 28
    height: int = 12

    def __post_init__(self) -> None:
        if not _valid_position(self.x) or not _valid_position(self.y):
            raise PersonaBuddyPreferenceError("persona_buddy_geometry_invalid")
        if not _valid_dimension(self.width) or not _valid_dimension(self.height):
            raise PersonaBuddyPreferenceError("persona_buddy_geometry_invalid")


@dataclass(frozen=True, slots=True)
class PersonaBuddyPreferences:
    """One immutable snapshot of Buddy UI and selection preferences."""

    enabled: bool = False
    selection: PersonaBuddySelection | None = None
    open: bool = True
    collapsed: bool = False
    geometry: PersonaBuddyGeometry = field(default_factory=PersonaBuddyGeometry)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise PersonaBuddyPreferenceError()
        if (
            self.selection is not None
            and type(self.selection) is not PersonaBuddySelection
        ):
            raise PersonaBuddyPreferenceError()
        if type(self.open) is not bool or type(self.collapsed) is not bool:
            raise PersonaBuddyPreferenceError()
        if type(self.geometry) is not PersonaBuddyGeometry:
            raise PersonaBuddyPreferenceError()


def _valid_persona_id(value: object) -> bool:
    return type(value) is str and _PERSONA_ID_PATTERN.fullmatch(value) is not None


def _valid_position(value: object) -> bool:
    return type(value) is int and 0 <= value <= _MAX_POSITION


def _valid_dimension(value: object) -> bool:
    return type(value) is int and 1 <= value <= _MAX_DIMENSION


def _boolean(value: object, default: bool) -> bool:
    return value if type(value) is bool else default


def _position(value: object, default: int) -> int:
    return value if _valid_position(value) else default


def _dimension(value: object, default: int) -> int:
    return value if _valid_dimension(value) else default


def parse_persona_buddy_preferences(
    section: Mapping[str, object],
) -> PersonaBuddyPreferences:
    """Parse one config section with independent safe field fallbacks.

    Args:
        section: Raw ``[persona_buddy]`` configuration values.

    Returns:
        A strict immutable preference snapshot.
    """

    if not isinstance(section, Mapping):
        return PersonaBuddyPreferences()

    defaults = PersonaBuddyPreferences()
    source = section.get("source")
    persona_id = section.get("local_persona_id")
    selection = None
    if source == "local" and _valid_persona_id(persona_id):
        selection = PersonaBuddySelection("local", persona_id)

    default_geometry = defaults.geometry
    geometry = PersonaBuddyGeometry(
        x=_position(section.get("x"), default_geometry.x),
        y=_position(section.get("y"), default_geometry.y),
        width=_dimension(section.get("width"), default_geometry.width),
        height=_dimension(section.get("height"), default_geometry.height),
    )
    return PersonaBuddyPreferences(
        enabled=_boolean(section.get("enabled"), defaults.enabled),
        selection=selection,
        open=_boolean(section.get("open"), defaults.open),
        collapsed=_boolean(section.get("collapsed"), defaults.collapsed),
        geometry=geometry,
    )


def serialize_persona_buddy_preferences(
    preferences: PersonaBuddyPreferences,
) -> dict[str, object]:
    """Serialize an exact preference snapshot for the incumbent config writer."""

    if type(preferences) is not PersonaBuddyPreferences:
        raise PersonaBuddyPreferenceError()
    selection = preferences.selection
    return {
        "enabled": preferences.enabled,
        "source": selection.source if selection is not None else "",
        "local_persona_id": (
            selection.local_persona_id if selection is not None else ""
        ),
        "open": preferences.open,
        "collapsed": preferences.collapsed,
        "x": preferences.geometry.x,
        "y": preferences.geometry.y,
        "width": preferences.geometry.width,
        "height": preferences.geometry.height,
    }


def persist_persona_buddy_preferences(
    preferences: PersonaBuddyPreferences,
) -> bool:
    """Persist one complete Buddy section through the incumbent config writer."""

    try:
        return save_settings_to_cli_config(
            {"persona_buddy": serialize_persona_buddy_preferences(preferences)}
        )
    except Exception:
        logger.error("persona_buddy_preferences_save_failed")
        return False
