"""Strict, profile-local Persona Buddy preference contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from loguru import logger

from tldw_chatbook.Persona_Buddy.preferences import (
    PersonaBuddyGeometry,
    PersonaBuddyPreferences,
    PersonaBuddySelection,
    parse_persona_buddy_preferences,
    persist_persona_buddy_preferences,
    serialize_persona_buddy_preferences,
)


def test_preferences_default_off_without_selection() -> None:
    preferences = parse_persona_buddy_preferences({})

    assert preferences == PersonaBuddyPreferences()
    assert preferences.enabled is False
    assert preferences.selection is None
    assert preferences.open is True
    assert preferences.collapsed is False
    assert not hasattr(preferences, "__dict__")

    with pytest.raises(FrozenInstanceError):
        preferences.enabled = True  # type: ignore[misc]


def test_preferences_round_trip_exact_local_selection_and_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        "enabled": True,
        "source": "local",
        "local_persona_id": "p-1",
        "open": False,
        "collapsed": True,
        "x": 9,
        "y": 4,
        "width": 28,
        "height": 12,
    }

    preferences = parse_persona_buddy_preferences(raw)

    assert preferences.selection == PersonaBuddySelection("local", "p-1")
    assert preferences.geometry == PersonaBuddyGeometry(9, 4, 28, 12)
    assert serialize_persona_buddy_preferences(preferences) == raw
    assert (
        parse_persona_buddy_preferences(
            serialize_persona_buddy_preferences(preferences)
        )
        == preferences
    )

    saved: list[dict[str, object]] = []

    def save(section_values: dict[str, object]) -> bool:
        saved.append(section_values)
        return True

    monkeypatch.setattr(
        "tldw_chatbook.Persona_Buddy.preferences.save_settings_to_cli_config",
        save,
    )

    assert persist_persona_buddy_preferences(preferences) is True
    assert saved == [{"persona_buddy": raw}]


@pytest.mark.parametrize(
    ("field", "malformed", "expected_attribute", "expected_value"),
    (
        ("enabled", "yes", "enabled", False),
        ("open", 1, "open", True),
        ("collapsed", None, "collapsed", False),
        ("x", True, "geometry", PersonaBuddyGeometry(0, 4, 36, 16)),
        ("y", -1, "geometry", PersonaBuddyGeometry(9, 0, 36, 16)),
        ("width", False, "geometry", PersonaBuddyGeometry(9, 4, 28, 16)),
        ("height", 0, "geometry", PersonaBuddyGeometry(9, 4, 36, 12)),
    ),
)
def test_preferences_reject_malformed_fields_independently(
    field: str,
    malformed: object,
    expected_attribute: str,
    expected_value: object,
) -> None:
    valid: dict[str, object] = {
        "enabled": True,
        "source": "local",
        "local_persona_id": "p-1",
        "open": False,
        "collapsed": True,
        "x": 9,
        "y": 4,
        "width": 36,
        "height": 16,
    }
    valid[field] = malformed

    preferences = parse_persona_buddy_preferences(valid)

    assert getattr(preferences, expected_attribute) == expected_value
    for unaffected_field, expected in {
        "enabled": True,
        "open": False,
        "collapsed": True,
    }.items():
        if field != unaffected_field:
            assert getattr(preferences, unaffected_field) is expected


@pytest.mark.parametrize(
    ("source", "persona_id"),
    (
        ("server", "p-1"),
        ("local", ""),
        ("local", "contains\x00control"),
        ("local", "x" * 129),
        ("local", "/private/profile/persona.json"),
    ),
)
def test_preference_failure_is_path_free(
    source: str,
    persona_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferences = parse_persona_buddy_preferences(
        {
            "enabled": True,
            "source": source,
            "local_persona_id": persona_id,
        }
    )

    assert preferences.selection is None
    if persona_id:
        assert persona_id not in repr(preferences)

    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:

        def fail(_section_values: object) -> bool:
            raise RuntimeError("/private/profile/config.toml")

        monkeypatch.setattr(
            "tldw_chatbook.Persona_Buddy.preferences.save_settings_to_cli_config",
            fail,
        )
        assert persist_persona_buddy_preferences(preferences) is False
    finally:
        logger.remove(sink)

    rendered = "".join(messages)
    assert rendered.strip() == "persona_buddy_preferences_save_failed"
    assert "/private/" not in rendered


def test_preferences_require_exact_string_local_source() -> None:
    class SpoofedLocal:
        def __eq__(self, other: object) -> bool:
            return other == "local"

        def __repr__(self) -> str:
            return "/private/profile/spoofed-source"

    source = SpoofedLocal()

    preferences = parse_persona_buddy_preferences(
        {"source": source, "local_persona_id": "p-1"}
    )

    assert preferences.selection is None
    with pytest.raises(ValueError, match="^persona_buddy_preferences_invalid$"):
        PersonaBuddySelection(source, "p-1")  # type: ignore[arg-type]
