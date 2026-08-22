"""Strict, profile-local Persona Buddy preference contracts."""

from __future__ import annotations

import tomllib
from dataclasses import FrozenInstanceError
from pathlib import Path

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
    assert preferences.geometry == PersonaBuddyGeometry(1_000_000, 1_000_000, 28, 12)
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
        ("x", True, "geometry", PersonaBuddyGeometry(1_000_000, 4, 36, 16)),
        ("y", -1, "geometry", PersonaBuddyGeometry(9, 1_000_000, 36, 16)),
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
    assert (
        rendered.strip()
        == "persona_buddy_preferences_save_failed exception_type=RuntimeError"
    )
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


def test_never_positioned_geometry_is_distinct_from_persisted_top_left() -> None:
    from tldw_chatbook.Persona_Buddy import preferences as preference_module

    sentinel = preference_module.PERSONA_BUDDY_UNPOSITIONED_COORDINATE
    defaults = parse_persona_buddy_preferences({})
    top_left = parse_persona_buddy_preferences({"x": 0, "y": 0})

    assert sentinel == 1_000_000
    assert defaults.geometry == PersonaBuddyGeometry(sentinel, sentinel, 28, 12)
    assert serialize_persona_buddy_preferences(defaults)["x"] == sentinel
    assert serialize_persona_buddy_preferences(defaults)["y"] == sentinel
    assert top_left.geometry == PersonaBuddyGeometry(0, 0, 28, 12)
    assert (
        parse_persona_buddy_preferences(
            serialize_persona_buddy_preferences(top_left)
        ).geometry
        == top_left.geometry
    )


def test_preferences_persist_through_real_incumbent_config_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "isolated" / "config.toml"
    config_path.parent.mkdir(mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    preferences = parse_persona_buddy_preferences(
        {
            "enabled": True,
            "source": "local",
            "local_persona_id": "p-real",
            "x": 0,
            "y": 0,
        }
    )

    assert persist_persona_buddy_preferences(preferences) is True

    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))["persona_buddy"]
    assert saved == serialize_persona_buddy_preferences(preferences)


@pytest.mark.parametrize(
    ("value", "field_name", "replacement"),
    (
        (PersonaBuddySelection("local", "p-1"), "local_persona_id", "p-2"),
        (PersonaBuddyGeometry(1, 2, 28, 12), "x", 3),
        (PersonaBuddyPreferences(), "enabled", True),
    ),
)
def test_preference_public_contracts_are_exactly_frozen_and_slotted(
    value: object,
    field_name: str,
    replacement: object,
) -> None:
    assert type(value).__dataclass_params__.frozen is True
    assert "__slots__" in vars(type(value))
    assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(value, field_name, replacement)
