"""Persisted Persona Buddy settings must reach app-owned runtime state."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

import tldw_chatbook.config as app_config
from tldw_chatbook.Persona_Buddy import (
    PersonaBuddyController,
    PersonaBuddyGeometry,
    PersonaBuddyPreferences,
    PersonaBuddySelection,
    parse_persona_buddy_preferences,
)


_BUDDY_TOML = """\
[persona_buddy]
enabled = true
source = "local"
local_persona_id = "persona-uat"
open = true
collapsed = true
x = 7
y = 5
width = 31
height = 14

[private_probe]
token = "must-not-project"
"""


@contextmanager
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str):
    cache_state = {
        "config_cache": app_config._CONFIG_CACHE,
        "config_cache_source": app_config._CONFIG_CACHE_SOURCE,
        "settings_cache": app_config._SETTINGS_CACHE,
        "settings_cache_source": app_config._SETTINGS_CACHE_SOURCE,
        "settings": app_config.settings,
        "config_generation": app_config._CONFIG_GENERATION,
        "last_failure": app_config._LAST_CONFIG_LOAD_FAILURE,
    }
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    original_path = os.environ.get("TLDW_CONFIG_PATH")
    try:
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
        app_config._CONFIG_CACHE = None
        app_config._CONFIG_CACHE_SOURCE = None
        app_config._SETTINGS_CACHE = None
        app_config._SETTINGS_CACHE_SOURCE = None
        app_config._LAST_CONFIG_LOAD_FAILURE = None
        yield config_path
    finally:
        if original_path is None:
            monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
        else:
            monkeypatch.setenv("TLDW_CONFIG_PATH", original_path)
        app_config._CONFIG_CACHE = cache_state["config_cache"]
        app_config._CONFIG_CACHE_SOURCE = cache_state["config_cache_source"]
        app_config._SETTINGS_CACHE = cache_state["settings_cache"]
        app_config._SETTINGS_CACHE_SOURCE = cache_state["settings_cache_source"]
        app_config.settings = cache_state["settings"]
        app_config._CONFIG_GENERATION = cache_state["config_generation"]
        app_config._LAST_CONFIG_LOAD_FAILURE = cache_state["last_failure"]


def _load_preferences() -> PersonaBuddyPreferences:
    settings = app_config.load_settings(force_reload=True)
    return parse_persona_buddy_preferences(settings["persona_buddy"])


def test_real_toml_persona_buddy_table_reaches_controller_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _scratch_config(tmp_path, monkeypatch, _BUDDY_TOML) as config_path:
        before = config_path.read_bytes()
        controller = PersonaBuddyController(preferences=_load_preferences())

        assert controller.current_preferences() == PersonaBuddyPreferences(
            enabled=True,
            selection=PersonaBuddySelection("local", "persona-uat"),
            open=True,
            collapsed=True,
            geometry=PersonaBuddyGeometry(7, 5, 31, 14),
        )
        assert config_path.read_bytes() == before


def test_projected_malformed_fields_keep_independent_safe_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = _BUDDY_TOML.replace("enabled = true", 'enabled = "yes"').replace(
        "x = 7", "x = true"
    )
    with _scratch_config(tmp_path, monkeypatch, malformed):
        preferences = _load_preferences()

    assert preferences.enabled is False
    assert preferences.selection == PersonaBuddySelection("local", "persona-uat")
    assert preferences.open is True
    assert preferences.collapsed is True
    assert preferences.geometry == PersonaBuddyGeometry(1_000_000, 5, 31, 14)


@pytest.mark.asyncio
async def test_first_persist_after_restart_preserves_loaded_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[PersonaBuddyPreferences] = []

    def writer(preferences: PersonaBuddyPreferences) -> bool:
        saved.append(preferences)
        return True

    with _scratch_config(tmp_path, monkeypatch, _BUDDY_TOML):
        controller = PersonaBuddyController(
            preferences=_load_preferences(),
            preference_writer=writer,
        )
        updated = replace(controller.current_preferences(), open=False)

        snapshot = await controller.update_preferences(updated)
        await controller.shutdown()

    assert snapshot.open is False
    assert saved == [updated]
    assert saved[0].geometry == PersonaBuddyGeometry(7, 5, 31, 14)


def test_projection_adds_only_the_buddy_top_level_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _scratch_config(tmp_path, monkeypatch, _BUDDY_TOML):
        settings = app_config.load_settings(force_reload=True)

    assert "persona_buddy" in settings
    assert "private_probe" not in settings
    assert settings["persona_buddy"] == {
        "enabled": True,
        "source": "local",
        "local_persona_id": "persona-uat",
        "open": True,
        "collapsed": True,
        "x": 7,
        "y": 5,
        "width": 31,
        "height": 14,
    }
