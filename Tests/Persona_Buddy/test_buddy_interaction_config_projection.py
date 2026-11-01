"""Persisted interaction choices must survive the production settings loader."""

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["conversation", "workspace"])
@private_profile_test
async def test_saved_buddy_binding_and_static_mode_reach_loaded_settings(
    request: pytest.FixtureRequest, kind: str
):
    from tldw_chatbook.config import (
        load_cli_config_and_ensure_existence,
        load_settings,
        save_settings_to_cli_config,
    )
    from tldw_chatbook.Persona_Buddy.interaction import parse_preferences

    raw = {
        "kind": kind,
        "target_id": "saved-owner",
        "animated": False,
        "speak_responses": False,
    }
    if kind == "conversation":
        raw.update(conversation_id="original-conversation", binding_revision=1)
    assert save_settings_to_cli_config({"buddy_interaction": raw})
    settings = load_settings(force_reload=True)
    assert settings["buddy_interaction"] == raw
    preferences = parse_preferences(settings["buddy_interaction"])
    assert preferences.binding.kind == kind
    assert preferences.binding.target_id == "saved-owner"
    assert preferences.animated is False
    assert preferences.speak_responses is False
    if kind == "conversation":
        assert preferences.binding.conversation_id == "original-conversation"
        assert preferences.binding.binding_revision == 1
    source = load_cli_config_and_ensure_existence()
    settings["buddy_interaction"]["animated"] = True
    assert source["buddy_interaction"]["animated"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, {"kind": "invalid", "animated": "no"}])
@private_profile_test
async def test_missing_or_malformed_buddy_settings_keep_safe_defaults(
    request: pytest.FixtureRequest, raw
):
    from tldw_chatbook.config import load_settings, save_settings_to_cli_config
    from tldw_chatbook.Persona_Buddy.interaction import (
        BuddyInteractionPreferences,
        parse_preferences,
    )

    if raw is not None:
        assert save_settings_to_cli_config({"buddy_interaction": raw})
    settings = load_settings(force_reload=True)
    assert ("buddy_interaction" in settings) is (raw is not None)
    assert (
        parse_preferences(settings.get("buddy_interaction", {}))
        == BuddyInteractionPreferences()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "reduce_motion", "expected_static"),
    [
        (None, False, False),
        ({"animated": True}, False, False),
        ({"animated": False}, False, True),
        ({"animated": 0}, False, False),
        ({"animated": ""}, False, False),
        ({"animated": []}, False, False),
        ({"animated": 0}, True, True),
        ({"animated": True}, True, True),
    ],
    ids=[
        "missing",
        "dynamic",
        "static",
        "zero",
        "empty-string",
        "empty-list",
        "invalid-global-override",
        "dynamic-global-override",
    ],
)
@private_profile_test
async def test_loaded_motion_settings_drive_actual_visual_controller(
    request: pytest.FixtureRequest, raw, reduce_motion: bool, expected_static: bool
):
    from tldw_chatbook.config import save_settings_to_cli_config

    settings = {
        "appearance": {"reduce_motion": reduce_motion},
        "model_catalog": {"auto_refresh_enabled": False},
    }
    if raw is not None:
        settings["buddy_interaction"] = raw
    assert save_settings_to_cli_config(settings)

    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    controller = app.ensure_persona_buddy_controller()
    assert controller is not None
    assert controller._current_reduced_motion() is expected_static
