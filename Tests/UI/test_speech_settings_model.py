"""Which settings belong to which provider, and whether it is set up.

The second question is the one the legacy screen could not answer. It
rendered eight identical collapsed boxes, so "is ElevenLabs configured?"
required opening each in turn. The spec's rule -- one block per provider,
only the configured ones expanded -- depends on this module.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Speech.speech_settings_model import (
    ALL_SETTINGS_CONTROLS,
    PROVIDER_SETTINGS,
    SETTINGS_ACTIONS,
    SETTINGS_STATUS,
    SETTINGS_PROVIDER_ORDER,
    configured_state,
    settings_for_provider,
)


@pytest.mark.unit
def test_a_provider_holding_a_custom_value_reads_as_configured():
    """This is what decides which groups open on arrival.

    Without it the spec's "only the configured ones expanded" rule cannot be
    implemented, and the user is back to eight identical closed boxes.
    """
    state = configured_state(
        "audio_cpp", {"audio-cpp-base-url-input": "http://192.168.1.5:9000"}
    )
    assert state == "configured"


@pytest.mark.unit
def test_a_provider_at_its_defaults_reads_as_default():
    assert configured_state("audio_cpp", {}) == "default"


@pytest.mark.unit
def test_blank_values_are_not_configuration():
    """An empty string is what an untouched Input holds. Treating it as a
    value would mark every provider configured and expand all eight."""
    assert configured_state("openai", {"openai-api-key-input": ""}) == "default"
    assert configured_state("openai", {"openai-api-key-input": "   "}) == "default"


@pytest.mark.unit
def test_every_id_these_tests_name_actually_exists():
    """A test naming a control that does not exist asserts nothing.

    The first draft of the incomplete/configured cases used
    `elevenlabs-voice-id-input`, which the screen has never had -- so
    "nothing is set" was trivially true and the case passed for the wrong
    reason until the model disagreed with it.
    """
    named = {
        "audio-cpp-base-url-input",
        "openai-api-key-input",
        "elevenlabs-api-key-input",
        "elevenlabs-stability-input",
    }
    assert named <= ALL_SETTINGS_CONTROLS, sorted(named - ALL_SETTINGS_CONTROLS)


@pytest.mark.unit
def test_a_half_filled_provider_reads_as_incomplete():
    """Half-configured is the state worth surfacing: it is the one that
    fails at generation time with nothing on screen having said so."""
    state = configured_state(
        "elevenlabs",
        {"elevenlabs-api-key-input": "", "elevenlabs-stability-input": "0.7"},
    )
    assert state == "incomplete"


@pytest.mark.unit
def test_a_fully_filled_provider_is_configured_not_incomplete():
    state = configured_state(
        "elevenlabs",
        {"elevenlabs-api-key-input": "sk-live", "elevenlabs-stability-input": "0.7"},
    )
    assert state == "configured"


@pytest.mark.unit
def test_an_unknown_provider_yields_no_settings_rather_than_raising():
    """compose() calls this; raising would take the screen down."""
    assert settings_for_provider("nonexistent") == ()
    assert configured_state("nonexistent", {"whatever": "x"}) == "default"


@pytest.mark.unit
def test_no_setting_is_claimed_by_two_providers():
    """A shared id would be written twice and read back wrong."""
    seen: set[str] = set()
    for controls in PROVIDER_SETTINGS.values():
        assert not (seen & set(controls)), sorted(seen & set(controls))
        seen |= set(controls)


@pytest.mark.unit
def test_every_provider_in_the_order_has_settings():
    """An ordered provider with no settings renders an empty group."""
    for provider in SETTINGS_PROVIDER_ORDER:
        assert settings_for_provider(provider), f"{provider} has no settings"


@pytest.mark.unit
def test_actions_are_not_filed_as_settings():
    """Save and the blend import/export are commands, not values to persist;
    filing them as settings would put them inside a provider group."""
    every_setting = {c for controls in PROVIDER_SETTINGS.values() for c in controls}
    assert not (set(SETTINGS_ACTIONS) & every_setting)


@pytest.mark.unit
def test_the_inventory_covers_every_legacy_id():
    """79 ids, classified. An unclassified control is one that goes missing."""
    classified = (
        {c for controls in PROVIDER_SETTINGS.values() for c in controls}
        | set(SETTINGS_ACTIONS)
        | set(SETTINGS_STATUS)
    )
    assert classified <= ALL_SETTINGS_CONTROLS
    unclassified = ALL_SETTINGS_CONTROLS - classified
    assert not unclassified, f"unclassified: {sorted(unclassified)}"
