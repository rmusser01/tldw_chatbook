"""Versioned per-conversation Console reply-speech preferences."""

from __future__ import annotations

import importlib
import json

import pytest

DESTINATION = "sha256:" + "a" * 64


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (DESTINATION, True),
        ("sha256:" + "0" * 64, True),
        (None, False),
        ("", False),
        ("sha256:" + "A" * 64, False),
        ("sha256:" + "a" * 63, False),
        ("sha256:" + "a" * 65, False),
        ("sha256:" + "g" * 64, False),
        (42, False),
    ],
)
def test_console_speech_destination_validator_is_exact(value, expected):
    speech = _speech_module()

    assert speech.is_console_speech_destination(value) is expected


@pytest.mark.parametrize(
    "destination",
    ["", "sha256:" + "A" * 64, "sha256:" + "a" * 63, 42],
)
def test_preferences_model_uses_the_canonical_destination_boundary(destination):
    speech = _speech_module()

    assert speech.is_console_speech_destination(destination) is False
    with pytest.raises(ValueError, match="canonical SHA-256"):
        speech.ConsoleSpeechPreferences(consent_destination=destination)


def test_preferences_model_accepts_the_canonical_destination_boundary():
    speech = _speech_module()

    assert speech.is_console_speech_destination(DESTINATION) is True
    assert (
        speech.ConsoleSpeechPreferences(
            consent_destination=DESTINATION
        ).consent_destination
        == DESTINATION
    )


def _speech_module():
    return importlib.import_module("tldw_chatbook.Chat.console_speech_preferences")


def test_missing_metadata_defaults_auto_speak_off():
    speech = _speech_module()

    assert (
        speech.parse_console_speech_preferences(None)
        == speech.ConsoleSpeechPreferences()
    )


@pytest.mark.parametrize(
    "owned",
    [
        {
            "auto_speak": 1,
            "paused": False,
            "consent_destination": None,
            "consent_version": 1,
        },
        {
            "auto_speak": False,
            "paused": 0,
            "consent_destination": None,
            "consent_version": 1,
        },
        {
            "auto_speak": True,
            "paused": False,
            "consent_destination": "sha256:ABC",
            "consent_version": 1,
        },
        {
            "auto_speak": True,
            "paused": False,
            "consent_destination": "sha256:" + "a" * 63,
            "consent_version": 1,
        },
        {
            "auto_speak": True,
            "paused": False,
            "consent_destination": DESTINATION,
            "consent_version": True,
        },
        {
            "auto_speak": True,
            "paused": False,
            "consent_destination": DESTINATION,
            "consent_version": 2,
        },
        {"auto_speak": True, "paused": False, "consent_destination": DESTINATION},
        ["not", "an", "object"],
    ],
)
def test_invalid_owned_metadata_fails_closed_to_all_defaults(owned):
    speech = _speech_module()

    assert speech.parse_console_speech_preferences({"console_speech": owned}) == (
        speech.ConsoleSpeechPreferences()
    )


def test_valid_preferences_parse_from_json_metadata():
    speech = _speech_module()
    metadata = json.dumps(
        {
            "console_speech": {
                "auto_speak": True,
                "paused": True,
                "consent_destination": DESTINATION,
                "consent_version": 1,
            }
        }
    )

    assert speech.parse_console_speech_preferences(metadata) == (
        speech.ConsoleSpeechPreferences(True, True, DESTINATION, 1)
    )


def test_mapping_owned_metadata_with_extra_key_fails_closed():
    speech = _speech_module()
    metadata = {
        "console_speech": {
            "auto_speak": True,
            "paused": False,
            "consent_destination": DESTINATION,
            "consent_version": 1,
            "future_flag": True,
        }
    }

    assert speech.parse_console_speech_preferences(metadata) == (
        speech.ConsoleSpeechPreferences()
    )


def test_json_owned_metadata_with_extra_key_fails_closed():
    speech = _speech_module()
    metadata = json.dumps(
        {
            "console_speech": {
                "auto_speak": True,
                "paused": False,
                "consent_destination": DESTINATION,
                "consent_version": 1,
                "future_flag": True,
            }
        }
    )

    assert speech.parse_console_speech_preferences(metadata) == (
        speech.ConsoleSpeechPreferences()
    )


def test_speech_preferences_merge_preserves_roleplay_and_unrelated_metadata():
    speech = _speech_module()
    metadata = {
        "console_roleplay_context": {
            "version": 1,
            "character_system_template": "Stay kind.",
        },
        "other": {"keep": True},
    }

    merged = speech.merge_console_speech_preferences(
        metadata,
        speech.ConsoleSpeechPreferences(True, True, DESTINATION, 1),
    )

    assert merged["console_roleplay_context"] == metadata["console_roleplay_context"]
    assert merged["other"] == metadata["other"]
    assert merged["console_speech"] == {
        "auto_speak": True,
        "paused": True,
        "consent_destination": DESTINATION,
        "consent_version": 1,
    }


def test_merge_rejects_future_mapping_payload_without_mutating_it():
    speech = _speech_module()
    metadata = {
        "console_speech": {
            "auto_speak": True,
            "paused": False,
            "consent_destination": DESTINATION,
            "consent_version": 2,
            "future_flag": "keep",
        },
        "other": {"keep": True},
    }
    original = json.loads(json.dumps(metadata))

    with pytest.raises(
        speech.ConsoleSpeechPreferencesVersionError,
        match="version 2",
    ):
        speech.merge_console_speech_preferences(
            metadata,
            speech.ConsoleSpeechPreferences(auto_speak=False),
        )

    assert metadata == original


def test_merge_rejects_future_json_payload_without_replacing_it():
    speech = _speech_module()
    metadata = json.dumps(
        {
            "console_speech": {
                "auto_speak": True,
                "paused": False,
                "consent_destination": DESTINATION,
                "consent_version": 2,
            },
            "other": {"keep": True},
        },
        sort_keys=True,
    )
    original = metadata

    with pytest.raises(
        speech.ConsoleSpeechPreferencesVersionError,
        match="version 2",
    ):
        speech.merge_console_speech_preferences(
            metadata,
            speech.ConsoleSpeechPreferences(auto_speak=False),
        )

    assert metadata == original


@pytest.mark.parametrize(
    "preferences",
    [
        (1, False, None, 1),
        (False, 0, None, 1),
        (True, False, "sha256:" + "B" * 64, 1),
        (True, False, DESTINATION, True),
        (True, False, DESTINATION, 2),
    ],
)
def test_merge_rejects_noncanonical_preference_values(preferences):
    speech = _speech_module()

    with pytest.raises(ValueError):
        speech.merge_console_speech_preferences(
            {"other": 1},
            speech.ConsoleSpeechPreferences(*preferences),
        )
