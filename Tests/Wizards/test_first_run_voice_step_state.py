from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_VOICE,
    TRACK_QUICK,
    setup_draft_checkpoint,
)
from tldw_chatbook.UI.Wizards.first_run_voice_step_state import (
    VOICE_PRESET_CUSTOM,
    VOICE_PRESET_OFFICIAL_OPENAI,
    VOICE_PRESET_POCKET_TTS,
    VoiceSetupDraft,
    apply_voice_preset,
    build_voice_setup_save_event,
    run_voice_sample,
    validate_voice_sample_text,
    validate_voice_setup_draft,
)


def _pocket_draft(**changes: object) -> VoiceSetupDraft:
    values: dict[str, object] = {
        "endpoint": "http://127.0.0.1:8765/v1/audio/speech",
        "authentication_mode": "none",
        "model_id": "pocket-tts",
        "voice_id": "alba",
        "response_format": "wav",
        "speed": 1.0,
        "sample_text": "Hello from Chatbook.",
        "use_as_default": False,
    }
    values.update(changes)
    return VoiceSetupDraft(**values)  # type: ignore[arg-type]


def test_voice_setup_draft_is_frozen_slotted_and_non_secret() -> None:
    draft = _pocket_draft()

    assert not hasattr(draft, "__dict__")
    assert "credential" not in draft.__slots__
    assert "api_key" not in draft.__slots__
    with pytest.raises(FrozenInstanceError):
        draft.voice_id = "new"  # type: ignore[misc]


def test_pocket_tts_draft_is_locally_valid_without_connection_or_key() -> None:
    validation = validate_voice_setup_draft(_pocket_draft())

    assert validation.configuration_valid is True
    assert validation.connection_state == "needs_test"
    assert validation.normalized_endpoint == ("http://127.0.0.1:8765/v1/audio/speech")


@pytest.mark.parametrize("value", ["   ", "x" * 501, None, 7])
def test_voice_sample_rejects_blank_overlong_and_non_text(value: object) -> None:
    with pytest.raises(ValueError, match="1 to 500"):
        validate_voice_sample_text(value)


def test_voice_sample_is_trimmed_at_the_validation_boundary() -> None:
    assert validate_voice_sample_text("  Hello.  ") == "Hello."


def test_voice_presets_enforce_authentication_without_erasing_custom_values() -> None:
    custom = _pocket_draft(
        endpoint="https://speech.example.test/custom",
        authentication_mode="api_key",
        model_id="custom-model",
        voice_id="custom-voice",
    )

    pocket = apply_voice_preset(custom, VOICE_PRESET_POCKET_TTS)
    official = apply_voice_preset(custom, VOICE_PRESET_OFFICIAL_OPENAI)
    restored = apply_voice_preset(custom, VOICE_PRESET_CUSTOM)

    assert pocket.endpoint == "http://127.0.0.1:8765/v1/audio/speech"
    assert pocket.authentication_mode == "none"
    assert official.endpoint == "https://api.openai.com/v1/audio/speech"
    assert official.authentication_mode == "api_key"
    assert restored.endpoint == custom.endpoint
    assert restored.model_id == custom.model_id


def test_voice_service_presets_replace_incompatible_model_and_voice_defaults() -> None:
    custom = _pocket_draft(
        endpoint="https://speech.example.test/custom",
        authentication_mode="api_key",
        model_id="custom-model",
        voice_id="custom-voice",
        response_format="flac",
        speed=1.25,
    )

    pocket = apply_voice_preset(custom, VOICE_PRESET_POCKET_TTS)
    official = apply_voice_preset(pocket, VOICE_PRESET_OFFICIAL_OPENAI)
    pocket_again = apply_voice_preset(official, VOICE_PRESET_POCKET_TTS)

    assert (pocket.model_id, pocket.voice_id) == ("pocket-tts", "alba")
    assert (official.model_id, official.voice_id) == ("tts-1-hd", "shimmer")
    assert official.response_format == "mp3"
    assert (pocket_again.model_id, pocket_again.voice_id) == ("pocket-tts", "alba")
    assert pocket_again.response_format == "wav"


def test_custom_preset_preserves_every_user_owned_field() -> None:
    custom = _pocket_draft(
        endpoint="https://speech.example.test/custom",
        authentication_mode="api_key",
        model_id="custom-model",
        voice_id="custom-voice",
        response_format="flac",
        speed=1.25,
        sample_text="Keep this exact sample.",
        use_as_default=True,
    )

    assert apply_voice_preset(custom, VOICE_PRESET_CUSTOM) == custom


def test_official_openai_rejects_none_but_custom_uses_exact_endpoint_rules() -> None:
    invalid_official = _pocket_draft(
        endpoint="https://api.openai.com/v1/audio/speech",
        authentication_mode="none",
    )
    ambiguous_custom = _pocket_draft(
        endpoint="https://speech.example.test/v1//audio/speech",
    )

    assert not validate_voice_setup_draft(invalid_official).configuration_valid
    assert not validate_voice_setup_draft(ambiguous_custom).configuration_valid


def test_voice_authentication_rejects_values_outside_explicit_control() -> None:
    validation = validate_voice_setup_draft(
        _pocket_draft(authentication_mode="unexpected")
    )

    assert validation.configuration_valid is False
    assert any("authentication" in error.casefold() for error in validation.errors)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:8765/v1/audio/speech",
        "http://127.42.0.9:8765/v1/audio/speech",
        "http://[::1]:8765/v1/audio/speech",
        "https://speech.example.test/v1/audio/speech",
    ],
)
def test_keyed_voice_transport_allows_https_and_normalized_loopback_http(
    endpoint: str,
) -> None:
    validation = validate_voice_setup_draft(
        _pocket_draft(endpoint=endpoint, authentication_mode="api_key")
    )

    assert validation.configuration_valid is True
    assert validation.normalized_endpoint is not None


def test_keyed_voice_transport_rejects_remote_plain_http_for_save() -> None:
    draft = _pocket_draft(
        endpoint="http://speech.example.test/v1/audio/speech",
        authentication_mode="api_key",
    )
    validation = validate_voice_setup_draft(draft)

    assert validation.configuration_valid is False
    assert validation.errors == (
        "API key authentication requires HTTPS or a loopback HTTP endpoint.",
    )
    with pytest.raises(ValueError, match="configuration is invalid"):
        build_voice_setup_save_event(draft)


@pytest.mark.asyncio
async def test_keyed_remote_http_sample_fails_before_request_or_header(
    monkeypatch,
) -> None:
    requests: list[object] = []

    def unexpected_client(*args, **kwargs):
        requests.append((args, kwargs))
        raise AssertionError("HTTP client must not be created")

    monkeypatch.setattr(
        "tldw_chatbook.UI.Wizards.first_run_voice_step_state.httpx.AsyncClient",
        unexpected_client,
    )
    draft = _pocket_draft(
        endpoint="http://speech.example.test/v1/audio/speech",
        authentication_mode="api_key",
    )

    with pytest.raises(ValueError, match="configuration is invalid"):
        await run_voice_sample(draft, credential="must-not-be-sent")

    assert requests == []


def test_save_event_is_opt_in_and_carries_only_exact_default_axes() -> None:
    without_default = build_voice_setup_save_event(_pocket_draft())
    with_default = build_voice_setup_save_event(
        _pocket_draft(use_as_default=True), request_id=9, reply_to=object()
    )

    assert without_default.settings == {
        "OPENAI_BASE_URL": "http://127.0.0.1:8765/v1/audio/speech",
        "OPENAI_AUTH_MODE": "none",
    }
    assert without_default.preferences is None
    assert without_default.commit_defaults_after_handoff is False
    assert with_default.preferences is not None
    assert with_default.preferences.provider_id == "openai"
    assert with_default.preferences.model_id == "pocket-tts"
    assert with_default.preferences.voice_id == "alba"
    assert with_default.preferences.response_format == "wav"
    assert with_default.preferences.speed == 1.0
    assert with_default.commit_defaults_after_handoff is True
    assert "endpoint" not in with_default.preferences.__slots__
    assert "authentication_mode" not in with_default.preferences.__slots__


def test_setup_recovery_checkpoints_every_non_secret_voice_value_only() -> None:
    secret = "must-not-survive"
    draft = setup_draft_checkpoint(
        track=TRACK_QUICK,
        active_step_id=STEP_VOICE,
        values={
            STEP_VOICE: {
                "endpoint": "http://127.0.0.1:8765/v1/audio/speech",
                "authentication_mode": "none",
                "model_id": "pocket-tts",
                "voice_id": "alba",
                "response_format": "wav",
                "speed": 1.0,
                "sample_text": "Resume this sample.",
                "use_as_default": True,
                "api_key": secret,
            }
        },
    )

    assert draft.values[STEP_VOICE] == {
        "endpoint": "http://127.0.0.1:8765/v1/audio/speech",
        "authentication_mode": "none",
        "model_id": "pocket-tts",
        "voice_id": "alba",
        "response_format": "wav",
        "speed": 1.0,
        "sample_text": "Resume this sample.",
        "use_as_default": True,
    }
    assert secret not in repr(draft)
