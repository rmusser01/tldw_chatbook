"""Tests for the public, per-selection legacy TTS request builder.

``build_legacy_speech_request`` is the promoted, explicit-field replacement
for ``request_admission._legacy_request``'s id/override derivation. Briefing
scripts need one legacy speech request per speaker, each with its own
provider/model/voice, so the builder cannot read a single shared
``TTSPreferencesSnapshot`` the way the app-wide chat experience does.

The round-trip tests are the load-bearing ones: every internal model id the
builder emits for a known legacy provider must resolve back through
``legacy_bridge.resolve_legacy_route`` without raising. That property is what
would have caught the existing drift between this builder's id table (mirrored
from ``request_admission._legacy_request``) and the differently-derived copy
in ``Event_Handlers/STTS_Events/stts_events.py``'s ``_legacy_internal_model_id``
(kokoro's onnx/pytorch suffix and alltalk's model-derived suffix there are
deliberately different — see the cross-reference comments at both sites).
"""

from __future__ import annotations

import pytest

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import resolve_legacy_route
from tldw_chatbook.TTS.legacy_request_builder import build_legacy_speech_request

pytestmark = pytest.mark.unit


# provider_id, model_id, response_format, expected_model, expected_format, expected_internal_id
_KNOWN_PROVIDER_CASES = (
    ("openai", "tts-1-hd", "opus", "tts-1-hd", "opus", "openai_official_tts-1-hd"),
    ("elevenlabs", "eleven_multilingual_v2", "wav", "elevenlabs", "mp3", "elevenlabs_elevenlabs"),
    ("kokoro", "kokoro", "mp3", "kokoro", "wav", "local_kokoro_default_onnx"),
    ("chatterbox", "chatterbox", "mp3", "chatterbox", "wav", "local_chatterbox_default"),
    ("higgs", "higgs-audio-v2", "wav", "higgs-audio-v2", "wav", "local_higgs_v2"),
    ("alltalk", "alltalk", "mp3", "alltalk", "wav", "alltalk_default"),
)


@pytest.mark.parametrize(
    (
        "provider_id",
        "model_id",
        "response_format",
        "expected_model",
        "expected_format",
        "expected_internal_id",
    ),
    _KNOWN_PROVIDER_CASES,
)
def test_id_table_matches_request_admission_exactly(
    provider_id: str,
    model_id: str,
    response_format: str,
    expected_model: str,
    expected_format: str,
    expected_internal_id: str,
) -> None:
    request, internal_model_id = build_legacy_speech_request(
        provider_id=provider_id,
        model_id=model_id,
        voice="Voice/Case",
        text="Character response",
        response_format=response_format,
    )

    assert internal_model_id == expected_internal_id
    assert request == OpenAISpeechRequest(
        model=expected_model,
        input="Character response",
        voice="voice/case",
        response_format=expected_format,
        speed=1.0,
    )


@pytest.mark.parametrize(
    ("provider_id", "model_id"),
    tuple((case[0], case[1]) for case in _KNOWN_PROVIDER_CASES),
)
def test_every_known_provider_internal_id_round_trips_through_legacy_bridge(
    provider_id: str,
    model_id: str,
) -> None:
    """The assertion that would have caught the kokoro/alltalk id drift."""
    _request, internal_model_id = build_legacy_speech_request(
        provider_id=provider_id,
        model_id=model_id,
        voice="voice",
        text="text",
    )

    route = resolve_legacy_route(internal_model_id)

    assert route.provider_id == provider_id
    assert route.internal_model_id == internal_model_id


def test_unrecognized_provider_falls_back_to_the_requests_model() -> None:
    """Unmapped-but-nonempty provider ids fall through to `request.model`."""
    request, internal_model_id = build_legacy_speech_request(
        provider_id="future_native",
        model_id="Some/Model",
        voice="voice",
        text="text",
    )

    assert internal_model_id == request.model == "some/model"


def test_voice_is_lowercased() -> None:
    _request, _internal_model_id = build_legacy_speech_request(
        provider_id="openai",
        model_id="tts-1",
        voice="ALLOY",
        text="text",
    )

    assert _request.voice == "alloy"


@pytest.mark.parametrize("empty_voice", ("", None))
def test_empty_or_none_voice_raises_value_error_naming_the_requirement(
    empty_voice: str | None,
) -> None:
    with pytest.raises(ValueError, match="voice"):
        build_legacy_speech_request(
            provider_id="openai",
            model_id="tts-1",
            voice=empty_voice,  # type: ignore[arg-type]
            text="text",
        )


def test_empty_provider_id_raises_value_error_naming_the_requirement() -> None:
    with pytest.raises(ValueError, match="provider"):
        build_legacy_speech_request(
            provider_id="",
            model_id="tts-1",
            voice="voice",
            text="text",
        )


def test_invalid_response_format_falls_back_to_wav() -> None:
    request, _internal_model_id = build_legacy_speech_request(
        provider_id="higgs",
        model_id="higgs-audio-v2",
        voice="voice",
        text="text",
        response_format="not-a-real-format",
    )

    assert request.response_format == "wav"


def test_provider_format_override_wins_over_requested_format() -> None:
    """elevenlabs is always mp3 regardless of the requested format."""
    request, _internal_model_id = build_legacy_speech_request(
        provider_id="elevenlabs",
        model_id="eleven_multilingual_v2",
        voice="voice",
        text="text",
        response_format="flac",
    )

    assert request.response_format == "mp3"


def test_speed_is_passed_through() -> None:
    request, _internal_model_id = build_legacy_speech_request(
        provider_id="higgs",
        model_id="higgs-audio-v2",
        voice="voice",
        text="text",
        speed=1.75,
    )

    assert request.speed == 1.75
