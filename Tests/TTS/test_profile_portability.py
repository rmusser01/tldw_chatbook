"""Tests for the sanitized TTS generation-profile portability payload."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
from types import ModuleType
from uuid import UUID

import pytest

from tldw_chatbook.TTS.profile_types import TTSProfileDraft
from tldw_chatbook.TTS.profile_errors import ProfileValidationError


def _portability_module() -> ModuleType:
    module_name = "tldw_chatbook.TTS.profile_portability"
    assert importlib.util.find_spec(module_name) is not None, (
        "the profile portability codec has not been implemented"
    )
    return importlib.import_module(module_name)


def test_version_one_payload_has_the_exact_sanitized_wire_shape() -> None:
    portability = _portability_module()
    portable = portability.PortableTTSProfile(
        profile_id=UUID("00000000-0000-4000-8000-000000000000"),
        draft=TTSProfileDraft(
            display_name="Character voice",
            provider_id="audio_cpp",
            model_id="supertonic-3",
            voice_id="M1",
            response_format="wav",
            speed=1.0,
            options={},
        ),
    )

    assert portability.portable_profile_payload(portable) == {
        "schema_version": 1,
        "profile_id": "00000000-0000-4000-8000-000000000000",
        "name": "Character voice",
        "provider_id": "audio_cpp",
        "model_id": "supertonic-3",
        "voice_id": "M1",
        "response_format": "wav",
        "speed": 1.0,
        "options": {},
    }


def _valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "profile_id": "00000000-0000-4000-8000-000000000000",
        "name": "Character voice",
        "provider_id": "audio_cpp",
        "model_id": "supertonic-3",
        "voice_id": "M1",
        "response_format": "wav",
        "speed": 1.0,
        "options": {},
    }
    payload.update(overrides)
    return payload


def test_standalone_json_is_deterministic_and_contains_only_wire_fields() -> None:
    portability = _portability_module()
    portable = portability.PortableTTSProfile(
        profile_id=UUID("00000000-0000-4000-8000-000000000000"),
        draft=TTSProfileDraft(
            display_name="Character voice",
            provider_id="audio_cpp",
            model_id="supertonic-3",
            voice_id="M1",
            response_format="wav",
            speed=1.0,
            options={},
        ),
    )

    encoded = portability.portable_profile_json(portable)

    assert encoded == json.dumps(
        _valid_payload(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    for forbidden in (
        "authority",
        "origin",
        "credential",
        "path",
        "health",
        "revision",
        "timestamp",
        "text",
    ):
        assert forbidden not in encoded.casefold()


def test_valid_payload_decodes_to_an_exact_profile_selection() -> None:
    portability = _portability_module()

    result = portability.decode_portable_profile(_valid_payload())

    assert result.status == "valid"
    assert result.warning_code is None
    assert result.profile == portability.PortableTTSProfile(
        profile_id=UUID("00000000-0000-4000-8000-000000000000"),
        draft=TTSProfileDraft(
            display_name="Character voice",
            provider_id="audio_cpp",
            model_id="supertonic-3",
            voice_id="M1",
            response_format="wav",
            speed=1.0,
            options={},
        ),
    )


def test_portable_profile_value_rejects_non_audio_cpp_selection() -> None:
    portability = _portability_module()

    with pytest.raises(ProfileValidationError) as caught:
        portability.PortableTTSProfile(
            profile_id=UUID("00000000-0000-4000-8000-000000000000"),
            draft=TTSProfileDraft(
                display_name="Unsupported voice",
                provider_id="future_tts",
                model_id="future-model",
                voice_id=None,
                response_format="wav",
                speed=1.0,
                options={},
            ),
        )

    assert caught.value.code == "audio_cpp"


@pytest.mark.parametrize(
    ("overrides", "warning_code"),
    [
        ({"schema_version": 2}, "unsupported_version"),
        ({"provider_id": "future_tts"}, "unsupported_provider"),
    ],
)
def test_unknown_version_or_provider_skips_with_a_bounded_warning(
    overrides: dict[str, object],
    warning_code: str,
) -> None:
    portability = _portability_module()

    result = portability.decode_portable_profile(_valid_payload(**overrides))

    assert result.status == "skipped"
    assert result.profile is None
    assert result.warning_code == warning_code
    assert "future_tts" not in repr(result)


@pytest.mark.parametrize(
    "payload",
    [
        {**_valid_payload(), "unexpected": True},
        _valid_payload(profile_id="not-a-uuid"),
        _valid_payload(name="\u200einvisible"),
        _valid_payload(model_id="x" * 257),
        _valid_payload(voice_id="x" * 257),
        _valid_payload(speed=math.nan),
        _valid_payload(speed=1.01),
        _valid_payload(response_format="mp3"),
        _valid_payload(options={"secret": True}),
        _valid_payload(options={"a": {"b": {"c": {"d": {}}}}}),
        _valid_payload(options={"blob": "x" * (16 * 1024)}),
    ],
)
def test_malformed_known_payload_is_rejected_without_echoing_values(
    payload: dict[str, object],
) -> None:
    portability = _portability_module()

    result = portability.decode_portable_profile(payload)

    assert result.status == "invalid"
    assert result.profile is None
    assert result.warning_code == "invalid_attachment"
    assert "secret" not in repr(result)


def test_size_and_depth_are_checked_before_skipping_an_unknown_provider() -> None:
    portability = _portability_module()
    oversized = _valid_payload(
        provider_id="future_tts",
        options={"blob": "x" * (16 * 1024)},
    )
    too_deep = _valid_payload(
        provider_id="future_tts",
        options={"a": {"b": {"c": {"d": {}}}}},
    )

    assert portability.decode_portable_profile(oversized).status == "invalid"
    assert portability.decode_portable_profile(too_deep).status == "invalid"
