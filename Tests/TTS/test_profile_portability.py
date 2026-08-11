"""Tests for the sanitized TTS generation-profile portability payload."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
from dataclasses import fields
from types import ModuleType
from uuid import UUID

import pytest

from tldw_chatbook.TTS.profile_errors import ProfileValidationError
from tldw_chatbook.TTS.profile_types import TTSProfileDraft


def _portability_module() -> ModuleType:
    module_name = "tldw_chatbook.TTS.profile_portability"
    assert importlib.util.find_spec(module_name) is not None, (
        "the profile portability codec has not been implemented"
    )
    return importlib.import_module(module_name)


def _forged_draft(draft: TTSProfileDraft, **updates: object) -> TTSProfileDraft:
    """Build an adversarial exact draft without domain revalidation.

    Mirrors `test_profile_service.py`'s `_forged_profile` helper: a plain
    `TTSProfileDraft(provider_id="future_tts", ...)` construction is rejected
    by the draft's own `__post_init__` (Task 1's provider contract table), so
    the only honest way to exercise `PortableTTSProfile.__post_init__`'s own
    provider-membership defense is a draft that bypasses its own validation,
    matching the pattern already established for `TTSGenerationProfile`.
    """

    forged = object.__new__(TTSProfileDraft)
    for draft_field in fields(TTSProfileDraft):
        object.__setattr__(
            forged,
            draft_field.name,
            updates.get(draft_field.name, getattr(draft, draft_field.name)),
        )
    return forged


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


def test_private_reference_storage_does_not_change_ordinary_wire_v1() -> None:
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

    payload = portability.portable_profile_payload(portable)
    encoded = portability.portable_profile_json(portable)

    assert tuple(payload) == (
        "schema_version",
        "profile_id",
        "name",
        "provider_id",
        "model_id",
        "voice_id",
        "response_format",
        "speed",
        "options",
    )
    assert payload["schema_version"] == 1
    for forbidden in (
        "reference",
        "reference_id",
        "reference_text",
        "wav_bytes",
        "sha256",
        "source_path",
    ):
        assert forbidden not in payload
        assert forbidden not in encoded


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


def test_portable_profile_value_accepts_legacy_provider_selection() -> None:
    portability = _portability_module()

    portable = portability.PortableTTSProfile(
        profile_id=UUID("00000000-0000-4000-8000-000000000000"),
        draft=TTSProfileDraft(
            display_name="Legacy voice",
            provider_id="openai",
            model_id="tts-1",
            voice_id="alloy",
            response_format="mp3",
            speed=1.0,
            options={},
        ),
    )

    assert portable.draft.provider_id == "openai"


def test_portable_profile_value_rejects_hostile_provider() -> None:
    """`PortableTTSProfile.__post_init__`'s own provider-membership check is
    defense-in-depth: no honestly-constructed `TTSProfileDraft` can carry an
    out-of-set `provider_id` (Task 1 already rejects that at the draft's own
    construction), so this branch is only reachable via a forged draft --
    e.g. one representing data that predates the current provider contract.
    """

    portability = _portability_module()
    hostile_draft = _forged_draft(
        TTSProfileDraft(
            display_name="Hostile voice",
            provider_id="openai",
            model_id="tts-1",
            voice_id="alloy",
            response_format="mp3",
            speed=1.0,
            options={},
        ),
        provider_id="future_tts",
    )

    with pytest.raises(ProfileValidationError) as caught:
        portability.PortableTTSProfile(
            profile_id=UUID("00000000-0000-4000-8000-000000000000"),
            draft=hostile_draft,
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


@pytest.mark.parametrize(
    "provider_id",
    sorted(("openai", "elevenlabs", "kokoro", "chatterbox", "higgs", "alltalk")),
)
def test_null_voice_legacy_attachment_decodes_invalid(provider_id: str) -> None:
    """A voiceless legacy card must never yield an assignable profile.

    Task 6c auto-assigns whatever this decodes as VALID, so a null voice must
    be refused here rather than becoming an assigned, unspeakable profile.
    """

    portability = _portability_module()
    payload = _valid_payload(
        provider_id=provider_id,
        response_format="mp3",
        voice_id=None,
    )

    result = portability.decode_portable_profile(payload)

    assert result.status == "invalid"
    assert result.profile is None
    assert result.warning_code == "invalid_attachment"


def test_null_voice_audio_cpp_attachment_still_decodes_valid() -> None:
    portability = _portability_module()

    result = portability.decode_portable_profile(_valid_payload(voice_id=None))

    assert result.status == "valid"
    assert result.profile is not None
    assert result.profile.draft.voice_id is None


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
