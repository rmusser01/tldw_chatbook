"""Unit tests for the app-wide default-voice-profile resolver (slice 3, task 4)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest
from loguru import logger

from tldw_chatbook.TTS.adapter_types import TTSRequest
from tldw_chatbook.TTS.character_request_resolver import CharacterTTSResolutionError
from tldw_chatbook.TTS.default_profile_request_resolver import resolve_default_profile
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError, ProfileServiceError
from tldw_chatbook.TTS.profile_service import LoadedTTSProfile
from tldw_chatbook.TTS.profile_reference_types import (
    TTSCloneReference,
    TTSCloneReferenceSummary,
)
from tldw_chatbook.TTS.profile_types import TTSGenerationProfile, TTSProfileDraft

_PROFILE_ID = UUID("44444444-4444-4444-8444-444444444444")
_CREATED_AT = datetime(2026, 8, 6, tzinfo=UTC)


def _profile(
    *,
    revision: int = 3,
    provider_id: str = "kokoro",
    model_id: str = "kokoro-v1",
    voice_id: str | None = "af_bella",
    response_format: str = "wav",
    reference: TTSCloneReferenceSummary | None = None,
) -> TTSGenerationProfile:
    draft = TTSProfileDraft(
        display_name="Narrator",
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        response_format=response_format,
        speed=1.0,
        options={},
    )
    return TTSGenerationProfile(
        profile_id=_PROFILE_ID,
        display_name=draft.display_name,
        normalized_name=draft.normalized_name,
        provider_id=draft.provider_id,
        model_id=draft.model_id,
        voice_id=draft.voice_id,
        response_format=draft.response_format,
        speed=draft.speed,
        options=draft.options,
        revision=revision,
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
        reference=reference,
    )


def _reference() -> TTSCloneReference:
    wav_bytes = b"default-private-reference"
    return TTSCloneReference(
        summary=TTSCloneReferenceSummary(
            reference_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            byte_length=len(wav_bytes),
            duration_ms=300,
            sample_rate_hz=24_000,
            channels=1,
            sample_encoding="pcm_s16le",
            created_at=_CREATED_AT,
            updated_at=_CREATED_AT,
        ),
        reference_text="Default private transcript",
        sha256=hashlib.sha256(wav_bytes).hexdigest(),
        wav_bytes=wav_bytes,
    )


class _FakeDefaultProfileService:
    def __init__(
        self, *, result: object = None, error: BaseException | None = None
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[UUID] = []
        self.reference_result: TTSCloneReference | None = None
        self.reference_error: BaseException | None = None
        self.reference_calls: list[tuple[UUID, int, int]] = []

    async def get_profile(self, profile_id: UUID) -> LoadedTTSProfile:
        self.calls.append(profile_id)
        if self.error is not None:
            raise self.error
        return cast(LoadedTTSProfile, self.result)

    async def get_reference(
        self,
        profile_id: UUID,
        *,
        expected_revision: int,
        expected_generation: int,
    ) -> TTSCloneReference:
        self.reference_calls.append(
            (profile_id, expected_revision, expected_generation)
        )
        if self.reference_error is not None:
            raise self.reference_error
        if self.reference_result is None:
            raise AssertionError("unexpected reference read")
        return self.reference_result


@pytest.mark.asyncio
async def test_loadable_default_profile_resolves_to_exact_request() -> None:
    profile = _profile(revision=5)
    service = _FakeDefaultProfileService(
        result=LoadedTTSProfile(repository_generation=7, profile=profile)
    )

    resolved = await resolve_default_profile(
        text="Read this aloud.",
        default_profile_id=str(_PROFILE_ID),
        profile_service=service,
    )

    assert resolved.source == "default_profile"
    assert resolved.request == TTSRequest(
        provider_id="kokoro",
        model_id="kokoro-v1",
        text="Read this aloud.",
        voice="af_bella",
        response_format="wav",
        speed=1.0,
        options={},
    )
    assert resolved.repository_generation == 7
    assert resolved.profile_id == _PROFILE_ID
    assert resolved.profile_revision == 5
    assert service.calls == [_PROFILE_ID]
    assert service.reference_calls == []


@pytest.mark.asyncio
async def test_default_profile_freezes_exact_reference_under_profile_fences() -> None:
    reference = _reference()
    profile = _profile(
        revision=5,
        provider_id="audio_cpp",
        model_id="pocket-tts",
        voice_id=None,
        reference=reference.summary,
    )
    service = _FakeDefaultProfileService(
        result=LoadedTTSProfile(repository_generation=7, profile=profile)
    )
    service.reference_result = reference

    resolved = await resolve_default_profile(
        text="Read with the clone.",
        default_profile_id=str(_PROFILE_ID),
        profile_service=service,
    )

    assert resolved.reference == reference
    assert resolved.profile_id == _PROFILE_ID
    assert resolved.profile_revision == 5
    assert resolved.repository_generation == 7
    assert service.reference_calls == [(_PROFILE_ID, 5, 7)]
    assert "Default private transcript" not in repr(resolved)


@pytest.mark.asyncio
async def test_default_profile_reference_delete_race_fails_closed() -> None:
    reference = _reference()
    profile = _profile(
        revision=5,
        provider_id="audio_cpp",
        model_id="pocket-tts",
        voice_id=None,
        reference=reference.summary,
    )
    service = _FakeDefaultProfileService(
        result=LoadedTTSProfile(repository_generation=7, profile=profile)
    )
    service.reference_error = ProfileRepositoryError("reference_unavailable")

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolve_default_profile(
            text="Do not substitute a deleted reference.",
            default_profile_id=str(_PROFILE_ID),
            profile_service=service,
        )

    assert caught.value.code == "default_profile_store_unavailable"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert service.reference_calls == [(_PROFILE_ID, 5, 7)]


@pytest.mark.asyncio
async def test_default_profile_reference_failure_severs_raw_exception() -> None:
    reference = _reference()
    service = _FakeDefaultProfileService(
        result=LoadedTTSProfile(
            repository_generation=7,
            profile=_profile(reference=reference.summary),
        )
    )
    service.reference_error = RuntimeError("PRIVATE_REFERENCE_CANARY")

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolve_default_profile(
            text="Do not expose a reference failure.",
            default_profile_id=str(_PROFILE_ID),
            profile_service=service,
        )

    assert caught.value.code == "default_profile_store_unavailable"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "PRIVATE_REFERENCE_CANARY" not in repr(caught.value)


@pytest.mark.asyncio
async def test_unbound_profile_store_refuses_with_default_specific_code() -> None:
    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolve_default_profile(
            text="Speak.",
            default_profile_id=str(_PROFILE_ID),
            profile_service=None,
        )

    assert caught.value.code == "default_profile_store_unavailable"
    assert caught.value.allow_global_override is True
    assert "character" not in str(caught.value).lower()


@pytest.mark.asyncio
async def test_malformed_stored_id_is_treated_as_missing() -> None:
    service = _FakeDefaultProfileService()

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolve_default_profile(
            text="Speak.",
            default_profile_id="not-a-uuid",
            profile_service=service,
        )

    assert caught.value.code == "default_profile_missing"
    assert caught.value.allow_global_override is True
    assert service.calls == []


@pytest.mark.asyncio
async def test_deleted_profile_is_reported_missing_not_store_unavailable() -> None:
    service = _FakeDefaultProfileService(error=ProfileRepositoryError("missing"))

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolve_default_profile(
            text="Speak.",
            default_profile_id=str(_PROFILE_ID),
            profile_service=service,
        )

    assert caught.value.code == "default_profile_missing"
    assert caught.value.allow_global_override is True
    assert "character" not in str(caught.value).lower()
    assert "default voice" in str(caught.value).lower()
    assert service.calls == [_PROFILE_ID]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    (
        ProfileRepositoryError("unavailable"),
        ProfileServiceError("operation_failed"),
        RuntimeError("boom"),
    ),
)
async def test_other_repository_failures_are_store_unavailable_and_private(
    error: BaseException,
) -> None:
    service = _FakeDefaultProfileService(error=error)
    log_messages: list[str] = []

    sink_id = logger.add(log_messages.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(CharacterTTSResolutionError) as caught:
            await resolve_default_profile(
                text="Private text should never be logged.",
                default_profile_id=str(_PROFILE_ID),
                profile_service=service,
            )
    finally:
        logger.remove(sink_id)

    assert caught.value.code == "default_profile_store_unavailable"
    assert caught.value.allow_global_override is True
    rendered_logs = "\n".join(log_messages)
    assert "operation=default_profile_tts_resolution" in rendered_logs
    assert "outcome_code=default_profile_store_unavailable" in rendered_logs
    assert "Private text should never be logged." not in rendered_logs


@pytest.mark.asyncio
async def test_malformed_loaded_profile_result_fails_closed() -> None:
    service = _FakeDefaultProfileService(result=object())

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolve_default_profile(
            text="Speak.",
            default_profile_id=str(_PROFILE_ID),
            profile_service=service,
        )

    assert caught.value.code == "default_profile_store_unavailable"
