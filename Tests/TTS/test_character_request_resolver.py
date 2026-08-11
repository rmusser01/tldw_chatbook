from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest
from loguru import logger

from tldw_chatbook.TTS.adapter_types import TTSRequest
from tldw_chatbook.TTS.character_request_resolver import (
    CharacterTTSRequestResolution,
    CharacterTTSRequestResolver,
    CharacterTTSResolutionError,
)
from tldw_chatbook.TTS.profile_errors import (
    ProfileRepositoryError,
    ProfileServiceError,
)
from tldw_chatbook.TTS.profile_service import LoadedCharacterTTSAssignment
from tldw_chatbook.TTS.profile_reference_types import (
    TTSCloneReference,
    TTSCloneReferenceSummary,
)
from tldw_chatbook.TTS.profile_types import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    TTSGenerationProfile,
    TTSProfileDraft,
)

_PROFILE_ID = UUID("11111111-1111-4111-8111-111111111111")
_CREATED_AT = datetime(2026, 7, 31, tzinfo=UTC)


def _character_ref(
    *,
    source: str = "server",
    authority_id: str = "server-user-v1:authority-a",
    character_id: str = "42",
) -> CharacterRef:
    return CharacterRef(
        source=source,  # type: ignore[arg-type]
        authority_id=authority_id,
        character_id=character_id,
    )


def _profile(
    *,
    revision: int = 4,
    provider_id: str = "audio_cpp",
    model_id: str = "supertonic-3",
    voice_id: str | None = "voice-7",
    response_format: str = "wav",
    reference: TTSCloneReferenceSummary | None = None,
) -> TTSGenerationProfile:
    draft = TTSProfileDraft(
        display_name="Mara",
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
    wav_bytes = b"character-private-reference"
    return TTSCloneReference(
        summary=TTSCloneReferenceSummary(
            reference_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            byte_length=len(wav_bytes),
            duration_ms=300,
            sample_rate_hz=24_000,
            channels=1,
            sample_encoding="pcm_s16le",
            created_at=_CREATED_AT,
            updated_at=_CREATED_AT,
        ),
        reference_text="Character private transcript",
        sha256=hashlib.sha256(wav_bytes).hexdigest(),
        wav_bytes=wav_bytes,
    )


def _loaded_assignment(
    character_ref: CharacterRef,
    *,
    generation: int = 9,
    revision: int = 4,
    provider_id: str = "audio_cpp",
    model_id: str = "supertonic-3",
    voice_id: str | None = "voice-7",
    response_format: str = "wav",
    reference: TTSCloneReferenceSummary | None = None,
) -> LoadedCharacterTTSAssignment:
    profile = _profile(
        revision=revision,
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        response_format=response_format,
        reference=reference,
    )
    return LoadedCharacterTTSAssignment(
        repository_generation=generation,
        snapshot=AssignedTTSProfileSnapshot(
            assignment=CharacterTTSAssignment(
                character_ref=character_ref,
                profile_id=profile.profile_id,
            ),
            profile=profile,
        ),
    )


class _FakeProfileService:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[CharacterRef] = []
        self.error: BaseException | None = None
        self.reference_result: TTSCloneReference | None = None
        self.reference_error: BaseException | None = None
        self.reference_calls: list[tuple[UUID, int, int]] = []

    async def get_assigned_profile(
        self,
        character_ref: CharacterRef,
    ) -> LoadedCharacterTTSAssignment:
        self.calls.append(character_ref)
        if self.error is not None:
            raise self.error
        return cast(LoadedCharacterTTSAssignment, self.result)

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

    async def observe_availability(self, *_args: object) -> object:
        raise AssertionError("runtime resolution must not preflight availability")


@pytest.mark.asyncio
@pytest.mark.parametrize("assistant_kind", (None, "generic", "persona"))
async def test_generic_or_persona_speech_resolves_global_without_profile_work(
    assistant_kind: str | None,
) -> None:
    service = _FakeProfileService(result=object())
    resolver = CharacterTTSRequestResolver(service)

    resolved = await resolver.resolve(
        text="Hello there.",
        assistant_kind=assistant_kind,
        character_ref=None,
    )

    assert resolved == CharacterTTSRequestResolution(
        source="global",
        request=None,
        repository_generation=None,
        profile_id=None,
        profile_revision=None,
        reference=None,
    )
    assert service.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("assistant_kind", ("", "charcter", "generic "))
async def test_unknown_assistant_kind_fails_closed_without_profile_work(
    assistant_kind: str,
) -> None:
    service = _FakeProfileService(result=object())
    resolver = CharacterTTSRequestResolver(service)

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolver.resolve(
            text="Do not silently route malformed authorship.",
            assistant_kind=assistant_kind,
            character_ref=None,
        )

    assert caught.value.code == "authorship_invalid"
    assert caught.value.allow_global_override is False
    assert service.calls == []


@pytest.mark.asyncio
async def test_unassigned_character_resolves_global_after_one_exact_joined_read() -> (
    None
):
    character_ref = _character_ref()
    service = _FakeProfileService(
        LoadedCharacterTTSAssignment(repository_generation=9, snapshot=None)
    )
    resolver = CharacterTTSRequestResolver(service)

    resolved = await resolver.resolve(
        text="Hello there.",
        assistant_kind="character",
        character_ref=character_ref,
    )

    assert resolved.source == "global"
    assert resolved.request is None
    assert service.calls == [character_ref]


@pytest.mark.asyncio
async def test_assigned_character_freezes_one_exact_request_without_preflight() -> None:
    character_ref = _character_ref()
    service = _FakeProfileService(_loaded_assignment(character_ref, revision=6))
    resolver = CharacterTTSRequestResolver(service)

    resolved = await resolver.resolve(
        text="A character-authored reply.",
        assistant_kind="character",
        character_ref=character_ref,
    )

    assert resolved.source == "assigned"
    assert resolved.request == TTSRequest(
        provider_id="audio_cpp",
        model_id="supertonic-3",
        text="A character-authored reply.",
        voice="voice-7",
        response_format="wav",
        speed=1.0,
        options={},
    )
    assert resolved.repository_generation == 9
    assert resolved.profile_id == _PROFILE_ID
    assert resolved.profile_revision == 6
    assert service.calls == [character_ref]
    assert service.reference_calls == []


@pytest.mark.asyncio
async def test_assigned_character_freezes_exact_reference_under_profile_fences() -> None:
    character_ref = _character_ref()
    reference = _reference()
    service = _FakeProfileService(
        _loaded_assignment(
            character_ref,
            revision=6,
            voice_id=None,
            reference=reference.summary,
        )
    )
    service.reference_result = reference
    resolver = CharacterTTSRequestResolver(service)

    resolved = await resolver.resolve(
        text="A cloned character reply.",
        assistant_kind="character",
        character_ref=character_ref,
    )

    assert resolved.reference == reference
    assert resolved.profile_id == _PROFILE_ID
    assert resolved.profile_revision == 6
    assert resolved.repository_generation == 9
    assert service.reference_calls == [(_PROFILE_ID, 6, 9)]
    assert "Character private transcript" not in repr(resolved)


@pytest.mark.asyncio
async def test_assigned_character_reference_edit_race_fails_closed() -> None:
    character_ref = _character_ref()
    reference = _reference()
    service = _FakeProfileService(
        _loaded_assignment(
            character_ref,
            revision=6,
            voice_id=None,
            reference=reference.summary,
        )
    )
    service.reference_error = ProfileRepositoryError("stale")
    resolver = CharacterTTSRequestResolver(service)

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolver.resolve(
            text="Do not substitute a changed reference.",
            assistant_kind="character",
            character_ref=character_ref,
        )

    assert caught.value.code == "assignment_invalid"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert service.reference_calls == [(_PROFILE_ID, 6, 9)]


@pytest.mark.asyncio
async def test_assigned_character_reference_failure_severs_raw_exception() -> None:
    character_ref = _character_ref()
    reference = _reference()
    service = _FakeProfileService(
        _loaded_assignment(character_ref, reference=reference.summary)
    )
    service.reference_error = RuntimeError("PRIVATE_REFERENCE_CANARY")
    resolver = CharacterTTSRequestResolver(service)

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolver.resolve(
            text="Do not expose a reference failure.",
            assistant_kind="character",
            character_ref=character_ref,
        )

    assert caught.value.code == "profile_store_unavailable"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "PRIVATE_REFERENCE_CANARY" not in repr(caught.value)


@pytest.mark.asyncio
async def test_assigned_openai_profile_resolves_to_exact_request() -> None:
    character_ref = _character_ref()
    service = _FakeProfileService(
        _loaded_assignment(
            character_ref,
            revision=6,
            provider_id="openai",
            model_id="pocket-tts",
            voice_id="marius",
            response_format="mp3",
        )
    )
    resolver = CharacterTTSRequestResolver(service)

    resolved = await resolver.resolve(
        text="A character-authored reply.",
        assistant_kind="character",
        character_ref=character_ref,
    )

    assert resolved.source == "assigned"
    assert resolved.request is not None
    assert resolved.request.provider_id == "openai"
    assert resolved.request.model_id == "pocket-tts"
    assert resolved.request.voice == "marius"
    assert resolved.request.response_format == "mp3"


@pytest.mark.asyncio
async def test_explicit_override_is_global_and_performs_no_assignment_read() -> None:
    service = _FakeProfileService(result=object())
    resolver = CharacterTTSRequestResolver(service)

    resolved = resolver.resolve_explicit_global_override(
        text="Use the global voice once."
    )

    assert resolved.source == "explicit_override"
    assert resolved.request is None
    assert service.calls == []


@pytest.mark.asyncio
async def test_character_speech_without_authority_fails_closed() -> None:
    service = _FakeProfileService(result=object())
    resolver = CharacterTTSRequestResolver(service)

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolver.resolve(
            text="Do not silently treat this as generic.",
            assistant_kind="character",
            character_ref=None,
        )

    assert caught.value.code == "authority_missing"
    assert caught.value.allow_global_override is True
    assert "authority-a" not in str(caught.value)
    assert service.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "exception_category", "detail_code"),
    (
        (
            ProfileRepositoryError("unavailable"),
            "ProfileRepositoryError",
            "unavailable",
        ),
        (
            ProfileServiceError("operation_failed"),
            "ProfileServiceError",
            "operation_failed",
        ),
        (
            RuntimeError(
                "https://user:credential@example.test/private/path submitted text"
            ),
            "RuntimeError",
            "not_available",
        ),
    ),
)
async def test_profile_store_failures_are_bounded_and_never_become_global(
    error: BaseException,
    exception_category: str,
    detail_code: str,
) -> None:
    character_ref = _character_ref()
    service = _FakeProfileService(result=object())
    service.error = error
    resolver = CharacterTTSRequestResolver(service)
    log_messages: list[str] = []

    sink_id = logger.add(log_messages.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(CharacterTTSResolutionError) as caught:
            await resolver.resolve(
                text="Private character reply.",
                assistant_kind="character",
                character_ref=character_ref,
            )
    finally:
        logger.remove(sink_id)

    assert caught.value.code == "profile_store_unavailable"
    assert caught.value.allow_global_override is True
    rendered = str(caught.value)
    rendered_logs = "\n".join(log_messages)
    assert "operation=character_tts_resolution" in rendered_logs
    assert "outcome_code=profile_store_unavailable" in rendered_logs
    assert f"exception_category={exception_category}" in rendered_logs
    assert f"detail_code={detail_code}" in rendered_logs
    for secret in (
        character_ref.authority_id,
        "credential",
        "example.test",
        "/private/path",
        "submitted text",
    ):
        assert secret not in rendered
        assert secret not in rendered_logs
    assert service.calls == [character_ref]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "exception_category"),
    (("wrong-type", "TypeError"), ("wrong-character", "ValueError")),
)
async def test_malformed_joined_assignment_fails_closed_without_fallback(
    case: str,
    exception_category: str,
) -> None:
    character_ref = _character_ref()
    if case == "wrong-type":
        result: object = object()
    else:
        result = _loaded_assignment(
            _character_ref(authority_id="server-user-v1:authority-b")
        )
    service = _FakeProfileService(result=result)
    resolver = CharacterTTSRequestResolver(service)
    log_messages: list[str] = []

    sink_id = logger.add(log_messages.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(CharacterTTSResolutionError) as caught:
            await resolver.resolve(
                text="Private character reply.",
                assistant_kind="character",
                character_ref=character_ref,
            )
    finally:
        logger.remove(sink_id)

    assert caught.value.code == "assignment_invalid"
    assert caught.value.allow_global_override is True
    rendered_logs = "\n".join(log_messages)
    assert "operation=character_tts_resolution" in rendered_logs
    assert "outcome_code=assignment_invalid" in rendered_logs
    assert f"exception_category={exception_category}" in rendered_logs
    assert "detail_code=not_available" in rendered_logs
    assert character_ref.authority_id not in rendered_logs
    assert "Private character reply." not in rendered_logs
    assert service.calls == [character_ref]


@pytest.mark.asyncio
async def test_same_character_id_isolated_by_source_and_authority() -> None:
    first_ref = _character_ref(
        source="local",
        authority_id="local-authority-a",
        character_id="42",
    )
    second_ref = _character_ref(
        source="server",
        authority_id="server-user-v1:authority-b",
        character_id="42",
    )
    first_service = _FakeProfileService(_loaded_assignment(first_ref))
    second_service = _FakeProfileService(
        LoadedCharacterTTSAssignment(repository_generation=9, snapshot=None)
    )

    assigned = await CharacterTTSRequestResolver(first_service).resolve(
        text="Assigned.",
        assistant_kind="character",
        character_ref=first_ref,
    )
    unassigned = await CharacterTTSRequestResolver(second_service).resolve(
        text="Unassigned.",
        assistant_kind="character",
        character_ref=second_ref,
    )

    assert assigned.source == "assigned"
    assert unassigned.source == "global"
    assert first_service.calls == [first_ref]
    assert second_service.calls == [second_ref]


@pytest.mark.parametrize(
    "code",
    ("default_profile_missing", "default_profile_store_unavailable"),
)
def test_default_profile_codes_are_bounded_and_overridable(code: str) -> None:
    """Slice-3 task-4: the default-profile codes reuse this bounded mechanism."""
    error = CharacterTTSResolutionError(code)  # type: ignore[arg-type]

    assert error.code == code
    assert error.allow_global_override is True
    assert "character" not in str(error).lower()
    assert "default voice" in str(error).lower()
    assert error.domain == "default_profile"


@pytest.mark.parametrize(
    "code",
    (
        "assignment_invalid",
        "authorship_invalid",
        "authority_missing",
        "profile_store_unavailable",
    ),
)
def test_character_domain_codes_report_the_character_domain(code: str) -> None:
    """Review round 2: the dialog must be able to tell the domains apart."""
    error = CharacterTTSResolutionError(code)  # type: ignore[arg-type]

    assert error.domain == "character"


def test_default_profile_resolution_accepts_the_assigned_shaped_request() -> None:
    """`source="default_profile"` reuses "assigned"'s exact-request shape."""
    character_ref_free_request = TTSRequest(
        provider_id="openai",
        model_id="gpt-4o-mini-tts",
        text="Read this.",
        voice="verse",
        response_format="mp3",
        speed=1.0,
        options={},
    )

    resolution = CharacterTTSRequestResolution(
        source="default_profile",
        request=character_ref_free_request,
        repository_generation=2,
        profile_id=_PROFILE_ID,
        profile_revision=1,
    )

    assert resolution.source == "default_profile"
    assert resolution.request is character_ref_free_request

    with pytest.raises(ValueError, match="default_profile resolution"):
        CharacterTTSRequestResolution(
            source="default_profile",
            request=None,
            repository_generation=None,
            profile_id=None,
            profile_revision=None,
        )


@pytest.mark.asyncio
async def test_inconsistent_noncharacter_authorship_is_rejected() -> None:
    service = _FakeProfileService(result=object())
    resolver = CharacterTTSRequestResolver(service)

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolver.resolve(
            text="Forged ownership.",
            assistant_kind="generic",
            character_ref=_character_ref(),
        )

    assert caught.value.code == "authorship_invalid"
    assert caught.value.allow_global_override is False
    assert service.calls == []
