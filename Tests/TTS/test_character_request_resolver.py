from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest

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


def _profile(*, revision: int = 4) -> TTSGenerationProfile:
    draft = TTSProfileDraft(
        display_name="Mara",
        provider_id="audio_cpp",
        model_id="supertonic-3",
        voice_id="voice-7",
        response_format="wav",
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
    )


def _loaded_assignment(
    character_ref: CharacterRef,
    *,
    generation: int = 9,
    revision: int = 4,
) -> LoadedCharacterTTSAssignment:
    profile = _profile(revision=revision)
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

    async def get_assigned_profile(
        self,
        character_ref: CharacterRef,
    ) -> LoadedCharacterTTSAssignment:
        self.calls.append(character_ref)
        if self.error is not None:
            raise self.error
        return cast(LoadedCharacterTTSAssignment, self.result)

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
    )
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
    "error",
    (
        ProfileRepositoryError("unavailable"),
        ProfileServiceError("operation_failed"),
        RuntimeError(
            "https://user:credential@example.test/private/path submitted text"
        ),
    ),
)
async def test_profile_store_failures_are_bounded_and_never_become_global(
    error: BaseException,
) -> None:
    character_ref = _character_ref()
    service = _FakeProfileService(result=object())
    service.error = error
    resolver = CharacterTTSRequestResolver(service)

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolver.resolve(
            text="Private character reply.",
            assistant_kind="character",
            character_ref=character_ref,
        )

    assert caught.value.code == "profile_store_unavailable"
    assert caught.value.allow_global_override is True
    rendered = str(caught.value)
    for secret in (
        character_ref.authority_id,
        "credential",
        "example.test",
        "/private/path",
        "submitted text",
    ):
        assert secret not in rendered
    assert service.calls == [character_ref]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("wrong-type", "wrong-character"))
async def test_malformed_joined_assignment_fails_closed_without_fallback(
    case: str,
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

    with pytest.raises(CharacterTTSResolutionError) as caught:
        await resolver.resolve(
            text="Private character reply.",
            assistant_kind="character",
            character_ref=character_ref,
        )

    assert caught.value.code == "assignment_invalid"
    assert caught.value.allow_global_override is True
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
