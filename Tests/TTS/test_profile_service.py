from __future__ import annotations

import asyncio
import traceback
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import UUID

import pytest

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSConfigurationRevisionError,
    TTSModelInfo,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.playground_types import (
    STTSGeneratedAudio,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.TTS.profile_errors import (
    ProfileRepositoryError,
    ProfileServiceError,
    ProfileValidationError,
)
from tldw_chatbook.TTS.profile_service import (
    LoadedTTSProfile,
    TTSPlaygroundSelectionPreset,
    TTSProfileAvailability,
    TTSProfileAvailabilitySnapshot,
    TTSProfilePageSnapshot,
    TTSProfileService,
)
from tldw_chatbook.TTS.profile_types import (
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileDraft,
    TTSProfilePage,
)

_CREATED_AT = datetime(2026, 7, 27, 12, tzinfo=UTC)
_PROFILE_ID = UUID("11111111-1111-4111-8111-111111111111")
_DUPLICATE_ID = UUID("22222222-2222-4222-8222-222222222222")


def _profile(
    *,
    profile_id: UUID = _PROFILE_ID,
    display_name: str = "Narrator",
    provider_id: str = "audio_cpp",
    model_id: str = "model-a",
    voice_id: str | None = None,
    response_format: str = "wav",
    speed: float = 1.0,
    options: dict[str, Any] | None = None,
    revision: int = 1,
) -> TTSGenerationProfile:
    draft = TTSProfileDraft(
        display_name=display_name,
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        response_format=response_format,
        speed=speed,
        options={} if options is None else options,
    )
    return TTSGenerationProfile(
        profile_id=profile_id,
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


def _forged_profile(
    profile: TTSGenerationProfile,
    **updates: object,
) -> TTSGenerationProfile:
    """Build an adversarial already-loaded value without domain revalidation."""

    forged = object.__new__(TTSGenerationProfile)
    for profile_field in fields(TTSGenerationProfile):
        object.__setattr__(
            forged,
            profile_field.name,
            updates.get(profile_field.name, getattr(profile, profile_field.name)),
        )
    return forged


def _model(
    model_id: str,
    *,
    formats: tuple[str, ...] = ("wav",),
    server_default: bool = True,
) -> TTSModelInfo:
    return TTSModelInfo(
        model_id=model_id,
        display_name=model_id,
        family="test",
        upstream_mode="tts",
        formats=formats,
        voices=(),
        supports_speed=False,
        omit_voice_uses_server_default=server_default,
    )


def _capability_snapshot(
    *,
    configuration_revision: int = 3,
    state: str = "complete",
    models: tuple[TTSModelInfo, ...] = (),
    voice_results: dict[str, TTSVoiceDiscoveryResult] | None = None,
    catalog_revision: int = 9,
    fresh: bool = True,
    health_state: str = "available",
) -> TTSNativeCapabilitySnapshot:
    return TTSNativeCapabilitySnapshot(
        provider_id="audio_cpp",
        configuration_revision=configuration_revision,
        state=state,  # type: ignore[arg-type]
        catalog=TTSProviderCatalog(
            provider_id="audio_cpp",
            revision=catalog_revision,
            health=ProviderHealth(
                state=health_state,  # type: ignore[arg-type]
                fresh=fresh,
            ),
            models=models,
        ),
        voice_results={} if voice_results is None else voice_results,
    )


def _artifact(
    *,
    selection: TTSRequestedSelectionSnapshot | None = None,
) -> STTSGeneratedAudio:
    return STTSGeneratedAudio(
        path=Path("/private/secret/result.wav"),
        provider_id="legacy-response-provider",
        model_id="mutable-response-model",
        voice_id="mutable-response-voice",
        source_text="private submitted text",
        operation_id="operation",
        audio_format="mp3",
        content_type="secret/content-type",
        metadata={"endpoint": "https://user:credential@example.test"},
        requested_selection=selection,
    )


def _selection(
    *,
    model_id: str = "selected-model",
    voice_id: str | None = "selected-voice",
    configuration_revision: int = 3,
) -> TTSRequestedSelectionSnapshot:
    return TTSRequestedSelectionSnapshot(
        provider_id="audio_cpp",
        model_id=model_id,
        voice_id=voice_id,
        response_format="wav",
        speed=1.0,
        options={},
        configuration_revision=configuration_revision,
    )


class _FakeRepository:
    def __init__(self) -> None:
        self.generation = 7
        self.calls: list[tuple[str, object]] = []
        self.page = TTSProfilePage(profiles=(), total=0)
        self.created_profile_id = _PROFILE_ID
        self.create_error: BaseException | None = None
        self.update_error: BaseException | None = None
        self.delete_error: BaseException | None = None
        self.count_value = 0
        self.count_generation: int | None = None
        self.advance_generation_during_count = False
        self.coordinator_probe: Callable[[], bool] | None = None
        self.coordinator_active_at_repository_calls: list[bool] = []

    def _record_coordinator_state(self) -> None:
        self.coordinator_active_at_repository_calls.append(
            False if self.coordinator_probe is None else self.coordinator_probe()
        )

    async def list_profiles(
        self,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ProfileStoreResult[TTSProfilePage]:
        self._record_coordinator_state()
        self.calls.append(("list", (search, limit, offset)))
        return ProfileStoreResult(generation=self.generation, value=self.page)

    async def create_profile(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID | None = None,
        *,
        expected_generation: int | None = None,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        self._record_coordinator_state()
        self.calls.append(
            ("create", (draft, profile_id, expected_generation, self.generation))
        )
        if self.create_error is not None:
            raise self.create_error
        persisted = _profile(
            profile_id=self.created_profile_id if profile_id is None else profile_id,
            display_name=draft.display_name,
            provider_id=draft.provider_id,
            model_id=draft.model_id,
            voice_id=draft.voice_id,
            response_format=draft.response_format,
            speed=draft.speed,
            options=dict(draft.options),
        )
        return ProfileStoreResult(generation=self.generation, value=persisted)

    async def update_profile(
        self,
        profile_id: UUID,
        expected_revision: int,
        draft: TTSProfileDraft,
        *,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        self._record_coordinator_state()
        self.calls.append(
            (
                "update",
                (
                    profile_id,
                    expected_revision,
                    draft,
                    expected_generation,
                    self.generation,
                ),
            )
        )
        if self.update_error is not None:
            raise self.update_error
        return ProfileStoreResult(
            generation=self.generation,
            value=_profile(
                profile_id=profile_id,
                display_name=draft.display_name,
                provider_id=draft.provider_id,
                model_id=draft.model_id,
                voice_id=draft.voice_id,
                response_format=draft.response_format,
                speed=draft.speed,
                options=dict(draft.options),
                revision=expected_revision + 1,
            ),
        )

    async def delete_profile(
        self,
        profile_id: UUID,
        *,
        expected_generation: int,
    ) -> ProfileStoreResult[None]:
        self._record_coordinator_state()
        self.calls.append(
            (
                "delete",
                (profile_id, expected_generation, self.generation),
            )
        )
        if self.delete_error is not None:
            raise self.delete_error
        return ProfileStoreResult(generation=self.generation, value=None)

    async def assignment_count(
        self,
        profile_id: UUID,
    ) -> ProfileStoreResult[int]:
        self._record_coordinator_state()
        self.calls.append(("count", (profile_id, self.generation)))
        generation = (
            self.generation if self.count_generation is None else self.count_generation
        )
        result = ProfileStoreResult(
            generation=generation,
            value=self.count_value,
        )
        if self.advance_generation_during_count:
            self.generation += 1
        return result


class _ExplodingSequence(Sequence[object]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> object:
        raise RuntimeError(
            "https://user:credential@example.test/private/path submitted text"
        )


class _FakeTTSService:
    def __init__(
        self,
        snapshot: TTSNativeCapabilitySnapshot | None = None,
    ) -> None:
        self.revision = 3
        self.snapshot = (
            _capability_snapshot(models=(_model("selected-model"),))
            if snapshot is None
            else snapshot
        )
        self.capability_calls: list[tuple[str, tuple[str, ...]]] = []
        self.revision_reads: list[str] = []
        self.revision_decisions: list[tuple[str, int]] = []
        self.stale_decision = False
        self.reconfigure_after_decision = False
        self.capability_hook: Callable[[], None] | None = None
        self.read_side_active = False

    async def get_native_capability_snapshot(
        self,
        provider_id: str,
        exact_voice_model_ids: tuple[str, ...],
    ) -> TTSNativeCapabilitySnapshot:
        self.capability_calls.append((provider_id, tuple(exact_voice_model_ids)))
        if self.capability_hook is not None:
            self.capability_hook()
        return self.snapshot

    def configuration_revision(self, provider_id: str) -> int:
        self.revision_reads.append(provider_id)
        return self.revision

    async def require_current_configuration_revision(
        self,
        provider_id: str,
        expected_revision: int,
    ) -> None:
        self.revision_decisions.append((provider_id, expected_revision))
        self.read_side_active = True
        try:
            await asyncio.sleep(0)
            if self.stale_decision or self.revision != expected_revision:
                raise TTSConfigurationRevisionError(
                    "https://user:credential@example.test/private/path"
                )
            if self.reconfigure_after_decision:
                self.revision += 1
        finally:
            self.read_side_active = False

    def provider_descriptors(self) -> tuple[object, ...]:
        raise AssertionError("native descriptors are not profile allowlist authority")

    async def get_catalog(self, _provider_id: str) -> object:
        raise AssertionError("profile save must not call catalog discovery")

    async def get_voices(self, _provider_id: str, _model_id: str) -> object:
        raise AssertionError("profile save must not call voice discovery")


def _service(
    *,
    repository: _FakeRepository | None = None,
    tts_service: _FakeTTSService | None = None,
) -> tuple[TTSProfileService, _FakeRepository, _FakeTTSService]:
    selected_repository = _FakeRepository() if repository is None else repository
    selected_tts_service = _FakeTTSService() if tts_service is None else tts_service
    selected_repository.coordinator_probe = lambda: (
        selected_tts_service.read_side_active
    )
    return (
        TTSProfileService(
            selected_repository,  # type: ignore[arg-type]
            selected_tts_service,  # type: ignore[arg-type]
        ),
        selected_repository,
        selected_tts_service,
    )


def _assert_safe_service_error(
    error: ProfileServiceError,
    code: str,
    *secrets: str,
) -> None:
    assert type(error) is ProfileServiceError
    assert error.code == code
    assert str(error) == f"TTS profile service failed: {code}"
    assert error.__cause__ is None
    assert error.__context__ is None
    visible = " ".join(
        (
            str(error),
            repr(error),
            "".join(traceback.format_exception(error)),
            *(str(note) for note in getattr(error, "__notes__", ())),
        )
    )
    for secret in secrets:
        assert secret not in visible


def test_service_values_are_immutable_and_defensively_freeze_containers() -> None:
    profile = _profile()
    source_profiles = [profile]
    page = TTSProfilePageSnapshot(
        repository_generation=7,
        profiles=source_profiles,
        total=1,
    )
    source_profiles.clear()
    availability = TTSProfileAvailability(
        profile_id=profile.profile_id,
        state="available",
        recovery_action="none",
    )
    source_availability = [availability]
    snapshot = TTSProfileAvailabilitySnapshot(
        repository_generation=7,
        configuration_revision=3,
        catalog_revision=9,
        profiles=source_availability,
    )
    source_availability.clear()
    source_options: dict[str, Any] = {"nested": ["value"]}
    preset = TTSPlaygroundSelectionPreset(
        provider_id="future_native",
        model_id="model",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options=source_options,
        availability="unavailable",
    )
    source_options["nested"].append("changed")

    assert page.profiles == (profile,)
    assert snapshot.profiles == (availability,)
    assert preset.options == {"nested": ("value",)}
    assert isinstance(preset.options, MappingProxyType)
    with pytest.raises(FrozenInstanceError):
        page.total = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        preset.options["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize(
    ("value_type", "kwargs", "code"),
    (
        (
            TTSProfilePageSnapshot,
            {"repository_generation": -1, "profiles": (), "total": 0},
            "generation",
        ),
        (
            LoadedTTSProfile,
            {"repository_generation": True, "profile": _profile()},
            "generation",
        ),
        (
            TTSProfileAvailability,
            {
                "profile_id": _PROFILE_ID,
                "state": "unknown",
                "recovery_action": "none",
            },
            "availability",
        ),
        (
            TTSProfileAvailability,
            {
                "profile_id": _PROFILE_ID,
                "state": "available",
                "recovery_action": "edit",
            },
            "recovery_action",
        ),
        (
            TTSProfileAvailabilitySnapshot,
            {
                "repository_generation": 1,
                "configuration_revision": 1,
                "catalog_revision": -1,
                "profiles": (),
            },
            "catalog_revision",
        ),
    ),
)
def test_service_values_reject_unbounded_state(
    value_type: type[object],
    kwargs: dict[str, object],
    code: str,
) -> None:
    with pytest.raises(ProfileValidationError) as caught:
        value_type(**kwargs)

    assert caught.value.code == code


def test_service_values_do_not_retain_hostile_container_errors() -> None:
    with pytest.raises(ProfileValidationError) as caught:
        TTSProfilePageSnapshot(
            repository_generation=1,
            profiles=_ExplodingSequence(),  # type: ignore[arg-type]
            total=1,
        )

    assert caught.value.code == "profiles"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    visible = " ".join(
        (
            str(caught.value),
            repr(caught.value),
            "".join(traceback.format_exception(caught.value)),
        )
    )
    assert "credential" not in visible
    assert "/private/path" not in visible
    assert "submitted text" not in visible


@pytest.mark.asyncio
async def test_list_profiles_delegates_with_fixed_fifty_row_limit() -> None:
    service, repository, _tts_service = _service()
    repository.page = TTSProfilePage(profiles=(_profile(),), total=81)

    page = await service.list_profiles(search=" nar ", offset=50)

    assert repository.calls == [("list", (" nar ", 50, 50))]
    assert page.repository_generation == 7
    assert page.profiles == repository.page.profiles
    assert page.total == 81


@pytest.mark.asyncio
async def test_availability_applies_exact_allowlist_before_capability_lookup() -> None:
    supported = _profile(
        profile_id=UUID(int=1),
        display_name="Supported",
        model_id="model-a",
        voice_id="voice-a",
    )
    default_voice = _profile(
        profile_id=UUID(int=2),
        display_name="Default",
        model_id="model-b",
    )
    future_native = _profile(
        profile_id=UUID(int=3),
        display_name="Future native",
        provider_id="future_native",
        model_id="future-model",
        voice_id="future-voice",
    )
    invalid_format = _forged_profile(
        supported,
        profile_id=UUID(int=4),
        display_name="Invalid format",
        normalized_name="invalid format",
        response_format="mp3",
    )
    invalid_speed = _forged_profile(
        supported,
        profile_id=UUID(int=5),
        display_name="Invalid speed",
        normalized_name="invalid speed",
        speed=1.25,
    )
    invalid_options = _forged_profile(
        supported,
        profile_id=UUID(int=6),
        display_name="Invalid options",
        normalized_name="invalid options",
        options=MappingProxyType({"endpoint": "secret"}),
    )
    voice_result = TTSVoiceDiscoveryResult(
        provider_id="audio_cpp",
        model_id="model-a",
        catalog_revision=9,
        voices=("voice-a",),
        state="complete",
    )
    tts_service = _FakeTTSService(
        _capability_snapshot(
            models=(_model("model-a"), _model("model-b")),
            voice_results={"model-a": voice_result},
        )
    )
    service, repository, tts_service = _service(tts_service=tts_service)
    page = TTSProfilePageSnapshot(
        repository_generation=repository.generation,
        profiles=(
            supported,
            default_voice,
            future_native,
            invalid_format,
            invalid_speed,
            invalid_options,
        ),
        total=6,
    )

    observed = await service.observe_availability(page)

    assert tts_service.capability_calls == [("audio_cpp", ("model-a",))]
    assert tuple(item.state for item in observed.profiles) == (
        "available",
        "available",
        "unavailable",
        "unavailable",
        "unavailable",
        "unavailable",
    )
    assert tuple(item.recovery_action for item in observed.profiles) == (
        "none",
        "none",
        "edit",
        "edit",
        "edit",
        "edit",
    )
    assert observed.repository_generation == repository.generation
    assert observed.configuration_revision == 3
    assert observed.catalog_revision == 9


@pytest.mark.asyncio
async def test_all_unsupported_profiles_do_not_observe_capabilities() -> None:
    unsupported = _profile(
        provider_id="future_native",
        model_id="model",
        voice_id="voice",
    )
    service, repository, tts_service = _service()

    observed = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(unsupported,),
            total=1,
        )
    )

    assert tts_service.capability_calls == []
    assert observed.profiles == (
        TTSProfileAvailability(
            profile_id=unsupported.profile_id,
            state="unavailable",
            recovery_action="edit",
        ),
    )


@pytest.mark.asyncio
async def test_availability_rejects_stale_repository_generation_before_tts_work() -> (
    None
):
    service, repository, tts_service = _service()

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.observe_availability(
            TTSProfilePageSnapshot(
                repository_generation=repository.generation - 1,
                profiles=(_profile(),),
                total=1,
            )
        )

    assert caught.value.code == "stale"
    assert tts_service.capability_calls == []
    assert tts_service.revision_reads == []


@pytest.mark.asyncio
async def test_availability_deduplicates_only_exact_voice_supported_models() -> None:
    first = _profile(
        profile_id=UUID(int=10),
        display_name="First",
        model_id="shared",
        voice_id="one",
    )
    second = _profile(
        profile_id=UUID(int=11),
        display_name="Second",
        model_id="shared",
        voice_id="two",
    )
    server_default = _profile(
        profile_id=UUID(int=12),
        display_name="Default",
        model_id="default-model",
    )
    unsupported = _profile(
        profile_id=UUID(int=13),
        display_name="Unsupported",
        provider_id="future_native",
        model_id="other",
        voice_id="other-voice",
    )
    voice_result = TTSVoiceDiscoveryResult(
        provider_id="audio_cpp",
        model_id="shared",
        catalog_revision=9,
        voices=("one",),
        state="complete",
    )
    tts_service = _FakeTTSService(
        _capability_snapshot(
            models=(_model("shared"), _model("default-model")),
            voice_results={"shared": voice_result},
        )
    )
    service, repository, tts_service = _service(tts_service=tts_service)

    observed = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(first, second, server_default, unsupported),
            total=4,
        )
    )

    assert tts_service.capability_calls == [("audio_cpp", ("shared",))]
    assert tuple(item.state for item in observed.profiles) == (
        "available",
        "unavailable",
        "available",
        "unavailable",
    )


@pytest.mark.asyncio
async def test_create_from_artifact_rejects_legacy_or_missing_provenance_safely() -> (
    None
):
    service, repository, tts_service = _service()

    with pytest.raises(ProfileServiceError) as caught:
        await service.create_from_artifact("Saved", _artifact())

    _assert_safe_service_error(
        caught.value,
        "artifact_ineligible",
        "private submitted text",
        "/private/secret/result.wav",
        "credential",
    )
    assert repository.calls == []
    assert tts_service.revision_decisions == []
    assert tts_service.capability_calls == []


@pytest.mark.asyncio
async def test_create_from_artifact_uses_only_immutable_requested_selection() -> None:
    service, repository, tts_service = _service()
    selection = _selection()

    loaded = await service.create_from_artifact(
        " Saved voice ", _artifact(selection=selection)
    )

    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert tts_service.capability_calls == []
    assert len(repository.calls) == 1
    call_name, call_value = repository.calls[0]
    assert call_name == "create"
    draft, profile_id, expected_generation, generation_at_call = call_value  # type: ignore[misc]
    assert draft == TTSProfileDraft(
        display_name="Saved voice",
        provider_id="audio_cpp",
        model_id="selected-model",
        voice_id="selected-voice",
        response_format="wav",
        speed=1.0,
        options={},
    )
    assert profile_id is None
    assert expected_generation is None
    assert generation_at_call == 7
    assert loaded.repository_generation == 7
    assert loaded.profile.model_id == "selected-model"
    assert loaded.profile.voice_id == "selected-voice"
    assert loaded.profile.provider_id != "legacy-response-provider"
    assert repository.coordinator_active_at_repository_calls == [False]


@pytest.mark.asyncio
async def test_create_from_artifact_maps_stale_revision_to_bounded_error() -> None:
    tts_service = _FakeTTSService()
    tts_service.stale_decision = True
    service, repository, tts_service = _service(tts_service=tts_service)

    with pytest.raises(ProfileServiceError) as caught:
        await service.create_from_artifact(
            "Saved",
            _artifact(selection=_selection()),
        )

    _assert_safe_service_error(
        caught.value,
        "stale_configuration",
        "credential",
        "example.test",
        "/private/path",
    )
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert repository.calls == []


@pytest.mark.asyncio
async def test_later_reconfiguration_does_not_roll_back_admitted_create() -> None:
    tts_service = _FakeTTSService()
    tts_service.reconfigure_after_decision = True
    service, repository, tts_service = _service(tts_service=tts_service)

    loaded = await service.create_from_artifact(
        "Saved",
        _artifact(selection=_selection()),
    )

    assert loaded.profile.display_name == "Saved"
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert tts_service.revision == 4
    assert [name for name, _value in repository.calls] == ["create"]


def test_profile_service_error_codes_are_bounded_and_value_independent() -> None:
    error = ProfileServiceError(
        "https://user:credential@example.test/private/path submitted text"
    )

    _assert_safe_service_error(
        error,
        "operation_failed",
        "credential",
        "/private/path",
        "submitted text",
    )


@pytest.mark.asyncio
async def test_rename_only_is_derived_from_loaded_generation_fields() -> None:
    unverified = _capability_snapshot(
        state="unverified",
        models=(_model("model-a"),),
    )
    tts_service = _FakeTTSService(unverified)
    service, repository, tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(revision=4),
    )
    renamed = TTSProfileDraft(
        display_name="Renamed",
        provider_id=loaded.profile.provider_id,
        model_id=loaded.profile.model_id,
        voice_id=loaded.profile.voice_id,
        response_format=loaded.profile.response_format,
        speed=loaded.profile.speed,
        options=loaded.profile.options,
    )

    updated = await service.update_profile(loaded, renamed)

    assert updated.profile.display_name == "Renamed"
    assert updated.profile.revision == 5
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert repository.calls == [
        (
            "update",
            (
                loaded.profile.profile_id,
                4,
                renamed,
                loaded.repository_generation,
                repository.generation,
            ),
        )
    ]


@pytest.mark.asyncio
async def test_generation_edit_requires_fresh_authority_and_revision_decision() -> None:
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-b"),)))
    service, repository, tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(revision=2),
    )
    changed = TTSProfileDraft(
        display_name="Changed",
        provider_id="audio_cpp",
        model_id="model-b",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
    )

    updated = await service.update_profile(loaded, changed)

    assert updated.profile.model_id == "model-b"
    assert tts_service.capability_calls == [("audio_cpp", ())]
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert repository.calls[0][0] == "update"
    (
        profile_id,
        expected_revision,
        submitted,
        expected_generation,
        _generation_at_call,
    ) = repository.calls[0][1]  # type: ignore[misc]
    assert profile_id == loaded.profile.profile_id
    assert expected_revision == loaded.profile.revision
    assert submitted is changed
    assert expected_generation == loaded.repository_generation
    assert repository.coordinator_active_at_repository_calls == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot", "code"),
    (
        (
            _capability_snapshot(
                state="unverified",
                models=(_model("model-b"),),
            ),
            "profile_unverified",
        ),
        (
            _capability_snapshot(
                models=(_model("different-model"),),
            ),
            "profile_unavailable",
        ),
    ),
)
async def test_generation_edit_rejects_non_authoritative_capability(
    snapshot: TTSNativeCapabilitySnapshot,
    code: str,
) -> None:
    tts_service = _FakeTTSService(snapshot)
    service, repository, tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    changed = TTSProfileDraft(
        display_name="Changed",
        provider_id="audio_cpp",
        model_id="model-b",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
    )

    with pytest.raises(ProfileServiceError) as caught:
        await service.update_profile(loaded, changed)

    _assert_safe_service_error(caught.value, code)
    assert tts_service.capability_calls == [("audio_cpp", ())]
    assert tts_service.revision_decisions == []
    assert repository.calls == []


@pytest.mark.asyncio
async def test_generation_edit_rejects_unreviewed_native_provider() -> None:
    service, repository, tts_service = _service()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    changed = TTSProfileDraft(
        display_name="Future",
        provider_id="future_native",
        model_id="model",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
    )

    with pytest.raises(ProfileServiceError) as caught:
        await service.update_profile(loaded, changed)

    _assert_safe_service_error(caught.value, "unsupported_profile")
    assert tts_service.capability_calls == []
    assert repository.calls == []


@pytest.mark.asyncio
async def test_duplicate_copies_immutable_loaded_version_at_revision_one() -> None:
    voice = TTSVoiceDiscoveryResult(
        provider_id="audio_cpp",
        model_id="model-a",
        catalog_revision=9,
        voices=("voice-a",),
        state="complete",
    )
    tts_service = _FakeTTSService(
        _capability_snapshot(
            models=(_model("model-a"),),
            voice_results={"model-a": voice},
        )
    )
    tts_service.reconfigure_after_decision = True
    repository = _FakeRepository()
    repository.created_profile_id = _DUPLICATE_ID
    service, repository, tts_service = _service(
        repository=repository,
        tts_service=tts_service,
    )
    source = _profile(voice_id="voice-a", revision=8)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=source,
    )

    duplicate = await service.duplicate_profile(loaded, "Duplicate")

    assert duplicate.profile.profile_id == _DUPLICATE_ID
    assert duplicate.profile.profile_id != source.profile_id
    assert duplicate.profile.revision == 1
    assert duplicate.profile.model_id == source.model_id
    assert duplicate.profile.voice_id == source.voice_id
    assert tts_service.capability_calls == [("audio_cpp", ("model-a",))]
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert tts_service.revision == 4
    assert repository.calls[0][0] == "create"
    draft, profile_id, expected_generation, _generation_at_call = repository.calls[0][1]  # type: ignore[misc]
    assert draft.display_name == "Duplicate"
    assert draft.model_id == source.model_id
    assert draft.voice_id == source.voice_id
    assert profile_id is None
    assert expected_generation == loaded.repository_generation
    assert repository.coordinator_active_at_repository_calls == [False]


@pytest.mark.asyncio
async def test_duplicate_requires_fresh_authoritative_capability() -> None:
    tts_service = _FakeTTSService(
        _capability_snapshot(
            state="unverified",
            models=(_model("model-a"),),
        )
    )
    service, repository, tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )

    with pytest.raises(ProfileServiceError) as caught:
        await service.duplicate_profile(loaded, "Duplicate")

    _assert_safe_service_error(caught.value, "profile_unverified")
    assert tts_service.revision_decisions == []
    assert repository.calls == []


@pytest.mark.asyncio
async def test_assignment_count_rejects_generation_change_before_publication() -> None:
    service, repository, _tts_service = _service()
    repository.count_value = 6
    repository.advance_generation_during_count = True
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.assignment_count(loaded)

    assert caught.value.code == "stale"
    assert repository.calls[0][0] == "count"


@pytest.mark.asyncio
async def test_assignment_count_returns_only_loaded_repository_generation() -> None:
    service, repository, _tts_service = _service()
    repository.count_value = 6
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )

    count = await service.assignment_count(loaded)

    assert count == 6
    assert repository.calls == [("count", (loaded.profile.profile_id, 7))]


@pytest.mark.asyncio
async def test_delete_supplies_loaded_generation_and_leaves_protection_to_repository() -> (
    None
):
    service, repository, _tts_service = _service()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    repository.delete_error = ProfileRepositoryError("conflict")

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.delete_profile(loaded)

    assert caught.value.code == "conflict"
    assert repository.calls == [
        (
            "delete",
            (
                loaded.profile.profile_id,
                loaded.repository_generation,
                repository.generation,
            ),
        )
    ]


@pytest.mark.asyncio
async def test_delete_succeeds_without_capability_or_count_preflight() -> None:
    service, repository, tts_service = _service()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )

    result = await service.delete_profile(loaded)

    assert result is None
    assert [name for name, _value in repository.calls] == ["delete"]
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []


@pytest.mark.asyncio
async def test_availability_rejects_generation_change_during_capability_work() -> None:
    service, repository, tts_service = _service()
    tts_service.capability_hook = lambda: setattr(
        repository,
        "generation",
        repository.generation + 1,
    )
    page = TTSProfilePageSnapshot(
        repository_generation=repository.generation,
        profiles=(_profile(),),
        total=1,
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.observe_availability(page)

    assert caught.value.code == "stale"
    assert tts_service.capability_calls == [("audio_cpp", ())]


@pytest.mark.asyncio
async def test_availability_rejects_snapshot_after_configuration_change() -> None:
    tts_service = _FakeTTSService(
        _capability_snapshot(
            configuration_revision=3,
            models=(_model("model-a"),),
        )
    )
    tts_service.revision = 4
    service, repository, _tts_service = _service(tts_service=tts_service)

    with pytest.raises(ProfileServiceError) as caught:
        await service.observe_availability(
            TTSProfilePageSnapshot(
                repository_generation=repository.generation,
                profiles=(_profile(),),
                total=1,
            )
        )

    _assert_safe_service_error(caught.value, "stale_configuration")


@pytest.mark.asyncio
async def test_unverified_snapshot_classifies_each_row_from_its_exact_evidence() -> (
    None
):
    profiles = (
        _profile(
            profile_id=UUID(int=31),
            display_name="Default available",
            model_id="default-ok",
        ),
        _profile(
            profile_id=UUID(int=32),
            display_name="Default unavailable",
            model_id="default-no",
        ),
        _profile(
            profile_id=UUID(int=33),
            display_name="Format unavailable",
            model_id="format-no",
        ),
        _profile(
            profile_id=UUID(int=34),
            display_name="Model unavailable",
            model_id="catalog-missing",
        ),
        _profile(
            profile_id=UUID(int=35),
            display_name="Voice available",
            model_id="voice-present",
            voice_id="wanted",
        ),
        _profile(
            profile_id=UUID(int=36),
            display_name="Voice unavailable",
            model_id="voice-absent",
            voice_id="wanted",
        ),
        _profile(
            profile_id=UUID(int=37),
            display_name="Voice model unavailable",
            model_id="voice-model-missing",
            voice_id="wanted",
        ),
        _profile(
            profile_id=UUID(int=38),
            display_name="Voice unverified",
            model_id="voice-ambiguous",
            voice_id="wanted",
        ),
    )
    voice_results = {
        "voice-present": TTSVoiceDiscoveryResult(
            provider_id="audio_cpp",
            model_id="voice-present",
            catalog_revision=9,
            voices=("wanted",),
            state="complete",
        ),
        "voice-absent": TTSVoiceDiscoveryResult(
            provider_id="audio_cpp",
            model_id="voice-absent",
            catalog_revision=9,
            voices=("other",),
            state="complete",
        ),
        "voice-model-missing": TTSVoiceDiscoveryResult(
            provider_id="audio_cpp",
            model_id="voice-model-missing",
            catalog_revision=9,
            voices=(),
            state="model_missing",
        ),
        "voice-ambiguous": TTSVoiceDiscoveryResult(
            provider_id="audio_cpp",
            model_id="voice-ambiguous",
            catalog_revision=9,
            voices=(),
            state="unverified",
        ),
    }
    tts_service = _FakeTTSService(
        _capability_snapshot(
            state="unverified",
            models=(
                _model("default-ok"),
                _model("default-no", server_default=False),
                _model("format-no", formats=("mp3",)),
                _model("voice-present"),
                _model("voice-absent"),
                _model("voice-ambiguous"),
            ),
            voice_results=voice_results,
        )
    )
    service, repository, _tts_service = _service(tts_service=tts_service)

    observed = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=profiles,
            total=len(profiles),
        )
    )

    assert tuple(item.state for item in observed.profiles) == (
        "available",
        "unavailable",
        "unavailable",
        "unavailable",
        "available",
        "unavailable",
        "unavailable",
        "unverified",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot_state", "fresh", "health_state", "expected"),
    (
        ("unverified", False, "available", "unverified"),
        ("complete", True, "reconfiguring", "unverified"),
        ("complete", True, "not_configured", "unavailable"),
    ),
)
async def test_availability_health_branches_return_bounded_row_states(
    snapshot_state: str,
    fresh: bool,
    health_state: str,
    expected: str,
) -> None:
    tts_service = _FakeTTSService(
        _capability_snapshot(
            state=snapshot_state,
            models=(_model("model-a"),),
            fresh=fresh,
            health_state=health_state,
        )
    )
    service, repository, _tts_service = _service(tts_service=tts_service)

    observed = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(_profile(),),
            total=1,
        )
    )

    assert observed.profiles[0].state == expected


def test_preview_preset_copies_only_persisted_selection_and_availability() -> None:
    service, repository, tts_service = _service()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(
            provider_id="future_native",
            model_id="opaque-model",
            voice_id="opaque-voice",
            response_format="flac",
            speed=1.5,
            options={"quality": {"level": 2}},
        ),
    )
    availability = TTSProfileAvailability(
        profile_id=loaded.profile.profile_id,
        state="unavailable",
        recovery_action="edit",
    )

    preset = service.preview_preset(loaded, availability)

    assert preset == TTSPlaygroundSelectionPreset(
        provider_id=loaded.profile.provider_id,
        model_id=loaded.profile.model_id,
        voice_id=loaded.profile.voice_id,
        response_format=loaded.profile.response_format,
        speed=loaded.profile.speed,
        options=loaded.profile.options,
        availability="unavailable",
    )
    assert repository.calls == []
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert not hasattr(tts_service, "synthesis_calls")


def test_preview_preset_rejects_availability_for_another_profile() -> None:
    service, repository, _tts_service = _service()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    availability = TTSProfileAvailability(
        profile_id=_DUPLICATE_ID,
        state="available",
        recovery_action="none",
    )

    with pytest.raises(ProfileValidationError) as caught:
        service.preview_preset(loaded, availability)

    assert caught.value.code == "profile_id"
