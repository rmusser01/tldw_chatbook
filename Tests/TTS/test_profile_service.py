from __future__ import annotations

import asyncio
import traceback
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
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
_UNSET = object()


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


def _forged_loaded_profile(
    profile: TTSGenerationProfile,
    *,
    repository_generation: object = 7,
) -> LoadedTTSProfile:
    """Build an adversarial loaded wrapper without service-value validation."""

    forged = object.__new__(LoadedTTSProfile)
    object.__setattr__(forged, "repository_generation", repository_generation)
    object.__setattr__(forged, "profile", profile)
    return forged


def _forged_page_snapshot(
    *,
    repository_generation: object,
    profiles: object,
    total: object,
) -> TTSProfilePageSnapshot:
    """Build an adversarial exact page without service-value validation."""

    forged = object.__new__(TTSProfilePageSnapshot)
    object.__setattr__(forged, "repository_generation", repository_generation)
    object.__setattr__(forged, "profiles", profiles)
    object.__setattr__(forged, "total", total)
    return forged


def _forged_capability_snapshot(
    snapshot: TTSNativeCapabilitySnapshot,
    **updates: object,
) -> TTSNativeCapabilitySnapshot:
    """Build an adversarial exact snapshot without adapter revalidation."""

    forged = object.__new__(TTSNativeCapabilitySnapshot)
    for snapshot_field in fields(TTSNativeCapabilitySnapshot):
        object.__setattr__(
            forged,
            snapshot_field.name,
            updates.get(snapshot_field.name, getattr(snapshot, snapshot_field.name)),
        )
    return forged


class _ExplodingStr(str):
    def __eq__(self, _other: object) -> bool:
        raise RuntimeError(
            "https://user:credential@example.test/private/path submitted text"
        )

    def __ne__(self, _other: object) -> bool:
        raise RuntimeError(
            "https://user:credential@example.test/private/path submitted text"
        )

    __hash__ = str.__hash__


class _AlwaysEqualStr(str):
    def __eq__(self, _other: object) -> bool:
        return True

    def __ne__(self, _other: object) -> bool:
        return False

    __hash__ = str.__hash__


class _GenerationAdvancingMapping(dict[str, Any]):
    def __init__(self, advance: Callable[[], None]) -> None:
        super().__init__()
        self._advance = advance

    def __iter__(self) -> Iterator[str]:
        self._advance()
        return super().__iter__()

    def items(self) -> Any:
        self._advance()
        return super().items()


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
    provider_id: str = "audio_cpp",
    configuration_revision: int = 3,
    state: str = "complete",
    models: tuple[TTSModelInfo, ...] = (),
    voice_results: dict[str, TTSVoiceDiscoveryResult] | None = None,
    catalog_revision: int = 9,
    fresh: bool = True,
    health_state: str = "available",
) -> TTSNativeCapabilitySnapshot:
    return TTSNativeCapabilitySnapshot(
        provider_id=provider_id,
        configuration_revision=configuration_revision,
        state=state,  # type: ignore[arg-type]
        catalog=TTSProviderCatalog(
            provider_id=provider_id,
            revision=catalog_revision,
            health=ProviderHealth(
                state=health_state,  # type: ignore[arg-type]
                fresh=fresh,
            ),
            models=models,
        ),
        voice_results={} if voice_results is None else voice_results,
    )


def _hostile_capability_snapshot(
    attack: str,
    *,
    model_id: str,
) -> TTSNativeCapabilitySnapshot:
    if attack == "health_state":
        return _capability_snapshot(
            models=(_model(model_id),),
            health_state=_ExplodingStr("available"),
        )
    if attack == "response_format":
        return _capability_snapshot(
            models=(
                _model(
                    model_id,
                    formats=(_ExplodingStr("wav"),),
                ),
            ),
        )
    if attack == "manufactured_response_format":
        return _capability_snapshot(
            models=(
                _model(
                    model_id,
                    formats=(_AlwaysEqualStr("mp3"),),
                ),
            ),
        )

    snapshot = _capability_snapshot(models=(_model(model_id),))
    if attack == "snapshot_state":
        return _forged_capability_snapshot(
            snapshot,
            state=_ExplodingStr("complete"),
        )
    if attack == "configuration_revision":
        return _forged_capability_snapshot(
            snapshot,
            configuration_revision=True,
        )
    raise AssertionError(f"Unknown hostile capability attack: {attack}")


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
        self.list_result: object = _UNSET
        self.create_result: object = _UNSET
        self.update_result: object = _UNSET
        self.delete_result: object = _UNSET
        self.count_result: object = _UNSET
        self.create_boundary: _AsyncBoundary | None = None

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
        if self.list_result is not _UNSET:
            return cast(ProfileStoreResult[TTSProfilePage], self.list_result)
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
        if self.create_boundary is not None:
            await self.create_boundary.wait()
        if self.create_error is not None:
            raise self.create_error
        if self.create_result is not _UNSET:
            return cast(
                ProfileStoreResult[TTSGenerationProfile],
                self.create_result,
            )
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
        if self.update_result is not _UNSET:
            return cast(
                ProfileStoreResult[TTSGenerationProfile],
                self.update_result,
            )
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
        if self.delete_result is not _UNSET:
            return cast(ProfileStoreResult[None], self.delete_result)
        return ProfileStoreResult(generation=self.generation, value=None)

    async def assignment_count(
        self,
        profile_id: UUID,
    ) -> ProfileStoreResult[int]:
        self._record_coordinator_state()
        self.calls.append(("count", (profile_id, self.generation)))
        if self.count_result is not _UNSET:
            return cast(ProfileStoreResult[int], self.count_result)
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


class _AsyncBoundary:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.settled = asyncio.Event()

    async def wait(self) -> None:
        self.entered.set()
        try:
            await self.release.wait()
        finally:
            self.settled.set()


class _HostileResult:
    def __getattribute__(self, _name: str) -> object:
        raise RuntimeError(
            "https://user:credential@example.test/private/path submitted text"
        )


class _ExplodingSequence(Sequence[object]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> object:
        raise RuntimeError(
            "https://user:credential@example.test/private/path submitted text"
        )


class _GuardedInfiniteSequence(Sequence[object]):
    def __init__(self, item: object) -> None:
        self.item = item
        self.items_requested = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> object:  # type: ignore[override]
        return self.item

    def __iter__(self) -> Iterator[object]:
        while True:
            self.items_requested += 1
            if self.items_requested > 51:
                raise RuntimeError(
                    "https://user:credential@example.test/private/path submitted text"
                )
            yield self.item


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
        self.capability_boundary: _AsyncBoundary | None = None
        self.revision_boundary: _AsyncBoundary | None = None
        self.read_side_active = False

    async def get_native_capability_snapshot(
        self,
        provider_id: str,
        exact_voice_model_ids: Iterable[str],
    ) -> TTSNativeCapabilitySnapshot:
        self.capability_calls.append((provider_id, tuple(exact_voice_model_ids)))
        if self.capability_hook is not None:
            self.capability_hook()
        if self.capability_boundary is not None:
            await self.capability_boundary.wait()
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
            if self.revision_boundary is not None:
                await self.revision_boundary.wait()
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
            selected_repository,
            selected_tts_service,
        ),
        selected_repository,
        selected_tts_service,
    )


def _profile_advancing_repository_generation(
    repository: _FakeRepository,
    profile: TTSGenerationProfile,
) -> TTSGenerationProfile:
    next_generation = repository.generation + 1
    options = _GenerationAdvancingMapping(
        lambda: setattr(repository, "generation", next_generation)
    )
    return _forged_profile(
        profile,
        options=MappingProxyType(options),
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
    loaded = LoadedTTSProfile(
        repository_generation=7,
        profile=profile,
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
    assert page.profiles[0] is not profile
    assert loaded.profile == profile
    assert loaded.profile is not profile
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


@pytest.mark.parametrize("wrapper", ("page", "loaded"))
def test_profile_wrappers_reject_forged_mutable_options_safely(
    wrapper: str,
) -> None:
    secret = "https://user:credential@example.test/private/path"
    forged = _forged_profile(
        _profile(
            provider_id="future_native",
            response_format="flac",
            speed=1.5,
        ),
        options={"endpoint": secret},
    )

    with pytest.raises(ProfileValidationError) as caught:
        if wrapper == "page":
            TTSProfilePageSnapshot(
                repository_generation=7,
                profiles=(forged,),
                total=1,
            )
        else:
            LoadedTTSProfile(
                repository_generation=7,
                profile=forged,
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


def test_loaded_profile_rejects_hostile_exact_profile_safely() -> None:
    forged = _forged_profile(
        _profile(),
        model_id=_ExplodingStr("model-a"),
    )

    with pytest.raises(ProfileValidationError) as caught:
        LoadedTTSProfile(
            repository_generation=7,
            profile=forged,
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


def test_page_snapshot_stops_lying_unbounded_sequence_at_item_fifty_one() -> None:
    profiles = _GuardedInfiniteSequence(_profile())

    with pytest.raises(ProfileValidationError) as caught:
        TTSProfilePageSnapshot(
            repository_generation=1,
            profiles=profiles,  # type: ignore[arg-type]
            total=51,
        )

    assert caught.value.code == "profiles"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert profiles.items_requested == 51


def test_availability_snapshot_stops_lying_unbounded_sequence_at_item_fifty_one() -> (
    None
):
    profiles = _GuardedInfiniteSequence(
        TTSProfileAvailability(
            profile_id=_PROFILE_ID,
            state="available",
            recovery_action="none",
        )
    )

    with pytest.raises(ProfileValidationError) as caught:
        TTSProfileAvailabilitySnapshot(
            repository_generation=1,
            configuration_revision=1,
            catalog_revision=1,
            profiles=profiles,  # type: ignore[arg-type]
        )

    assert caught.value.code == "profiles"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert profiles.items_requested == 51


@pytest.mark.asyncio
async def test_list_profiles_delegates_with_fixed_fifty_row_limit() -> None:
    service, repository, _tts_service = _service()
    collaborator_profile = _profile()
    repository.page = TTSProfilePage(profiles=(collaborator_profile,), total=81)

    page = await service.list_profiles(search=" nar ", offset=50)

    assert repository.calls == [("list", (" nar ", 50, 50))]
    assert page.repository_generation == 7
    assert page.profiles == repository.page.profiles
    assert page.profiles[0] is not collaborator_profile
    assert page.total == 81


@pytest.mark.asyncio
async def test_list_profiles_rechecks_generation_after_profile_canonicalization() -> (
    None
):
    repository = _FakeRepository()
    repository.page = TTSProfilePage(
        profiles=(
            _profile_advancing_repository_generation(
                repository,
                _profile(),
            ),
        ),
        total=1,
    )
    service, repository, tts_service = _service(repository=repository)

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.list_profiles()

    assert caught.value.code == "stale"
    assert [name for name, _value in repository.calls] == ["list"]
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert tts_service.revision_reads == []


@pytest.mark.asyncio
async def test_list_profiles_rejects_forged_mutable_profile_safely() -> None:
    repository = _FakeRepository()
    repository.page = TTSProfilePage(
        profiles=(
            _forged_profile(
                _profile(
                    provider_id="future_native",
                    response_format="flac",
                    speed=1.5,
                ),
                options={
                    "endpoint": ("https://user:credential@example.test/private/path"),
                },
            ),
        ),
        total=1,
    )
    service, repository, _tts_service = _service(repository=repository)

    with pytest.raises(ProfileServiceError) as caught:
        await service.list_profiles()

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "/private/path",
        "submitted text",
    )
    assert [name for name, _value in repository.calls] == ["list"]


@pytest.mark.asyncio
async def test_list_profiles_rejects_hostile_repository_result_safely() -> None:
    repository = _FakeRepository()
    repository.list_result = _HostileResult()
    service, repository, _tts_service = _service(repository=repository)

    with pytest.raises(ProfileServiceError) as caught:
        await service.list_profiles()

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "/private/path",
        "submitted text",
    )
    assert [name for name, _value in repository.calls] == ["list"]


@pytest.mark.asyncio
async def test_availability_rejects_forged_page_over_fifty_before_tts_work() -> None:
    service, repository, tts_service = _service()
    profiles = tuple(
        _profile(
            profile_id=UUID(int=index + 1),
            display_name=f"Profile {index + 1}",
        )
        for index in range(51)
    )
    page = _forged_page_snapshot(
        repository_generation=repository.generation,
        profiles=profiles,
        total=len(profiles),
    )

    with pytest.raises(ProfileValidationError) as caught:
        await service.observe_availability(page)

    assert caught.value.code == "profiles"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert tts_service.revision_reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hostile_value", ("container", "profile"))
async def test_availability_sanitizes_hostile_forged_page_before_tts_work(
    hostile_value: str,
) -> None:
    service, repository, tts_service = _service()
    profiles: object
    if hostile_value == "container":
        profiles = _ExplodingSequence()
    else:
        profiles = (
            _forged_profile(
                _profile(),
                model_id=_ExplodingStr("model-a"),
            ),
        )
    page = _forged_page_snapshot(
        repository_generation=repository.generation,
        profiles=profiles,
        total=1,
    )

    with pytest.raises(ProfileValidationError) as caught:
        await service.observe_availability(page)

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
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert tts_service.revision_reads == []


@pytest.mark.asyncio
async def test_availability_rechecks_generation_after_page_canonicalization() -> None:
    service, repository, tts_service = _service()
    profile = _profile_advancing_repository_generation(
        repository,
        _profile(),
    )
    page = _forged_page_snapshot(
        repository_generation=repository.generation,
        profiles=(profile,),
        total=1,
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.observe_availability(page)

    assert caught.value.code == "stale"
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert tts_service.revision_reads == []


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
    invalid_format = _profile(
        profile_id=UUID(int=4),
        display_name="Invalid format",
        provider_id="future_native",
        model_id="future-format",
        response_format="mp3",
    )
    invalid_speed = _profile(
        profile_id=UUID(int=5),
        display_name="Invalid speed",
        provider_id="future_native",
        model_id="future-speed",
        speed=1.25,
    )
    invalid_options = _profile(
        profile_id=UUID(int=6),
        display_name="Invalid options",
        provider_id="future_native",
        model_id="future-options",
        options={"quality": "high"},
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
@pytest.mark.parametrize("snapshot_state", ("complete", "unverified"))
async def test_availability_rejects_wrong_provider_snapshot_before_classification(
    snapshot_state: str,
) -> None:
    tts_service = _FakeTTSService(
        _capability_snapshot(
            provider_id="future_native",
            state=snapshot_state,
            models=(_model("model-a"),),
        )
    )
    service, repository, tts_service = _service(tts_service=tts_service)

    with pytest.raises(ProfileServiceError) as caught:
        await service.observe_availability(
            TTSProfilePageSnapshot(
                repository_generation=repository.generation,
                profiles=(_profile(),),
                total=1,
            )
        )

    _assert_safe_service_error(caught.value, "operation_failed")
    assert tts_service.capability_calls == [("audio_cpp", ())]
    assert tts_service.revision_decisions == []
    assert tts_service.revision_reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attack",
    (
        "snapshot_state",
        "configuration_revision",
        "health_state",
        "response_format",
        "manufactured_response_format",
    ),
)
async def test_availability_sanitizes_malformed_exact_capability_snapshot(
    attack: str,
) -> None:
    tts_service = _FakeTTSService(
        _hostile_capability_snapshot(attack, model_id="model-a")
    )
    service, repository, tts_service = _service(tts_service=tts_service)

    with pytest.raises(ProfileServiceError) as caught:
        await service.observe_availability(
            TTSProfilePageSnapshot(
                repository_generation=repository.generation,
                profiles=(_profile(),),
                total=1,
            )
        )

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "/private/path",
        "submitted text",
    )
    assert tts_service.capability_calls == [("audio_cpp", ())]
    assert tts_service.revision_decisions == []
    assert tts_service.revision_reads == []
    assert repository.calls == []


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


@pytest.mark.asyncio
async def test_create_rechecks_generation_after_profile_canonicalization() -> None:
    repository = _FakeRepository()
    persisted = _profile_advancing_repository_generation(
        repository,
        _profile(
            display_name="Saved",
            model_id="selected-model",
            voice_id="selected-voice",
        ),
    )
    repository.create_result = ProfileStoreResult(
        generation=repository.generation,
        value=persisted,
    )
    service, repository, tts_service = _service(repository=repository)

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.create_from_artifact(
            "Saved",
            _artifact(selection=_selection()),
        )

    assert caught.value.code == "stale"
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert [name for name, _value in repository.calls] == ["create"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    (
        "hostile_envelope",
        "hostile_value",
        "wrong_generation",
        "changed_display_name",
        "changed_normalized_name",
        "changed_generation_fields",
        "noncanonical_speed",
        "mutable_options",
        "wrong_revision",
    ),
)
async def test_create_from_artifact_rejects_hostile_repository_result(
    invalid_result: str,
) -> None:
    repository = _FakeRepository()
    persisted = _profile(
        display_name="Saved",
        model_id="selected-model",
        voice_id="selected-voice",
    )
    if invalid_result == "hostile_envelope":
        repository.create_result = _HostileResult()
    elif invalid_result == "hostile_value":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_HostileResult(),
        )
    elif invalid_result == "wrong_generation":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation + 1,
            value=persisted,
        )
    elif invalid_result == "changed_display_name":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(
                persisted,
                display_name="Different",
                normalized_name="different",
            ),
        )
    elif invalid_result == "changed_normalized_name":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, normalized_name="different"),
        )
    elif invalid_result == "changed_generation_fields":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(
                persisted,
                model_id="https://user:credential@example.test/private/path",
            ),
        )
    elif invalid_result == "noncanonical_speed":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, speed=1),
        )
    elif invalid_result == "mutable_options":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, options={}),
        )
    else:
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, revision=2),
        )
    service, repository, tts_service = _service(repository=repository)

    with pytest.raises(ProfileServiceError) as caught:
        await service.create_from_artifact(
            "Saved",
            _artifact(selection=_selection()),
        )

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "/private/path",
        "submitted text",
    )
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert [name for name, _value in repository.calls] == ["create"]


@pytest.mark.asyncio
async def test_create_from_artifact_returns_canonical_copy_of_repository_profile() -> (
    None
):
    repository = _FakeRepository()
    persisted = _profile(
        display_name="Saved",
        model_id="selected-model",
        voice_id="selected-voice",
    )
    repository.create_result = ProfileStoreResult(
        generation=repository.generation,
        value=persisted,
    )
    service, _repository, _tts_service = _service(repository=repository)

    loaded = await service.create_from_artifact(
        "Saved",
        _artifact(selection=_selection()),
    )

    assert loaded.profile == persisted
    assert loaded.profile is not persisted
    assert type(loaded.profile.speed) is float
    assert type(loaded.profile.options) is MappingProxyType


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


def test_profile_service_requires_repository_generation_protocol_member() -> None:
    repository = _FakeRepository()
    del repository.generation

    with pytest.raises(ProfileServiceError) as caught:
        TTSProfileService(repository, _FakeTTSService())

    _assert_safe_service_error(caught.value, "operation_failed")


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
async def test_update_rejects_hostile_forged_loaded_profile_before_comparison() -> None:
    service, repository, tts_service = _service()
    loaded = _forged_loaded_profile(
        _forged_profile(
            _profile(revision=4),
            model_id=_AlwaysEqualStr("model-a"),
        ),
        repository_generation=repository.generation,
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

    with pytest.raises(ProfileValidationError) as caught:
        await service.update_profile(loaded, changed)

    assert caught.value.code == "profiles"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert repository.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    (
        "hostile_envelope",
        "hostile_value",
        "wrong_generation",
        "changed_display_name",
        "changed_normalized_name",
        "changed_generation_fields",
        "noncanonical_speed",
        "mutable_options",
        "changed_profile_id",
        "wrong_revision",
    ),
)
async def test_update_rejects_hostile_repository_result(
    invalid_result: str,
) -> None:
    repository = _FakeRepository()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(revision=4),
    )
    draft = TTSProfileDraft(
        display_name="Renamed",
        provider_id=loaded.profile.provider_id,
        model_id=loaded.profile.model_id,
        voice_id=loaded.profile.voice_id,
        response_format=loaded.profile.response_format,
        speed=loaded.profile.speed,
        options=loaded.profile.options,
    )
    persisted = _profile(
        profile_id=loaded.profile.profile_id,
        display_name=draft.display_name,
        revision=loaded.profile.revision + 1,
    )
    if invalid_result == "hostile_envelope":
        repository.update_result = _HostileResult()
    elif invalid_result == "hostile_value":
        repository.update_result = ProfileStoreResult(
            generation=repository.generation,
            value=_HostileResult(),
        )
    elif invalid_result == "wrong_generation":
        repository.update_result = ProfileStoreResult(
            generation=repository.generation + 1,
            value=persisted,
        )
    elif invalid_result == "changed_display_name":
        repository.update_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(
                persisted,
                display_name="Different",
                normalized_name="different",
            ),
        )
    elif invalid_result == "changed_normalized_name":
        repository.update_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, normalized_name="different"),
        )
    elif invalid_result == "changed_generation_fields":
        repository.update_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(
                persisted,
                model_id="https://user:credential@example.test/private/path",
            ),
        )
    elif invalid_result == "noncanonical_speed":
        repository.update_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, speed=1),
        )
    elif invalid_result == "mutable_options":
        repository.update_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, options={}),
        )
    elif invalid_result == "changed_profile_id":
        repository.update_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, profile_id=_DUPLICATE_ID),
        )
    else:
        repository.update_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(
                persisted,
                revision=loaded.profile.revision + 2,
            ),
        )
    service, repository, tts_service = _service(repository=repository)

    with pytest.raises(ProfileServiceError) as caught:
        await service.update_profile(loaded, draft)

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "/private/path",
        "submitted text",
    )
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert [name for name, _value in repository.calls] == ["update"]


@pytest.mark.asyncio
async def test_update_rechecks_generation_after_profile_canonicalization() -> None:
    repository = _FakeRepository()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(revision=4),
    )
    draft = TTSProfileDraft(
        display_name="Renamed",
        provider_id=loaded.profile.provider_id,
        model_id=loaded.profile.model_id,
        voice_id=loaded.profile.voice_id,
        response_format=loaded.profile.response_format,
        speed=loaded.profile.speed,
        options=loaded.profile.options,
    )
    persisted = _profile_advancing_repository_generation(
        repository,
        _profile(
            profile_id=loaded.profile.profile_id,
            display_name=draft.display_name,
            revision=loaded.profile.revision + 1,
        ),
    )
    repository.update_result = ProfileStoreResult(
        generation=repository.generation,
        value=persisted,
    )
    service, repository, tts_service = _service(repository=repository)

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.update_profile(loaded, draft)

    assert caught.value.code == "stale"
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert [name for name, _value in repository.calls] == ["update"]


@pytest.mark.asyncio
async def test_update_rejects_stale_loaded_generation_before_capability_work() -> None:
    service, repository, tts_service = _service(
        tts_service=_FakeTTSService(_capability_snapshot(models=(_model("model-b"),)))
    )
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation - 1,
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

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.update_profile(loaded, changed)

    assert caught.value.code == "stale"
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert repository.calls == []


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
    ("snapshot", "code", "revision_decisions"),
    (
        (
            _capability_snapshot(
                state="unverified",
                models=(_model("model-b"),),
            ),
            "profile_unverified",
            (),
        ),
        (
            _capability_snapshot(
                models=(_model("different-model"),),
            ),
            "profile_unavailable",
            (("audio_cpp", 3),),
        ),
    ),
)
async def test_generation_edit_rejects_non_authoritative_capability(
    snapshot: TTSNativeCapabilitySnapshot,
    code: str,
    revision_decisions: tuple[tuple[str, int], ...],
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
    assert tts_service.revision_decisions == list(revision_decisions)
    assert repository.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "negative_evidence",
    (
        "missing_model",
        "unsupported_format",
        "server_default_disallowed",
        "exact_voice_absent",
        "voice_model_missing",
    ),
)
async def test_stale_complete_negative_capability_checks_revision_first(
    negative_evidence: str,
) -> None:
    voice_id = (
        "wanted"
        if negative_evidence in {"exact_voice_absent", "voice_model_missing"}
        else None
    )
    models = ()
    voice_results: dict[str, TTSVoiceDiscoveryResult] = {}
    if negative_evidence == "unsupported_format":
        models = (_model("model-b", formats=("mp3",)),)
    elif negative_evidence == "server_default_disallowed":
        models = (_model("model-b", server_default=False),)
    elif negative_evidence == "exact_voice_absent":
        models = (_model("model-b"),)
        voice_results["model-b"] = TTSVoiceDiscoveryResult(
            provider_id="audio_cpp",
            model_id="model-b",
            catalog_revision=9,
            voices=("other",),
            state="complete",
        )
    elif negative_evidence == "voice_model_missing":
        voice_results["model-b"] = TTSVoiceDiscoveryResult(
            provider_id="audio_cpp",
            model_id="model-b",
            catalog_revision=9,
            voices=(),
            state="model_missing",
        )

    tts_service = _FakeTTSService(
        _capability_snapshot(
            configuration_revision=3,
            models=models,
            voice_results=voice_results,
        )
    )
    tts_service.revision = 4
    service, repository, tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    changed = TTSProfileDraft(
        display_name="Changed",
        provider_id="audio_cpp",
        model_id="model-b",
        voice_id=voice_id,
        response_format="wav",
        speed=1.0,
        options={},
    )

    with pytest.raises(ProfileServiceError) as caught:
        await service.update_profile(loaded, changed)

    _assert_safe_service_error(
        caught.value,
        "stale_configuration",
        "credential",
        "example.test",
        "/private/path",
    )
    expected_voice_models = () if voice_id is None else ("model-b",)
    assert tts_service.capability_calls == [
        ("audio_cpp", expected_voice_models),
    ]
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert repository.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_state", ("complete", "unverified"))
async def test_capability_rejects_mismatched_provider_before_state_or_decision(
    snapshot_state: str,
) -> None:
    tts_service = _FakeTTSService(
        _capability_snapshot(
            provider_id="future_native",
            state=snapshot_state,
            models=(_model("model-b"),),
        )
    )
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

    _assert_safe_service_error(caught.value, "operation_failed")
    assert tts_service.capability_calls == [("audio_cpp", ())]
    assert tts_service.revision_decisions == []
    assert repository.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attack",
    (
        "snapshot_state",
        "configuration_revision",
        "health_state",
        "response_format",
        "manufactured_response_format",
    ),
)
async def test_generation_edit_sanitizes_malformed_exact_capability_snapshot(
    attack: str,
) -> None:
    tts_service = _FakeTTSService(
        _hostile_capability_snapshot(attack, model_id="model-b")
    )
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

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "/private/path",
        "submitted text",
    )
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
@pytest.mark.parametrize(
    "invalid_result",
    (
        "hostile_envelope",
        "hostile_value",
        "wrong_generation",
        "changed_display_name",
        "changed_normalized_name",
        "changed_generation_fields",
        "noncanonical_speed",
        "mutable_options",
        "reused_profile_id",
        "wrong_revision",
    ),
)
async def test_duplicate_rejects_hostile_repository_result(
    invalid_result: str,
) -> None:
    repository = _FakeRepository()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    persisted = _profile(
        profile_id=_DUPLICATE_ID,
        display_name="Duplicate",
    )
    if invalid_result == "hostile_envelope":
        repository.create_result = _HostileResult()
    elif invalid_result == "hostile_value":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_HostileResult(),
        )
    elif invalid_result == "wrong_generation":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation + 1,
            value=persisted,
        )
    elif invalid_result == "changed_display_name":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(
                persisted,
                display_name="Different",
                normalized_name="different",
            ),
        )
    elif invalid_result == "changed_normalized_name":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, normalized_name="different"),
        )
    elif invalid_result == "changed_generation_fields":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(
                persisted,
                model_id="https://user:credential@example.test/private/path",
            ),
        )
    elif invalid_result == "noncanonical_speed":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, speed=1),
        )
    elif invalid_result == "mutable_options":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, options={}),
        )
    elif invalid_result == "reused_profile_id":
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(
                persisted,
                profile_id=loaded.profile.profile_id,
            ),
        )
    else:
        repository.create_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(persisted, revision=2),
        )
    service, repository, tts_service = _service(
        repository=repository,
        tts_service=_FakeTTSService(_capability_snapshot(models=(_model("model-a"),))),
    )

    with pytest.raises(ProfileServiceError) as caught:
        await service.duplicate_profile(loaded, "Duplicate")

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "/private/path",
        "submitted text",
    )
    assert tts_service.capability_calls == [("audio_cpp", ())]
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert [name for name, _value in repository.calls] == ["create"]


@pytest.mark.asyncio
async def test_duplicate_rechecks_generation_after_profile_canonicalization() -> None:
    repository = _FakeRepository()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    persisted = _profile_advancing_repository_generation(
        repository,
        _profile(
            profile_id=_DUPLICATE_ID,
            display_name="Duplicate",
        ),
    )
    repository.create_result = ProfileStoreResult(
        generation=repository.generation,
        value=persisted,
    )
    service, repository, tts_service = _service(
        repository=repository,
        tts_service=_FakeTTSService(_capability_snapshot(models=(_model("model-a"),))),
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.duplicate_profile(loaded, "Duplicate")

    assert caught.value.code == "stale"
    assert tts_service.capability_calls == [("audio_cpp", ())]
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert [name for name, _value in repository.calls] == ["create"]


@pytest.mark.asyncio
async def test_duplicate_rejects_stale_loaded_generation_before_capability_work() -> (
    None
):
    service, repository, tts_service = _service(
        tts_service=_FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    )
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation - 1,
        profile=_profile(),
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.duplicate_profile(loaded, "Duplicate")

    assert caught.value.code == "stale"
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert repository.calls == []


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
async def test_assignment_count_rejects_hostile_repository_result_safely() -> None:
    repository = _FakeRepository()
    repository.count_result = _HostileResult()
    service, repository, _tts_service = _service(repository=repository)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )

    with pytest.raises(ProfileServiceError) as caught:
        await service.assignment_count(loaded)

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "/private/path",
        "submitted text",
    )
    assert [name for name, _value in repository.calls] == ["count"]


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
async def test_delete_rejects_hostile_repository_result_safely() -> None:
    repository = _FakeRepository()
    repository.delete_result = _HostileResult()
    service, repository, _tts_service = _service(repository=repository)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )

    with pytest.raises(ProfileServiceError) as caught:
        await service.delete_profile(loaded)

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "/private/path",
        "submitted text",
    )
    assert [name for name, _value in repository.calls] == ["delete"]


@pytest.mark.asyncio
async def test_delete_rejects_stale_loaded_generation_before_repository_work() -> None:
    service, repository, tts_service = _service()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation - 1,
        profile=_profile(),
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.delete_profile(loaded)

    assert caught.value.code == "stale"
    assert repository.calls == []
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []


@pytest.mark.asyncio
async def test_cancellation_propagates_unchanged_from_capability_wait() -> None:
    boundary = _AsyncBoundary()
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    tts_service.capability_boundary = boundary
    service, repository, tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    existing_tasks = set(asyncio.all_tasks())
    operation = asyncio.create_task(
        service.duplicate_profile(loaded, "Duplicate"),
        name="profile_service_capability_wait",
    )
    async with asyncio.timeout(1):
        await boundary.entered.wait()
    cancellation_identity = object()

    operation.cancel(cancellation_identity)
    with pytest.raises(asyncio.CancelledError) as caught:
        await operation

    assert caught.value.args == (cancellation_identity,)
    assert caught.value.args[0] is cancellation_identity
    assert boundary.settled.is_set()
    assert not boundary.release.is_set()
    assert operation.done()
    assert operation not in asyncio.all_tasks()
    assert set(asyncio.all_tasks()) == existing_tasks
    assert tts_service.capability_calls == [("audio_cpp", ())]
    assert tts_service.revision_decisions == []
    assert repository.calls == []


@pytest.mark.asyncio
async def test_cancellation_propagates_unchanged_from_revision_decision_wait() -> None:
    boundary = _AsyncBoundary()
    tts_service = _FakeTTSService()
    tts_service.revision_boundary = boundary
    service, repository, tts_service = _service(tts_service=tts_service)
    existing_tasks = set(asyncio.all_tasks())
    operation = asyncio.create_task(
        service.create_from_artifact(
            "Saved",
            _artifact(selection=_selection()),
        ),
        name="profile_service_revision_wait",
    )
    async with asyncio.timeout(1):
        await boundary.entered.wait()
    cancellation_identity = object()

    operation.cancel(cancellation_identity)
    with pytest.raises(asyncio.CancelledError) as caught:
        await operation

    assert caught.value.args == (cancellation_identity,)
    assert caught.value.args[0] is cancellation_identity
    assert boundary.settled.is_set()
    assert not boundary.release.is_set()
    assert operation.done()
    assert operation not in asyncio.all_tasks()
    assert set(asyncio.all_tasks()) == existing_tasks
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert not tts_service.read_side_active
    assert repository.calls == []


@pytest.mark.asyncio
async def test_cancellation_propagates_unchanged_from_repository_wait() -> None:
    boundary = _AsyncBoundary()
    repository = _FakeRepository()
    repository.create_boundary = boundary
    service, repository, tts_service = _service(repository=repository)
    existing_tasks = set(asyncio.all_tasks())
    operation = asyncio.create_task(
        service.create_from_artifact(
            "Saved",
            _artifact(selection=_selection()),
        ),
        name="profile_service_repository_wait",
    )
    async with asyncio.timeout(1):
        await boundary.entered.wait()
    cancellation_identity = object()

    operation.cancel(cancellation_identity)
    with pytest.raises(asyncio.CancelledError) as caught:
        await operation

    assert caught.value.args == (cancellation_identity,)
    assert caught.value.args[0] is cancellation_identity
    assert boundary.settled.is_set()
    assert not boundary.release.is_set()
    assert operation.done()
    assert operation not in asyncio.all_tasks()
    assert set(asyncio.all_tasks()) == existing_tasks
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert [name for name, _value in repository.calls] == ["create"]
    assert repository.coordinator_active_at_repository_calls == [False]


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
@pytest.mark.parametrize("snapshot_state", ("complete", "unverified"))
@pytest.mark.parametrize("evidence_state", ("available", "unavailable"))
async def test_stale_availability_evidence_uses_writer_ordered_decision_first(
    snapshot_state: str,
    evidence_state: str,
) -> None:
    models = (_model("model-a"),) if evidence_state == "available" else ()
    tts_service = _FakeTTSService(
        _capability_snapshot(
            configuration_revision=3,
            state=snapshot_state,
            models=models,
        )
    )
    tts_service.revision = 4
    service, repository, tts_service = _service(tts_service=tts_service)

    with pytest.raises(ProfileServiceError) as caught:
        await service.observe_availability(
            TTSProfilePageSnapshot(
                repository_generation=repository.generation,
                profiles=(_profile(),),
                total=1,
            )
        )

    _assert_safe_service_error(
        caught.value,
        "stale_configuration",
        "credential",
        "/private/path",
    )
    assert tts_service.capability_calls == [("audio_cpp", ())]
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert tts_service.revision_reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_state", ("complete", "unverified"))
async def test_availability_rejects_snapshot_after_configuration_change(
    snapshot_state: str,
) -> None:
    tts_service = _FakeTTSService(
        _capability_snapshot(
            configuration_revision=3,
            state=snapshot_state,
            models=(_model("model-a"),),
        )
    )
    tts_service.reconfigure_after_decision = True
    service, repository, tts_service = _service(tts_service=tts_service)

    with pytest.raises(ProfileServiceError) as caught:
        await service.observe_availability(
            TTSProfilePageSnapshot(
                repository_generation=repository.generation,
                profiles=(_profile(),),
                total=1,
            )
        )

    _assert_safe_service_error(caught.value, "stale_configuration")
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert tts_service.revision_reads == ["audio_cpp"]


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
    service, repository, tts_service = _service(tts_service=tts_service)

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
    assert tts_service.revision_decisions == [("audio_cpp", 3)]


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


def test_preview_preset_forces_unsupported_profile_unavailable_before_enrichment() -> (
    None
):
    service, repository, tts_service = _service()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(
            provider_id="openai",
            model_id="tts-1",
            voice_id="alloy",
        ),
    )
    pending = TTSProfileAvailability(
        profile_id=loaded.profile.profile_id,
        state="unverified",
        recovery_action="refresh",
    )

    preset = service.preview_preset(loaded, pending)

    assert preset.provider_id == "openai"
    assert preset.availability == "unavailable"
    assert repository.calls == []
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []


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
