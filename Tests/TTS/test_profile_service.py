from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import inspect
import os
import struct
import threading
import traceback
import wave
from collections.abc import Callable, Coroutine, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar, cast
from uuid import UUID

import pytest

import tldw_chatbook.TTS.profile_service as profile_service
import tldw_chatbook.TTS.sample_audio_validation as sample_audio_validation
import tldw_chatbook.TTS.TTS_Generation as tts_generation
from tldw_chatbook.TTS.adapter_registry import TTSProviderConfigurationSnapshot
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppAcceptedPackage,
    AudioCppSettingsConfig,
)
from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
    AudioCppArtifactConsumerRequirement,
)
from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY
from tldw_chatbook.TTS.adapter_types import (
    _TTS_CLONE_GENERATION_EVIDENCE_TOKEN,
    ProviderHealth,
    TTSCloneGenerationEvidence,
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
from tldw_chatbook.TTS.profile_portability import PortableTTSProfile
from tldw_chatbook.TTS.profile_reference_types import (
    CanonicalTTSCloneReference,
    TTSCloneReference,
    TTSCloneRecipeRequirement,
    TTSCloneReferenceSummary,
)
from tldw_chatbook.TTS.profile_service import (
    LoadedCharacterTTSAssignment,
    LoadedTTSProfile,
    PortableProfileAvailabilityObservation,
    PortableProfileImportPlan,
    TTSPlaygroundSelectionPreset,
    TTSProfileAvailability,
    TTSProfileAvailabilitySnapshot,
    TTSProfilePageSnapshot,
    TTSProfileService,
)
from tldw_chatbook.TTS.profile_types import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileCollisionSnapshot,
    TTSProfileDraft,
    TTSProfilePage,
    profile_options_fingerprint,
)

_CREATED_AT = datetime(2026, 7, 27, 12, tzinfo=UTC)
_PROFILE_ID = UUID("11111111-1111-4111-8111-111111111111")
_DUPLICATE_ID = UUID("22222222-2222-4222-8222-222222222222")
_PORTABLE_COPY_ID = UUID("33333333-3333-4333-8333-333333333333")
_UNSET = object()
_TaskResult = TypeVar("_TaskResult")


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
    reference: TTSCloneReferenceSummary | None = None,
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
        reference=reference,
    )


def _reference() -> TTSCloneReference:
    wav_bytes = b"canonical-private-reference"
    return TTSCloneReference(
        summary=TTSCloneReferenceSummary(
            reference_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            byte_length=len(wav_bytes),
            duration_ms=250,
            sample_rate_hz=24_000,
            channels=1,
            sample_encoding="pcm_s16le",
            created_at=_CREATED_AT,
            updated_at=_CREATED_AT,
        ),
        reference_text="Private transcript",
        sha256=hashlib.sha256(wav_bytes).hexdigest(),
        wav_bytes=wav_bytes,
    )


def _reference_with_requirement(
    requirement: TTSCloneRecipeRequirement | None,
) -> TTSCloneReference:
    reference = _reference()
    summary = TTSCloneReferenceSummary(
        reference_id=reference.summary.reference_id,
        byte_length=reference.summary.byte_length,
        duration_ms=reference.summary.duration_ms,
        sample_rate_hz=reference.summary.sample_rate_hz,
        channels=reference.summary.channels,
        sample_encoding=reference.summary.sample_encoding,
        created_at=reference.summary.created_at,
        updated_at=reference.summary.updated_at,
        recipe_requirement=requirement,
    )
    return TTSCloneReference(
        summary=summary,
        reference_text=reference.reference_text,
        sha256=reference.sha256,
        wav_bytes=reference.wav_bytes,
        recipe_requirement=requirement,
    )


def test_reference_canonicalizers_reconstruct_exact_recipe_provenance() -> None:
    requirement = TTSCloneRecipeRequirement(
        recipe_id="audio-cpp-0.5.1.supertonic.supertonic_3_orig",
        recipe_revision=1,
        model_id="model-a",
    )
    wav_bytes = b"canonical-private-reference"
    summary = TTSCloneReferenceSummary(
        reference_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        byte_length=len(wav_bytes),
        duration_ms=250,
        sample_rate_hz=24_000,
        channels=1,
        sample_encoding="pcm_s16le",
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
        recipe_requirement=requirement,
    )
    reference = TTSCloneReference(
        summary=summary,
        recipe_requirement=requirement,
        reference_text="Private transcript",
        sha256=hashlib.sha256(wav_bytes).hexdigest(),
        wav_bytes=wav_bytes,
    )

    canonical_summary = profile_service._canonicalize_exact_reference_summary(summary)
    assert canonical_summary == summary
    assert canonical_summary.recipe_requirement is not requirement
    canonical_reference = profile_service._canonicalize_exact_reference(reference)
    assert canonical_reference == reference
    assert canonical_reference.summary.recipe_requirement is not requirement
    assert canonical_reference.recipe_requirement is not requirement
    assert (
        canonical_reference.summary.recipe_requirement
        is canonical_reference.recipe_requirement
    )
    assert _profile(model_id="model-a", reference=summary).reference == summary
    with pytest.raises(ProfileValidationError, match=r"reference_invalid"):
        _profile(model_id="model-b", reference=summary)


def test_reference_canonicalizer_rejects_forged_recipe_provenance() -> None:
    summary = TTSCloneReferenceSummary(
        reference_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        byte_length=1,
        duration_ms=1,
        sample_rate_hz=24_000,
        channels=1,
        sample_encoding="pcm_s16le",
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )
    forged = object.__new__(TTSCloneReferenceSummary)
    for summary_field in fields(TTSCloneReferenceSummary):
        object.__setattr__(
            forged, summary_field.name, getattr(summary, summary_field.name)
        )
    object.__setattr__(forged, "recipe_requirement", object())

    with pytest.raises(ProfileValidationError, match=r"reference_invalid"):
        profile_service._canonicalize_exact_reference_summary(forged)


def test_reference_canonicalizer_rejects_forged_direct_recipe_provenance() -> None:
    requirement = TTSCloneRecipeRequirement(
        recipe_id="audio-cpp-0.5.1.supertonic.supertonic_3_orig",
        recipe_revision=1,
        model_id="model-a",
    )
    wav_bytes = b"canonical-private-reference"
    summary = TTSCloneReferenceSummary(
        reference_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        byte_length=len(wav_bytes),
        duration_ms=250,
        sample_rate_hz=24_000,
        channels=1,
        sample_encoding="pcm_s16le",
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
        recipe_requirement=requirement,
    )
    reference = TTSCloneReference(
        summary=summary,
        recipe_requirement=requirement,
        reference_text="Private transcript",
        sha256=hashlib.sha256(wav_bytes).hexdigest(),
        wav_bytes=wav_bytes,
    )

    class _AlwaysEqualRecipeRequirement:
        def __eq__(self, _other: object) -> bool:
            return True

    forged = object.__new__(TTSCloneReference)
    for reference_field in fields(TTSCloneReference):
        object.__setattr__(
            forged,
            reference_field.name,
            getattr(reference, reference_field.name),
        )
    object.__setattr__(forged, "recipe_requirement", _AlwaysEqualRecipeRequirement())

    with pytest.raises(ProfileValidationError, match=r"reference_invalid"):
        profile_service._canonicalize_exact_reference(forged)


def _portable_profile(
    *,
    profile_id: UUID = _PROFILE_ID,
    display_name: str = "Imported voice",
    provider_id: str = "audio_cpp",
    model_id: str = "model-a",
    voice_id: str | None = None,
    response_format: str = "wav",
    speed: float = 1.0,
) -> PortableTTSProfile:
    return PortableTTSProfile(
        profile_id=profile_id,
        draft=TTSProfileDraft(
            display_name=display_name,
            provider_id=provider_id,
            model_id=model_id,
            voice_id=voice_id,
            response_format=response_format,
            speed=speed,
            options={},
        ),
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


def _character_ref(
    *,
    source: str = "server",
    authority_id: str = "server-user-v1:authority",
    character_id: str = "character-a",
) -> CharacterRef:
    return CharacterRef(
        source=source,  # type: ignore[arg-type]
        authority_id=authority_id,
        character_id=character_id,
    )


def _assignment(
    *,
    character_ref: CharacterRef | None = None,
    profile_id: UUID = _PROFILE_ID,
) -> CharacterTTSAssignment:
    return CharacterTTSAssignment(
        character_ref=_character_ref() if character_ref is None else character_ref,
        profile_id=profile_id,
    )


def _forged_character_ref(
    character_ref: CharacterRef,
    **updates: object,
) -> CharacterRef:
    """Build an adversarial exact character reference without revalidation."""

    forged = object.__new__(CharacterRef)
    for reference_field in fields(CharacterRef):
        object.__setattr__(
            forged,
            reference_field.name,
            updates.get(
                reference_field.name,
                getattr(character_ref, reference_field.name),
            ),
        )
    return forged


def _forged_assignment(
    assignment: CharacterTTSAssignment,
    **updates: object,
) -> CharacterTTSAssignment:
    """Build an adversarial exact assignment without revalidation."""

    forged = object.__new__(CharacterTTSAssignment)
    for assignment_field in fields(CharacterTTSAssignment):
        object.__setattr__(
            forged,
            assignment_field.name,
            updates.get(
                assignment_field.name,
                getattr(assignment, assignment_field.name),
            ),
        )
    return forged


def _forged_assigned_snapshot(
    snapshot: AssignedTTSProfileSnapshot,
    **updates: object,
) -> AssignedTTSProfileSnapshot:
    """Build an adversarial joined snapshot without domain revalidation."""

    forged = object.__new__(AssignedTTSProfileSnapshot)
    for snapshot_field in fields(AssignedTTSProfileSnapshot):
        object.__setattr__(
            forged,
            snapshot_field.name,
            updates.get(
                snapshot_field.name,
                getattr(snapshot, snapshot_field.name),
            ),
        )
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


def _manufactured_equal_character_ref(
    character_ref: CharacterRef,
) -> CharacterRef:
    hostile_ref = _forged_character_ref(
        character_ref,
        authority_id=_AlwaysEqualStr("different-authority"),
        character_id=_AlwaysEqualStr("different-character"),
    )
    assert type(hostile_ref) is CharacterRef
    assert hostile_ref == character_ref
    assert str(hostile_ref.authority_id) == "different-authority"
    assert str(hostile_ref.character_id) == "different-character"
    return hostile_ref


def _manufactured_equal_assignment(
    assignment: CharacterTTSAssignment,
) -> CharacterTTSAssignment:
    hostile_assignment = _forged_assignment(
        assignment,
        character_ref=_manufactured_equal_character_ref(assignment.character_ref),
    )
    assert type(hostile_assignment) is CharacterTTSAssignment
    assert type(hostile_assignment.character_ref) is CharacterRef
    assert hostile_assignment == assignment
    return hostile_assignment


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
    clone_evidence: TTSCloneGenerationEvidence | None = None,
) -> STTSGeneratedAudio:
    return STTSGeneratedAudio(
        path=Path("/private/secret/result.wav"),
        provider_id=(
            "audio_cpp" if clone_evidence is not None else "legacy-response-provider"
        ),
        model_id=(
            "selected-model" if clone_evidence is not None else "mutable-response-model"
        ),
        voice_id=(
            "selected-voice" if clone_evidence is not None else "mutable-response-voice"
        ),
        source_text="private submitted text",
        operation_id="operation",
        audio_format=("wav" if clone_evidence is not None else "mp3"),
        content_type="secret/content-type",
        metadata={"endpoint": "https://user:credential@example.test"},
        requested_selection=selection,
        clone_evidence=clone_evidence,
    )


def _successful_artifact(
    selection: TTSRequestedSelectionSnapshot,
    path: Path,
    *,
    provider_id: str | None = None,
    model_id: str | None = None,
    voice_id: str | None | object = _UNSET,
    audio_format: str | None = None,
) -> STTSGeneratedAudio:
    selected_voice = selection.voice_id if voice_id is _UNSET else voice_id
    return STTSGeneratedAudio(
        path=path,
        provider_id=selection.provider_id if provider_id is None else provider_id,
        model_id=selection.model_id if model_id is None else model_id,
        voice_id=cast(str | None, selected_voice),
        source_text="private submitted text",
        operation_id="operation",
        audio_format=(
            selection.response_format if audio_format is None else audio_format
        ),
        content_type=(
            "audio/wav" if selection.response_format == "wav" else "audio/mpeg"
        ),
        metadata={"endpoint": "https://user:credential@example.test"},
        requested_selection=selection,
    )


@pytest.fixture
def successful_audio_path(tmp_path: Path) -> Path:
    path = tmp_path / "completed.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(struct.pack("<h", 100) * 32)
    return path


def _selection(
    *,
    provider_id: str = "audio_cpp",
    model_id: str = "selected-model",
    voice_id: str | None = "selected-voice",
    response_format: str = "wav",
    speed: float = 1.0,
    configuration_revision: int = 3,
) -> TTSRequestedSelectionSnapshot:
    return TTSRequestedSelectionSnapshot(
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        response_format=response_format,
        speed=speed,
        options={},
        configuration_revision=configuration_revision,
    )


def _clone_canonical() -> CanonicalTTSCloneReference:
    frames = 32
    sample_rate = 16_000
    pcm = struct.pack("<h", 3) * frames
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )
    wav = b"RIFF" + struct.pack("<I", len(body)) + body
    return CanonicalTTSCloneReference(
        wav_bytes=wav,
        reference_text="Private reference transcript",
        sha256=hashlib.sha256(wav).hexdigest(),
        byte_length=len(wav),
        duration_ms=2,
        sample_rate_hz=sample_rate,
        channels=1,
        sample_encoding="pcm_s16le",
    )


def _clone_evidence() -> TTSCloneGenerationEvidence:
    return TTSCloneGenerationEvidence(
        _TTS_CLONE_GENERATION_EVIDENCE_TOKEN,
        canonical_reference=_clone_canonical(),
        model_id="selected-model",
        recipe_id="pocket_tts",
        recipe_revision=1,
        provider_configuration_revision=3,
        applied_provider_generation=2,
        process_generation=7,
    )


def _guided_clone_config(*, model_id: str = "clone-model") -> dict[str, Any]:
    recipe = next(
        item
        for item in AUDIO_CPP_RECIPE_REGISTRY.recipes
        if "clone" in item.capabilities
        and item.reference_requirement.value == "required"
    )
    accepted = AudioCppAcceptedPackage(
        package_uuid="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        recipe_id=recipe.recipe_id,
        recipe_revision=recipe.recipe_revision,
        package_variant=recipe.package_variant,
        public_model_id=model_id,
        canonical_root="/private/model",
        canonical_root_identity="1" * 64,
        configuration_identity="2" * 64,
        weight_identity="3" * 64,
        projection=recipe.projection,
    )
    return AudioCppSettingsConfig(
        mode="managed",
        managed_setup_source="guided",
        guided_binary_path="/private/audiocpp",
        guided_packages=(accepted,),
        guided_default_model_id=model_id,
    ).model_dump(mode="json")


def _guided_requirement(*, model_id: str = "clone-model") -> TTSCloneRecipeRequirement:
    config = AudioCppSettingsConfig.from_mapping(
        _guided_clone_config(model_id=model_id)
    )
    accepted = config.guided_packages[0]
    return TTSCloneRecipeRequirement(
        recipe_id=accepted.recipe_id,
        recipe_revision=accepted.recipe_revision,
        model_id=model_id,
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
        self.set_error: BaseException | None = None
        self.create_with_assignment_error: BaseException | None = None
        self.remove_error: BaseException | None = None
        self.get_assignment_error: BaseException | None = None
        self.get_profile_error: BaseException | None = None
        self.get_reference_error: BaseException | None = None
        self.count_value = 0
        self.count_generation: int | None = None
        self.advance_generation_during_count = False
        self.coordinator_probe: Callable[[], bool] | None = None
        self.coordinator_active_at_repository_calls: list[bool] = []
        self.list_result: object = _UNSET
        self.create_result: object = _UNSET
        self.create_with_reference_result: object = _UNSET
        self.update_result: object = _UNSET
        self.delete_result: object = _UNSET
        self.set_result: object = _UNSET
        self.remove_result: object = _UNSET
        self.get_assignment_result: object = _UNSET
        self.get_profile_result: object = _UNSET
        self.get_reference_result: object = _UNSET
        self.count_result: object = _UNSET
        self.collision_result = TTSProfileCollisionSnapshot(None, None)
        self.collision_reads = 0
        self.create_boundary: _AsyncBoundary | None = None
        self.set_boundary: _AsyncBoundary | None = None
        self.last_expected_profile: TTSGenerationProfile | None = None
        self.remove_boundary: _AsyncBoundary | None = None
        self.advance_generation_after_get_assignment = False
        self.advance_generation_after_get_profile = False

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

    async def create_profile_with_reference(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID,
        canonical: CanonicalTTSCloneReference,
        recipe_requirement: TTSCloneRecipeRequirement,
        *,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        self._record_coordinator_state()
        self.calls.append(
            (
                "create_with_reference",
                (
                    draft,
                    profile_id,
                    canonical,
                    recipe_requirement,
                    expected_generation,
                    self.generation,
                ),
            )
        )
        if self.create_with_reference_result is not _UNSET:
            return cast(
                ProfileStoreResult[TTSGenerationProfile],
                self.create_with_reference_result,
            )
        summary = TTSCloneReferenceSummary(
            reference_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            byte_length=canonical.byte_length,
            duration_ms=canonical.duration_ms,
            sample_rate_hz=canonical.sample_rate_hz,
            channels=canonical.channels,
            sample_encoding=canonical.sample_encoding,
            created_at=_CREATED_AT,
            updated_at=_CREATED_AT,
            recipe_requirement=recipe_requirement,
        )
        persisted = _profile(
            profile_id=profile_id,
            display_name=draft.display_name,
            provider_id=draft.provider_id,
            model_id=draft.model_id,
            voice_id=draft.voice_id,
            response_format=draft.response_format,
            speed=draft.speed,
            options=dict(draft.options),
            revision=2,
            reference=summary,
        )
        return ProfileStoreResult(generation=self.generation, value=persisted)

    async def create_profile_with_assignment(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID,
        character_ref: CharacterRef,
        *,
        expected_generation: int,
        expected_current_profile_id: UUID | None,
    ) -> ProfileStoreResult[AssignedTTSProfileSnapshot]:
        self._record_coordinator_state()
        self.calls.append(
            (
                "create_with_assignment",
                (
                    draft,
                    profile_id,
                    character_ref,
                    expected_generation,
                    expected_current_profile_id,
                    self.generation,
                ),
            )
        )
        if self.create_with_assignment_error is not None:
            raise self.create_with_assignment_error
        profile = _profile(
            profile_id=profile_id,
            display_name=draft.display_name,
            provider_id=draft.provider_id,
            model_id=draft.model_id,
            voice_id=draft.voice_id,
            response_format=draft.response_format,
            speed=draft.speed,
            options=dict(draft.options),
        )
        return ProfileStoreResult(
            generation=self.generation,
            value=AssignedTTSProfileSnapshot(
                assignment=CharacterTTSAssignment(character_ref, profile_id),
                profile=profile,
            ),
        )

    async def get_profile_collisions(
        self,
        profile_id: UUID,
        draft: TTSProfileDraft,
    ) -> ProfileStoreResult[TTSProfileCollisionSnapshot]:
        self._record_coordinator_state()
        self.calls.append(("collisions", (profile_id, draft, self.generation)))
        self.collision_reads += 1
        value = (
            self.collision_result
            if self.collision_reads == 1
            else TTSProfileCollisionSnapshot(None, None)
        )
        return ProfileStoreResult(generation=self.generation, value=value)

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

    async def set_assignment(
        self,
        character_ref: CharacterRef,
        profile_id: UUID,
        *,
        expected_generation: int,
        expected_profile_revision: int,
        expected_current_profile_id: UUID | None,
        expected_profile: TTSGenerationProfile | None = None,
    ) -> ProfileStoreResult[CharacterTTSAssignment]:
        self._record_coordinator_state()
        self.last_expected_profile = expected_profile
        self.calls.append(
            (
                "set_assignment",
                (
                    character_ref,
                    profile_id,
                    expected_generation,
                    expected_profile_revision,
                    expected_current_profile_id,
                    self.generation,
                ),
            )
        )
        if self.set_boundary is not None:
            await self.set_boundary.wait()
        if self.set_error is not None:
            raise self.set_error
        if self.set_result is not _UNSET:
            return cast(
                ProfileStoreResult[CharacterTTSAssignment],
                self.set_result,
            )
        return ProfileStoreResult(
            generation=self.generation,
            value=CharacterTTSAssignment(
                character_ref=character_ref,
                profile_id=profile_id,
            ),
        )

    async def remove_assignment(
        self,
        character_ref: CharacterRef,
        *,
        expected_generation: int,
        expected_profile_id: UUID,
    ) -> ProfileStoreResult[None]:
        self._record_coordinator_state()
        self.calls.append(
            (
                "remove_assignment",
                (
                    character_ref,
                    expected_generation,
                    expected_profile_id,
                    self.generation,
                ),
            )
        )
        if self.remove_boundary is not None:
            await self.remove_boundary.wait()
        if self.remove_error is not None:
            raise self.remove_error
        if self.remove_result is not _UNSET:
            return cast(ProfileStoreResult[None], self.remove_result)
        return ProfileStoreResult(generation=self.generation, value=None)

    async def get_assigned_profile(
        self,
        character_ref: CharacterRef,
    ) -> ProfileStoreResult[AssignedTTSProfileSnapshot | None]:
        self._record_coordinator_state()
        self.calls.append(("get_assigned_profile", character_ref))
        if self.get_assignment_error is not None:
            raise self.get_assignment_error
        if self.get_assignment_result is not _UNSET:
            result = cast(
                ProfileStoreResult[AssignedTTSProfileSnapshot | None],
                self.get_assignment_result,
            )
        else:
            result = ProfileStoreResult(generation=self.generation, value=None)
        if self.advance_generation_after_get_assignment:
            self.generation += 1
        return result

    async def get_profile(
        self,
        profile_id: UUID,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        self._record_coordinator_state()
        self.calls.append(("get_profile", profile_id))
        if self.get_profile_error is not None:
            raise self.get_profile_error
        if self.get_profile_result is not _UNSET:
            result = cast(
                ProfileStoreResult[TTSGenerationProfile],
                self.get_profile_result,
            )
        else:
            result = ProfileStoreResult(
                generation=self.generation,
                value=_profile(profile_id=profile_id),
            )
        if self.advance_generation_after_get_profile:
            self.generation += 1
        return result

    async def get_reference(
        self,
        profile_id: UUID,
        *,
        expected_revision: int,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSCloneReference]:
        self._record_coordinator_state()
        self.calls.append(
            (
                "get_reference",
                (profile_id, expected_revision, expected_generation),
            )
        )
        if self.get_reference_error is not None:
            raise self.get_reference_error
        if self.get_reference_result is not _UNSET:
            return cast(
                ProfileStoreResult[TTSCloneReference],
                self.get_reference_result,
            )
        return ProfileStoreResult(generation=self.generation, value=_reference())


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


async def _wait_for_boundary_or_task_failure(
    boundary: _AsyncBoundary,
    operation: asyncio.Task[Any],
) -> None:
    entered = asyncio.create_task(boundary.entered.wait())
    try:
        done, _pending = await asyncio.wait(
            (entered, operation),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation in done:
            await operation
        await entered
    finally:
        if not entered.done():
            entered.cancel()
        await asyncio.gather(entered, return_exceptions=True)


async def _start_at_boundary(
    operation: Coroutine[Any, Any, _TaskResult],
    boundary: _AsyncBoundary,
) -> asyncio.Task[_TaskResult]:
    task = asyncio.create_task(operation)
    try:
        async with asyncio.timeout(1):
            await _wait_for_boundary_or_task_failure(boundary, task)
    except BaseException:
        boundary.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    return task


async def _settle_boundary_task(
    boundary: _AsyncBoundary,
    task: asyncio.Task[Any],
) -> None:
    boundary.release.set()
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


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
        self.revisions: dict[str, int] = {}
        self.saved_revisions: dict[str, int] = {}
        self.applied_revisions: dict[str, int] = {}
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
        self.dependency_snapshots: dict[
            TTSCloneRecipeRequirement,
            tts_generation.AudioCppGuidedDependencySnapshot,
        ] = {}
        self.dependency_calls: list[TTSCloneRecipeRequirement] = []

    async def audio_cpp_guided_dependency_snapshot(
        self,
        requirement: TTSCloneRecipeRequirement,
    ) -> tts_generation.AudioCppGuidedDependencySnapshot:
        self.dependency_calls.append(requirement)
        return self.dependency_snapshots.get(
            requirement,
            tts_generation.AudioCppGuidedDependencySnapshot(
                state="exact",
                provider_configuration_revision=self.revision,
                saved_generation=1,
                applied_generation=1,
                pending_configuration=False,
                saved_requirement=requirement,
                applied_requirement=requirement,
            ),
        )

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
        return self.revisions.get(provider_id, self.revision)

    def saved_configuration_revision(self, provider_id: str) -> int:
        return self.saved_revisions.get(provider_id, 0)

    def applied_configuration_revision(self, provider_id: str) -> int:
        return self.applied_revisions.get(provider_id, 0)

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
            current_revision = self.revisions.get(provider_id, self.revision)
            if self.stale_decision or current_revision != expected_revision:
                raise TTSConfigurationRevisionError(
                    "https://user:credential@example.test/private/path"
                )
            if self.reconfigure_after_decision:
                if provider_id in self.revisions:
                    self.revisions[provider_id] += 1
                else:
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
    artifact_lease_coordinator: object | None = None,
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
            artifact_lease_coordinator=artifact_lease_coordinator,
        ),
        selected_repository,
        selected_tts_service,
    )


class _ArtifactLeaseCoordinator:
    def __init__(self) -> None:
        self.active = False
        self.calls: list[tuple[AudioCppArtifactConsumerRequirement, ...]] = []

    @asynccontextmanager
    async def lease_consumers(self, consumers):
        exact = tuple(consumers)
        self.calls.append(exact)
        self.active = True
        try:
            yield
        finally:
            self.active = False


@pytest.mark.asyncio
async def test_bounded_consumer_snapshot_serializes_constant_generation_reorder() -> (
    None
):
    first_page_entered = asyncio.Event()
    release_first_page = asyncio.Event()
    update_called = asyncio.Event()
    first = _profile(display_name="A")
    middle = tuple(
        _profile(
            profile_id=UUID(int=index + 2),
            display_name=f"C{index:02d}",
        )
        for index in range(49)
    )
    target = _profile(
        profile_id=UUID(int=100),
        display_name="TARGET",
    )
    last = _profile(profile_id=UUID(int=101), display_name="Z")

    class ReorderingRepository(_FakeRepository):
        def __init__(self) -> None:
            super().__init__()
            self.profiles = {
                profile.profile_id: profile
                for profile in (first, *middle, target, last)
            }

        async def list_profiles(
            self,
            search: str | None = None,
            limit: int = 50,
            offset: int = 0,
        ) -> ProfileStoreResult[TTSProfilePage]:
            assert search is None
            ordered = tuple(
                sorted(self.profiles.values(), key=lambda item: item.normalized_name)
            )
            if offset == 0:
                first_page_entered.set()
                await release_first_page.wait()
            return ProfileStoreResult(
                generation=self.generation,
                value=TTSProfilePage(
                    profiles=ordered[offset : offset + limit],
                    total=len(ordered),
                ),
            )

        async def get_profile(
            self,
            profile_id: UUID,
        ) -> ProfileStoreResult[TTSGenerationProfile]:
            return ProfileStoreResult(
                generation=self.generation,
                value=self.profiles[profile_id],
            )

        async def assignment_count(
            self,
            profile_id: UUID,
        ) -> ProfileStoreResult[int]:
            assert profile_id in self.profiles
            return ProfileStoreResult(generation=self.generation, value=0)

        async def update_profile(
            self,
            profile_id: UUID,
            expected_revision: int,
            draft: TTSProfileDraft,
            *,
            expected_generation: int,
        ) -> ProfileStoreResult[TTSGenerationProfile]:
            update_called.set()
            updated = _profile(
                profile_id=profile_id,
                display_name=draft.display_name,
                provider_id=draft.provider_id,
                model_id=draft.model_id,
                voice_id=draft.voice_id,
                response_format=draft.response_format,
                speed=draft.speed,
                options=dict(draft.options),
                revision=expected_revision + 1,
            )
            self.profiles[profile_id] = updated
            return ProfileStoreResult(generation=expected_generation, value=updated)

    repository = ReorderingRepository()
    service, _repository, _tts_service = _service(repository=repository)
    loaded_first = LoadedTTSProfile(repository.generation, first)
    renamed = TTSProfileDraft(
        display_name="Y",
        provider_id=first.provider_id,
        model_id=first.model_id,
        voice_id=first.voice_id,
        response_format=first.response_format,
        speed=first.speed,
        options=first.options,
    )

    snapshot = asyncio.create_task(service.bounded_profile_assignment_snapshot())
    await first_page_entered.wait()
    rename = asyncio.create_task(service.update_profile(loaded_first, renamed))
    await asyncio.sleep(0)
    assert update_called.is_set() is False

    release_first_page.set()
    captured = await snapshot
    await rename

    assert target.profile_id in {profile.profile_id for profile, _count in captured}
    assert len(captured) == 52
    assert update_called.is_set() is True


@pytest.mark.asyncio
async def test_bounded_consumer_snapshot_rejects_inventory_over_limit() -> None:
    repository = _FakeRepository()
    repository.page = TTSProfilePage(profiles=(), total=201)
    service, _repository, _tts_service = _service(repository=repository)

    with pytest.raises(ProfileServiceError, match="operation_failed"):
        await service.bounded_profile_assignment_snapshot()


@pytest.mark.asyncio
async def test_artifact_lease_covers_profile_create_repository_commit() -> None:
    coordinator = _ArtifactLeaseCoordinator()
    repository = _FakeRepository()
    repository.coordinator_probe = lambda: coordinator.active
    service, repository, _tts_service = _service(
        repository=repository,
        artifact_lease_coordinator=coordinator,
    )
    repository.coordinator_probe = lambda: coordinator.active

    await service.create_from_artifact("Saved", _artifact(selection=_selection()))

    assert repository.coordinator_active_at_repository_calls == [True]
    assert coordinator.calls == [
        (
            AudioCppArtifactConsumerRequirement(
                provider_id="audio_cpp",
                model_id="selected-model",
            ),
        )
    ]


@pytest.mark.asyncio
async def test_artifact_lease_covers_profile_delete_repository_commit() -> None:
    coordinator = _ArtifactLeaseCoordinator()
    repository = _FakeRepository()
    repository.coordinator_probe = lambda: coordinator.active
    service, repository, _tts_service = _service(
        repository=repository,
        artifact_lease_coordinator=coordinator,
    )
    repository.coordinator_probe = lambda: coordinator.active
    loaded = LoadedTTSProfile(repository.generation, _profile())

    await service.delete_profile(loaded)

    assert repository.coordinator_active_at_repository_calls == [True]
    assert coordinator.calls == [
        (
            AudioCppArtifactConsumerRequirement(
                provider_id="audio_cpp",
                model_id="model-a",
            ),
        )
    ]


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
    preset = TTSPlaygroundSelectionPreset(
        provider_id="audio_cpp",
        model_id="model",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        availability="unavailable",
    )

    assert page.profiles == (profile,)
    assert page.profiles[0] is not profile
    assert loaded.profile == profile
    assert loaded.profile is not profile
    assert snapshot.profiles == (availability,)
    assert preset.options == {}
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
            TTSProfileAvailability,
            {
                "profile_id": _PROFILE_ID,
                "state": "available",
                "recovery_action": "none",
                "provider_configuration_revision": True,
            },
            "configuration_revision",
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
            provider_id="openai",
            voice_id="alloy",
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
                    provider_id="openai",
                    voice_id="alloy",
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
    """audio.cpp profiles still gate on native catalog data; legacy-provider
    profiles are now structurally in the allowlist too, but classify as the
    interim honest "unverified" state without ever being probed against the
    native (audio.cpp) catalog, since legacy providers have no catalog
    authority -- their model ids are excluded from the capability batch
    entirely (task 2b), not merely left unclassified."""

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
    legacy_with_voice = _profile(
        profile_id=UUID(int=3),
        display_name="Legacy voice",
        provider_id="openai",
        model_id="legacy-model",
        voice_id="legacy-voice",
    )
    legacy_other_format = _profile(
        profile_id=UUID(int=4),
        display_name="Legacy format",
        provider_id="openai",
        model_id="legacy-format",
        voice_id="legacy-voice",
        response_format="mp3",
    )
    legacy_custom_speed = _profile(
        profile_id=UUID(int=5),
        display_name="Legacy speed",
        provider_id="openai",
        model_id="legacy-speed",
        voice_id="legacy-voice",
        speed=1.25,
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
            legacy_with_voice,
            legacy_other_format,
            legacy_custom_speed,
        ),
        total=5,
    )

    observed = await service.observe_availability(page)

    assert tts_service.capability_calls == [("audio_cpp", ("model-a",))]
    assert tuple(item.state for item in observed.profiles) == (
        "available",
        "available",
        "unverified",
        "unverified",
        "unverified",
    )
    assert tuple(item.recovery_action for item in observed.profiles) == (
        "none",
        "none",
        "none",
        "none",
        "none",
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
            provider_id="openai",
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
    # Every recognized provider is now structurally valid at construction
    # (Task 1's per-provider contract table), so the only profile that can
    # still fail the allowlist is one referencing a provider outside the
    # closed seven-provider set -- and since `TTSGenerationProfile`
    # construction itself enforces that same contract, such a profile can
    # only exist as forged (pre-expansion or otherwise hostile) data.
    # `observe_availability` fails closed on it before any capability
    # lookup, matching the file's other hostile-page tests.
    unsupported = _forged_profile(
        _profile(model_id="model", voice_id="voice"),
        provider_id="unrecognized_future_provider",
    )
    service, repository, tts_service = _service()
    page = _forged_page_snapshot(
        repository_generation=repository.generation,
        profiles=(unsupported,),
        total=1,
    )

    with pytest.raises(ProfileValidationError) as caught:
        await service.observe_availability(page)

    assert caught.value.code == "profiles"
    assert tts_service.capability_calls == []


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
            profiles=(first, second, server_default),
            total=3,
        )
    )

    assert tts_service.capability_calls == [("audio_cpp", ("shared",))]
    assert tuple(item.state for item in observed.profiles) == (
        "available",
        "unavailable",
        "available",
    )


@pytest.mark.asyncio
async def test_availability_all_legacy_page_skips_native_capability_call() -> None:
    def _raise() -> None:
        raise RuntimeError("get_native_capability_snapshot must not be called")

    openai_profile = _profile(
        profile_id=UUID(int=20),
        display_name="OpenAI voice",
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
    )
    elevenlabs_profile = _profile(
        profile_id=UUID(int=21),
        display_name="ElevenLabs voice",
        provider_id="elevenlabs",
        model_id="eleven_multilingual_v2",
        voice_id="rachel",
        response_format="mp3",
    )
    tts_service = _FakeTTSService()
    tts_service.revisions = {
        "audio_cpp": 3,
        "openai": 9,
        "elevenlabs": 11,
    }
    tts_service.capability_hook = _raise
    service, repository, tts_service = _service(tts_service=tts_service)

    observed = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(openai_profile, elevenlabs_profile),
            total=2,
        )
    )

    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert tts_service.revision_reads == [
        "openai",
        "elevenlabs",
        "audio_cpp",
        "openai",
        "elevenlabs",
    ]
    assert tuple(item.state for item in observed.profiles) == (
        "unverified",
        "unverified",
    )
    assert tuple(item.recovery_action for item in observed.profiles) == (
        "none",
        "none",
    )
    assert tuple(
        item.provider_configuration_revision for item in observed.profiles
    ) == (9, 11)
    assert observed.catalog_revision is None
    assert observed.repository_generation == repository.generation
    assert observed.configuration_revision == tts_service.revision


def test_availability_value_admits_an_inert_recovery_for_unverified() -> None:
    """ "unverified" now has two honest recoveries, one per provider class."""

    profile_id = _PROFILE_ID
    refreshable = TTSProfileAvailability(
        profile_id=profile_id,
        state="unverified",
        recovery_action="refresh",
    )
    inert = TTSProfileAvailability(
        profile_id=profile_id,
        state="unverified",
        recovery_action="none",
    )

    assert refreshable.recovery_action == "refresh"
    assert inert.recovery_action == "none"
    with pytest.raises(ProfileValidationError) as caught:
        TTSProfileAvailability(
            profile_id=profile_id,
            state="unverified",
            recovery_action="edit",
        )
    assert caught.value.code == "recovery_action"


@pytest.mark.asyncio
async def test_availability_mixed_page_probes_only_audio_cpp_models() -> None:
    audio_cpp_profile = _profile(
        profile_id=UUID(int=30),
        display_name="Native",
        model_id="model-a",
        voice_id="voice-a",
    )
    openai_profile = _profile(
        profile_id=UUID(int=31),
        display_name="OpenAI voice",
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
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
            models=(_model("model-a"),),
            voice_results={"model-a": voice_result},
        )
    )
    tts_service.revisions = {"audio_cpp": 3, "openai": 9}
    service, repository, tts_service = _service(tts_service=tts_service)

    observed = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(audio_cpp_profile, openai_profile),
            total=2,
        )
    )

    assert tts_service.capability_calls == [("audio_cpp", ("model-a",))]
    assert tuple(item.state for item in observed.profiles) == (
        "available",
        "unverified",
    )
    # ADR-031: a legacy provider has no catalog to preflight, so its
    # "unverified" is permanent and Refresh must not be offered as a
    # recovery it can never perform. audio.cpp keeps "refresh".
    assert tuple(item.recovery_action for item in observed.profiles) == (
        "none",
        "none",
    )
    assert tuple(
        item.provider_configuration_revision for item in observed.profiles
    ) == (3, 9)


def test_current_revision_reads_requested_active_provider_not_publication_counters() -> (
    None
):
    tts_service = _FakeTTSService()
    tts_service.revisions = {"audio_cpp": 2, "openai": 41}
    tts_service.saved_revisions = {"openai": 7}
    tts_service.applied_revisions = {"openai": 7}
    service, _repository, _tts_service = _service(tts_service=tts_service)

    assert service._current_configuration_revision("openai") == 41
    assert tts_service.revision_reads == ["openai"]


@pytest.mark.asyncio
async def test_openai_profile_created_from_sample_is_available_this_process(
    successful_audio_path: Path,
) -> None:
    tts_service = _FakeTTSService()
    tts_service.revisions = {"audio_cpp": 2, "openai": 41}
    tts_service.saved_revisions = {"openai": 7}
    tts_service.applied_revisions = {"openai": 7}
    service, repository, _tts_service = _service(tts_service=tts_service)
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )

    loaded = await service.create_from_artifact(
        "Pocket Alba",
        _successful_artifact(selection, successful_audio_path),
    )
    observed = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(loaded.profile,),
            total=1,
        )
    )

    assert observed.configuration_revision == 2
    assert observed.catalog_revision is None
    assert observed.profiles[0].state == "available"
    assert observed.profiles[0].provider_configuration_revision == 41
    evidence = service._sample_evidence[loaded.profile.profile_id]
    assert evidence.profile_revision == loaded.profile.revision
    assert evidence.options_fingerprint == profile_options_fingerprint({})
    assert "credential" not in repr(evidence)
    assert "submitted text" not in repr(evidence)


@pytest.mark.asyncio
async def test_openai_profile_evidence_invalidates_on_active_revision_change(
    successful_audio_path: Path,
) -> None:
    tts_service = _FakeTTSService()
    tts_service.revisions = {"audio_cpp": 2, "openai": 41}
    service, repository, _tts_service = _service(tts_service=tts_service)
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )
    loaded = await service.create_from_artifact(
        "Pocket Alba",
        _successful_artifact(selection, successful_audio_path),
    )

    tts_service.revisions["openai"] = 42
    observed = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(loaded.profile,),
            total=1,
        )
    )

    assert observed.profiles[0].state == "unverified"
    assert observed.profiles[0].provider_configuration_revision == 42

    tts_service.revisions["openai"] = 41
    observed_after_revert = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(loaded.profile,),
            total=1,
        )
    )
    assert observed_after_revert.profiles[0].state == "unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "artifact_overrides",
    (
        {"provider_id": "elevenlabs"},
        {"model_id": "other-model"},
        {"voice_id": "other-voice"},
        {"audio_format": "mp3"},
    ),
)
async def test_sample_evidence_rejects_malformed_or_mismatched_artifact(
    artifact_overrides: dict[str, object],
    successful_audio_path: Path,
) -> None:
    tts_service = _FakeTTSService()
    tts_service.revisions = {"audio_cpp": 2, "openai": 41}
    service, repository, _tts_service = _service(tts_service=tts_service)
    profile = _profile(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
    )
    loaded = LoadedTTSProfile(repository.generation, profile)
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )

    service.record_sample_evidence(
        loaded,
        _successful_artifact(
            selection,
            successful_audio_path,
            **artifact_overrides,
        ),  # type: ignore[arg-type]
    )

    assert profile.profile_id not in service._sample_evidence


@pytest.mark.parametrize(
    "invalid_kind",
    ("missing", "empty", "malformed", "oversized", "directory", "symlink"),
)
def test_sample_evidence_requires_bounded_playable_regular_audio(
    tmp_path: Path,
    invalid_kind: str,
    successful_audio_path: Path,
) -> None:
    path = tmp_path / f"{invalid_kind}.wav"
    if invalid_kind == "empty":
        path.write_bytes(b"")
    elif invalid_kind == "malformed":
        path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    elif invalid_kind == "oversized":
        path.write_bytes(b"x" * ((8 * 1024 * 1024) + 1))
    elif invalid_kind == "directory":
        path.mkdir()
    elif invalid_kind == "symlink":
        path.symlink_to(successful_audio_path)

    tts_service = _FakeTTSService()
    tts_service.revisions = {"openai": 41}
    service, repository, _tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository.generation,
        _profile(
            provider_id="openai",
            model_id="pocket-tts",
            voice_id="alba",
            response_format="wav",
        ),
    )
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )

    service.record_sample_evidence(loaded, _successful_artifact(selection, path))

    assert service._sample_evidence == {}


def test_sample_evidence_rejects_relative_artifact_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    relative = Path("relative.wav")
    monkeypatch.chdir(tmp_path)
    with wave.open(str(relative), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(struct.pack("<h", 100) * 32)
    tts_service = _FakeTTSService()
    tts_service.revisions = {"openai": 41}
    service, repository, _tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository.generation,
        _profile(
            provider_id="openai",
            model_id="pocket-tts",
            voice_id="alba",
            response_format="wav",
        ),
    )
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )

    service.record_sample_evidence(loaded, _successful_artifact(selection, relative))

    assert service._sample_evidence == {}


def test_sample_evidence_rejects_path_replaced_after_bounded_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    successful_audio_path: Path,
) -> None:
    path = tmp_path / "replaceable.wav"
    path.write_bytes(successful_audio_path.read_bytes())
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(successful_audio_path.read_bytes())
    original_read = sample_audio_validation._read_bounded_regular_file

    def replace_after_read(
        artifact_path: Path,
        max_bytes: int,
    ) -> tuple[bytes, os.stat_result] | None:
        result = original_read(artifact_path, max_bytes)
        os.replace(replacement, artifact_path)
        return result

    monkeypatch.setattr(
        sample_audio_validation,
        "_read_bounded_regular_file",
        replace_after_read,
    )
    tts_service = _FakeTTSService()
    tts_service.revisions = {"openai": 41}
    service, repository, _tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository.generation,
        _profile(
            provider_id="openai",
            model_id="pocket-tts",
            voice_id="alba",
            response_format="wav",
        ),
    )
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )

    service.record_sample_evidence(loaded, _successful_artifact(selection, path))

    assert service._sample_evidence == {}


def test_sample_evidence_rejects_failed_cancelled_or_forged_values() -> None:
    tts_service = _FakeTTSService()
    tts_service.revisions = {"openai": 41}
    service, repository, _tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository.generation,
        _profile(
            provider_id="openai",
            model_id="pocket-tts",
            voice_id="alba",
            response_format="mp3",
        ),
    )
    forged = object.__new__(STTSGeneratedAudio)

    service.record_sample_evidence(loaded, cast(STTSGeneratedAudio, object()))
    service.record_sample_evidence(loaded, forged)

    assert service._sample_evidence == {}


@pytest.mark.asyncio
async def test_edit_and_delete_clear_process_sample_evidence(
    successful_audio_path: Path,
) -> None:
    tts_service = _FakeTTSService()
    tts_service.revisions = {"openai": 41}
    service, _repository, _tts_service = _service(tts_service=tts_service)
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )
    loaded = await service.create_from_artifact(
        "Pocket Alba",
        _successful_artifact(selection, successful_audio_path),
    )
    assert loaded.profile.profile_id in service._sample_evidence

    updated = await service.update_profile(
        loaded,
        TTSProfileDraft(
            display_name="Pocket Alba edited",
            provider_id="openai",
            model_id="pocket-tts",
            voice_id="alba",
            response_format="wav",
            speed=1.0,
            options={},
        ),
    )
    assert loaded.profile.profile_id not in service._sample_evidence

    service.record_sample_evidence(
        updated,
        _successful_artifact(selection, successful_audio_path),
    )
    assert updated.profile.profile_id in service._sample_evidence
    await service.delete_profile(updated)
    assert updated.profile.profile_id not in service._sample_evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("edit", "delete"))
async def test_inflight_sample_cannot_reinsert_after_profile_mutation(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    successful_audio_path: Path,
) -> None:
    tts_service = _FakeTTSService()
    tts_service.revisions = {"openai": 41}
    service, repository, _tts_service = _service(tts_service=tts_service)
    profile = _profile(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
    )
    loaded = LoadedTTSProfile(repository.generation, profile)
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )
    artifact = _successful_artifact(selection, successful_audio_path)
    validation_started = threading.Event()
    validation_release = threading.Event()
    real_validate = profile_service.validate_playable_audio_file

    def blocked_validation(*args: object, **kwargs: object) -> object:
        validation_started.set()
        assert validation_release.wait(2)
        return real_validate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        profile_service,
        "validate_playable_audio_file",
        blocked_validation,
    )
    worker = threading.Thread(
        target=service.record_sample_evidence,
        args=(loaded, artifact),
    )
    worker.start()
    assert await asyncio.to_thread(validation_started.wait, 2)

    if mutation == "edit":
        await service.update_profile(
            loaded,
            TTSProfileDraft(
                display_name="Edited while validating",
                provider_id="openai",
                model_id="pocket-tts",
                voice_id="alba",
                response_format="wav",
                speed=1.0,
                options={},
            ),
        )
    else:
        await service.delete_profile(loaded)
    validation_release.set()
    await asyncio.to_thread(worker.join, 2)

    assert not worker.is_alive()
    assert profile.profile_id not in service._sample_evidence
    service.record_sample_evidence(loaded, artifact)
    assert profile.profile_id not in service._sample_evidence


@pytest.mark.asyncio
async def test_concurrent_observation_cannot_publish_deleted_profile_as_available(
    monkeypatch: pytest.MonkeyPatch,
    successful_audio_path: Path,
) -> None:
    tts_service = _FakeTTSService()
    tts_service.revisions = {"audio_cpp": 2, "openai": 41}
    service, repository, _tts_service = _service(tts_service=tts_service)
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )
    loaded = await service.create_from_artifact(
        "Pocket Alba",
        _successful_artifact(selection, successful_audio_path),
    )
    page = TTSProfilePageSnapshot(
        repository_generation=repository.generation,
        profiles=(loaded.profile,),
        total=1,
    )
    classification_started = threading.Event()
    classification_release = threading.Event()
    original_classify = service._classify_profile_with_evidence

    def blocked_classification(*args: object, **kwargs: object) -> object:
        classification_started.set()
        assert classification_release.wait(2)
        return original_classify(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        service,
        "_classify_profile_with_evidence",
        blocked_classification,
    )
    observed: list[TTSProfileAvailabilitySnapshot] = []

    def observe_in_thread() -> None:
        observed.append(asyncio.run(service.observe_availability(page)))

    worker = threading.Thread(target=observe_in_thread)
    worker.start()
    assert await asyncio.to_thread(classification_started.wait, 2)
    await service.delete_profile(loaded)
    classification_release.set()
    await asyncio.to_thread(worker.join, 2)

    assert not worker.is_alive()
    assert observed[0].profiles[0].state == "unverified"


@pytest.mark.asyncio
async def test_new_service_and_unrecorded_profile_have_no_sample_evidence(
    successful_audio_path: Path,
) -> None:
    tts_service = _FakeTTSService()
    tts_service.revisions = {"audio_cpp": 2, "openai": 41}
    first, repository, _tts_service = _service(tts_service=tts_service)
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )
    loaded = await first.create_from_artifact(
        "Pocket Alba",
        _successful_artifact(selection, successful_audio_path),
    )
    restarted = TTSProfileService(repository, tts_service)

    observed = await restarted.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(loaded.profile,),
            total=1,
        )
    )

    assert observed.profiles[0].state == "unverified"
    assert restarted._sample_evidence == {}


def test_sample_evidence_cache_concurrent_admission_retains_every_bounded_id(
    successful_audio_path: Path,
) -> None:
    tts_service = _FakeTTSService()
    tts_service.revisions = {"openai": 41}
    service, repository, _tts_service = _service(tts_service=tts_service)
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )
    artifact = _successful_artifact(selection, successful_audio_path)
    loaded_profiles = tuple(
        LoadedTTSProfile(
            repository.generation,
            _profile(
                profile_id=UUID(int=index + 1),
                provider_id="openai",
                model_id="pocket-tts",
                voice_id="alba",
                response_format="wav",
            ),
        )
        for index in range(128)
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(
            executor.map(
                lambda loaded: service.record_sample_evidence(loaded, artifact),
                loaded_profiles,
            )
        )

    assert set(service._sample_evidence) == {
        loaded.profile.profile_id for loaded in loaded_profiles
    }


def test_sample_evidence_cache_uses_deterministic_fifo_eviction(
    successful_audio_path: Path,
) -> None:
    tts_service = _FakeTTSService()
    tts_service.revisions = {"openai": 41}
    service, repository, _tts_service = _service(tts_service=tts_service)
    selection = _selection(
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        configuration_revision=41,
    )
    artifact = _successful_artifact(selection, successful_audio_path)

    for index in range(300):
        service.record_sample_evidence(
            LoadedTTSProfile(
                repository.generation,
                _profile(
                    profile_id=UUID(int=index + 1),
                    provider_id="openai",
                    model_id="pocket-tts",
                    voice_id="alba",
                    response_format="wav",
                ),
            ),
            artifact,
        )

    assert tuple(service._sample_evidence) == tuple(
        UUID(int=index) for index in range(45, 301)
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
async def test_create_clone_from_artifact_uses_exact_success_evidence_atomically() -> (
    None
):
    service, repository, tts_service = _service()
    selection = _selection()
    evidence = _clone_evidence()
    artifact = _artifact(selection=selection, clone_evidence=evidence)

    loaded = await service.create_clone_from_artifact(" Clone voice ", artifact)

    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert len(repository.calls) == 1
    call_name, call_value = repository.calls[0]
    assert call_name == "create_with_reference"
    (
        draft,
        profile_id,
        canonical,
        requirement,
        expected_generation,
        generation_at_call,
    ) = call_value  # type: ignore[misc]
    assert draft == TTSProfileDraft(
        display_name="Clone voice",
        provider_id="audio_cpp",
        model_id="selected-model",
        voice_id="selected-voice",
        response_format="wav",
        speed=1.0,
        options={},
    )
    assert type(profile_id) is UUID
    assert canonical == evidence.canonical_reference
    assert requirement == TTSCloneRecipeRequirement(
        recipe_id=evidence.recipe_id,
        recipe_revision=evidence.recipe_revision,
        model_id=evidence.model_id,
    )
    assert expected_generation == 7
    assert generation_at_call == 7
    assert loaded.repository_generation == 7
    assert loaded.profile.revision == 2
    assert loaded.profile.reference is not None
    assert loaded.profile.reference.recipe_requirement == requirement


@pytest.mark.asyncio
async def test_create_clone_from_artifact_rejects_missing_or_mismatched_evidence() -> (
    None
):
    service, repository, tts_service = _service()
    selection = _selection()

    with pytest.raises(ProfileServiceError) as missing:
        await service.create_clone_from_artifact(
            "Clone voice",
            _artifact(selection=selection),
        )
    _assert_safe_service_error(missing.value, "artifact_ineligible")

    mismatched = TTSRequestedSelectionSnapshot(
        provider_id="audio_cpp",
        model_id="other-model",
        voice_id="selected-voice",
        response_format="wav",
        speed=1.0,
        options={},
        configuration_revision=3,
    )
    with pytest.raises(ProfileServiceError) as mismatch:
        await service.create_clone_from_artifact(
            "Clone voice",
            _artifact(selection=mismatched, clone_evidence=_clone_evidence()),
        )
    _assert_safe_service_error(mismatch.value, "artifact_ineligible")
    assert repository.calls == []
    assert tts_service.revision_decisions == []


@pytest.mark.asyncio
async def test_create_from_artifact_accepts_legacy_selection() -> None:
    service, repository, tts_service = _service()
    selection = _selection(
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        response_format="mp3",
        speed=1.0,
    )

    await service.create_from_artifact("OpenAI voice", _artifact(selection=selection))

    assert tts_service.revision_decisions == [("openai", 3)]
    assert tts_service.capability_calls == []
    assert [name for name, _ in repository.calls] == ["create"]


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
async def test_guided_dependency_snapshot_reports_exact_without_provider_work() -> None:
    config = _guided_clone_config()

    class _PureRegistry:
        acquire_calls = 0

        def descriptors(self) -> tuple[object, ...]:
            return ()

        async def provider_configuration_snapshot(
            self,
            provider_id: str,
        ) -> TTSProviderConfigurationSnapshot:
            assert provider_id == "audio_cpp"
            return TTSProviderConfigurationSnapshot(
                revision=4,
                applied_generation=2,
                applied_config=config,
                staged_generation=None,
                staged_config=None,
            )

        async def acquire(self, _provider_id: str) -> object:
            self.acquire_calls += 1
            raise AssertionError("pure dependency inspection acquired an adapter")

    registry = _PureRegistry()
    service = tts_generation.TTSService(cast(Any, registry))

    snapshot = await service.audio_cpp_guided_dependency_snapshot(_guided_requirement())

    assert snapshot.state == "exact"
    assert snapshot.saved_requirement == _guided_requirement()
    assert snapshot.applied_requirement == _guided_requirement()
    assert snapshot.pending_configuration is False
    assert registry.acquire_calls == 0


@pytest.mark.parametrize("pending_configuration", (False, True))
@pytest.mark.parametrize("applied_kind", ("none", "exact", "other"))
@pytest.mark.parametrize("saved_kind", ("none", "exact", "other"))
@pytest.mark.parametrize("state", ("exact", "missing", "mismatch", "pending"))
def test_guided_dependency_snapshot_validator_enforces_full_producer_matrix(
    state: str,
    saved_kind: str,
    applied_kind: str,
    pending_configuration: bool,
) -> None:
    requirement = _guided_requirement()
    other = _guided_requirement(model_id="other-model")
    observed = {
        "none": None,
        "exact": requirement,
        "other": other,
    }
    snapshot = tts_generation.AudioCppGuidedDependencySnapshot(
        state=state,  # type: ignore[arg-type]
        provider_configuration_revision=4,
        saved_generation=2 if pending_configuration else 1,
        applied_generation=1,
        pending_configuration=pending_configuration,
        saved_requirement=observed[saved_kind],
        applied_requirement=observed[applied_kind],
    )
    if "other" in {saved_kind, applied_kind}:
        expected = False
    elif not pending_configuration and saved_kind != applied_kind:
        expected = False
    elif applied_kind == "exact":
        expected = state == "exact"
    elif pending_configuration and saved_kind == "exact":
        expected = state == "pending"
    else:
        expected = state in {"missing", "mismatch"}

    validated = tts_generation.validate_audio_cpp_guided_dependency_snapshot(
        snapshot,
        requirement,
    )

    assert (validated is not None) is expected
    if expected:
        assert validated == snapshot
        assert validated is not snapshot
        if snapshot.saved_requirement is not None:
            assert validated.saved_requirement is not snapshot.saved_requirement
        if snapshot.applied_requirement is not None:
            assert validated.applied_requirement is not snapshot.applied_requirement


def test_guided_dependency_snapshot_validation_owns_canonical_evidence() -> None:
    source_requirement = _guided_requirement()
    source = tts_generation.AudioCppGuidedDependencySnapshot(
        state="exact",
        provider_configuration_revision=4,
        saved_generation=1,
        applied_generation=1,
        pending_configuration=False,
        saved_requirement=source_requirement,
        applied_requirement=source_requirement,
    )

    validated = tts_generation.validate_audio_cpp_guided_dependency_snapshot(
        source,
        source_requirement,
    )
    assert validated is not None
    object.__setattr__(source_requirement, "model_id", "hostile-model")
    object.__setattr__(source, "state", "mismatch")

    assert validated.state == "exact"
    assert validated.saved_requirement == _guided_requirement()
    assert validated.applied_requirement == _guided_requirement()


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("provider_configuration_revision", True),
        ("saved_generation", False),
        ("applied_generation", True),
        ("pending_configuration", 0),
    ),
)
def test_guided_dependency_snapshot_validator_requires_strict_bool_and_int_fields(
    field: str,
    invalid: object,
) -> None:
    requirement = _guided_requirement()
    values: dict[str, object] = {
        "state": "exact",
        "provider_configuration_revision": 4,
        "saved_generation": 1,
        "applied_generation": 1,
        "pending_configuration": False,
        "saved_requirement": requirement,
        "applied_requirement": requirement,
    }
    values[field] = invalid
    snapshot = tts_generation.AudioCppGuidedDependencySnapshot(**values)  # type: ignore[arg-type]

    assert (
        tts_generation.validate_audio_cpp_guided_dependency_snapshot(
            snapshot,
            requirement,
        )
        is None
    )


def test_guided_dependency_snapshot_validator_contains_hollow_exact_type_objects() -> (
    None
):
    requirement = _guided_requirement()
    hollow_snapshot = object.__new__(tts_generation.AudioCppGuidedDependencySnapshot)
    hollow_requirement = object.__new__(TTSCloneRecipeRequirement)
    snapshot_with_hollow_nested = tts_generation.AudioCppGuidedDependencySnapshot(
        state="exact",
        provider_configuration_revision=4,
        saved_generation=1,
        applied_generation=1,
        pending_configuration=False,
        saved_requirement=hollow_requirement,
        applied_requirement=hollow_requirement,
    )
    exact_snapshot = tts_generation.AudioCppGuidedDependencySnapshot(
        state="exact",
        provider_configuration_revision=4,
        saved_generation=1,
        applied_generation=1,
        pending_configuration=False,
        saved_requirement=requirement,
        applied_requirement=requirement,
    )

    assert (
        tts_generation.validate_audio_cpp_guided_dependency_snapshot(
            hollow_snapshot,
            requirement,
        )
        is None
    )
    assert (
        tts_generation.validate_audio_cpp_guided_dependency_snapshot(
            snapshot_with_hollow_nested,
            requirement,
        )
        is None
    )
    assert (
        tts_generation.validate_audio_cpp_guided_dependency_snapshot(
            exact_snapshot,
            hollow_requirement,
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requirement", "expected_state"),
    (
        (
            TTSCloneRecipeRequirement(
                recipe_id="future.valid.recipe",
                recipe_revision=1,
                model_id="clone-model",
            ),
            "missing",
        ),
        (
            TTSCloneRecipeRequirement(
                recipe_id=_guided_requirement().recipe_id,
                recipe_revision=_guided_requirement().recipe_revision + 1,
                model_id="clone-model",
            ),
            "mismatch",
        ),
    ),
)
async def test_guided_dependency_snapshot_classifies_missing_and_mismatch(
    requirement: TTSCloneRecipeRequirement,
    expected_state: str,
) -> None:
    config = _guided_clone_config()

    class _PureRegistry:
        def descriptors(self) -> tuple[object, ...]:
            return ()

        async def provider_configuration_snapshot(
            self, _provider_id: str
        ) -> TTSProviderConfigurationSnapshot:
            return TTSProviderConfigurationSnapshot(
                revision=4,
                applied_generation=2,
                applied_config=config,
                staged_generation=None,
                staged_config=None,
            )

        async def acquire(self, _provider_id: str) -> object:
            raise AssertionError("dependency inspection acquired an adapter")

    service = tts_generation.TTSService(cast(Any, _PureRegistry()))

    snapshot = await service.audio_cpp_guided_dependency_snapshot(requirement)

    assert snapshot.state == expected_state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drift",
    ("projection_family", "package_variant", "recipe_revision", "model_id"),
)
async def test_guided_dependency_snapshot_preserves_present_config_drift_as_mismatch(
    drift: str,
) -> None:
    config = _guided_clone_config()
    package = config["guided_packages"][0]
    if drift == "projection_family":
        package["projection"]["family"] = "drifted_family"
    elif drift == "package_variant":
        package["package_variant"] = "drifted_variant"
    elif drift == "recipe_revision":
        package["recipe_revision"] += 1
    else:
        package["public_model_id"] = "different-model"
        config["guided_default_model_id"] = "different-model"

    class _PureRegistry:
        acquire_calls = 0

        def descriptors(self) -> tuple[object, ...]:
            return ()

        async def provider_configuration_snapshot(
            self, _provider_id: str
        ) -> TTSProviderConfigurationSnapshot:
            return TTSProviderConfigurationSnapshot(
                revision=4,
                applied_generation=2,
                applied_config=config,
                staged_generation=None,
                staged_config=None,
            )

        async def acquire(self, _provider_id: str) -> object:
            self.acquire_calls += 1
            raise AssertionError("dependency inspection acquired an adapter")

    registry = _PureRegistry()
    service = tts_generation.TTSService(cast(Any, registry))

    snapshot = await service.audio_cpp_guided_dependency_snapshot(_guided_requirement())

    assert snapshot.state == "mismatch"
    assert registry.acquire_calls == 0


@pytest.mark.asyncio
async def test_guided_dependency_snapshot_reports_present_unknown_recipe_as_missing() -> (
    None
):
    config = _guided_clone_config()
    package = config["guided_packages"][0]
    package["recipe_id"] = "future.valid.recipe"
    package["recipe_revision"] = 1
    requirement = TTSCloneRecipeRequirement(
        recipe_id="future.valid.recipe",
        recipe_revision=1,
        model_id="clone-model",
    )

    class _PureRegistry:
        acquire_calls = 0

        def descriptors(self) -> tuple[object, ...]:
            return ()

        async def provider_configuration_snapshot(
            self, _provider_id: str
        ) -> TTSProviderConfigurationSnapshot:
            return TTSProviderConfigurationSnapshot(
                revision=4,
                applied_generation=2,
                applied_config=config,
                staged_generation=None,
                staged_config=None,
            )

        async def acquire(self, _provider_id: str) -> object:
            self.acquire_calls += 1
            raise AssertionError("dependency inspection acquired an adapter")

    registry = _PureRegistry()
    service = tts_generation.TTSService(cast(Any, registry))

    snapshot = await service.audio_cpp_guided_dependency_snapshot(requirement)

    assert snapshot.state == "missing"
    assert registry.acquire_calls == 0


@pytest.mark.asyncio
async def test_guided_dependency_snapshot_reports_pending_saved_configuration() -> None:
    saved_config = _guided_clone_config()

    class _PureRegistry:
        def descriptors(self) -> tuple[object, ...]:
            return ()

        async def provider_configuration_snapshot(
            self, _provider_id: str
        ) -> TTSProviderConfigurationSnapshot:
            return TTSProviderConfigurationSnapshot(
                revision=4,
                applied_generation=1,
                applied_config={},
                staged_generation=2,
                staged_config=saved_config,
            )

        async def acquire(self, _provider_id: str) -> object:
            raise AssertionError("dependency inspection acquired an adapter")

    service = tts_generation.TTSService(cast(Any, _PureRegistry()))
    service._settings_persisted_provider_generations["audio_cpp"] = 2
    service._settings_persisted_provider_configs["audio_cpp"] = dict(saved_config)
    before = dict(service._settings_persisted_provider_configs["audio_cpp"])

    snapshot = await service.audio_cpp_guided_dependency_snapshot(_guided_requirement())

    assert snapshot.state == "pending"
    assert snapshot.saved_requirement == _guided_requirement()
    assert snapshot.applied_requirement is None
    assert snapshot.pending_configuration is True
    assert service._settings_persisted_provider_configs["audio_cpp"] == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_state"),
    (
        ("applied_exact_saved_exact", "exact"),
        ("applied_absent_saved_exact", "pending"),
        ("applied_absent_saved_absent", "missing"),
        ("applied_absent_saved_drift", "mismatch"),
        ("applied_drift_saved_exact", "pending"),
        ("applied_exact_saved_drift", "exact"),
    ),
)
async def test_guided_dependency_snapshot_applied_saved_precedence_matrix(
    case: str,
    expected_state: str,
) -> None:
    exact_config = _guided_clone_config()
    drift_config = _guided_clone_config()
    drift_config["guided_packages"][0]["projection"]["family"] = "drifted_family"
    if case.startswith("applied_exact"):
        applied_config = exact_config
    elif case.startswith("applied_drift"):
        applied_config = drift_config
    else:
        applied_config = {}
    if case.endswith("saved_exact"):
        saved_config = exact_config
    elif case.endswith("saved_drift"):
        saved_config = drift_config
    else:
        saved_config = {}

    class _PureRegistry:
        acquire_calls = 0

        def descriptors(self) -> tuple[object, ...]:
            return ()

        async def provider_configuration_snapshot(
            self, _provider_id: str
        ) -> TTSProviderConfigurationSnapshot:
            return TTSProviderConfigurationSnapshot(
                revision=4,
                applied_generation=1,
                applied_config=applied_config,
                staged_generation=2,
                staged_config=saved_config,
            )

        async def acquire(self, _provider_id: str) -> object:
            self.acquire_calls += 1
            raise AssertionError("dependency inspection acquired an adapter")

    registry = _PureRegistry()
    service = tts_generation.TTSService(cast(Any, registry))
    service._settings_persisted_provider_generations["audio_cpp"] = 2
    service._settings_persisted_provider_configs["audio_cpp"] = dict(saved_config)

    snapshot = await service.audio_cpp_guided_dependency_snapshot(_guided_requirement())

    assert snapshot.state == expected_state
    assert snapshot.pending_configuration is True
    assert registry.acquire_calls == 0


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
async def test_reference_profile_generation_edit_is_rejected_without_capability_work() -> (
    None
):
    service, repository, tts_service = _service()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(revision=4, reference=_reference().summary),
    )
    changed = TTSProfileDraft(
        display_name=loaded.profile.display_name,
        provider_id="audio_cpp",
        model_id=loaded.profile.model_id,
        voice_id="other-voice",
        response_format="wav",
        speed=1.0,
        options={},
    )

    with pytest.raises(ProfileServiceError) as caught:
        await service.update_profile(loaded, changed)

    _assert_safe_service_error(caught.value, "operation_failed")
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert repository.calls == []


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
            provider_id="openai",
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


def test_unknown_provider_draft_is_unconstructable() -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: provider_id$"
    ):
        TTSProfileDraft(
            display_name="Future",
            provider_id="future_native",
            model_id="model",
            voice_id=None,
            response_format="wav",
            speed=1.0,
            options={},
        )


@pytest.mark.asyncio
async def test_update_profile_accepts_openai_draft_without_native_calls() -> None:
    coordinator = _ArtifactLeaseCoordinator()
    service, repository, tts_service = _service(artifact_lease_coordinator=coordinator)
    repository.coordinator_probe = lambda: coordinator.active
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation, profile=_profile()
    )
    draft = TTSProfileDraft(
        display_name="Narrator",
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="marius",
        response_format="mp3",
        speed=1.25,
        options={},
    )

    await service.update_profile(loaded, draft)

    assert tts_service.capability_calls == []
    assert [name for name, _ in repository.calls] == ["update"]
    assert repository.coordinator_active_at_repository_calls == [True]


@pytest.mark.asyncio
async def test_duplicate_of_legacy_profile_skips_native_capability() -> None:
    repository = _FakeRepository()
    repository.created_profile_id = _DUPLICATE_ID
    service, repository, tts_service = _service(repository=repository)
    source = _profile(
        provider_id="elevenlabs",
        model_id="eleven_multilingual_v2",
        voice_id="21m00Tcm4TlvDq8ikWAM",
        response_format="mp3",
        speed=1.0,
    )
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation, profile=source
    )

    await service.duplicate_profile(loaded, "Narrator copy")

    assert tts_service.capability_calls == []
    assert [name for name, _ in repository.calls] == ["create"]


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
    coordinator = _ArtifactLeaseCoordinator()
    service, repository, tts_service = _service(
        repository=repository,
        tts_service=tts_service,
        artifact_lease_coordinator=coordinator,
    )
    repository.coordinator_probe = lambda: coordinator.active
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
    assert repository.coordinator_active_at_repository_calls == [True]


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
async def test_get_assigned_profile_returns_one_exact_immutable_joined_read() -> None:
    service, repository, tts_service = _service()
    character_ref = _character_ref()
    persisted = AssignedTTSProfileSnapshot(
        assignment=_assignment(character_ref=character_ref),
        profile=_profile(voice_id="voice-a", revision=6),
    )
    repository.get_assignment_result = ProfileStoreResult(
        generation=repository.generation,
        value=persisted,
    )

    loaded = await service.get_assigned_profile(character_ref)

    assert type(loaded) is LoadedCharacterTTSAssignment
    assert loaded.repository_generation == repository.generation
    assert loaded.snapshot == persisted
    assert loaded.snapshot is not persisted
    assert loaded.snapshot is not None
    assert loaded.snapshot.assignment is not persisted.assignment
    assert loaded.snapshot.assignment.character_ref is not character_ref
    assert loaded.snapshot.profile is not persisted.profile
    assert loaded.snapshot.profile.revision == 6
    assert len(repository.calls) == 1
    call_name, forwarded_ref = repository.calls[0]
    assert call_name == "get_assigned_profile"
    assert type(forwarded_ref) is CharacterRef
    assert forwarded_ref == character_ref
    assert forwarded_ref is not character_ref
    assert tts_service.capability_calls == []
    assert tts_service.revision_reads == []
    assert tts_service.revision_decisions == []


@pytest.mark.asyncio
async def test_get_assigned_profile_preserves_exact_unassigned_generation() -> None:
    service, repository, tts_service = _service()

    loaded = await service.get_assigned_profile(_character_ref())

    assert loaded == LoadedCharacterTTSAssignment(
        repository_generation=repository.generation,
        snapshot=None,
    )
    assert [name for name, _value in repository.calls] == ["get_assigned_profile"]
    assert tts_service.capability_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    (
        "snapshot-subclass",
        "other-character",
        "profile-mismatch",
        "hostile-envelope",
    ),
)
async def test_get_assigned_profile_rejects_nonexact_repository_success(
    case: str,
) -> None:
    class SnapshotSubclass(AssignedTTSProfileSnapshot):
        pass

    repository = _FakeRepository()
    character_ref = _character_ref()
    exact = AssignedTTSProfileSnapshot(
        assignment=_assignment(character_ref=character_ref),
        profile=_profile(),
    )
    if case == "snapshot-subclass":
        value: object = SnapshotSubclass(
            assignment=exact.assignment,
            profile=exact.profile,
        )
    elif case == "other-character":
        value = AssignedTTSProfileSnapshot(
            assignment=_assignment(
                character_ref=_character_ref(character_id="different-character"),
            ),
            profile=exact.profile,
        )
    elif case == "profile-mismatch":
        value = _forged_assigned_snapshot(
            exact,
            profile=_profile(profile_id=_DUPLICATE_ID),
        )
    else:
        assert case == "hostile-envelope"
        repository.get_assignment_result = _HostileResult()
        value = None
    if case != "hostile-envelope":
        repository.get_assignment_result = ProfileStoreResult(
            generation=repository.generation,
            value=cast(AssignedTTSProfileSnapshot, value),
        )
    service, repository, tts_service = _service(repository=repository)

    with pytest.raises(ProfileServiceError) as caught:
        await service.get_assigned_profile(character_ref)

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "different-character",
        "credential",
        "example.test",
        "/private/path",
        "submitted text",
    )
    assert [name for name, _value in repository.calls] == ["get_assigned_profile"]
    assert tts_service.capability_calls == []


@pytest.mark.asyncio
async def test_get_assigned_profile_rejects_generation_change_before_publication() -> (
    None
):
    repository = _FakeRepository()
    repository.get_assignment_result = ProfileStoreResult(
        generation=repository.generation,
        value=None,
    )
    repository.advance_generation_after_get_assignment = True
    service, repository, tts_service = _service(repository=repository)

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.get_assigned_profile(_character_ref())

    assert caught.value.code == "stale"
    assert [name for name, _value in repository.calls] == ["get_assigned_profile"]
    assert tts_service.capability_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_type", "expected_code"),
    (
        (
            ProfileRepositoryError("unavailable"),
            ProfileRepositoryError,
            "unavailable",
        ),
        (
            RuntimeError("https://user:credential@example.test/private/path"),
            ProfileServiceError,
            "operation_failed",
        ),
    ),
)
async def test_get_assigned_profile_maps_repository_failures_safely(
    error: BaseException,
    expected_type: type[BaseException],
    expected_code: str,
) -> None:
    repository = _FakeRepository()
    repository.get_assignment_error = error
    service, repository, tts_service = _service(repository=repository)

    with pytest.raises(expected_type) as caught:
        await service.get_assigned_profile(_character_ref())

    assert getattr(caught.value, "code", None) == expected_code
    assert "credential" not in str(caught.value)
    assert "example.test" not in str(caught.value)
    assert [name for name, _value in repository.calls] == ["get_assigned_profile"]
    assert tts_service.capability_calls == []


@pytest.mark.asyncio
async def test_get_profile_returns_one_exact_immutable_loaded_profile() -> None:
    service, repository, tts_service = _service()
    persisted = _profile(voice_id="voice-a", revision=6)
    repository.get_profile_result = ProfileStoreResult(
        generation=repository.generation,
        value=persisted,
    )

    loaded = await service.get_profile(_PROFILE_ID)

    assert type(loaded) is LoadedTTSProfile
    assert loaded.repository_generation == repository.generation
    assert loaded.profile == persisted
    assert loaded.profile is not persisted
    assert loaded.profile.revision == 6
    assert len(repository.calls) == 1
    call_name, forwarded_id = repository.calls[0]
    assert call_name == "get_profile"
    assert type(forwarded_id) is UUID
    assert forwarded_id == _PROFILE_ID
    assert forwarded_id is not _PROFILE_ID
    assert tts_service.capability_calls == []


@pytest.mark.asyncio
async def test_get_profile_preserves_exact_reference_summary() -> None:
    service, repository, _tts_service = _service()
    reference = _reference()
    persisted = _profile(reference=reference.summary)
    repository.get_profile_result = ProfileStoreResult(
        generation=repository.generation,
        value=persisted,
    )

    loaded = await service.get_profile(_PROFILE_ID)

    assert loaded.profile.reference == reference.summary
    assert loaded.profile.reference is not reference.summary


@pytest.mark.asyncio
async def test_get_reference_returns_one_exact_private_snapshot_under_fences() -> None:
    service, repository, tts_service = _service()
    persisted = _reference()
    repository.get_reference_result = ProfileStoreResult(
        generation=repository.generation,
        value=persisted,
    )

    loaded = await service.get_reference(
        _PROFILE_ID,
        expected_revision=6,
        expected_generation=repository.generation,
    )

    assert loaded == persisted
    assert loaded is not persisted
    assert loaded.summary is not persisted.summary
    assert repository.calls == [
        ("get_reference", (_PROFILE_ID, 6, repository.generation))
    ]
    assert "Private transcript" not in repr(loaded)
    assert tts_service.capability_calls == []


@pytest.mark.asyncio
async def test_get_reference_rejects_repository_generation_change() -> None:
    service, repository, _tts_service = _service()
    expected_generation = repository.generation
    repository.generation += 1
    repository.get_reference_result = ProfileStoreResult(
        generation=repository.generation,
        value=_reference(),
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.get_reference(
            _PROFILE_ID,
            expected_revision=6,
            expected_generation=expected_generation,
        )

    assert caught.value.code == "stale"


@pytest.mark.asyncio
async def test_get_reference_rejects_malformed_private_result_without_detail() -> None:
    service, repository, _tts_service = _service()
    repository.get_reference_result = ProfileStoreResult(
        generation=repository.generation,
        value=object(),
    )

    with pytest.raises(ProfileServiceError) as caught:
        await service.get_reference(
            _PROFILE_ID,
            expected_revision=6,
            expected_generation=repository.generation,
        )

    assert caught.value.code == "operation_failed"
    assert "object" not in str(caught.value)


@pytest.mark.asyncio
async def test_get_profile_rejects_nonuuid_profile_id() -> None:
    service, repository, tts_service = _service()

    with pytest.raises(ProfileValidationError) as caught:
        await service.get_profile(str(_PROFILE_ID))  # type: ignore[arg-type]

    assert caught.value.code == "profile_id"
    assert repository.calls == []
    assert tts_service.capability_calls == []


@pytest.mark.asyncio
async def test_get_profile_rejects_generation_change_before_publication() -> None:
    repository = _FakeRepository()
    repository.get_profile_result = ProfileStoreResult(
        generation=repository.generation,
        value=_profile(),
    )
    repository.advance_generation_after_get_profile = True
    service, repository, tts_service = _service(repository=repository)

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.get_profile(_PROFILE_ID)

    assert caught.value.code == "stale"
    assert [name for name, _value in repository.calls] == ["get_profile"]
    assert tts_service.capability_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ("wrong-id", "hostile-envelope"),
)
async def test_get_profile_rejects_nonexact_repository_success(case: str) -> None:
    repository = _FakeRepository()
    if case == "wrong-id":
        repository.get_profile_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_profile(_profile(), profile_id=_DUPLICATE_ID),
        )
    else:
        assert case == "hostile-envelope"
        repository.get_profile_result = _HostileResult()
    service, repository, tts_service = _service(repository=repository)

    with pytest.raises(ProfileServiceError) as caught:
        await service.get_profile(_PROFILE_ID)

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "example.test",
        "/private/path",
        "submitted text",
    )
    assert [name for name, _value in repository.calls] == ["get_profile"]
    assert tts_service.capability_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_type", "expected_code"),
    (
        (
            ProfileRepositoryError("missing"),
            ProfileRepositoryError,
            "missing",
        ),
        (
            RuntimeError("https://user:credential@example.test/private/path"),
            ProfileServiceError,
            "operation_failed",
        ),
    ),
)
async def test_get_profile_maps_repository_failures_safely(
    error: BaseException,
    expected_type: type[BaseException],
    expected_code: str,
) -> None:
    repository = _FakeRepository()
    repository.get_profile_error = error
    service, repository, tts_service = _service(repository=repository)

    with pytest.raises(expected_type) as caught:
        await service.get_profile(_PROFILE_ID)

    assert getattr(caught.value, "code", None) == expected_code
    assert "credential" not in str(caught.value)
    assert "example.test" not in str(caught.value)
    assert [name for name, _value in repository.calls] == ["get_profile"]
    assert tts_service.capability_calls == []


@pytest.mark.parametrize(
    ("method_name", "required_sections"),
    (
        ("get_assigned_profile", ("Args:", "Returns:", "Raises:")),
        ("get_profile", ("Args:", "Returns:", "Raises:")),
        ("set_assignment", ("Args:", "Returns:", "Raises:")),
        ("detach_assignment", ("Args:", "Raises:")),
    ),
)
def test_assignment_mutation_methods_document_their_public_contract(
    method_name: str,
    required_sections: tuple[str, ...],
) -> None:
    method = getattr(TTSProfileService, method_name)
    docstring = inspect.getdoc(method)

    assert docstring is not None
    for parameter_name in inspect.signature(method).parameters:
        if parameter_name != "self":
            assert f"{parameter_name}:" in docstring
    for section in required_sections:
        assert section in docstring


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected_current_profile_id",
    (None, _DUPLICATE_ID),
    ids=("unassigned", "replacement"),
)
async def test_set_assignment_uses_fresh_loaded_authority_and_exact_expected_state(
    expected_current_profile_id: UUID | None,
) -> None:
    voice = TTSVoiceDiscoveryResult(
        provider_id="audio_cpp",
        model_id="model-a",
        catalog_revision=9,
        voices=("voice-a",),
        state="complete",
    )
    repository = _FakeRepository()
    tts_service = _FakeTTSService(
        _capability_snapshot(
            models=(_model("model-a"),),
            voice_results={"model-a": voice},
        )
    )
    coordinator = _ArtifactLeaseCoordinator()
    service, repository, tts_service = _service(
        repository=repository,
        tts_service=tts_service,
        artifact_lease_coordinator=coordinator,
    )
    repository.coordinator_probe = lambda: coordinator.active
    character_ref = _character_ref()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(voice_id="voice-a", revision=4),
    )
    persisted = _assignment(
        character_ref=character_ref,
        profile_id=loaded.profile.profile_id,
    )
    repository.set_result = ProfileStoreResult(
        generation=loaded.repository_generation,
        value=persisted,
    )
    page = TTSProfilePageSnapshot(
        repository_generation=loaded.repository_generation,
        profiles=(loaded.profile,),
        total=1,
    )
    observed = await service.observe_availability(page)
    assert observed.profiles[0].state == "available"

    expected_current = (
        None
        if expected_current_profile_id is None
        else _assignment(
            character_ref=character_ref,
            profile_id=expected_current_profile_id,
        )
    )

    assigned = await service.set_assignment(
        character_ref,
        loaded,
        expected_current,
    )

    assert assigned == persisted
    assert type(assigned) is CharacterTTSAssignment
    assert assigned is not persisted
    assert assigned.character_ref is not persisted.character_ref
    assert len(repository.calls) == 1
    call_name, call_value = repository.calls[0]
    assert call_name == "set_assignment"
    (
        forwarded_ref,
        forwarded_profile_id,
        forwarded_generation,
        forwarded_revision,
        forwarded_current_profile_id,
        generation_at_call,
    ) = call_value  # type: ignore[misc]
    assert type(forwarded_ref) is CharacterRef
    assert forwarded_ref == character_ref
    assert forwarded_ref is not character_ref
    assert forwarded_profile_id == loaded.profile.profile_id
    assert forwarded_generation == loaded.repository_generation
    assert forwarded_revision == loaded.profile.revision
    assert forwarded_current_profile_id == expected_current_profile_id
    assert generation_at_call == loaded.repository_generation
    assert repository.coordinator_active_at_repository_calls == [True]

    assert tts_service.capability_calls == [
        ("audio_cpp", ("model-a",)),
        ("audio_cpp", ("model-a",)),
    ]
    assert tts_service.revision_decisions == [("audio_cpp", 3)] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    (
        "character-ref-subclass",
        "character-ref-exploding-field",
        "character-ref-impostor",
        "character-ref-manufactured-equality",
        "loaded-subclass",
        "loaded-malformed-profile",
        "loaded-impostor",
        "expected-current-subclass",
        "expected-current-malformed-profile-id",
        "expected-current-impostor",
        "expected-current-other-character",
        "expected-current-manufactured-equality",
    ),
)
async def test_set_assignment_rejects_nonexact_domain_inputs_before_work(
    case: str,
) -> None:
    class CharacterRefSubclass(CharacterRef):
        pass

    class LoadedProfileSubclass(LoadedTTSProfile):
        pass

    class AssignmentSubclass(CharacterTTSAssignment):
        pass

    service, repository, tts_service = _service()
    character_ref = _character_ref()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    expected_current = _assignment(
        character_ref=character_ref,
        profile_id=_DUPLICATE_ID,
    )
    secret = "https://user:credential@example.test/private/path"
    candidate_ref: object = character_ref
    candidate_loaded: object = loaded
    candidate_current: object = expected_current

    if case == "character-ref-subclass":
        candidate_ref = CharacterRefSubclass(
            source="server",
            authority_id="server-user-v1:authority",
            character_id="character-a",
        )
    elif case == "character-ref-exploding-field":
        candidate_ref = _forged_character_ref(
            character_ref,
            authority_id=_ExplodingStr(secret),
        )
    elif case == "character-ref-impostor":
        candidate_ref = object()
    elif case == "character-ref-manufactured-equality":
        candidate_ref = _manufactured_equal_character_ref(character_ref)
    elif case == "loaded-subclass":
        candidate_loaded = LoadedProfileSubclass(
            repository_generation=repository.generation,
            profile=_profile(),
        )
    elif case == "loaded-malformed-profile":
        candidate_loaded = _forged_loaded_profile(
            _forged_profile(
                _profile(),
                model_id=_ExplodingStr(secret),
            ),
            repository_generation=repository.generation,
        )
    elif case == "loaded-impostor":
        candidate_loaded = object()
    elif case == "expected-current-subclass":
        candidate_current = AssignmentSubclass(
            character_ref=character_ref,
            profile_id=_DUPLICATE_ID,
        )
    elif case == "expected-current-malformed-profile-id":
        candidate_current = _forged_assignment(expected_current, profile_id=secret)
    elif case == "expected-current-impostor":
        candidate_current = object()
    elif case == "expected-current-other-character":
        candidate_current = _assignment(
            character_ref=_character_ref(character_id="different-character"),
            profile_id=_DUPLICATE_ID,
        )
    else:
        assert case == "expected-current-manufactured-equality"
        candidate_current = _manufactured_equal_assignment(expected_current)

    with pytest.raises(ProfileValidationError) as caught:
        await service.set_assignment(
            cast(CharacterRef, candidate_ref),
            cast(LoadedTTSProfile, candidate_loaded),
            cast(CharacterTTSAssignment, candidate_current),
        )

    expected_code = "profiles" if case.startswith("loaded-") else "assignment"
    assert caught.value.code == expected_code
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert repository.calls == []
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot", "stale_configuration", "expected_code"),
    (
        (_capability_snapshot(models=()), False, "profile_unavailable"),
        (
            _capability_snapshot(
                state="unverified",
                models=(_model("model-a"),),
            ),
            False,
            "profile_unverified",
        ),
        (
            TTSNativeCapabilitySnapshot(
                provider_id="audio_cpp",
                configuration_revision=3,
                state="unverified",
                catalog=None,
                voice_results={},
            ),
            False,
            "profile_unverified",
        ),
        (_HostileResult(), False, "operation_failed"),
        (
            _capability_snapshot(models=(_model("model-a"),)),
            True,
            "stale_configuration",
        ),
    ),
    ids=(
        "model-unavailable",
        "catalog-unverified",
        "missing-catalog-authority",
        "malformed-capability-success",
        "stale-configuration",
    ),
)
async def test_set_assignment_rejects_non_authoritative_capability_outcomes(
    snapshot: object,
    stale_configuration: bool,
    expected_code: str,
) -> None:
    tts_service = _FakeTTSService(
        cast(TTSNativeCapabilitySnapshot, snapshot),
    )
    tts_service.stale_decision = stale_configuration
    service, repository, tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(revision=5),
    )

    with pytest.raises(ProfileServiceError) as caught:
        await service.set_assignment(_character_ref(), loaded, None)

    _assert_safe_service_error(
        caught.value,
        expected_code,
        "credential",
        "example.test",
        "/private/path",
        "submitted text",
    )
    assert tts_service.capability_calls == [("audio_cpp", ())]
    assert repository.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("race", "expected_code"),
    (
        ("configuration", "stale_configuration"),
        ("catalog", "profile_unverified"),
    ),
)
async def test_set_assignment_fails_closed_at_capability_authority_barriers(
    race: str,
    expected_code: str,
) -> None:
    boundary = _AsyncBoundary()
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    if race == "configuration":
        tts_service.revision_boundary = boundary
    else:
        tts_service.capability_boundary = boundary
    service, repository, tts_service = _service(tts_service=tts_service)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    operation = await _start_at_boundary(
        service.set_assignment(_character_ref(), loaded, None),
        boundary,
    )
    try:
        if race == "configuration":
            tts_service.revision += 1
        else:
            tts_service.snapshot = _capability_snapshot(
                state="unverified",
                models=(_model("model-a"),),
                catalog_revision=10,
            )
        boundary.release.set()

        with pytest.raises(ProfileServiceError) as caught:
            await operation
    finally:
        await _settle_boundary_task(boundary, operation)

    _assert_safe_service_error(caught.value, expected_code)
    assert boundary.settled.is_set()
    assert repository.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("race_stage", ("before", "capability", "repository"))
async def test_set_assignment_rechecks_generation_across_lifecycle_barriers(
    race_stage: str,
) -> None:
    boundary = _AsyncBoundary()
    repository = _FakeRepository()
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    character_ref = _character_ref()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    repository.set_result = ProfileStoreResult(
        generation=loaded.repository_generation,
        value=_assignment(
            character_ref=character_ref,
            profile_id=loaded.profile.profile_id,
        ),
    )
    if race_stage == "capability":
        tts_service.capability_boundary = boundary
    elif race_stage == "repository":
        repository.set_boundary = boundary
    service, repository, tts_service = _service(
        repository=repository,
        tts_service=tts_service,
    )
    if race_stage == "before":
        repository.generation += 1

        with pytest.raises(ProfileRepositoryError) as caught:
            await service.set_assignment(character_ref, loaded, None)

        assert caught.value.code == "stale"
        assert repository.calls == []
        assert tts_service.capability_calls == []
        return

    operation = await _start_at_boundary(
        service.set_assignment(character_ref, loaded, None),
        boundary,
    )
    try:
        repository.generation += 1
        boundary.release.set()

        with pytest.raises(ProfileRepositoryError) as caught:
            await operation
    finally:
        await _settle_boundary_task(boundary, operation)

    assert type(caught.value) is ProfileRepositoryError
    assert caught.value.code == "stale"
    assert boundary.settled.is_set()
    expected_calls = [] if race_stage == "capability" else ["set_assignment"]
    assert [name for name, _value in repository.calls] == expected_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", ("conflict", "stale"))
async def test_set_assignment_preserves_bounded_repository_race_errors(
    error_code: str,
) -> None:
    boundary = _AsyncBoundary()
    repository = _FakeRepository()
    repository.set_boundary = boundary
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    service, repository, _tts_service = _service(
        repository=repository,
        tts_service=tts_service,
    )
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    operation = await _start_at_boundary(
        service.set_assignment(_character_ref(), loaded, None),
        boundary,
    )
    try:
        repository.set_error = ProfileRepositoryError(error_code)
        boundary.release.set()

        with pytest.raises(ProfileRepositoryError) as caught:
            await operation
    finally:
        await _settle_boundary_task(boundary, operation)

    assert type(caught.value) is ProfileRepositoryError
    assert caught.value.code == error_code
    assert str(caught.value) == f"TTS profile repository failed: {error_code}"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert boundary.settled.is_set()
    assert [name for name, _value in repository.calls] == ["set_assignment"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result_case",
    (
        "hostile-envelope",
        "wrong-generation",
        "assignment-subclass",
        "assignment-exploding-nested-ref",
        "assignment-other-character",
        "assignment-other-profile",
        "assignment-manufactured-equality",
    ),
)
async def test_set_assignment_rejects_nonexact_repository_success(
    result_case: str,
) -> None:
    class AssignmentSubclass(CharacterTTSAssignment):
        pass

    repository = _FakeRepository()
    character_ref = _character_ref()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(),
    )
    persisted: object = _assignment(
        character_ref=character_ref,
        profile_id=loaded.profile.profile_id,
    )
    if result_case == "hostile-envelope":
        invalid_result: object = _HostileResult()
    elif result_case == "wrong-generation":
        invalid_result = ProfileStoreResult(
            generation=repository.generation + 1,
            value=persisted,
        )
    elif result_case == "assignment-subclass":
        invalid_result = ProfileStoreResult(
            generation=repository.generation,
            value=AssignmentSubclass(
                character_ref=character_ref,
                profile_id=loaded.profile.profile_id,
            ),
        )
    elif result_case == "assignment-exploding-nested-ref":
        invalid_result = ProfileStoreResult(
            generation=repository.generation,
            value=_forged_assignment(
                cast(CharacterTTSAssignment, persisted),
                character_ref=_forged_character_ref(
                    character_ref,
                    authority_id=_ExplodingStr(
                        "https://user:credential@example.test/private/path"
                    ),
                ),
            ),
        )
    elif result_case == "assignment-other-character":
        invalid_result = ProfileStoreResult(
            generation=repository.generation,
            value=_assignment(
                character_ref=_character_ref(character_id="different-character"),
                profile_id=loaded.profile.profile_id,
            ),
        )
    elif result_case == "assignment-other-profile":
        invalid_result = ProfileStoreResult(
            generation=repository.generation,
            value=_assignment(
                character_ref=character_ref,
                profile_id=_DUPLICATE_ID,
            ),
        )
    else:
        assert result_case == "assignment-manufactured-equality"
        hostile_assignment = _manufactured_equal_assignment(
            cast(CharacterTTSAssignment, persisted),
        )
        invalid_result = ProfileStoreResult(
            generation=repository.generation,
            value=hostile_assignment,
        )
    service, repository, _tts_service = _service(
        repository=repository,
        tts_service=_FakeTTSService(_capability_snapshot(models=(_model("model-a"),))),
    )
    repository.set_result = invalid_result

    with pytest.raises(ProfileServiceError) as caught:
        await service.set_assignment(character_ref, loaded, None)

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "example.test",
        "/private/path",
        "submitted text",
        character_ref.authority_id,
        character_ref.character_id,
        "different-authority",
        "different-character",
    )
    assert [name for name, _value in repository.calls] == ["set_assignment"]


@pytest.mark.asyncio
async def test_detach_assignment_forwards_exact_state_without_capability_work() -> None:
    coordinator = _ArtifactLeaseCoordinator()
    service, repository, tts_service = _service(artifact_lease_coordinator=coordinator)
    repository.coordinator_probe = lambda: coordinator.active
    assignment = _assignment(profile_id=_DUPLICATE_ID)

    result = await service.detach_assignment(
        assignment,
        repository.generation,
    )

    assert result is None
    assert len(repository.calls) == 2
    assert repository.calls[0][0] == "get_profile"
    call_name, call_value = repository.calls[1]
    assert call_name == "remove_assignment"
    forwarded_ref, forwarded_generation, forwarded_profile_id, generation_at_call = (
        call_value  # type: ignore[misc]
    )
    assert type(forwarded_ref) is CharacterRef
    assert forwarded_ref == assignment.character_ref
    assert forwarded_ref is not assignment.character_ref
    assert forwarded_generation == repository.generation
    assert forwarded_profile_id == assignment.profile_id
    assert generation_at_call == repository.generation
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []
    assert tts_service.revision_reads == []
    assert repository.coordinator_active_at_repository_calls == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    (
        "assignment-subclass",
        "assignment-exploding-nested-ref",
        "assignment-impostor",
        "generation-bool",
        "generation-negative",
        "generation-subclass",
        "generation-stale",
    ),
)
async def test_detach_assignment_rejects_nonexact_or_stale_caller_state(
    case: str,
) -> None:
    class AssignmentSubclass(CharacterTTSAssignment):
        pass

    class GenerationSubclass(int):
        pass

    service, repository, tts_service = _service()
    assignment = _assignment()
    secret = "https://user:credential@example.test/private/path"
    candidate: object = assignment
    generation: object = repository.generation
    if case == "assignment-subclass":
        candidate = AssignmentSubclass(
            character_ref=assignment.character_ref,
            profile_id=assignment.profile_id,
        )
    elif case == "assignment-exploding-nested-ref":
        candidate = _forged_assignment(
            assignment,
            character_ref=_forged_character_ref(
                assignment.character_ref,
                character_id=_ExplodingStr(secret),
            ),
        )
    elif case == "assignment-impostor":
        candidate = object()
    elif case == "generation-bool":
        generation = True
    elif case == "generation-negative":
        generation = -1
    elif case == "generation-subclass":
        generation = GenerationSubclass(repository.generation)
    else:
        assert case == "generation-stale"
        generation = repository.generation - 1

    expected_error = (
        ProfileRepositoryError if case == "generation-stale" else ProfileValidationError
    )
    with pytest.raises(expected_error) as caught:
        await service.detach_assignment(
            cast(CharacterTTSAssignment, candidate),
            cast(int, generation),
        )

    expected_code = (
        "stale"
        if case == "generation-stale"
        else "generation"
        if case.startswith("generation-")
        else "assignment"
    )
    assert caught.value.code == expected_code  # type: ignore[attr-defined]
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert repository.calls == []
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_code",
    ("conflict", "stale"),
)
async def test_detach_assignment_preserves_bounded_repository_errors(
    error_code: str,
) -> None:
    repository = _FakeRepository()
    service, repository, tts_service = _service(repository=repository)
    repository.remove_error = ProfileRepositoryError(error_code)

    with pytest.raises(ProfileRepositoryError) as caught:
        await service.detach_assignment(
            _assignment(),
            repository.generation,
        )

    assert type(caught.value) is ProfileRepositoryError
    assert caught.value.code == error_code
    assert str(caught.value) == f"TTS profile repository failed: {error_code}"
    assert [name for name, _value in repository.calls] == [
        "get_profile",
        "remove_assignment",
    ]
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result_case",
    ("hostile-envelope", "wrong-generation", "unexpected-value"),
)
async def test_detach_assignment_rejects_nonexact_repository_success(
    result_case: str,
) -> None:
    repository = _FakeRepository()
    service, repository, tts_service = _service(repository=repository)
    assignment = _assignment()
    if result_case == "hostile-envelope":
        invalid_result: object = _HostileResult()
    elif result_case == "wrong-generation":
        invalid_result = ProfileStoreResult(
            generation=repository.generation + 1,
            value=None,
        )
    else:
        assert result_case == "unexpected-value"
        invalid_result = ProfileStoreResult(
            generation=repository.generation,
            value=_HostileResult(),
        )
    repository.remove_result = invalid_result

    with pytest.raises(ProfileServiceError) as caught:
        await service.detach_assignment(
            assignment,
            repository.generation,
        )

    _assert_safe_service_error(
        caught.value,
        "operation_failed",
        "credential",
        "example.test",
        "/private/path",
        "submitted text",
        assignment.character_ref.authority_id,
        assignment.character_ref.character_id,
    )
    assert [name for name, _value in repository.calls] == [
        "get_profile",
        "remove_assignment",
    ]
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == []


@pytest.mark.asyncio
async def test_detach_assignment_rechecks_generation_after_repository_result() -> None:
    boundary = _AsyncBoundary()
    repository = _FakeRepository()
    repository.remove_boundary = boundary
    repository.remove_result = ProfileStoreResult(
        generation=repository.generation,
        value=None,
    )
    service, repository, tts_service = _service(repository=repository)
    operation = await _start_at_boundary(
        service.detach_assignment(
            _assignment(),
            repository.generation,
        ),
        boundary,
    )
    try:
        repository.generation += 1
        boundary.release.set()

        with pytest.raises(ProfileRepositoryError) as caught:
            await operation
    finally:
        await _settle_boundary_task(boundary, operation)

    assert caught.value.code == "stale"
    assert boundary.settled.is_set()
    assert [name for name, _value in repository.calls] == [
        "get_profile",
        "remove_assignment",
    ]
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
    coordinator = _ArtifactLeaseCoordinator()
    service, repository, tts_service = _service(
        repository=repository,
        artifact_lease_coordinator=coordinator,
    )
    repository.coordinator_probe = lambda: coordinator.active
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
    await asyncio.sleep(0)
    assert operation.done() is False
    assert coordinator.active is True
    boundary.release.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await operation

    assert caught.value.args == (cancellation_identity,)
    assert caught.value.args[0] is cancellation_identity
    assert boundary.settled.is_set()
    assert boundary.release.is_set()
    assert operation.done()
    assert operation not in asyncio.all_tasks()
    assert set(asyncio.all_tasks()) == existing_tasks
    assert tts_service.capability_calls == []
    assert tts_service.revision_decisions == [("audio_cpp", 3)]
    assert [name for name, _value in repository.calls] == ["create"]
    assert repository.coordinator_active_at_repository_calls == [True]


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dependency_state", "reason", "action", "display"),
    (
        ("exact", "none", "none", None),
        (
            "missing",
            "recipe_missing",
            "open_audio_cpp_settings",
            "Needs compatible model",
        ),
        (
            "mismatch",
            "recipe_mismatch",
            "open_audio_cpp_settings",
            "Needs compatible model",
        ),
        (
            "pending",
            "recipe_pending_apply",
            "open_speech_lab_apply",
            "Compatible model saved; apply settings",
        ),
    ),
)
async def test_reference_availability_projects_exact_dependency_truth(
    dependency_state: str,
    reason: str,
    action: str,
    display: str | None,
) -> None:
    requirement = _guided_requirement(model_id="model-a")
    reference = _reference_with_requirement(requirement)
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    tts_service.dependency_snapshots[requirement] = (
        tts_generation.AudioCppGuidedDependencySnapshot(
            state=dependency_state,  # type: ignore[arg-type]
            provider_configuration_revision=tts_service.revision,
            saved_generation=2 if dependency_state == "pending" else 1,
            applied_generation=1,
            pending_configuration=dependency_state == "pending",
            saved_requirement=(
                requirement if dependency_state in {"exact", "pending"} else None
            ),
            applied_requirement=(requirement if dependency_state == "exact" else None),
        )
    )
    service, repository, _ = _service(tts_service=tts_service)

    observed = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(
                _profile(
                    model_id="model-a",
                    reference=reference.summary,
                ),
            ),
            total=1,
        )
    )

    availability = observed.profiles[0]
    assert availability.dependency.reason == reason
    assert availability.dependency.action == action
    assert availability.dependency.display == display
    assert availability.dependency.advisory == "none"
    assert availability.state == (
        "available" if dependency_state == "exact" else "unavailable"
    )
    assert tts_service.dependency_calls == [requirement]


@pytest.mark.asyncio
async def test_reference_availability_bounds_invalid_dependency_evidence() -> None:
    requirement = _guided_requirement(model_id="model-a")
    reference = _reference_with_requirement(requirement)
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    tts_service.dependency_snapshots[requirement] = cast(
        Any,
        object.__new__(tts_generation.AudioCppGuidedDependencySnapshot),
    )
    service, repository, _ = _service(tts_service=tts_service)

    with pytest.raises(ProfileServiceError) as caught:
        await service.observe_availability(
            TTSProfilePageSnapshot(
                repository_generation=repository.generation,
                profiles=(
                    _profile(
                        model_id="model-a",
                        reference=reference.summary,
                    ),
                ),
                total=1,
            )
        )

    _assert_safe_service_error(caught.value, "operation_failed")
    assert tts_service.dependency_calls == [requirement]


@pytest.mark.asyncio
async def test_migrated_reference_keeps_provenance_advisory_beside_provider_blocker() -> (
    None
):
    reference = _reference_with_requirement(None)
    tts_service = _FakeTTSService(
        _capability_snapshot(
            models=(_model("model-a"),),
            health_state="not_configured",
        )
    )
    service, repository, _ = _service(tts_service=tts_service)

    observed = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(_profile(reference=reference.summary),),
            total=1,
        )
    )

    availability = observed.profiles[0]
    assert availability.state == "unavailable"
    assert availability.recovery_action == "edit"
    assert availability.dependency.reason == "none"
    assert availability.dependency.advisory == "recipe_provenance_unavailable"
    assert availability.dependency.advisory_display == ("Recipe provenance unavailable")
    assert availability.dependency.advisory_action == "generate_new_profile"
    assert tts_service.dependency_calls == []


@pytest.mark.asyncio
async def test_provider_blocker_precedes_recipe_blocker_but_keeps_advisory() -> None:
    requirement = _guided_requirement(model_id="model-a")
    reference = _reference_with_requirement(requirement)
    tts_service = _FakeTTSService(
        _capability_snapshot(
            models=(_model("model-a"),),
            health_state="not_configured",
        )
    )
    tts_service.dependency_snapshots[requirement] = (
        tts_generation.AudioCppGuidedDependencySnapshot(
            state="missing",
            provider_configuration_revision=tts_service.revision,
            saved_generation=1,
            applied_generation=1,
            pending_configuration=False,
            saved_requirement=None,
            applied_requirement=None,
        )
    )
    service, repository, _ = _service(tts_service=tts_service)

    observed = await service.observe_availability(
        TTSProfilePageSnapshot(
            repository_generation=repository.generation,
            profiles=(_profile(reference=reference.summary),),
            total=1,
        )
    )

    availability = observed.profiles[0]
    assert availability.state == "unavailable"
    assert availability.recovery_action == "edit"
    assert availability.dependency.reason == "none"
    assert availability.dependency.advisory == "none"


def test_preview_preset_copies_only_persisted_selection_and_availability() -> None:
    service, repository, tts_service = _service()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(
            provider_id="openai",
            model_id="opaque-model",
            voice_id="opaque-voice",
            response_format="flac",
            speed=2.0,
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


def test_reference_profile_preview_carries_only_exact_repository_identity() -> None:
    service, repository, tts_service = _service()
    reference = _reference()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation,
        profile=_profile(reference=reference.summary, revision=4),
    )
    availability = TTSProfileAvailability(
        profile_id=loaded.profile.profile_id,
        state="available",
        recovery_action="none",
    )

    preset = service.preview_preset(loaded, availability)

    assert preset.profile_id == loaded.profile.profile_id
    assert preset.repository_generation == loaded.repository_generation
    assert preset.profile_revision == loaded.profile.revision
    assert not hasattr(preset, "reference")
    assert not hasattr(preset, "wav_bytes")
    assert not hasattr(preset, "reference_text")
    assert not hasattr(preset, "source_path")
    rendered = repr(preset)
    assert reference.reference_text not in rendered
    assert reference.sha256 not in rendered
    assert reference.wav_bytes.decode() not in rendered
    assert repository.calls == []
    assert tts_service.capability_calls == []


def test_reference_preview_identity_is_all_or_none_and_exactly_typed() -> None:
    values: dict[str, object] = {
        "provider_id": "audio_cpp",
        "model_id": "model-a",
        "voice_id": None,
        "response_format": "wav",
        "speed": 1.0,
        "options": {},
        "availability": "available",
    }

    with pytest.raises(ValueError, match="preview identity"):
        TTSPlaygroundSelectionPreset(
            **values,  # type: ignore[arg-type]
            profile_id=_PROFILE_ID,
        )
    with pytest.raises(TypeError, match="repository_generation"):
        TTSPlaygroundSelectionPreset(
            **values,  # type: ignore[arg-type]
            profile_id=_PROFILE_ID,
            repository_generation=True,  # type: ignore[arg-type]
            profile_revision=1,
        )


def test_preview_preset_forces_unsupported_profile_unavailable_before_enrichment() -> (
    None
):
    service, repository, tts_service = _service()
    # Every recognized provider is now structurally supported at
    # construction (Task 1), so the only way a loaded profile can still fail
    # the allowlist is to forge one referencing an unrecognized provider,
    # bypassing both `TTSGenerationProfile.__post_init__` and
    # `LoadedTTSProfile.__post_init__`. `preview_preset` fails closed on it
    # (via `_validate_loaded`'s canonicalization) rather than trusting the
    # (falsely optimistic) passed-in `pending` state.
    loaded = _forged_loaded_profile(
        _forged_profile(
            _profile(model_id="tts-1", voice_id="alloy"),
            provider_id="unrecognized_future_provider",
        ),
        repository_generation=repository.generation,
    )
    pending = TTSProfileAvailability(
        profile_id=loaded.profile.profile_id,
        state="unverified",
        recovery_action="refresh",
    )

    with pytest.raises(ProfileValidationError) as caught:
        service.preview_preset(loaded, pending)

    assert caught.value.code == "profiles"
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


@pytest.mark.asyncio
async def test_portable_observation_reports_unavailable_without_writing() -> None:
    repository = _FakeRepository()
    tts_service = _FakeTTSService(
        _capability_snapshot(models=(), health_state="unavailable")
    )
    service = TTSProfileService(repository, tts_service)

    observation = await service.observe_portable_profile(_portable_profile())

    assert observation.availability == "unavailable"
    assert observation.repository_generation == repository.generation
    assert repository.calls == []
    assert tts_service.capability_calls == [("audio_cpp", ())]


@pytest.mark.asyncio
async def test_portable_observation_of_legacy_provider_skips_native_capability() -> (
    None
):
    repository = _FakeRepository()
    tts_service = _FakeTTSService()
    service = TTSProfileService(repository, tts_service)
    portable = _portable_profile(
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        response_format="mp3",
    )

    observation = await service.observe_portable_profile(portable)

    assert observation.availability == "unverified"
    assert observation.repository_generation == repository.generation
    assert repository.calls == []
    assert tts_service.capability_calls == []


def test_portable_import_plan_rejects_candidate_with_different_generation() -> None:
    portable = _portable_profile()
    observation = PortableProfileAvailabilityObservation(
        repository_generation=7,
        configuration_revision=3,
        profile=portable,
        availability="available",
    )
    mismatched = PortableTTSProfile(
        profile_id=_PORTABLE_COPY_ID,
        draft=TTSProfileDraft(
            display_name="Imported voice copy",
            provider_id="audio_cpp",
            model_id="another-model",
            voice_id=portable.draft.voice_id,
            response_format=portable.draft.response_format,
            speed=portable.draft.speed,
            options=portable.draft.options,
        ),
    )

    with pytest.raises(ProfileValidationError) as caught:
        PortableProfileImportPlan(
            observation=observation,
            allowed_choices=("copy",),
            reuse_profile=None,
            copy_candidate=mismatched,
        )

    assert caught.value.code == "profiles"


@pytest.mark.parametrize(
    (
        "collisions",
        "choices",
        "reuse_id",
        "copy_keeps_id",
        "copy_keeps_name",
    ),
    [
        (TTSProfileCollisionSnapshot(None, None), ("create",), None, True, True),
        (
            TTSProfileCollisionSnapshot(
                _profile(display_name="UUID match"),
                None,
            ),
            ("reuse", "copy"),
            _PROFILE_ID,
            False,
            True,
        ),
        (
            TTSProfileCollisionSnapshot(
                _profile(display_name="UUID match", model_id="other-model"),
                None,
            ),
            ("copy",),
            None,
            False,
            True,
        ),
        (
            TTSProfileCollisionSnapshot(
                None,
                _profile(profile_id=_DUPLICATE_ID, display_name="Imported voice"),
            ),
            ("reuse", "copy"),
            _DUPLICATE_ID,
            True,
            False,
        ),
        (
            TTSProfileCollisionSnapshot(
                None,
                _profile(
                    profile_id=_DUPLICATE_ID,
                    display_name="Imported voice",
                    model_id="other-model",
                ),
            ),
            ("copy",),
            None,
            True,
            False,
        ),
        (
            TTSProfileCollisionSnapshot(
                _profile(display_name="Imported voice"),
                _profile(display_name="Imported voice"),
            ),
            ("reuse", "copy"),
            _PROFILE_ID,
            False,
            False,
        ),
        (
            TTSProfileCollisionSnapshot(
                _profile(display_name="UUID match"),
                _profile(profile_id=_DUPLICATE_ID, display_name="Imported voice"),
            ),
            ("copy",),
            None,
            False,
            False,
        ),
    ],
)
@pytest.mark.asyncio
async def test_portable_collision_matrix_is_explicit_and_never_mutates(
    collisions: TTSProfileCollisionSnapshot,
    choices: tuple[str, ...],
    reuse_id: UUID | None,
    copy_keeps_id: bool,
    copy_keeps_name: bool,
) -> None:
    repository = _FakeRepository()
    repository.collision_result = collisions
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    service = TTSProfileService(
        repository,
        tts_service,
        _uuid_factory=lambda: _PORTABLE_COPY_ID,
    )
    portable = _portable_profile()
    observation = await service.observe_portable_profile(portable)

    plan = await service.inspect_portable_profile_import(observation)

    assert plan.allowed_choices == choices
    assert (
        None if plan.reuse_profile is None else plan.reuse_profile.profile_id
    ) == reuse_id
    assert (plan.copy_candidate.profile_id == portable.profile_id) is copy_keeps_id
    assert (
        plan.copy_candidate.draft.display_name == portable.draft.display_name
    ) is copy_keeps_name
    assert all(call[0] == "collisions" for call in repository.calls)


@pytest.mark.asyncio
async def test_portable_commit_revalidates_available_selection_and_assigns_atomically() -> (
    None
):
    repository = _FakeRepository()
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    service = TTSProfileService(repository, tts_service)
    observation = await service.observe_portable_profile(_portable_profile())
    plan = await service.inspect_portable_profile_import(observation)
    character_ref = _character_ref(source="local", authority_id="local-db")

    result = await service.commit_portable_profile_import(
        plan,
        "create",
        character_ref,
        expected_current=None,
    )

    assert result.created is True
    assert result.availability == "available"
    assert result.assignment is not None
    assert result.assignment.character_ref == character_ref
    assert result.loaded.profile.profile_id == _PROFILE_ID
    assert [call[0] for call in repository.calls] == [
        "collisions",
        "create_with_assignment",
    ]
    assert tts_service.capability_calls == [
        ("audio_cpp", ()),
        ("audio_cpp", ()),
    ]


@pytest.mark.asyncio
async def test_portable_commit_persists_for_repair_when_availability_changes() -> None:
    repository = _FakeRepository()
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    service = TTSProfileService(repository, tts_service)
    observation = await service.observe_portable_profile(_portable_profile())
    plan = await service.inspect_portable_profile_import(observation)
    tts_service.snapshot = _capability_snapshot(
        models=(),
        health_state="unavailable",
    )

    result = await service.commit_portable_profile_import(
        plan,
        "create",
        _character_ref(source="local", authority_id="local-db"),
        expected_current=None,
    )

    assert result.created is True
    assert result.availability == "unavailable"
    assert result.assignment is None
    assert [call[0] for call in repository.calls] == ["collisions", "create"]


@pytest.mark.asyncio
async def test_reusing_unavailable_profile_never_replaces_existing_assignment() -> None:
    repository = _FakeRepository()
    existing = _profile(display_name="Imported voice")
    repository.collision_result = TTSProfileCollisionSnapshot(existing, existing)
    tts_service = _FakeTTSService(
        _capability_snapshot(models=(), health_state="unavailable")
    )
    service = TTSProfileService(repository, tts_service)
    observation = await service.observe_portable_profile(_portable_profile())
    plan = await service.inspect_portable_profile_import(observation)
    current = _assignment(profile_id=_DUPLICATE_ID)

    result = await service.commit_portable_profile_import(
        plan,
        "reuse",
        current.character_ref,
        expected_current=current,
    )

    assert result.created is False
    assert result.availability == "unavailable"
    assert result.assignment is None
    assert result.loaded.profile == existing
    assert [call[0] for call in repository.calls] == ["collisions"]


@pytest.mark.asyncio
async def test_reusing_available_profile_assigns_only_the_observed_profile_identity() -> (
    None
):
    repository = _FakeRepository()
    existing = _profile(display_name="Imported voice")
    repository.collision_result = TTSProfileCollisionSnapshot(existing, existing)
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    service = TTSProfileService(repository, tts_service)
    observation = await service.observe_portable_profile(_portable_profile())
    plan = await service.inspect_portable_profile_import(observation)
    character_ref = _character_ref(source="local", authority_id="local-db")

    result = await service.commit_portable_profile_import(
        plan,
        "reuse",
        character_ref,
        expected_current=None,
    )

    assert result.assignment == CharacterTTSAssignment(
        character_ref,
        existing.profile_id,
    )
    assert repository.last_expected_profile == existing


@pytest.mark.asyncio
async def test_portable_commit_auto_applies_unverified_legacy_profile_on_create() -> (
    None
):
    """Task-6c (TASK-2450 AC#8): an imported legacy-provider profile is always
    classified 'unverified' (observe_portable_profile's early return for any
    non-audio_cpp provider) -- it must auto-apply exactly like an 'available'
    one, not fall through to the unassigned-for-repair branch."""

    repository = _FakeRepository()
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    service = TTSProfileService(repository, tts_service)
    portable = _portable_profile(
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        response_format="mp3",
    )
    observation = await service.observe_portable_profile(portable)
    assert observation.availability == "unverified"
    plan = await service.inspect_portable_profile_import(observation)
    character_ref = _character_ref(source="local", authority_id="local-db")

    result = await service.commit_portable_profile_import(
        plan,
        "create",
        character_ref,
        expected_current=None,
    )

    assert result.created is True
    # Note: the successful-assign branch reports the literal "available" here
    # regardless of the observed state, matching the pre-existing (unchanged)
    # behavior on the already-passing audio_cpp path -- this field is never
    # read by any caller (confirmed by grep across tldw_chatbook/), so it is
    # left untouched; what this test pins is that assignment actually
    # happened, which is the live-user-visible outcome.
    assert result.availability == "available"
    assert result.assignment is not None
    assert result.assignment.character_ref == character_ref
    assert [call[0] for call in repository.calls] == [
        "collisions",
        "create_with_assignment",
    ]


@pytest.mark.asyncio
async def test_portable_commit_auto_applies_unverified_legacy_profile_on_reuse() -> (
    None
):
    """Complementary reuse-path pin for the same fix."""

    repository = _FakeRepository()
    existing = _profile(
        display_name="Imported voice",
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        response_format="mp3",
    )
    repository.collision_result = TTSProfileCollisionSnapshot(existing, existing)
    tts_service = _FakeTTSService(_capability_snapshot(models=(_model("model-a"),)))
    service = TTSProfileService(repository, tts_service)
    portable = _portable_profile(
        display_name="Imported voice",
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        response_format="mp3",
    )
    observation = await service.observe_portable_profile(portable)
    assert observation.availability == "unverified"
    plan = await service.inspect_portable_profile_import(observation)
    character_ref = _character_ref(source="local", authority_id="local-db")

    result = await service.commit_portable_profile_import(
        plan,
        "reuse",
        character_ref,
        expected_current=None,
    )

    assert result.assignment == CharacterTTSAssignment(
        character_ref,
        existing.profile_id,
    )
    # Same pre-existing hardcoded "available" on the success branch as the
    # create-path test above -- unchanged, not user-visible.
    assert result.availability == "available"
