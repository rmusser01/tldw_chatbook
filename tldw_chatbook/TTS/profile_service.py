"""Native-only TTS generation-profile service values and operations."""

from __future__ import annotations

import asyncio
import math
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import islice
from threading import RLock
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, cast, runtime_checkable
from uuid import UUID, uuid4

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSCloneGenerationEvidence,
    TTSConfigurationRevisionError,
    TTSModelInfo,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
    AudioCppArtifactConsumerRequirement,
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
from tldw_chatbook.TTS.profile_types import (
    AUDIO_CPP_PROFILE_SPEED,
    PROFILE_PROVIDER_FORMATS,
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileCollisionSnapshot,
    TTSProfileDraft,
    TTSProfilePage,
    TTSProfileVerificationEvidence,
    profile_options_fingerprint,
)
from tldw_chatbook.TTS.sample_audio_validation import validate_playable_audio_file
from tldw_chatbook.TTS.TTS_Generation import (
    AudioCppGuidedDependencySnapshot,
    validate_audio_cpp_guided_dependency_snapshot,
)

ProfileAvailabilityState: TypeAlias = Literal[
    "available",
    "unavailable",
    "unverified",
]
ProfileRecoveryAction: TypeAlias = Literal["none", "refresh", "edit"]
ProfileDependencyReason: TypeAlias = Literal[
    "none",
    "recipe_missing",
    "recipe_mismatch",
    "recipe_pending_apply",
]
ProfileDependencyAction: TypeAlias = Literal[
    "none",
    "open_audio_cpp_settings",
    "open_speech_lab_apply",
]
ProfilePortabilityAdvisory: TypeAlias = Literal[
    "none",
    "recipe_provenance_unavailable",
]
ProfilePortabilityAction: TypeAlias = Literal["none", "generate_new_profile"]
PortableProfileImportChoice: TypeAlias = Literal["create", "reuse", "copy"]

_PROFILE_PROVIDER_ID = "audio_cpp"
_PROFILE_PAGE_LIMIT = 50
_PROFILE_SAMPLE_EVIDENCE_LIMIT = 256
_PROFILE_CONSUMER_SNAPSHOT_LIMIT = 200
_CHARACTER_REF_TYPE: type[CharacterRef] = CharacterRef
_CHARACTER_TTS_ASSIGNMENT_TYPE: type[CharacterTTSAssignment] = CharacterTTSAssignment
_ASSIGNED_TTS_PROFILE_SNAPSHOT_TYPE: type[AssignedTTSProfileSnapshot] = (
    AssignedTTSProfileSnapshot
)
_TTS_PROFILE_COLLISION_SNAPSHOT_TYPE: type[TTSProfileCollisionSnapshot] = (
    TTSProfileCollisionSnapshot
)
_TTS_GENERATION_PROFILE_TYPE: type[TTSGenerationProfile] = TTSGenerationProfile
_TTS_PROFILE_DRAFT_TYPE: type[TTSProfileDraft] = TTSProfileDraft
_TTS_CLONE_REFERENCE_TYPE: type[TTSCloneReference] = TTSCloneReference
_TTS_CLONE_REFERENCE_SUMMARY_TYPE: type[TTSCloneReferenceSummary] = (
    TTSCloneReferenceSummary
)
_TTS_CLONE_RECIPE_REQUIREMENT_TYPE: type[TTSCloneRecipeRequirement] = (
    TTSCloneRecipeRequirement
)
_PORTABLE_TTS_PROFILE_TYPE: type[PortableTTSProfile] = PortableTTSProfile
_TTS_NATIVE_CAPABILITY_SNAPSHOT_TYPE: type[TTSNativeCapabilitySnapshot] = (
    TTSNativeCapabilitySnapshot
)
_BoundedValue = TypeVar("_BoundedValue")
_AVAILABILITY_RECOVERY: Mapping[
    ProfileAvailabilityState,
    ProfileRecoveryAction,
] = MappingProxyType(
    {
        "available": "none",
        "unavailable": "edit",
        "unverified": "refresh",
    }
)
#: The recovery actions each state may honestly carry.
#:
#: "unverified" admits two, one per provider class: audio.cpp is unverified
#: only until its next capability preflight, so "refresh" is a real recovery;
#: the legacy providers have no catalog to preflight (`observe_availability`
#: skips them by design), so their "unverified" is permanent and the only
#: honest action is the inert one -- ADR-031 forbids a control that claims a
#: recovery it can never perform.
_ALLOWED_RECOVERY_ACTIONS: Mapping[
    ProfileAvailabilityState,
    frozenset[ProfileRecoveryAction],
] = MappingProxyType(
    {
        "available": frozenset({"none"}),
        "unavailable": frozenset({"edit"}),
        "unverified": frozenset({"refresh", "none"}),
    }
)


@runtime_checkable
class _ProfileRepositoryProtocol(Protocol):
    @property
    def generation(self) -> int: ...

    async def list_profiles(
        self,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ProfileStoreResult[TTSProfilePage]: ...

    async def create_profile(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID | None = None,
        *,
        expected_generation: int | None = None,
    ) -> ProfileStoreResult[TTSGenerationProfile]: ...

    async def create_profile_with_reference(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID,
        canonical: CanonicalTTSCloneReference,
        recipe_requirement: TTSCloneRecipeRequirement,
        *,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSGenerationProfile]: ...

    async def update_profile(
        self,
        profile_id: UUID,
        expected_revision: int,
        draft: TTSProfileDraft,
        *,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSGenerationProfile]: ...

    async def delete_profile(
        self,
        profile_id: UUID,
        *,
        expected_generation: int,
    ) -> ProfileStoreResult[None]: ...

    async def assignment_count(
        self,
        profile_id: UUID,
    ) -> ProfileStoreResult[int]: ...

    async def set_assignment(
        self,
        character_ref: CharacterRef,
        profile_id: UUID,
        *,
        expected_generation: int,
        expected_profile_revision: int,
        expected_current_profile_id: UUID | None,
        expected_profile: TTSGenerationProfile | None = None,
    ) -> ProfileStoreResult[CharacterTTSAssignment]: ...

    async def remove_assignment(
        self,
        character_ref: CharacterRef,
        *,
        expected_generation: int,
        expected_profile_id: UUID,
    ) -> ProfileStoreResult[None]: ...

    async def get_assigned_profile(
        self,
        character_ref: CharacterRef,
    ) -> ProfileStoreResult[AssignedTTSProfileSnapshot | None]: ...

    async def get_profile(
        self,
        profile_id: UUID,
    ) -> ProfileStoreResult[TTSGenerationProfile]: ...

    async def get_reference(
        self,
        profile_id: UUID,
        *,
        expected_revision: int,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSCloneReference]: ...


@runtime_checkable
class _PortableProfileRepositoryProtocol(Protocol):
    """Repository additions required only by explicit portability workflows."""

    async def create_profile_with_assignment(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID,
        character_ref: CharacterRef,
        *,
        expected_generation: int,
        expected_current_profile_id: UUID | None,
    ) -> ProfileStoreResult[AssignedTTSProfileSnapshot]: ...

    async def get_profile_collisions(
        self,
        profile_id: UUID,
        draft: TTSProfileDraft,
    ) -> ProfileStoreResult[TTSProfileCollisionSnapshot]: ...


@runtime_checkable
class _ProfileTTSServiceProtocol(Protocol):
    def configuration_revision(self, provider_id: str) -> int: ...

    async def get_native_capability_snapshot(
        self,
        provider_id: str,
        exact_voice_model_ids: Iterable[str],
    ) -> TTSNativeCapabilitySnapshot: ...

    async def require_current_configuration_revision(
        self,
        provider_id: str,
        expected_revision: int,
    ) -> None: ...

    async def audio_cpp_guided_dependency_snapshot(
        self,
        requirement: TTSCloneRecipeRequirement,
    ) -> AudioCppGuidedDependencySnapshot: ...


@runtime_checkable
class _ArtifactLeaseCoordinatorProtocol(Protocol):
    def lease_consumers(
        self,
        consumers: Iterable[AudioCppArtifactConsumerRequirement],
    ) -> object: ...


def _validate_nonnegative_integer(value: object, code: str) -> int:
    if type(value) is not int or value < 0:
        raise ProfileValidationError(code)
    return value


def _freeze_bounded_sequence(
    values: object,
    expected_type: type[_BoundedValue],
    *,
    maximum: int = _PROFILE_PAGE_LIMIT,
) -> tuple[_BoundedValue, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ProfileValidationError("profiles")
    source_length = 0
    source_length_failed = False
    try:
        source_length = len(values)
    except Exception:  # noqa: BLE001 - hostile sequences fail closed
        source_length_failed = True
    if source_length_failed or source_length > maximum:
        raise ProfileValidationError("profiles")

    frozen_sample: tuple[object, ...] = ()
    freeze_failed = False
    try:
        frozen_sample = tuple(islice(values, maximum + 1))
    except Exception:  # noqa: BLE001 - hostile sequences fail closed
        freeze_failed = True
    if freeze_failed or len(frozen_sample) > maximum:
        raise ProfileValidationError("profiles")
    if not all(type(item) is expected_type for item in frozen_sample):
        raise ProfileValidationError("profiles")
    return cast(tuple[_BoundedValue, ...], frozen_sample)


def _validate_availability_state(value: object) -> ProfileAvailabilityState:
    if type(value) is not str or value not in _AVAILABILITY_RECOVERY:
        raise ProfileValidationError("availability")
    return cast(ProfileAvailabilityState, value)


def _validate_recovery_action(
    value: object,
    state: ProfileAvailabilityState,
) -> ProfileRecoveryAction:
    if type(value) is not str or value not in {"none", "refresh", "edit"}:
        raise ProfileValidationError("recovery_action")
    action = cast(ProfileRecoveryAction, value)
    if action not in _ALLOWED_RECOVERY_ACTIONS[state]:
        raise ProfileValidationError("recovery_action")
    return action


def _mapping_is_empty(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        return next(iter(value), None) is None and len(value) == 0
    except Exception:  # noqa: BLE001 - hostile mappings are unsupported
        return False


def _selection_is_profile_safe(
    provider_id: object,
    response_format: object,
    speed: object,
    options: object,
) -> bool:
    if type(provider_id) is not str or type(response_format) is not str:
        return False
    formats = PROFILE_PROVIDER_FORMATS.get(provider_id)
    if formats is None or response_format not in formats:
        return False
    if type(speed) is not float or not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
        return False
    if not _mapping_is_empty(options):
        return False
    if provider_id == _PROFILE_PROVIDER_ID:
        return speed == AUDIO_CPP_PROFILE_SPEED
    return True


def _matches_exact_canonical_value(value: object, canonical: object) -> bool:
    if type(value) is not type(canonical):
        return False
    if isinstance(canonical, Mapping):
        value_mapping = cast(Mapping[object, object], value)
        if len(value_mapping) != len(canonical):
            return False
        return all(
            key in value_mapping
            and _matches_exact_canonical_value(value_mapping[key], item)
            for key, item in canonical.items()
        )
    if isinstance(canonical, tuple):
        value_sequence = cast(tuple[object, ...], value)
        return len(value_sequence) == len(canonical) and all(
            _matches_exact_canonical_value(actual, expected)
            for actual, expected in zip(value_sequence, canonical, strict=True)
        )
    return value == canonical


def _canonicalize_exact_character_ref(value: object) -> CharacterRef:
    """Return a fresh exact character reference or fail closed."""

    if type(value) is not _CHARACTER_REF_TYPE:
        raise ProfileValidationError("assignment")
    character_ref = cast(CharacterRef, value)
    canonical: CharacterRef | None = None
    valid = False
    failed = False
    try:
        canonical = CharacterRef(
            source=character_ref.source,
            authority_id=character_ref.authority_id,
            character_id=character_ref.character_id,
        )
        valid = all(
            _matches_exact_canonical_value(source, expected)
            for source, expected in (
                (character_ref.source, canonical.source),
                (character_ref.authority_id, canonical.authority_id),
                (character_ref.character_id, canonical.character_id),
            )
        )
    except Exception:  # noqa: BLE001 - hostile identity values fail closed
        failed = True
    if failed or not valid or canonical is None:
        raise ProfileValidationError("assignment")
    return canonical


def _canonicalize_exact_assignment(value: object) -> CharacterTTSAssignment:
    """Return a fresh exact character assignment or fail closed."""

    if type(value) is not _CHARACTER_TTS_ASSIGNMENT_TYPE:
        raise ProfileValidationError("assignment")
    assignment = cast(CharacterTTSAssignment, value)
    canonical: CharacterTTSAssignment | None = None
    valid = False
    failed = False
    try:
        character_ref = _canonicalize_exact_character_ref(assignment.character_ref)
        profile_id = assignment.profile_id
        if type(profile_id) is not UUID or type(profile_id.int) is not int:
            raise TypeError
        canonical_profile_id = UUID(int=profile_id.int)
        canonical = CharacterTTSAssignment(
            character_ref=character_ref,
            profile_id=canonical_profile_id,
        )
        valid = (
            _matches_exact_canonical_value(
                assignment.character_ref,
                canonical.character_ref,
            )
            and profile_id.int == canonical_profile_id.int
        )
    except Exception:  # noqa: BLE001 - hostile assignment values fail closed
        failed = True
    if failed or not valid or canonical is None:
        raise ProfileValidationError("assignment")
    return canonical


def _canonicalize_exact_profile_id(value: object) -> UUID:
    """Return a fresh exact profile UUID or fail closed.

    Mirrors `_canonicalize_exact_character_ref`'s defensive shape: a hostile
    `UUID` subclass overriding comparison or attribute access cannot slip a
    noncanonical identity past this boundary, since the returned value is
    always rebuilt from the plain integer representation.
    """

    canonical: UUID | None = None
    failed = False
    try:
        if type(value) is not UUID or type(value.int) is not int:
            raise TypeError
        canonical = UUID(int=value.int)
    except Exception:  # noqa: BLE001 - hostile identity values fail closed
        failed = True
    if failed or canonical is None:
        raise ProfileValidationError("profile_id")
    return canonical


def _canonicalize_exact_recipe_requirement(
    value: object,
) -> TTSCloneRecipeRequirement | None:
    """Return a fresh exact clone recipe requirement or legacy absence."""

    if value is None:
        return None
    if type(value) is not _TTS_CLONE_RECIPE_REQUIREMENT_TYPE:
        raise ProfileValidationError("reference_invalid")
    requirement = cast(TTSCloneRecipeRequirement, value)
    try:
        canonical = TTSCloneRecipeRequirement(
            recipe_id=requirement.recipe_id,
            recipe_revision=requirement.recipe_revision,
            model_id=requirement.model_id,
        )
    except Exception:
        raise ProfileValidationError("reference_invalid") from None
    return canonical


def _canonicalize_exact_reference_summary(
    value: object,
) -> TTSCloneReferenceSummary:
    """Return a fresh exact private-reference summary or fail closed."""

    if type(value) is not _TTS_CLONE_REFERENCE_SUMMARY_TYPE:
        raise ProfileValidationError("reference_invalid")
    summary = cast(TTSCloneReferenceSummary, value)
    try:
        canonical = TTSCloneReferenceSummary(
            reference_id=_canonicalize_exact_profile_id(summary.reference_id),
            byte_length=summary.byte_length,
            duration_ms=summary.duration_ms,
            sample_rate_hz=summary.sample_rate_hz,
            channels=summary.channels,
            sample_encoding=summary.sample_encoding,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            recipe_requirement=_canonicalize_exact_recipe_requirement(
                summary.recipe_requirement
            ),
        )
    except Exception:
        raise ProfileValidationError("reference_invalid") from None
    if canonical != summary:
        raise ProfileValidationError("reference_invalid")
    return canonical


def _canonicalize_exact_reference(value: object) -> TTSCloneReference:
    """Return a fresh exact private reference without exposing its values."""

    if type(value) is not _TTS_CLONE_REFERENCE_TYPE:
        raise ProfileValidationError("reference_invalid")
    reference = cast(TTSCloneReference, value)
    try:
        summary = _canonicalize_exact_reference_summary(reference.summary)
        direct_requirement = _canonicalize_exact_recipe_requirement(
            reference.recipe_requirement
        )
        if summary.recipe_requirement != direct_requirement:
            raise ValueError
        canonical = TTSCloneReference(
            summary=summary,
            reference_text=reference.reference_text,
            sha256=reference.sha256,
            wav_bytes=reference.wav_bytes,
            recipe_requirement=summary.recipe_requirement,
        )
    except Exception:
        raise ProfileValidationError("reference_invalid") from None
    if (
        type(reference.reference_text) is not str
        or type(reference.sha256) is not str
        or type(reference.wav_bytes) is not bytes
        or canonical.reference_text != reference.reference_text
        or canonical.sha256 != reference.sha256
        or canonical.wav_bytes != reference.wav_bytes
    ):
        raise ProfileValidationError("reference_invalid")
    return canonical


def _canonicalize_exact_profile(value: object) -> TTSGenerationProfile:
    """Return a fresh profile only when every source field is already canonical."""

    if type(value) is not _TTS_GENERATION_PROFILE_TYPE:
        raise ProfileValidationError("profiles")
    profile = cast(TTSGenerationProfile, value)
    canonical: TTSGenerationProfile | None = None
    valid = False
    failed = False
    try:
        canonical = TTSGenerationProfile(
            profile_id=profile.profile_id,
            display_name=profile.display_name,
            normalized_name=profile.normalized_name,
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
            revision=profile.revision,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
            reference=(
                None
                if profile.reference is None
                else _canonicalize_exact_reference_summary(profile.reference)
            ),
        )
        valid = all(
            _matches_exact_canonical_value(source, expected)
            for source, expected in (
                (profile.profile_id, canonical.profile_id),
                (profile.display_name, canonical.display_name),
                (profile.normalized_name, canonical.normalized_name),
                (profile.provider_id, canonical.provider_id),
                (profile.model_id, canonical.model_id),
                (profile.voice_id, canonical.voice_id),
                (profile.response_format, canonical.response_format),
                (profile.speed, canonical.speed),
                (profile.options, canonical.options),
                (profile.revision, canonical.revision),
                (profile.created_at, canonical.created_at),
                (profile.updated_at, canonical.updated_at),
                (profile.reference, canonical.reference),
            )
        )
    except Exception:  # noqa: BLE001 - hostile profile values fail closed
        failed = True
    if failed or not valid or canonical is None:
        raise ProfileValidationError("profiles")
    return canonical


def _canonicalize_exact_draft(value: object) -> TTSProfileDraft:
    """Return a fresh profile draft only when all source fields are canonical."""

    if type(value) is not _TTS_PROFILE_DRAFT_TYPE:
        raise ProfileValidationError("profiles")
    draft = cast(TTSProfileDraft, value)
    canonical: TTSProfileDraft | None = None
    valid = False
    try:
        canonical = TTSProfileDraft(
            display_name=draft.display_name,
            provider_id=draft.provider_id,
            model_id=draft.model_id,
            voice_id=draft.voice_id,
            response_format=draft.response_format,
            speed=draft.speed,
            options=draft.options,
        )
        valid = all(
            _matches_exact_canonical_value(source, expected)
            for source, expected in (
                (draft.display_name, canonical.display_name),
                (draft.provider_id, canonical.provider_id),
                (draft.model_id, canonical.model_id),
                (draft.voice_id, canonical.voice_id),
                (draft.response_format, canonical.response_format),
                (draft.speed, canonical.speed),
                (draft.options, canonical.options),
            )
        )
    except Exception:  # noqa: BLE001 - hostile typed values fail closed
        canonical = None
    if not valid or canonical is None:
        raise ProfileValidationError("profiles")
    return canonical


def _canonicalize_exact_portable_profile(value: object) -> PortableTTSProfile:
    """Return a fresh exact portable profile or fail closed."""

    if type(value) is not _PORTABLE_TTS_PROFILE_TYPE:
        raise ProfileValidationError("profiles")
    portable = cast(PortableTTSProfile, value)
    if type(portable.profile_id) is not UUID:
        raise ProfileValidationError("profile_id")
    return PortableTTSProfile(
        profile_id=UUID(int=portable.profile_id.int),
        draft=_canonicalize_exact_draft(portable.draft),
    )


def _canonicalize_exact_collision_snapshot(
    value: object,
) -> TTSProfileCollisionSnapshot:
    """Copy an exact collision read without trusting collaborator values."""

    if type(value) is not _TTS_PROFILE_COLLISION_SNAPSHOT_TYPE:
        raise ProfileValidationError("profiles")
    snapshot = cast(TTSProfileCollisionSnapshot, value)
    return TTSProfileCollisionSnapshot(
        profile_id_match=(
            None
            if snapshot.profile_id_match is None
            else _canonicalize_exact_profile(snapshot.profile_id_match)
        ),
        normalized_name_match=(
            None
            if snapshot.normalized_name_match is None
            else _canonicalize_exact_profile(snapshot.normalized_name_match)
        ),
    )


def _canonicalize_exact_assigned_profile(
    value: object,
) -> AssignedTTSProfileSnapshot:
    """Return a fresh exact joined assignment/profile value or fail closed."""

    if type(value) is not _ASSIGNED_TTS_PROFILE_SNAPSHOT_TYPE:
        raise ProfileValidationError("assignment")
    snapshot = cast(AssignedTTSProfileSnapshot, value)
    canonical: AssignedTTSProfileSnapshot | None = None
    failed = False
    try:
        canonical = AssignedTTSProfileSnapshot(
            assignment=_canonicalize_exact_assignment(snapshot.assignment),
            profile=_canonicalize_exact_profile(snapshot.profile),
        )
    except Exception:  # noqa: BLE001 - hostile joined values fail closed
        failed = True
    if failed or canonical is None:
        raise ProfileValidationError("assignment")
    return canonical


def _canonicalize_consumed_capability_snapshot(
    value: object,
    *,
    relevant_model_ids: tuple[str, ...],
) -> TTSNativeCapabilitySnapshot:
    """Copy only profile-classification capability fields into safe values."""

    canonical: TTSNativeCapabilitySnapshot | None = None
    failed = False
    try:
        if type(value) is not _TTS_NATIVE_CAPABILITY_SNAPSHOT_TYPE:
            raise TypeError
        snapshot = cast(TTSNativeCapabilitySnapshot, value)

        provider_id = snapshot.provider_id
        if type(provider_id) is not str or provider_id != _PROFILE_PROVIDER_ID:
            raise ValueError
        configuration_revision = snapshot.configuration_revision
        if type(configuration_revision) is not int or configuration_revision < 0:
            raise ValueError
        state = snapshot.state
        if type(state) is not str or state not in ("complete", "unverified"):
            raise ValueError
        if (
            type(relevant_model_ids) is not tuple
            or len(relevant_model_ids) > _PROFILE_PAGE_LIMIT
            or any(type(model_id) is not str for model_id in relevant_model_ids)
        ):
            raise ValueError
        relevant_models = set(relevant_model_ids)

        catalog = snapshot.catalog
        canonical_catalog: TTSProviderCatalog | None = None
        if catalog is not None:
            if type(catalog) is not TTSProviderCatalog:
                raise TypeError
            catalog_provider_id = catalog.provider_id
            if (
                type(catalog_provider_id) is not str
                or catalog_provider_id != provider_id
            ):
                raise ValueError
            catalog_revision = catalog.revision
            if type(catalog_revision) is not int or catalog_revision < 0:
                raise ValueError
            health = catalog.health
            if type(health) is not ProviderHealth:
                raise TypeError
            health_state = health.state
            if type(health_state) is not str or health_state not in (
                "available",
                "unavailable",
                "not_configured",
                "reconfiguring",
                "closed",
            ):
                raise ValueError
            health_fresh = health.fresh
            if type(health_fresh) is not bool:
                raise TypeError
            models = catalog.models
            if type(models) is not tuple:
                raise TypeError

            canonical_models: list[TTSModelInfo] = []
            seen_relevant_models: set[str] = set()
            for model in models:
                if type(model) is not TTSModelInfo:
                    raise TypeError
                model_id = model.model_id
                if type(model_id) is not str:
                    raise TypeError
                if model_id not in relevant_models:
                    continue
                if model_id in seen_relevant_models:
                    raise ValueError
                seen_relevant_models.add(model_id)
                formats = model.formats
                if type(formats) is not tuple or any(
                    type(response_format) is not str for response_format in formats
                ):
                    raise TypeError
                omit_voice_uses_server_default = model.omit_voice_uses_server_default
                if type(omit_voice_uses_server_default) is not bool:
                    raise TypeError
                canonical_models.append(
                    TTSModelInfo(
                        model_id=model_id,
                        display_name="",
                        family="",
                        upstream_mode="",
                        formats=formats,
                        voices=(),
                        supports_speed=False,
                        supports_options=(),
                        omit_voice_uses_server_default=(omit_voice_uses_server_default),
                    )
                )
            canonical_catalog = TTSProviderCatalog(
                provider_id=provider_id,
                revision=catalog_revision,
                health=ProviderHealth(
                    state=cast(Any, health_state),
                    fresh=health_fresh,
                ),
                models=tuple(canonical_models),
            )

        voice_results = snapshot.voice_results
        if (
            type(voice_results) is not MappingProxyType
            or len(voice_results) > _PROFILE_PAGE_LIMIT
        ):
            raise TypeError
        canonical_voice_results: dict[str, TTSVoiceDiscoveryResult] = {}
        for model_id in relevant_model_ids:
            result = voice_results.get(model_id)
            if result is None:
                continue
            if type(result) is not TTSVoiceDiscoveryResult:
                raise TypeError
            result_provider_id = result.provider_id
            result_model_id = result.model_id
            result_catalog_revision = result.catalog_revision
            result_voices = result.voices
            result_state = result.state
            if (
                type(result_provider_id) is not str
                or result_provider_id != provider_id
                or type(result_model_id) is not str
                or result_model_id != model_id
                or type(result_catalog_revision) is not int
                or result_catalog_revision < 0
                or type(result_voices) is not tuple
                or any(type(voice_id) is not str for voice_id in result_voices)
                or type(result_state) is not str
                or result_state not in ("complete", "model_missing", "unverified")
            ):
                raise ValueError
            canonical_voice_results[model_id] = TTSVoiceDiscoveryResult(
                provider_id=provider_id,
                model_id=model_id,
                catalog_revision=result_catalog_revision,
                voices=result_voices,
                state=cast(Any, result_state),
            )

        canonical = TTSNativeCapabilitySnapshot(
            provider_id=provider_id,
            configuration_revision=configuration_revision,
            state=cast(Any, state),
            catalog=canonical_catalog,
            voice_results=canonical_voice_results,
        )
    except Exception:  # noqa: BLE001 - hostile capability values fail closed
        failed = True
    if failed or canonical is None:
        raise ProfileServiceError("operation_failed")
    return canonical


def _profile_is_structurally_supported(profile: TTSGenerationProfile) -> bool:
    return _selection_is_profile_safe(
        profile.provider_id,
        profile.response_format,
        profile.speed,
        profile.options,
    )


def _recovery_action(
    provider_id: str,
    state: ProfileAvailabilityState,
) -> ProfileRecoveryAction:
    """Return the only recovery this provider can actually perform."""

    if state == "unverified" and provider_id != _PROFILE_PROVIDER_ID:
        return "none"
    return _AVAILABILITY_RECOVERY[state]


def _availability(
    profile_id: UUID,
    state: ProfileAvailabilityState,
    provider_id: str,
    dependency: TTSProfileDependencyProjection | None = None,
    provider_configuration_revision: int | None = None,
) -> TTSProfileAvailability:
    return TTSProfileAvailability(
        profile_id=profile_id,
        state=state,
        recovery_action=_recovery_action(provider_id, state),
        dependency=(
            TTSProfileDependencyProjection() if dependency is None else dependency
        ),
        provider_configuration_revision=provider_configuration_revision,
    )


@dataclass(frozen=True, slots=True)
class TTSPlaygroundSelectionPreset:
    """One immutable exact profile selection handed to the Playground."""

    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: Mapping[str, Any] = field(default_factory=dict)
    availability: ProfileAvailabilityState = "unverified"
    profile_id: UUID | None = None
    repository_generation: int | None = None
    profile_revision: int | None = None

    def __post_init__(self) -> None:
        draft = TTSProfileDraft(
            display_name="Profile preview",
            provider_id=self.provider_id,
            model_id=self.model_id,
            voice_id=self.voice_id,
            response_format=self.response_format,
            speed=self.speed,
            options=self.options,
        )
        state = _validate_availability_state(self.availability)
        identity = (
            self.profile_id,
            self.repository_generation,
            self.profile_revision,
        )
        if any(value is not None for value in identity):
            if any(value is None for value in identity):
                raise ValueError("Reference preview identity must be complete")
            if type(self.profile_id) is not UUID:
                raise TypeError("profile_id must be a UUID")
            if type(self.repository_generation) is not int:
                raise TypeError("repository_generation must be an integer")
            if self.repository_generation < 0:
                raise ValueError("repository_generation must be nonnegative")
            if type(self.profile_revision) is not int:
                raise TypeError("profile_revision must be an integer")
            if self.profile_revision < 1:
                raise ValueError("profile_revision must be positive")
        object.__setattr__(self, "provider_id", draft.provider_id)
        object.__setattr__(self, "model_id", draft.model_id)
        object.__setattr__(self, "voice_id", draft.voice_id)
        object.__setattr__(self, "response_format", draft.response_format)
        object.__setattr__(self, "speed", draft.speed)
        object.__setattr__(self, "options", draft.options)
        object.__setattr__(self, "availability", state)


@dataclass(frozen=True, slots=True)
class TTSProfilePageSnapshot:
    """One bounded immutable repository page and its lifecycle generation."""

    repository_generation: int
    profiles: tuple[TTSGenerationProfile, ...]
    total: int

    def __post_init__(self) -> None:
        generation = _validate_nonnegative_integer(
            self.repository_generation,
            "generation",
        )
        profiles = _freeze_bounded_sequence(
            self.profiles,
            TTSGenerationProfile,
        )
        canonical_profiles = tuple(
            _canonicalize_exact_profile(profile) for profile in profiles
        )
        total = _validate_nonnegative_integer(self.total, "total")
        if total < len(canonical_profiles):
            raise ProfileValidationError("total")
        object.__setattr__(self, "repository_generation", generation)
        object.__setattr__(self, "profiles", canonical_profiles)
        object.__setattr__(self, "total", total)


@dataclass(frozen=True, slots=True)
class LoadedTTSProfile:
    """One immutable profile version paired with its repository generation."""

    repository_generation: int
    profile: TTSGenerationProfile

    def __post_init__(self) -> None:
        generation = _validate_nonnegative_integer(
            self.repository_generation,
            "generation",
        )
        profile = _canonicalize_exact_profile(self.profile)
        object.__setattr__(self, "repository_generation", generation)
        object.__setattr__(self, "profile", profile)


@dataclass(frozen=True, slots=True)
class LoadedCharacterTTSAssignment:
    """One exact joined assignment read paired with its store generation."""

    repository_generation: int
    snapshot: AssignedTTSProfileSnapshot | None

    def __post_init__(self) -> None:
        generation = _validate_nonnegative_integer(
            self.repository_generation,
            "generation",
        )
        snapshot = self.snapshot
        if snapshot is not None:
            snapshot = _canonicalize_exact_assigned_profile(snapshot)
        object.__setattr__(self, "repository_generation", generation)
        object.__setattr__(self, "snapshot", snapshot)


@dataclass(frozen=True, slots=True)
class TTSProfileDependencyProjection:
    """Bounded dependency blocker plus an independent portability advisory."""

    reason: ProfileDependencyReason = "none"
    display: str | None = None
    action: ProfileDependencyAction = "none"
    advisory: ProfilePortabilityAdvisory = "none"
    advisory_display: str | None = None
    advisory_action: ProfilePortabilityAction = "none"

    def __post_init__(self) -> None:
        blockers = {
            "none": (None, "none"),
            "recipe_missing": ("Needs compatible model", "open_audio_cpp_settings"),
            "recipe_mismatch": ("Needs compatible model", "open_audio_cpp_settings"),
            "recipe_pending_apply": (
                "Compatible model saved; apply settings",
                "open_speech_lab_apply",
            ),
        }
        advisories = {
            "none": (None, "none"),
            "recipe_provenance_unavailable": (
                "Recipe provenance unavailable",
                "generate_new_profile",
            ),
        }
        if (
            type(self.reason) is not str
            or self.reason not in blockers
            or type(self.action) is not str
            or (self.display, self.action) != blockers[self.reason]
            or type(self.advisory) is not str
            or self.advisory not in advisories
            or type(self.advisory_action) is not str
            or (self.advisory_display, self.advisory_action)
            != advisories[self.advisory]
        ):
            raise ProfileValidationError("dependency")


@dataclass(frozen=True, slots=True)
class TTSProfileAvailability:
    """The current bounded availability state for one exact profile UUID."""

    profile_id: UUID
    state: ProfileAvailabilityState
    recovery_action: ProfileRecoveryAction
    dependency: TTSProfileDependencyProjection = field(
        default_factory=TTSProfileDependencyProjection
    )
    provider_configuration_revision: int | None = None

    def __post_init__(self) -> None:
        if type(self.profile_id) is not UUID:
            raise ProfileValidationError("profile_id")
        state = _validate_availability_state(self.state)
        action = _validate_recovery_action(self.recovery_action, state)
        dependency = self.dependency
        if type(dependency) is not TTSProfileDependencyProjection:
            raise ProfileValidationError("dependency")
        dependency = TTSProfileDependencyProjection(
            reason=dependency.reason,
            display=dependency.display,
            action=dependency.action,
            advisory=dependency.advisory,
            advisory_display=dependency.advisory_display,
            advisory_action=dependency.advisory_action,
        )
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "recovery_action", action)
        object.__setattr__(self, "dependency", dependency)
        provider_revision = self.provider_configuration_revision
        if provider_revision is not None:
            provider_revision = _validate_nonnegative_integer(
                provider_revision,
                "configuration_revision",
            )
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "recovery_action", action)
        object.__setattr__(
            self,
            "provider_configuration_revision",
            provider_revision,
        )


@dataclass(frozen=True, slots=True)
class TTSProfileAvailabilitySnapshot:
    """One page's availability under explicit store and provider revisions."""

    repository_generation: int
    configuration_revision: int
    catalog_revision: int | None
    profiles: tuple[TTSProfileAvailability, ...]

    def __post_init__(self) -> None:
        repository_generation = _validate_nonnegative_integer(
            self.repository_generation,
            "generation",
        )
        configuration_revision = _validate_nonnegative_integer(
            self.configuration_revision,
            "configuration_revision",
        )
        catalog_revision = self.catalog_revision
        if catalog_revision is not None:
            catalog_revision = _validate_nonnegative_integer(
                catalog_revision,
                "catalog_revision",
            )
        profiles = _freeze_bounded_sequence(
            self.profiles,
            TTSProfileAvailability,
        )
        if len({item.profile_id for item in profiles}) != len(profiles):
            raise ProfileValidationError("profiles")
        object.__setattr__(
            self,
            "repository_generation",
            repository_generation,
        )
        object.__setattr__(
            self,
            "configuration_revision",
            configuration_revision,
        )
        object.__setattr__(self, "catalog_revision", catalog_revision)
        object.__setattr__(self, "profiles", profiles)


def _provenance_projection(
    profile: TTSGenerationProfile,
) -> TTSProfileDependencyProjection:
    reference = profile.reference
    if reference is not None and reference.recipe_requirement is None:
        return TTSProfileDependencyProjection(
            advisory="recipe_provenance_unavailable",
            advisory_display="Recipe provenance unavailable",
            advisory_action="generate_new_profile",
        )
    return TTSProfileDependencyProjection()


def _dependency_projection(
    state: str,
    *,
    advisory: TTSProfileDependencyProjection,
) -> TTSProfileDependencyProjection:
    blockers: dict[
        str, tuple[ProfileDependencyReason, str | None, ProfileDependencyAction]
    ] = {
        "exact": ("none", None, "none"),
        "missing": (
            "recipe_missing",
            "Needs compatible model",
            "open_audio_cpp_settings",
        ),
        "mismatch": (
            "recipe_mismatch",
            "Needs compatible model",
            "open_audio_cpp_settings",
        ),
        "pending": (
            "recipe_pending_apply",
            "Compatible model saved; apply settings",
            "open_speech_lab_apply",
        ),
    }
    try:
        reason, display, action = blockers[state]
    except (KeyError, TypeError):
        raise ProfileServiceError("operation_failed") from None
    return TTSProfileDependencyProjection(
        reason=reason,
        display=display,
        action=action,
        advisory=advisory.advisory,
        advisory_display=advisory.advisory_display,
        advisory_action=advisory.advisory_action,
    )


@dataclass(frozen=True, slots=True)
class _ProfileEvidenceLifecycle:
    """Latest profile revision or tombstone observed by this service."""

    revision: int
    deleted: bool


@dataclass(frozen=True, slots=True)
class PortableProfileAvailabilityObservation:
    """Current local capability state for one sanitized portable profile."""

    repository_generation: int
    configuration_revision: int
    profile: PortableTTSProfile
    availability: ProfileAvailabilityState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_generation",
            _validate_nonnegative_integer(self.repository_generation, "generation"),
        )
        object.__setattr__(
            self,
            "configuration_revision",
            _validate_nonnegative_integer(
                self.configuration_revision,
                "configuration_revision",
            ),
        )
        object.__setattr__(
            self,
            "profile",
            _canonicalize_exact_portable_profile(self.profile),
        )
        object.__setattr__(
            self,
            "availability",
            _validate_availability_state(self.availability),
        )


@dataclass(frozen=True, slots=True)
class PortableProfileImportPlan:
    """Non-mutating collision decision for one observed portable profile."""

    observation: PortableProfileAvailabilityObservation
    allowed_choices: tuple[PortableProfileImportChoice, ...]
    reuse_profile: TTSGenerationProfile | None
    copy_candidate: PortableTTSProfile

    def __post_init__(self) -> None:
        if type(self.observation) is not PortableProfileAvailabilityObservation:
            raise ProfileValidationError("profiles")
        observation = PortableProfileAvailabilityObservation(
            repository_generation=self.observation.repository_generation,
            configuration_revision=self.observation.configuration_revision,
            profile=self.observation.profile,
            availability=self.observation.availability,
        )
        choices = self.allowed_choices
        if type(choices) is not tuple or choices not in (
            ("create",),
            ("copy",),
            ("reuse", "copy"),
        ):
            raise ProfileValidationError("choice")
        reuse_profile = self.reuse_profile
        if reuse_profile is not None:
            reuse_profile = _canonicalize_exact_profile(reuse_profile)
        if ("reuse" in choices) != (reuse_profile is not None):
            raise ProfileValidationError("choice")
        copy_candidate = _canonicalize_exact_portable_profile(self.copy_candidate)
        source_draft = observation.profile.draft
        candidate_draft = copy_candidate.draft
        generation_fields = (
            "provider_id",
            "model_id",
            "voice_id",
            "response_format",
            "speed",
            "options",
        )
        if not all(
            _matches_exact_canonical_value(
                getattr(candidate_draft, field_name),
                getattr(source_draft, field_name),
            )
            for field_name in generation_fields
        ):
            raise ProfileValidationError("profiles")
        if choices == ("create",):
            if copy_candidate != observation.profile:
                raise ProfileValidationError("profiles")
        elif copy_candidate == observation.profile:
            raise ProfileValidationError("profiles")
        if reuse_profile is not None:
            if not all(
                _matches_exact_canonical_value(
                    getattr(reuse_profile, field_name),
                    getattr(source_draft, field_name),
                )
                for field_name in generation_fields
            ):
                raise ProfileValidationError("profiles")
            if (
                reuse_profile.profile_id != observation.profile.profile_id
                and reuse_profile.normalized_name != source_draft.normalized_name
            ):
                raise ProfileValidationError("profiles")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "reuse_profile", reuse_profile)
        object.__setattr__(self, "copy_candidate", copy_candidate)


@dataclass(frozen=True, slots=True)
class PortableProfileImportResult:
    """Structured profile persistence and assignment outcome."""

    created: bool
    availability: ProfileAvailabilityState
    loaded: LoadedTTSProfile
    assignment: CharacterTTSAssignment | None

    def __post_init__(self) -> None:
        if type(self.created) is not bool:
            raise ProfileValidationError("created")
        object.__setattr__(
            self,
            "availability",
            _validate_availability_state(self.availability),
        )
        if type(self.loaded) is not LoadedTTSProfile:
            raise ProfileValidationError("profiles")
        loaded = LoadedTTSProfile(
            repository_generation=self.loaded.repository_generation,
            profile=self.loaded.profile,
        )
        assignment = self.assignment
        if assignment is not None:
            assignment = _canonicalize_exact_assignment(assignment)
            if assignment.profile_id != loaded.profile.profile_id:
                raise ProfileValidationError("assignment")
        object.__setattr__(self, "loaded", loaded)
        object.__setattr__(self, "assignment", assignment)


class TTSProfileService:
    """Manage native audio.cpp profiles over existing app-owned dependencies."""

    def __init__(
        self,
        repository: _ProfileRepositoryProtocol,
        tts_service: _ProfileTTSServiceProtocol,
        *,
        artifact_lease_coordinator: _ArtifactLeaseCoordinatorProtocol | None = None,
        _uuid_factory: Callable[[], UUID] | None = None,
    ) -> None:
        validation_failed = False
        try:
            if (
                not isinstance(repository, _ProfileRepositoryProtocol)
                or not isinstance(tts_service, _ProfileTTSServiceProtocol)
                or (
                    artifact_lease_coordinator is not None
                    and not isinstance(
                        artifact_lease_coordinator,
                        _ArtifactLeaseCoordinatorProtocol,
                    )
                )
                or (_uuid_factory is not None and not callable(_uuid_factory))
            ):
                validation_failed = True
        except Exception:  # noqa: BLE001 - hostile collaborators fail closed
            validation_failed = True
        if validation_failed:
            raise ProfileServiceError("operation_failed")
        self._repository = repository
        self._tts_service = tts_service
        self._sample_evidence: dict[UUID, TTSProfileVerificationEvidence] = {}
        self._sample_evidence_lock = RLock()
        self._sample_evidence_lifecycle: dict[UUID, _ProfileEvidenceLifecycle] = {}
        self._sample_evidence_epoch = 0
        self._consumer_mutation_lock = asyncio.Lock()
        if artifact_lease_coordinator is not None:
            self._artifact_lease_coordinator = artifact_lease_coordinator
        if _uuid_factory is not None:
            self._uuid_factory = _uuid_factory

    def _require_portable_repository(self) -> _PortableProfileRepositoryProtocol:
        """Return portability operations without expanding constructor needs."""

        validation_failed = False
        try:
            if not isinstance(
                self._repository,
                _PortableProfileRepositoryProtocol,
            ):
                validation_failed = True
        except Exception:  # noqa: BLE001 - hostile collaborators fail closed
            validation_failed = True
        if validation_failed:
            raise ProfileServiceError("operation_failed")
        return cast(_PortableProfileRepositoryProtocol, self._repository)

    @staticmethod
    def _artifact_consumer(
        provider_id: str,
        model_id: str,
        requirement: TTSCloneRecipeRequirement | None = None,
    ) -> AudioCppArtifactConsumerRequirement:
        return AudioCppArtifactConsumerRequirement(
            provider_id=provider_id,
            model_id=model_id,
            recipe_requirement=requirement,
        )

    @asynccontextmanager
    async def consumer_mutation_fence(self) -> AsyncIterator[None]:
        """Serialize one external repository mutation with bounded snapshots."""

        async with self._consumer_mutation_lock:
            yield

    @asynccontextmanager
    async def _lease_artifact_consumers(
        self,
        *consumers: AudioCppArtifactConsumerRequirement,
    ) -> AsyncIterator[None]:
        coordinator = getattr(self, "_artifact_lease_coordinator", None)
        if coordinator is None:
            async with self.consumer_mutation_fence():
                yield
            return
        async with cast(Any, coordinator.lease_consumers(consumers)):
            async with self.consumer_mutation_fence():
                yield

    async def _run_owned_repository_call(self, awaitable: Awaitable[Any]) -> Any:
        """Keep one admitted repository mutation leased through settlement."""

        task: asyncio.Future[Any] = asyncio.ensure_future(awaitable)
        cancellation: asyncio.CancelledError | None = None
        waiter = asyncio.current_task()
        requests = waiter.cancelling() if waiter is not None else 0
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                current = waiter.cancelling() if waiter is not None else 0
                if current > requests:
                    cancellation = cancellation or error
                    requests = current
            except BaseException:
                if not task.done():
                    raise
        try:
            result = task.result()
        except BaseException as error:
            if cancellation is not None:
                cancellation.add_note(
                    "profile repository mutation also failed after cancellation"
                )
                raise cancellation from None
            raise error
        if cancellation is not None:
            raise cancellation
        return result

    async def list_profiles(
        self,
        *,
        search: str | None = None,
        offset: int = 0,
    ) -> TTSProfilePageSnapshot:
        """Return one repository page using the fixed Slice 2B limit."""

        failed = False
        result = None
        try:
            result = await self._repository.list_profiles(
                search=search,
                limit=_PROFILE_PAGE_LIMIT,
                offset=offset,
            )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        generation, value = self._extract_store_result(result)
        self._require_repository_generation(generation)
        if type(value) is not TTSProfilePage:
            raise ProfileServiceError("operation_failed")
        page = cast(TTSProfilePage, value)
        # Preserve exact-forged Sequence ingress without iterating here; the
        # snapshot's bounded normalizer remains the runtime authority.
        page_profiles = cast(tuple[TTSGenerationProfile, ...], page.profiles)
        validation_failed = False
        snapshot = None
        try:
            snapshot = TTSProfilePageSnapshot(
                repository_generation=generation,
                profiles=page_profiles,
                total=page.total,
            )
        except Exception:  # noqa: BLE001 - hostile results fail closed
            validation_failed = True
        if validation_failed or snapshot is None:
            raise ProfileServiceError("operation_failed")
        self._require_repository_generation(generation)
        return snapshot

    async def bounded_profile_assignment_snapshot(
        self,
    ) -> tuple[tuple[TTSGenerationProfile, int], ...]:
        """Read one bounded complete inventory while profile mutations wait."""

        async with self._consumer_mutation_lock:
            captured: list[tuple[TTSGenerationProfile, int]] = []
            seen_profile_ids: set[UUID] = set()
            expected_total: int | None = None
            expected_generation: int | None = None
            offset = 0
            while len(captured) < _PROFILE_CONSUMER_SNAPSHOT_LIMIT:
                page = await self.list_profiles(search=None, offset=offset)
                if page.total > _PROFILE_CONSUMER_SNAPSHOT_LIMIT:
                    raise ProfileServiceError("operation_failed")
                if expected_total is None:
                    expected_total = page.total
                    expected_generation = page.repository_generation
                elif (
                    page.total != expected_total
                    or page.repository_generation != expected_generation
                ):
                    raise ProfileServiceError("operation_failed")
                if not page.profiles:
                    break
                if len(captured) + len(page.profiles) > page.total:
                    raise ProfileServiceError("operation_failed")
                for profile in page.profiles:
                    if profile.profile_id in seen_profile_ids:
                        raise ProfileServiceError("operation_failed")
                    seen_profile_ids.add(profile.profile_id)
                    loaded = LoadedTTSProfile(page.repository_generation, profile)
                    count = await self.assignment_count(loaded)
                    captured.append((loaded.profile, count))
                offset += len(page.profiles)
                if offset >= page.total:
                    break
            if expected_total is None or len(captured) != expected_total:
                raise ProfileServiceError("operation_failed")
            return tuple(captured)

    async def get_assigned_profile(
        self,
        character_ref: CharacterRef,
    ) -> LoadedCharacterTTSAssignment:
        """Read one exact character assignment and immutable profile revision.

        Args:
            character_ref: Exact source, authority, and character identity.

        Returns:
            The joined assignment/profile snapshot, or unassigned state, paired
            with the repository generation that produced it.

        Raises:
            ProfileValidationError: If the character reference is noncanonical.
            ProfileRepositoryError: If repository state changes or the read
                fails with a bounded repository error.
            ProfileServiceError: If the repository returns malformed state or
                an unexpected collaborator failure occurs.
        """

        canonical_ref = _canonicalize_exact_character_ref(character_ref)
        failed = False
        result = None
        try:
            result = await self._repository.get_assigned_profile(canonical_ref)
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")

        generation, value = self._extract_store_result(result)
        self._require_repository_generation(generation)
        snapshot: AssignedTTSProfileSnapshot | None = None
        validation_failed = False
        try:
            if value is not None:
                snapshot = _canonicalize_exact_assigned_profile(value)
                if snapshot.assignment.character_ref != canonical_ref:
                    raise ValueError
        except Exception:  # noqa: BLE001 - hostile joined values fail closed
            validation_failed = True
        if validation_failed:
            raise ProfileServiceError("operation_failed")
        self._require_repository_generation(generation)
        return LoadedCharacterTTSAssignment(
            repository_generation=generation,
            snapshot=snapshot,
        )

    async def get_profile(self, profile_id: UUID) -> LoadedTTSProfile:
        """Load one exact stored profile revision by id, if it still exists.

        A thin passthrough onto the repository's own `get_profile` (added
        for callers that hold a stored profile id, such as the briefings
        voice resolver, and need the current revision without paging
        `list_profiles`). Mirrors `get_assigned_profile`'s exact shape.

        Args:
            profile_id: Exact profile UUID to load.

        Returns:
            The loaded profile paired with the repository generation that
            produced it.

        Raises:
            ProfileValidationError: If `profile_id` is not an exact `UUID`.
            ProfileRepositoryError: If the profile no longer exists (code
                `"missing"`) or the repository read fails safely.
            ProfileServiceError: If the repository returns malformed state or
                an unexpected collaborator failure occurs.
        """

        canonical_id = _canonicalize_exact_profile_id(profile_id)
        failed = False
        result = None
        try:
            result = await self._repository.get_profile(canonical_id)
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")

        generation, value = self._extract_store_result(result)
        self._require_repository_generation(generation)
        profile: TTSGenerationProfile | None = None
        validation_failed = False
        try:
            profile = _canonicalize_exact_profile(value)
            if profile.profile_id != canonical_id:
                raise ValueError
        except Exception:  # noqa: BLE001 - hostile results fail closed
            validation_failed = True
        if validation_failed or profile is None:
            raise ProfileServiceError("operation_failed")
        self._require_repository_generation(generation)
        return LoadedTTSProfile(
            repository_generation=generation,
            profile=profile,
        )

    async def get_reference(
        self,
        profile_id: UUID,
        *,
        expected_revision: int,
        expected_generation: int,
    ) -> TTSCloneReference:
        """Read and revalidate one exact private reference under store fences."""

        canonical_id = _canonicalize_exact_profile_id(profile_id)
        if type(expected_revision) is not int or expected_revision < 1:
            raise ProfileValidationError("revision")
        generation = _validate_nonnegative_integer(
            expected_generation,
            "generation",
        )
        failed = False
        result = None
        try:
            result = await self._repository.get_reference(
                canonical_id,
                expected_revision=expected_revision,
                expected_generation=generation,
            )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide private collaborator detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")

        value = self._require_admitted_store_result(result, generation)
        reference: TTSCloneReference | None = None
        validation_failed = False
        try:
            reference = _canonicalize_exact_reference(value)
        except Exception:  # noqa: BLE001 - hostile private values fail closed
            validation_failed = True
        if validation_failed or reference is None:
            raise ProfileServiceError("operation_failed")
        self._require_repository_generation(generation)
        return reference

    async def observe_availability(
        self,
        page: TTSProfilePageSnapshot,
    ) -> TTSProfileAvailabilitySnapshot:
        """Observe one bounded capability snapshot for structurally valid rows."""

        if type(page) is not TTSProfilePageSnapshot:
            raise ProfileValidationError("profiles")
        expected_generation = _validate_nonnegative_integer(
            page.repository_generation,
            "generation",
        )
        self._require_repository_generation(expected_generation)
        canonical_page: TTSProfilePageSnapshot | None = None
        validation_failed = False
        try:
            canonical_page = TTSProfilePageSnapshot(
                repository_generation=expected_generation,
                profiles=page.profiles,
                total=page.total,
            )
        except Exception:  # noqa: BLE001 - forged page values fail closed
            validation_failed = True
        if validation_failed or canonical_page is None:
            raise ProfileValidationError("profiles")
        self._require_repository_generation(expected_generation)
        page = canonical_page

        provider_revisions = {
            provider_id: self._current_configuration_revision(provider_id)
            for provider_id in dict.fromkeys(
                profile.provider_id for profile in page.profiles
            )
            if provider_id != _PROFILE_PROVIDER_ID
        }
        audio_cpp_profiles = tuple(
            profile
            for profile in page.profiles
            if profile.provider_id == _PROFILE_PROVIDER_ID
        )
        compatibility_revision: int | None = None
        if not any(
            _profile_is_structurally_supported(profile)
            for profile in audio_cpp_profiles
        ):
            compatibility_revision = self._current_configuration_revision(
                _PROFILE_PROVIDER_ID
            )
            if audio_cpp_profiles:
                provider_revisions[_PROFILE_PROVIDER_ID] = compatibility_revision

        supported_profiles = tuple(
            profile
            for profile in page.profiles
            if _profile_is_structurally_supported(profile)
        )
        if not supported_profiles:
            self._require_repository_generation(page.repository_generation)
            self._require_provider_revisions_unchanged(provider_revisions)
            return TTSProfileAvailabilitySnapshot(
                repository_generation=page.repository_generation,
                configuration_revision=compatibility_revision,
                catalog_revision=None,
                profiles=tuple(
                    _availability(
                        profile.profile_id,
                        "unavailable",
                        profile.provider_id,
                        dependency=_provenance_projection(profile),
                        provider_configuration_revision=provider_revisions[
                            profile.provider_id
                        ],
                    )
                    for profile in page.profiles
                ),
            )

        audio_cpp_supported = tuple(
            profile
            for profile in supported_profiles
            if profile.provider_id == _PROFILE_PROVIDER_ID
        )
        if not audio_cpp_supported:
            self._require_repository_generation(page.repository_generation)
            self._require_provider_revisions_unchanged(provider_revisions)
            return TTSProfileAvailabilitySnapshot(
                repository_generation=page.repository_generation,
                configuration_revision=compatibility_revision,
                catalog_revision=None,
                profiles=tuple(
                    TTSProfileAvailability(
                        profile_id=item.profile_id,
                        state=item.state,
                        recovery_action=item.recovery_action,
                        dependency=_provenance_projection(profile),
                        provider_configuration_revision=(
                            item.provider_configuration_revision
                        ),
                    )
                    for profile, item in (
                        (
                            profile,
                            self._classify_profile_with_evidence(
                                profile,
                                provider_revisions[profile.provider_id],
                            ),
                        )
                        for profile in page.profiles
                    )
                ),
            )

        relevant_models: dict[str, None] = {}
        exact_voice_models: dict[str, None] = {}
        for profile in audio_cpp_supported:
            relevant_models.setdefault(profile.model_id, None)
            if profile.voice_id is not None:
                exact_voice_models.setdefault(profile.model_id, None)

        failed = False
        snapshot = None
        try:
            snapshot = await self._tts_service.get_native_capability_snapshot(
                _PROFILE_PROVIDER_ID,
                tuple(exact_voice_models),
            )
        except Exception:  # noqa: BLE001 - capability detail is not public
            failed = True
        if failed:
            raise ProfileServiceError("operation_failed")
        snapshot = _canonicalize_consumed_capability_snapshot(
            snapshot,
            relevant_model_ids=tuple(relevant_models),
        )
        await self._require_configuration_revision(
            _PROFILE_PROVIDER_ID,
            snapshot.configuration_revision,
        )
        compatibility_revision = snapshot.configuration_revision
        provider_revisions[_PROFILE_PROVIDER_ID] = snapshot.configuration_revision

        projected: list[TTSProfileAvailability] = []
        for profile in page.profiles:
            if profile.provider_id != _PROFILE_PROVIDER_ID:
                evidence_item = self._classify_profile_with_evidence(
                    profile,
                    provider_revisions[profile.provider_id],
                )
                projected.append(
                    TTSProfileAvailability(
                        profile_id=evidence_item.profile_id,
                        state=evidence_item.state,
                        recovery_action=evidence_item.recovery_action,
                        dependency=_provenance_projection(profile),
                        provider_configuration_revision=(
                            evidence_item.provider_configuration_revision
                        ),
                    )
                )
                continue
            base = self._classify_profile(profile, snapshot)
            advisory = _provenance_projection(profile)
            requirement = (
                None
                if profile.reference is None
                else profile.reference.recipe_requirement
            )
            if base.state != "available" or requirement is None:
                projected.append(
                    _availability(
                        profile.profile_id,
                        base.state,
                        profile.provider_id,
                        dependency=advisory,
                        provider_configuration_revision=(
                            snapshot.configuration_revision
                        ),
                    )
                )
                continue
            dependency_failed = False
            dependency: AudioCppGuidedDependencySnapshot | None = None
            try:
                dependency = (
                    await self._tts_service.audio_cpp_guided_dependency_snapshot(
                        requirement
                    )
                )
            except Exception:  # noqa: BLE001 - collaborator detail stays private
                dependency_failed = True
            dependency = validate_audio_cpp_guided_dependency_snapshot(
                dependency,
                requirement,
            )
            if dependency_failed or dependency is None:
                raise ProfileServiceError("operation_failed") from None
            await self._require_configuration_revision(
                _PROFILE_PROVIDER_ID,
                dependency.provider_configuration_revision,
            )
            dependency_projection = _dependency_projection(
                dependency.state,
                advisory=advisory,
            )
            projected.append(
                _availability(
                    profile.profile_id,
                    (
                        "available"
                        if dependency_projection.reason == "none"
                        else "unavailable"
                    ),
                    profile.provider_id,
                    dependency=dependency_projection,
                    provider_configuration_revision=snapshot.configuration_revision,
                )
            )
        availability = tuple(projected)
        self._require_repository_generation(page.repository_generation)
        self._require_provider_revisions_unchanged(provider_revisions)
        assert compatibility_revision is not None
        return TTSProfileAvailabilitySnapshot(
            repository_generation=page.repository_generation,
            configuration_revision=compatibility_revision,
            catalog_revision=(
                None if snapshot.catalog is None else snapshot.catalog.revision
            ),
            profiles=availability,
        )

    async def observe_portable_profile(
        self,
        profile: PortableTTSProfile,
    ) -> PortableProfileAvailabilityObservation:
        """Observe current audio.cpp availability without mutating either store."""

        portable = _canonicalize_exact_portable_profile(profile)
        draft = portable.draft
        if not _selection_is_profile_safe(
            draft.provider_id,
            draft.response_format,
            draft.speed,
            draft.options,
        ):
            raise ProfileServiceError("unsupported_profile")

        repository_generation = self._current_repository_generation()
        if draft.provider_id != _PROFILE_PROVIDER_ID:
            revision = self._current_configuration_revision(_PROFILE_PROVIDER_ID)
            self._require_repository_generation(repository_generation)
            return PortableProfileAvailabilityObservation(
                repository_generation=repository_generation,
                configuration_revision=revision,
                profile=portable,
                availability="unverified",
            )
        exact_voice_models = () if draft.voice_id is None else (draft.model_id,)
        failed = False
        snapshot = None
        try:
            snapshot = await self._tts_service.get_native_capability_snapshot(
                _PROFILE_PROVIDER_ID,
                exact_voice_models,
            )
        except Exception:  # noqa: BLE001 - capability detail is not public
            failed = True
        if failed:
            raise ProfileServiceError("operation_failed")
        snapshot = _canonicalize_consumed_capability_snapshot(
            snapshot,
            relevant_model_ids=(draft.model_id,),
        )
        await self._require_configuration_revision(
            _PROFILE_PROVIDER_ID,
            snapshot.configuration_revision,
        )
        availability: ProfileAvailabilityState = (
            "unverified"
            if snapshot.state != "complete"
            else self._classify_selection(
                provider_id=draft.provider_id,
                model_id=draft.model_id,
                voice_id=draft.voice_id,
                response_format=draft.response_format,
                speed=draft.speed,
                options=draft.options,
                snapshot=snapshot,
            )
        )
        self._require_repository_generation(repository_generation)
        if (
            self._current_configuration_revision(_PROFILE_PROVIDER_ID)
            != snapshot.configuration_revision
        ):
            raise ProfileServiceError("stale_configuration")
        return PortableProfileAvailabilityObservation(
            repository_generation=repository_generation,
            configuration_revision=snapshot.configuration_revision,
            profile=portable,
            availability=availability,
        )

    async def inspect_portable_profile_import(
        self,
        observation: PortableProfileAvailabilityObservation,
    ) -> PortableProfileImportPlan:
        """Classify local UUID/name collisions without writing profile state."""

        if type(observation) is not PortableProfileAvailabilityObservation:
            raise ProfileValidationError("profiles")
        canonical_observation = PortableProfileAvailabilityObservation(
            repository_generation=observation.repository_generation,
            configuration_revision=observation.configuration_revision,
            profile=observation.profile,
            availability=observation.availability,
        )
        expected_generation = canonical_observation.repository_generation
        self._require_repository_generation(expected_generation)
        portable = canonical_observation.profile
        collisions = await self._read_portable_collisions(
            portable,
            expected_generation,
        )

        id_match = collisions.profile_id_match
        name_match = collisions.normalized_name_match
        if id_match is not None and id_match.profile_id != portable.profile_id:
            raise ProfileServiceError("operation_failed")
        if (
            name_match is not None
            and name_match.normalized_name != portable.draft.normalized_name
        ):
            raise ProfileServiceError("operation_failed")
        if (
            id_match is not None
            and name_match is not None
            and id_match.profile_id == name_match.profile_id
            and id_match != name_match
        ):
            raise ProfileServiceError("operation_failed")

        distinct_matches = {
            match.profile_id: match
            for match in (id_match, name_match)
            if match is not None
        }
        reuse_profile: TTSGenerationProfile | None = None
        if len(distinct_matches) == 1:
            only_match = next(iter(distinct_matches.values()))
            if self._generation_fields_match(only_match, portable.draft):
                reuse_profile = only_match

        if not distinct_matches:
            choices: tuple[PortableProfileImportChoice, ...] = ("create",)
            copy_candidate = portable
        else:
            choices = ("reuse", "copy") if reuse_profile is not None else ("copy",)
            copy_candidate = await self._collision_free_copy_candidate(
                portable,
                replace_profile_id=id_match is not None,
                replace_name=name_match is not None,
                expected_generation=expected_generation,
                verify=reuse_profile is None,
            )

        self._require_repository_generation(expected_generation)
        return PortableProfileImportPlan(
            observation=canonical_observation,
            allowed_choices=choices,
            reuse_profile=reuse_profile,
            copy_candidate=copy_candidate,
        )

    async def commit_portable_profile_import(
        self,
        plan: PortableProfileImportPlan,
        choice: PortableProfileImportChoice,
        character_ref: CharacterRef,
        *,
        expected_current: CharacterTTSAssignment | None,
    ) -> PortableProfileImportResult:
        """Commit an inspected choice with current capability and CAS guards."""

        if type(plan) is not PortableProfileImportPlan:
            raise ProfileValidationError("profiles")
        canonical_plan = PortableProfileImportPlan(
            observation=plan.observation,
            allowed_choices=plan.allowed_choices,
            reuse_profile=plan.reuse_profile,
            copy_candidate=plan.copy_candidate,
        )
        if type(choice) is not str or choice not in canonical_plan.allowed_choices:
            raise ProfileValidationError("choice")
        canonical_choice = cast(PortableProfileImportChoice, choice)
        canonical_ref = _canonicalize_exact_character_ref(character_ref)
        expected_assignment = (
            None
            if expected_current is None
            else _canonicalize_exact_assignment(expected_current)
        )
        if (
            expected_assignment is not None
            and expected_assignment.character_ref != canonical_ref
        ):
            raise ProfileValidationError("assignment")

        expected_generation = canonical_plan.observation.repository_generation
        self._require_repository_generation(expected_generation)
        current = await self.observe_portable_profile(
            canonical_plan.observation.profile
        )
        if current.repository_generation != expected_generation:
            raise ProfileRepositoryError("stale")
        self._require_repository_generation(expected_generation)

        if canonical_choice == "reuse":
            profile = canonical_plan.reuse_profile
            if profile is None:
                raise ProfileValidationError("choice")
            loaded = LoadedTTSProfile(expected_generation, profile)
            if current.availability == "unavailable":
                return PortableProfileImportResult(
                    created=False,
                    availability=current.availability,
                    loaded=loaded,
                    assignment=None,
                )
            assignment = await self._set_assignment_after_observation(
                canonical_ref,
                loaded.profile,
                expected_generation=expected_generation,
                expected_current=expected_assignment,
            )
            return PortableProfileImportResult(
                created=False,
                availability="available",
                loaded=loaded,
                assignment=assignment,
            )

        candidate = canonical_plan.copy_candidate
        if canonical_choice == "copy":
            candidate = await self._collision_free_copy_candidate(
                candidate,
                replace_profile_id=False,
                replace_name=False,
                expected_generation=expected_generation,
                verify=True,
            )
        if current.availability != "unavailable":
            snapshot = await self._create_profile_with_assignment(
                candidate,
                canonical_ref,
                expected_generation=expected_generation,
                expected_current=expected_assignment,
            )
            loaded = LoadedTTSProfile(expected_generation, snapshot.profile)
            return PortableProfileImportResult(
                created=True,
                availability="available",
                loaded=loaded,
                assignment=snapshot.assignment,
            )

        loaded = await self._create_portable_profile_unassigned(
            candidate,
            expected_generation=expected_generation,
        )
        return PortableProfileImportResult(
            created=True,
            availability=current.availability,
            loaded=loaded,
            assignment=None,
        )

    async def create_from_artifact(
        self,
        display_name: str,
        artifact: STTSGeneratedAudio,
    ) -> LoadedTTSProfile:
        """Create a profile from immutable successful native provenance only."""

        if (
            type(artifact) is not STTSGeneratedAudio
            or type(artifact.requested_selection) is not TTSRequestedSelectionSnapshot
        ):
            raise ProfileServiceError("artifact_ineligible")
        selection = artifact.requested_selection
        if not _selection_is_profile_safe(
            selection.provider_id,
            selection.response_format,
            selection.speed,
            selection.options,
        ):
            raise ProfileServiceError("unsupported_profile")
        draft = TTSProfileDraft(
            display_name=display_name,
            provider_id=selection.provider_id,
            model_id=selection.model_id,
            voice_id=selection.voice_id,
            response_format=selection.response_format,
            speed=selection.speed,
            options=selection.options,
        )
        await self._require_configuration_revision(
            selection.provider_id,
            selection.configuration_revision,
        )
        repository_generation = self._current_repository_generation()

        failed = False
        result = None
        try:
            async with self._lease_artifact_consumers(
                self._artifact_consumer(draft.provider_id, draft.model_id)
            ):
                result = await self._run_owned_repository_call(
                    self._repository.create_profile(draft)
                )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(
            result,
            repository_generation,
        )
        profile = self._require_profile_mutation_result(
            value,
            draft,
            expected_revision=1,
        )
        self._require_repository_generation(repository_generation)
        loaded = LoadedTTSProfile(
            repository_generation=repository_generation,
            profile=profile,
        )
        self._mark_profile_evidence_current(profile)
        self.record_sample_evidence(loaded, artifact)
        return loaded

    def record_sample_evidence(
        self,
        loaded: LoadedTTSProfile,
        artifact: STTSGeneratedAudio,
    ) -> None:
        """Remember exact successful sample provenance for this process only."""

        if type(artifact) is not STTSGeneratedAudio:
            return
        try:
            profile = self._validate_loaded(loaded)
            with self._sample_evidence_lock:
                lifecycle = self._sample_evidence_lifecycle.get(profile.profile_id)
                if lifecycle is not None and (
                    lifecycle.deleted or lifecycle.revision != profile.revision
                ):
                    return
                admission_epoch = self._sample_evidence_epoch
            selection = artifact.requested_selection
            if type(selection) is not TTSRequestedSelectionSnapshot:
                return
            if not all(
                _matches_exact_canonical_value(source, expected)
                for source, expected in (
                    (selection.provider_id, profile.provider_id),
                    (selection.model_id, profile.model_id),
                    (selection.voice_id, profile.voice_id),
                    (selection.response_format, profile.response_format),
                    (selection.speed, profile.speed),
                    (selection.options, profile.options),
                    (artifact.provider_id, profile.provider_id),
                    (artifact.model_id, profile.model_id),
                    (artifact.voice_id, profile.voice_id),
                )
            ):
                return
            audio_format = artifact.audio_format
            if (
                type(audio_format) is not str
                or audio_format.removeprefix(".") != profile.response_format
            ):
                return
            if (
                validate_playable_audio_file(
                    artifact.path,
                    profile.response_format,
                    artifact.content_type,
                    artifact.metadata,
                )
                is None
            ):
                return
            provider_revision = self._current_configuration_revision(
                profile.provider_id
            )
            if provider_revision != selection.configuration_revision:
                return
            evidence = TTSProfileVerificationEvidence(
                profile_id=profile.profile_id,
                profile_revision=profile.revision,
                provider_id=profile.provider_id,
                model_id=profile.model_id,
                voice_id=profile.voice_id,
                response_format=profile.response_format,
                speed=profile.speed,
                options_fingerprint=profile_options_fingerprint(profile.options),
                provider_configuration_revision=provider_revision,
            )
            if (
                self._current_configuration_revision(profile.provider_id)
                != provider_revision
            ):
                return
        except Exception:  # noqa: BLE001 - malformed artifacts are ineligible
            return

        with self._sample_evidence_lock:
            lifecycle = self._sample_evidence_lifecycle.get(evidence.profile_id)
            if (
                self._sample_evidence_epoch != admission_epoch
                or lifecycle is not None
                and (lifecycle.deleted or lifecycle.revision != profile.revision)
            ):
                return
            # FIFO is intentional: re-recording an existing UUID does not
            # extend its residency; only first admission establishes order.
            self._sample_evidence[evidence.profile_id] = evidence
            while len(self._sample_evidence) > _PROFILE_SAMPLE_EVIDENCE_LIMIT:
                oldest_profile_id = next(iter(self._sample_evidence))
                self._sample_evidence.pop(oldest_profile_id, None)

    async def create_clone_from_artifact(
        self,
        display_name: str,
        artifact: STTSGeneratedAudio,
    ) -> LoadedTTSProfile:
        """Atomically persist one exact successful clone artifact.

        Args:
            display_name: User-selected name for the new profile.
            artifact: Exact retained STTS result carrying clone evidence.

        Returns:
            The committed revision-2 profile and repository generation.

        Raises:
            ProfileServiceError: If the artifact or collaborator result is
                ineligible, incoherent, or unsafe.
            ProfileRepositoryError: If persistence or freshness fails.
            ProfileValidationError: If the selected profile values are invalid.
        """

        if (
            type(artifact) is not STTSGeneratedAudio
            or type(artifact.requested_selection) is not TTSRequestedSelectionSnapshot
            or type(artifact.clone_evidence) is not TTSCloneGenerationEvidence
        ):
            raise ProfileServiceError("artifact_ineligible")
        selection = artifact.requested_selection
        evidence = artifact.clone_evidence
        assert selection is not None
        assert evidence is not None
        if (
            artifact.provider_id != "audio_cpp"
            or artifact.model_id != selection.model_id
            or artifact.voice_id != selection.voice_id
            or artifact.audio_format.removeprefix(".") != selection.response_format
            or selection.provider_id != "audio_cpp"
            or evidence.model_id != selection.model_id
            or evidence.provider_configuration_revision
            != selection.configuration_revision
            or not _selection_is_profile_safe(
                selection.provider_id,
                selection.response_format,
                selection.speed,
                selection.options,
            )
        ):
            raise ProfileServiceError("artifact_ineligible")
        draft = TTSProfileDraft(
            display_name=display_name,
            provider_id=selection.provider_id,
            model_id=selection.model_id,
            voice_id=selection.voice_id,
            response_format=selection.response_format,
            speed=selection.speed,
            options=selection.options,
        )
        await self._require_configuration_revision(
            selection.provider_id,
            evidence.provider_configuration_revision,
        )
        repository_generation = self._current_repository_generation()
        profile_id = self._next_portable_uuid(set())
        recipe_requirement = TTSCloneRecipeRequirement(
            recipe_id=evidence.recipe_id,
            recipe_revision=evidence.recipe_revision,
            model_id=evidence.model_id,
        )

        failed = False
        result = None
        try:
            async with self._lease_artifact_consumers(
                self._artifact_consumer(
                    draft.provider_id,
                    draft.model_id,
                    recipe_requirement,
                )
            ):
                result = await self._run_owned_repository_call(
                    self._repository.create_profile_with_reference(
                        draft,
                        profile_id,
                        evidence.canonical_reference,
                        recipe_requirement,
                        expected_generation=repository_generation,
                    )
                )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(result, repository_generation)
        profile = self._require_profile_mutation_result(
            value,
            draft,
            expected_revision=2,
            required_profile_id=profile_id,
        )
        reference = profile.reference
        canonical = evidence.canonical_reference
        if (
            reference is None
            or reference.byte_length != canonical.byte_length
            or reference.duration_ms != canonical.duration_ms
            or reference.sample_rate_hz != canonical.sample_rate_hz
            or reference.channels != canonical.channels
            or reference.sample_encoding != canonical.sample_encoding
            or reference.recipe_requirement != recipe_requirement
        ):
            raise ProfileServiceError("operation_failed")
        self._require_repository_generation(repository_generation)
        return LoadedTTSProfile(repository_generation, profile)

    async def update_profile(
        self,
        loaded: LoadedTTSProfile,
        draft: TTSProfileDraft,
    ) -> LoadedTTSProfile:
        """Update one exact loaded revision after service-owned validation."""

        loaded_profile = self._validate_loaded_and_draft(loaded, draft)
        self._require_repository_generation(loaded.repository_generation)
        if not _selection_is_profile_safe(
            draft.provider_id,
            draft.response_format,
            draft.speed,
            draft.options,
        ):
            raise ProfileServiceError("unsupported_profile")
        if loaded_profile.reference is not None and not self._generation_fields_match(
            loaded_profile,
            draft,
        ):
            raise ProfileServiceError("operation_failed")
        if not self._generation_fields_match(loaded_profile, draft):
            await self._require_authoritative_capability(draft)

        failed = False
        result = None
        try:
            async with self._lease_artifact_consumers(
                self._artifact_consumer(
                    loaded_profile.provider_id,
                    loaded_profile.model_id,
                    (
                        None
                        if loaded_profile.reference is None
                        else loaded_profile.reference.recipe_requirement
                    ),
                ),
                self._artifact_consumer(draft.provider_id, draft.model_id),
            ):
                result = await self._run_owned_repository_call(
                    self._repository.update_profile(
                        loaded_profile.profile_id,
                        loaded_profile.revision,
                        draft,
                        expected_generation=loaded.repository_generation,
                    )
                )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(
            result,
            loaded.repository_generation,
        )
        profile = self._require_profile_mutation_result(
            value,
            draft,
            expected_revision=loaded_profile.revision + 1,
            required_profile_id=loaded_profile.profile_id,
        )
        self._require_repository_generation(loaded.repository_generation)
        self._mark_profile_evidence_current(profile)
        return LoadedTTSProfile(
            repository_generation=loaded.repository_generation,
            profile=profile,
        )

    async def duplicate_profile(
        self,
        loaded: LoadedTTSProfile,
        display_name: str,
    ) -> LoadedTTSProfile:
        """Copy the immutable loaded version under a new profile identity."""

        source = self._validate_loaded(loaded)
        self._require_repository_generation(loaded.repository_generation)
        draft = TTSProfileDraft(
            display_name=display_name,
            provider_id=source.provider_id,
            model_id=source.model_id,
            voice_id=source.voice_id,
            response_format=source.response_format,
            speed=source.speed,
            options=source.options,
        )
        if not _selection_is_profile_safe(
            draft.provider_id,
            draft.response_format,
            draft.speed,
            draft.options,
        ):
            raise ProfileServiceError("unsupported_profile")
        await self._require_authoritative_capability(draft)

        failed = False
        result = None
        try:
            async with self._lease_artifact_consumers(
                self._artifact_consumer(
                    source.provider_id,
                    source.model_id,
                    (
                        None
                        if source.reference is None
                        else source.reference.recipe_requirement
                    ),
                )
            ):
                result = await self._run_owned_repository_call(
                    self._repository.create_profile(
                        draft,
                        expected_generation=loaded.repository_generation,
                    )
                )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(
            result,
            loaded.repository_generation,
        )
        profile = self._require_profile_mutation_result(
            value,
            draft,
            expected_revision=1,
            forbidden_profile_id=source.profile_id,
        )
        self._require_repository_generation(loaded.repository_generation)
        return LoadedTTSProfile(
            repository_generation=loaded.repository_generation,
            profile=profile,
        )

    async def assignment_count(self, loaded: LoadedTTSProfile) -> int:
        """Return the advisory count only for the loaded store generation."""

        profile = self._validate_loaded(loaded)
        self._require_repository_generation(loaded.repository_generation)
        failed = False
        result = None
        try:
            result = await self._repository.assignment_count(profile.profile_id)
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(
            result,
            loaded.repository_generation,
        )
        if type(value) is not int or value < 0:
            raise ProfileValidationError("assignment_count")
        return value

    async def set_assignment(
        self,
        character_ref: CharacterRef,
        loaded: LoadedTTSProfile,
        expected_current: CharacterTTSAssignment | None,
    ) -> CharacterTTSAssignment:
        """Set one exact character assignment from caller-held profile state.

        Args:
            character_ref: Exact source, authority, and character identity to
                assign.
            loaded: Exact profile and repository generation selected by the
                caller.
            expected_current: Exact assignment observed by the caller, or
                ``None`` when the character was observed as unassigned.

        Returns:
            The exact persisted character assignment.

        Raises:
            ProfileValidationError: If caller-held values are invalid,
                noncanonical, or refer to different characters.
            ProfileRepositoryError: If repository state is stale or rejects
                the compare-and-set mutation.
            ProfileServiceError: If capability authority is unavailable or
                unverified, configuration changes, or a collaborator returns
                an invalid result.
        """

        canonical_ref = _canonicalize_exact_character_ref(character_ref)
        profile = self._validate_loaded(loaded)
        expected_assignment = (
            None
            if expected_current is None
            else _canonicalize_exact_assignment(expected_current)
        )
        if (
            expected_assignment is not None
            and expected_assignment.character_ref != canonical_ref
        ):
            raise ProfileValidationError("assignment")

        repository_generation = loaded.repository_generation
        self._require_repository_generation(repository_generation)
        draft = TTSProfileDraft(
            display_name=profile.display_name,
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
        )
        await self._require_authoritative_capability(draft)
        self._require_repository_generation(repository_generation)

        failed = False
        result = None
        try:
            async with self._lease_artifact_consumers(
                self._artifact_consumer(
                    profile.provider_id,
                    profile.model_id,
                    (
                        None
                        if profile.reference is None
                        else profile.reference.recipe_requirement
                    ),
                )
            ):
                result = await self._run_owned_repository_call(
                    self._repository.set_assignment(
                        canonical_ref,
                        profile.profile_id,
                        expected_generation=repository_generation,
                        expected_profile_revision=profile.revision,
                        expected_current_profile_id=(
                            None
                            if expected_assignment is None
                            else expected_assignment.profile_id
                        ),
                        expected_profile=profile,
                    )
                )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(
            result,
            repository_generation,
        )
        assignment = self._require_assignment_mutation_result(
            value,
            canonical_ref,
            profile.profile_id,
        )
        self._require_repository_generation(repository_generation)
        return assignment

    async def detach_assignment(
        self,
        assignment: CharacterTTSAssignment,
        repository_generation: int,
    ) -> None:
        """Detach one exact caller-held assignment without capability work.

        Args:
            assignment: Exact character and profile assignment observed by the
                caller.
            repository_generation: Exact repository lifecycle generation
                observed with the assignment.

        Raises:
            ProfileValidationError: If caller-held values are invalid or
                noncanonical.
            ProfileRepositoryError: If repository state is stale or the
                observed assignment has been replaced.
            ProfileServiceError: If the repository returns an invalid result.
        """

        canonical_assignment = _canonicalize_exact_assignment(assignment)
        expected_generation = _validate_nonnegative_integer(
            repository_generation,
            "generation",
        )
        self._require_repository_generation(expected_generation)

        loaded_profile: TTSGenerationProfile | None = None
        failed = False
        try:
            loaded_result = await self._repository.get_profile(
                canonical_assignment.profile_id
            )
            loaded_value = self._require_admitted_store_result(
                loaded_result,
                expected_generation,
            )
            loaded_profile = _canonicalize_exact_profile(loaded_value)
            if loaded_profile.profile_id != canonical_assignment.profile_id:
                raise ProfileValidationError("assignment")
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or loaded_profile is None:
            raise ProfileServiceError("operation_failed")

        failed = False
        result = None
        try:
            async with self._lease_artifact_consumers(
                self._artifact_consumer(
                    loaded_profile.provider_id,
                    loaded_profile.model_id,
                    (
                        None
                        if loaded_profile.reference is None
                        else loaded_profile.reference.recipe_requirement
                    ),
                )
            ):
                result = await self._run_owned_repository_call(
                    self._repository.remove_assignment(
                        canonical_assignment.character_ref,
                        expected_generation=expected_generation,
                        expected_profile_id=canonical_assignment.profile_id,
                    )
                )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(
            result,
            expected_generation,
        )
        if value is not None:
            raise ProfileServiceError("operation_failed")
        self._require_repository_generation(expected_generation)

    async def delete_profile(self, loaded: LoadedTTSProfile) -> None:
        """Delete one loaded profile while retaining repository protection."""

        profile = self._validate_loaded(loaded)
        self._require_repository_generation(loaded.repository_generation)
        failed = False
        result = None
        try:
            async with self._lease_artifact_consumers(
                self._artifact_consumer(
                    profile.provider_id,
                    profile.model_id,
                    (
                        None
                        if profile.reference is None
                        else profile.reference.recipe_requirement
                    ),
                )
            ):
                result = await self._run_owned_repository_call(
                    self._repository.delete_profile(
                        profile.profile_id,
                        expected_generation=loaded.repository_generation,
                    )
                )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(
            result,
            loaded.repository_generation,
        )
        if value is not None:
            raise ProfileServiceError("operation_failed")
        self._require_repository_generation(loaded.repository_generation)
        self._mark_profile_evidence_deleted(profile)

    async def _read_portable_collisions(
        self,
        portable: PortableTTSProfile,
        expected_generation: int,
    ) -> TTSProfileCollisionSnapshot:
        failed = False
        result = None
        repository = self._require_portable_repository()
        try:
            result = await repository.get_profile_collisions(
                portable.profile_id,
                portable.draft,
            )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(result, expected_generation)
        try:
            collisions = _canonicalize_exact_collision_snapshot(value)
        except ProfileValidationError:
            raise ProfileServiceError("operation_failed") from None
        if (
            collisions.profile_id_match is not None
            and collisions.profile_id_match.profile_id != portable.profile_id
        ):
            raise ProfileServiceError("operation_failed")
        if (
            collisions.normalized_name_match is not None
            and collisions.normalized_name_match.normalized_name
            != portable.draft.normalized_name
        ):
            raise ProfileServiceError("operation_failed")
        return collisions

    def _next_portable_uuid(self, disallowed: set[UUID]) -> UUID:
        for _ in range(32):
            failed = False
            candidate: object = None
            try:
                factory = getattr(self, "_uuid_factory", uuid4)
                candidate = factory()
            except Exception:  # noqa: BLE001 - factory detail is not public
                failed = True
            if failed:
                raise ProfileServiceError("operation_failed")
            if type(candidate) is UUID and candidate not in disallowed:
                return candidate
        raise ProfileServiceError("operation_failed")

    @staticmethod
    def _portable_copy_name(display_name: str, index: int) -> str:
        suffix = " (imported)" if index == 1 else f" (imported {index})"
        base = display_name[: 128 - len(suffix)].rstrip()
        return f"{base}{suffix}"

    async def _collision_free_copy_candidate(
        self,
        portable: PortableTTSProfile,
        *,
        replace_profile_id: bool,
        replace_name: bool,
        expected_generation: int,
        verify: bool,
    ) -> PortableTTSProfile:
        used_ids = {portable.profile_id}
        profile_id = (
            self._next_portable_uuid(used_ids)
            if replace_profile_id
            else portable.profile_id
        )
        used_ids.add(profile_id)
        name_index = 1
        display_name = (
            self._portable_copy_name(portable.draft.display_name, name_index)
            if replace_name
            else portable.draft.display_name
        )

        if not verify:
            return PortableTTSProfile(
                profile_id=profile_id,
                draft=TTSProfileDraft(
                    display_name=display_name,
                    provider_id=portable.draft.provider_id,
                    model_id=portable.draft.model_id,
                    voice_id=portable.draft.voice_id,
                    response_format=portable.draft.response_format,
                    speed=portable.draft.speed,
                    options=portable.draft.options,
                ),
            )

        for _ in range(32):
            draft = TTSProfileDraft(
                display_name=display_name,
                provider_id=portable.draft.provider_id,
                model_id=portable.draft.model_id,
                voice_id=portable.draft.voice_id,
                response_format=portable.draft.response_format,
                speed=portable.draft.speed,
                options=portable.draft.options,
            )
            candidate = PortableTTSProfile(profile_id=profile_id, draft=draft)
            collisions = await self._read_portable_collisions(
                candidate,
                expected_generation,
            )
            if (
                collisions.profile_id_match is None
                and collisions.normalized_name_match is None
            ):
                return candidate
            if collisions.profile_id_match is not None:
                profile_id = self._next_portable_uuid(used_ids)
                used_ids.add(profile_id)
            if collisions.normalized_name_match is not None:
                name_index += 1
                display_name = self._portable_copy_name(
                    portable.draft.display_name,
                    name_index,
                )
        raise ProfileServiceError("operation_failed")

    async def _create_portable_profile_unassigned(
        self,
        portable: PortableTTSProfile,
        *,
        expected_generation: int,
    ) -> LoadedTTSProfile:
        failed = False
        result = None
        try:
            async with self._lease_artifact_consumers(
                self._artifact_consumer(
                    portable.draft.provider_id,
                    portable.draft.model_id,
                )
            ):
                result = await self._run_owned_repository_call(
                    self._repository.create_profile(
                        portable.draft,
                        portable.profile_id,
                        expected_generation=expected_generation,
                    )
                )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(result, expected_generation)
        profile = self._require_profile_mutation_result(
            value,
            portable.draft,
            expected_revision=1,
            required_profile_id=portable.profile_id,
        )
        return LoadedTTSProfile(expected_generation, profile)

    async def _create_profile_with_assignment(
        self,
        portable: PortableTTSProfile,
        character_ref: CharacterRef,
        *,
        expected_generation: int,
        expected_current: CharacterTTSAssignment | None,
    ) -> AssignedTTSProfileSnapshot:
        failed = False
        result = None
        repository = self._require_portable_repository()
        try:
            async with self._lease_artifact_consumers(
                self._artifact_consumer(
                    portable.draft.provider_id,
                    portable.draft.model_id,
                )
            ):
                result = await self._run_owned_repository_call(
                    repository.create_profile_with_assignment(
                        portable.draft,
                        portable.profile_id,
                        character_ref,
                        expected_generation=expected_generation,
                        expected_current_profile_id=(
                            None
                            if expected_current is None
                            else expected_current.profile_id
                        ),
                    )
                )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(result, expected_generation)
        try:
            snapshot = _canonicalize_exact_assigned_profile(value)
        except ProfileValidationError:
            raise ProfileServiceError("operation_failed") from None
        profile = self._require_profile_mutation_result(
            snapshot.profile,
            portable.draft,
            expected_revision=1,
            required_profile_id=portable.profile_id,
        )
        assignment = self._require_assignment_mutation_result(
            snapshot.assignment,
            character_ref,
            profile.profile_id,
        )
        return AssignedTTSProfileSnapshot(assignment=assignment, profile=profile)

    async def _set_assignment_after_observation(
        self,
        character_ref: CharacterRef,
        profile: TTSGenerationProfile,
        *,
        expected_generation: int,
        expected_current: CharacterTTSAssignment | None,
    ) -> CharacterTTSAssignment:
        failed = False
        result = None
        try:
            async with self._lease_artifact_consumers(
                self._artifact_consumer(
                    profile.provider_id,
                    profile.model_id,
                    (
                        None
                        if profile.reference is None
                        else profile.reference.recipe_requirement
                    ),
                )
            ):
                result = await self._run_owned_repository_call(
                    self._repository.set_assignment(
                        character_ref,
                        profile.profile_id,
                        expected_generation=expected_generation,
                        expected_profile_revision=profile.revision,
                        expected_current_profile_id=(
                            None
                            if expected_current is None
                            else expected_current.profile_id
                        ),
                        expected_profile=profile,
                    )
                )
        except (ProfileRepositoryError, ProfileValidationError):
            raise
        except Exception:  # noqa: BLE001 - hide unexpected repository detail
            failed = True
        if failed or result is None:
            raise ProfileServiceError("operation_failed")
        value = self._require_admitted_store_result(result, expected_generation)
        return self._require_assignment_mutation_result(
            value,
            character_ref,
            profile.profile_id,
        )

    def preview_preset(
        self,
        loaded: LoadedTTSProfile,
        availability: TTSProfileAvailability,
    ) -> TTSPlaygroundSelectionPreset:
        """Copy persisted generation values into one exact no-synthesis preset."""

        profile = self._validate_loaded(loaded)
        if type(availability) is not TTSProfileAvailability:
            raise ProfileValidationError("availability")
        if availability.profile_id != profile.profile_id:
            raise ProfileValidationError("profile_id")
        effective_availability = (
            availability.state
            if _profile_is_structurally_supported(profile)
            else "unavailable"
        )
        return TTSPlaygroundSelectionPreset(
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
            availability=effective_availability,
            profile_id=(profile.profile_id if profile.reference is not None else None),
            repository_generation=(
                loaded.repository_generation if profile.reference is not None else None
            ),
            profile_revision=(
                profile.revision if profile.reference is not None else None
            ),
        )

    @staticmethod
    def _extract_store_result(result: object) -> tuple[int, object]:
        if type(result) is not ProfileStoreResult:
            raise ProfileServiceError("operation_failed")
        store_result = cast(ProfileStoreResult[object], result)
        extraction_failed = False
        generation = 0
        value: object = None
        try:
            generation = store_result.generation
            value = store_result.value
        except Exception:  # noqa: BLE001 - hostile results fail closed
            extraction_failed = True
        if extraction_failed or type(generation) is not int or generation < 0:
            raise ProfileServiceError("operation_failed")
        return generation, value

    def _current_repository_generation(self) -> int:
        read_failed = False
        generation = 0
        try:
            generation = self._repository.generation
        except Exception:  # noqa: BLE001 - hostile collaborators fail closed
            read_failed = True
        if read_failed or type(generation) is not int or generation < 0:
            raise ProfileServiceError("operation_failed")
        return generation

    def _require_repository_generation(self, expected_generation: int) -> None:
        if self._current_repository_generation() != expected_generation:
            raise ProfileRepositoryError("stale")

    def _require_admitted_store_result(
        self,
        result: object,
        expected_generation: int,
    ) -> object:
        result_generation, value = self._extract_store_result(result)
        self._require_repository_generation(expected_generation)
        if result_generation != expected_generation:
            raise ProfileServiceError("operation_failed")
        return value

    @classmethod
    def _require_profile_mutation_result(
        cls,
        value: object,
        draft: TTSProfileDraft,
        *,
        expected_revision: int,
        required_profile_id: UUID | None = None,
        forbidden_profile_id: UUID | None = None,
    ) -> TTSGenerationProfile:
        profile: TTSGenerationProfile | None = None
        failed = False
        valid = False
        try:
            profile = _canonicalize_exact_profile(value)
            valid = (
                _matches_exact_canonical_value(
                    draft.display_name,
                    profile.display_name,
                )
                and _matches_exact_canonical_value(
                    draft.normalized_name,
                    profile.normalized_name,
                )
                and cls._generation_fields_match(profile, draft)
                and profile.revision == expected_revision
                and (
                    required_profile_id is None
                    or profile.profile_id == required_profile_id
                )
                and (
                    forbidden_profile_id is None
                    or profile.profile_id != forbidden_profile_id
                )
            )
        except Exception:  # noqa: BLE001 - hostile results fail closed
            failed = True
        if failed or not valid or profile is None:
            raise ProfileServiceError("operation_failed")
        return profile

    @staticmethod
    def _require_assignment_mutation_result(
        value: object,
        character_ref: CharacterRef,
        profile_id: UUID,
    ) -> CharacterTTSAssignment:
        assignment: CharacterTTSAssignment | None = None
        failed = False
        valid = False
        try:
            assignment = _canonicalize_exact_assignment(value)
            valid = (
                assignment.character_ref == character_ref
                and assignment.profile_id == profile_id
            )
        except Exception:  # noqa: BLE001 - hostile results fail closed
            failed = True
        if failed or not valid or assignment is None:
            raise ProfileServiceError("operation_failed")
        return assignment

    def _current_configuration_revision(self, provider_id: str) -> int:
        """Return one provider's active runtime revision, not publication state."""

        failed = False
        revision = None
        try:
            revision = self._tts_service.configuration_revision(provider_id)
        except Exception:  # noqa: BLE001 - configuration detail is not public
            failed = True
        if failed or type(revision) is not int or revision < 0:
            raise ProfileServiceError("operation_failed")
        return revision

    def _require_provider_revisions_unchanged(
        self,
        expected: Mapping[str, int],
    ) -> None:
        for provider_id, revision in expected.items():
            if self._current_configuration_revision(provider_id) != revision:
                raise ProfileServiceError("stale_configuration")

    async def _require_configuration_revision(
        self,
        provider_id: str,
        expected_revision: int,
    ) -> None:
        stale = False
        failed = False
        try:
            await self._tts_service.require_current_configuration_revision(
                provider_id,
                expected_revision,
            )
        except TTSConfigurationRevisionError:
            stale = True
        except Exception:  # noqa: BLE001 - configuration detail is not public
            failed = True
        if stale:
            raise ProfileServiceError("stale_configuration")
        if failed:
            raise ProfileServiceError("operation_failed")

    async def _require_authoritative_capability(
        self,
        draft: TTSProfileDraft,
    ) -> None:
        if draft.provider_id != _PROFILE_PROVIDER_ID:
            return
        exact_voice_models = () if draft.voice_id is None else (draft.model_id,)
        failed = False
        snapshot = None
        try:
            snapshot = await self._tts_service.get_native_capability_snapshot(
                _PROFILE_PROVIDER_ID,
                exact_voice_models,
            )
        except Exception:  # noqa: BLE001 - capability detail is not public
            failed = True
        if failed:
            raise ProfileServiceError("operation_failed")
        snapshot = _canonicalize_consumed_capability_snapshot(
            snapshot,
            relevant_model_ids=(draft.model_id,),
        )
        if snapshot.state != "complete":
            raise ProfileServiceError("profile_unverified")
        await self._require_configuration_revision(
            _PROFILE_PROVIDER_ID,
            snapshot.configuration_revision,
        )
        state = self._classify_selection(
            provider_id=draft.provider_id,
            model_id=draft.model_id,
            voice_id=draft.voice_id,
            response_format=draft.response_format,
            speed=draft.speed,
            options=draft.options,
            snapshot=snapshot,
        )
        if state == "unavailable":
            raise ProfileServiceError("profile_unavailable")
        if state != "available":
            raise ProfileServiceError("profile_unverified")

    @staticmethod
    def _validate_loaded(loaded: LoadedTTSProfile) -> TTSGenerationProfile:
        if type(loaded) is not LoadedTTSProfile:
            raise ProfileValidationError("profiles")
        _validate_nonnegative_integer(
            loaded.repository_generation,
            "generation",
        )
        return _canonicalize_exact_profile(loaded.profile)

    @classmethod
    def _validate_loaded_and_draft(
        cls,
        loaded: LoadedTTSProfile,
        draft: TTSProfileDraft,
    ) -> TTSGenerationProfile:
        profile = cls._validate_loaded(loaded)
        if type(draft) is not TTSProfileDraft:
            raise ProfileValidationError("profiles")
        return profile

    @staticmethod
    def _generation_fields_match(
        profile: TTSGenerationProfile,
        draft: TTSProfileDraft,
    ) -> bool:
        return all(
            _matches_exact_canonical_value(source, expected)
            for source, expected in (
                (draft.provider_id, profile.provider_id),
                (draft.model_id, profile.model_id),
                (draft.voice_id, profile.voice_id),
                (draft.response_format, profile.response_format),
                (draft.speed, profile.speed),
                (draft.options, profile.options),
            )
        )

    def _mark_profile_evidence_current(self, profile: TTSGenerationProfile) -> None:
        with self._sample_evidence_lock:
            self._sample_evidence_epoch += 1
            self._sample_evidence_lifecycle[profile.profile_id] = (
                _ProfileEvidenceLifecycle(profile.revision, False)
            )
            self._sample_evidence.pop(profile.profile_id, None)

    def _mark_profile_evidence_deleted(self, profile: TTSGenerationProfile) -> None:
        with self._sample_evidence_lock:
            self._sample_evidence_epoch += 1
            self._sample_evidence_lifecycle[profile.profile_id] = (
                _ProfileEvidenceLifecycle(profile.revision, True)
            )
            self._sample_evidence.pop(profile.profile_id, None)

    def _classify_profile_with_evidence(
        self,
        profile: TTSGenerationProfile,
        provider_configuration_revision: int,
    ) -> TTSProfileAvailability:
        if not _profile_is_structurally_supported(profile):
            state: ProfileAvailabilityState = "unavailable"
        else:
            expected = TTSProfileVerificationEvidence(
                profile_id=profile.profile_id,
                profile_revision=profile.revision,
                provider_id=profile.provider_id,
                model_id=profile.model_id,
                voice_id=profile.voice_id,
                response_format=profile.response_format,
                speed=profile.speed,
                options_fingerprint=profile_options_fingerprint(profile.options),
                provider_configuration_revision=provider_configuration_revision,
            )
            with self._sample_evidence_lock:
                lifecycle = self._sample_evidence_lifecycle.get(profile.profile_id)
                evidence = (
                    None
                    if lifecycle is not None
                    and (lifecycle.deleted or lifecycle.revision != profile.revision)
                    else self._sample_evidence.get(profile.profile_id)
                )
                if evidence is not None and evidence != expected:
                    self._sample_evidence.pop(profile.profile_id, None)
                    evidence = None
            state = "available" if evidence == expected else "unverified"
        return _availability(
            profile.profile_id,
            state,
            profile.provider_id,
            provider_configuration_revision=provider_configuration_revision,
        )

    @staticmethod
    def _classify_profile(
        profile: TTSGenerationProfile,
        snapshot: TTSNativeCapabilitySnapshot,
    ) -> TTSProfileAvailability:
        state = TTSProfileService._classify_selection(
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
            snapshot=snapshot,
        )
        return _availability(profile.profile_id, state, profile.provider_id)

    @staticmethod
    def _classify_selection(
        *,
        provider_id: object,
        model_id: object,
        voice_id: object,
        response_format: object,
        speed: object,
        options: object,
        snapshot: TTSNativeCapabilitySnapshot,
    ) -> ProfileAvailabilityState:
        if not _selection_is_profile_safe(
            provider_id,
            response_format,
            speed,
            options,
        ):
            return "unavailable"
        if provider_id != _PROFILE_PROVIDER_ID:
            return "unverified"
        catalog = snapshot.catalog
        if (
            type(catalog) is not TTSProviderCatalog
            or type(catalog.health) is not ProviderHealth
            or not catalog.health.fresh
        ):
            return "unverified"
        if catalog.health.state == "reconfiguring":
            return "unverified"
        if catalog.health.state != "available":
            return "unavailable"

        model = next(
            (
                candidate
                for candidate in catalog.models
                if candidate.model_id == model_id
            ),
            None,
        )
        if model is None:
            return "unavailable"
        if response_format not in model.formats:
            return "unavailable"
        if voice_id is None:
            return (
                "available" if model.omit_voice_uses_server_default else "unavailable"
            )

        voice_result = snapshot.voice_results.get(cast(str, model_id))
        if type(voice_result) is not TTSVoiceDiscoveryResult:
            return "unverified"
        if voice_result.state == "complete":
            return "available" if voice_id in voice_result.voices else "unavailable"
        if voice_result.state == "model_missing":
            return "unavailable"
        return "unverified"
