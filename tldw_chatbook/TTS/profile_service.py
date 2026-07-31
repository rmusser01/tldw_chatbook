"""Native-only TTS generation-profile service values and operations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, cast, runtime_checkable
from uuid import UUID

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
from tldw_chatbook.TTS.profile_types import (
    AUDIO_CPP_PROFILE_RESPONSE_FORMAT,
    AUDIO_CPP_PROFILE_SPEED,
    CharacterRef,
    CharacterTTSAssignment,
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileDraft,
    TTSProfilePage,
)

ProfileAvailabilityState: TypeAlias = Literal[
    "available",
    "unavailable",
    "unverified",
]
ProfileRecoveryAction: TypeAlias = Literal["none", "refresh", "edit"]

_PROFILE_PROVIDER_ID = "audio_cpp"
_PROFILE_PAGE_LIMIT = 50
_CHARACTER_REF_TYPE: type[CharacterRef] = CharacterRef
_CHARACTER_TTS_ASSIGNMENT_TYPE: type[CharacterTTSAssignment] = CharacterTTSAssignment
_TTS_GENERATION_PROFILE_TYPE: type[TTSGenerationProfile] = TTSGenerationProfile
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
    ) -> ProfileStoreResult[CharacterTTSAssignment]: ...

    async def remove_assignment(
        self,
        character_ref: CharacterRef,
        *,
        expected_generation: int,
        expected_profile_id: UUID,
    ) -> ProfileStoreResult[None]: ...


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
    if action != _AVAILABILITY_RECOVERY[state]:
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
    return (
        type(provider_id) is str
        and provider_id == _PROFILE_PROVIDER_ID
        and type(response_format) is str
        and response_format == AUDIO_CPP_PROFILE_RESPONSE_FORMAT
        and type(speed) is float
        and speed == AUDIO_CPP_PROFILE_SPEED
        and _mapping_is_empty(options)
    )


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
            )
        )
    except Exception:  # noqa: BLE001 - hostile profile values fail closed
        failed = True
    if failed or not valid or canonical is None:
        raise ProfileValidationError("profiles")
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


def _availability(
    profile_id: UUID,
    state: ProfileAvailabilityState,
) -> TTSProfileAvailability:
    return TTSProfileAvailability(
        profile_id=profile_id,
        state=state,
        recovery_action=_AVAILABILITY_RECOVERY[state],
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
class TTSProfileAvailability:
    """The current bounded availability state for one exact profile UUID."""

    profile_id: UUID
    state: ProfileAvailabilityState
    recovery_action: ProfileRecoveryAction

    def __post_init__(self) -> None:
        if type(self.profile_id) is not UUID:
            raise ProfileValidationError("profile_id")
        state = _validate_availability_state(self.state)
        action = _validate_recovery_action(self.recovery_action, state)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "recovery_action", action)


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


class TTSProfileService:
    """Manage native audio.cpp profiles over existing app-owned dependencies."""

    def __init__(
        self,
        repository: _ProfileRepositoryProtocol,
        tts_service: _ProfileTTSServiceProtocol,
    ) -> None:
        validation_failed = False
        try:
            if not isinstance(repository, _ProfileRepositoryProtocol) or not isinstance(
                tts_service,
                _ProfileTTSServiceProtocol,
            ):
                validation_failed = True
        except Exception:  # noqa: BLE001 - hostile collaborators fail closed
            validation_failed = True
        if validation_failed:
            raise ProfileServiceError("operation_failed")
        self._repository = repository
        self._tts_service = tts_service

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

        supported_profiles = tuple(
            profile
            for profile in page.profiles
            if _profile_is_structurally_supported(profile)
        )
        if not supported_profiles:
            revision = self._current_configuration_revision()
            self._require_repository_generation(page.repository_generation)
            return TTSProfileAvailabilitySnapshot(
                repository_generation=page.repository_generation,
                configuration_revision=revision,
                catalog_revision=None,
                profiles=tuple(
                    _availability(profile.profile_id, "unavailable")
                    for profile in page.profiles
                ),
            )

        relevant_models: dict[str, None] = {}
        exact_voice_models: dict[str, None] = {}
        for profile in supported_profiles:
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

        availability = tuple(
            self._classify_profile(profile, snapshot) for profile in page.profiles
        )
        self._require_repository_generation(page.repository_generation)
        if self._current_configuration_revision() != snapshot.configuration_revision:
            raise ProfileServiceError("stale_configuration")
        return TTSProfileAvailabilitySnapshot(
            repository_generation=page.repository_generation,
            configuration_revision=snapshot.configuration_revision,
            catalog_revision=(
                None if snapshot.catalog is None else snapshot.catalog.revision
            ),
            profiles=availability,
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
            result = await self._repository.create_profile(draft)
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
        return LoadedTTSProfile(
            repository_generation=repository_generation,
            profile=profile,
        )

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
        if not self._generation_fields_match(loaded_profile, draft):
            await self._require_authoritative_capability(draft)

        failed = False
        result = None
        try:
            result = await self._repository.update_profile(
                loaded_profile.profile_id,
                loaded_profile.revision,
                draft,
                expected_generation=loaded.repository_generation,
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
            result = await self._repository.create_profile(
                draft,
                expected_generation=loaded.repository_generation,
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
        """Set one exact character assignment from caller-held profile state."""

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
            result = await self._repository.set_assignment(
                canonical_ref,
                profile.profile_id,
                expected_generation=repository_generation,
                expected_profile_revision=profile.revision,
                expected_current_profile_id=(
                    None
                    if expected_assignment is None
                    else expected_assignment.profile_id
                ),
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
        """Detach one exact caller-held assignment without capability work."""

        canonical_assignment = _canonicalize_exact_assignment(assignment)
        expected_generation = _validate_nonnegative_integer(
            repository_generation,
            "generation",
        )
        self._require_repository_generation(expected_generation)

        failed = False
        result = None
        try:
            result = await self._repository.remove_assignment(
                canonical_assignment.character_ref,
                expected_generation=expected_generation,
                expected_profile_id=canonical_assignment.profile_id,
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
            result = await self._repository.delete_profile(
                profile.profile_id,
                expected_generation=loaded.repository_generation,
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

    def _current_configuration_revision(self) -> int:
        failed = False
        revision = None
        try:
            revision = self._tts_service.configuration_revision(_PROFILE_PROVIDER_ID)
        except Exception:  # noqa: BLE001 - configuration detail is not public
            failed = True
        if failed or type(revision) is not int or revision < 0:
            raise ProfileServiceError("operation_failed")
        return revision

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
        return _availability(profile.profile_id, state)

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
