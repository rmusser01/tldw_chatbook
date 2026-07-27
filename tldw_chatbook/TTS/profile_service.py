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
_PROFILE_RESPONSE_FORMAT = "wav"
_PROFILE_SPEED = 1.0
_PROFILE_PAGE_LIMIT = 50
_TTS_GENERATION_PROFILE_TYPE: type[TTSGenerationProfile] = TTSGenerationProfile
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
        and response_format == _PROFILE_RESPONSE_FORMAT
        and type(speed) is float
        and speed == _PROFILE_SPEED
        and _mapping_is_empty(options)
    )


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
        total = _validate_nonnegative_integer(self.total, "total")
        if total < len(profiles):
            raise ProfileValidationError("total")
        object.__setattr__(self, "repository_generation", generation)
        object.__setattr__(self, "profiles", profiles)
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
        if type(self.profile) is not TTSGenerationProfile:
            raise ProfileValidationError("profiles")
        object.__setattr__(self, "repository_generation", generation)


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
        validation_failed = False
        snapshot = None
        try:
            snapshot = TTSProfilePageSnapshot(
                repository_generation=generation,
                profiles=page.profiles,
                total=page.total,
            )
        except Exception:  # noqa: BLE001 - hostile results fail closed
            validation_failed = True
        if validation_failed or snapshot is None:
            raise ProfileServiceError("operation_failed")
        return snapshot

    async def observe_availability(
        self,
        page: TTSProfilePageSnapshot,
    ) -> TTSProfileAvailabilitySnapshot:
        """Observe one bounded capability snapshot for structurally valid rows."""

        if type(page) is not TTSProfilePageSnapshot:
            raise ProfileValidationError("profiles")
        self._require_repository_generation(page.repository_generation)

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

        exact_voice_models: dict[str, None] = {}
        for profile in supported_profiles:
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
        if failed or type(snapshot) is not TTSNativeCapabilitySnapshot:
            raise ProfileServiceError("operation_failed")

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

        self._validate_loaded_and_draft(loaded, draft)
        self._require_repository_generation(loaded.repository_generation)
        if not _selection_is_profile_safe(
            draft.provider_id,
            draft.response_format,
            draft.speed,
            draft.options,
        ):
            raise ProfileServiceError("unsupported_profile")
        if not self._generation_fields_match(loaded.profile, draft):
            await self._require_authoritative_capability(draft)

        failed = False
        result = None
        try:
            result = await self._repository.update_profile(
                loaded.profile.profile_id,
                loaded.profile.revision,
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
            expected_revision=loaded.profile.revision + 1,
            required_profile_id=loaded.profile.profile_id,
        )
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

        self._validate_loaded(loaded)
        self._require_repository_generation(loaded.repository_generation)
        source = loaded.profile
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
        return LoadedTTSProfile(
            repository_generation=loaded.repository_generation,
            profile=profile,
        )

    async def assignment_count(self, loaded: LoadedTTSProfile) -> int:
        """Return the advisory count only for the loaded store generation."""

        self._validate_loaded(loaded)
        self._require_repository_generation(loaded.repository_generation)
        failed = False
        result = None
        try:
            result = await self._repository.assignment_count(loaded.profile.profile_id)
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

    async def delete_profile(self, loaded: LoadedTTSProfile) -> None:
        """Delete one loaded profile while retaining repository protection."""

        self._validate_loaded(loaded)
        self._require_repository_generation(loaded.repository_generation)
        failed = False
        result = None
        try:
            result = await self._repository.delete_profile(
                loaded.profile.profile_id,
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

        self._validate_loaded(loaded)
        if type(availability) is not TTSProfileAvailability:
            raise ProfileValidationError("availability")
        if availability.profile_id != loaded.profile.profile_id:
            raise ProfileValidationError("profile_id")
        profile = loaded.profile
        return TTSPlaygroundSelectionPreset(
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
            availability=availability.state,
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
        if type(value) is not _TTS_GENERATION_PROFILE_TYPE:
            raise ProfileServiceError("operation_failed")
        profile = cast(TTSGenerationProfile, value)
        validation_failed = False
        valid = False
        try:
            revalidated = TTSGenerationProfile(
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
            valid = (
                revalidated == profile
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
            validation_failed = True
        if validation_failed or not valid:
            raise ProfileServiceError("operation_failed")
        return profile

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
        if failed or type(snapshot) is not TTSNativeCapabilitySnapshot:
            raise ProfileServiceError("operation_failed")
        if snapshot.state != "complete":
            raise ProfileServiceError("profile_unverified")
        if snapshot.provider_id != _PROFILE_PROVIDER_ID:
            raise ProfileServiceError("operation_failed")
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
    def _validate_loaded(loaded: LoadedTTSProfile) -> None:
        if type(loaded) is not LoadedTTSProfile:
            raise ProfileValidationError("profiles")

    @classmethod
    def _validate_loaded_and_draft(
        cls,
        loaded: LoadedTTSProfile,
        draft: TTSProfileDraft,
    ) -> None:
        cls._validate_loaded(loaded)
        if type(draft) is not TTSProfileDraft:
            raise ProfileValidationError("profiles")

    @staticmethod
    def _generation_fields_match(
        profile: TTSGenerationProfile,
        draft: TTSProfileDraft,
    ) -> bool:
        return (
            profile.provider_id == draft.provider_id
            and profile.model_id == draft.model_id
            and profile.voice_id == draft.voice_id
            and profile.response_format == draft.response_format
            and profile.speed == draft.speed
            and profile.options == draft.options
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
