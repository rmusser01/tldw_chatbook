"""Native-only TTS generation-profile service values and operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, TypeVar, cast
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
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.TTS.profile_types import (
    TTSGenerationProfile,
    TTSProfileDraft,
)
from tldw_chatbook.TTS.TTS_Generation import TTSService

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
        repository: TTSProfileRepository,
        tts_service: TTSService,
    ) -> None:
        if not isinstance(repository, TTSProfileRepository):
            repository_methods = (
                "assignment_count",
                "create_profile",
                "delete_profile",
                "list_profiles",
                "update_profile",
            )
            if not all(
                callable(getattr(repository, name, None)) for name in repository_methods
            ):
                raise ProfileServiceError("operation_failed")
        if not isinstance(tts_service, TTSService):
            service_methods = (
                "configuration_revision",
                "get_native_capability_snapshot",
                "require_current_configuration_revision",
            )
            if not all(
                callable(getattr(tts_service, name, None)) for name in service_methods
            ):
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
        if result.generation != self._repository.generation:
            raise ProfileRepositoryError("stale")
        return TTSProfilePageSnapshot(
            repository_generation=result.generation,
            profiles=result.value.profiles,
            total=result.value.total,
        )

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
        if result.generation != self._repository.generation:
            raise ProfileRepositoryError("stale")
        return LoadedTTSProfile(
            repository_generation=result.generation,
            profile=result.value,
        )

    async def update_profile(
        self,
        loaded: LoadedTTSProfile,
        draft: TTSProfileDraft,
    ) -> LoadedTTSProfile:
        """Update one exact loaded revision after service-owned validation."""

        self._validate_loaded_and_draft(loaded, draft)
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
        self._require_mutation_result_generation(
            loaded.repository_generation,
            result.generation,
        )
        return LoadedTTSProfile(
            repository_generation=result.generation,
            profile=result.value,
        )

    async def duplicate_profile(
        self,
        loaded: LoadedTTSProfile,
        display_name: str,
    ) -> LoadedTTSProfile:
        """Copy the immutable loaded version under a new profile identity."""

        self._validate_loaded(loaded)
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
        self._require_mutation_result_generation(
            loaded.repository_generation,
            result.generation,
        )
        return LoadedTTSProfile(
            repository_generation=result.generation,
            profile=result.value,
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
        self._require_mutation_result_generation(
            loaded.repository_generation,
            result.generation,
        )
        if type(result.value) is not int or result.value < 0:
            raise ProfileValidationError("assignment_count")
        return result.value

    async def delete_profile(self, loaded: LoadedTTSProfile) -> None:
        """Delete one loaded profile while retaining repository protection."""

        self._validate_loaded(loaded)
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
        self._require_mutation_result_generation(
            loaded.repository_generation,
            result.generation,
        )
        if result.value is not None:
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

    def _require_repository_generation(self, expected_generation: int) -> None:
        if self._repository.generation != expected_generation:
            raise ProfileRepositoryError("stale")

    def _require_mutation_result_generation(
        self,
        expected_generation: int,
        result_generation: int,
    ) -> None:
        if (
            result_generation != expected_generation
            or self._repository.generation != expected_generation
        ):
            raise ProfileRepositoryError("stale")

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
        await self._require_configuration_revision(
            draft.provider_id,
            snapshot.configuration_revision,
        )

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
