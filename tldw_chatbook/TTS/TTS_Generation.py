from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Mapping,
)
from contextlib import asynccontextmanager, nullcontext
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal, Protocol
from uuid import uuid4

from tldw_chatbook.TTS._async_lifecycle import (
    join_retained_task,
    shutdown_deadline_scope,
)
from tldw_chatbook.TTS.adapter_registry import (
    ReconfigureResult,
    TTSAdapterLease,
    TTSAdapterRegistry,
    TTSReconfigurationTicket,
)
from tldw_chatbook.TTS.adapter_types import (
    AudioCppCloneCapabilityAdmission,
    CapabilitySnapshotState,
    CleanupCallback,
    ProgressSink,
    ProviderHealth,
    TTSAudioResponse,
    TTSCloneGenerationEvidence,
    TTSConfigurationRevisionError,
    TTSNativeCapabilityObservation,
    TTSNativeCapabilitySnapshot,
    TTSNativeCloneAdapter,
    TTSNativeCloneDependencyAdapter,
    TTSOutboundEndpointAdapter,
    TTSOperationError,
    TTSProgress,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
    TTSRequest,
    TTSStructuredVoiceAdapter,
    TTSVoiceDiscoveryResult,
    _new_admitted_audio_cpp_clone_request,
    _new_tts_clone_generation_evidence,
)
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_guided_config import AudioCppSettingsConfig
from tldw_chatbook.TTS.audio_cpp_recipes import (
    AUDIO_CPP_RECIPE_REGISTRY,
    audio_cpp_guided_default_is_text_ready,
)
from tldw_chatbook.TTS.audio_cpp_supervisor import (
    AudioCppProcessAdmissionSnapshot,
    AudioCppProcessSnapshot,
    AudioCppSupervisor,
    AudioCppTTSCapability,
    _AudioCppGenerationChanged,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.effective_settings import (
    NativeCapabilityReader,
    TTSCharacterProfileSelection,
    TTSDefaultProfileSelection,
    TTSEffectiveSelectionSnapshot,
    TTSSelectionOverrides,
    TTSStudioDraftSelection,
    tts_configuration_is_active,
)
from tldw_chatbook.TTS.legacy_bridge import resolve_legacy_route
from tldw_chatbook.TTS.playground_types import (
    STTSPlaygroundCloneSnapshot,
    STTSPlaygroundProfilePreview,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_reference_materialization import (
    TTSCloneMaterializationError,
    TTSCloneReferenceMaterialization,
    TTSCloneReferenceMaterializer,
)
from tldw_chatbook.TTS.profile_reference_types import (
    CanonicalTTSCloneReference,
    TTSCloneReference,
    TTSCloneRecipeRequirement,
)
from tldw_chatbook.TTS.request_admission import (
    TTSAdmissionAuthorizer,
    TTSProfileReferenceResolver,
    TTSRequestAdmissionCoordinator,
    _ResolvedTTSCloneExecutionAuthority,
)
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot

logger = logging.getLogger(__name__)

_CLEANUP_FAILURE_NOTE = "TTS cleanup also failed while preserving the original error"
_TTS_SETTINGS_FOREGROUND_TIMEOUT_SECONDS = 2.0
_NATIVE_CAPABILITY_TIMEOUT_SECONDS = 10.0
_NATIVE_CAPABILITY_VOICE_CONCURRENCY = 4
_NATIVE_CAPABILITY_MAX_MODEL_ROWS = 50


def _canonical_clone_reference(
    reference: CanonicalTTSCloneReference | TTSCloneReference,
) -> CanonicalTTSCloneReference:
    """Detach one admitted reference into exact canonical evidence."""

    if type(reference) is CanonicalTTSCloneReference:
        return CanonicalTTSCloneReference(
            wav_bytes=reference.wav_bytes,
            reference_text=reference.reference_text,
            sha256=reference.sha256,
            byte_length=reference.byte_length,
            duration_ms=reference.duration_ms,
            sample_rate_hz=reference.sample_rate_hz,
            channels=reference.channels,
            sample_encoding=reference.sample_encoding,
        )
    if type(reference) is TTSCloneReference:
        summary = reference.summary
        return CanonicalTTSCloneReference(
            wav_bytes=reference.wav_bytes,
            reference_text=reference.reference_text,
            sha256=reference.sha256,
            byte_length=summary.byte_length,
            duration_ms=summary.duration_ms,
            sample_rate_hz=summary.sample_rate_hz,
            channels=summary.channels,
            sample_encoding=summary.sample_encoding,
        )
    raise TypeError("Clone reference authority is invalid")


def _native_capability_deadline() -> float:
    """Return the one aggregate deadline for a capability observation."""
    return asyncio.get_running_loop().time() + _NATIVE_CAPABILITY_TIMEOUT_SECONDS


def _audio_cpp_clone_setup_projection(
    settings: AudioCppSettingsConfig,
    selected_model_id: str | None,
) -> AudioCppCloneSetupProjection | None:
    """Project exact Guided clone setup metadata without exposing package paths."""

    if selected_model_id is None:
        return None
    if (
        type(selected_model_id) is not str
        or not selected_model_id
        or selected_model_id != selected_model_id.strip()
        or len(selected_model_id) > 256
        or any(ord(character) < 32 for character in selected_model_id)
    ):
        raise ValueError("audio.cpp selected model ID is invalid")
    if settings.mode != "managed" or settings.managed_setup_source != "guided":
        return None
    accepted = next(
        (
            package
            for package in settings.guided_packages
            if package.public_model_id == selected_model_id
        ),
        None,
    )
    if accepted is None:
        return None
    try:
        recipe = AUDIO_CPP_RECIPE_REGISTRY.validate_accepted(accepted)
    except ValueError:
        return None
    if (
        "clone" not in recipe.capabilities
        or recipe.reference_requirement.value != "required"
    ):
        return None
    family_labels = {
        "pocket_tts": "Pocket TTS",
        "supertonic": "Supertonic",
    }
    return AudioCppCloneSetupProjection(
        model_id=selected_model_id,
        recipe_id=recipe.recipe_id,
        recipe_revision=recipe.recipe_revision,
        family_label=family_labels.get(
            recipe.family,
            recipe.family.replace("_", " ").title(),
        ),
        recipe_label=recipe.display_name,
        reference_requirement=recipe.reference_requirement.value,
        voice_reference_policy=recipe.voice_reference_policy.value,
    )


_GuidedCloneObservationState = Literal["absent", "missing", "mismatch", "exact"]


@dataclass(frozen=True, slots=True)
class _GuidedCloneObservation:
    state: _GuidedCloneObservationState
    requirement: TTSCloneRecipeRequirement | None = None


def _guided_clone_observation(
    settings: AudioCppSettingsConfig,
    requirement: TTSCloneRecipeRequirement,
) -> _GuidedCloneObservation:
    """Preserve absence, unknown recipe, drift, and exact Guided evidence."""

    if settings.mode != "managed" or settings.managed_setup_source != "guided":
        return _GuidedCloneObservation("absent")
    model_match = next(
        (
            package
            for package in settings.guided_packages
            if package.public_model_id == requirement.model_id
        ),
        None,
    )
    recipe_match = next(
        (
            package
            for package in settings.guided_packages
            if package.recipe_id == requirement.recipe_id
        ),
        None,
    )
    accepted = model_match if model_match is not None else recipe_match
    if accepted is None:
        return _GuidedCloneObservation("absent")
    installed = next(
        (
            recipe
            for recipe in AUDIO_CPP_RECIPE_REGISTRY.recipes
            if recipe.recipe_id == requirement.recipe_id
        ),
        None,
    )
    if installed is None:
        return _GuidedCloneObservation("missing")
    if accepted.recipe_id != requirement.recipe_id:
        return _GuidedCloneObservation("mismatch")
    try:
        recipe = AUDIO_CPP_RECIPE_REGISTRY.validate_accepted(accepted)
    except ValueError:
        return _GuidedCloneObservation("mismatch")
    if "clone" not in recipe.capabilities:
        return _GuidedCloneObservation("mismatch")
    observed = TTSCloneRecipeRequirement(
        recipe_id=recipe.recipe_id,
        recipe_revision=recipe.recipe_revision,
        model_id=accepted.public_model_id,
    )
    if observed != requirement:
        return _GuidedCloneObservation("mismatch")
    return _GuidedCloneObservation("exact", observed)


TTSSettingsProviderStatus = Literal[
    "applied",
    "pending",
    "unchanged",
    "superseded",
    "unavailable",
]


@dataclass(frozen=True, slots=True)
class AudioCppCloneSetupProjection:
    """Path-free setup guidance for one exact selected Guided clone model."""

    model_id: str
    recipe_id: str
    recipe_revision: int
    family_label: str
    recipe_label: str
    reference_requirement: Literal["none", "optional", "required"]
    voice_reference_policy: Literal[
        "native_only",
        "reference_only",
        "either",
        "both_required_combined",
    ]

    def __post_init__(self) -> None:
        for value, label, limit in (
            (self.model_id, "model ID", 256),
            (self.recipe_id, "recipe ID", 256),
            (self.family_label, "family label", 128),
            (self.recipe_label, "recipe label", 256),
        ):
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or len(value) > limit
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(f"audio.cpp clone setup {label} is invalid")
        if type(self.recipe_revision) is not int or self.recipe_revision < 1:
            raise ValueError("audio.cpp clone setup recipe revision is invalid")
        if self.reference_requirement not in {"none", "optional", "required"}:
            raise ValueError("audio.cpp clone setup reference requirement is invalid")
        if self.voice_reference_policy not in {
            "native_only",
            "reference_only",
            "either",
            "both_required_combined",
        }:
            raise ValueError("audio.cpp clone setup voice policy is invalid")


AudioCppGuidedDependencyState = Literal[
    "exact",
    "missing",
    "mismatch",
    "pending",
]


@dataclass(frozen=True, slots=True)
class AudioCppGuidedDependencySnapshot:
    """Pure saved/applied Guided dependency facts for one clone requirement."""

    state: AudioCppGuidedDependencyState
    provider_configuration_revision: int
    saved_generation: int
    applied_generation: int
    pending_configuration: bool
    saved_requirement: TTSCloneRecipeRequirement | None
    applied_requirement: TTSCloneRecipeRequirement | None


def validate_audio_cpp_guided_dependency_snapshot(
    value: object,
    requirement: TTSCloneRecipeRequirement,
) -> AudioCppGuidedDependencySnapshot | None:
    """Fail closed on forged pure dependency evidence."""

    if (
        type(value) is not AudioCppGuidedDependencySnapshot
        or type(requirement) is not TTSCloneRecipeRequirement
    ):
        return None

    def canonical_requirement(
        observed: object,
    ) -> TTSCloneRecipeRequirement | None:
        if observed is None:
            return None
        if not isinstance(observed, TTSCloneRecipeRequirement):
            raise ValueError
        if type(observed) is not TTSCloneRecipeRequirement:
            raise ValueError
        return TTSCloneRecipeRequirement(
            recipe_id=observed.recipe_id,
            recipe_revision=observed.recipe_revision,
            model_id=observed.model_id,
        )

    try:
        state = value.state
        provider_revision = value.provider_configuration_revision
        saved_generation = value.saved_generation
        applied_generation = value.applied_generation
        pending_configuration = value.pending_configuration
        exact_requirement = canonical_requirement(requirement)
        saved_requirement = canonical_requirement(value.saved_requirement)
        applied_requirement = canonical_requirement(value.applied_requirement)
    except Exception:  # noqa: BLE001 - hostile evidence fails closed
        return None

    if (
        type(state) is not str
        or state not in {"exact", "missing", "mismatch", "pending"}
        or any(
            type(item) is not int or item < 0
            for item in (
                provider_revision,
                saved_generation,
                applied_generation,
            )
        )
        or type(pending_configuration) is not bool
        or exact_requirement is None
        or pending_configuration != (saved_generation != applied_generation)
    ):
        return None
    if saved_requirement not in {
        None,
        exact_requirement,
    } or applied_requirement not in {
        None,
        exact_requirement,
    }:
        return None
    if not pending_configuration and saved_requirement != applied_requirement:
        return None
    if applied_requirement == exact_requirement:
        valid = state == "exact"
    elif pending_configuration and saved_requirement == exact_requirement:
        valid = state == "pending"
    else:
        valid = state in {"missing", "mismatch"}
    if not valid:
        return None
    if state == "exact":
        canonical_state: AudioCppGuidedDependencyState = "exact"
    elif state == "missing":
        canonical_state = "missing"
    elif state == "mismatch":
        canonical_state = "mismatch"
    else:
        canonical_state = "pending"
    return AudioCppGuidedDependencySnapshot(
        state=canonical_state,
        provider_configuration_revision=provider_revision,
        saved_generation=saved_generation,
        applied_generation=applied_generation,
        pending_configuration=pending_configuration,
        saved_requirement=saved_requirement,
        applied_requirement=applied_requirement,
    )


@dataclass(frozen=True, slots=True)
class AudioCppRuntimeObservation:
    """One passive, generation-coherent view of audio.cpp runtime state."""

    saved_mode: Literal["external", "managed"]
    saved_configuration_generation: int
    applied_mode: Literal["external", "managed"]
    applied_configuration_generation: int
    provider_configuration_revision: int
    pending_configuration: bool
    process: AudioCppProcessSnapshot
    catalog_revision: int | None
    catalog_fresh: bool
    catalog_observed_at: datetime | None
    tts_capability: AudioCppTTSCapability
    service_closed: bool
    saved_managed_binary_path: str | None = field(repr=False)
    saved_managed_server_json_path: str | None = field(repr=False)
    applied_managed_binary_path: str | None = field(repr=False)
    applied_managed_server_json_path: str | None = field(repr=False)
    active_endpoint: str | None = field(default=None, repr=False)
    saved_managed_setup_source: Literal["user_json", "guided"] | None = None
    applied_managed_setup_source: Literal["user_json", "guided"] | None = None
    saved_guided_model_ids: tuple[str, ...] = ()
    applied_guided_model_ids: tuple[str, ...] = ()
    saved_guided_default_model_id: str | None = None
    applied_guided_default_model_id: str | None = None
    saved_guided_text_ready: bool = False
    applied_guided_text_ready: bool = False
    clone_setup: AudioCppCloneSetupProjection | None = None


@dataclass(frozen=True, slots=True)
class TTSSettingsPersistenceOutcome:
    """Structured result of atomic settings replacement and cache refresh."""

    file_replaced: bool
    caches_reloaded: bool
    failure_phase: Literal["before_replace", "cache_reload"] | None

    def __post_init__(self) -> None:
        if (
            type(self.file_replaced) is not bool
            or type(self.caches_reloaded) is not bool
        ):
            raise TypeError("TTS settings persistence flags must be booleans")
        if self.failure_phase not in {None, "before_replace", "cache_reload"}:
            raise ValueError("Unknown TTS settings persistence failure phase")
        if self.caches_reloaded and not self.file_replaced:
            raise ValueError("Caches cannot reload before settings replacement")
        if self.failure_phase == "before_replace" and self.file_replaced:
            raise ValueError("Pre-replacement failure cannot replace the settings file")
        if self.failure_phase == "cache_reload" and (
            not self.file_replaced or self.caches_reloaded
        ):
            raise ValueError("Cache-reload failure requires a replaced settings file")


TTSDefaultActivationStatus = Literal[
    "activation_not_ready",
    "committed",
    "rolled_back",
    "rollback_failed",
]


@dataclass(frozen=True, slots=True)
class TTSDefaultActivationOutcome:
    """Truthful result of one generation-fenced default activation attempt."""

    status: TTSDefaultActivationStatus

    def __post_init__(self) -> None:
        if self.status not in {
            "activation_not_ready",
            "committed",
            "rolled_back",
            "rollback_failed",
        }:
            raise ValueError("Unknown TTS default activation status")

    @property
    def activated(self) -> bool:
        return self.status == "committed"

    def __bool__(self) -> bool:
        return self.activated


@dataclass(frozen=True, slots=True)
class TTSSettingsPublication:
    """One safe settings-publication result for foreground or final observers."""

    generation: int
    preferences: TTSPreferencesSnapshot
    persistence: TTSSettingsPersistenceOutcome
    provider_statuses: Mapping[str, TTSSettingsProviderStatus]
    provider_revisions: Mapping[str, int]
    published: bool
    staged_provider_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_statuses",
            MappingProxyType(dict(self.provider_statuses)),
        )
        object.__setattr__(
            self,
            "provider_revisions",
            MappingProxyType(dict(self.provider_revisions)),
        )
        staged_provider_ids = frozenset(self.staged_provider_ids)
        if not staged_provider_ids.issubset(self.provider_statuses):
            raise ValueError("Staged TTS providers require publication statuses")
        if any(
            self.provider_statuses[provider_id] != "pending"
            for provider_id in staged_provider_ids
        ):
            raise ValueError(
                "Staged TTS providers require pending publication statuses"
            )
        object.__setattr__(self, "staged_provider_ids", staged_provider_ids)


@dataclass(frozen=True, slots=True)
class TTSSettingsPublicationTicket:
    """Service-owned settings operation with bounded and definitive views."""

    generation: int
    foreground: asyncio.Future[TTSSettingsPublication]
    completion: asyncio.Task[TTSSettingsPublication]


class TTSSettingsPublicationLease(Protocol):
    """Process-local exact lease transferred into one settings publication."""

    def adopt(self) -> None:
        """Accept definitive ownership before the publication task starts."""

    async def release(self) -> None:
        """Release or retain exact cleanup authority for a later retry."""

    def abandon(self) -> None:
        """Release an unadopted transfer; do nothing after adoption."""


def _sanitized_shutdown_error(*failures: BaseException) -> RuntimeError:
    failure_types = ", ".join(sorted({type(failure).__name__ for failure in failures}))
    return RuntimeError(f"TTS shutdown cleanup failed ({failure_types})")


def _record_cleanup_failure(
    primary_error: BaseException,
    cleanup_error: BaseException,
) -> None:
    primary_error.add_note(_CLEANUP_FAILURE_NOTE)
    sanitized_error = RuntimeError("cleanup failure details redacted")
    logger.warning(
        "TTS cleanup failed while preserving an earlier error: %s",
        type(cleanup_error).__name__,
        exc_info=(
            type(sanitized_error),
            sanitized_error,
            cleanup_error.__traceback__,
        ),
    )


async def _join_retained_task(task: asyncio.Task[None]) -> None:
    """Join a retained cleanup task without abandoning it on cancellation.

    Args:
        task: Cleanup task owned by the TTS service or registry.
    """
    await join_retained_task(
        task,
        on_failure_after_cancellation=_record_cleanup_failure,
    )


async def _cleanup_preserving_primary(
    cleanup: Callable[[], Awaitable[None]],
    primary_error: BaseException,
) -> None:
    waiter = asyncio.current_task()
    cancellation_requests = waiter.cancelling() if waiter is not None else 0
    try:
        await cleanup()
    except asyncio.CancelledError as cleanup_error:
        if waiter is not None and waiter.cancelling() > cancellation_requests:
            raise
        _record_cleanup_failure(primary_error, cleanup_error)
    except BaseException as cleanup_error:
        _record_cleanup_failure(primary_error, cleanup_error)


class _OperationCapacityReservation:
    """One idempotent reservation of a service concurrency slot."""

    def __init__(self, operation_limit: asyncio.Semaphore) -> None:
        self._operation_limit = operation_limit
        self._transferred = False
        self._released = False

    def transfer_to_resources(self) -> None:
        if self._released or self._transferred:
            raise RuntimeError("The TTS operation capacity is not transferable")
        self._transferred = True

    def release_if_untransferred(self) -> None:
        if not self._transferred:
            self._release()

    def release_from_resources(self) -> None:
        if not self._transferred:
            raise RuntimeError("The TTS operation capacity was not transferred")
        self._release()

    def _release(self) -> None:
        if self._released:
            return
        self._released = True
        self._operation_limit.release()


@dataclass(frozen=True, slots=True)
class _AudioCppPreparation:
    """One deliberate operation's managed-generation admission fence."""

    require_existing: AudioCppProcessAdmissionSnapshot | None


def _adapter_admission_scope(
    adapter: object,
    preparation: _AudioCppPreparation | None,
) -> Any:
    if preparation is None:
        return nullcontext()
    scope_factory = getattr(adapter, "managed_admission", None)
    if not callable(scope_factory):
        return nullcontext()
    return scope_factory(preparation.require_existing)


class _OperationResources:
    def __init__(
        self,
        lease: TTSAdapterLease,
        capacity: _OperationCapacityReservation,
    ) -> None:
        self._lease = lease
        self._capacity = capacity
        self._capacity.transfer_to_resources()
        self._cleanup_task: asyncio.Task[None] | None = None

    async def close(self) -> None:
        await _join_retained_task(self.start_close())

    def start_close(self) -> asyncio.Task[None]:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._release())
        return self._cleanup_task

    async def _release(self) -> None:
        try:
            await self._lease.release()
        finally:
            self._capacity.release_from_resources()


class _ManagedAudioResponse(TTSAudioResponse):
    def __init__(
        self,
        response: TTSAudioResponse,
        resources: _OperationResources,
        on_closed: Callable[["_ManagedAudioResponse"], None],
        *,
        protected_close_chain: bool = False,
    ) -> None:
        super().__init__(
            provider_id=response.provider_id,
            model_id=response.model_id,
            audio_format=response.audio_format,
            content_type=response.content_type,
            byte_stream=response.byte_stream,
            sample_rate=response.sample_rate,
            metadata=response.metadata,
        )
        self._response = response
        self._resources = resources
        self._on_closed = on_closed
        self._protected_close_chain = protected_close_chain
        self._response_close_task: asyncio.Task[None] | None = None

    def add_cleanup(self, callback: CleanupCallback) -> None:
        self._response.add_cleanup(callback)

    def start_close(self) -> asyncio.Task[None]:
        if self._response_close_task is None:
            self._response_close_task = asyncio.create_task(self._close())
        return self._response_close_task

    def start_resource_release(self) -> asyncio.Task[None]:
        if self._protected_close_chain:
            return self.start_close()
        return self._resources.start_close()

    async def aclose(self) -> None:
        await _join_retained_task(self.start_close())

    async def _close(self) -> None:
        try:
            try:
                await self._response.aclose()
            except BaseException as error:
                await _cleanup_preserving_primary(self._resources.close, error)
                raise
            else:
                await self._resources.close()
        finally:
            self._on_closed(self)


class _AdmittedTTSOperation:
    """Single-use synthesis operation with already-admitted resources."""

    def __init__(
        self,
        *,
        request: TTSRequest,
        resources: _OperationResources,
        close_signal: asyncio.Event,
        on_finished: Callable[["_AdmittedTTSOperation"], None],
        manage_response: Callable[
            [TTSAudioResponse, _OperationResources, bool],
            _ManagedAudioResponse,
        ],
        observe_cleanup: Callable[[asyncio.Task[None]], None],
        audio_cpp_preparation: _AudioCppPreparation | None,
        clone_execution: _ResolvedTTSCloneExecutionAuthority | None,
        clone_requirement: TTSCloneRecipeRequirement | None,
        clone_materializer: TTSCloneReferenceMaterializer | None,
        admission_authorizer: TTSAdmissionAuthorizer | None,
    ) -> None:
        self._request = request
        self._resources = resources
        self._close_signal = close_signal
        self._on_finished = on_finished
        self._manage_response = manage_response
        self._observe_cleanup = observe_cleanup
        self._audio_cpp_preparation = audio_cpp_preparation
        self._clone_execution = clone_execution
        self._clone_requirement = clone_requirement
        self._clone_materializer = clone_materializer
        self._admission_authorizer = admission_authorizer
        self._claimed = False
        self._used = False
        self._executing = False
        self._close_task: asyncio.Task[None] | None = None
        self._shutdown_wait_task: asyncio.Task[None] | None = None
        self._finished = asyncio.Event()

    def claim(self) -> None:
        """Transfer a pending operation to its immediate execution owner."""
        if self._used or self._claimed:
            raise RuntimeError("The admitted TTS operation cannot be claimed")
        self._claimed = True

    async def synthesize(
        self,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        """Execute the admitted request exactly once."""
        response, _evidence = await self.synthesize_with_evidence(progress_sink)
        return response

    async def synthesize_with_evidence(
        self,
        progress_sink: ProgressSink | None = None,
    ) -> tuple[TTSAudioResponse, TTSCloneGenerationEvidence | None]:
        """Execute once and retain exact clone-success evidence internally."""
        if self._used:
            raise RuntimeError("The admitted TTS operation has already been used")
        self._claimed = True
        self._used = True
        self._executing = True

        if self._close_signal.is_set():
            closed_error = TTSRegistryClosedError("The TTS service is closed")
            try:
                await _cleanup_preserving_primary(self._resources.close, closed_error)
            finally:
                self._finish_tracking()
            raise closed_error

        lease = self._resources._lease
        safe_sink = _isolate_progress_sink(progress_sink)
        generation_changed = False
        materialization: TTSCloneReferenceMaterialization | None = None
        capability: AudioCppCloneCapabilityAdmission | None = None
        clone_evidence: TTSCloneGenerationEvidence | None = None
        try:
            with _adapter_admission_scope(
                lease.adapter,
                self._audio_cpp_preparation,
            ):
                await lease.adapter.ensure_ready()
                if self._clone_execution is None:
                    self._authorize_outbound(lease)
                    response = await lease.adapter.synthesize(self._request, safe_sink)
                else:
                    adapter = lease.adapter
                    materializer = self._clone_materializer
                    if (
                        not isinstance(adapter, TTSNativeCloneAdapter)
                        or materializer is None
                    ):
                        raise TTSOperationError(
                            code="request_invalid",
                            message="The selected TTS request is unavailable",
                            retryable=False,
                            operation_id=uuid4().hex,
                            recovery_action="check_profile",
                        ) from None
                    capability = adapter.admit_clone_capability(self._request)
                    requirement = self._clone_requirement
                    expected_process = (
                        None
                        if self._audio_cpp_preparation is None
                        or self._audio_cpp_preparation.require_existing is None
                        else self._audio_cpp_preparation.require_existing.process_generation
                    )
                    if requirement is not None and (
                        capability.model_id != requirement.model_id
                        or capability.recipe_id != requirement.recipe_id
                        or capability.recipe_revision != requirement.recipe_revision
                        or type(capability.process_generation) is not int
                        or capability.process_generation < 1
                        or (
                            expected_process is not None
                            and capability.process_generation != expected_process
                        )
                    ):
                        adapter.release_clone_capability(capability)
                        capability = None
                        raise TTSOperationError(
                            code="dependency_changed",
                            message="The clone voice dependency changed",
                            retryable=False,
                            operation_id=uuid4().hex,
                            recovery_action="refresh",
                        ) from None
                    materialization_failure: str | None = None
                    try:
                        materialization = await materializer.materialize(
                            self._clone_execution.reference
                        )
                    except TTSCloneMaterializationError as error:
                        materialization_failure = error.code
                    if materialization_failure is not None:
                        if materialization_failure == "closed":
                            raise TTSOperationError(
                                code="connection_unavailable",
                                message="Clone reference materialization is unavailable",
                                retryable=True,
                                operation_id=uuid4().hex,
                                recovery_action="retry",
                            ) from None
                        raise TTSOperationError(
                            code=(
                                "request_invalid"
                                if materialization_failure == "unsupported"
                                else "generation_failed"
                            ),
                            message="Clone reference materialization is unavailable",
                            retryable=False,
                            operation_id=uuid4().hex,
                            recovery_action="check_profile",
                        ) from None
                    assert materialization is not None
                    admitted_request = _new_admitted_audio_cpp_clone_request(
                        request=self._request,
                        materialization=materialization,
                        capability=capability,
                        provider_revision=lease.configuration_revision,
                        applied_provider_generation=lease.applied_generation,
                    )
                    self._authorize_outbound(lease)
                    response = await adapter.synthesize_clone(
                        admitted_request,
                        safe_sink,
                    )
                    try:
                        clone_evidence = _new_tts_clone_generation_evidence(
                            capability=capability,
                            canonical_reference=_canonical_clone_reference(
                                self._clone_execution.reference
                            ),
                            provider_configuration_revision=(
                                lease.configuration_revision
                            ),
                            applied_provider_generation=lease.applied_generation,
                        )
                    except BaseException as error:
                        await _cleanup_preserving_primary(response.aclose, error)
                        raise
                    response.add_cleanup(materialization.aclose)
                    materialization = None
                    capability = None
        except _AudioCppGenerationChanged:
            generation_changed = True
        except BaseException as error:
            if capability is not None and isinstance(
                lease.adapter, TTSNativeCloneAdapter
            ):
                lease.adapter.release_clone_capability(capability)
            if materialization is not None:
                await _cleanup_preserving_primary(materialization.aclose, error)
            try:
                await _cleanup_preserving_primary(self._resources.close, error)
            finally:
                self._finish_tracking()
            raise
        if generation_changed:
            unavailable = TTSOperationError(
                code="connection_unavailable",
                message="The audio.cpp server is unavailable",
                retryable=True,
                operation_id=uuid4().hex,
                recovery_action="retry",
            )
            try:
                await _cleanup_preserving_primary(self._resources.close, unavailable)
            finally:
                self._finish_tracking()
            raise unavailable

        try:
            response_provider_id = response.provider_id
            admitted_provider_id = self._resources._lease.provider_id
            if (
                type(response_provider_id) is not str
                or response_provider_id != admitted_provider_id
            ):
                raise TTSOperationError(
                    code="audio_response_invalid",
                    message="TTS adapter returned invalid audio",
                    retryable=False,
                    operation_id=uuid4().hex,
                    recovery_action="check_provider",
                )
            managed_response = self._manage_response(
                response,
                self._resources,
                self._clone_execution is not None,
            )
        except BaseException as error:

            async def close_unmanaged_response() -> None:
                try:
                    await response.aclose()
                finally:
                    await self._resources.close()

            try:
                await _cleanup_preserving_primary(close_unmanaged_response, error)
            finally:
                self._finish_tracking()
            raise

        self._finish_tracking()
        if self._close_signal.is_set():
            closed_error = TTSRegistryClosedError("The TTS service is closed")
            cleanup_tasks = (
                managed_response.start_close(),
                managed_response.start_resource_release(),
            )
            for cleanup_task in cleanup_tasks:
                cleanup_task.add_done_callback(self._observe_cleanup)
            raise closed_error
        return managed_response, clone_evidence

    def _authorize_outbound(self, lease: TTSAdapterLease) -> None:
        authorizer = self._admission_authorizer
        if authorizer is None:
            return
        adapter = lease.adapter
        authorized = False
        try:
            if isinstance(adapter, TTSOutboundEndpointAdapter):
                authorized = authorizer(
                    lease.provider_id,
                    adapter.admitted_outbound_endpoint(),
                )
        except Exception:
            authorized = False
        if authorized is not True:
            raise TTSConfigurationRevisionError(
                "TTS provider destination authorization changed"
            )

    async def close(self) -> None:
        """Release an admitted operation that has not started execution."""
        await _join_retained_task(self.start_close())

    def start_close(self) -> asyncio.Task[None]:
        """Start idempotent resource cleanup for an abandoned operation."""
        if self._close_task is None:
            if self._used:
                raise RuntimeError("The admitted TTS operation has already been used")
            self._used = True
            self._close_task = asyncio.create_task(self._close())
        return self._close_task

    def start_close_if_pending(self) -> asyncio.Task[None] | None:
        """Start cleanup only when synthesis has not begun."""
        if self._executing or self._claimed:
            return None
        return self.start_close()

    def start_shutdown_cleanup(self) -> asyncio.Task[None] | None:
        """Release resources for an operation still tracked after the drain."""
        if self._claimed and not self._used:
            return None
        if self._clone_execution is not None and self._executing:
            if self._shutdown_wait_task is None:
                self._shutdown_wait_task = asyncio.create_task(
                    self._wait_until_finished()
                )
            return self._shutdown_wait_task
        if self._close_task is None:
            self._used = True
            self._close_task = asyncio.create_task(self._close())
        return self._close_task

    async def _wait_until_finished(self) -> None:
        await self._finished.wait()

    async def _close(self) -> None:
        try:
            await self._resources.close()
        finally:
            self._finish_tracking()

    def _finish_tracking(self) -> None:
        self._executing = False
        self._finished.set()
        self._on_finished(self)


class TTSService:
    """Coordinate registry-backed TTS operations and response lifetimes."""

    def __init__(
        self,
        registry: TTSAdapterRegistry,
        *,
        max_concurrent_operations: int = 4,
        preferences_snapshot: TTSPreferencesSnapshot | None = None,
        studio_preferences_loader: Callable[[], StudioTTSPreferencesSnapshot]
        | None = None,
        native_capability_reader: NativeCapabilityReader | None = None,
        audio_cpp_supervisor: AudioCppSupervisor | None = None,
        clone_materializer: TTSCloneReferenceMaterializer | None = None,
    ) -> None:
        if max_concurrent_operations < 1:
            raise ValueError("max_concurrent_operations must be positive")
        self.registry = registry
        self._audio_cpp_supervisor = audio_cpp_supervisor
        self._clone_materializer = clone_materializer
        self._audio_cpp_preparation: ContextVar[_AudioCppPreparation | None] = (
            ContextVar(
                f"tts_audio_cpp_preparation_{id(self)}",
                default=None,
            )
        )
        self._audio_cpp_lifecycle_lock = asyncio.Lock()
        self._audio_cpp_lifecycle_tasks: set[asyncio.Task[Any]] = set()
        self._operation_limit = asyncio.Semaphore(max_concurrent_operations)
        self._close_signal = asyncio.Event()
        self._registry_close_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._shutdown_deadline: float | None = None
        self._responses: set[_ManagedAudioResponse] = set()
        self._admitted_operations: set[_AdmittedTTSOperation] = set()
        self._settings_generation = 0
        self._settings_persisted_provider_generations: dict[str, int] = {}
        self._settings_persisted_provider_configs: dict[str, dict[str, Any]] = {}
        self._settings_admission_fences: dict[str, int] = {}
        self._settings_staged_preferences: dict[
            str,
            tuple[int, TTSPreferencesSnapshot],
        ] = {}
        self._settings_publication_tasks: set[asyncio.Task[TTSSettingsPublication]] = (
            set()
        )
        self._settings_publication_leases: set[TTSSettingsPublicationLease] = set()
        self._native_capability_observations: dict[
            str,
            TTSNativeCapabilityObservation,
        ] = {}
        self._native_catalog_request_generations: dict[str, int] = {}
        self._native_voice_request_generations: dict[tuple[str, str], int] = {}
        self._audio_cpp_catalog_process_generation: int | None = None
        self._audio_cpp_catalog_observation_version: int | None = None
        candidate_preferences = (
            TTSPreferencesSnapshot.from_settings({})
            if preferences_snapshot is None
            else preferences_snapshot
        )
        canonical_provider_ids = frozenset(self._canonical_provider_ids())
        initial_preferences = (
            candidate_preferences
            if candidate_preferences.provider_id in canonical_provider_ids
            else None
        )
        self._request_admission = TTSRequestAdmissionCoordinator(
            self,
            initial_preferences,
            studio_preferences_loader,
            native_capability_reader,
        )

    @asynccontextmanager
    async def _prepared_provider_read(
        self,
        provider_id: str,
        *,
        deliberate: bool,
    ) -> AsyncIterator[_AudioCppPreparation | None]:
        """Fence a passive read or prepare a deliberate provider operation."""
        supervisor = self._audio_cpp_supervisor
        if provider_id != "audio_cpp" or supervisor is None:
            async with self._request_admission._gate.read():
                self._raise_if_settings_admission_fenced(provider_id)
                token = self._audio_cpp_preparation.set(None)
                try:
                    yield None
                finally:
                    self._audio_cpp_preparation.reset(token)
            return

        if not deliberate:
            async with self._request_admission._gate.read():
                self._raise_if_settings_admission_fenced(provider_id)
                passive_preparation = _AudioCppPreparation(
                    require_existing=supervisor.admission_snapshot()
                )
                token = self._audio_cpp_preparation.set(passive_preparation)
                try:
                    yield passive_preparation
                finally:
                    self._audio_cpp_preparation.reset(token)
            return

        while True:
            wait_for_stage_boundary = False
            async with self._request_admission._publication_lock:
                configuration = await self.registry.provider_configuration_snapshot(
                    "audio_cpp"
                )
                admission = supervisor.admission_snapshot()
                if (
                    configuration.staged_config is not None
                    and admission.stage_application_eligible
                ):
                    async with self._request_admission._gate.write():
                        current_configuration = (
                            await self.registry.provider_configuration_snapshot(
                                "audio_cpp"
                            )
                        )
                        current_admission = supervisor.admission_snapshot()
                        if (
                            current_configuration.staged_config is None
                            or not current_admission.stage_application_eligible
                        ):
                            continue
                        await self.registry.run_exclusive_provider_transition(
                            "audio_cpp",
                            on_draining=supervisor.begin_draining,
                            action=self._stop_audio_cpp_for_transition,
                            apply_staged=True,
                        )
                        self._publish_staged_preferences_if_applied("audio_cpp")
                    continue
                if configuration.staged_config is not None and admission.state in {
                    "stopped",
                    "unavailable",
                }:
                    wait_for_stage_boundary = True

            if wait_for_stage_boundary:
                await supervisor.wait_for_stage_application_boundary()
                continue

            async with self._request_admission._gate.read():
                self._raise_if_settings_admission_fenced(provider_id)
                configuration = await self.registry.provider_configuration_snapshot(
                    "audio_cpp"
                )
                admission = supervisor.admission_snapshot()
                if (
                    configuration.staged_config is not None
                    and admission.stage_application_eligible
                ):
                    continue
                applied = AudioCppConfig.from_mapping(configuration.applied_config)
                preparation = (
                    _AudioCppPreparation(
                        require_existing=(
                            admission
                            if admission.state in {"starting", "running", "unhealthy"}
                            else None
                        )
                    )
                    if applied.mode == "managed"
                    else None
                )
                token = self._audio_cpp_preparation.set(preparation)
                try:
                    yield preparation
                finally:
                    self._audio_cpp_preparation.reset(token)
                return

    async def admit(
        self,
        request: TTSRequest,
        *,
        expected_configuration_revision: int | None = None,
        admission_authorizer: TTSAdmissionAuthorizer | None = None,
    ) -> _AdmittedTTSOperation:
        """Reserve service capacity and a revision-matched provider lease.

        Args:
            request: Native provider, model, and audio options.
            expected_configuration_revision: Optional selected provider revision.

        Returns:
            A single-use operation that owns its admitted resources.
        """
        if type(request) is not TTSRequest:
            raise TypeError("TTS request is invalid")
        reservation: _OperationCapacityReservation | None = None
        try:
            reservation = await self._reserve_operation_capacity()
            while True:
                try:
                    async with self._prepared_provider_read(
                        request.provider_id,
                        deliberate=True,
                    ):
                        return await self._admit_reserved(
                            request,
                            reservation,
                            expected_configuration_revision=(
                                expected_configuration_revision
                            ),
                            admission_authorizer=admission_authorizer,
                        )
                except _AudioCppGenerationChanged:
                    continue
        except BaseException:
            if reservation is not None:
                reservation.release_if_untransferred()
            raise

    async def _reserve_operation_capacity(
        self,
    ) -> _OperationCapacityReservation:
        """Reserve service capacity before entering request-selection gates."""
        await self._acquire_operation_slot()
        return _OperationCapacityReservation(self._operation_limit)

    def _require_operation_admission_open(self) -> None:
        """Reject closed requests before resolving private clone authority."""

        if self._close_signal.is_set():
            raise TTSRegistryClosedError("The TTS service is closed")

    async def _preflight_audio_cpp_clone_source(self) -> None:
        """Reject unauthorized clone sources before readiness or catalog I/O."""
        revision = self.configuration_revision("audio_cpp")
        lease = await self.registry.acquire("audio_cpp", expected_revision=revision)
        try:
            adapter = lease.adapter
            if not isinstance(adapter, TTSNativeCloneAdapter):
                raise TTSOperationError(
                    code="request_invalid",
                    message="The selected TTS request is unavailable",
                    retryable=False,
                    operation_id=uuid4().hex,
                    recovery_action="check_profile",
                ) from None
            adapter.preflight_clone_source()
        except BaseException as error:
            await _cleanup_preserving_primary(lease.release, error)
            raise
        else:
            await lease.release()

    async def _preflight_audio_cpp_clone_dependency(
        self,
        requirement: TTSCloneRecipeRequirement,
    ) -> None:
        """Repeat exact applied adapter configuration checks before readiness."""

        revision = self.configuration_revision("audio_cpp")
        lease = await self.registry.acquire("audio_cpp", expected_revision=revision)
        try:
            adapter = lease.adapter
            if not isinstance(adapter, TTSNativeCloneDependencyAdapter):
                raise TTSOperationError(
                    code="dependency_changed",
                    message="The clone voice dependency changed",
                    retryable=False,
                    operation_id=uuid4().hex,
                    recovery_action="refresh",
                ) from None
            adapter.preflight_clone_dependency(requirement)
        except BaseException as error:
            await _cleanup_preserving_primary(lease.release, error)
            raise
        else:
            await lease.release()

    async def _admit_reserved(
        self,
        request: TTSRequest,
        reservation: _OperationCapacityReservation,
        *,
        expected_configuration_revision: int | None = None,
        clone_execution: _ResolvedTTSCloneExecutionAuthority | None = None,
        admission_authorizer: TTSAdmissionAuthorizer | None = None,
    ) -> _AdmittedTTSOperation:
        """Acquire a provider lease using capacity reserved by this service."""
        try:
            lease = await self.registry.acquire(
                request.provider_id,
                expected_revision=expected_configuration_revision,
            )
        except BaseException:
            reservation.release_if_untransferred()
            raise

        preparation = self._audio_cpp_preparation.get()
        clone_reference = None if clone_execution is None else clone_execution.reference
        clone_requirement = (
            clone_reference.recipe_requirement
            if isinstance(clone_reference, TTSCloneReference)
            else None
        )
        if clone_execution is not None:
            adapter = lease.adapter
            try:
                if lease.provider_id != "audio_cpp" or not isinstance(
                    adapter, TTSNativeCloneAdapter
                ):
                    raise TTSOperationError(
                        code="request_invalid",
                        message="The selected TTS request is unavailable",
                        retryable=False,
                        operation_id=uuid4().hex,
                        recovery_action="check_profile",
                    ) from None
                adapter.preflight_clone_source()
                if clone_requirement is not None:
                    if not isinstance(adapter, TTSNativeCloneDependencyAdapter):
                        raise TTSOperationError(
                            code="dependency_changed",
                            message="The clone voice dependency changed",
                            retryable=False,
                            operation_id=uuid4().hex,
                            recovery_action="refresh",
                        ) from None
                    dependency_preflight_failed = False
                    try:
                        adapter.preflight_clone_request_dependency(
                            request,
                            clone_requirement,
                        )
                    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                        raise
                    except TTSOperationError:
                        raise
                    except Exception:
                        dependency_preflight_failed = True
                    if dependency_preflight_failed:
                        raise TTSOperationError(
                            code="dependency_changed",
                            message="The clone voice dependency changed",
                            retryable=False,
                            operation_id=uuid4().hex,
                            recovery_action="refresh",
                        ) from None
            except BaseException as error:
                await _cleanup_preserving_primary(lease.release, error)
                reservation.release_if_untransferred()
                raise
        if preparation is not None and lease.provider_id == "audio_cpp":
            try:
                configuration_revision = self.configuration_revision("audio_cpp")
                with _adapter_admission_scope(lease.adapter, preparation):
                    await lease.adapter.ensure_ready()
                    catalog = await lease.adapter.get_catalog(refresh=False)
                self._publish_native_catalog(
                    "audio_cpp",
                    configuration_revision,
                    catalog,
                    audio_cpp_adapter=lease.adapter,
                )
                supervisor = self._audio_cpp_supervisor
                if supervisor is not None:
                    admission = supervisor.admission_snapshot()
                    if admission.state in {"starting", "running", "unhealthy"}:
                        preparation = _AudioCppPreparation(
                            require_existing=admission,
                        )
            except BaseException as error:
                await _cleanup_preserving_primary(lease.release, error)
                reservation.release_if_untransferred()
                raise

        resources = _OperationResources(lease, reservation)
        if self._close_signal.is_set():
            closed_error = TTSRegistryClosedError("The TTS service is closed")
            await _cleanup_preserving_primary(resources.close, closed_error)
            raise closed_error

        operation = _AdmittedTTSOperation(
            request=request,
            resources=resources,
            close_signal=self._close_signal,
            on_finished=self._admitted_operations.discard,
            manage_response=self._manage_response,
            observe_cleanup=self._observe_shutdown_result,
            audio_cpp_preparation=preparation,
            clone_execution=clone_execution,
            clone_requirement=clone_requirement,
            clone_materializer=self._clone_materializer,
            admission_authorizer=admission_authorizer,
        )
        self._admitted_operations.add(operation)
        if self._close_signal.is_set():
            closed_error = TTSRegistryClosedError("The TTS service is closed")
            await _cleanup_preserving_primary(operation.close, closed_error)
            raise closed_error
        return operation

    async def resolve_provider_outbound_endpoint(self, provider_id: str) -> str:
        """Resolve the exact endpoint of one ready adapter under its lease."""
        lease = await self.registry.acquire(provider_id)
        try:
            if provider_id == "audio_cpp":
                await lease.adapter.ensure_ready()
            adapter = lease.adapter
            if not isinstance(adapter, TTSOutboundEndpointAdapter):
                raise TTSConfigurationRevisionError(
                    "TTS provider destination is unavailable"
                )
            endpoint = adapter.admitted_outbound_endpoint()
            if not isinstance(endpoint, str) or not endpoint:
                raise TTSConfigurationRevisionError(
                    "TTS provider destination is unavailable"
                )
            return endpoint
        finally:
            await lease.release()

    async def _close_admitted_operation_preserving_primary(
        self,
        operation: _AdmittedTTSOperation,
        primary_error: BaseException,
    ) -> None:
        """Close a claimed operation without replacing its primary failure."""
        await _cleanup_preserving_primary(operation.close, primary_error)

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
        admission_authorizer: TTSAdmissionAuthorizer | None = None,
    ) -> TTSAudioResponse:
        """Synthesize audio while retaining provider resources for the response.

        Args:
            request: Native provider, model, and audio options.
            progress_sink: Optional asynchronous progress reporter.

        Returns:
            A response that releases its registry lease when closed.
        """
        if type(request) is not TTSRequest:
            raise TypeError("TTS request is invalid")
        operation = await self.admit(
            request,
            admission_authorizer=admission_authorizer,
        )
        return await operation.synthesize(progress_sink)

    async def synthesize_exact(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> tuple[TTSAudioResponse, TTSRequestedSelectionSnapshot]:
        """Synthesize one exact native request with admitted provenance."""
        if type(request) is not TTSRequest:
            raise TypeError("TTS request is invalid")
        if type(request.provider_id) is not str or request.provider_id != "audio_cpp":
            raise ValueError("Exact provenance requires exact audio_cpp provider")
        return await self._request_admission.synthesize_exact(
            request,
            progress_sink,
        )

    async def require_current_configuration_revision(
        self,
        provider_id: str,
        expected_revision: int,
    ) -> None:
        """Make one writer-ordered decision about exact provider provenance."""
        if type(provider_id) is not str or provider_id not in (
            self._canonical_provider_ids()
        ):
            raise ValueError("TTS provider must be an exact canonical provider")
        if type(expected_revision) is not int:
            raise TypeError("Expected configuration revision must be an integer")
        if expected_revision < 0:
            raise ValueError("Expected configuration revision must be nonnegative")
        await self._request_admission.require_current_configuration_revision(
            provider_id,
            expected_revision,
        )

    # Native capability snapshot orchestration

    def latest_native_capability_observation(
        self,
        provider_id: str,
    ) -> TTSNativeCapabilityObservation | None:
        """Return accepted in-memory capability state without provider work.

        Args:
            provider_id: Exact canonical identifier for a native provider.

        Returns:
            The latest accepted observation, or ``None`` when none is available.

        Raises:
            TypeError: If ``provider_id`` is not a string.
            ValueError: If ``provider_id`` does not identify a native provider.
        """
        self._require_native_provider(provider_id)
        return self._native_capability_observations.get(provider_id)

    def _reserve_native_catalog_request(self, provider_id: str) -> int:
        generation = self._native_catalog_request_generations.get(provider_id, 0) + 1
        self._native_catalog_request_generations[provider_id] = generation
        return generation

    def _native_catalog_request_is_current(
        self,
        provider_id: str,
        generation: int,
    ) -> bool:
        return self._native_catalog_request_generations.get(provider_id) == generation

    def _reserve_native_voice_request(self, provider_id: str, model_id: str) -> int:
        key = (provider_id, model_id)
        generation = self._native_voice_request_generations.get(key, 0) + 1
        self._native_voice_request_generations[key] = generation
        return generation

    def _native_voice_request_is_current(
        self,
        provider_id: str,
        model_id: str,
        generation: int,
    ) -> bool:
        return (
            self._native_voice_request_generations.get((provider_id, model_id))
            == generation
        )

    def _publish_native_capability_snapshot(
        self,
        snapshot: TTSNativeCapabilitySnapshot,
    ) -> None:
        """Accept only a snapshot for the provider's still-current revision."""
        try:
            current_revision = self.configuration_revision(snapshot.provider_id)
        except (KeyError, TTSRegistryClosedError):
            return
        if current_revision != snapshot.configuration_revision:
            return
        self._native_capability_observations[snapshot.provider_id] = (
            TTSNativeCapabilityObservation(
                snapshot=snapshot,
                observed_at=datetime.now(timezone.utc),
            )
        )

    def _publish_native_catalog(
        self,
        provider_id: str,
        configuration_revision: int,
        catalog: TTSProviderCatalog,
        *,
        audio_cpp_adapter: object | None = None,
    ) -> None:
        """Publish a catalog and retain only voices from its exact revision."""
        if catalog.provider_id != provider_id:
            return
        try:
            current_revision = self.configuration_revision(provider_id)
        except (KeyError, TTSRegistryClosedError):
            return
        if current_revision != configuration_revision:
            return

        audio_cpp_evidence: tuple[int | None, int | None] | None = None
        if provider_id == "audio_cpp" and audio_cpp_adapter is not None:
            evidence_reader = getattr(
                audio_cpp_adapter,
                "catalog_publication_evidence",
                None,
            )
            if callable(evidence_reader):
                audio_cpp_evidence = evidence_reader(catalog)
                if audio_cpp_evidence is None:
                    return
                process_generation, observation_version = audio_cpp_evidence
                if process_generation is not None or observation_version is not None:
                    if process_generation is None or observation_version is None:
                        return
                    supervisor = self._audio_cpp_supervisor
                    process = None if supervisor is None else supervisor.snapshot()
                    if not (
                        process is not None
                        and process.state in {"running", "draining"}
                        and process.tts_capability in {"available", "not_configured"}
                        and process.process_generation == process_generation
                        and process.observation_version == observation_version
                    ):
                        return

        retained_voices: Mapping[str, TTSVoiceDiscoveryResult] = {}
        previous = self._native_capability_observations.get(provider_id)
        if (
            catalog.health.fresh
            and previous is not None
            and previous.snapshot.configuration_revision == configuration_revision
            and previous.snapshot.catalog is not None
            and previous.snapshot.catalog.revision == catalog.revision
        ):
            retained_voices = previous.snapshot.voice_results
        try:
            snapshot = TTSNativeCapabilitySnapshot(
                provider_id=provider_id,
                configuration_revision=configuration_revision,
                state="unverified",
                catalog=catalog,
                voice_results=retained_voices,
            )
        except (TypeError, ValueError):
            return
        self._publish_native_capability_snapshot(snapshot)
        if provider_id == "audio_cpp":
            if audio_cpp_evidence is not None:
                (
                    self._audio_cpp_catalog_process_generation,
                    self._audio_cpp_catalog_observation_version,
                ) = audio_cpp_evidence
                return
            supervisor = self._audio_cpp_supervisor
            process = None if supervisor is None else supervisor.snapshot()
            process_has_fresh_evidence = bool(
                process is not None
                and process.state in {"running", "draining"}
                and process.tts_capability in {"available", "not_configured"}
            )
            self._audio_cpp_catalog_process_generation = (
                process.process_generation
                if process_has_fresh_evidence and process is not None
                else None
            )
            self._audio_cpp_catalog_observation_version = (
                process.observation_version
                if process_has_fresh_evidence and process is not None
                else None
            )

    def _publish_native_voice_result(
        self,
        provider_id: str,
        configuration_revision: int,
        result: TTSVoiceDiscoveryResult,
    ) -> None:
        """Merge one model-scoped voice result into a matching catalog."""
        previous = self._native_capability_observations.get(provider_id)
        if previous is None:
            return
        snapshot = previous.snapshot
        catalog = snapshot.catalog
        if (
            snapshot.configuration_revision != configuration_revision
            or catalog is None
            or not catalog.health.fresh
            or result.provider_id != provider_id
            or result.catalog_revision != catalog.revision
            or result.model_id not in {model.model_id for model in catalog.models}
        ):
            return
        voice_results = dict(snapshot.voice_results)
        voice_results[result.model_id] = result
        try:
            merged = TTSNativeCapabilitySnapshot(
                provider_id=provider_id,
                configuration_revision=configuration_revision,
                state="unverified",
                catalog=catalog,
                voice_results=voice_results,
            )
        except (TypeError, ValueError):
            return
        self._publish_native_capability_snapshot(merged)

    async def get_native_capability_snapshot(
        self,
        provider_id: str,
        exact_voice_model_ids: Iterable[str],
    ) -> TTSNativeCapabilitySnapshot:
        """Observe one bounded native capability snapshot without exposing a lease."""
        return await self._get_native_capability_snapshot(
            provider_id,
            exact_voice_model_ids,
            already_prepared=False,
        )

    async def _get_native_capability_snapshot_already_prepared(
        self,
        provider_id: str,
        exact_voice_model_ids: Iterable[str],
    ) -> TTSNativeCapabilitySnapshot:
        """Observe capabilities while the caller holds the admission read side."""
        return await self._get_native_capability_snapshot(
            provider_id,
            exact_voice_model_ids,
            already_prepared=True,
        )

    async def _get_native_capability_snapshot(
        self,
        provider_id: str,
        exact_voice_model_ids: Iterable[str],
        *,
        already_prepared: bool,
    ) -> TTSNativeCapabilitySnapshot:
        """Build one native snapshot through prepared or public admission."""
        self._require_native_provider(provider_id)
        revision = self.configuration_revision(provider_id)
        if type(revision) is not int:
            raise TypeError("Capability configuration revision must be an integer")
        if revision < 0:
            raise ValueError("Capability configuration revision must be nonnegative")
        deadline = _native_capability_deadline()
        lease: TTSAdapterLease | None = None
        result = TTSNativeCapabilitySnapshot(
            provider_id=provider_id,
            configuration_revision=revision,
            state="unverified",
            catalog=None,
            voice_results={},
        )
        model_ids = self._distinct_capability_model_ids(exact_voice_model_ids)
        if provider_id == "audio_cpp" and self.saved_configuration_revision(
            provider_id
        ) != self.applied_configuration_revision(provider_id):
            return result
        if asyncio.get_running_loop().time() >= deadline:
            return result
        catalog_request_generation = self._reserve_native_catalog_request(provider_id)
        voice_request_generations = {
            model_id: self._reserve_native_voice_request(provider_id, model_id)
            for model_id in model_ids
        }
        primary_error: BaseException | None = None
        publication_adapter: object | None = None
        preparation = self._audio_cpp_preparation.get()
        process_fence_required = False
        process_fence: (
            tuple[
                AudioCppProcessAdmissionSnapshot,
                AudioCppProcessSnapshot,
            ]
            | None
        ) = None
        try:
            async with asyncio.timeout_at(deadline):
                if already_prepared:
                    revision = self.configuration_revision(provider_id)
                    lease = await self.registry.acquire(
                        provider_id,
                        expected_revision=revision,
                    )
                else:
                    (
                        revision,
                        lease,
                        preparation,
                    ) = await self._request_admission.acquire_native_capability_lease(
                        provider_id
                    )
                publication_adapter = lease.adapter
                with _adapter_admission_scope(
                    lease.adapter,
                    preparation,
                ):
                    result = await self._observe_native_capabilities(
                        provider_id,
                        revision,
                        lease.adapter,
                        model_ids,
                    )
                required_process = (
                    preparation.require_existing
                    if provider_id == "audio_cpp" and preparation is not None
                    else None
                )
                process_fence_required = (
                    required_process is not None
                    and required_process.state in {"starting", "running", "unhealthy"}
                )
                supervisor = self._audio_cpp_supervisor
                if process_fence_required and supervisor is not None:
                    assert required_process is not None
                    admission = supervisor.admission_snapshot()
                    process = supervisor.snapshot()
                    if (
                        admission.lifecycle_epoch == required_process.lifecycle_epoch
                        and admission.process_generation
                        == required_process.process_generation
                        and admission.state in {"running", "unhealthy"}
                        and process.process_generation == admission.process_generation
                        and process.state == admission.state
                        and process.endpoint is not None
                        and process.tts_capability in {"available", "not_configured"}
                        and process.consecutive_health_failures == 0
                        and process.last_failure is None
                    ):
                        process_fence = (admission, process)
                if self._close_signal.is_set():
                    result = self._unverified_native_capabilities(
                        provider_id,
                        revision,
                        result.catalog,
                    )
        except asyncio.CancelledError as error:
            primary_error = error
            raise
        except Exception as error:  # noqa: BLE001 - capability failures are unverified
            primary_error = error
            result = TTSNativeCapabilitySnapshot(
                provider_id=provider_id,
                configuration_revision=revision,
                state="unverified",
                catalog=None,
                voice_results={},
            )
        finally:
            if lease is not None:
                if primary_error is None:
                    await lease.release()
                else:
                    await _cleanup_preserving_primary(
                        lease.release,
                        primary_error,
                    )

        def finalize_result() -> TTSNativeCapabilitySnapshot:
            finalized = result
            process_fence_current = True
            if process_fence_required:
                supervisor = self._audio_cpp_supervisor
                if process_fence is None or supervisor is None:
                    process_fence_current = False
                else:
                    admission_fence, process_snapshot_fence = process_fence
                    current_admission = supervisor.admission_snapshot()
                    current_process = supervisor.snapshot()
                    process_fence_current = (
                        current_admission == admission_fence
                        and current_process.state == process_snapshot_fence.state
                        and current_process.process_generation
                        == process_snapshot_fence.process_generation
                        and current_process.observation_version
                        == process_snapshot_fence.observation_version
                        and current_process.endpoint == process_snapshot_fence.endpoint
                        and current_process.tts_capability
                        == process_snapshot_fence.tts_capability
                        and current_process.consecutive_health_failures
                        == process_snapshot_fence.consecutive_health_failures
                        and current_process.last_failure
                        == process_snapshot_fence.last_failure
                    )
            if not process_fence_current:
                finalized = self._unverified_native_capabilities(
                    provider_id,
                    revision,
                    None,
                )
            elif self.configuration_revision(provider_id) != revision:
                finalized = self._unverified_native_capabilities(
                    provider_id,
                    revision,
                    None,
                )
            elif provider_id == "audio_cpp" and self.saved_configuration_revision(
                provider_id
            ) != self.applied_configuration_revision(provider_id):
                finalized = self._unverified_native_capabilities(
                    provider_id,
                    revision,
                    None,
                )
            elif self._close_signal.is_set():
                finalized = self._unverified_native_capabilities(
                    provider_id,
                    revision,
                    finalized.catalog,
                )
            self._publish_native_snapshot_result(
                finalized,
                catalog_request_generation=catalog_request_generation,
                voice_request_generations=voice_request_generations,
                audio_cpp_adapter=publication_adapter,
            )
            return finalized

        if already_prepared:
            return finalize_result()
        async with self._request_admission._gate.read():
            return finalize_result()

    def _publish_native_snapshot_result(
        self,
        snapshot: TTSNativeCapabilitySnapshot,
        *,
        catalog_request_generation: int,
        voice_request_generations: Mapping[str, int],
        audio_cpp_adapter: object | None = None,
    ) -> None:
        """Merge only still-current catalog and model-scoped observations."""
        catalog = snapshot.catalog
        if catalog is not None and self._native_catalog_request_is_current(
            snapshot.provider_id,
            catalog_request_generation,
        ):
            self._publish_native_catalog(
                snapshot.provider_id,
                snapshot.configuration_revision,
                catalog,
                audio_cpp_adapter=audio_cpp_adapter,
            )
        for model_id, result in snapshot.voice_results.items():
            request_generation = voice_request_generations.get(model_id)
            if request_generation is None or not self._native_voice_request_is_current(
                snapshot.provider_id,
                model_id,
                request_generation,
            ):
                continue
            self._publish_native_voice_result(
                snapshot.provider_id,
                snapshot.configuration_revision,
                result,
            )

    async def _observe_native_capabilities(
        self,
        provider_id: str,
        configuration_revision: int,
        adapter: object,
        model_ids: tuple[str, ...],
    ) -> TTSNativeCapabilitySnapshot:
        """Observe at most two complete catalog generations on one adapter."""
        last_snapshot = TTSNativeCapabilitySnapshot(
            provider_id=provider_id,
            configuration_revision=configuration_revision,
            state="unverified",
            catalog=None,
            voice_results={},
        )
        for attempt in range(2):
            catalog = await adapter.get_catalog(refresh=True)  # type: ignore[attr-defined]
            if (
                type(catalog) is not TTSProviderCatalog
                or catalog.provider_id != provider_id
            ):
                return last_snapshot
            if not self._catalog_is_fresh(catalog):
                return self._unverified_native_capabilities(
                    provider_id,
                    configuration_revision,
                    catalog,
                )
            if model_ids:
                if not isinstance(adapter, TTSStructuredVoiceAdapter):
                    return self._unverified_native_capabilities(
                        provider_id,
                        configuration_revision,
                        catalog,
                    )
                semaphore = asyncio.Semaphore(_NATIVE_CAPABILITY_VOICE_CONCURRENCY)
                observed = await self._observe_native_voice_batch(
                    adapter,
                    semaphore,
                    model_ids,
                )
                voice_results = dict(zip(model_ids, observed, strict=True))
            else:
                voice_results = {}
            final_catalog = await adapter.get_catalog(refresh=False)  # type: ignore[attr-defined]
            if (
                type(final_catalog) is not TTSProviderCatalog
                or final_catalog.provider_id != provider_id
            ):
                return TTSNativeCapabilitySnapshot(
                    provider_id=provider_id,
                    configuration_revision=configuration_revision,
                    state="unverified",
                    catalog=catalog,
                    voice_results=voice_results,
                )
            if not self._catalog_is_fresh(final_catalog):
                return self._unverified_native_capabilities(
                    provider_id,
                    configuration_revision,
                    final_catalog,
                )

            catalog_moved = final_catalog.revision != catalog.revision
            if catalog_moved:
                last_snapshot = self._unverified_native_capabilities(
                    provider_id,
                    configuration_revision,
                    final_catalog,
                )
                if attempt == 0:
                    continue
                return last_snapshot

            authoritative = all(
                result.provider_id == provider_id
                and result.model_id == model_id
                and result.catalog_revision == catalog.revision
                and result.state != "unverified"
                for model_id, result in voice_results.items()
            )
            state: CapabilitySnapshotState = (
                "complete" if authoritative else "unverified"
            )
            last_snapshot = TTSNativeCapabilitySnapshot(
                provider_id=provider_id,
                configuration_revision=configuration_revision,
                state=state,
                catalog=final_catalog,
                voice_results=(
                    voice_results
                    if authoritative
                    else self._safe_unverified_voice_results(
                        final_catalog,
                        voice_results,
                    )
                ),
            )
            return last_snapshot

        return last_snapshot

    async def _observe_native_voice_batch(
        self,
        adapter: TTSStructuredVoiceAdapter,
        semaphore: asyncio.Semaphore,
        model_ids: tuple[str, ...],
    ) -> tuple[TTSVoiceDiscoveryResult, ...]:
        tasks = tuple(
            asyncio.create_task(
                self._observe_native_voices(adapter, semaphore, model_id),
                name="tts_native_capability_voice",
            )
            for model_id in model_ids
        )
        try:
            return tuple(await asyncio.gather(*tasks))
        except BaseException as primary_error:
            cleanup_task = asyncio.create_task(
                self._cancel_and_join_capability_tasks(tasks),
                name="tts_native_capability_voice_cleanup",
            )
            try:
                await _join_retained_task(cleanup_task)
            except asyncio.CancelledError:
                if isinstance(primary_error, asyncio.CancelledError):
                    raise primary_error from None
                raise
            except BaseException as cleanup_error:
                _record_cleanup_failure(primary_error, cleanup_error)
            raise

    @staticmethod
    async def _cancel_and_join_capability_tasks(
        tasks: tuple[asyncio.Task[TTSVoiceDiscoveryResult], ...],
    ) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _observe_native_voices(
        adapter: TTSStructuredVoiceAdapter,
        semaphore: asyncio.Semaphore,
        model_id: str,
    ) -> TTSVoiceDiscoveryResult:
        async with semaphore:
            return await adapter.observe_voices(model_id, refresh=True)

    @staticmethod
    def _catalog_is_fresh(catalog: TTSProviderCatalog) -> bool:
        return (
            type(catalog.health) is ProviderHealth
            and type(catalog.health.fresh) is bool
            and catalog.health.fresh
        )

    @staticmethod
    def _safe_unverified_voice_results(
        catalog: TTSProviderCatalog,
        voice_results: Mapping[str, TTSVoiceDiscoveryResult],
    ) -> dict[str, TTSVoiceDiscoveryResult]:
        return {
            model_id: result
            for model_id, result in voice_results.items()
            if result.state == "unverified"
            or (catalog.health.fresh and result.catalog_revision == catalog.revision)
        }

    @staticmethod
    def _unverified_native_capabilities(
        provider_id: str,
        configuration_revision: int,
        catalog: TTSProviderCatalog | None,
    ) -> TTSNativeCapabilitySnapshot:
        return TTSNativeCapabilitySnapshot(
            provider_id=provider_id,
            configuration_revision=configuration_revision,
            state="unverified",
            catalog=catalog,
            voice_results={},
        )

    # Capability request boundaries

    def _require_native_provider(self, provider_id: str) -> None:
        if type(provider_id) is not str:
            raise TypeError("TTS provider ID must be a string")
        native_ids = {
            descriptor.provider_id
            for descriptor in self.provider_descriptors()
            if descriptor.native
        }
        if provider_id not in native_ids:
            raise ValueError("TTS provider must be an exact native provider")

    @staticmethod
    def _distinct_capability_model_ids(
        model_ids: Iterable[str],
    ) -> tuple[str, ...]:
        distinct: dict[str, None] = {}
        for row_number, model_id in enumerate(model_ids, start=1):
            if row_number > _NATIVE_CAPABILITY_MAX_MODEL_ROWS:
                raise ValueError("Capability model selection accepts at most 50 rows")
            if type(model_id) is not str or not model_id:
                raise ValueError("Capability model ID is invalid")
            try:
                model_id.encode("utf-8", errors="strict")
            except UnicodeError:
                raise ValueError("Capability model ID is invalid") from None
            distinct.setdefault(model_id, None)
        return tuple(distinct)

    async def _release_lease_preserving_primary(
        self,
        lease: TTSAdapterLease,
        primary_error: BaseException,
    ) -> None:
        await _cleanup_preserving_primary(lease.release, primary_error)

    def preferences_snapshot(self) -> TTSPreferencesSnapshot | None:
        """Return canonical default preferences, or None while unconfigured."""
        return self._request_admission.preferences_snapshot()

    def preferences_generation(self) -> int:
        """Return the latest saved settings generation published in memory."""
        return self._request_admission.preferences_generation()

    async def synthesize_default(
        self,
        *,
        text: str,
        voice_override: str | None = None,
        progress_sink: ProgressSink | None = None,
        admission_authorizer: TTSAdmissionAuthorizer | None = None,
    ) -> TTSAudioResponse:
        """Resolve and synthesize one revision-coherent default request."""
        return await self._request_admission.synthesize_default(
            text=text,
            voice_override=voice_override,
            progress_sink=progress_sink,
            admission_authorizer=admission_authorizer,
        )

    async def synthesize_effective(
        self,
        *,
        text: str,
        explicit: TTSSelectionOverrides | None = None,
        character_profile: TTSCharacterProfileSelection | None = None,
        default_profile: TTSDefaultProfileSelection | None = None,
        studio_draft: TTSStudioDraftSelection | None = None,
        studio_preferences: StudioTTSPreferencesSnapshot | None = None,
        clone_audition: STTSPlaygroundCloneSnapshot | None = None,
        profile_preview: STTSPlaygroundProfilePreview | None = None,
        profile_reference_resolver: TTSProfileReferenceResolver | None = None,
        progress_sink: ProgressSink | None = None,
        admission_authorizer: TTSAdmissionAuthorizer | None = None,
    ) -> tuple[TTSAudioResponse, TTSEffectiveSelectionSnapshot]:
        """Resolve owner-scoped settings and synthesize one admitted request."""

        return await self._request_admission.synthesize_effective(
            text=text,
            explicit=explicit,
            character_profile=character_profile,
            default_profile=default_profile,
            studio_draft=studio_draft,
            studio_preferences=studio_preferences,
            clone_audition=clone_audition,
            profile_preview=profile_preview,
            profile_reference_resolver=profile_reference_resolver,
            progress_sink=progress_sink,
            admission_authorizer=admission_authorizer,
        )

    async def synthesize_effective_with_evidence(
        self,
        *,
        text: str,
        explicit: TTSSelectionOverrides | None = None,
        character_profile: TTSCharacterProfileSelection | None = None,
        default_profile: TTSDefaultProfileSelection | None = None,
        studio_draft: TTSStudioDraftSelection | None = None,
        studio_preferences: StudioTTSPreferencesSnapshot | None = None,
        clone_audition: STTSPlaygroundCloneSnapshot | None = None,
        profile_preview: STTSPlaygroundProfilePreview | None = None,
        profile_reference_resolver: TTSProfileReferenceResolver | None = None,
        progress_sink: ProgressSink | None = None,
        admission_authorizer: TTSAdmissionAuthorizer | None = None,
    ) -> tuple[
        TTSAudioResponse,
        TTSEffectiveSelectionSnapshot,
        TTSCloneGenerationEvidence | None,
    ]:
        """Synthesize and retain exact clone-success evidence for STTS."""

        return await self._request_admission.synthesize_effective_with_evidence(
            text=text,
            explicit=explicit,
            character_profile=character_profile,
            default_profile=default_profile,
            studio_draft=studio_draft,
            studio_preferences=studio_preferences,
            clone_audition=clone_audition,
            profile_preview=profile_preview,
            profile_reference_resolver=profile_reference_resolver,
            progress_sink=progress_sink,
            admission_authorizer=admission_authorizer,
        )

    async def generate_audio_stream(
        self,
        request: OpenAISpeechRequest,
        internal_model_id: str,
        progress_sink: ProgressSink | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream audio through the legacy OpenAI-compatible request interface.

        Args:
            request: Existing OpenAI-compatible speech request.
            internal_model_id: Exact legacy model route identifier.
            progress_sink: Optional asynchronous progress reporter.

        Yields:
            Audio byte chunks from the selected provider.
        """
        route = resolve_legacy_route(internal_model_id)
        native_request = TTSRequest(
            provider_id=route.provider_id,
            model_id=request.model,
            text=request.input,
            voice=request.voice,
            response_format=request.response_format,
            speed=request.speed,
            options={
                "_legacy_openai_request": request,
                "_legacy_internal_model_id": internal_model_id,
            },
        )
        response = await self.synthesize(native_request, progress_sink)
        try:
            async for chunk in response.byte_stream:
                yield chunk
        except GeneratorExit:
            await response.aclose()
            raise
        except BaseException as error:
            await _cleanup_preserving_primary(response.aclose, error)
            raise
        else:
            await response.aclose()

    def provider_descriptors(self) -> tuple[TTSProviderDescriptor, ...]:
        """Return ordered provider metadata without materializing adapters."""
        return self.registry.descriptors()

    def _canonical_provider_ids(self) -> tuple[str, ...]:
        """Return ordered exact provider IDs admitted by this service."""
        return tuple(
            descriptor.provider_id for descriptor in self.registry.descriptors()
        )

    def configuration_revision(self, provider_id: str) -> int:
        """Return the current registry configuration revision for a provider.

        Args:
            provider_id: Canonical provider identifier.

        Returns:
            The provider's monotonically increasing configuration revision.
        """
        return self.registry.configuration_revision(provider_id)

    def saved_configuration_revision(self, provider_id: str) -> int:
        """Return the latest durably published configuration generation.

        A provider starts at generation zero. Persistence advances this value
        before a runtime handoff, so a failed handoff cannot make evidence
        from the prior adapter configuration appear current.
        """

        self.registry.configuration_revision(provider_id)
        return self._settings_persisted_provider_generations.get(provider_id, 0)

    def applied_configuration_revision(self, provider_id: str) -> int:
        """Return the saved generation currently applied by the adapter slot."""

        return self.registry.configuration_generation(provider_id)

    async def _get_catalog_already_prepared(
        self,
        provider_id: str,
        refresh: bool = False,
    ) -> TTSProviderCatalog:
        """Read a catalog while the caller owns the shared admission side."""
        catalog, _adapter = await self._get_catalog_and_adapter_already_prepared(
            provider_id,
            refresh=refresh,
        )
        return catalog

    async def _get_catalog_and_adapter_already_prepared(
        self,
        provider_id: str,
        refresh: bool = False,
    ) -> tuple[TTSProviderCatalog, object]:
        """Read a catalog and retain its adapter identity for fenced publication."""
        lease = await self.registry.acquire(provider_id)
        try:
            preparation = self._audio_cpp_preparation.get()
            with _adapter_admission_scope(
                lease.adapter,
                preparation,
            ):
                if provider_id == "audio_cpp" and preparation is not None:
                    await lease.adapter.ensure_ready()
                catalog = await lease.adapter.get_catalog(refresh=refresh)
                return catalog, lease.adapter
        finally:
            await lease.release()

    async def get_catalog(
        self,
        provider_id: str,
        refresh: bool = False,
    ) -> TTSProviderCatalog:
        """Return a provider catalog, optionally refreshing its contents.

        Args:
            provider_id: Canonical provider identifier.
            refresh: Whether to refresh the provider catalog.

        Returns:
            The provider's current catalog.
        """
        native_provider = any(
            descriptor.provider_id == provider_id and descriptor.native
            for descriptor in self.provider_descriptors()
        )
        request_generation = (
            self._reserve_native_catalog_request(provider_id)
            if native_provider
            else None
        )
        while True:
            lease: TTSAdapterLease | None = None
            publication_adapter: object | None = None
            try:
                async with self._prepared_provider_read(
                    provider_id,
                    deliberate=refresh,
                ) as preparation:
                    configuration_revision = (
                        self.configuration_revision(provider_id)
                        if native_provider
                        else None
                    )
                    lease = await self.registry.acquire(provider_id)
                    publication_adapter = lease.adapter
                    with _adapter_admission_scope(lease.adapter, preparation):
                        catalog = await lease.adapter.get_catalog(refresh=refresh)
            except _AudioCppGenerationChanged:
                continue
            finally:
                if lease is not None:
                    await lease.release()
            break
        if (
            configuration_revision is not None
            and request_generation is not None
            and self._native_catalog_request_is_current(
                provider_id,
                request_generation,
            )
        ):
            self._publish_native_catalog(
                provider_id,
                configuration_revision,
                catalog,
                audio_cpp_adapter=publication_adapter,
            )
        return catalog

    async def get_voices(
        self,
        provider_id: str,
        model_id: str,
        refresh: bool = False,
    ) -> tuple[str, ...]:
        """Return voices for one provider model.

        Args:
            provider_id: Canonical provider identifier.
            model_id: Exact provider model identifier.
            refresh: Whether to refresh the provider's voice data.

        Returns:
            The provider's current voices for the selected model.
        """
        while True:
            lease: TTSAdapterLease | None = None
            try:
                async with self._prepared_provider_read(
                    provider_id,
                    deliberate=refresh,
                ) as preparation:
                    lease = await self.registry.acquire(provider_id)
                    with _adapter_admission_scope(lease.adapter, preparation):
                        result = await lease.adapter.get_voices(
                            model_id,
                            refresh=refresh,
                        )
            except _AudioCppGenerationChanged:
                continue
            finally:
                if lease is not None:
                    await lease.release()
            return result

    async def observe_voices(
        self,
        provider_id: str,
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult:
        """Return structured voice authority for one native provider model."""
        self._require_native_provider(provider_id)
        request_generation = self._reserve_native_voice_request(
            provider_id,
            model_id,
        )
        while True:
            lease: TTSAdapterLease | None = None
            try:
                async with self._prepared_provider_read(
                    provider_id,
                    deliberate=refresh,
                ) as preparation:
                    configuration_revision = self.configuration_revision(provider_id)
                    lease = await self.registry.acquire(provider_id)
                    adapter = lease.adapter
                    if not isinstance(adapter, TTSStructuredVoiceAdapter):
                        raise TypeError(
                            "TTS provider does not expose structured voice discovery"
                        )
                    with _adapter_admission_scope(adapter, preparation):
                        result = await adapter.observe_voices(
                            model_id,
                            refresh=refresh,
                        )
            except _AudioCppGenerationChanged:
                continue
            finally:
                if lease is not None:
                    await lease.release()
            break
        if self._native_voice_request_is_current(
            provider_id,
            model_id,
            request_generation,
        ):
            self._publish_native_voice_result(
                provider_id,
                configuration_revision,
                result,
            )
        return result

    async def reconfigure_provider(
        self,
        provider_id: str,
        config: Mapping[str, Any],
    ) -> ReconfigureResult:
        """Apply provider configuration through the registry lifecycle.

        Args:
            provider_id: Canonical provider identifier.
            config: Replacement provider configuration.

        Returns:
            The registry's reconfiguration result.
        """
        return await self.registry.reconfigure_provider(provider_id, config)

    def audio_cpp_process_snapshot(self) -> AudioCppProcessSnapshot:
        """Return the current passive snapshot of the app-owned audio.cpp child."""
        supervisor = self._audio_cpp_supervisor
        if supervisor is None:
            raise RuntimeError("Managed audio.cpp lifecycle is unavailable")
        return supervisor.snapshot()

    async def audio_cpp_guided_dependency_snapshot(
        self,
        requirement: TTSCloneRecipeRequirement,
    ) -> AudioCppGuidedDependencySnapshot:
        """Return pure saved/applied Guided recipe facts without provider work."""

        if type(requirement) is not TTSCloneRecipeRequirement:
            raise TypeError("Exact clone recipe requirement is required")
        exact_requirement = TTSCloneRecipeRequirement(
            recipe_id=requirement.recipe_id,
            recipe_revision=requirement.recipe_revision,
            model_id=requirement.model_id,
        )
        async with self._request_admission._publication_lock:
            async with self._request_admission._gate.read():
                configuration = await self.registry.provider_configuration_snapshot(
                    "audio_cpp"
                )
                saved_generation = self._settings_persisted_provider_generations.get(
                    "audio_cpp",
                    configuration.applied_generation,
                )
                saved_config = self._settings_persisted_provider_configs.get(
                    "audio_cpp"
                )
                if saved_config is None:
                    if (
                        configuration.staged_config is not None
                        and configuration.staged_generation == saved_generation
                    ):
                        saved_config = dict(configuration.staged_config)
                    else:
                        saved_config = dict(configuration.applied_config)
                saved_observation = _guided_clone_observation(
                    AudioCppSettingsConfig.from_mapping(saved_config),
                    exact_requirement,
                )
                applied_observation = _guided_clone_observation(
                    AudioCppSettingsConfig.from_mapping(configuration.applied_config),
                    exact_requirement,
                )

        pending = saved_generation != configuration.applied_generation
        saved = saved_observation.requirement
        applied = applied_observation.requirement
        installed = next(
            (
                recipe
                for recipe in AUDIO_CPP_RECIPE_REGISTRY.recipes
                if recipe.recipe_id == exact_requirement.recipe_id
            ),
            None,
        )
        if installed is None:
            state: AudioCppGuidedDependencyState = "missing"
        elif installed.recipe_revision != exact_requirement.recipe_revision:
            state = "mismatch"
        elif applied_observation.state == "exact":
            state = "exact"
        elif pending and saved_observation.state == "exact":
            state = "pending"
        elif (
            applied_observation.state == "mismatch"
            or saved_observation.state == "mismatch"
        ):
            state = "mismatch"
        else:
            state = "missing"
        return AudioCppGuidedDependencySnapshot(
            state=state,
            provider_configuration_revision=configuration.revision,
            saved_generation=saved_generation,
            applied_generation=configuration.applied_generation,
            pending_configuration=pending,
            saved_requirement=saved,
            applied_requirement=applied,
        )

    async def _require_audio_cpp_clone_dependency(
        self,
        requirement: TTSCloneRecipeRequirement,
    ) -> None:
        """Reject persisted dependency drift before adapter or provider work."""

        if type(requirement) is not TTSCloneRecipeRequirement:
            raise TypeError("Exact clone recipe requirement is required")
        exact = TTSCloneRecipeRequirement(
            recipe_id=requirement.recipe_id,
            recipe_revision=requirement.recipe_revision,
            model_id=requirement.model_id,
        )
        failed = False
        snapshot: AudioCppGuidedDependencySnapshot | None = None
        try:
            snapshot = await self.audio_cpp_guided_dependency_snapshot(exact)
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
        snapshot = validate_audio_cpp_guided_dependency_snapshot(snapshot, exact)
        if failed or snapshot is None:
            raise TTSOperationError(
                code="dependency_changed",
                message="The clone voice dependency changed",
                retryable=False,
                operation_id=uuid4().hex,
                recovery_action="refresh",
            ) from None
        if snapshot.state == "exact" and snapshot.applied_requirement == exact:
            return
        missing = snapshot.state == "missing"
        raise TTSOperationError(
            code="dependency_missing" if missing else "dependency_changed",
            message=(
                "The clone voice dependency is unavailable"
                if missing
                else "The clone voice dependency changed"
            ),
            retryable=False,
            operation_id=uuid4().hex,
            recovery_action="open_settings",
        ) from None

    async def audio_cpp_runtime_observation(
        self,
        *,
        selected_model_id: str | None = None,
    ) -> AudioCppRuntimeObservation:
        """Return passive runtime state and selected Guided-model setup metadata."""
        supervisor = self._audio_cpp_supervisor
        if supervisor is None:
            raise RuntimeError("Managed audio.cpp lifecycle is unavailable")

        async with self._request_admission._publication_lock:
            async with self._request_admission._gate.read():
                configuration = await self.registry.provider_configuration_snapshot(
                    "audio_cpp"
                )
                saved_generation = self._settings_persisted_provider_generations.get(
                    "audio_cpp",
                    0,
                )
                saved_config = self._settings_persisted_provider_configs.get(
                    "audio_cpp"
                )
                if saved_config is None:
                    if (
                        configuration.staged_config is not None
                        and configuration.staged_generation == saved_generation
                    ):
                        saved_config = dict(configuration.staged_config)
                    else:
                        saved_config = dict(configuration.applied_config)

                saved_settings = AudioCppSettingsConfig.from_mapping(saved_config)
                applied_settings = AudioCppSettingsConfig.from_mapping(
                    configuration.applied_config
                )
                saved = AudioCppConfig.from_mapping(saved_config)
                applied = AudioCppConfig.from_mapping(configuration.applied_config)
                latest_capability = self._native_capability_observations.get(
                    "audio_cpp"
                )
                capability = (
                    latest_capability
                    if latest_capability is not None
                    and latest_capability.snapshot.configuration_revision
                    == configuration.revision
                    else None
                )
                catalog = (
                    capability.snapshot.catalog if capability is not None else None
                )
                process = supervisor.snapshot()
                service_closed = self._close_signal.is_set()
                catalog_fresh = bool(
                    catalog is not None and catalog.health.fresh and not service_closed
                )
                if applied.mode == "managed":
                    catalog_fresh = bool(
                        catalog_fresh
                        and process.state in {"running", "draining"}
                        and process.tts_capability in {"available", "not_configured"}
                        and self._audio_cpp_catalog_process_generation
                        == process.process_generation
                        and self._audio_cpp_catalog_observation_version
                        == process.observation_version
                    )
                tts_capability: AudioCppTTSCapability = process.tts_capability
                if applied.mode == "external":
                    tts_capability = "unknown"
                    if catalog_fresh and catalog is not None:
                        if catalog.health.state == "available":
                            tts_capability = "available"
                        elif catalog.health.state == "not_configured":
                            tts_capability = "not_configured"
                clone_settings = applied_settings
                if (
                    saved_generation != configuration.applied_generation
                    and saved.mode == "managed"
                    and process.state in {"stopped", "unavailable"}
                ):
                    clone_settings = saved_settings
                clone_setup = _audio_cpp_clone_setup_projection(
                    clone_settings,
                    selected_model_id,
                )

                return AudioCppRuntimeObservation(
                    saved_mode=saved.mode,
                    saved_configuration_generation=saved_generation,
                    applied_mode=applied.mode,
                    applied_configuration_generation=(configuration.applied_generation),
                    provider_configuration_revision=configuration.revision,
                    pending_configuration=(
                        saved_generation != configuration.applied_generation
                    ),
                    process=process,
                    catalog_revision=None if catalog is None else catalog.revision,
                    catalog_fresh=catalog_fresh,
                    catalog_observed_at=(
                        None if capability is None else capability.observed_at
                    ),
                    tts_capability=tts_capability,
                    service_closed=service_closed,
                    saved_managed_binary_path=(
                        saved.managed_binary_path
                        if saved.mode == "managed"
                        and saved_settings.managed_setup_source == "user_json"
                        else None
                    ),
                    saved_managed_server_json_path=(
                        saved.managed_server_json_path
                        if saved.mode == "managed"
                        and saved_settings.managed_setup_source == "user_json"
                        else None
                    ),
                    applied_managed_binary_path=(
                        applied.managed_binary_path
                        if applied.mode == "managed"
                        and applied_settings.managed_setup_source == "user_json"
                        else None
                    ),
                    applied_managed_server_json_path=(
                        applied.managed_server_json_path
                        if applied.mode == "managed"
                        and applied_settings.managed_setup_source == "user_json"
                        else None
                    ),
                    active_endpoint=(
                        process.endpoint
                        if applied.mode == "managed"
                        else applied.base_url
                    ),
                    saved_managed_setup_source=(
                        saved_settings.managed_setup_source.value
                        if saved.mode == "managed"
                        else None
                    ),
                    applied_managed_setup_source=(
                        applied_settings.managed_setup_source.value
                        if applied.mode == "managed"
                        else None
                    ),
                    saved_guided_model_ids=tuple(
                        package.public_model_id
                        for package in saved_settings.guided_packages
                    ),
                    applied_guided_model_ids=tuple(
                        package.public_model_id
                        for package in applied_settings.guided_packages
                    ),
                    saved_guided_default_model_id=(
                        saved_settings.guided_default_model_id
                        if saved.mode == "managed"
                        and saved_settings.managed_setup_source == "guided"
                        else None
                    ),
                    applied_guided_default_model_id=(
                        applied_settings.guided_default_model_id
                        if applied.mode == "managed"
                        and applied_settings.managed_setup_source == "guided"
                        else None
                    ),
                    saved_guided_text_ready=bool(
                        saved.mode == "managed"
                        and saved_settings.managed_setup_source == "guided"
                        and audio_cpp_guided_default_is_text_ready(saved_settings)
                    ),
                    applied_guided_text_ready=bool(
                        applied.mode == "managed"
                        and applied_settings.managed_setup_source == "guided"
                        and audio_cpp_guided_default_is_text_ready(applied_settings)
                    ),
                    clone_setup=clone_setup,
                )

    async def start_and_test_audio_cpp(self) -> TTSProviderCatalog:
        """Deliberately prepare audio.cpp and refresh its native catalog."""
        return await self.get_catalog("audio_cpp", refresh=True)

    async def restart_audio_cpp(self) -> TTSProviderCatalog | None:
        """Drain audio.cpp, apply the latest stage, and restart Managed mode."""
        if self._close_signal.is_set():
            raise TTSRegistryClosedError("The TTS service is closed")
        task = self._start_audio_cpp_lifecycle(
            self._restart_audio_cpp(),
            name="tts_audio_cpp_restart",
        )
        await join_retained_task(task)
        return task.result()

    async def shutdown_audio_cpp(self) -> None:
        """Drain and stop audio.cpp while promoting the latest stage lazily."""
        if self._close_signal.is_set():
            raise TTSRegistryClosedError("The TTS service is closed")
        task = self._start_audio_cpp_lifecycle(
            self._shutdown_audio_cpp(),
            name="tts_audio_cpp_shutdown",
        )
        await join_retained_task(task)

    def _start_audio_cpp_lifecycle(
        self,
        operation: Coroutine[Any, Any, Any],
        *,
        name: str,
    ) -> asyncio.Task[Any]:
        task: asyncio.Task[Any] = asyncio.create_task(operation, name=name)
        self._audio_cpp_lifecycle_tasks.add(task)
        task.add_done_callback(self._audio_cpp_lifecycle_tasks.discard)
        task.add_done_callback(self._observe_background_result)
        return task

    async def _restart_audio_cpp(self) -> TTSProviderCatalog | None:
        async with self._audio_cpp_lifecycle_lock:
            supervisor = self._audio_cpp_supervisor
            if supervisor is None:
                raise RuntimeError("Managed audio.cpp lifecycle is unavailable")
            request_generation = self._reserve_native_catalog_request("audio_cpp")
            async with self._request_admission._publication_lock:
                async with self._request_admission._gate.write():
                    await self.registry.run_exclusive_provider_transition(
                        "audio_cpp",
                        on_draining=supervisor.begin_draining,
                        action=self._stop_audio_cpp_for_transition,
                        apply_staged=True,
                    )
                    self._publish_staged_preferences_if_applied("audio_cpp")
                    configuration = await self.registry.provider_configuration_snapshot(
                        "audio_cpp"
                    )
                    if (
                        AudioCppConfig.from_mapping(configuration.applied_config).mode
                        == "external"
                    ):
                        return None
                    configuration_revision = self.configuration_revision("audio_cpp")
                    preparation = _AudioCppPreparation(require_existing=None)
                    token = self._audio_cpp_preparation.set(preparation)
                    try:
                        (
                            catalog,
                            publication_adapter,
                        ) = await self._get_catalog_and_adapter_already_prepared(
                            "audio_cpp",
                            refresh=True,
                        )
                    finally:
                        self._audio_cpp_preparation.reset(token)
            if self._native_catalog_request_is_current(
                "audio_cpp",
                request_generation,
            ):
                self._publish_native_catalog(
                    "audio_cpp",
                    configuration_revision,
                    catalog,
                    audio_cpp_adapter=publication_adapter,
                )
            return catalog

    async def _shutdown_audio_cpp(self) -> None:
        async with self._audio_cpp_lifecycle_lock:
            await self._transition_audio_cpp_lifecycle(apply_staged=True)

    async def _transition_audio_cpp_lifecycle(
        self,
        *,
        apply_staged: bool,
    ) -> ReconfigureResult:
        supervisor = self._audio_cpp_supervisor
        if supervisor is None:
            raise RuntimeError("Managed audio.cpp lifecycle is unavailable")
        async with self._request_admission._publication_lock:
            async with self._request_admission._gate.write():
                result = await self.registry.run_exclusive_provider_transition(
                    "audio_cpp",
                    on_draining=supervisor.begin_draining,
                    action=self._stop_audio_cpp_for_transition,
                    apply_staged=apply_staged,
                )
                self._publish_staged_preferences_if_applied("audio_cpp")
                return result

    async def _stop_audio_cpp_for_transition(self) -> None:
        """Stop a transition under the outer shutdown deadline, if active."""
        supervisor = self._audio_cpp_supervisor
        if supervisor is None:
            raise RuntimeError("Managed audio.cpp lifecycle is unavailable")
        with shutdown_deadline_scope(self._shutdown_deadline):
            await supervisor.stop()

    def begin_preferences_publication(
        self,
        preferences: TTSPreferencesSnapshot,
        provider_configs: Mapping[str, Mapping[str, Any]],
        persistence: Callable[[], TTSSettingsPersistenceOutcome],
        *,
        foreground_timeout_seconds: float = (_TTS_SETTINGS_FOREGROUND_TIMEOUT_SECONDS),
        publish_preferences: bool = True,
        publication_lease: TTSSettingsPublicationLease | None = None,
    ) -> TTSSettingsPublicationTicket:
        """Start one retained settings persistence and runtime publication.

        Args:
            preferences: Complete immutable preferences to publish after the
                settings file is replaced.
            provider_configs: Complete replacement configurations keyed by
                canonical provider ID.
            persistence: Blocking atomic persistence operation.
            foreground_timeout_seconds: Maximum foreground wait for provider
                handoffs after persistence succeeds.
            publication_lease: Optional process-local lease already acquired
                across the before/after managed artifact union.

        Returns:
            A service-owned ticket with bounded foreground and final views.

        Raises:
            TTSRegistryClosedError: If service shutdown has started.
            TypeError: If an input does not match the publication contract.
            ValueError: If provider IDs or the timeout are invalid.
        """
        if self._close_signal.is_set():
            raise TTSRegistryClosedError("The TTS service is closed")
        if not isinstance(preferences, TTSPreferencesSnapshot):
            raise TypeError("preferences must be a TTSPreferencesSnapshot")
        if not isinstance(provider_configs, Mapping):
            raise TypeError("provider_configs must be a mapping")
        if not callable(persistence):
            raise TypeError("persistence must be callable")
        if type(publish_preferences) is not bool:
            raise TypeError("publish_preferences must be a boolean")
        if publication_lease is not None and (
            not callable(getattr(publication_lease, "adopt", None))
            or not callable(getattr(publication_lease, "release", None))
        ):
            raise TypeError("publication_lease must support adopt and release")
        if (
            isinstance(foreground_timeout_seconds, bool)
            or not isinstance(foreground_timeout_seconds, (int, float))
            or not math.isfinite(foreground_timeout_seconds)
            or foreground_timeout_seconds < 0
        ):
            raise ValueError("foreground timeout must be finite and non-negative")

        canonical_ids = self._canonical_provider_ids()
        canonical_id_set = frozenset(canonical_ids)
        if preferences.provider_id not in canonical_id_set:
            raise ValueError("preferences must use a canonical registered provider ID")
        validated_configs: dict[str, dict[str, Any]] = {}
        for provider_id, config in provider_configs.items():
            if not isinstance(provider_id, str) or provider_id not in canonical_id_set:
                raise ValueError("provider_configs must use canonical provider IDs")
            if not isinstance(config, Mapping):
                raise TypeError("Each TTS provider config must be a mapping")
            validated_configs[provider_id] = deepcopy(dict(config))
        copied_configs = {
            provider_id: validated_configs[provider_id]
            for provider_id in canonical_ids
            if provider_id in validated_configs
        }

        generation = self.registry.reserve_reconfiguration_generation()
        self._settings_generation = generation
        foreground: asyncio.Future[TTSSettingsPublication] = (
            asyncio.get_running_loop().create_future()
        )
        if publication_lease is not None:
            publication_lease.adopt()
            self._settings_publication_leases.add(publication_lease)
        completion = asyncio.create_task(
            self._run_owned_preferences_publication(
                generation=generation,
                preferences=preferences,
                provider_configs=copied_configs,
                persistence=persistence,
                foreground_timeout_seconds=float(foreground_timeout_seconds),
                foreground=foreground,
                publish_preferences=publish_preferences,
                publication_lease=publication_lease,
            ),
            name=f"tts_settings_publication_{generation}",
        )
        self._settings_publication_tasks.add(completion)
        completion.add_done_callback(self._settings_publication_tasks.discard)
        completion.add_done_callback(self._observe_settings_publication)
        return TTSSettingsPublicationTicket(
            generation=generation,
            foreground=foreground,
            completion=completion,
        )

    async def _run_owned_preferences_publication(
        self,
        *,
        generation: int,
        preferences: TTSPreferencesSnapshot,
        provider_configs: Mapping[str, Mapping[str, Any]],
        persistence: Callable[[], TTSSettingsPersistenceOutcome],
        foreground_timeout_seconds: float,
        foreground: asyncio.Future[TTSSettingsPublication],
        publish_preferences: bool,
        publication_lease: TTSSettingsPublicationLease | None,
    ) -> TTSSettingsPublication:
        """Run publication and release its transferred exact lease last."""

        try:
            return await self._run_preferences_publication(
                generation=generation,
                preferences=preferences,
                provider_configs=provider_configs,
                persistence=persistence,
                foreground_timeout_seconds=foreground_timeout_seconds,
                foreground=foreground,
                publish_preferences=publish_preferences,
            )
        finally:
            if publication_lease is not None:
                try:
                    await publication_lease.release()
                except Exception:
                    pass
                else:
                    self._settings_publication_leases.discard(publication_lease)

    async def _run_preferences_publication(
        self,
        *,
        generation: int,
        preferences: TTSPreferencesSnapshot,
        provider_configs: Mapping[str, Mapping[str, Any]],
        persistence: Callable[[], TTSSettingsPersistenceOutcome],
        foreground_timeout_seconds: float,
        foreground: asyncio.Future[TTSSettingsPublication],
        publish_preferences: bool,
    ) -> TTSSettingsPublication:
        tickets: dict[str, TTSReconfigurationTicket] = {}
        provider_statuses: dict[str, TTSSettingsProviderStatus] = {}
        provider_revisions: dict[str, int] = {}
        staged_provider_ids: set[str] = set()
        admission_fenced_provider_ids: set[str] = set()
        persistence_outcome = TTSSettingsPersistenceOutcome(
            file_replaced=False,
            caches_reloaded=False,
            failure_phase="before_replace",
        )

        async with self._request_admission._publication_lock:
            try:
                persisted = await asyncio.to_thread(persistence)
                if not isinstance(persisted, TTSSettingsPersistenceOutcome):
                    raise TypeError("Unexpected TTS settings persistence result")
                persistence_outcome = persisted
            except BaseException:
                persistence_outcome = TTSSettingsPersistenceOutcome(
                    file_replaced=False,
                    caches_reloaded=False,
                    failure_phase="before_replace",
                )

            if not persistence_outcome.file_replaced:
                provider_statuses.update(
                    {provider_id: "unchanged" for provider_id in provider_configs}
                )
                provider_revisions.update(
                    self._safe_provider_revisions(provider_configs)
                )
                result = self._settings_publication_result(
                    generation=generation,
                    preferences=preferences,
                    persistence=persistence_outcome,
                    provider_statuses=provider_statuses,
                    provider_revisions=provider_revisions,
                    published=False,
                )
                self._resolve_settings_foreground(foreground, result)
                return result

            async with self._request_admission._gate.write():
                for provider_id in provider_configs:
                    previous_generation = (
                        self._settings_persisted_provider_generations.get(
                            provider_id,
                            0,
                        )
                    )
                    if generation >= previous_generation:
                        self._settings_persisted_provider_generations[provider_id] = (
                            generation
                        )
                        if provider_id == "audio_cpp":
                            self._settings_persisted_provider_configs[provider_id] = (
                                deepcopy(dict(provider_configs[provider_id]))
                            )
                transition_failed = False
                staging_failed_provider_id: str | None = None
                for provider_id, config in provider_configs.items():
                    try:
                        staged_status = await self._stage_managed_boundary(
                            provider_id,
                            config,
                            generation=generation,
                        )
                    except BaseException:
                        staging_failed_provider_id = provider_id
                        transition_failed = True
                        break
                    if staged_status is not None:
                        provider_statuses[provider_id] = staged_status
                        if staged_status == "pending":
                            staged_provider_ids.add(provider_id)
                        continue
                    try:
                        ticket = await self.registry.begin_reconfigure_provider(
                            provider_id,
                            config,
                            generation=generation,
                        )
                        tickets[provider_id] = ticket
                        if ticket.admission_fenced:
                            admission_fenced_provider_ids.add(provider_id)
                    except BaseException:
                        transition_failed = True
                        break

                if transition_failed:
                    if (
                        staging_failed_provider_id is not None
                        or tickets
                        or staged_provider_ids
                    ):
                        await self._seal_provider_configs(provider_configs)
                    provider_statuses.update(
                        {provider_id: "unavailable" for provider_id in provider_configs}
                    )
                    staged_provider_ids.clear()
                else:
                    provider_statuses.update(
                        await self._bounded_reconfiguration_statuses(
                            tickets,
                            timeout_seconds=foreground_timeout_seconds,
                        )
                    )

                preferences_can_activate = publish_preferences and (
                    staging_failed_provider_id == preferences.provider_id
                    or self._preferences_can_activate(
                        preferences,
                        generation,
                        provider_configs,
                        provider_statuses,
                        admission_fenced_provider_ids,
                    )
                )
                if publish_preferences:
                    if preferences_can_activate:
                        self._request_admission._publish_preferences(
                            preferences,
                            generation,
                        )
                        if preferences.provider_id in provider_configs:
                            self._discard_staged_preferences_through(
                                preferences.provider_id,
                                generation,
                            )
                    elif provider_statuses.get(preferences.provider_id) == "pending":
                        if preferences.provider_id in staged_provider_ids:
                            self._settings_staged_preferences[
                                preferences.provider_id
                            ] = (generation, preferences)
                        else:
                            self._settings_admission_fences[preferences.provider_id] = (
                                generation
                            )
                provider_revisions.update(
                    self._safe_provider_revisions(provider_configs)
                )
                foreground_result = self._settings_publication_result(
                    generation=generation,
                    preferences=preferences,
                    persistence=persistence_outcome,
                    provider_statuses=provider_statuses,
                    provider_revisions=provider_revisions,
                    published=True,
                    staged_provider_ids=staged_provider_ids,
                )
                self._resolve_settings_foreground(foreground, foreground_result)

        final_statuses = dict(provider_statuses)
        for provider_id, ticket in tickets.items():
            if final_statuses.get(provider_id) == "pending":
                final_statuses[provider_id] = await self._reconfiguration_status(
                    provider_id,
                    ticket,
                )
                continue
            try:
                await asyncio.shield(ticket.completion)
            except BaseException:
                pass
        if publish_preferences:
            async with self._request_admission._gate.write():
                if self._preferences_can_activate(
                    preferences,
                    generation,
                    provider_configs,
                    final_statuses,
                    admission_fenced_provider_ids,
                ):
                    self._request_admission._publish_preferences(
                        preferences,
                        generation,
                    )
                    if preferences.provider_id in provider_configs:
                        self._discard_staged_preferences_through(
                            preferences.provider_id,
                            generation,
                        )
                self._clear_settings_admission_fence(
                    preferences.provider_id,
                    generation,
                )
        final_revisions = self._safe_provider_revisions(provider_configs)
        return self._settings_publication_result(
            generation=generation,
            preferences=preferences,
            persistence=persistence_outcome,
            provider_statuses=final_statuses,
            provider_revisions=final_revisions,
            published=True,
            staged_provider_ids=staged_provider_ids,
        )

    def _preferences_can_activate(
        self,
        preferences: TTSPreferencesSnapshot,
        generation: int,
        provider_configs: Mapping[str, Mapping[str, Any]],
        provider_statuses: Mapping[str, TTSSettingsProviderStatus],
        admission_fenced_provider_ids: Collection[str],
    ) -> bool:
        """Fence one default snapshot against its provider handoff."""

        provider_id = preferences.provider_id
        if provider_id not in provider_configs:
            return True
        provider_status = provider_statuses.get(provider_id)
        if provider_status == "pending":
            return provider_id in admission_fenced_provider_ids or (
                tts_configuration_is_active(self, provider_id, generation)
            )
        if provider_status not in {"applied", "unchanged"}:
            return False
        return tts_configuration_is_active(self, provider_id, generation)

    def _raise_if_settings_admission_fenced(self, provider_id: str) -> None:
        if provider_id in self._settings_admission_fences:
            raise TTSProviderReconfiguringError(
                f"TTS provider is reconfiguring: {provider_id}"
            )

    def _clear_settings_admission_fence(
        self,
        provider_id: str,
        generation: int,
    ) -> None:
        fenced_generation = self._settings_admission_fences.get(provider_id)
        if fenced_generation == generation:
            self._settings_admission_fences.pop(provider_id, None)

    def _discard_staged_preferences_through(
        self,
        provider_id: str,
        generation: int,
    ) -> None:
        pending = self._settings_staged_preferences.get(provider_id)
        if pending is not None and pending[0] <= generation:
            self._settings_staged_preferences.pop(provider_id, None)

    def _publish_staged_preferences_if_applied(self, provider_id: str) -> None:
        pending = self._settings_staged_preferences.get(provider_id)
        if pending is None:
            return
        generation, preferences = pending
        applied_generation = self.registry.configuration_generation(provider_id)
        if applied_generation < generation:
            return
        if applied_generation == generation:
            self._request_admission._publish_preferences(preferences, generation)
        self._settings_staged_preferences.pop(provider_id, None)
        self._clear_settings_admission_fence(provider_id, generation)

    async def commit_voice_setup_default(
        self,
        preferences: TTSPreferencesSnapshot,
        *,
        expected_saved_revision: int,
    ) -> bool:
        """Publish one default only while its exact provider generation is active."""

        outcome = await self._commit_voice_setup_default(
            preferences,
            expected_saved_revision=expected_saved_revision,
        )
        return outcome.activated

    async def _commit_voice_setup_default(
        self,
        preferences: TTSPreferencesSnapshot,
        *,
        expected_saved_revision: int,
        persistence: Callable[[], TTSSettingsPersistenceOutcome] | None = None,
        rollback: (
            Callable[
                [TTSPreferencesSnapshot | None],
                TTSSettingsPersistenceOutcome,
            ]
            | None
        ) = None,
    ) -> TTSDefaultActivationOutcome:
        """Fence optional defaults persistence and active-snapshot publication."""

        if not isinstance(preferences, TTSPreferencesSnapshot):
            raise TypeError("preferences must be a TTSPreferencesSnapshot")
        if type(expected_saved_revision) is not int or expected_saved_revision < 0:
            raise ValueError("expected_saved_revision must be nonnegative")
        if persistence is not None and not callable(persistence):
            raise TypeError("persistence must be callable")
        if rollback is not None and not callable(rollback):
            raise TypeError("rollback must be callable")

        async def compensate(prior: TTSPreferencesSnapshot | None) -> bool:
            if rollback is None:
                logger.error(
                    "TTS default activation failed after persistence without rollback"
                )
                return False
            try:
                outcome = await asyncio.to_thread(rollback, prior)
            except BaseException as error:
                logger.error(
                    "TTS default rollback raised %s",
                    type(error).__name__,
                )
                return False
            restored = bool(
                isinstance(outcome, TTSSettingsPersistenceOutcome)
                and outcome.file_replaced
                and outcome.caches_reloaded
            )
            if not restored:
                logger.error("TTS default rollback did not restore persisted settings")
            return restored

        async with self._request_admission._publication_lock:
            async with self._request_admission._gate.write():
                prior_preferences = self.preferences_snapshot()
                prior_generation = self.preferences_generation()
                if self.preferences_generation() > expected_saved_revision:
                    return TTSDefaultActivationOutcome("activation_not_ready")
                if not tts_configuration_is_active(
                    self,
                    preferences.provider_id,
                    expected_saved_revision,
                ):
                    return TTSDefaultActivationOutcome("activation_not_ready")
                if persistence is not None:
                    try:
                        outcome = await asyncio.to_thread(persistence)
                    except BaseException:
                        return TTSDefaultActivationOutcome("activation_not_ready")
                    if not isinstance(outcome, TTSSettingsPersistenceOutcome):
                        return TTSDefaultActivationOutcome("activation_not_ready")
                    if not outcome.file_replaced:
                        return TTSDefaultActivationOutcome("activation_not_ready")
                    if not outcome.caches_reloaded:
                        restored = await compensate(prior_preferences)
                        return TTSDefaultActivationOutcome(
                            "rolled_back" if restored else "rollback_failed"
                        )
                    if not tts_configuration_is_active(
                        self,
                        preferences.provider_id,
                        expected_saved_revision,
                    ):
                        restored = await compensate(prior_preferences)
                        return TTSDefaultActivationOutcome(
                            "rolled_back" if restored else "rollback_failed"
                        )
                try:
                    self._request_admission._publish_preferences(
                        preferences,
                        expected_saved_revision,
                    )
                except BaseException:
                    restored = False
                    if persistence is not None:
                        restored = await compensate(prior_preferences)
                    self._request_admission._preferences = prior_preferences
                    self._request_admission._preferences_generation = prior_generation
                    return TTSDefaultActivationOutcome(
                        "rolled_back"
                        if persistence is not None and restored
                        else "rollback_failed"
                        if persistence is not None
                        else "activation_not_ready"
                    )
                activated = bool(
                    self.preferences_generation() == expected_saved_revision
                    and self.preferences_snapshot() == preferences
                )
                if activated:
                    return TTSDefaultActivationOutcome("committed")
                restored = False
                if persistence is not None:
                    restored = await compensate(prior_preferences)
                self._request_admission._preferences = prior_preferences
                self._request_admission._preferences_generation = prior_generation
                return TTSDefaultActivationOutcome(
                    "rolled_back"
                    if persistence is not None and restored
                    else "rollback_failed"
                    if persistence is not None
                    else "activation_not_ready"
                )

    async def _stage_managed_boundary(
        self,
        provider_id: str,
        config: Mapping[str, Any],
        *,
        generation: int,
    ) -> TTSSettingsProviderStatus | None:
        """Stage audio.cpp changes that enter, leave, or alter Managed mode."""
        if provider_id != "audio_cpp":
            return None
        snapshot = await self.registry.provider_configuration_snapshot(provider_id)
        applied_config = dict(snapshot.applied_config)
        desired_config = dict(config)
        applied = AudioCppConfig.from_mapping(applied_config)
        desired = AudioCppConfig.from_mapping(desired_config)
        if desired_config != applied_config and (
            applied.mode != "managed" and desired.mode != "managed"
        ):
            return None

        result = await self.registry.stage_provider_configuration(
            provider_id,
            desired_config,
            generation=generation,
        )
        if result is ReconfigureResult.CHANGED:
            return "pending"
        if result is ReconfigureResult.UNCHANGED:
            return "unchanged"
        return "superseded"

    async def _bounded_reconfiguration_statuses(
        self,
        tickets: Mapping[str, TTSReconfigurationTicket],
        *,
        timeout_seconds: float,
    ) -> dict[str, TTSSettingsProviderStatus]:
        if not tickets:
            return {}
        completion_to_provider = {
            ticket.completion: provider_id for provider_id, ticket in tickets.items()
        }
        done, pending = await asyncio.wait(
            completion_to_provider,
            timeout=timeout_seconds,
        )
        statuses: dict[str, TTSSettingsProviderStatus] = {}
        for completion in done:
            provider_id = completion_to_provider[completion]
            statuses[provider_id] = await self._reconfiguration_status(
                provider_id,
                tickets[provider_id],
            )
        for completion in pending:
            statuses[completion_to_provider[completion]] = "pending"
        return {provider_id: statuses[provider_id] for provider_id in tickets}

    async def _reconfiguration_status(
        self,
        provider_id: str,
        ticket: TTSReconfigurationTicket,
    ) -> TTSSettingsProviderStatus:
        try:
            result = await asyncio.shield(ticket.completion)
        except BaseException:
            await self._seal_provider_configs((provider_id,))
            return "unavailable"
        if result is ReconfigureResult.CHANGED:
            return "applied"
        if result is ReconfigureResult.UNCHANGED:
            return "unchanged"
        if (
            self._settings_persisted_provider_generations.get(provider_id, 0)
            > ticket.generation
        ):
            return "superseded"
        await self._seal_provider_configs((provider_id,))
        return "unavailable"

    async def _seal_provider_configs(
        self,
        provider_configs: Mapping[str, object] | tuple[str, ...],
    ) -> None:
        for provider_id in reversed(tuple(provider_configs)):
            try:
                await self.registry.seal_provider_unavailable(provider_id)
            except BaseException:
                pass

    def _safe_provider_revisions(
        self,
        provider_configs: Mapping[str, object],
    ) -> dict[str, int]:
        revisions: dict[str, int] = {}
        for provider_id in provider_configs:
            try:
                revisions[provider_id] = self.configuration_revision(provider_id)
            except BaseException:
                continue
        return revisions

    @staticmethod
    def _settings_publication_result(
        *,
        generation: int,
        preferences: TTSPreferencesSnapshot,
        persistence: TTSSettingsPersistenceOutcome,
        provider_statuses: Mapping[str, TTSSettingsProviderStatus],
        provider_revisions: Mapping[str, int],
        published: bool,
        staged_provider_ids: Collection[str] = (),
    ) -> TTSSettingsPublication:
        return TTSSettingsPublication(
            generation=generation,
            preferences=preferences,
            persistence=persistence,
            provider_statuses=provider_statuses,
            provider_revisions=provider_revisions,
            published=published,
            staged_provider_ids=frozenset(staged_provider_ids),
        )

    @staticmethod
    def _resolve_settings_foreground(
        foreground: asyncio.Future[TTSSettingsPublication],
        result: TTSSettingsPublication,
    ) -> None:
        if not foreground.done():
            foreground.set_result(result)

    @staticmethod
    def _observe_settings_publication(
        task: asyncio.Task[TTSSettingsPublication],
    ) -> None:
        try:
            task.exception()
        except BaseException:
            pass

    async def close(self) -> None:
        """Seal admission and begin bounded provider shutdown."""
        registry_close_task, _ = self._start_close()
        caller = asyncio.current_task()
        cancellation_requests = caller.cancelling() if caller is not None else 0
        caller_cancellation: asyncio.CancelledError | None = None
        sanitized_error: RuntimeError | None = None
        try:
            await _join_retained_task(registry_close_task)
        except asyncio.CancelledError as error:
            if caller is not None and caller.cancelling() > cancellation_requests:
                caller_cancellation = asyncio.CancelledError()
                if _CLEANUP_FAILURE_NOTE in getattr(error, "__notes__", ()):
                    caller_cancellation.add_note(_CLEANUP_FAILURE_NOTE)
            else:
                sanitized_error = _sanitized_shutdown_error(error)
        except BaseException as error:
            sanitized_error = _sanitized_shutdown_error(error)
        if caller_cancellation is not None:
            raise caller_cancellation from None
        if sanitized_error is not None:
            raise sanitized_error from None

    async def wait_closed(self) -> None:
        """Join service-owned cleanup and report sanitized shutdown failures."""
        _, shutdown_task = self._start_close()
        await _join_retained_task(shutdown_task)

    async def _acquire_operation_slot(self) -> None:
        if self._close_signal.is_set():
            raise TTSRegistryClosedError("The TTS service is closed")

        acquire_task = asyncio.create_task(self._operation_limit.acquire())
        close_task = asyncio.create_task(self._close_signal.wait())
        try:
            await asyncio.wait(
                {acquire_task, close_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if close_task.done() or self._close_signal.is_set():
                raise TTSRegistryClosedError("The TTS service is closed")
            close_task.cancel()
            await asyncio.gather(close_task, return_exceptions=True)
            if self._close_signal.is_set():
                raise TTSRegistryClosedError("The TTS service is closed")
            acquire_task.result()
        except BaseException:
            cleanup_task = asyncio.create_task(
                self._cancel_admission(acquire_task, close_task)
            )
            await _join_retained_task(cleanup_task)
            raise

    async def _cancel_admission(
        self,
        acquire_task: asyncio.Task[bool],
        close_task: asyncio.Task[bool],
    ) -> None:
        acquired = self._task_acquired_slot(acquire_task)
        acquire_task.cancel()
        close_task.cancel()
        await asyncio.gather(
            acquire_task,
            close_task,
            return_exceptions=True,
        )
        if acquired or self._task_acquired_slot(acquire_task):
            self._operation_limit.release()

    def _start_close(self) -> tuple[asyncio.Task[None], asyncio.Task[None]]:
        if self._registry_close_task is None:
            self._close_signal.set()
            if self._clone_materializer is not None:
                self._clone_materializer.seal()
            publication_tasks = tuple(self._settings_publication_tasks)
            lifecycle_tasks = tuple(self._audio_cpp_lifecycle_tasks)
            operation_tasks = tuple(
                cleanup_task
                for operation in tuple(self._admitted_operations)
                if (cleanup_task := operation.start_close_if_pending()) is not None
            )
            deadline = (
                asyncio.get_running_loop().time()
                + self.registry.shutdown_timeout_seconds
            )
            self._shutdown_deadline = deadline
            with shutdown_deadline_scope(deadline):
                self._registry_close_task = asyncio.create_task(self.registry.close())
                self._shutdown_task = asyncio.create_task(
                    self._complete_shutdown(
                        self._registry_close_task,
                        operation_tasks,
                        publication_tasks,
                        lifecycle_tasks,
                    )
                )
            self._shutdown_task.add_done_callback(self._observe_shutdown_result)
        assert self._shutdown_task is not None
        return self._registry_close_task, self._shutdown_task

    async def _complete_shutdown(
        self,
        registry_close_task: asyncio.Task[None],
        operation_tasks: tuple[asyncio.Task[None], ...],
        publication_tasks: tuple[asyncio.Task[TTSSettingsPublication], ...],
        lifecycle_tasks: tuple[asyncio.Task[Any], ...],
    ) -> None:
        failures: list[BaseException] = []
        supervisor = self._audio_cpp_supervisor
        try:
            if supervisor is not None:
                try:
                    begin_terminal_shutdown = getattr(
                        supervisor,
                        "begin_terminal_shutdown",
                        None,
                    )
                    if callable(begin_terminal_shutdown):
                        await begin_terminal_shutdown(self._shutdown_deadline)
                except BaseException as error:
                    failures.append(error)
            try:
                await asyncio.shield(registry_close_task)
            except BaseException as error:
                failures.append(error)

            late_operation_tasks = tuple(
                cleanup_task
                for operation in tuple(self._admitted_operations)
                if (cleanup_task := operation.start_shutdown_cleanup()) is not None
                and cleanup_task not in operation_tasks
            )
            responses = tuple(self._responses)
            response_tasks = [response.start_close() for response in responses]
            resource_tasks = [
                response.start_resource_release() for response in responses
            ]
            registry_wait_task = asyncio.create_task(self.registry.wait_closed())
            terminal_shutdown_tasks = (
                registry_wait_task,
                *operation_tasks,
                *resource_tasks,
                *publication_tasks,
                *lifecycle_tasks,
            )
            results = await asyncio.gather(
                *terminal_shutdown_tasks,
                *late_operation_tasks,
                return_exceptions=True,
            )
            for publication_lease in tuple(self._settings_publication_leases):
                try:
                    await publication_lease.release()
                except BaseException as error:
                    failures.append(error)
                else:
                    self._settings_publication_leases.discard(publication_lease)
            # A still-executing operation may later produce the primary failure.
            # Its joined resource error remains available to that cleanup path and
            # must not be promoted ahead of the unfinished execution.
            failures.extend(
                result
                for result in results[: len(terminal_shutdown_tasks)]
                if isinstance(result, BaseException)
            )
            await asyncio.sleep(0)
            for response_task in response_tasks:
                if not response_task.done():
                    response_task.add_done_callback(self._observe_shutdown_result)
                    continue
                try:
                    response_task.result()
                except BaseException as error:
                    failures.append(error)
            if self._clone_materializer is not None:
                try:
                    await self._clone_materializer.close()
                except BaseException as error:
                    failures.append(error)
        except BaseException as error:
            failures.append(error)
        finally:
            if supervisor is not None:
                try:
                    await supervisor.close()
                except BaseException as error:
                    failures.append(error)
                try:
                    await supervisor.wait_closed()
                except BaseException as error:
                    failures.append(error)
        if failures:
            raise _sanitized_shutdown_error(*failures) from None

    def _manage_response(
        self,
        response: TTSAudioResponse,
        resources: _OperationResources,
        protected_close_chain: bool = False,
    ) -> _ManagedAudioResponse:
        managed_response = _ManagedAudioResponse(
            response,
            resources,
            self._responses.discard,
            protected_close_chain=protected_close_chain,
        )
        self._responses.add(managed_response)
        return managed_response

    @staticmethod
    def _task_acquired_slot(task: asyncio.Task[bool]) -> bool:
        if not task.done() or task.cancelled():
            return False
        try:
            return task.result()
        except BaseException:
            return False

    @staticmethod
    def _observe_shutdown_result(task: asyncio.Task[None]) -> None:
        try:
            task.exception()
        except BaseException:
            pass

    @staticmethod
    def _observe_background_result(task: asyncio.Task[Any]) -> None:
        try:
            task.exception()
        except BaseException:
            pass


def _isolate_progress_sink(
    progress_sink: ProgressSink | None,
) -> ProgressSink | None:
    if progress_sink is None:
        return None

    async def report(progress: TTSProgress) -> None:
        try:
            await progress_sink(progress)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("TTS progress sink failed")

    return report


_bound_tts_service: TTSService | None = None
_bound_tts_close_service: TTSService | None = None
_bound_tts_close_task: asyncio.Task[None] | None = None


def bind_tts_service(service: TTSService) -> None:
    """Bind the application-owned TTS service.

    Args:
        service: Service instance owned by the application lifecycle.

    Raises:
        RuntimeError: If a different service is already bound.
    """
    global _bound_tts_service
    if _bound_tts_service is not None and _bound_tts_service is not service:
        raise RuntimeError("A different TTS service is already bound")
    _bound_tts_service = service


async def get_tts_service(
    app_config: Mapping[str, Any] | None = None,
) -> TTSService:
    """Return the explicitly bound service without retaining caller config.

    Args:
        app_config: Compatibility argument that is intentionally ignored.

    Returns:
        The application-owned service.

    Raises:
        RuntimeError: If no service is bound.
    """
    del app_config
    if _bound_tts_service is None:
        raise RuntimeError("The application TTS service is not bound")
    return _bound_tts_service


def reset_tts_service_binding(
    *,
    expected: TTSService | None = None,
) -> None:
    """Clear the binding when it is absent or identical to the expected owner.

    Args:
        expected: Optional service identity allowed to clear the binding.

    Raises:
        RuntimeError: If another service currently owns the binding.
    """
    global _bound_tts_service
    if (
        expected is not None
        and _bound_tts_service is not None
        and _bound_tts_service is not expected
    ):
        raise RuntimeError("Refusing to reset a different TTS service")
    _bound_tts_service = None


async def _close_bound_service(service: TTSService) -> None:
    global _bound_tts_close_service, _bound_tts_close_task
    try:
        try:
            await service.close()
        except BaseException as error:
            await _cleanup_preserving_primary(service.wait_closed, error)
            raise
        else:
            await service.wait_closed()
    finally:
        try:
            reset_tts_service_binding(expected=service)
        finally:
            if _bound_tts_close_task is asyncio.current_task():
                _bound_tts_close_service = None
                _bound_tts_close_task = None


async def close_tts_resources() -> None:
    """Close the bound service before releasing its application binding."""
    global _bound_tts_close_service, _bound_tts_close_task
    service = _bound_tts_service
    if service is None:
        return
    close_task = _bound_tts_close_task
    if close_task is None or _bound_tts_close_service is not service:
        close_task = asyncio.create_task(_close_bound_service(service))
        _bound_tts_close_service = service
        _bound_tts_close_task = close_task
    await _join_retained_task(close_task)
