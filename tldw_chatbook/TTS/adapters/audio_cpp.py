"""Bounded discovery adapter for an external audio.cpp HTTP server."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Literal
from unicodedata import category
from uuid import uuid4

import httpx

from tldw_chatbook.TTS._async_lifecycle import join_retained_task
from tldw_chatbook.TTS.adapter_types import (
    AudioCppCloneCapabilityAdmission,
    ProgressSink,
    ProviderHealth,
    TTSAudioResponse,
    TTSModelInfo,
    TTSOperationCode,
    TTSOperationError,
    TTSProgress,
    TTSProviderCatalog,
    TTSRequest,
    TTSVoiceDiscoveryResult,
    _AdmittedAudioCppCloneRequest,
    _new_audio_cpp_clone_capability,
)
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppManagedSetupSource,
    AudioCppSettingsConfig,
)
from tldw_chatbook.TTS.audio_cpp_guided_launch import (
    AudioCppGeneratedLaunchArtifact,
    AudioCppGuidedLaunchError,
    materialize_audio_cpp_guided_launch,
    take_audio_cpp_guided_cleanup_owner,
)
from tldw_chatbook.TTS.audio_cpp_contract import (
    AudioCppContractError,
    AudioCppModel,
    Pcm16WavInfo,
    TimingMetadata,
    parse_health_response,
    parse_models_response,
    parse_server_busy_response,
    parse_timing_headers,
    parse_voices_response,
    validate_pcm16_wav,
)
from tldw_chatbook.TTS.audio_cpp_managed_config import (
    AudioCppExpectedModel,
    AudioCppManagedLaunchConfig,
    validate_audio_cpp_managed_launch,
)
from tldw_chatbook.TTS.audio_cpp_recipes import (
    AUDIO_CPP_RECIPE_REGISTRY,
    AudioCppPackageRecipe,
)
from tldw_chatbook.TTS.audio_cpp_supervisor import (
    _AUDIO_CPP_SUPERVISOR_OWNER_TOKEN,
    AudioCppGenerationHooks,
    AudioCppProcessAdmissionSnapshot,
    AudioCppReadyEndpoint,
    AudioCppSupervisor,
    AudioCppTTSCapability,
)
from tldw_chatbook.TTS.profile_reference_materialization import (
    TTSCloneMaterializationError,
    TTSCloneReferenceMaterialization,
)
from tldw_chatbook.TTS.profile_reference_types import TTSCloneRecipeRequirement

_PROVIDER_ID = "audio_cpp"
_TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_GET_ATTEMPTS = 2
_MAX_CONTENT_LENGTH_DIGITS = len(str(sys.maxsize))
_MAX_VOICE_CACHE_ENTRIES = 32
_MAX_VOICE_CACHE_BYTES = 8 * 1024 * 1024
_VOICE_CACHE_ENTRY_OVERHEAD_BYTES = 256
_UNSAFE_VOICE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn"})
_ACCEPTED_AUDIO_CONTENT_TYPES = frozenset(
    {"audio/wav", "audio/x-wav", "application/octet-stream"}
)
_HTTP_LOGGER_NAMES = (
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
)
_HTTP_LOG_SUPPRESSION_ACTIVE: ContextVar[bool] = ContextVar(
    "audio_cpp_http_log_suppression_active",
    default=False,
)
_VoiceCacheKey = tuple[int, str]
_SpeechOutcomeKind = Literal[
    "success",
    "server_busy",
    "transient_failure",
    "contract_failure",
    "refresh_models",
    "generation_failure",
    "invalid_audio",
]

_INITIAL_HEALTH = ProviderHealth(
    state="unavailable",
    fresh=False,
    diagnostic="The audio.cpp server is unavailable",
    retryable=True,
    recovery_action="retry",
)
_AVAILABLE_HEALTH = ProviderHealth(state="available", fresh=True)
_NOT_CONFIGURED_HEALTH = ProviderHealth(
    state="not_configured",
    fresh=True,
    diagnostic="No audio.cpp TTS models are configured",
    recovery_action="check_server",
)
_TRANSIENT_FAILURE_HEALTH = _INITIAL_HEALTH
_CONTRACT_FAILURE_HEALTH = ProviderHealth(
    state="unavailable",
    fresh=False,
    diagnostic="The audio.cpp server response is incompatible",
    recovery_action="check_server",
)
_CLOSED_HEALTH = ProviderHealth(
    state="closed",
    fresh=False,
    diagnostic="The audio.cpp adapter is closed",
)


class _TransientHttpFailure(Exception):
    """Internal value-free marker for a retryable safe-GET failure."""


class _HttpContractFailure(Exception):
    """Internal value-free marker for a non-retryable HTTP contract failure."""


@dataclass(frozen=True, slots=True)
class _OperationFailure:
    code: TTSOperationCode
    message: str
    retryable: bool
    recovery_action: str


_REQUEST_INVALID = _OperationFailure(
    code="request_invalid",
    message="The audio.cpp speech request is invalid",
    retryable=False,
    recovery_action="edit_request",
)
_CONNECTION_UNAVAILABLE = _OperationFailure(
    code="connection_unavailable",
    message="The audio.cpp server is unavailable",
    retryable=True,
    recovery_action="retry",
)
_CLOSED_UNAVAILABLE = _OperationFailure(
    code="connection_unavailable",
    message="The audio.cpp server is unavailable",
    retryable=False,
    recovery_action="check_server",
)
_CONTRACT_INCOMPATIBLE = _OperationFailure(
    code="contract_incompatible",
    message="The audio.cpp server response is incompatible",
    retryable=False,
    recovery_action="check_server",
)
_NOT_CONFIGURED = _OperationFailure(
    code="not_configured",
    message="No audio.cpp TTS models are configured",
    retryable=False,
    recovery_action="configure_server",
)
_MODEL_INVALID = _OperationFailure(
    code="model_invalid",
    message="The requested audio.cpp model is unavailable",
    retryable=False,
    recovery_action="refresh_models",
)
_SERVER_BUSY = _OperationFailure(
    code="server_busy",
    message="The audio.cpp server is busy",
    retryable=True,
    recovery_action="retry",
)
_GENERATION_FAILED = _OperationFailure(
    code="generation_failed",
    message="audio.cpp could not generate speech",
    retryable=False,
    recovery_action="check_server",
)
_AUDIO_RESPONSE_INVALID = _OperationFailure(
    code="audio_response_invalid",
    message="audio.cpp returned invalid audio",
    retryable=False,
    recovery_action="check_server",
)
_GENERATION_TIMEOUT = _OperationFailure(
    code="generation_timeout",
    message="audio.cpp speech generation timed out",
    retryable=True,
    recovery_action="retry",
)
_MANAGED_CONFIGURATION_INVALID = _OperationFailure(
    code="configuration_invalid",
    message="Managed audio.cpp configuration is invalid",
    retryable=False,
    recovery_action="open_settings",
)
_MANAGED_PORT_UNAVAILABLE = _OperationFailure(
    code="port_in_use",
    message="A private audio.cpp loopback port is unavailable",
    retryable=True,
    recovery_action="retry",
)
_MANAGED_ARTIFACT_FAILURE = _OperationFailure(
    code="process_spawn_failed",
    message="The audio.cpp server could not be started",
    retryable=True,
    recovery_action="retry",
)
_MANAGED_CLEANUP_FAILURE_MESSAGE = "audio.cpp generation cleanup failed"
_DEPENDENCY_CHANGED = _OperationFailure(
    code="dependency_changed",
    message="The clone voice dependency changed",
    retryable=False,
    recovery_action="open_settings",
)


@dataclass(frozen=True, slots=True)
class _VoiceCacheEntry:
    result: TTSVoiceDiscoveryResult
    estimated_bytes: int


@dataclass(frozen=True, slots=True)
class _SpeechOutcome:
    kind: _SpeechOutcomeKind
    audio: bytes | None = None
    wav_info: Pcm16WavInfo | None = None
    timing: TimingMetadata | None = None


@dataclass(slots=True)
class _ManagedGenerationBundle:
    """Adapter-owned HTTP resources for one exact managed process generation."""

    process_generation: int
    request_client: httpx.AsyncClient
    health_client: httpx.AsyncClient
    expected_models: tuple[AudioCppExpectedModel, ...] = ()
    cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    request_client_closed: bool = False
    health_client_closed: bool = False
    supervisor_cleanup_failed: bool = False

    async def close_remaining(self) -> None:
        """Close unfinished clients once and retain failed steps for retry."""
        failed = False
        cancellation: asyncio.CancelledError | None = None
        async with self.cleanup_lock:
            if not self.request_client_closed:
                try:
                    await self.request_client.aclose()
                except asyncio.CancelledError as error:
                    cancellation = error
                except BaseException:
                    failed = True
                else:
                    self.request_client_closed = True
            if not self.health_client_closed:
                try:
                    await self.health_client.aclose()
                except asyncio.CancelledError as error:
                    cancellation = cancellation or error
                except BaseException:
                    failed = True
                else:
                    self.health_client_closed = True
        if cancellation is not None:
            raise cancellation
        if failed:
            raise RuntimeError(_MANAGED_CLEANUP_FAILURE_MESSAGE)

    async def supervisor_cleanup(self) -> None:
        """Run generation cleanup while remembering swallowed supervisor errors."""
        cleanup_failed = False
        try:
            await self.close_remaining()
        except asyncio.CancelledError:
            raise
        except BaseException:
            cleanup_failed = True
        if cleanup_failed:
            self.supervisor_cleanup_failed = True
            raise RuntimeError(_MANAGED_CLEANUP_FAILURE_MESSAGE)


async def _complete_wav_stream(audio: bytes) -> AsyncIterator[bytes]:
    """Yield one already validated complete WAV without another body copy."""
    yield audio


def _estimate_voice_cache_entry_bytes(
    key: _VoiceCacheKey,
    voices: tuple[str, ...],
) -> int:
    """Conservatively estimate retained Python memory for one cache entry."""
    revision, model_id = key
    return (
        _VOICE_CACHE_ENTRY_OVERHEAD_BYTES
        + sys.getsizeof(key)
        + sys.getsizeof(revision)
        + sys.getsizeof(model_id)
        + sys.getsizeof(voices)
        + sum(sys.getsizeof(voice) for voice in voices)
    )


class _HttpxPrivacyFilter(logging.Filter):
    """Suppress HTTP-library records only inside this adapter's request."""

    def filter(self, record: logging.LogRecord) -> bool:
        del record
        return not _HTTP_LOG_SUPPRESSION_ACTIVE.get()


class AudioCppAdapter:
    """Discover metadata and synthesize bounded WAVs from audio.cpp.

    External construction creates its owned HTTP client without network I/O.
    Managed construction remains client-, process-, and network-lazy.

    Args:
        config: Validated immutable audio.cpp configuration.
        transport: Optional fake HTTP transport for deterministic tests.
        supervisor: App-scoped managed process owner. External mode ignores it.
        guided_settings: Complete structured Managed settings for lazy launch.
    """

    def __init__(
        self,
        config: AudioCppConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        supervisor: AudioCppSupervisor | None = None,
        guided_settings: AudioCppSettingsConfig | None = None,
    ) -> None:
        self._config = config
        self._guided_settings = guided_settings
        self._transport = transport
        self._supervisor = supervisor
        self._client: httpx.AsyncClient | None = (
            self._new_request_client(config.base_url)
            if config.mode == "external"
            else None
        )
        self._managed_bundle: _ManagedGenerationBundle | None = None
        self._managed_launch: AudioCppManagedLaunchConfig | None = None
        self._pending_guided_cleanup: AudioCppGeneratedLaunchArtifact | None = None
        self._managed_preparation_lock = asyncio.Lock()
        self._managed_process_generation: int | None = None
        self._managed_catalog_process_generation: int | None = None
        self._managed_catalog_observation_version: int | None = None
        self._managed_stop_complete = False
        self._managed_required_admission: ContextVar[
            AudioCppProcessAdmissionSnapshot | None
        ] = ContextVar(
            f"audio_cpp_managed_required_admission_{id(self)}",
            default=None,
        )
        self._catalog = TTSProviderCatalog(
            provider_id=_PROVIDER_ID,
            revision=0,
            health=_INITIAL_HEALTH,
            models=(),
        )
        self._refresh_lock = asyncio.Lock()
        self._refresh_generation = 0
        self._voice_cache: OrderedDict[_VoiceCacheKey, _VoiceCacheEntry] = OrderedDict()
        self._voice_cache_bytes = 0
        self._voice_generation: dict[_VoiceCacheKey, int] = {}
        self._voice_locks: dict[_VoiceCacheKey, asyncio.Lock] = {}
        self._voice_lock_users: dict[_VoiceCacheKey, int] = {}
        self._voice_shared_results: dict[
            _VoiceCacheKey,
            tuple[int, TTSVoiceDiscoveryResult],
        ] = {}
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False
        self._clone_adapter_identity = object()
        self._clone_capabilities: dict[object, AudioCppCloneCapabilityAdmission] = {}
        self._httpx_privacy_filter = _HttpxPrivacyFilter()
        self._http_log_suppression_users = 0
        self._http_log_client_closed = False
        self._http_log_filters_installed = True
        for logger_name in _HTTP_LOGGER_NAMES:
            logging.getLogger(logger_name).addFilter(self._httpx_privacy_filter)

    def _new_request_client(self, base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url,
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
            headers={"Accept-Encoding": "identity"},
            timeout=httpx.Timeout(
                connect=self._config.connect_timeout_seconds,
                read=None,
                write=None,
                pool=None,
            ),
        )

    def _new_health_client(self, base_url: str) -> httpx.AsyncClient:
        timeout = min(
            self._config.connect_timeout_seconds,
            self._config.managed_health_check_interval_seconds,
        )
        return httpx.AsyncClient(
            base_url=base_url,
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
            headers={"Accept-Encoding": "identity"},
            timeout=httpx.Timeout(timeout),
        )

    @contextmanager
    def managed_admission(
        self,
        snapshot: AudioCppProcessAdmissionSnapshot | None,
    ) -> Iterator[None]:
        """Fence one deliberate managed operation to an observed generation."""
        token = self._managed_required_admission.set(snapshot)
        try:
            yield
        finally:
            self._managed_required_admission.reset(token)

    async def ensure_ready(self) -> None:
        """Perform the first authoritative refresh and cache fresh readiness."""
        await self._refresh_catalog(force=False)

    def admitted_outbound_endpoint(self) -> str:
        """Return the exact base URL bound to this ready adapter generation."""
        client = self._require_request_client()
        return str(client.base_url).rstrip("/")

    def preflight_clone_source(self) -> None:
        """Reject non-Guided local-reference authority without side effects."""
        if not self._clone_source_authorized():
            raise self._operation_error(
                _MANAGED_CONFIGURATION_INVALID,
                uuid4().hex,
            ) from None

    def preflight_clone_dependency(
        self,
        requirement: TTSCloneRecipeRequirement,
    ) -> None:
        """Compare exact Guided recipe/model configuration without readiness work."""

        self.preflight_clone_source()
        if type(requirement) is not TTSCloneRecipeRequirement:
            raise self._operation_error(_DEPENDENCY_CHANGED, uuid4().hex) from None
        recipe = self._guided_recipe_for_model(requirement.model_id)
        if (
            recipe is None
            or recipe.recipe_id != requirement.recipe_id
            or recipe.recipe_revision != requirement.recipe_revision
            or "clone" not in recipe.capabilities
        ):
            raise self._operation_error(_DEPENDENCY_CHANGED, uuid4().hex) from None

    def preflight_clone_request_dependency(
        self,
        request: TTSRequest,
        requirement: TTSCloneRecipeRequirement,
    ) -> None:
        """Apply clone voice/reference policy to one exact resolved request."""

        self.preflight_clone_dependency(requirement)
        if type(request) is not TTSRequest or request.model_id != requirement.model_id:
            raise self._operation_error(_DEPENDENCY_CHANGED, uuid4().hex) from None
        recipe = self._guided_recipe_for_model(request.model_id)
        if recipe is None or not recipe.admits_voice_reference(
            has_voice=request.voice is not None,
            has_reference=True,
        ):
            raise self._operation_error(_DEPENDENCY_CHANGED, uuid4().hex) from None

    def admit_clone_capability(
        self,
        request: TTSRequest,
    ) -> AudioCppCloneCapabilityAdmission:
        """Issue single-use authority for one ready Guided model generation.

        Args:
            request: Exact public request already bound to this adapter lease.

        Returns:
            Opaque single-use authority for the matching Guided process and
            recipe generation.

        Raises:
            TTSOperationError: If the source, request, catalog, recipe, or
                managed process generation cannot admit clone synthesis.
        """
        self.preflight_clone_source()
        if type(request) is not TTSRequest or not self._valid_speech_request(request):
            raise self._operation_error(_REQUEST_INVALID, uuid4().hex) from None
        if not self._catalog_contains(self._catalog, request.model_id):
            raise self._operation_error(_MODEL_INVALID, uuid4().hex) from None
        readiness = self._readiness_failure()
        if readiness is not None:
            raise self._operation_error(readiness, uuid4().hex) from None
        recipe = self._guided_recipe_for_model(request.model_id)
        if recipe is None:
            raise self._operation_error(
                _MANAGED_CONFIGURATION_INVALID,
                uuid4().hex,
            ) from None
        if "clone" not in recipe.capabilities or not recipe.admits_voice_reference(
            has_voice=request.voice is not None,
            has_reference=True,
        ):
            raise self._operation_error(_REQUEST_INVALID, uuid4().hex) from None
        process_generation = self._current_clone_process_generation()
        if process_generation is None:
            raise self._operation_error(_CONNECTION_UNAVAILABLE, uuid4().hex) from None
        capability_token = object()
        capability = _new_audio_cpp_clone_capability(
            adapter_identity=self._clone_adapter_identity,
            capability_token=capability_token,
            model_id=request.model_id,
            recipe_id=recipe.recipe_id,
            recipe_revision=recipe.recipe_revision,
            process_generation=process_generation,
            request=request,
        )
        self._clone_capabilities[capability_token] = capability
        return capability

    def release_clone_capability(
        self,
        capability: AudioCppCloneCapabilityAdmission,
    ) -> None:
        """Discard one exact unused capability without affecting other work."""
        if not isinstance(capability, AudioCppCloneCapabilityAdmission):
            return
        token = capability._capability_token
        if (
            capability._adapter_identity is self._clone_adapter_identity
            and self._clone_capabilities.get(token) is capability
        ):
            self._clone_capabilities.pop(token, None)

    async def get_catalog(self, refresh: bool = False) -> TTSProviderCatalog:
        """Return the immutable catalog, optionally forcing one refresh."""
        if self._config.mode == "managed" and not refresh:
            return self._catalog
        await self._refresh_catalog(force=refresh)
        return self._catalog

    def catalog_publication_evidence(
        self,
        catalog: TTSProviderCatalog,
    ) -> tuple[int | None, int | None] | None:
        """Fence service publication to this adapter's current catalog evidence.

        Args:
            catalog: Catalog instance proposed for service-level publication.

        Returns:
            Managed process and observation generations, ``(None, None)`` for
            External or stale evidence, or ``None`` when publication is unsafe.
        """
        if self._closed or catalog is not self._catalog:
            return None
        if self._config.mode != "managed" or not catalog.health.fresh:
            return (None, None)
        process_generation = self._managed_catalog_process_generation
        observation_version = self._managed_catalog_observation_version
        if process_generation is None or observation_version is None:
            return None
        return (process_generation, observation_version)

    async def get_voices(
        self,
        model_id: str,
        refresh: bool = False,
    ) -> tuple[str, ...]:
        """Return the compatibility tuple projection of voice observation."""
        result = await self.observe_voices(model_id, refresh=refresh)
        return result.voices if result.state == "complete" else ()

    async def observe_voices(
        self,
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult:
        """Observe optional voices without discarding discovery authority."""
        if self._config.mode == "managed" and not refresh:
            catalog = self._catalog
            if (
                self._closed
                or not catalog.health.fresh
                or self._managed_catalog_process_generation
                != self._managed_process_generation
            ):
                return self._unverified_voice_result(model_id, catalog.revision)
            if not self._catalog_contains(catalog, model_id):
                return TTSVoiceDiscoveryResult(
                    provider_id=_PROVIDER_ID,
                    model_id=model_id,
                    catalog_revision=catalog.revision,
                    voices=(),
                    state="model_missing",
                )
            cached = self._cached_voice_result((catalog.revision, model_id))
            return cached or self._unverified_voice_result(
                model_id,
                catalog.revision,
            )

        await self.ensure_ready()
        force = refresh

        while True:
            catalog = self._catalog
            if self._closed or not catalog.health.fresh:
                return self._unverified_voice_result(model_id, catalog.revision)
            if not self._catalog_contains(catalog, model_id):
                return TTSVoiceDiscoveryResult(
                    provider_id=_PROVIDER_ID,
                    model_id=model_id,
                    catalog_revision=catalog.revision,
                    voices=(),
                    state="model_missing",
                )

            key = (catalog.revision, model_id)
            started_generation = self._voice_generation.get(key, 0)
            lock = self._voice_locks.setdefault(key, asyncio.Lock())
            self._voice_lock_users[key] = self._voice_lock_users.get(key, 0) + 1
            try:
                async with lock:
                    current = self._catalog
                    if (
                        current.revision != catalog.revision
                        or not self._catalog_contains(current, model_id)
                    ):
                        continue
                    if self._closed or not current.health.fresh:
                        return self._unverified_voice_result(
                            model_id,
                            current.revision,
                        )

                    current_generation = self._voice_generation.get(key, 0)
                    if current_generation != started_generation:
                        shared = self._voice_shared_results.get(key)
                        if shared is not None and shared[0] == current_generation:
                            return shared[1]
                        cached = self._cached_voice_result(key)
                        if cached is not None:
                            return cached
                    if not force:
                        cached = self._cached_voice_result(key)
                        if cached is not None:
                            return cached

                    result = await self._fetch_voices(
                        model_id,
                        catalog.revision,
                    )
                    current = self._catalog
                    if self._closed or not current.health.fresh:
                        return self._unverified_voice_result(
                            model_id,
                            current.revision,
                        )
                    if current.revision != catalog.revision:
                        continue

                    next_generation = current_generation + 1
                    self._voice_generation[key] = next_generation
                    self._voice_shared_results[key] = (
                        next_generation,
                        result,
                    )
                    self._cache_voice_result(key, result)
                    return result
            finally:
                self._release_voice_lock_user(key, lock)

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        """Validate and perform one non-retried complete-WAV speech request."""
        return await self._synthesize_request(
            request,
            progress_sink,
            clone_request=None,
        )

    async def synthesize_clone(
        self,
        request: _AdmittedAudioCppCloneRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        """Perform one exact internal Guided clone request."""
        failure = self._clone_request_failure(request, consume=False)
        if failure is not None:
            if type(request) is _AdmittedAudioCppCloneRequest:
                try:
                    capability = request.capability
                except Exception:
                    capability = None
                if type(capability) is AudioCppCloneCapabilityAdmission:
                    self.release_clone_capability(capability)
            raise self._operation_error(failure, uuid4().hex) from None
        return await self._synthesize_request(
            request.request,
            progress_sink,
            clone_request=request,
        )

    async def _synthesize_request(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None,
        *,
        clone_request: _AdmittedAudioCppCloneRequest | None,
    ) -> TTSAudioResponse:
        operation_id = uuid4().hex
        failure: _OperationFailure | None = None
        outcome: _SpeechOutcome | None = None
        payload: dict[str, str] | None = None
        validated_voice_ref: Path | None = None
        process_generation: int | None = None

        if type(request) is not TTSRequest or not self._valid_speech_request(request):
            failure = _REQUEST_INVALID
        elif clone_request is not None:
            failure = self._clone_request_failure(clone_request, consume=False)
            if failure is None:
                failure = self._readiness_failure()
        else:
            await self.ensure_ready()
            failure = self._readiness_failure()

        if failure is None and clone_request is None and self._uses_guided_launch():
            recipe = self._guided_recipe_for_model(request.model_id)
            if recipe is not None and not recipe.admits_voice_reference(
                has_voice=request.voice is not None,
                has_reference=False,
            ):
                failure = _REQUEST_INVALID

        if failure is None and not self._catalog_contains(
            self._catalog,
            request.model_id,
        ):
            if clone_request is not None:
                failure = _MODEL_INVALID
            else:
                await self._refresh_catalog(force=True)
                failure = self._readiness_failure()
                if failure is None and not self._catalog_contains(
                    self._catalog,
                    request.model_id,
                ):
                    failure = _MODEL_INVALID

        if failure is None:
            process_generation = self._managed_process_generation
            await self._report_progress(
                progress_sink,
                TTSProgress(status="Generating", fraction=None),
            )
            if self._closed:
                failure = _CLOSED_UNAVAILABLE
            else:
                if clone_request is not None:
                    failure = self._clone_request_failure(
                        clone_request,
                        consume=True,
                    )
                    if failure is None:
                        try:
                            validated_voice_ref = await (
                                clone_request.materialization.validated_voice_ref()
                            )
                        except TTSCloneMaterializationError:
                            failure = _REQUEST_INVALID
                    supervisor = self._supervisor
                    suppress_diagnostics = (
                        None
                        if supervisor is None
                        else getattr(supervisor, "suppress_clone_diagnostics", None)
                    )
                    if failure is None and (
                        not callable(suppress_diagnostics)
                        or not suppress_diagnostics(clone_request.process_generation)
                    ):
                        failure = _CONNECTION_UNAVAILABLE
                if failure is not None:
                    payload = None
                else:
                    payload = {
                        "model": request.model_id,
                        "input": request.text,
                        "response_format": "wav",
                    }
                    if request.voice is not None:
                        payload["voice"] = request.voice
                    if clone_request is not None:
                        assert validated_voice_ref is not None
                        payload["voice_ref"] = str(validated_voice_ref)
                        payload["reference_text"] = (
                            clone_request.materialization.reference_text
                        )

            if failure is None and payload is not None:
                suppression_token = self._begin_http_log_suppression()
                try:
                    try:
                        async with asyncio.timeout(
                            self._config.synthesis_timeout_seconds
                        ):
                            outcome = await self._post_speech(payload)
                    except asyncio.CancelledError:
                        raise
                    except TimeoutError:
                        failure = (
                            _CLOSED_UNAVAILABLE if self._closed else _GENERATION_TIMEOUT
                        )
                    except (
                        _TransientHttpFailure,
                        httpx.StreamError,
                        httpx.TransportError,
                    ):
                        if self._closed:
                            failure = _CLOSED_UNAVAILABLE
                        else:
                            self._mark_catalog_stale(_TRANSIENT_FAILURE_HEALTH)
                            failure = _CONNECTION_UNAVAILABLE
                    except RuntimeError:
                        if not self._closed:
                            raise
                        failure = _CLOSED_UNAVAILABLE
                finally:
                    self._end_http_log_suppression(suppression_token)

        if failure is None and outcome is not None:
            if self._closed:
                failure = _CLOSED_UNAVAILABLE
            elif outcome.kind == "transient_failure":
                self._mark_catalog_stale(_TRANSIENT_FAILURE_HEALTH)
                failure = _CONNECTION_UNAVAILABLE
            elif outcome.kind == "contract_failure":
                self._mark_catalog_stale(_CONTRACT_FAILURE_HEALTH)
                failure = _CONTRACT_INCOMPATIBLE
            elif outcome.kind == "server_busy":
                failure = _SERVER_BUSY
            elif outcome.kind == "generation_failure":
                failure = _GENERATION_FAILED
            elif outcome.kind == "invalid_audio":
                failure = _AUDIO_RESPONSE_INVALID
            elif outcome.kind == "refresh_models":
                await self._refresh_catalog(force=True)
                failure = self._readiness_failure()
                if failure is _NOT_CONFIGURED:
                    failure = _MODEL_INVALID
                elif failure is None:
                    failure = (
                        _GENERATION_FAILED
                        if self._catalog_contains(
                            self._catalog,
                            request.model_id,
                        )
                        else _MODEL_INVALID
                    )

        if failure is not None:
            del request, progress_sink, payload, outcome
            raise self._operation_error(failure, operation_id)

        if (
            outcome is None
            or outcome.kind != "success"
            or outcome.audio is None
            or outcome.wav_info is None
        ):
            del request, progress_sink, payload, outcome
            raise self._operation_error(_GENERATION_FAILED, operation_id)

        metadata: dict[str, str | int | float | bool | None] = {
            "adapter": "audio_cpp",
            "contract": "audio_cpp_http_v1",
            "delivery": "complete_wav",
            "channels": outcome.wav_info.channels,
            "frame_count": outcome.wav_info.frame_count,
            "data_size": outcome.wav_info.data_size,
        }
        if process_generation is not None:
            metadata["process_generation"] = process_generation
        if outcome.timing is not None:
            metadata.update(outcome.timing)
        response = TTSAudioResponse(
            provider_id=_PROVIDER_ID,
            model_id=request.model_id,
            audio_format="wav",
            content_type="audio/wav",
            byte_stream=_complete_wav_stream(outcome.audio),
            sample_rate=outcome.wav_info.sample_rate,
            metadata=metadata,
        )
        await self._report_progress(
            progress_sink,
            TTSProgress(status="Complete", fraction=1.0),
        )
        return response

    def _valid_speech_request(self, request: TTSRequest) -> bool:
        text = request.text
        speed = request.speed
        voice = request.voice
        if request.provider_id != _PROVIDER_ID:
            return False
        if (
            not isinstance(text, str)
            or len(text) > self._config.max_input_characters
            or not text.strip()
        ):
            return False
        if request.response_format != "wav":
            return False
        if isinstance(speed, bool) or not isinstance(speed, Real) or speed != 1:
            return False
        if request.options:
            return False
        if voice is None:
            return True
        return (
            isinstance(voice, str)
            and bool(voice)
            and len(voice) <= self._config.max_identifier_characters
            and voice == voice.strip()
            and not any(
                category(character) in _UNSAFE_VOICE_CATEGORIES for character in voice
            )
        )

    def _clone_source_authorized(self) -> bool:
        return (
            not self._closed
            and self._config.mode == "managed"
            and self._supervisor is not None
            and getattr(self._supervisor, "_application_owner_token", None)
            is _AUDIO_CPP_SUPERVISOR_OWNER_TOKEN
            and self._uses_guided_launch()
        )

    def _guided_recipe_for_model(
        self,
        model_id: str,
    ) -> AudioCppPackageRecipe | None:
        settings = self._guided_settings
        if settings is None:
            return None
        accepted = next(
            (
                package
                for package in settings.guided_packages
                if package.public_model_id == model_id
            ),
            None,
        )
        if accepted is None:
            return None
        try:
            return AUDIO_CPP_RECIPE_REGISTRY.validate_accepted(accepted)
        except (TypeError, ValueError):
            return None

    def _current_clone_process_generation(self) -> int | None:
        supervisor = self._supervisor
        bundle = self._managed_bundle
        process_generation = self._managed_process_generation
        if (
            supervisor is None
            or bundle is None
            or process_generation is None
            or bundle.process_generation != process_generation
        ):
            return None
        try:
            snapshot = supervisor.snapshot()
        except Exception:
            return None
        if (
            snapshot.state not in {"running", "draining"}
            or snapshot.process_generation != process_generation
        ):
            return None
        return process_generation

    def _clone_request_failure(
        self,
        request: _AdmittedAudioCppCloneRequest,
        *,
        consume: bool,
    ) -> _OperationFailure | None:
        if type(request) is not _AdmittedAudioCppCloneRequest:
            return _REQUEST_INVALID
        try:
            service_sealed = request._is_service_sealed()
            capability = request.capability
            base_request = request.request
            materialization = request.materialization
            provider_revision = request.provider_revision
            applied_generation = request.applied_provider_generation
            recipe_id = request.recipe_id
            recipe_revision = request.recipe_revision
            process_generation = request.process_generation
        except Exception:
            return _REQUEST_INVALID
        if (
            not service_sealed
            or type(base_request) is not TTSRequest
            or not self._valid_speech_request(base_request)
            or type(materialization) is not TTSCloneReferenceMaterialization
            or type(provider_revision) is not int
            or provider_revision < 0
            or type(applied_generation) is not int
            or applied_generation < 0
            or type(capability) is not AudioCppCloneCapabilityAdmission
            or capability._adapter_identity is not self._clone_adapter_identity
            or base_request is not capability._request
            or materialization is not capability._materialization
            or provider_revision != capability._provider_revision
            or applied_generation != capability._applied_provider_generation
            or base_request.model_id != capability.model_id
            or recipe_id != capability.recipe_id
            or recipe_revision != capability.recipe_revision
            or process_generation != capability.process_generation
            or not self._clone_source_authorized()
        ):
            return _REQUEST_INVALID
        assert isinstance(capability, AudioCppCloneCapabilityAdmission)
        assert isinstance(materialization, TTSCloneReferenceMaterialization)
        capability_token = capability._capability_token
        if self._current_clone_process_generation() != process_generation:
            return _CONNECTION_UNAVAILABLE
        try:
            live_owner = materialization._is_live_owner()
        except Exception:
            live_owner = False
        if (
            self._clone_capabilities.get(capability_token) is not capability
            or not live_owner
        ):
            return _REQUEST_INVALID
        recipe = self._guided_recipe_for_model(base_request.model_id)
        if (
            recipe is None
            or recipe.recipe_id != recipe_id
            or recipe.recipe_revision != recipe_revision
            or "clone" not in recipe.capabilities
            or not recipe.admits_voice_reference(
                has_voice=base_request.voice is not None,
                has_reference=True,
            )
        ):
            return _REQUEST_INVALID
        if consume:
            self._clone_capabilities.pop(capability_token, None)
        return None

    def _readiness_failure(self) -> _OperationFailure | None:
        health = self._catalog.health
        if self._closed or health.state == "closed":
            return _CLOSED_UNAVAILABLE
        if health.state == "not_configured":
            return _NOT_CONFIGURED
        if health.state != "available" or not health.fresh:
            return (
                _CONNECTION_UNAVAILABLE if health.retryable else _CONTRACT_INCOMPATIBLE
            )
        if not self._catalog.models:
            return _NOT_CONFIGURED
        return None

    @staticmethod
    async def _report_progress(
        progress_sink: ProgressSink | None,
        progress: TTSProgress,
    ) -> None:
        if progress_sink is None:
            return
        try:
            await progress_sink(progress)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _post_speech(
        self,
        payload: Mapping[str, str],
    ) -> _SpeechOutcome:
        client = self._require_request_client()
        process_generation = self._managed_client_generation(client)
        runtime_failure: RuntimeError | None = None
        client.cookies.clear()
        try:
            async with client.stream(
                "POST",
                "/v1/audio/speech",
                json=payload,
            ) as response:
                client.cookies.clear()
                status = response.status_code
                if status == 200:
                    if not self._accepted_audio_content_type(response):
                        return _SpeechOutcome(kind="invalid_audio")
                    try:
                        body = await self._read_bounded_response(
                            response,
                            max_bytes=self._config.max_response_bytes,
                        )
                        wav_info = validate_pcm16_wav(body)
                    except (
                        AudioCppContractError,
                        _HttpContractFailure,
                        httpx.StreamError,
                    ):
                        return _SpeechOutcome(kind="invalid_audio")
                    return _SpeechOutcome(
                        kind="success",
                        audio=body,
                        wav_info=wav_info,
                        timing=parse_timing_headers(response.headers),
                    )

                if status == 503:
                    try:
                        body = await self._read_bounded_response(
                            response,
                            max_bytes=self._config.max_metadata_bytes,
                        )
                        parse_server_busy_response(
                            body,
                            max_metadata_bytes=self._config.max_metadata_bytes,
                        )
                    except (
                        AudioCppContractError,
                        _HttpContractFailure,
                        httpx.StreamError,
                    ):
                        return _SpeechOutcome(kind="transient_failure")
                    return _SpeechOutcome(kind="server_busy")

                if status == 500:
                    return _SpeechOutcome(kind="refresh_models")
                if status in {408, 429, 502, 504}:
                    return _SpeechOutcome(kind="transient_failure")
                if status == 404 or 300 <= status < 400:
                    return _SpeechOutcome(kind="contract_failure")
                return _SpeechOutcome(kind="generation_failure")
        except RuntimeError as error:
            runtime_failure = error
        finally:
            client.cookies.clear()
        assert runtime_failure is not None
        if self._managed_client_invalidated(client, process_generation):
            raise _TransientHttpFailure
        raise runtime_failure

    def _require_request_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None:
            raise RuntimeError("audio.cpp request client is not bound")
        return client

    @staticmethod
    def _accepted_audio_content_type(response: httpx.Response) -> bool:
        content_types = response.headers.get_list("content-type")
        if len(content_types) != 1:
            return False
        media_type, _separator, _parameters = content_types[0].partition(";")
        return media_type.strip().casefold() in _ACCEPTED_AUDIO_CONTENT_TYPES

    async def close(self) -> None:
        """Seal admission and join the one retained cleanup task."""
        async with self._close_lock:
            if (
                self._config.mode == "managed"
                and self._close_task is not None
                and self._close_task.done()
                and not self._close_task.cancelled()
                and self._close_task.exception() is not None
            ):
                self._close_task = None
            if self._close_task is None:
                self._closed = True
                self._clone_capabilities.clear()
                self._close_task = asyncio.create_task(self._complete_close())
            close_task = self._close_task
        await join_retained_task(close_task)

    async def _complete_close(self) -> None:
        async with self._refresh_lock:
            current = self._catalog
            self._catalog = TTSProviderCatalog(
                provider_id=_PROVIDER_ID,
                revision=current.revision,
                health=_CLOSED_HEALTH,
                models=current.models,
            )
            self._clear_voice_state()

        try:
            if self._config.mode == "managed":
                await self._complete_managed_close()
            else:
                await self._require_request_client().aclose()
        finally:
            self._http_log_client_closed = True
            self._remove_http_log_filters_if_idle()

    async def _complete_managed_close(self) -> None:
        bundle = self._managed_bundle
        if (
            not self._managed_stop_complete
            and self._managed_process_generation is not None
        ):
            supervisor = self._supervisor
            if supervisor is None:
                raise RuntimeError(_MANAGED_CLEANUP_FAILURE_MESSAGE)
            await supervisor.stop(
                expected_process_generation=self._managed_process_generation
            )
            self._managed_stop_complete = True

        if bundle is not None:
            if bundle.supervisor_cleanup_failed:
                bundle.supervisor_cleanup_failed = False
                raise RuntimeError(_MANAGED_CLEANUP_FAILURE_MESSAGE)
            await bundle.close_remaining()

        pending = self._pending_guided_cleanup
        if pending is None:
            return
        cleanup_task = asyncio.create_task(asyncio.to_thread(pending.cleanup))
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(cleanup_task)
            except BaseException:
                pass
            raise
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            raise RuntimeError(_MANAGED_CLEANUP_FAILURE_MESSAGE) from None
        self._pending_guided_cleanup = None

    async def _refresh_catalog(self, *, force: bool) -> None:
        if self._config.mode == "managed":
            catalog_generation = self._managed_catalog_process_generation
            endpoint = await self._ensure_managed_running()
            if self._closed:
                return
            if (
                catalog_generation != endpoint.process_generation
                and self._managed_catalog_process_generation
                == endpoint.process_generation
            ):
                self._managed_catalog_observation_version = endpoint.observation_version
                return
            supervisor = self._supervisor
            process = None if supervisor is None else supervisor.snapshot()
            if (
                process is not None
                and process.state == "draining"
                and self._managed_required_admission.get() is not None
                and self._managed_catalog_process_generation
                == endpoint.process_generation
                and self._catalog.health.fresh
            ):
                return
            bundle = self._managed_bundle
            if (
                bundle is None
                or bundle.process_generation != endpoint.process_generation
            ):
                raise self._operation_error(
                    _CONNECTION_UNAVAILABLE,
                    uuid4().hex,
                ) from None
            capability = await self._refresh_catalog_with_client(
                bundle.request_client,
                force=(
                    force
                    or self._managed_catalog_observation_version
                    != endpoint.observation_version
                ),
                process_generation=endpoint.process_generation,
                raise_on_failure=False,
                expected_models=bundle.expected_models,
            )
            if (
                capability in {"available", "not_configured"}
                and self._managed_catalog_process_generation
                == endpoint.process_generation
            ):
                process = None if supervisor is None else supervisor.snapshot()
                if (
                    process is not None
                    and process.process_generation == endpoint.process_generation
                    and process.state in {"running", "draining"}
                    and process.tts_capability in {"available", "not_configured"}
                    and self._catalog.health.fresh
                ):
                    self._managed_catalog_observation_version = (
                        process.observation_version
                    )
                else:
                    self._managed_catalog_process_generation = None
                    self._mark_catalog_stale(_TRANSIENT_FAILURE_HEALTH)
            return

        await self._refresh_catalog_with_client(
            self._require_request_client(),
            force=force,
            process_generation=None,
            raise_on_failure=False,
        )

    async def _refresh_catalog_with_client(
        self,
        client: httpx.AsyncClient,
        *,
        force: bool,
        process_generation: int | None,
        raise_on_failure: bool,
        expected_models: tuple[AudioCppExpectedModel, ...] = (),
    ) -> AudioCppTTSCapability:
        started_generation = self._refresh_generation
        async with self._refresh_lock:
            if self._closed:
                return "unknown"
            if (
                process_generation is not None
                and self._managed_process_generation != process_generation
            ):
                return "unknown"
            if self._refresh_generation != started_generation:
                return "unknown"
            if (
                not force
                and self._catalog.health.fresh
                and (
                    process_generation is None
                    or self._managed_catalog_process_generation == process_generation
                )
            ):
                return "available" if self._catalog.models else "not_configured"

            previous = self._catalog
            try:
                async with asyncio.timeout(self._config.connect_timeout_seconds):
                    health_body = await self._safe_get("/health", client=client)
                    upstream_health = parse_health_response(
                        health_body,
                        max_metadata_bytes=self._config.max_metadata_bytes,
                        max_identifier_characters=(
                            self._config.max_identifier_characters
                        ),
                        max_models=self._config.max_catalog_models,
                    )
                    models_body = await self._safe_get("/v1/models", client=client)
                    upstream_models = parse_models_response(
                        models_body,
                        max_metadata_bytes=self._config.max_metadata_bytes,
                        max_identifier_characters=(
                            self._config.max_identifier_characters
                        ),
                        max_models=self._config.max_catalog_models,
                        require_speech_tasks=bool(expected_models),
                    )
                    if expected_models and upstream_health.models != len(
                        expected_models
                    ):
                        raise _HttpContractFailure
                    generated_capabilities = self._generated_catalog_capabilities(
                        upstream_models,
                        expected_models,
                    )
                    if generated_capabilities is None:
                        raise _HttpContractFailure
            except asyncio.CancelledError:
                raise
            except (TimeoutError, _TransientHttpFailure):
                if not self._closed:
                    self._catalog = self._failed_catalog(
                        previous,
                        _TRANSIENT_FAILURE_HEALTH,
                    )
                    if process_generation is not None:
                        self._managed_catalog_process_generation = None
                        self._managed_catalog_observation_version = None
                    self._refresh_generation += 1
                if raise_on_failure:
                    raise self._operation_error(
                        _CONNECTION_UNAVAILABLE,
                        uuid4().hex,
                    ) from None
                return "unknown"
            except (_HttpContractFailure, AudioCppContractError):
                if not self._closed:
                    self._catalog = self._failed_catalog(
                        previous,
                        _CONTRACT_FAILURE_HEALTH,
                    )
                    if process_generation is not None:
                        self._managed_catalog_process_generation = None
                        self._managed_catalog_observation_version = None
                    self._refresh_generation += 1
                if raise_on_failure:
                    raise self._operation_error(
                        _CONTRACT_INCOMPATIBLE,
                        uuid4().hex,
                    ) from None
                return "unknown"

            if self._closed:
                return "unknown"
            if (
                process_generation is not None
                and self._managed_process_generation != process_generation
            ):
                return "unknown"
            if self._refresh_generation != started_generation:
                return "unknown"
            models = tuple(
                TTSModelInfo(
                    model_id=model.model_id,
                    display_name=model.model_id,
                    family=model.family,
                    upstream_mode=model.mode,
                    formats=("wav",),
                    voices=(),
                    supports_speed=False,
                    speech_capabilities=(
                        (model.task,)
                        if not expected_models
                        else generated_capabilities[model.model_id]
                    ),
                    supports_options=(),
                    omit_voice_uses_server_default=True,
                )
                for model in upstream_models
            )
            health = _AVAILABLE_HEALTH if models else _NOT_CONFIGURED_HEALTH
            self._catalog = TTSProviderCatalog(
                provider_id=_PROVIDER_ID,
                revision=previous.revision + 1,
                health=health,
                models=models,
            )
            if process_generation is not None:
                self._managed_catalog_process_generation = process_generation
            self._refresh_generation += 1
            self._clear_voice_state()
            return "available" if models else "not_configured"

    @staticmethod
    def _generated_catalog_capabilities(
        upstream_models: tuple[AudioCppModel, ...],
        expected_models: tuple[AudioCppExpectedModel, ...],
    ) -> dict[str, tuple[Literal["tts", "clone"], ...]] | None:
        if not expected_models:
            return {}
        expected_by_id = {model.model_id: model for model in expected_models}
        if (
            len(expected_by_id) != len(expected_models)
            or {model.model_id for model in upstream_models} != expected_by_id.keys()
        ):
            return None
        capabilities: dict[str, tuple[Literal["tts", "clone"], ...]] = {}
        for model in upstream_models:
            expected = expected_by_id[model.model_id]
            if (
                model.family != expected.family
                or model.task != expected.task
                or model.mode != expected.mode
            ):
                return None
            capabilities[model.model_id] = expected.speech_capabilities
        return capabilities

    async def _ensure_managed_running(self) -> AudioCppReadyEndpoint:
        if self._uses_guided_launch():
            async with self._managed_preparation_lock:
                return await self._ensure_managed_running_unlocked()
        return await self._ensure_managed_running_unlocked()

    def _uses_guided_launch(self) -> bool:
        settings = self._guided_settings
        return (
            settings is not None
            and self._config.mode == "managed"
            and settings.mode == "managed"
            and settings.managed_setup_source is AudioCppManagedSetupSource.GUIDED
        )

    async def _ensure_managed_running_unlocked(self) -> AudioCppReadyEndpoint:
        supervisor = self._supervisor
        if supervisor is None:
            raise self._operation_error(
                _MANAGED_CONFIGURATION_INVALID,
                uuid4().hex,
            ) from None
        admission = supervisor.admission_snapshot()
        launch = self._managed_launch
        reuse_launch = launch is not None and admission.state in {
            "starting",
            "running",
            "unhealthy",
            "draining",
        }
        invalid_launch = False
        if not reuse_launch:
            self._managed_launch = None
            guided_failure: _OperationFailure | None = None
            if self._uses_guided_launch():
                if self._pending_guided_cleanup is not None:
                    raise self._operation_error(
                        _MANAGED_ARTIFACT_FAILURE,
                        uuid4().hex,
                    ) from None
                settings = self._guided_settings
                assert settings is not None
                try:
                    launch = await materialize_audio_cpp_guided_launch(settings)
                except AudioCppGuidedLaunchError as error:
                    cleanup_owner = error.take_cleanup_owner()
                    if cleanup_owner is not None:
                        self._pending_guided_cleanup = cleanup_owner
                    if error.code == "port_unavailable":
                        guided_failure = _MANAGED_PORT_UNAVAILABLE
                    elif error.code in {
                        "artifact_create_failed",
                        "artifact_cleanup_failed",
                    }:
                        guided_failure = _MANAGED_ARTIFACT_FAILURE
                    else:
                        guided_failure = _MANAGED_CONFIGURATION_INVALID
                except asyncio.CancelledError as error:
                    cleanup_owner = take_audio_cpp_guided_cleanup_owner(error)
                    if cleanup_owner is not None:
                        self._pending_guided_cleanup = cleanup_owner
                    raise
                except BaseException as error:
                    cleanup_owner = take_audio_cpp_guided_cleanup_owner(error)
                    if cleanup_owner is not None:
                        self._pending_guided_cleanup = cleanup_owner
                    raise
                if guided_failure is not None:
                    raise self._operation_error(
                        guided_failure,
                        uuid4().hex,
                    ) from None
            else:
                try:
                    launch = validate_audio_cpp_managed_launch(self._config)
                except (TypeError, ValueError):
                    invalid_launch = True
            if not invalid_launch:
                self._managed_launch = launch
        if invalid_launch:
            raise self._operation_error(
                _MANAGED_CONFIGURATION_INVALID,
                uuid4().hex,
            )
        assert launch is not None

        async def generation_hooks_factory(
            process_generation: int,
        ) -> AudioCppGenerationHooks:
            return await self._create_managed_generation_hooks(
                launch,
                process_generation,
            )

        try:
            endpoint = await supervisor.ensure_running(
                launch,
                generation_hooks_factory=generation_hooks_factory,
                require_existing=self._managed_required_admission.get(),
            )
        except TTSOperationError:
            if self._uses_guided_launch():
                self._managed_launch = None
            raise
        bundle = self._managed_bundle
        if bundle is None or bundle.process_generation != endpoint.process_generation:
            raise self._operation_error(
                _CONNECTION_UNAVAILABLE,
                uuid4().hex,
            ) from None
        return endpoint

    async def _create_managed_generation_hooks(
        self,
        launch: AudioCppManagedLaunchConfig,
        process_generation: int,
    ) -> AudioCppGenerationHooks:
        request_client: httpx.AsyncClient | None = None
        health_client: httpx.AsyncClient | None = None
        setup_failed = False
        try:
            request_client = self._new_request_client(launch.base_url)
            health_client = self._new_health_client(launch.base_url)
        except asyncio.CancelledError:
            await self._close_partial_managed_clients(
                request_client,
                health_client,
            )
            raise
        except BaseException:
            setup_failed = True
        if setup_failed:
            await self._close_partial_managed_clients(
                request_client,
                health_client,
            )
            raise RuntimeError("audio.cpp generation resource setup failed")

        bundle = _ManagedGenerationBundle(
            process_generation=process_generation,
            request_client=request_client,
            health_client=health_client,
            expected_models=launch.expected_models,
        )
        previous = self._managed_bundle
        if previous is not None and previous.process_generation != process_generation:
            previous_cleanup_failed = False
            try:
                await previous.close_remaining()
            except BaseException:
                previous_cleanup_failed = True
            if previous_cleanup_failed:
                await bundle.close_remaining()
                raise RuntimeError(_MANAGED_CLEANUP_FAILURE_MESSAGE)

        self._managed_bundle = bundle
        self._managed_process_generation = process_generation
        self._clone_capabilities.clear()
        self._managed_catalog_process_generation = None
        self._managed_catalog_observation_version = None
        self._managed_stop_complete = False
        self._client = request_client
        self._catalog = TTSProviderCatalog(
            provider_id=_PROVIDER_ID,
            revision=self._catalog.revision,
            health=_INITIAL_HEALTH,
            models=(),
        )
        self._clear_voice_state()

        async def health_probe() -> bool:
            return await self._probe_managed_health(bundle)

        async def contract_probe() -> AudioCppTTSCapability:
            return await self._refresh_catalog_with_client(
                bundle.request_client,
                force=True,
                process_generation=process_generation,
                raise_on_failure=True,
                expected_models=launch.expected_models,
            )

        def generation_invalidate() -> None:
            if self._managed_process_generation == process_generation:
                self._clone_capabilities.clear()
                self._managed_process_generation = None
                self._managed_catalog_process_generation = None
                self._managed_catalog_observation_version = None
                self._clear_voice_state()
                self._mark_catalog_stale(_TRANSIENT_FAILURE_HEALTH)

        async def generation_cleanup() -> None:
            generation_invalidate()
            await bundle.supervisor_cleanup()

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=health_probe,
            cleanup=generation_cleanup,
            invalidate=generation_invalidate,
        )

    @staticmethod
    async def _close_partial_managed_clients(
        request_client: httpx.AsyncClient | None,
        health_client: httpx.AsyncClient | None,
    ) -> None:
        """Attempt every allocated client cleanup during transactional setup."""
        clients = tuple(
            client for client in (health_client, request_client) if client is not None
        )
        if clients:
            await asyncio.gather(
                *(client.aclose() for client in clients),
                return_exceptions=True,
            )

    async def _probe_managed_health(
        self,
        bundle: _ManagedGenerationBundle,
    ) -> bool:
        try:
            timeout = min(
                self._config.connect_timeout_seconds,
                self._config.managed_health_check_interval_seconds,
            )
            async with asyncio.timeout(timeout):
                body = await self._safe_get(
                    "/health",
                    client=bundle.health_client,
                )
                parse_health_response(
                    body,
                    max_metadata_bytes=self._config.max_metadata_bytes,
                    max_identifier_characters=self._config.max_identifier_characters,
                    max_models=self._config.max_catalog_models,
                )
        except asyncio.CancelledError:
            raise
        except BaseException:
            if self._managed_process_generation == bundle.process_generation:
                self._mark_catalog_stale(_TRANSIENT_FAILURE_HEALTH)
                self._managed_catalog_process_generation = None
                self._managed_catalog_observation_version = None
            return False
        return True

    async def _fetch_voices(
        self,
        model_id: str,
        catalog_revision: int,
    ) -> TTSVoiceDiscoveryResult:
        try:
            async with asyncio.timeout(self._config.connect_timeout_seconds):
                body = await self._safe_get(
                    "/v1/audio/voices",
                    params={"model": model_id},
                    client=self._require_request_client(),
                )
                voices = parse_voices_response(
                    body,
                    max_metadata_bytes=self._config.max_metadata_bytes,
                    max_identifier_characters=(self._config.max_identifier_characters),
                    max_voices=self._config.max_voices_per_model,
                )
                return TTSVoiceDiscoveryResult(
                    provider_id=_PROVIDER_ID,
                    model_id=model_id,
                    catalog_revision=catalog_revision,
                    voices=voices,
                    state="complete",
                )
        except asyncio.CancelledError:
            raise
        except (
            TimeoutError,
            _TransientHttpFailure,
            _HttpContractFailure,
            AudioCppContractError,
        ):
            return self._unverified_voice_result(model_id, catalog_revision)

    async def _safe_get(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> bytes:
        active_client = client or self._require_request_client()
        process_generation = self._managed_client_generation(active_client)
        for attempt in range(_MAX_GET_ATTEMPTS):
            runtime_failure: RuntimeError | None = None
            suppression_token = self._begin_http_log_suppression()
            try:
                active_client.cookies.clear()
                async with active_client.stream(
                    "GET",
                    path,
                    params=params,
                ) as response:
                    active_client.cookies.clear()
                    if response.status_code != 200:
                        if response.status_code in _TRANSIENT_STATUSES:
                            if attempt + 1 < _MAX_GET_ATTEMPTS:
                                continue
                            raise _TransientHttpFailure
                        raise _HttpContractFailure
                    return await self._read_bounded_metadata(response)
            except asyncio.CancelledError:
                raise
            except _HttpContractFailure:
                raise
            except _TransientHttpFailure:
                raise
            except httpx.StreamError:
                raise _HttpContractFailure from None
            except RuntimeError as error:
                runtime_failure = error
            except (TimeoutError, httpx.TransportError):
                if self._closed or attempt + 1 >= _MAX_GET_ATTEMPTS:
                    raise _TransientHttpFailure from None
            finally:
                self._end_http_log_suppression(suppression_token)
            if runtime_failure is not None:
                if self._managed_client_invalidated(
                    active_client,
                    process_generation,
                ):
                    raise _TransientHttpFailure
                raise runtime_failure
        raise _TransientHttpFailure

    def _managed_client_generation(
        self,
        client: httpx.AsyncClient,
    ) -> int | None:
        bundle = self._managed_bundle
        if (
            self._config.mode == "managed"
            and bundle is not None
            and client in {bundle.request_client, bundle.health_client}
        ):
            return bundle.process_generation
        return None

    def _managed_client_invalidated(
        self,
        client: httpx.AsyncClient,
        process_generation: int | None,
    ) -> bool:
        if self._closed:
            return True
        if process_generation is None:
            return False
        bundle = self._managed_bundle
        return (
            self._managed_process_generation != process_generation
            or bundle is None
            or bundle.process_generation != process_generation
            or client not in {bundle.request_client, bundle.health_client}
        )

    async def _read_bounded_metadata(
        self,
        response: httpx.Response,
    ) -> bytes:
        return await self._read_bounded_response(
            response,
            max_bytes=self._config.max_metadata_bytes,
        )

    async def _read_bounded_response(
        self,
        response: httpx.Response,
        *,
        max_bytes: int,
    ) -> bytes:
        content_encodings = response.headers.get_list("content-encoding")
        if len(content_encodings) > 1:
            raise _HttpContractFailure
        if content_encodings and content_encodings[0].strip().casefold() != "identity":
            raise _HttpContractFailure

        declared_lengths = response.headers.get_list("content-length")
        if len(declared_lengths) > 1:
            raise _HttpContractFailure
        declared_length: int | None = None
        if declared_lengths:
            raw_length = declared_lengths[0]
            if (
                not raw_length
                or len(raw_length) > _MAX_CONTENT_LENGTH_DIGITS
                or not raw_length.isascii()
                or not raw_length.isdecimal()
            ):
                raise _HttpContractFailure
            declared_length = int(raw_length)
            if declared_length > sys.maxsize or declared_length > max_bytes:
                raise _HttpContractFailure

        body = bytearray()
        async for chunk in response.aiter_raw():
            if not chunk:
                continue
            remaining = max_bytes - len(body)
            if len(chunk) > remaining:
                raise _HttpContractFailure
            body.extend(chunk)
        if declared_length is not None and len(body) != declared_length:
            raise _HttpContractFailure
        return bytes(body)

    def _begin_http_log_suppression(self) -> Token[bool]:
        if not self._http_log_filters_installed:
            for logger_name in _HTTP_LOGGER_NAMES:
                logging.getLogger(logger_name).addFilter(self._httpx_privacy_filter)
            self._http_log_filters_installed = True
        self._http_log_suppression_users += 1
        return _HTTP_LOG_SUPPRESSION_ACTIVE.set(True)

    def _end_http_log_suppression(self, token: Token[bool]) -> None:
        try:
            _HTTP_LOG_SUPPRESSION_ACTIVE.reset(token)
        finally:
            self._http_log_suppression_users -= 1
            self._remove_http_log_filters_if_idle()

    def _remove_http_log_filters_if_idle(self) -> None:
        if (
            not self._http_log_client_closed
            or self._http_log_suppression_users
            or not self._http_log_filters_installed
        ):
            return
        for logger_name in _HTTP_LOGGER_NAMES:
            logging.getLogger(logger_name).removeFilter(self._httpx_privacy_filter)
        self._http_log_filters_installed = False

    def _cached_voice_result(
        self,
        key: _VoiceCacheKey,
    ) -> TTSVoiceDiscoveryResult | None:
        entry = self._voice_cache.get(key)
        if entry is None:
            return None
        self._voice_cache.move_to_end(key)
        return entry.result

    def _cache_voice_result(
        self,
        key: _VoiceCacheKey,
        result: TTSVoiceDiscoveryResult,
    ) -> None:
        existing = self._voice_cache.pop(key, None)
        if existing is not None:
            self._voice_cache_bytes -= existing.estimated_bytes

        estimated_bytes = _estimate_voice_cache_entry_bytes(key, result.voices)
        if (
            _MAX_VOICE_CACHE_ENTRIES <= 0
            or _MAX_VOICE_CACHE_BYTES <= 0
            or estimated_bytes > _MAX_VOICE_CACHE_BYTES
        ):
            self._discard_voice_key_state_if_idle(key)
            return

        self._voice_cache[key] = _VoiceCacheEntry(
            result=result,
            estimated_bytes=estimated_bytes,
        )
        self._voice_cache_bytes += estimated_bytes
        while self._voice_cache and (
            len(self._voice_cache) > _MAX_VOICE_CACHE_ENTRIES
            or self._voice_cache_bytes > _MAX_VOICE_CACHE_BYTES
        ):
            evicted_key, evicted = self._voice_cache.popitem(last=False)
            self._voice_cache_bytes -= evicted.estimated_bytes
            self._discard_voice_key_state_if_idle(evicted_key)

    def _release_voice_lock_user(
        self,
        key: _VoiceCacheKey,
        lock: asyncio.Lock,
    ) -> None:
        users = self._voice_lock_users.get(key)
        if users is None:
            return
        if users > 1:
            self._voice_lock_users[key] = users - 1
            return

        self._voice_lock_users.pop(key, None)
        self._voice_shared_results.pop(key, None)
        if key not in self._voice_cache:
            self._voice_generation.pop(key, None)
            if self._voice_locks.get(key) is lock:
                self._voice_locks.pop(key, None)

    def _discard_voice_key_state_if_idle(self, key: _VoiceCacheKey) -> None:
        if self._voice_lock_users.get(key, 0) > 0:
            return
        self._voice_generation.pop(key, None)
        self._voice_locks.pop(key, None)
        self._voice_lock_users.pop(key, None)
        self._voice_shared_results.pop(key, None)

    def _clear_voice_state(self) -> None:
        self._voice_cache.clear()
        self._voice_cache_bytes = 0
        self._voice_generation.clear()
        self._voice_locks.clear()
        self._voice_lock_users.clear()
        self._voice_shared_results.clear()

    def _mark_catalog_stale(self, health: ProviderHealth) -> None:
        if self._closed:
            return
        if self._config.mode == "managed":
            self._managed_catalog_observation_version = None
        self._catalog = self._failed_catalog(self._catalog, health)
        self._refresh_generation += 1

    @staticmethod
    def _unverified_voice_result(
        model_id: str,
        catalog_revision: int,
    ) -> TTSVoiceDiscoveryResult:
        return TTSVoiceDiscoveryResult(
            provider_id=_PROVIDER_ID,
            model_id=model_id,
            catalog_revision=catalog_revision,
            voices=(),
            state="unverified",
        )

    @staticmethod
    def _operation_error(
        failure: _OperationFailure,
        operation_id: str,
    ) -> TTSOperationError:
        return TTSOperationError(
            code=failure.code,
            message=failure.message,
            retryable=failure.retryable,
            operation_id=operation_id,
            recovery_action=failure.recovery_action,
        )

    @staticmethod
    def _failed_catalog(
        previous: TTSProviderCatalog,
        health: ProviderHealth,
    ) -> TTSProviderCatalog:
        return TTSProviderCatalog(
            provider_id=_PROVIDER_ID,
            revision=previous.revision,
            health=health,
            models=previous.models,
        )

    @staticmethod
    def _catalog_contains(
        catalog: TTSProviderCatalog,
        model_id: str,
    ) -> bool:
        return any(model.model_id == model_id for model in catalog.models)
