from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

from tldw_chatbook.TTS._async_lifecycle import join_retained_task
from tldw_chatbook.TTS.adapter_registry import (
    ReconfigureResult,
    TTSAdapterLease,
    TTSAdapterRegistry,
    TTSReconfigurationTicket,
)
from tldw_chatbook.TTS.adapter_types import (
    CapabilitySnapshotState,
    CleanupCallback,
    ProgressSink,
    TTSAudioResponse,
    TTSNativeCapabilitySnapshot,
    TTSOperationError,
    TTSProgress,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSRegistryClosedError,
    TTSRequest,
    TTSStructuredVoiceAdapter,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import resolve_legacy_route
from tldw_chatbook.TTS.playground_types import TTSRequestedSelectionSnapshot
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.request_admission import TTSRequestAdmissionCoordinator

logger = logging.getLogger(__name__)

_CLEANUP_FAILURE_NOTE = "TTS cleanup also failed while preserving the original error"
_TTS_SETTINGS_FOREGROUND_TIMEOUT_SECONDS = 2.0
_NATIVE_CAPABILITY_TIMEOUT_SECONDS = 10.0
_NATIVE_CAPABILITY_VOICE_CONCURRENCY = 4

TTSSettingsProviderStatus = Literal[
    "applied",
    "pending",
    "unchanged",
    "superseded",
    "unavailable",
]


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


@dataclass(frozen=True, slots=True)
class TTSSettingsPublication:
    """One safe settings-publication result for foreground or final observers."""

    generation: int
    preferences: TTSPreferencesSnapshot
    persistence: TTSSettingsPersistenceOutcome
    provider_statuses: Mapping[str, TTSSettingsProviderStatus]
    provider_revisions: Mapping[str, int]
    published: bool

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


@dataclass(frozen=True, slots=True)
class TTSSettingsPublicationTicket:
    """Service-owned settings operation with bounded and definitive views."""

    generation: int
    foreground: asyncio.Future[TTSSettingsPublication]
    completion: asyncio.Task[TTSSettingsPublication]


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
        self._response_close_task: asyncio.Task[None] | None = None

    def add_cleanup(self, callback: CleanupCallback) -> None:
        self._response.add_cleanup(callback)

    def start_close(self) -> asyncio.Task[None]:
        if self._response_close_task is None:
            self._response_close_task = asyncio.create_task(self._close())
        return self._response_close_task

    def start_resource_release(self) -> asyncio.Task[None]:
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
            [TTSAudioResponse, _OperationResources],
            _ManagedAudioResponse,
        ],
        observe_cleanup: Callable[[asyncio.Task[None]], None],
    ) -> None:
        self._request = request
        self._resources = resources
        self._close_signal = close_signal
        self._on_finished = on_finished
        self._manage_response = manage_response
        self._observe_cleanup = observe_cleanup
        self._claimed = False
        self._used = False
        self._executing = False
        self._close_task: asyncio.Task[None] | None = None

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
        try:
            await lease.adapter.ensure_ready()
            response = await lease.adapter.synthesize(self._request, safe_sink)
        except BaseException as error:
            try:
                await _cleanup_preserving_primary(self._resources.close, error)
            finally:
                self._finish_tracking()
            raise

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
            managed_response = self._manage_response(response, self._resources)
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
        return managed_response

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
        if self._close_task is None:
            self._used = True
            self._close_task = asyncio.create_task(self._close())
        return self._close_task

    async def _close(self) -> None:
        try:
            await self._resources.close()
        finally:
            self._finish_tracking()

    def _finish_tracking(self) -> None:
        self._executing = False
        self._on_finished(self)


class TTSService:
    """Coordinate registry-backed TTS operations and response lifetimes."""

    def __init__(
        self,
        registry: TTSAdapterRegistry,
        *,
        max_concurrent_operations: int = 4,
        preferences_snapshot: TTSPreferencesSnapshot | None = None,
    ) -> None:
        if max_concurrent_operations < 1:
            raise ValueError("max_concurrent_operations must be positive")
        self.registry = registry
        self._operation_limit = asyncio.Semaphore(max_concurrent_operations)
        self._close_signal = asyncio.Event()
        self._registry_close_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._responses: set[_ManagedAudioResponse] = set()
        self._admitted_operations: set[_AdmittedTTSOperation] = set()
        self._settings_generation = 0
        self._settings_persisted_provider_generations: dict[str, int] = {}
        self._settings_publication_tasks: set[asyncio.Task[TTSSettingsPublication]] = (
            set()
        )
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
        )

    async def admit(
        self,
        request: TTSRequest,
        *,
        expected_configuration_revision: int | None = None,
    ) -> _AdmittedTTSOperation:
        """Reserve service capacity and a revision-matched provider lease.

        Args:
            request: Native provider, model, and audio options.
            expected_configuration_revision: Optional selected provider revision.

        Returns:
            A single-use operation that owns its admitted resources.
        """
        reservation: _OperationCapacityReservation | None = None
        try:
            reservation = await self._reserve_operation_capacity()
            return await self._admit_reserved(
                request,
                reservation,
                expected_configuration_revision=expected_configuration_revision,
            )
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

    async def _admit_reserved(
        self,
        request: TTSRequest,
        reservation: _OperationCapacityReservation,
        *,
        expected_configuration_revision: int | None = None,
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
        )
        self._admitted_operations.add(operation)
        if self._close_signal.is_set():
            closed_error = TTSRegistryClosedError("The TTS service is closed")
            await _cleanup_preserving_primary(operation.close, closed_error)
            raise closed_error
        return operation

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
    ) -> TTSAudioResponse:
        """Synthesize audio while retaining provider resources for the response.

        Args:
            request: Native provider, model, and audio options.
            progress_sink: Optional asynchronous progress reporter.

        Returns:
            A response that releases its registry lease when closed.
        """
        operation = await self.admit(request)
        return await operation.synthesize(progress_sink)

    async def synthesize_exact(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> tuple[TTSAudioResponse, TTSRequestedSelectionSnapshot]:
        """Synthesize one exact native request with admitted provenance."""
        self._require_native_provider(request.provider_id)
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

    async def get_native_capability_snapshot(
        self,
        provider_id: str,
        exact_voice_model_ids: Iterable[str],
    ) -> TTSNativeCapabilitySnapshot:
        """Observe one bounded native capability snapshot without exposing a lease."""
        self._require_native_provider(provider_id)
        model_ids = self._distinct_capability_model_ids(exact_voice_model_ids)
        revision = 0
        lease: TTSAdapterLease | None = None
        result = TTSNativeCapabilitySnapshot(
            provider_id=provider_id,
            configuration_revision=revision,
            state="unverified",
            catalog=None,
            voice_results={},
        )
        primary_error: BaseException | None = None
        deadline = (
            asyncio.get_running_loop().time() + _NATIVE_CAPABILITY_TIMEOUT_SECONDS
        )
        try:
            async with asyncio.timeout_at(deadline):
                (
                    revision,
                    lease,
                ) = await self._request_admission.acquire_native_capability_lease(
                    provider_id
                )
                result = await self._observe_native_capabilities(
                    provider_id,
                    revision,
                    lease.adapter,
                    model_ids,
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
        return result

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
            if model_ids and not isinstance(adapter, TTSStructuredVoiceAdapter):
                return TTSNativeCapabilitySnapshot(
                    provider_id=provider_id,
                    configuration_revision=configuration_revision,
                    state="unverified",
                    catalog=catalog,
                    voice_results={},
                )

            semaphore = asyncio.Semaphore(_NATIVE_CAPABILITY_VOICE_CONCURRENCY)
            assert isinstance(adapter, TTSStructuredVoiceAdapter)
            observed = await asyncio.gather(
                *(
                    self._observe_native_voices(
                        adapter,
                        semaphore,
                        model_id,
                    )
                    for model_id in model_ids
                )
            )
            voice_results = dict(zip(model_ids, observed, strict=True))
            final_catalog = await adapter.get_catalog(refresh=True)  # type: ignore[attr-defined]
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

            catalog_moved = final_catalog.revision != catalog.revision
            authoritative = all(
                result.provider_id == provider_id
                and result.model_id == model_id
                and result.catalog_revision == catalog.revision
                and result.state != "unverified"
                for model_id, result in voice_results.items()
            )
            state: CapabilitySnapshotState = (
                "complete" if not catalog_moved and authoritative else "unverified"
            )
            last_snapshot = TTSNativeCapabilitySnapshot(
                provider_id=provider_id,
                configuration_revision=configuration_revision,
                state=state,
                catalog=final_catalog,
                voice_results=voice_results,
            )
            if not catalog_moved or attempt == 1:
                return last_snapshot

        return last_snapshot

    @staticmethod
    async def _observe_native_voices(
        adapter: TTSStructuredVoiceAdapter,
        semaphore: asyncio.Semaphore,
        model_id: str,
    ) -> TTSVoiceDiscoveryResult:
        async with semaphore:
            return await adapter.observe_voices(model_id, refresh=True)

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
        for model_id in model_ids:
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
    ) -> TTSAudioResponse:
        """Resolve and synthesize one revision-coherent default request."""
        return await self._request_admission.synthesize_default(
            text=text,
            voice_override=voice_override,
            progress_sink=progress_sink,
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
        return await self.registry.get_catalog(provider_id, refresh=refresh)

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
        return await self.registry.get_voices(
            provider_id,
            model_id,
            refresh=refresh,
        )

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

    def begin_preferences_publication(
        self,
        preferences: TTSPreferencesSnapshot,
        provider_configs: Mapping[str, Mapping[str, Any]],
        persistence: Callable[[], TTSSettingsPersistenceOutcome],
        *,
        foreground_timeout_seconds: float = (_TTS_SETTINGS_FOREGROUND_TIMEOUT_SECONDS),
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
        completion = asyncio.create_task(
            self._run_preferences_publication(
                generation=generation,
                preferences=preferences,
                provider_configs=copied_configs,
                persistence=persistence,
                foreground_timeout_seconds=float(foreground_timeout_seconds),
                foreground=foreground,
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

    async def _run_preferences_publication(
        self,
        *,
        generation: int,
        preferences: TTSPreferencesSnapshot,
        provider_configs: Mapping[str, Mapping[str, Any]],
        persistence: Callable[[], TTSSettingsPersistenceOutcome],
        foreground_timeout_seconds: float,
        foreground: asyncio.Future[TTSSettingsPublication],
    ) -> TTSSettingsPublication:
        tickets: dict[str, TTSReconfigurationTicket] = {}
        provider_statuses: dict[str, TTSSettingsProviderStatus] = {}
        provider_revisions: dict[str, int] = {}
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

            # Expose durable, provider-scoped proof before a newer handoff can
            # wake an older publication that it superseded.
            for provider_id in provider_configs:
                self._settings_persisted_provider_generations[provider_id] = max(
                    generation,
                    self._settings_persisted_provider_generations.get(provider_id, 0),
                )

            async with self._request_admission._gate.write():
                transition_failed = False
                for provider_id, config in provider_configs.items():
                    try:
                        ticket = await self.registry.begin_reconfigure_provider(
                            provider_id,
                            config,
                            generation=generation,
                        )
                        tickets[provider_id] = ticket
                    except BaseException:
                        transition_failed = True
                        break

                if transition_failed:
                    await self._seal_provider_configs(provider_configs)
                    provider_statuses.update(
                        {provider_id: "unavailable" for provider_id in provider_configs}
                    )
                else:
                    provider_statuses.update(
                        await self._bounded_reconfiguration_statuses(
                            tickets,
                            timeout_seconds=foreground_timeout_seconds,
                        )
                    )

                self._request_admission._publish_preferences(
                    preferences,
                    generation,
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
        final_revisions = self._safe_provider_revisions(provider_configs)
        return self._settings_publication_result(
            generation=generation,
            preferences=preferences,
            persistence=persistence_outcome,
            provider_statuses=final_statuses,
            provider_revisions=final_revisions,
            published=True,
        )

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
    ) -> TTSSettingsPublication:
        return TTSSettingsPublication(
            generation=generation,
            preferences=preferences,
            persistence=persistence,
            provider_statuses=provider_statuses,
            provider_revisions=provider_revisions,
            published=published,
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
            publication_tasks = tuple(self._settings_publication_tasks)
            operation_tasks = tuple(
                cleanup_task
                for operation in tuple(self._admitted_operations)
                if (cleanup_task := operation.start_close_if_pending()) is not None
            )
            self._registry_close_task = asyncio.create_task(self.registry.close())
            self._shutdown_task = asyncio.create_task(
                self._complete_shutdown(
                    self._registry_close_task,
                    operation_tasks,
                    publication_tasks,
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
    ) -> None:
        failures: list[BaseException] = []
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
        resource_tasks = [response.start_resource_release() for response in responses]
        registry_wait_task = asyncio.create_task(self.registry.wait_closed())
        terminal_shutdown_tasks = (
            registry_wait_task,
            *operation_tasks,
            *resource_tasks,
            *publication_tasks,
        )
        results = await asyncio.gather(
            *terminal_shutdown_tasks,
            *late_operation_tasks,
            return_exceptions=True,
        )
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
        if failures:
            raise _sanitized_shutdown_error(*failures) from None

    def _manage_response(
        self,
        response: TTSAudioResponse,
        resources: _OperationResources,
    ) -> _ManagedAudioResponse:
        managed_response = _ManagedAudioResponse(
            response,
            resources,
            self._responses.discard,
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
