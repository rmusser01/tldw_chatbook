from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from tldw_chatbook.TTS._async_lifecycle import join_retained_task
from tldw_chatbook.TTS.adapter_registry import (
    ReconfigureResult,
    TTSAdapterLease,
    TTSAdapterRegistry,
)
from tldw_chatbook.TTS.adapter_types import (
    CleanupCallback,
    ProgressSink,
    TTSAudioResponse,
    TTSProgress,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSRequest,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import resolve_legacy_route
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.request_admission import TTSRequestAdmissionCoordinator

logger = logging.getLogger(__name__)

_CLEANUP_FAILURE_NOTE = "TTS cleanup also failed while preserving the original error"


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
        initial_preferences = (
            TTSPreferencesSnapshot.from_settings({})
            if preferences_snapshot is None
            else preferences_snapshot
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

    def preferences_snapshot(self) -> TTSPreferencesSnapshot:
        """Return the current immutable default TTS preference snapshot."""
        return self._request_admission.preferences_snapshot()

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
                )
            )
            self._shutdown_task.add_done_callback(self._observe_shutdown_result)
        assert self._shutdown_task is not None
        return self._registry_close_task, self._shutdown_task

    async def _complete_shutdown(
        self,
        registry_close_task: asyncio.Task[None],
        operation_tasks: tuple[asyncio.Task[None], ...],
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
