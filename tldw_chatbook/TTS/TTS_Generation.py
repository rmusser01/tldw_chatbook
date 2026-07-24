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
    TTSRegistryClosedError,
    TTSRequest,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import resolve_legacy_route

logger = logging.getLogger(__name__)

_CLEANUP_FAILURE_NOTE = "TTS cleanup also failed while preserving the original error"
_SHUTDOWN_RESPONSE_FAILURE_NOTE = (
    "TTS response cleanup also failed during service shutdown"
)


def _record_cleanup_failure(primary_error: BaseException) -> None:
    primary_error.add_note(_CLEANUP_FAILURE_NOTE)
    logger.warning("TTS cleanup failed while preserving an earlier error")


async def _join_retained_task(task: asyncio.Task[None]) -> None:
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
    except asyncio.CancelledError:
        if waiter is not None and waiter.cancelling() > cancellation_requests:
            raise
        _record_cleanup_failure(primary_error)
    except BaseException:
        _record_cleanup_failure(primary_error)


class _OperationResources:
    def __init__(
        self,
        lease: TTSAdapterLease,
        operation_limit: asyncio.Semaphore,
    ) -> None:
        self._lease = lease
        self._operation_limit = operation_limit
        self._cleanup_task: asyncio.Task[None] | None = None

    async def close(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._release())
        await _join_retained_task(self._cleanup_task)

    async def _release(self) -> None:
        try:
            await self._lease.release()
        finally:
            self._operation_limit.release()


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
        )
        self._response = response
        self._resources = resources
        self._on_closed = on_closed
        self._response_close_task: asyncio.Task[None] | None = None
        self._resource_close_task: asyncio.Task[None] | None = None

    def add_cleanup(self, callback: CleanupCallback) -> None:
        self._response.add_cleanup(callback)

    def _ensure_response_close_task(self) -> asyncio.Task[None]:
        if self._response_close_task is None:
            self._response_close_task = asyncio.create_task(self._response.aclose())
            self._response_close_task.add_done_callback(
                lambda _task: self._on_closed(self)
            )
        return self._response_close_task

    def _ensure_resource_close_task(self) -> asyncio.Task[None]:
        if self._resource_close_task is None:
            self._resource_close_task = asyncio.create_task(self._resources.close())
        return self._resource_close_task

    def start_shutdown_close(
        self,
    ) -> tuple[asyncio.Task[None], asyncio.Task[None]]:
        return (
            self._ensure_response_close_task(),
            self._ensure_resource_close_task(),
        )

    async def aclose(self) -> None:
        await _join_retained_task(self._ensure_response_close_task())


class TTSService:
    """Coordinate registry-backed TTS operations and response lifetimes."""

    def __init__(
        self,
        registry: TTSAdapterRegistry,
        *,
        max_concurrent_operations: int = 4,
    ) -> None:
        if max_concurrent_operations < 1:
            raise ValueError("max_concurrent_operations must be positive")
        self.registry = registry
        self._operation_limit = asyncio.Semaphore(max_concurrent_operations)
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._admission_waiters: set[asyncio.Task[bool]] = set()
        self._syntheses_in_progress = 0
        self._synthesis_idle = asyncio.Event()
        self._synthesis_idle.set()
        self._responses: dict[_ManagedAudioResponse, None] = {}
        self._shutdown_response_tasks: dict[asyncio.Task[None], None] = {}

    async def _acquire_operation_slot(self) -> None:
        current_task = asyncio.current_task()
        async with self._lifecycle_lock:
            if self._closing:
                raise TTSRegistryClosedError("The TTS service is closing")
            waiter = asyncio.create_task(self._operation_limit.acquire())
            self._admission_waiters.add(waiter)

        try:
            await waiter
        except asyncio.CancelledError:
            acquired = (
                waiter.done() and not waiter.cancelled() and waiter.exception() is None
            )
            async with self._lifecycle_lock:
                self._admission_waiters.discard(waiter)
                closing = self._closing
            if acquired:
                self._operation_limit.release()
            if closing and current_task is not None and current_task.cancelling() == 0:
                raise TTSRegistryClosedError("The TTS service is closing") from None
            raise

        async with self._lifecycle_lock:
            self._admission_waiters.discard(waiter)
            if self._closing:
                self._operation_limit.release()
                raise TTSRegistryClosedError("The TTS service is closing")
            self._syntheses_in_progress += 1
            self._synthesis_idle.clear()

    async def _finish_synthesis_creation(self) -> None:
        async with self._lifecycle_lock:
            self._syntheses_in_progress -= 1
            if self._syntheses_in_progress == 0:
                self._synthesis_idle.set()

    async def _ensure_open(self) -> None:
        async with self._lifecycle_lock:
            if self._closing:
                raise TTSRegistryClosedError("The TTS service is closing")

    def _response_closed(self, response: _ManagedAudioResponse) -> None:
        self._responses.pop(response, None)

    @staticmethod
    def _observe_task_result(task: asyncio.Task[None]) -> None:
        try:
            task.exception()
        except BaseException:
            pass

    def _track_shutdown_response_task(self, task: asyncio.Task[None]) -> None:
        if task in self._shutdown_response_tasks:
            return
        self._shutdown_response_tasks[task] = None
        task.add_done_callback(self._observe_task_result)

    async def _start_response_shutdown(self) -> list[BaseException]:
        async with self._lifecycle_lock:
            responses = tuple(self._responses)

        resource_tasks: list[asyncio.Task[None]] = []
        for response in responses:
            response_task, resource_task = response.start_shutdown_close()
            self._track_shutdown_response_task(response_task)
            resource_tasks.append(resource_task)

        if not resource_tasks:
            return []
        return [
            result
            for result in await asyncio.gather(
                *resource_tasks,
                return_exceptions=True,
            )
            if isinstance(result, BaseException)
        ]

    @staticmethod
    def _raise_with_shutdown_failures(
        primary_error: BaseException | None,
        secondary_errors: list[BaseException],
    ) -> None:
        unique_secondary: list[BaseException] = []
        seen = {id(primary_error)} if primary_error is not None else set()
        for error in secondary_errors:
            if id(error) in seen:
                continue
            seen.add(id(error))
            unique_secondary.append(error)

        if primary_error is None and unique_secondary:
            primary_error = unique_secondary.pop(0)
        if primary_error is None:
            return
        for _error in unique_secondary:
            primary_error.add_note(_SHUTDOWN_RESPONSE_FAILURE_NOTE)
            logger.warning("TTS response cleanup failed during service shutdown")
        raise primary_error

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
        await self._acquire_operation_slot()
        try:
            try:
                lease = await self.registry.acquire(request.provider_id)
            except BaseException:
                self._operation_limit.release()
                raise

            resources = _OperationResources(lease, self._operation_limit)
            safe_sink = _isolate_progress_sink(progress_sink)
            try:
                await lease.adapter.ensure_ready()
                response = await lease.adapter.synthesize(request, safe_sink)
            except BaseException as error:
                await _cleanup_preserving_primary(resources.close, error)
                raise

            try:
                response.add_cleanup(resources.close)
            except BaseException as error:
                await _cleanup_preserving_primary(resources.close, error)
                raise

            managed = _ManagedAudioResponse(
                response,
                resources,
                self._response_closed,
            )
            async with self._lifecycle_lock:
                if not self._closing:
                    self._responses[managed] = None
                    return managed

            response_task, resource_task = managed.start_shutdown_close()
            self._track_shutdown_response_task(response_task)
            closing_error = TTSRegistryClosedError("The TTS service is closing")
            try:
                await _join_retained_task(resource_task)
            except BaseException:
                closing_error.add_note(_SHUTDOWN_RESPONSE_FAILURE_NOTE)
                logger.warning("TTS response resources failed during service shutdown")
            raise closing_error
        finally:
            await self._finish_synthesis_creation()

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

    async def get_catalog(
        self,
        provider_id: str,
        refresh: bool = False,
    ) -> TTSProviderCatalog:
        """Return a provider catalog, optionally refreshing its contents."""
        await self._ensure_open()
        return await self.registry.get_catalog(provider_id, refresh=refresh)

    async def reconfigure_provider(
        self,
        provider_id: str,
        config: Mapping[str, Any],
    ) -> ReconfigureResult:
        """Apply provider configuration through the registry lifecycle."""
        await self._ensure_open()
        return await self.registry.reconfigure_provider(provider_id, config)

    async def close(self) -> None:
        """Seal service admission and run bounded registry/response shutdown."""
        close_task = await self._ensure_close_task()
        await _join_retained_task(close_task)

    async def wait_closed(self) -> None:
        """Join bounded shutdown, active responses, and definitive registry close."""
        close_task = await self._ensure_close_task()
        bounded_error: BaseException | None = None
        try:
            await _join_retained_task(close_task)
        except BaseException as error:
            bounded_error = error

        await self._synthesis_idle.wait()
        resource_errors = await self._start_response_shutdown()
        response_tasks = tuple(self._shutdown_response_tasks)
        response_errors = [
            result
            for result in await asyncio.gather(
                *response_tasks,
                return_exceptions=True,
            )
            if isinstance(result, BaseException)
        ]

        registry_error: BaseException | None = None
        try:
            await self.registry.wait_closed()
        except BaseException as error:
            registry_error = error

        self._raise_with_shutdown_failures(
            registry_error or bounded_error,
            resource_errors + response_errors,
        )

    async def _ensure_close_task(self) -> asyncio.Task[None]:
        async with self._lifecycle_lock:
            if self._close_task is not None:
                return self._close_task

            self._closing = True
            for waiter in tuple(self._admission_waiters):
                if not waiter.done():
                    waiter.cancel()
            self._close_task = asyncio.create_task(self._bounded_close())
            self._close_task.add_done_callback(self._observe_task_result)
            return self._close_task

    async def _bounded_close(self) -> None:
        registry_error: BaseException | None = None
        try:
            await self.registry.close()
        except BaseException as error:
            registry_error = error

        resource_errors = await self._start_response_shutdown()
        self._raise_with_shutdown_failures(registry_error, resource_errors)


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
