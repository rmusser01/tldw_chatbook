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
    TTSRequest,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import resolve_legacy_route

logger = logging.getLogger(__name__)

_CLEANUP_FAILURE_NOTE = "TTS cleanup also failed while preserving the original error"


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
        self._on_closed = on_closed
        self._response_close_task: asyncio.Task[None] | None = None

    def add_cleanup(self, callback: CleanupCallback) -> None:
        self._response.add_cleanup(callback)

    def start_close(self) -> asyncio.Task[None]:
        if self._response_close_task is None:
            self._response_close_task = asyncio.create_task(self._close())
        return self._response_close_task

    async def aclose(self) -> None:
        await _join_retained_task(self.start_close())

    async def _close(self) -> None:
        try:
            await self._response.aclose()
        finally:
            self._on_closed(self)


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
        self._close_signal = asyncio.Event()
        self._registry_close_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._responses: set[_ManagedAudioResponse] = set()

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
            lease = await self.registry.acquire(request.provider_id)
        except BaseException:
            self._operation_limit.release()
            raise

        resources = _OperationResources(lease, self._operation_limit)
        if self._close_signal.is_set():
            closed_error = TTSRegistryClosedError("The TTS service is closed")
            await _cleanup_preserving_primary(resources.close, closed_error)
            raise closed_error

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

        if self._close_signal.is_set():
            closed_error = TTSRegistryClosedError("The TTS service is closed")
            await _cleanup_preserving_primary(response.aclose, closed_error)
            raise closed_error
        managed_response = _ManagedAudioResponse(response, self._responses.discard)
        self._responses.add(managed_response)
        return managed_response

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
        """Return a provider catalog, optionally refreshing its contents.

        Args:
            provider_id: Canonical provider identifier.
            refresh: Whether to refresh the provider catalog.

        Returns:
            The provider's current catalog.
        """
        return await self.registry.get_catalog(provider_id, refresh=refresh)

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
        await _join_retained_task(registry_close_task)

    async def wait_closed(self) -> None:
        """Join response and provider cleanup and report sanitized failures."""
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
            self._registry_close_task = asyncio.create_task(self.registry.close())
            self._shutdown_task = asyncio.create_task(
                self._complete_shutdown(self._registry_close_task)
            )
            self._shutdown_task.add_done_callback(self._observe_shutdown_result)
        assert self._shutdown_task is not None
        return self._registry_close_task, self._shutdown_task

    async def _complete_shutdown(
        self,
        registry_close_task: asyncio.Task[None],
    ) -> None:
        failures: list[BaseException] = []
        try:
            await asyncio.shield(registry_close_task)
        except BaseException as error:
            failures.append(error)

        response_tasks = [response.start_close() for response in tuple(self._responses)]
        registry_wait_task = asyncio.create_task(self.registry.wait_closed())
        results = await asyncio.gather(
            registry_wait_task,
            *response_tasks,
            return_exceptions=True,
        )
        failures.extend(
            result for result in results if isinstance(result, BaseException)
        )
        if failures:
            failure_types = ", ".join(
                sorted({type(failure).__name__ for failure in failures})
            )
            raise RuntimeError(
                f"TTS shutdown cleanup failed ({failure_types})"
            ) from None

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
