from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tldw_chatbook.TTS._async_lifecycle import join_retained_task
from tldw_chatbook.TTS.adapter_types import (
    ProgressSink,
    TTSAudioResponse,
    TTSProgress,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSProviderSpec,
    TTSRequest,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_catalogs import (
    ELEVENLABS_MODELS,
    legacy_catalog,
)

if TYPE_CHECKING:
    from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager


class UnknownLegacyModelError(LookupError):
    """Raised when a compatibility internal model ID is not enumerated."""


@dataclass(frozen=True, slots=True)
class LegacyRoute:
    provider_id: str
    internal_model_id: str


OPENAI_INTERNAL_IDS = (
    "openai_official_tts-1",
    "openai_official_tts-1-hd",
    "openai_official_tts1",
    "openai_official_tts1hd",
)
LEGACY_PROVIDER_IDS = (
    "openai",
    "elevenlabs",
    "kokoro",
    "chatterbox",
    "higgs",
    "alltalk",
)
_STATIC_ROUTES = {
    "local_kokoro_default_onnx": "kokoro",
    "local_kokoro_default_pytorch": "kokoro",
    "local_chatterbox_default": "chatterbox",
    "local_higgs_default": "higgs",
    "local_higgs_v2": "higgs",
    "alltalk_default": "alltalk",
    "alltalk_alltalk": "alltalk",
}
LEGACY_ROUTES = {
    **{internal_id: "openai" for internal_id in OPENAI_INTERNAL_IDS},
    **{f"elevenlabs_{model_id}": "elevenlabs" for model_id in ELEVENLABS_MODELS},
    **_STATIC_ROUTES,
}

_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "elevenlabs": "ElevenLabs",
    "kokoro": "Kokoro (Local)",
    "chatterbox": "Chatterbox (Local)",
    "higgs": "Higgs Audio (Local)",
    "alltalk": "AllTalk (Local)",
}
_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
}


def resolve_legacy_route(internal_model_id: str) -> LegacyRoute:
    provider_id = LEGACY_ROUTES.get(internal_model_id)
    if provider_id is None:
        raise UnknownLegacyModelError("The selected TTS model is not available")
    return LegacyRoute(provider_id, internal_model_id)


def _content_type(audio_format: str) -> str:
    return _CONTENT_TYPES.get(audio_format, "application/octet-stream")


def _legacy_progress_callback(
    progress_sink: ProgressSink,
) -> Callable[[Mapping[str, Any]], Awaitable[None]]:
    async def report(info: Mapping[str, Any]) -> None:
        raw_fraction = info.get("progress")
        fraction = (
            max(0.0, min(1.0, float(raw_fraction)))
            if isinstance(raw_fraction, (int, float))
            else None
        )
        raw_metrics = info.get("metrics")
        metrics = {
            str(key): value
            for key, value in (
                raw_metrics.items() if isinstance(raw_metrics, Mapping) else ()
            )
            if isinstance(value, (str, int, float, bool))
        }
        try:
            await progress_sink(
                TTSProgress(
                    status=str(info.get("status") or "Generating"),
                    fraction=fraction,
                    processed=(
                        int(info["processed"])
                        if isinstance(info.get("processed"), int)
                        else None
                    ),
                    total=(
                        int(info["total"])
                        if isinstance(info.get("total"), int)
                        else None
                    ),
                    metrics=metrics,
                )
            )
        except Exception:
            return

    return report


async def _close_delegated_stream(
    stream: AsyncGenerator[bytes, None],
) -> None:
    close_task = asyncio.create_task(stream.aclose())
    cancellation: asyncio.CancelledError | None = None
    while not close_task.done():
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
    close_task.result()
    if cancellation is not None:
        raise cancellation


class _LegacyOperation(AsyncIterator[bytes]):
    def __init__(
        self,
        host: LegacyBackendHost,
        internal_model_id: str,
        request: OpenAISpeechRequest,
        progress_sink: ProgressSink | None,
    ) -> None:
        self._iterator = host._run_operation(
            self,
            internal_model_id,
            request,
            progress_sink,
        )
        self._drive_lock = asyncio.Lock()
        self._driver_task: asyncio.Task[Any] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False

    def __aiter__(self) -> _LegacyOperation:
        return self

    async def __anext__(self) -> bytes:
        if self._closing:
            raise StopAsyncIteration
        async with self._drive_lock:
            if self._closing:
                raise StopAsyncIteration
            self._driver_task = asyncio.current_task()
            try:
                return await anext(self._iterator)
            finally:
                self._driver_task = None

    async def aclose(self) -> None:
        await join_retained_task(self.start_close())

    def start_close(self) -> asyncio.Task[None]:
        if self._close_task is None:
            self._closing = True
            driver_task = self._driver_task
            if driver_task is not None and driver_task is not asyncio.current_task():
                driver_task.cancel()
            self._close_task = asyncio.create_task(self._complete_close())
        return self._close_task

    async def _complete_close(self) -> None:
        async with self._drive_lock:
            await self._iterator.aclose()


class LegacyBackendHost:
    def __init__(
        self,
        *,
        provider_id: str,
        app_config: Mapping[str, Any],
        manager_factory: Callable[[dict[str, Any]], TTSBackendManager],
        shutdown_timeout_seconds: float = 10.0,
    ) -> None:
        if shutdown_timeout_seconds < 0:
            raise ValueError("shutdown_timeout_seconds cannot be negative")

        self.provider_id = provider_id
        self._app_config = deepcopy(dict(app_config))
        self._manager_factory = manager_factory
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._manager: TTSBackendManager | None = None
        self._manager_lock = asyncio.Lock()
        self._operation_locks: dict[str, asyncio.Lock] = {}
        self._active_operations = 0
        self._active_operation_handles: set[_LegacyOperation] = set()
        self._operations_drained = asyncio.Event()
        self._operations_drained.set()
        self._closed = False
        self._manager_detached = False
        self._close_task: asyncio.Task[None] | None = None
        self._manager_close_task: asyncio.Task[None] | None = None

    async def _get_manager(self) -> TTSBackendManager:
        async with self._manager_lock:
            if self._manager is None:
                if self._manager_detached:
                    raise RuntimeError("Legacy TTS host is closed")
                self._manager = self._manager_factory(deepcopy(self._app_config))
            return self._manager

    async def _admit_operation(self, operation: _LegacyOperation) -> None:
        async with self._manager_lock:
            if self._closed:
                raise RuntimeError("Legacy TTS host is closed")
            self._active_operations += 1
            self._active_operation_handles.add(operation)
            self._operations_drained.clear()

    async def _release_operation(self, operation: _LegacyOperation) -> None:
        async with self._manager_lock:
            self._active_operations -= 1
            self._active_operation_handles.discard(operation)
            if self._active_operations == 0:
                self._operations_drained.set()

    def generate(
        self,
        internal_model_id: str,
        request: OpenAISpeechRequest,
        progress_sink: ProgressSink | None,
    ) -> AsyncIterator[bytes]:
        return _LegacyOperation(
            self,
            internal_model_id,
            request,
            progress_sink,
        )

    async def _run_operation(
        self,
        operation: _LegacyOperation,
        internal_model_id: str,
        request: OpenAISpeechRequest,
        progress_sink: ProgressSink | None,
    ) -> AsyncGenerator[bytes, None]:
        await self._admit_operation(operation)
        try:
            lock = self._operation_locks.setdefault(
                internal_model_id,
                asyncio.Lock(),
            )
            async with lock:
                manager = await self._get_manager()
                backend = await manager.get_backend(internal_model_id)
                if backend is None:
                    raise ValueError(f"TTS model '{request.model}' is not available")
                backend.set_progress_callback(
                    _legacy_progress_callback(progress_sink)
                    if progress_sink is not None
                    else None
                )
                try:
                    stream = backend.generate_speech_stream(request)
                    try:
                        async for chunk in stream:
                            yield bytes(chunk)
                    finally:
                        await _close_delegated_stream(stream)
                finally:
                    backend.set_progress_callback(None)
        finally:
            await self._release_operation(operation)

    async def close(self) -> None:
        async with self._manager_lock:
            if self._close_task is None:
                self._closed = True
                self._close_task = asyncio.create_task(self._close_when_drained())
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def _close_when_drained(self) -> None:
        if not self._operations_drained.is_set():
            try:
                await asyncio.wait_for(
                    self._operations_drained.wait(),
                    timeout=self._shutdown_timeout_seconds,
                )
            except TimeoutError:
                pass

        operation_error: BaseException | None = None
        operation_tasks: set[asyncio.Task[None]] = set()
        if not self._operations_drained.is_set():
            async with self._manager_lock:
                operations = set(self._active_operation_handles)
            operation_tasks = {operation.start_close() for operation in operations}
            await self._wait_pending(operation_tasks)

        async with self._manager_lock:
            manager = self._manager
            self._manager = None
            self._manager_detached = True
        manager_error: BaseException | None = None
        if manager is not None:
            try:
                await self._close_manager(manager)
            except BaseException as error:
                manager_error = error

        pending_operations: set[asyncio.Task[Any]] = {
            operation_task
            for operation_task in operation_tasks
            if not operation_task.done()
        }
        for operation_task in pending_operations:
            operation_task.add_done_callback(self._observe_task_result)
        for operation_task in operation_tasks - pending_operations:
            try:
                operation_task.result()
            except BaseException as error:
                operation_error = operation_error or error

        if manager_error is not None:
            raise manager_error
        if pending_operations:
            raise TimeoutError("Legacy TTS operations did not stop before shutdown")
        if operation_error is not None:
            raise operation_error

    async def _wait_pending(
        self,
        tasks: set[asyncio.Task[Any]],
    ) -> set[asyncio.Task[Any]]:
        if not tasks:
            return set()
        await asyncio.sleep(0)
        pending = {task for task in tasks if not task.done()}
        if pending and self._shutdown_timeout_seconds:
            _, pending = await asyncio.wait(
                pending,
                timeout=self._shutdown_timeout_seconds,
            )
        return pending

    async def _close_manager(self, manager: TTSBackendManager) -> None:
        close_task = asyncio.create_task(manager.close_all_backends())
        self._manager_close_task = close_task
        pending = await self._wait_pending({close_task})
        if not pending:
            close_task.result()
            return

        close_task.cancel()
        pending = await self._wait_pending({close_task})
        if pending:
            close_task.add_done_callback(self._observe_task_result)
        else:
            try:
                close_task.result()
            except asyncio.CancelledError:
                pass
        raise TimeoutError("Legacy TTS manager did not close before shutdown")

    @staticmethod
    def _observe_task_result(task: asyncio.Task[Any]) -> None:
        try:
            task.exception()
        except BaseException:
            pass


class LegacyTTSAdapter:
    _allowed_options = {
        "_legacy_openai_request",
        "_legacy_internal_model_id",
    }

    def __init__(
        self,
        provider_id: str,
        host: LegacyBackendHost,
        catalog: TTSProviderCatalog,
    ) -> None:
        self.provider_id = provider_id
        self.host = host
        self._catalog = catalog

    async def ensure_ready(self) -> None:
        return

    async def get_catalog(
        self,
        refresh: bool = False,
    ) -> TTSProviderCatalog:
        del refresh
        return self._catalog

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        if request.provider_id != self.provider_id:
            raise ValueError("TTS request does not match provider")
        if set(request.options) != self._allowed_options:
            raise ValueError("Invalid legacy adapter options")
        legacy_request = request.options["_legacy_openai_request"]
        internal_id = request.options["_legacy_internal_model_id"]
        if not isinstance(legacy_request, OpenAISpeechRequest):
            raise TypeError("Legacy request must be OpenAISpeechRequest")
        if not isinstance(internal_id, str):
            raise TypeError("Legacy internal model ID must be str")
        route = resolve_legacy_route(internal_id)
        if route.provider_id != self.provider_id:
            raise ValueError("Legacy route does not match provider")
        return TTSAudioResponse(
            provider_id=self.provider_id,
            model_id=request.model_id,
            audio_format=legacy_request.response_format,
            content_type=_content_type(legacy_request.response_format),
            byte_stream=self.host.generate(
                route.internal_model_id,
                legacy_request,
                progress_sink,
            ),
        )

    async def close(self) -> None:
        await self.host.close()


def legacy_provider_specs(
    app_config: Mapping[str, Any],
    *,
    manager_factory: Callable[
        [str, dict[str, Any]],
        TTSBackendManager,
    ]
    | None = None,
    shutdown_timeout_seconds: float = 10.0,
) -> tuple[TTSProviderSpec, ...]:
    if shutdown_timeout_seconds < 0:
        raise ValueError("shutdown_timeout_seconds cannot be negative")

    def default_manager_factory(
        _provider_id: str,
        config: dict[str, Any],
    ) -> TTSBackendManager:
        from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager

        return TTSBackendManager(app_config=config)

    create_manager = manager_factory or default_manager_factory
    config_snapshot = deepcopy(dict(app_config))
    specs: list[TTSProviderSpec] = []
    for provider_id in LEGACY_PROVIDER_IDS:

        def create_adapter(
            config: Mapping[str, Any],
            selected_provider: str = provider_id,
        ) -> LegacyTTSAdapter:
            provider_config = deepcopy(dict(config["app_config"]))
            host = LegacyBackendHost(
                provider_id=selected_provider,
                app_config=provider_config,
                manager_factory=lambda current_config: create_manager(
                    selected_provider,
                    current_config,
                ),
                shutdown_timeout_seconds=shutdown_timeout_seconds,
            )
            return LegacyTTSAdapter(
                selected_provider,
                host,
                legacy_catalog(selected_provider),
            )

        specs.append(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id=provider_id,
                    display_name=_DISPLAY_NAMES[provider_id],
                    native=False,
                ),
                factory=create_adapter,
                initial_config={"app_config": deepcopy(config_snapshot)},
            )
        )
    return tuple(specs)
