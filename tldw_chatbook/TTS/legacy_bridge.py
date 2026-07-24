from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tldw_chatbook.TTS._async_lifecycle import (
    current_shutdown_deadline,
    join_retained_task,
)
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

logger = logging.getLogger(__name__)
_DELEGATED_CLEANUP_FAILURE_NOTE = (
    "Legacy TTS cleanup also failed during caller cancellation"
)


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
_APP_TTS_PREFIXES = {
    "openai": "OPENAI_",
    "elevenlabs": "ELEVENLABS_",
    "kokoro": "KOKORO_",
    "chatterbox": "CHATTERBOX_",
    "higgs": "HIGGS_",
    "alltalk": "ALLTALK_",
}
_BACKEND_PREFIXES = {
    "openai": "openai_official",
    "elevenlabs": "elevenlabs_",
    "kokoro": "local_kokoro_",
    "chatterbox": "local_chatterbox_",
    "higgs": "local_higgs_",
    "alltalk": "alltalk_",
}


def _legacy_config_snapshot(
    app_config: Mapping[str, Any],
) -> dict[str, Any]:
    nested_raw = app_config.get("COMPREHENSIVE_CONFIG_RAW")
    source = nested_raw if isinstance(nested_raw, Mapping) else app_config
    snapshot = deepcopy(dict(source))
    if "app_tts" not in snapshot:
        normalized_tts = app_config.get("APP_TTS_CONFIG", {})
        snapshot["app_tts"] = (
            deepcopy(dict(normalized_tts))
            if isinstance(normalized_tts, Mapping)
            else {}
        )
    return snapshot


def _first_mapping_value(
    configuration: Mapping[str, Any],
    locations: tuple[tuple[str, str], ...],
) -> Any:
    for section, key in locations:
        values = configuration.get(section)
        if isinstance(values, Mapping):
            value = values.get(key)
            if value:
                return value
    return None


def legacy_provider_config(
    provider_id: str,
    app_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the effective legacy-manager input for one provider.

    Args:
        provider_id: Exact legacy provider identifier.
        app_config: Normalized settings with an optional raw configuration.

    Returns:
        Provider-scoped registry configuration for the legacy adapter factory.

    Raises:
        ValueError: If the provider identifier is not registered by the bridge.
    """
    if provider_id not in LEGACY_PROVIDER_IDS:
        raise ValueError(f"Unknown legacy TTS provider: {provider_id}")

    raw = _legacy_config_snapshot(app_config)
    raw_tts = raw.get("app_tts")
    effective_tts = dict(raw_tts) if isinstance(raw_tts, Mapping) else {}

    prefix = _APP_TTS_PREFIXES[provider_id]
    projected: dict[str, Any] = {
        "app_tts": {
            key: deepcopy(value)
            for key, value in effective_tts.items()
            if str(key).startswith(prefix)
        }
    }
    global_settings = raw.get("global_tts_settings")
    if isinstance(global_settings, Mapping):
        projected["global_tts_settings"] = deepcopy(dict(global_settings))

    backend_prefix = _BACKEND_PREFIXES[provider_id]
    projected.update(
        {
            str(key): deepcopy(dict(value))
            for key, value in raw.items()
            if str(key).startswith(backend_prefix) and isinstance(value, Mapping)
        }
    )

    if provider_id == "openai":
        openai_key_locations = (
            ("api_settings.openai", "api_key"),
            ("openai_api", "api_key"),
            ("API", "openai_api_key"),
        )
        api_key = os.getenv("OPENAI_API_KEY") or _first_mapping_value(
            raw,
            openai_key_locations,
        )
        api_key = api_key or _first_mapping_value(
            app_config,
            openai_key_locations,
        )
        if api_key:
            projected["openai_api"] = {"api_key": api_key}
    elif provider_id == "elevenlabs":
        elevenlabs_key_locations = (
            ("API", "elevenlabs_api_key"),
            ("elevenlabs_api", "api_key"),
        )
        api_key = os.getenv("ELEVENLABS_API_KEY") or _first_mapping_value(
            raw,
            elevenlabs_key_locations,
        )
        api_key = api_key or _first_mapping_value(
            app_config,
            elevenlabs_key_locations,
        )
        if api_key:
            projected["elevenlabs_api"] = {"api_key": api_key}
    elif provider_id == "kokoro":
        if "KOKORO_MODEL_PATH" in os.environ:
            model_path = os.environ["KOKORO_MODEL_PATH"]
            projected["app_tts"]["KOKORO_ONNX_MODEL_PATH_DEFAULT"] = model_path
            projected["app_tts"]["KOKORO_PT_MODEL_PATH_DEFAULT"] = model_path
        if "KOKORO_VOICES_PATH" in os.environ:
            projected["app_tts"]["KOKORO_ONNX_VOICES_JSON_DEFAULT"] = os.environ[
                "KOKORO_VOICES_PATH"
            ]
    elif provider_id == "higgs":
        higgs_settings = raw.get("HiggsSettings")
        effective_higgs = (
            deepcopy(dict(higgs_settings))
            if isinstance(higgs_settings, Mapping)
            else {}
        )
        for key, value in raw.items():
            if str(key).startswith("HIGGS_"):
                effective_higgs[str(key).removeprefix("HIGGS_").lower()] = deepcopy(
                    value
                )
        if "HIGGS_MODEL_PATH" in os.environ:
            effective_higgs["model_path"] = os.environ["HIGGS_MODEL_PATH"]
        projected["HiggsSettings"] = effective_higgs
        projected["app_tts"] = {}

    return {"app_config": projected}


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
        except BaseException:
            break
    close_error: BaseException | None = None
    try:
        close_task.result()
    except BaseException as error:
        close_error = error
    if cancellation is not None:
        if close_error is not None and not isinstance(
            close_error,
            asyncio.CancelledError,
        ):
            _record_delegated_cleanup_failure(cancellation, close_error)
        raise cancellation
    if close_error is not None:
        raise close_error


def _record_delegated_cleanup_failure(
    cancellation: asyncio.CancelledError,
    cleanup_error: BaseException,
) -> None:
    cancellation.add_note(_DELEGATED_CLEANUP_FAILURE_NOTE)
    logger.warning(
        "Legacy TTS delegated cleanup failed during cancellation: %s",
        type(cleanup_error).__name__,
    )


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
            driver_task = asyncio.current_task()
            cancellation_requests = (
                driver_task.cancelling() if driver_task is not None else 0
            )
            self._driver_task = driver_task
            delegated_error: BaseException | None = None
            try:
                chunk = await anext(self._iterator)
            except BaseException as error:
                delegated_error = error
            finally:
                self._driver_task = None
            if (
                driver_task is not None
                and driver_task.cancelling() > cancellation_requests
            ):
                cancellation = asyncio.CancelledError()
                if delegated_error is not None and not isinstance(
                    delegated_error,
                    asyncio.CancelledError,
                ):
                    _record_delegated_cleanup_failure(
                        cancellation,
                        delegated_error,
                    )
                raise cancellation
            if delegated_error is not None:
                raise delegated_error
            return chunk

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
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self._shutdown_timeout_seconds
                inherited_deadline = current_shutdown_deadline()
                if inherited_deadline is not None:
                    deadline = min(deadline, inherited_deadline)
                self._close_task = asyncio.create_task(
                    self._close_when_drained(deadline)
                )
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def _close_when_drained(self, deadline: float) -> None:
        if not self._operations_drained.is_set():
            remaining = self._remaining_time(deadline)
            try:
                if remaining > 0:
                    await asyncio.wait_for(
                        self._operations_drained.wait(),
                        timeout=remaining,
                    )
            except TimeoutError:
                pass

        operation_error: BaseException | None = None
        operation_tasks: set[asyncio.Task[None]] = set()
        if not self._operations_drained.is_set():
            async with self._manager_lock:
                operations = set(self._active_operation_handles)
            operation_tasks = {operation.start_close() for operation in operations}
            await self._wait_pending(operation_tasks, deadline)

        async with self._manager_lock:
            manager = self._manager
            self._manager = None
            self._manager_detached = True
        manager_error: BaseException | None = None
        if manager is not None:
            try:
                await self._close_manager(manager, deadline)
            except BaseException as error:
                manager_error = error

        await asyncio.sleep(0)
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
        deadline: float,
    ) -> set[asyncio.Task[Any]]:
        if not tasks:
            return set()
        await asyncio.sleep(0)
        pending = {task for task in tasks if not task.done()}
        remaining = self._remaining_time(deadline)
        if pending:
            _, pending = await asyncio.wait(
                pending,
                timeout=remaining,
            )
        return pending

    async def _close_manager(
        self,
        manager: TTSBackendManager,
        deadline: float,
    ) -> None:
        close_task = asyncio.create_task(manager.close_all_backends())
        self._manager_close_task = close_task
        pending = await self._wait_pending({close_task}, deadline)
        if not pending:
            close_task.result()
            return

        close_task.cancel()
        pending = await self._wait_pending({close_task}, deadline)
        if pending:
            close_task.add_done_callback(self._observe_task_result)
        else:
            try:
                close_task.result()
            except asyncio.CancelledError:
                pass
        raise TimeoutError("Legacy TTS manager did not close before shutdown")

    @staticmethod
    def _remaining_time(deadline: float) -> float:
        return max(0.0, deadline - asyncio.get_running_loop().time())

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
                initial_config=legacy_provider_config(provider_id, app_config),
            )
        )
    return tuple(specs)
