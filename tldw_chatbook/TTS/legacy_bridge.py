from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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


class LegacyBackendHost:
    def __init__(
        self,
        *,
        provider_id: str,
        app_config: Mapping[str, Any],
        manager_factory: Callable[[dict[str, Any]], TTSBackendManager],
    ) -> None:
        self.provider_id = provider_id
        self._app_config = deepcopy(dict(app_config))
        self._manager_factory = manager_factory
        self._manager: TTSBackendManager | None = None
        self._manager_lock = asyncio.Lock()
        self._operation_locks: dict[str, asyncio.Lock] = {}
        self._active_operations = 0
        self._operations_drained = asyncio.Event()
        self._operations_drained.set()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def _get_manager(self) -> TTSBackendManager:
        async with self._manager_lock:
            if self._manager is None:
                self._manager = self._manager_factory(deepcopy(self._app_config))
            return self._manager

    async def _admit_operation(self) -> None:
        async with self._manager_lock:
            if self._closed:
                raise RuntimeError("Legacy TTS host is closed")
            self._active_operations += 1
            self._operations_drained.clear()

    async def _release_operation(self) -> None:
        async with self._manager_lock:
            self._active_operations -= 1
            if self._active_operations == 0:
                self._operations_drained.set()

    async def generate(
        self,
        internal_model_id: str,
        request: OpenAISpeechRequest,
        progress_sink: ProgressSink | None,
    ) -> AsyncIterator[bytes]:
        await self._admit_operation()
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
                    async with aclosing(
                        backend.generate_speech_stream(request)
                    ) as stream:
                        async for chunk in stream:
                            yield bytes(chunk)
                finally:
                    backend.set_progress_callback(None)
        finally:
            await self._release_operation()

    async def close(self) -> None:
        async with self._manager_lock:
            if self._close_task is None:
                self._closed = True
                self._close_task = asyncio.create_task(self._close_when_drained())
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def _close_when_drained(self) -> None:
        await self._operations_drained.wait()
        async with self._manager_lock:
            manager = self._manager
            self._manager = None
        if manager is not None:
            await manager.close_all_backends()


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
) -> tuple[TTSProviderSpec, ...]:
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
