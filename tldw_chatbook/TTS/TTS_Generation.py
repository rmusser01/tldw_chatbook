from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

from tldw_chatbook.TTS.adapter_registry import (
    ReconfigureResult,
    TTSAdapterRegistry,
)
from tldw_chatbook.TTS.adapter_types import (
    ProgressSink,
    TTSAudioResponse,
    TTSProgress,
    TTSProviderCatalog,
    TTSRequest,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import resolve_legacy_route

logger = logging.getLogger(__name__)


class TTSService:
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

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        await self._operation_limit.acquire()
        try:
            lease = await self.registry.acquire(request.provider_id)
        except BaseException:
            self._operation_limit.release()
            raise

        safe_sink = _isolate_progress_sink(progress_sink)
        try:
            await lease.adapter.ensure_ready()
            response = await lease.adapter.synthesize(request, safe_sink)
        except BaseException:
            try:
                await lease.release()
            finally:
                self._operation_limit.release()
            raise

        response.add_cleanup(lease.release)
        response.add_cleanup(self._release_operation_slot)
        return response

    async def _release_operation_slot(self) -> None:
        self._operation_limit.release()

    async def generate_audio_stream(
        self,
        request: OpenAISpeechRequest,
        internal_model_id: str,
        progress_sink: ProgressSink | None = None,
    ) -> AsyncIterator[bytes]:
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
        finally:
            await response.aclose()

    async def get_catalog(
        self,
        provider_id: str,
        refresh: bool = False,
    ) -> TTSProviderCatalog:
        return await self.registry.get_catalog(provider_id, refresh=refresh)

    async def reconfigure_provider(
        self,
        provider_id: str,
        config: Mapping[str, Any],
    ) -> ReconfigureResult:
        return await self.registry.reconfigure_provider(provider_id, config)

    async def close(self) -> None:
        await self.registry.close()


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


def bind_tts_service(service: TTSService) -> None:
    global _bound_tts_service
    if _bound_tts_service is not None and _bound_tts_service is not service:
        raise RuntimeError("A different TTS service is already bound")
    _bound_tts_service = service


async def get_tts_service(
    app_config: Mapping[str, Any] | None = None,
) -> TTSService:
    del app_config
    if _bound_tts_service is None:
        raise RuntimeError("The application TTS service is not bound")
    return _bound_tts_service


def reset_tts_service_binding(
    *,
    expected: TTSService | None = None,
) -> None:
    global _bound_tts_service
    if expected is not None and _bound_tts_service not in (None, expected):
        raise RuntimeError("Refusing to reset a different TTS service")
    _bound_tts_service = None


async def close_tts_resources() -> None:
    service = _bound_tts_service
    if service is None:
        return
    try:
        await service.close()
    finally:
        reset_tts_service_binding(expected=service)
