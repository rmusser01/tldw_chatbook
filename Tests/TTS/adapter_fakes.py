from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    ProgressSink,
    TTSAudioResponse,
    TTSModelInfo,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSProviderSpec,
    TTSRequest,
)


class FakeAdapter:
    def __init__(
        self,
        provider_id: str,
        *,
        chunks: tuple[bytes, ...] = (b"audio",),
        close_order: list[str] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.chunks = chunks
        self.close_order = close_order
        self.ensure_ready_calls = 0
        self.synthesize_calls = 0
        self.close_calls = 0
        self.response_close_calls = 0

    async def ensure_ready(self) -> None:
        self.ensure_ready_calls += 1

    async def get_catalog(self, refresh: bool = False) -> TTSProviderCatalog:
        del refresh
        return TTSProviderCatalog(
            provider_id=self.provider_id,
            revision=1,
            health=ProviderHealth(state="available", fresh=True),
            models=(
                TTSModelInfo(
                    model_id="model",
                    display_name="Model",
                    family="fake",
                    upstream_mode="tts",
                    formats=("wav",),
                    voices=("default",),
                    supports_speed=True,
                ),
            ),
        )

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        self.synthesize_calls += 1
        if progress_sink is not None:
            from tldw_chatbook.TTS.adapter_types import TTSProgress

            await progress_sink(TTSProgress(status="Generating"))

        async def stream():
            yield_chunks = self.chunks
            for chunk in yield_chunks:
                yield chunk

        async def cleanup() -> None:
            self.response_close_calls += 1

        return TTSAudioResponse(
            provider_id=self.provider_id,
            model_id=request.model_id,
            audio_format=request.response_format,
            content_type="audio/wav",
            byte_stream=stream(),
            cleanup=cleanup,
        )

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_order is not None:
            self.close_order.append(self.provider_id)


class FakeAdapterFactory:
    def __init__(
        self,
        provider_id: str,
        *,
        close_order: list[str] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.close_order = close_order
        self.calls = 0
        self.instances: list[FakeAdapter] = []

    def __call__(self, config: Mapping[str, Any]) -> FakeAdapter:
        del config
        self.calls += 1
        adapter = FakeAdapter(self.provider_id, close_order=self.close_order)
        self.instances.append(adapter)
        return adapter


def provider_spec(
    provider_id: str,
    factory: FakeAdapterFactory,
    config: Mapping[str, Any] | None = None,
    *,
    exclusive: bool = False,
) -> TTSProviderSpec:
    return TTSProviderSpec(
        descriptor=TTSProviderDescriptor(
            provider_id=provider_id,
            display_name=provider_id,
            native=True,
        ),
        factory=factory,
        initial_config={} if config is None else config,
        exclusive_reconfigure=exclusive,
    )
