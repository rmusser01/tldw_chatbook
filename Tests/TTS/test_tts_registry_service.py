from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from typing import Any, cast

import pytest

from Tests.TTS.adapter_fakes import FakeAdapter, FakeAdapterFactory
from tldw_chatbook.TTS.adapter_bootstrap import (
    _legacy_config_snapshot,
    build_default_tts_service,
)
from tldw_chatbook.TTS.adapter_registry import (
    ReconfigureResult,
    TTSAdapterRegistry,
)
from tldw_chatbook.TTS.adapter_types import (
    ProgressSink,
    TTSAudioResponse,
    TTSProgress,
    TTSProviderDescriptor,
    TTSProviderSpec,
    TTSRequest,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.TTS_Generation import (
    TTSService,
    bind_tts_service,
    close_tts_resources,
    get_tts_service,
    reset_tts_service_binding,
)


def tts_request(provider_id: str = "openai") -> TTSRequest:
    return TTSRequest(
        provider_id=provider_id,
        model_id="tts-1",
        text="hello",
        voice="alloy",
        response_format="mp3",
    )


def speech_request() -> OpenAISpeechRequest:
    return OpenAISpeechRequest(
        model="tts-1",
        input="hello",
        voice="alloy",
        response_format="mp3",
    )


def registry_for_adapter(adapter: FakeAdapter) -> TTSAdapterRegistry:
    replacements = FakeAdapterFactory(adapter.provider_id)
    calls = 0

    def factory(config: Mapping[str, Any]) -> FakeAdapter:
        nonlocal calls
        del config
        calls += 1
        return adapter if calls == 1 else replacements({})

    return TTSAdapterRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id=adapter.provider_id,
                    display_name=adapter.provider_id,
                    native=True,
                ),
                factory=factory,
                initial_config={"revision": 1},
            ),
        ),
        aliases={},
    )


def service_for_adapter(adapter: FakeAdapter) -> TTSService:
    return TTSService(registry_for_adapter(adapter))


@pytest.mark.asyncio
async def test_synthesize_holds_lease_until_response_close() -> None:
    adapter = FakeAdapter("openai", chunks=(b"a", b"b"))
    registry = registry_for_adapter(adapter)
    service = TTSService(registry, max_concurrent_operations=4)

    response = await service.synthesize(tts_request())
    assert adapter.close_calls == 0
    await registry.reconfigure_provider("openai", {"revision": 2})
    assert adapter.close_calls == 0

    await response.aclose()
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_synthesis_failure_releases_slot_when_retired_adapter_close_fails() -> (
    None
):
    synthesis_started = asyncio.Event()
    fail_synthesis = asyncio.Event()

    class FailingAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            del request, progress_sink
            synthesis_started.set()
            await fail_synthesis.wait()
            raise RuntimeError("synthesis failed")

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("adapter close failed")

    adapter = FailingAdapter("openai")
    registry = registry_for_adapter(adapter)
    service = TTSService(registry, max_concurrent_operations=1)
    failed_request = asyncio.create_task(service.synthesize(tts_request()))
    await synthesis_started.wait()
    await registry.reconfigure_provider("openai", {"revision": 2})
    fail_synthesis.set()

    with pytest.raises(RuntimeError, match="adapter close failed"):
        await failed_request

    response = await asyncio.wait_for(
        service.synthesize(tts_request()),
        timeout=1,
    )
    await response.aclose()


@pytest.mark.asyncio
async def test_default_concurrency_limit_holds_four_open_responses() -> None:
    service = service_for_adapter(FakeAdapter("openai"))
    responses = [await service.synthesize(tts_request()) for _ in range(4)]
    fifth_task = asyncio.create_task(service.synthesize(tts_request()))

    await asyncio.sleep(0)
    assert not fifth_task.done()

    await responses.pop().aclose()
    fifth = await fifth_task
    await fifth.aclose()
    for response in responses:
        await response.aclose()


def test_service_concurrency_limit_is_instance_scoped_across_event_loops() -> None:
    first = service_for_adapter(FakeAdapter("openai"))
    second = service_for_adapter(FakeAdapter("openai"))

    async def consume(service: TTSService) -> bytes:
        response = await service.synthesize(tts_request())
        try:
            return b"".join([chunk async for chunk in response.byte_stream])
        finally:
            await response.aclose()

    assert asyncio.run(consume(first)) == b"audio"
    assert asyncio.run(consume(second)) == b"audio"
    assert first._operation_limit is not second._operation_limit


@pytest.mark.asyncio
async def test_compatibility_generator_closes_after_partial_consumption() -> None:
    adapter = FakeAdapter("openai", chunks=(b"one", b"two"))
    service = service_for_adapter(adapter)
    stream = service.generate_audio_stream(
        speech_request(),
        "openai_official_tts-1",
    )

    assert await anext(stream) == b"one"
    await cast(AsyncGenerator[bytes, None], stream).aclose()

    assert adapter.response_close_calls == 1
    await service.registry.reconfigure_provider("openai", {"revision": 2})
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_compatibility_generator_releases_response_on_cancellation() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class CancellationAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            del progress_sink

            async def stream():
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()
                yield b"unreachable"

            async def cleanup() -> None:
                self.response_close_calls += 1

            return TTSAudioResponse(
                provider_id=self.provider_id,
                model_id=request.model_id,
                audio_format=request.response_format,
                content_type="audio/mpeg",
                byte_stream=stream(),
                cleanup=cleanup,
            )

    adapter = CancellationAdapter("openai")
    service = service_for_adapter(adapter)

    async def consume_one() -> bytes:
        return await anext(
            service.generate_audio_stream(
                speech_request(),
                "openai_official_tts-1",
            )
        )

    task = asyncio.create_task(consume_one())
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert adapter.response_close_calls == 1
    await service.registry.reconfigure_provider("openai", {"revision": 2})
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_progress_sink_failure_does_not_fail_synthesis() -> None:
    async def broken_sink(_progress: TTSProgress) -> None:
        raise RuntimeError("display failed")

    service = service_for_adapter(FakeAdapter("openai"))
    response = await service.synthesize(
        tts_request(),
        progress_sink=broken_sink,
    )
    assert b"".join([chunk async for chunk in response.byte_stream]) == b"audio"
    await response.aclose()


@pytest.mark.asyncio
async def test_catalog_and_reconfigure_delegate_to_registry() -> None:
    adapter = FakeAdapter("openai")
    service = service_for_adapter(adapter)

    catalog = await service.get_catalog("openai", refresh=True)
    result = await service.reconfigure_provider(
        "openai",
        {"revision": 2},
    )

    assert catalog.provider_id == "openai"
    assert adapter.ensure_ready_calls == 1
    assert result is ReconfigureResult.CHANGED
    assert adapter.close_calls == 1


def test_bootstrap_preserves_nested_raw_provider_configuration() -> None:
    source: dict[str, Any] = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "API": {"openai_api_key": "secret"},
            "app_tts": {"default_format": "wav"},
        },
        "APP_TTS_CONFIG": {"default_format": "mp3"},
    }

    snapshot = _legacy_config_snapshot(source)
    source["COMPREHENSIVE_CONFIG_RAW"]["API"]["openai_api_key"] = "changed"

    assert snapshot == {
        "API": {"openai_api_key": "secret"},
        "app_tts": {"default_format": "wav"},
    }


def test_bootstrap_falls_back_to_normalized_tts_configuration() -> None:
    source = {"APP_TTS_CONFIG": {"default_format": "mp3"}}

    snapshot = _legacy_config_snapshot(source)
    source["APP_TTS_CONFIG"]["default_format"] = "wav"

    assert snapshot["app_tts"] == {"default_format": "mp3"}


def test_default_bootstrap_has_six_exact_ids_no_aliases_and_is_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_ids = (
        "openai",
        "elevenlabs",
        "kokoro",
        "chatterbox",
        "higgs",
        "alltalk",
    )
    factories = {
        provider_id: FakeAdapterFactory(provider_id) for provider_id in provider_ids
    }

    def provider_specs(
        config: Mapping[str, Any],
    ) -> tuple[TTSProviderSpec, ...]:
        assert config == {"app_tts": {}}
        return tuple(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id=provider_id,
                    display_name=provider_id,
                    native=False,
                ),
                factory=factories[provider_id],
                initial_config={},
            )
            for provider_id in provider_ids
        )

    monkeypatch.setattr(
        "tldw_chatbook.TTS.adapter_bootstrap.legacy_provider_specs",
        provider_specs,
    )

    service = build_default_tts_service({})

    assert (
        tuple(item.provider_id for item in service.registry.descriptors())
        == provider_ids
    )
    assert service.registry.aliases() == {}
    assert all(factory.calls == 0 for factory in factories.values())


@pytest.mark.asyncio
async def test_accessor_requires_an_explicit_binding() -> None:
    reset_tts_service_binding()

    with pytest.raises(RuntimeError, match="not bound"):
        await get_tts_service({"app_tts": {"default_provider": "openai"}})


@pytest.mark.asyncio
async def test_accessor_returns_bound_service_without_retaining_config() -> None:
    first = service_for_adapter(FakeAdapter("openai"))
    bind_tts_service(first)
    try:
        assert await get_tts_service({"value": "first"}) is first
        assert await get_tts_service({"value": "second"}) is first
    finally:
        reset_tts_service_binding(expected=first)


def test_binding_rejects_a_different_live_service() -> None:
    first = service_for_adapter(FakeAdapter("openai"))
    second = service_for_adapter(FakeAdapter("openai"))
    bind_tts_service(first)
    try:
        with pytest.raises(RuntimeError, match="already bound"):
            bind_tts_service(second)
    finally:
        reset_tts_service_binding(expected=first)


@pytest.mark.asyncio
async def test_close_resources_is_idempotent_and_clears_binding() -> None:
    adapter = FakeAdapter("openai")
    service = service_for_adapter(adapter)
    response = await service.synthesize(tts_request())
    await response.aclose()
    bind_tts_service(service)

    await close_tts_resources()
    await close_tts_resources()

    assert adapter.close_calls == 1
    with pytest.raises(RuntimeError, match="not bound"):
        await get_tts_service()
