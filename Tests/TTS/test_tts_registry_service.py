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


def registry_for_adapter(
    adapter: FakeAdapter,
    *,
    shutdown_timeout_seconds: float = 10.0,
) -> TTSAdapterRegistry:
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
        shutdown_timeout_seconds=shutdown_timeout_seconds,
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
async def test_closed_adapter_response_does_not_leak_lease_or_slot() -> None:
    class ClosedResponseAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            response = await super().synthesize(request, progress_sink)
            await response.aclose()
            return response

    adapter = ClosedResponseAdapter("openai")
    registry = registry_for_adapter(adapter)
    service = TTSService(registry, max_concurrent_operations=1)

    with pytest.raises(RuntimeError, match="Cannot add cleanup"):
        await service.synthesize(tts_request())

    await registry.reconfigure_provider("openai", {"revision": 2})
    assert adapter.close_calls == 1
    replacement_response = await asyncio.wait_for(
        service.synthesize(tts_request()),
        timeout=1,
    )
    await replacement_response.aclose()


@pytest.mark.asyncio
async def test_cancelled_response_close_waits_for_lease_and_slot_release() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class BlockingCloseAdapter(FakeAdapter):
        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    adapter = BlockingCloseAdapter("openai")
    registry = registry_for_adapter(adapter)
    service = TTSService(registry, max_concurrent_operations=1)
    response = await service.synthesize(tts_request())
    await registry.reconfigure_provider("openai", {"revision": 2})
    close_response = asyncio.create_task(response.aclose())
    await close_started.wait()

    close_response.cancel()
    await asyncio.sleep(0)
    close_returned_before_release = close_response.done()
    adapter_closed_before_release = close_finished.is_set()

    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await close_response

    replacement_response = await asyncio.wait_for(
        service.synthesize(tts_request()),
        timeout=1,
    )
    await replacement_response.aclose()

    assert close_returned_before_release is False
    assert adapter_closed_before_release is False
    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_cancelled_concurrent_response_close_waits_for_owner_cleanup() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class BlockingCloseAdapter(FakeAdapter):
        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    adapter = BlockingCloseAdapter("openai")
    registry = registry_for_adapter(adapter)
    service = TTSService(registry, max_concurrent_operations=1)
    response = await service.synthesize(tts_request())
    await registry.reconfigure_provider("openai", {"revision": 2})
    owner_close = asyncio.create_task(response.aclose())
    await close_started.wait()
    concurrent_close = asyncio.create_task(response.aclose())
    await asyncio.sleep(0)

    concurrent_close.cancel()
    await asyncio.sleep(0)
    concurrent_returned_before_release = concurrent_close.done()

    allow_close.set()
    await owner_close
    with pytest.raises(asyncio.CancelledError):
        await concurrent_close

    replacement_response = await asyncio.wait_for(
        service.synthesize(tts_request()),
        timeout=1,
    )
    await replacement_response.aclose()

    assert concurrent_returned_before_release is False
    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_adapter_cleanup_cancellation_is_primary_without_caller_cancel() -> None:
    class CancelledCloseAdapter(FakeAdapter):
        async def close(self) -> None:
            self.close_calls += 1
            raise asyncio.CancelledError("adapter cleanup cancelled")

    adapter = CancelledCloseAdapter("openai")
    registry = registry_for_adapter(adapter)
    service = TTSService(registry, max_concurrent_operations=1)
    response = await service.synthesize(tts_request())
    await registry.reconfigure_provider("openai", {"revision": 2})

    with pytest.raises(asyncio.CancelledError) as error:
        await response.aclose()

    assert getattr(error.value, "__notes__", []) == []
    replacement_response = await asyncio.wait_for(
        service.synthesize(tts_request()),
        timeout=1,
    )
    await replacement_response.aclose()


@pytest.mark.asyncio
async def test_caller_cancellation_precedes_later_resource_cleanup_failure() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class BlockingFailingCloseAdapter(FakeAdapter):
        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await allow_close.wait()
            raise RuntimeError("resource cleanup failed")

    adapter = BlockingFailingCloseAdapter("openai")
    registry = registry_for_adapter(adapter)
    service = TTSService(registry, max_concurrent_operations=1)
    response = await service.synthesize(tts_request())
    await registry.reconfigure_provider("openai", {"revision": 2})
    close_response = asyncio.create_task(response.aclose())
    await close_started.wait()

    close_response.cancel()
    await asyncio.sleep(0)
    close_returned_before_cleanup = close_response.done()
    allow_close.set()

    with pytest.raises(asyncio.CancelledError) as error:
        await close_response

    assert close_returned_before_cleanup is False
    assert error.value.__notes__ == [
        "TTS cleanup also failed while preserving the original error"
    ]
    replacement_response = await asyncio.wait_for(
        service.synthesize(tts_request()),
        timeout=1,
    )
    await replacement_response.aclose()


@pytest.mark.asyncio
async def test_synthesis_failure_preserves_primary_when_lease_cleanup_fails() -> None:
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

    with pytest.raises(RuntimeError, match="synthesis failed") as error:
        await failed_request
    assert error.value.__notes__ == [
        "TTS cleanup also failed while preserving the original error"
    ]

    response = await asyncio.wait_for(
        service.synthesize(tts_request()),
        timeout=1,
    )
    await response.aclose()


@pytest.mark.asyncio
async def test_caller_cancellation_supersedes_synthesis_failure_during_cleanup() -> (
    None
):
    synthesis_started = asyncio.Event()
    fail_synthesis = asyncio.Event()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

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
            close_started.set()
            await allow_close.wait()

    adapter = FailingAdapter("openai")
    registry = registry_for_adapter(adapter)
    service = TTSService(registry, max_concurrent_operations=1)
    failed_request = asyncio.create_task(service.synthesize(tts_request()))
    await synthesis_started.wait()
    await registry.reconfigure_provider("openai", {"revision": 2})
    fail_synthesis.set()
    await close_started.wait()

    failed_request.cancel()
    await asyncio.sleep(0)
    returned_before_cleanup = failed_request.done()
    allow_close.set()

    with pytest.raises(asyncio.CancelledError) as error:
        await failed_request

    assert returned_before_cleanup is False
    assert getattr(error.value, "__notes__", []) == []
    assert adapter.close_calls == 1
    response = await asyncio.wait_for(service.synthesize(tts_request()), timeout=1)
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
async def test_partial_generator_close_propagates_response_cleanup_failure() -> None:
    class CleanupFailureAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            del progress_sink

            async def stream():
                yield b"one"
                yield b"two"

            async def cleanup() -> None:
                self.response_close_calls += 1
                raise RuntimeError("response cleanup failed")

            return TTSAudioResponse(
                provider_id=self.provider_id,
                model_id=request.model_id,
                audio_format=request.response_format,
                content_type="audio/mpeg",
                byte_stream=stream(),
                cleanup=cleanup,
            )

    adapter = CleanupFailureAdapter("openai")
    service = service_for_adapter(adapter)
    stream = service.generate_audio_stream(
        speech_request(),
        "openai_official_tts-1",
    )

    assert await anext(stream) == b"one"
    with pytest.raises(RuntimeError, match="response cleanup failed"):
        await cast(AsyncGenerator[bytes, None], stream).aclose()

    assert adapter.response_close_calls == 1
    await service.registry.reconfigure_provider("openai", {"revision": 2})
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_stream_failure_preserves_primary_when_response_cleanup_fails() -> None:
    class StreamFailureAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            del progress_sink

            async def stream():
                raise RuntimeError("stream failed")
                yield b"unreachable"

            async def cleanup() -> None:
                self.response_close_calls += 1
                raise RuntimeError("response cleanup failed")

            return TTSAudioResponse(
                provider_id=self.provider_id,
                model_id=request.model_id,
                audio_format=request.response_format,
                content_type="audio/mpeg",
                byte_stream=stream(),
                cleanup=cleanup,
            )

    adapter = StreamFailureAdapter("openai")
    service = service_for_adapter(adapter)
    stream = service.generate_audio_stream(
        speech_request(),
        "openai_official_tts-1",
    )

    with pytest.raises(RuntimeError, match="stream failed") as error:
        await anext(stream)

    assert adapter.response_close_calls == 1
    assert error.value.__notes__ == [
        "TTS cleanup also failed while preserving the original error"
    ]
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
async def test_generator_cancellation_preserves_primary_when_cleanup_fails() -> None:
    started = asyncio.Event()

    class CancellationCleanupFailureAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            del progress_sink

            async def stream():
                started.set()
                await asyncio.Future()
                yield b"unreachable"

            async def cleanup() -> None:
                self.response_close_calls += 1
                raise RuntimeError("response cleanup failed")

            return TTSAudioResponse(
                provider_id=self.provider_id,
                model_id=request.model_id,
                audio_format=request.response_format,
                content_type="audio/mpeg",
                byte_stream=stream(),
                cleanup=cleanup,
            )

    adapter = CancellationCleanupFailureAdapter("openai")
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

    with pytest.raises(asyncio.CancelledError) as error:
        await task

    assert adapter.response_close_calls == 1
    assert error.value.__notes__ == [
        "TTS cleanup also failed while preserving the original error"
    ]
    await service.registry.reconfigure_provider("openai", {"revision": 2})
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_stream_cancellation_precedes_cleanup_originated_cancellation() -> None:
    started = asyncio.Event()

    class CancellationCleanupCancellationAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            del progress_sink

            async def stream():
                started.set()
                await asyncio.Future()
                yield b"unreachable"

            async def cleanup() -> None:
                self.response_close_calls += 1
                raise asyncio.CancelledError("cleanup cancelled")

            return TTSAudioResponse(
                provider_id=self.provider_id,
                model_id=request.model_id,
                audio_format=request.response_format,
                content_type="audio/mpeg",
                byte_stream=stream(),
                cleanup=cleanup,
            )

    adapter = CancellationCleanupCancellationAdapter("openai")
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
    task.cancel("caller cancelled")

    with pytest.raises(asyncio.CancelledError) as error:
        await task

    assert error.value.args == ("caller cancelled",)
    assert error.value.__notes__ == [
        "TTS cleanup also failed while preserving the original error"
    ]
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
async def test_value_equal_stale_service_cannot_reset_current_binding() -> None:
    class ValueEqualService(TTSService):
        def __eq__(self, other: object) -> bool:
            return isinstance(other, TTSService)

    current = ValueEqualService(registry_for_adapter(FakeAdapter("openai")))
    stale = ValueEqualService(registry_for_adapter(FakeAdapter("openai")))
    bind_tts_service(current)
    try:
        with pytest.raises(RuntimeError, match="different TTS service"):
            reset_tts_service_binding(expected=stale)
        assert await get_tts_service() is current
    finally:
        reset_tts_service_binding(expected=current)


@pytest.mark.asyncio
async def test_service_wait_closed_joins_bounded_registry_shutdown() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class BlockingCloseAdapter(FakeAdapter):
        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await allow_close.wait()

    adapter = BlockingCloseAdapter("openai")
    service = TTSService(registry_for_adapter(adapter, shutdown_timeout_seconds=0))
    response = await service.synthesize(tts_request())
    await response.aclose()

    await service.close()
    await close_started.wait()
    wait_for_close = asyncio.create_task(service.wait_closed())
    await asyncio.sleep(0)

    assert wait_for_close.done() is False
    allow_close.set()
    await wait_for_close
    await service.wait_closed()
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_zero_timeout_shutdown_retains_binding_until_adapter_closes() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class BlockingCloseAdapter(FakeAdapter):
        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await allow_close.wait()

    adapter = BlockingCloseAdapter("openai")
    service = TTSService(registry_for_adapter(adapter, shutdown_timeout_seconds=0))
    response = await service.synthesize(tts_request())
    await response.aclose()
    bind_tts_service(service)
    shutdown = asyncio.create_task(close_tts_resources())
    try:
        await close_started.wait()
        await asyncio.sleep(0)

        assert shutdown.done() is False
        assert await get_tts_service() is service
        assert adapter.close_calls == 1

        allow_close.set()
        await shutdown
        with pytest.raises(RuntimeError, match="not bound"):
            await get_tts_service()
        assert adapter.close_calls == 1
    finally:
        allow_close.set()
        await asyncio.gather(shutdown, return_exceptions=True)
        reset_tts_service_binding(expected=service)


@pytest.mark.asyncio
async def test_cancelled_resource_shutdown_retains_binding_until_close_finishes() -> (
    None
):
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class BlockingCloseAdapter(FakeAdapter):
        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    adapter = BlockingCloseAdapter("openai")
    service = service_for_adapter(adapter)
    response = await service.synthesize(tts_request())
    await response.aclose()
    bind_tts_service(service)
    first_close = asyncio.create_task(close_tts_resources())
    await close_started.wait()
    second_close = asyncio.create_task(close_tts_resources())
    await asyncio.sleep(0)

    first_close.cancel()
    await asyncio.sleep(0)
    first_returned_before_close = first_close.done()
    second_returned_before_close = second_close.done()
    try:
        binding_retained_before_close = await get_tts_service() is service
    except RuntimeError:
        binding_retained_before_close = False

    allow_close.set()
    first_result, second_result = await asyncio.gather(
        first_close,
        second_close,
        return_exceptions=True,
    )

    assert first_returned_before_close is False
    assert second_returned_before_close is False
    assert binding_retained_before_close is True
    assert isinstance(first_result, asyncio.CancelledError)
    assert second_result is None
    assert close_finished.is_set()
    assert adapter.close_calls == 1
    with pytest.raises(RuntimeError, match="not bound"):
        await get_tts_service()


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
