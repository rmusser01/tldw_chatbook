from __future__ import annotations

import asyncio
import logging
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
    TTSOperationError,
    TTSProgress,
    TTSProviderDescriptor,
    TTSProviderSpec,
    TTSRequest,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import legacy_provider_specs
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
    registry_type: type[TTSAdapterRegistry] = TTSAdapterRegistry,
) -> TTSAdapterRegistry:
    replacements = FakeAdapterFactory(adapter.provider_id)
    calls = 0

    def factory(config: Mapping[str, Any]) -> FakeAdapter:
        nonlocal calls
        del config
        calls += 1
        return adapter if calls == 1 else replacements({})

    return registry_type(
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

    response = await service.synthesize(tts_request())
    await response.aclose()

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
async def test_caller_cancellation_precedes_later_resource_cleanup_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    secret = "SENSITIVE_CLEANUP_PAYLOAD_9f04e7"
    cleanup_error = RuntimeError(f"provider cleanup exposed {secret}")

    class BlockingFailingCloseAdapter(FakeAdapter):
        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await allow_close.wait()
            raise cleanup_error

    adapter = BlockingFailingCloseAdapter("openai")
    registry = registry_for_adapter(adapter)
    service = TTSService(registry, max_concurrent_operations=1)
    response = await service.synthesize(tts_request())
    await registry.reconfigure_provider("openai", {"revision": 2})
    caplog.set_level(logging.WARNING, logger="tldw_chatbook.TTS.TTS_Generation")
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
    cleanup_record = next(
        record
        for record in caplog.records
        if record.getMessage().startswith(
            "TTS cleanup failed while preserving an earlier error"
        )
    )
    assert "RuntimeError" in cleanup_record.getMessage()
    assert cleanup_record.exc_info is not None
    assert cleanup_record.exc_info[2] is cleanup_error.__traceback__
    assert secret not in caplog.text
    assert str(cleanup_error) not in caplog.text
    assert cleanup_record.exc_info[1] is not cleanup_error
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
async def test_safe_operation_error_survives_response_cleanup_failure() -> None:
    primary_error = TTSOperationError(
        code="generation_failed",
        message="Audio generation failed",
        retryable=False,
        operation_id="op-test",
    )

    class SafeStreamFailureAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            del progress_sink

            async def stream():
                raise primary_error
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

    adapter = SafeStreamFailureAdapter("openai")
    service = service_for_adapter(adapter)
    stream = service.generate_audio_stream(
        speech_request(),
        "openai_official_tts-1",
    )

    with pytest.raises(TTSOperationError) as error:
        await anext(stream)

    assert error.value is primary_error
    assert error.value.code == "generation_failed"
    assert str(error.value) == "Audio generation failed"
    assert error.value.__notes__ == [
        "TTS cleanup also failed while preserving the original error"
    ]
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
async def test_catalog_voice_and_reconfigure_delegate_to_registry() -> None:
    adapter = FakeAdapter("openai")
    service = service_for_adapter(adapter)

    catalog = await service.get_catalog("openai", refresh=True)
    voices = await service.get_voices("openai", "model", refresh=True)
    result = await service.reconfigure_provider(
        "openai",
        {"revision": 2},
    )

    assert catalog.provider_id == "openai"
    assert voices == ("default",)
    assert adapter.ensure_ready_calls == 2
    assert adapter.get_voices_requests == [("model", True)]
    assert result is ReconfigureResult.CHANGED
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_managed_response_preserves_immutable_metadata_and_lease() -> None:
    source = {"operation_id": "op-test", "generation_ms": 12.5}
    adapter = FakeAdapter("openai", response_metadata=source)
    registry = registry_for_adapter(adapter)
    service = TTSService(registry)

    response = await service.synthesize(tts_request())
    source["operation_id"] = "changed"
    await registry.reconfigure_provider("openai", {"revision": 2})

    assert response.metadata == {
        "operation_id": "op-test",
        "generation_ms": 12.5,
    }
    with pytest.raises(TypeError):
        response.metadata["operation_id"] = "changed"  # type: ignore[index]
    assert adapter.close_calls == 0

    await response.aclose()
    assert adapter.response_close_calls == 1
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_managed_response_rejects_nested_metadata_and_releases_resources() -> (
    None
):
    class NestedMetadataAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            response = await super().synthesize(request, progress_sink)
            response.metadata = {"unsafe": []}  # type: ignore[assignment]
            return response

    adapter = NestedMetadataAdapter("openai")
    registry = registry_for_adapter(adapter)
    service = TTSService(registry)
    managed_response: TTSAudioResponse | None = None

    try:
        with pytest.raises(
            TypeError,
            match="TTS audio response metadata values must be immutable scalars",
        ):
            managed_response = await service.synthesize(tts_request())
    finally:
        if managed_response is not None:
            await managed_response.aclose()

    assert adapter.response_close_calls == 1
    assert registry._total_leases() == 0


@pytest.mark.asyncio
async def test_legacy_voice_discovery_uses_static_catalog_without_manager() -> None:
    manager_calls = 0

    def manager_factory(_provider_id: str, _config: dict[str, Any]) -> Any:
        nonlocal manager_calls
        manager_calls += 1
        raise AssertionError("voice discovery must not materialize the legacy manager")

    service = TTSService(
        TTSAdapterRegistry(
            specs=legacy_provider_specs({}, manager_factory=manager_factory),
            aliases={},
        )
    )

    catalog = await service.get_catalog("openai")
    voices = await service.get_voices("openai", "tts-1", refresh=True)
    unknown = await service.get_voices("openai", "missing")

    assert voices == catalog.models[0].voices
    assert unknown == ()
    assert manager_calls == 0
    await service.close()
    await service.wait_closed()


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
        assert config == {}
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
async def test_service_shutdown_closes_abandoned_response_and_wakes_waiter() -> None:
    adapter = FakeAdapter("openai")
    service = TTSService(
        registry_for_adapter(adapter, shutdown_timeout_seconds=0),
        max_concurrent_operations=1,
    )
    abandoned = await service.synthesize(tts_request())
    blocked = asyncio.create_task(service.synthesize(tts_request()))
    await asyncio.sleep(0)
    assert blocked.done() is False

    try:
        await service.close()
        await asyncio.wait_for(service.wait_closed(), timeout=1)

        assert blocked.done()
        with pytest.raises(TTSRegistryClosedError):
            await asyncio.wait_for(blocked, timeout=0.1)
        assert adapter.response_close_calls == 1
        assert service._operation_limit._value == 1
    finally:
        if not blocked.done():
            blocked.cancel()
        await asyncio.gather(blocked, return_exceptions=True)
        await abandoned.aclose()


@pytest.mark.asyncio
async def test_service_close_releases_slot_when_admission_and_signal_both_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission_wait_started = asyncio.Event()
    original_wait = asyncio.wait
    control_admission_wait = True
    adapter = FakeAdapter("openai")

    async def wait_until_both_finish(
        futures: set[asyncio.Task[bool]],
        *,
        timeout: float | None = None,
        return_when: str = asyncio.ALL_COMPLETED,
    ):
        nonlocal control_admission_wait
        if control_admission_wait:
            control_admission_wait = False
            admission_wait_started.set()
            done, pending = await original_wait(
                futures,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
            assert len(done) == 2
            assert pending == set()
            return done, pending
        return await original_wait(
            futures,
            timeout=timeout,
            return_when=return_when,
        )

    monkeypatch.setattr(
        "tldw_chatbook.TTS.TTS_Generation.asyncio.wait",
        wait_until_both_finish,
    )
    service = TTSService(
        registry_for_adapter(adapter, shutdown_timeout_seconds=0),
        max_concurrent_operations=1,
    )
    generation = asyncio.create_task(service.synthesize(tts_request()))
    await admission_wait_started.wait()
    close = asyncio.create_task(service.close())

    with pytest.raises(TTSRegistryClosedError):
        await asyncio.wait_for(generation, timeout=1)
    await close
    await service.wait_closed()

    assert service._operation_limit._value == 1
    assert adapter.synthesize_calls == 0


@pytest.mark.asyncio
async def test_shutdown_joins_already_started_manual_response_close() -> None:
    response_close_started = asyncio.Event()
    allow_response_close = asyncio.Event()

    class BlockingResponseCloseAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            del progress_sink
            self.synthesize_calls += 1

            async def stream():
                yield b"audio"

            async def cleanup() -> None:
                self.response_close_calls += 1
                response_close_started.set()
                await allow_response_close.wait()

            return TTSAudioResponse(
                provider_id=self.provider_id,
                model_id=request.model_id,
                audio_format=request.response_format,
                content_type="audio/wav",
                byte_stream=stream(),
                cleanup=cleanup,
            )

    adapter = BlockingResponseCloseAdapter("openai")
    service = TTSService(
        registry_for_adapter(adapter, shutdown_timeout_seconds=0),
        max_concurrent_operations=1,
    )
    response = await service.synthesize(tts_request())
    manual_close = asyncio.create_task(response.aclose())
    await response_close_started.wait()
    wait_closed: asyncio.Task[None] | None = None
    try:
        assert response in service._responses

        await service.close()
        wait_closed = asyncio.create_task(service.wait_closed())
        await asyncio.wait({wait_closed}, timeout=0.2)

        assert response in service._responses
        assert manual_close.done() is False
        assert wait_closed.done()
        await wait_closed
        assert adapter.response_close_calls == 1
        assert service._operation_limit._value == 1
    finally:
        allow_response_close.set()
        tasks = [manual_close]
        if wait_closed is not None:
            tasks.append(wait_closed)
        await asyncio.gather(*tasks, return_exceptions=True)

    assert response not in service._responses
    assert adapter.response_close_calls == 1
    assert service._operation_limit._value == 1


@pytest.mark.asyncio
async def test_service_shutdown_attempts_all_response_cleanup_and_sanitizes_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "SENSITIVE_RESPONSE_CLEANUP_81d19c"
    cleanup_calls: list[int] = []

    class OneFailingResponseAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            del progress_sink
            self.synthesize_calls += 1
            response_number = self.synthesize_calls

            async def stream():
                yield b"audio"

            async def cleanup() -> None:
                cleanup_calls.append(response_number)
                self.response_close_calls += 1
                if response_number == 1:
                    raise RuntimeError(f"provider exposed {secret}")

            return TTSAudioResponse(
                provider_id=self.provider_id,
                model_id=request.model_id,
                audio_format=request.response_format,
                content_type="audio/wav",
                byte_stream=stream(),
                cleanup=cleanup,
            )

    adapter = OneFailingResponseAdapter("openai")
    service = TTSService(
        registry_for_adapter(adapter, shutdown_timeout_seconds=0),
        max_concurrent_operations=2,
    )
    first = await service.synthesize(tts_request())
    second = await service.synthesize(tts_request())
    caplog.set_level(logging.WARNING, logger="tldw_chatbook.TTS.TTS_Generation")

    try:
        await service.close()
        with pytest.raises(RuntimeError) as error:
            await service.wait_closed()

        assert sorted(cleanup_calls) == [1, 2]
        assert adapter.response_close_calls == 2
        assert service._operation_limit._value == 2
        assert service.registry._total_leases() == 0
        assert secret not in str(error.value)
        assert secret not in caplog.text
    finally:
        await asyncio.gather(
            first.aclose(),
            second.aclose(),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_cancelled_close_caller_waits_for_retained_bounded_close() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    adapter = FakeAdapter("openai")

    class DelayedCloseRegistry(TTSAdapterRegistry):
        async def close(self) -> None:
            close_started.set()
            await allow_close.wait()
            await super().close()

    service = TTSService(
        registry_for_adapter(
            adapter,
            shutdown_timeout_seconds=0,
            registry_type=DelayedCloseRegistry,
        )
    )
    first = asyncio.create_task(service.close())
    await close_started.wait()
    second = asyncio.create_task(service.close())
    first.cancel()
    await asyncio.sleep(0)

    assert first.done() is False
    assert second.done() is False

    allow_close.set()
    first_result, second_result = await asyncio.gather(
        first,
        second,
        return_exceptions=True,
    )
    assert isinstance(first_result, asyncio.CancelledError)
    assert second_result is None
    await service.wait_closed()


@pytest.mark.asyncio
async def test_cancelled_wait_closed_caller_joins_shared_terminal_shutdown() -> None:
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
    first = asyncio.create_task(service.wait_closed())
    second = asyncio.create_task(service.wait_closed())
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.sleep(0)

    assert first.done() is False
    assert second.done() is False

    allow_close.set()
    first_result, second_result = await asyncio.gather(
        first,
        second,
        return_exceptions=True,
    )
    assert isinstance(first_result, asyncio.CancelledError)
    assert second_result is None
    await service.close()
    await service.wait_closed()
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_in_flight_synthesis_cannot_escape_after_service_seals() -> None:
    synthesis_started = asyncio.Event()
    allow_synthesis = asyncio.Event()

    class BlockingSynthesisAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            synthesis_started.set()
            await allow_synthesis.wait()
            return await super().synthesize(request, progress_sink)

    adapter = BlockingSynthesisAdapter("openai")
    service = TTSService(
        registry_for_adapter(adapter, shutdown_timeout_seconds=0),
        max_concurrent_operations=1,
    )
    generation = asyncio.create_task(service.synthesize(tts_request()))
    await synthesis_started.wait()
    await service.close()
    allow_synthesis.set()

    try:
        with pytest.raises(TTSRegistryClosedError):
            await asyncio.wait_for(generation, timeout=1)
    finally:
        if generation.done() and not generation.cancelled():
            response = generation.exception()
            if response is None:
                await generation.result().aclose()
        await service.wait_closed()

    assert adapter.response_close_calls == 1
    assert service._operation_limit._value == 1


@pytest.mark.asyncio
async def test_post_seal_synthesis_observes_all_cleanup_task_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    synthesis_started = asyncio.Event()
    allow_synthesis = asyncio.Event()
    response_cleanup_attempted = asyncio.Event()
    resource_release_attempted = asyncio.Event()
    observations_complete = asyncio.Event()
    response_secret = "SENSITIVE_POST_SEAL_RESPONSE_f79e"
    resource_secret = "SENSITIVE_POST_SEAL_RESOURCE_36a1"
    observed_error_types: list[type[BaseException]] = []

    class FailingResponseAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            del progress_sink
            synthesis_started.set()
            await allow_synthesis.wait()

            async def stream() -> AsyncGenerator[bytes, None]:
                yield b"unreachable"

            async def cleanup() -> None:
                self.response_close_calls += 1
                response_cleanup_attempted.set()
                raise RuntimeError(f"provider exposed {response_secret}")

            return TTSAudioResponse(
                provider_id=self.provider_id,
                model_id=request.model_id,
                audio_format=request.response_format,
                content_type="audio/wav",
                byte_stream=stream(),
                cleanup=cleanup,
            )

    class FailingReleaseRegistry(TTSAdapterRegistry):
        async def _release(self, slot: Any, record: Any) -> None:
            await super()._release(slot, record)
            resource_release_attempted.set()
            raise RuntimeError(f"provider exposed {resource_secret}")

    class ObservingService(TTSService):
        @staticmethod
        def _observe_shutdown_result(task: asyncio.Task[None]) -> None:
            try:
                error = task.exception()
            except BaseException as error:
                observed_error_types.append(type(error))
            else:
                if error is not None:
                    observed_error_types.append(type(error))
            if len(observed_error_types) == 2:
                observations_complete.set()

    adapter = FailingResponseAdapter("openai")
    service = ObservingService(
        registry_for_adapter(
            adapter,
            shutdown_timeout_seconds=0,
            registry_type=FailingReleaseRegistry,
        ),
        max_concurrent_operations=1,
    )
    generation = asyncio.create_task(service.synthesize(tts_request()))
    await synthesis_started.wait()
    await service.close()
    await service.wait_closed()

    loop = asyncio.get_running_loop()
    previous_exception_handler = loop.get_exception_handler()
    unobserved_contexts: list[dict[str, Any]] = []
    loop.set_exception_handler(
        lambda _loop, context: unobserved_contexts.append(context)
    )
    caplog.set_level(logging.WARNING)
    try:
        allow_synthesis.set()
        with pytest.raises(TTSRegistryClosedError):
            await generation
        await response_cleanup_attempted.wait()
        await resource_release_attempted.wait()
        await asyncio.wait_for(observations_complete.wait(), timeout=1)
        await asyncio.sleep(0)

        assert observed_error_types == [RuntimeError, RuntimeError]
        assert unobserved_contexts == []
        assert adapter.response_close_calls == 1
        assert service._operation_limit._value == 1
        assert service.registry._closing_records[0].leases == 0
        assert service.registry._total_leases() == 0
        assert service._responses == set()
        assert response_secret not in caplog.text
        assert resource_secret not in caplog.text
        await service.wait_closed()
    finally:
        loop.set_exception_handler(previous_exception_handler)
        allow_synthesis.set()
        await asyncio.gather(
            generation,
            service.wait_closed(),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_service_shutdown_cancels_legacy_stream_after_drain_deadline() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingBackend:
        def set_progress_callback(self, callback: object) -> None:
            del callback

        async def generate_speech_stream(
            self,
            request: OpenAISpeechRequest,
        ) -> AsyncGenerator[bytes, None]:
            del request
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield b"unreachable"

    class Manager:
        def __init__(self) -> None:
            self.backend = BlockingBackend()
            self.close_calls = 0

        async def get_backend(self, internal_model_id: str) -> BlockingBackend:
            del internal_model_id
            return self.backend

        async def close_all_backends(self) -> None:
            self.close_calls += 1

    async def consume_legacy_stream(service: TTSService) -> bytes:
        return b"".join(
            [
                chunk
                async for chunk in service.generate_audio_stream(
                    speech_request(),
                    "local_kokoro_default_onnx",
                )
            ]
        )

    manager = Manager()
    specs = legacy_provider_specs(
        {},
        manager_factory=lambda _provider_id, _config: manager,
        shutdown_timeout_seconds=0.01,
    )
    service = TTSService(
        TTSAdapterRegistry(
            specs=specs,
            aliases={},
            shutdown_timeout_seconds=0.01,
        )
    )
    generation = asyncio.create_task(consume_legacy_stream(service))
    await started.wait()

    await service.close()
    try:
        await asyncio.wait_for(service.wait_closed(), timeout=0.2)
    finally:
        if not generation.done():
            generation.cancel()
        await asyncio.gather(generation, return_exceptions=True)
        await asyncio.wait_for(service.wait_closed(), timeout=0.2)

    assert cancelled.is_set()
    assert manager.close_calls == 1


@pytest.mark.asyncio
async def test_legacy_stream_cancellation_precedes_delegated_finalizer_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    started = asyncio.Event()
    secret = "SENSITIVE_CANCELLED_LEGACY_FINALIZER_b4d0a7"

    class FailingFinalizerBackend:
        def set_progress_callback(self, callback: object) -> None:
            del callback

        async def generate_speech_stream(
            self,
            request: OpenAISpeechRequest,
        ) -> AsyncGenerator[bytes, None]:
            del request
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                raise RuntimeError(f"provider exposed {secret}")
            yield b"unreachable"

    class Manager:
        def __init__(self) -> None:
            self.backend = FailingFinalizerBackend()
            self.close_calls = 0

        async def get_backend(self, internal_model_id: str) -> FailingFinalizerBackend:
            del internal_model_id
            return self.backend

        async def close_all_backends(self) -> None:
            self.close_calls += 1

    async def consume(service: TTSService) -> asyncio.CancelledError:
        try:
            async for _chunk in service.generate_audio_stream(
                speech_request(),
                "local_kokoro_default_onnx",
            ):
                pass
        except asyncio.CancelledError as error:
            return error
        raise AssertionError("legacy stream did not preserve cancellation")

    manager = Manager()
    service = TTSService(
        TTSAdapterRegistry(
            specs=legacy_provider_specs(
                {},
                manager_factory=lambda _provider_id, _config: manager,
                shutdown_timeout_seconds=0.01,
            ),
            aliases={},
            shutdown_timeout_seconds=0.01,
        ),
        max_concurrent_operations=1,
    )
    caplog.set_level(logging.WARNING)
    consumer = asyncio.create_task(consume(service))
    await started.wait()
    consumer.cancel()
    result = await consumer

    assert result.__context__ is None
    assert result.__cause__ is None
    assert any("cleanup" in note.lower() for note in result.__notes__)
    assert secret not in " ".join(result.__notes__)
    assert secret not in caplog.text
    assert service._operation_limit._value == 1
    assert service.registry._total_leases() == 0

    await service.close()
    await service.wait_closed()
    assert manager.close_calls == 1


@pytest.mark.asyncio
async def test_wait_closed_releases_resources_without_joining_stuck_response_finalizer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    started = asyncio.Event()
    finalizer_started = asyncio.Event()
    allow_finalizer = asyncio.Event()
    secret = "SENSITIVE_STUCK_FINALIZER_51b6ec"

    class UncooperativeBackend:
        def set_progress_callback(self, callback: object) -> None:
            del callback

        async def generate_speech_stream(
            self,
            request: OpenAISpeechRequest,
        ) -> AsyncGenerator[bytes, None]:
            del request
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finalizer_started.set()
                while not allow_finalizer.is_set():
                    try:
                        await allow_finalizer.wait()
                    except asyncio.CancelledError:
                        continue
                raise RuntimeError(f"provider exposed {secret}")
            yield b"unreachable"

    class Manager:
        def __init__(self) -> None:
            self.backend = UncooperativeBackend()
            self.close_calls = 0

        async def get_backend(self, internal_model_id: str) -> UncooperativeBackend:
            del internal_model_id
            return self.backend

        async def close_all_backends(self) -> None:
            self.close_calls += 1

    manager = Manager()
    service = TTSService(
        TTSAdapterRegistry(
            specs=legacy_provider_specs(
                {},
                manager_factory=lambda _provider_id, _config: manager,
                shutdown_timeout_seconds=0.01,
            ),
            aliases={},
            shutdown_timeout_seconds=0.01,
        ),
        max_concurrent_operations=1,
    )
    native_request = TTSRequest(
        provider_id="kokoro",
        model_id="tts-1",
        text="hello",
        voice="af_heart",
        response_format="mp3",
        options={
            "_legacy_openai_request": speech_request(),
            "_legacy_internal_model_id": "local_kokoro_default_onnx",
        },
    )
    response = await service.synthesize(native_request)
    drive_stream = asyncio.create_task(anext(response.byte_stream))
    await started.wait()
    caplog.set_level(logging.WARNING)

    await service.close()
    wait_closed = asyncio.create_task(service.wait_closed())
    try:
        await asyncio.wait({wait_closed}, timeout=0.5)
        assert wait_closed.done()
        with pytest.raises(RuntimeError) as error:
            await wait_closed

        assert finalizer_started.is_set()
        assert response in service._responses
        assert drive_stream.done() is False
        assert service._operation_limit._value == 1
        assert service.registry._total_leases() == 0
        assert secret not in str(error.value)
        assert secret not in caplog.text
    finally:
        allow_finalizer.set()
        await asyncio.gather(
            drive_stream,
            response.aclose(),
            wait_closed,
            return_exceptions=True,
        )

    assert response not in service._responses
    assert manager.close_calls == 1


@pytest.mark.asyncio
async def test_cooperative_legacy_close_finishes_with_registry_deadline_remaining() -> (
    None
):
    timeout = 0.1
    manager_close_started = asyncio.Event()
    manager_close_finished = asyncio.Event()

    class Backend:
        def set_progress_callback(self, callback: object) -> None:
            del callback

        async def generate_speech_stream(
            self,
            request: OpenAISpeechRequest,
        ) -> AsyncGenerator[bytes, None]:
            del request
            yield b"audio"

    class Manager:
        def __init__(self) -> None:
            self.backend = Backend()
            self.close_calls = 0

        async def get_backend(self, internal_model_id: str) -> Backend:
            del internal_model_id
            return self.backend

        async def close_all_backends(self) -> None:
            self.close_calls += 1
            manager_close_started.set()
            await asyncio.sleep(0.01)
            manager_close_finished.set()

    manager = Manager()
    service = TTSService(
        TTSAdapterRegistry(
            specs=legacy_provider_specs(
                {},
                manager_factory=lambda _provider_id, _config: manager,
                shutdown_timeout_seconds=timeout,
            ),
            aliases={},
            shutdown_timeout_seconds=timeout,
        )
    )
    native_request = TTSRequest(
        provider_id="kokoro",
        model_id="tts-1",
        text="hello",
        voice="af_heart",
        response_format="mp3",
        options={
            "_legacy_openai_request": speech_request(),
            "_legacy_internal_model_id": "local_kokoro_default_onnx",
        },
    )
    response = await service.synthesize(native_request)
    assert b"".join([chunk async for chunk in response.byte_stream]) == b"audio"

    loop = asyncio.get_running_loop()
    shutdown_started = loop.time()
    close = asyncio.create_task(service.close())
    await asyncio.sleep(0.02)
    await response.aclose()
    await close
    await service.wait_closed()

    assert loop.time() - shutdown_started < timeout
    assert manager_close_started.is_set()
    assert manager_close_finished.is_set()
    assert manager.close_calls == 1


@pytest.mark.asyncio
async def test_legacy_shutdown_uses_one_deadline_for_all_cleanup_phases(
    caplog: pytest.LogCaptureFixture,
) -> None:
    timeout = 0.05
    started = asyncio.Event()
    finalizer_started = asyncio.Event()
    allow_finalizer = asyncio.Event()
    finalizer_finished = asyncio.Event()
    manager_close_started = asyncio.Event()
    allow_manager_close = asyncio.Event()
    manager_close_finished = asyncio.Event()
    operation_secret = "SENSITIVE_DEADLINE_OPERATION_8d6f"
    manager_secret = "SENSITIVE_DEADLINE_MANAGER_c230"

    class UncooperativeBackend:
        def set_progress_callback(self, callback: object) -> None:
            del callback

        async def generate_speech_stream(
            self,
            request: OpenAISpeechRequest,
        ) -> AsyncGenerator[bytes, None]:
            del request
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finalizer_started.set()
                while not allow_finalizer.is_set():
                    try:
                        await allow_finalizer.wait()
                    except asyncio.CancelledError:
                        continue
                finalizer_finished.set()
                raise RuntimeError(f"provider exposed {operation_secret}")
            yield b"unreachable"

    class UncooperativeManager:
        def __init__(self) -> None:
            self.backend = UncooperativeBackend()
            self.close_calls = 0

        async def get_backend(
            self,
            internal_model_id: str,
        ) -> UncooperativeBackend:
            del internal_model_id
            return self.backend

        async def close_all_backends(self) -> None:
            self.close_calls += 1
            manager_close_started.set()
            while not allow_manager_close.is_set():
                try:
                    await allow_manager_close.wait()
                except asyncio.CancelledError:
                    continue
            manager_close_finished.set()
            raise RuntimeError(f"provider exposed {manager_secret}")

    manager = UncooperativeManager()
    service = TTSService(
        TTSAdapterRegistry(
            specs=legacy_provider_specs(
                {},
                manager_factory=lambda _provider_id, _config: manager,
                shutdown_timeout_seconds=timeout,
            ),
            aliases={},
            shutdown_timeout_seconds=timeout,
        ),
        max_concurrent_operations=1,
    )
    native_request = TTSRequest(
        provider_id="kokoro",
        model_id="tts-1",
        text="hello",
        voice="af_heart",
        response_format="mp3",
        options={
            "_legacy_openai_request": speech_request(),
            "_legacy_internal_model_id": "local_kokoro_default_onnx",
        },
    )
    response = await service.synthesize(native_request)
    drive_stream = asyncio.create_task(anext(response.byte_stream))
    await started.wait()
    caplog.set_level(logging.WARNING)

    loop = asyncio.get_running_loop()
    shutdown_started = loop.time()
    close_error: RuntimeError | None = None
    try:
        try:
            await service.close()
        except RuntimeError as error:
            close_error = error
        with pytest.raises(RuntimeError) as wait_error:
            await service.wait_closed()
        elapsed = loop.time() - shutdown_started

        assert elapsed < timeout * 3.5
        assert finalizer_started.is_set()
        assert manager_close_started.is_set()
        assert response in service._responses
        assert drive_stream.done() is False
        assert service._operation_limit._value == 1
        assert service.registry._total_leases() == 0
        assert operation_secret not in str(wait_error.value)
        assert manager_secret not in str(wait_error.value)
        if close_error is not None:
            assert operation_secret not in str(close_error)
            assert manager_secret not in str(close_error)
        assert operation_secret not in caplog.text
        assert manager_secret not in caplog.text
    finally:
        allow_finalizer.set()
        allow_manager_close.set()
        await asyncio.gather(
            drive_stream,
            response.aclose(),
            service.wait_closed(),
            return_exceptions=True,
        )

    assert finalizer_finished.is_set()
    assert manager_close_finished.is_set()
    assert response not in service._responses
    assert manager.close_calls == 1


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


@pytest.mark.parametrize("cleanup_error_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.asyncio
async def test_bound_shutdown_sanitizes_adapter_close_failure(
    cleanup_error_type: type[BaseException],
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "SENSITIVE_BOUND_SHUTDOWN_73b8e1"
    raw_error = cleanup_error_type(f"provider cleanup exposed {secret}")

    class FailingCloseAdapter(FakeAdapter):
        async def close(self) -> None:
            self.close_calls += 1
            raise raw_error

    adapter = FailingCloseAdapter("openai")
    service = TTSService(
        registry_for_adapter(adapter, shutdown_timeout_seconds=1),
    )
    response = await service.synthesize(tts_request())
    await response.aclose()
    bind_tts_service(service)
    caplog.set_level(logging.WARNING, logger="tldw_chatbook.TTS.TTS_Generation")

    with pytest.raises(RuntimeError) as error:
        await close_tts_resources()

    assert str(error.value) == (
        f"TTS shutdown cleanup failed ({cleanup_error_type.__name__})"
    )
    assert error.value.__context__ is None
    assert secret not in str(error.value)
    assert str(raw_error) not in str(error.value)
    assert secret not in caplog.text
    assert str(raw_error) not in caplog.text
    assert service._shutdown_task is not None
    assert service._shutdown_task.done()
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
