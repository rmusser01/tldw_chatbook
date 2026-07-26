from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from Tests.TTS.adapter_fakes import FakeAdapter
from tldw_chatbook.TTS.adapter_registry import (
    TTSAdapterLease,
    TTSAdapterRegistry,
)
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
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import (
    LegacyBackendHost,
    LegacyTTSAdapter,
    legacy_provider_specs,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import TTSService

_WAIT_SECONDS = 1.0


def _admission_api() -> tuple[type[Any], type[Any]]:
    try:
        from tldw_chatbook.TTS.request_admission import (
            TTSRequestAdmissionCoordinator,
            _WriterPreferredGate,
        )
    except ModuleNotFoundError:
        pytest.fail("the TTS request-admission module is not implemented")
    return _WriterPreferredGate, TTSRequestAdmissionCoordinator


async def _hold_gate(
    context: Any,
    entered: asyncio.Event,
    release: asyncio.Event,
    order: list[str] | None = None,
    label: str = "",
) -> None:
    async with context:
        if order is not None:
            order.append(label)
        entered.set()
        await release.wait()


@pytest.mark.asyncio
async def test_gate_allows_concurrent_readers() -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    release = asyncio.Event()
    entered = (asyncio.Event(), asyncio.Event())
    readers = tuple(
        asyncio.create_task(_hold_gate(gate.read(), event, release))
        for event in entered
    )

    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered)),
            timeout=_WAIT_SECONDS,
        )
    finally:
        release.set()
        await asyncio.gather(*readers, return_exceptions=True)


@pytest.mark.asyncio
async def test_waiting_writer_blocks_later_readers() -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    first_reader_entered = asyncio.Event()
    release_first_reader = asyncio.Event()
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()
    late_reader_entered = asyncio.Event()
    release_late_reader = asyncio.Event()
    order: list[str] = []

    first_reader = asyncio.create_task(
        _hold_gate(
            gate.read(),
            first_reader_entered,
            release_first_reader,
            order,
            "first-reader",
        )
    )
    await asyncio.wait_for(first_reader_entered.wait(), timeout=_WAIT_SECONDS)
    writer = asyncio.create_task(
        _hold_gate(
            gate.write(),
            writer_entered,
            release_writer,
            order,
            "writer",
        )
    )
    await asyncio.sleep(0)
    late_reader = asyncio.create_task(
        _hold_gate(
            gate.read(),
            late_reader_entered,
            release_late_reader,
            order,
            "late-reader",
        )
    )

    try:
        await asyncio.sleep(0)
        assert not late_reader_entered.is_set()
        release_first_reader.set()
        await asyncio.wait_for(writer_entered.wait(), timeout=_WAIT_SECONDS)
        assert not late_reader_entered.is_set()
        release_writer.set()
        await asyncio.wait_for(late_reader_entered.wait(), timeout=_WAIT_SECONDS)
        assert order == ["first-reader", "writer", "late-reader"]
    finally:
        release_first_reader.set()
        release_writer.set()
        release_late_reader.set()
        await asyncio.gather(
            first_reader,
            writer,
            late_reader,
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_cancelling_waiting_writer_unblocks_later_readers() -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    first_reader_entered = asyncio.Event()
    release_first_reader = asyncio.Event()
    writer_started = asyncio.Event()

    async def wait_for_write() -> None:
        writer_started.set()
        async with gate.write():
            pytest.fail("cancelled writer must not enter")

    first_reader = asyncio.create_task(
        _hold_gate(
            gate.read(),
            first_reader_entered,
            release_first_reader,
        )
    )
    await asyncio.wait_for(first_reader_entered.wait(), timeout=_WAIT_SECONDS)
    writer = asyncio.create_task(wait_for_write())
    await asyncio.wait_for(writer_started.wait(), timeout=_WAIT_SECONDS)
    await asyncio.sleep(0)
    writer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer

    later_reader_entered = asyncio.Event()
    release_later_reader = asyncio.Event()
    later_reader = asyncio.create_task(
        _hold_gate(
            gate.read(),
            later_reader_entered,
            release_later_reader,
        )
    )
    try:
        await asyncio.wait_for(later_reader_entered.wait(), timeout=_WAIT_SECONDS)
    finally:
        release_first_reader.set()
        release_later_reader.set()
        await asyncio.gather(first_reader, later_reader, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_waiting_reader_leaves_gate_usable() -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()
    writer = asyncio.create_task(
        _hold_gate(gate.write(), writer_entered, release_writer)
    )
    await asyncio.wait_for(writer_entered.wait(), timeout=_WAIT_SECONDS)

    async def wait_for_read() -> None:
        async with gate.read():
            pytest.fail("cancelled reader must not enter")

    reader = asyncio.create_task(wait_for_read())
    await asyncio.sleep(0)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader

    release_writer.set()
    await writer
    next_writer_entered = asyncio.Event()
    release_next_writer = asyncio.Event()
    next_writer = asyncio.create_task(
        _hold_gate(gate.write(), next_writer_entered, release_next_writer)
    )
    try:
        await asyncio.wait_for(next_writer_entered.wait(), timeout=_WAIT_SECONDS)
    finally:
        release_next_writer.set()
        await asyncio.gather(next_writer, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancelled_mode", "next_mode"), (("read", "write"), ("write", "read"))
)
async def test_cancelling_entered_context_releases_exactly_once(
    cancelled_mode: str,
    next_mode: str,
) -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    entered = asyncio.Event()
    never_release = asyncio.Event()
    cancelled_context = getattr(gate, cancelled_mode)()
    holder = asyncio.create_task(_hold_gate(cancelled_context, entered, never_release))
    await asyncio.wait_for(entered.wait(), timeout=_WAIT_SECONDS)

    holder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder

    next_entered = asyncio.Event()
    release_next = asyncio.Event()
    next_holder = asyncio.create_task(
        _hold_gate(getattr(gate, next_mode)(), next_entered, release_next)
    )
    try:
        await asyncio.wait_for(next_entered.wait(), timeout=_WAIT_SECONDS)
    finally:
        release_next.set()
        await asyncio.gather(next_holder, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancelled_mode", "next_mode"),
    (("read", "write"), ("write", "read")),
)
async def test_repeated_cancellation_cannot_interrupt_context_release(
    cancelled_mode: str,
    next_mode: str,
) -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    entered = asyncio.Event()
    never_release = asyncio.Event()
    holder = asyncio.create_task(
        _hold_gate(
            getattr(gate, cancelled_mode)(),
            entered,
            never_release,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=_WAIT_SECONDS)

    async with gate._condition:
        holder.cancel()
        await asyncio.sleep(0)
        assert not holder.done()
        holder.cancel()
        await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await holder

    next_entered = asyncio.Event()
    release_next = asyncio.Event()
    next_holder = asyncio.create_task(
        _hold_gate(getattr(gate, next_mode)(), next_entered, release_next)
    )
    try:
        await asyncio.wait_for(next_entered.wait(), timeout=_WAIT_SECONDS)
    finally:
        release_next.set()
        next_holder.cancel()
        await asyncio.gather(next_holder, return_exceptions=True)


class _RecordingRegistry(TTSAdapterRegistry):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.expected_revisions: list[tuple[str, int | None]] = []

    async def acquire(
        self,
        provider_id: str,
        *,
        expected_revision: int | None = None,
    ) -> TTSAdapterLease:
        self.expected_revisions.append((provider_id, expected_revision))
        return await super().acquire(
            provider_id,
            expected_revision=expected_revision,
        )


class _CapturingAdapter(FakeAdapter):
    def __init__(
        self,
        provider_id: str,
        *,
        models: tuple[TTSModelInfo, ...] = (),
        synthesis_error: BaseException | None = None,
        generation: str = "",
    ) -> None:
        super().__init__(provider_id)
        self.models = models
        self.synthesis_error = synthesis_error
        self.generation = generation
        self.catalog_calls = 0
        self.requests: list[TTSRequest] = []

    async def get_catalog(self, refresh: bool = False) -> TTSProviderCatalog:
        assert refresh is False
        self.catalog_calls += 1
        return TTSProviderCatalog(
            provider_id=self.provider_id,
            revision=self.catalog_calls,
            health=ProviderHealth(state="available", fresh=True),
            models=self.models,
        )

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        self.requests.append(request)
        if self.synthesis_error is not None:
            raise self.synthesis_error
        return await super().synthesize(request, progress_sink)


def _model(model_id: str) -> TTSModelInfo:
    return TTSModelInfo(
        model_id=model_id,
        display_name=model_id,
        family="tts",
        upstream_mode="offline",
        formats=("wav",),
        voices=(),
        supports_speed=False,
        omit_voice_uses_server_default=True,
    )


def _snapshot(
    *,
    provider_id: str = "audio_cpp",
    model_mode: str = "exact",
    model_id: str | None = "Model/Case-Sensitive",
    voice_mode: str = "server_default",
    voice_id: str | None = None,
    response_format: str = "wav",
    speed: float = 1.0,
) -> TTSPreferencesSnapshot:
    return TTSPreferencesSnapshot(
        provider_id=provider_id,
        model_mode=model_mode,  # type: ignore[arg-type]
        model_id=model_id,
        voice_mode=voice_mode,  # type: ignore[arg-type]
        voice_id=voice_id,
        response_format=response_format,
        speed=speed,
    )


def _native_service(
    adapter: _CapturingAdapter,
    snapshot: TTSPreferencesSnapshot,
) -> tuple[TTSService, _RecordingRegistry]:
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id=adapter.provider_id,
                    display_name=adapter.provider_id,
                    native=True,
                ),
                factory=lambda _config: adapter,
                initial_config={"generation": adapter.generation or "initial"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    return TTSService(registry, preferences_snapshot=snapshot), registry


@pytest.mark.asyncio
async def test_audio_cpp_exact_default_is_admitted_without_rewriting_values() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    snapshot = _snapshot(
        model_id="Model/Byte-For-Byte",
        voice_mode="exact",
        voice_id="Voice/Byte-For-Byte",
    )
    service, registry = _native_service(adapter, snapshot)

    response = await service.synthesize_default(text="Character response")
    try:
        assert adapter.requests == [
            TTSRequest(
                provider_id="audio_cpp",
                model_id="Model/Byte-For-Byte",
                text="Character response",
                voice="Voice/Byte-For-Byte",
                response_format="wav",
                speed=1.0,
                options={},
            )
        ]
        assert registry.expected_revisions == [("audio_cpp", 1)]
    finally:
        await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_audio_cpp_first_available_is_resolved_once_and_never_falls_back() -> (
    None
):
    failure = RuntimeError("synthetic generation failure")
    adapter = _CapturingAdapter(
        "audio_cpp",
        models=(_model("First/Model"), _model("Second/Model")),
        synthesis_error=failure,
    )
    service, registry = _native_service(
        adapter,
        _snapshot(model_mode="first_available", model_id=None),
    )

    try:
        with pytest.raises(RuntimeError) as raised:
            await service.synthesize_default(text="Character response")
        assert raised.value is failure
        assert adapter.catalog_calls == 1
        assert [request.model_id for request in adapter.requests] == ["First/Model"]
        assert registry.expected_revisions == [
            ("audio_cpp", None),
            ("audio_cpp", 1),
        ]
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_audio_cpp_server_default_omits_voice_and_uses_locked_options() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    service, _registry = _native_service(
        adapter,
        _snapshot(voice_mode="server_default", voice_id=None),
    )

    response = await service.synthesize_default(
        text="Character response",
        voice_override=None,
    )
    try:
        request = adapter.requests[0]
        assert request.voice is None
        assert request.response_format == "wav"
        assert request.speed == 1.0
        assert request.options == {}
    finally:
        await response.aclose()
        await service.close()
        await service.wait_closed()


class _PauseOnceService(TTSService):
    def __init__(
        self,
        registry: TTSAdapterRegistry,
        snapshot: TTSPreferencesSnapshot,
    ) -> None:
        self.admission_started = asyncio.Event()
        self.allow_admission = asyncio.Event()
        self.frozen_requests: list[TTSRequest] = []
        self._pause_next_admission = True
        super().__init__(registry, preferences_snapshot=snapshot)

    async def admit(
        self,
        request: TTSRequest,
        *,
        expected_configuration_revision: int | None = None,
    ) -> Any:
        if self._pause_next_admission:
            self._pause_next_admission = False
            self.frozen_requests.append(request)
            self.admission_started.set()
            await self.allow_admission.wait()
        return await super().admit(
            request,
            expected_configuration_revision=expected_configuration_revision,
        )


@pytest.mark.asyncio
async def test_default_request_never_mixes_preference_and_adapter_generations() -> None:
    adapters: list[_CapturingAdapter] = []

    def factory(config: Mapping[str, Any]) -> _CapturingAdapter:
        adapter = _CapturingAdapter(
            "audio_cpp",
            generation=str(config["generation"]),
        )
        adapters.append(adapter)
        return adapter

    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=factory,
                initial_config={"generation": "old"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    old_snapshot = _snapshot(model_id="Old/Model")
    new_snapshot = _snapshot(model_id="New/Model")
    service = _PauseOnceService(registry, old_snapshot)
    coordinator = service._request_admission
    request_task = asyncio.create_task(
        service.synthesize_default(text="Character response")
    )
    await asyncio.wait_for(service.admission_started.wait(), timeout=_WAIT_SECONDS)

    async def publish_new_generation() -> None:
        async with coordinator._gate.write():
            await service.reconfigure_provider(
                "audio_cpp",
                {"generation": "new"},
            )
            coordinator._preferences = new_snapshot

    publication = asyncio.create_task(publish_new_generation())
    await asyncio.sleep(0)
    service.allow_admission.set()
    first_response = await asyncio.wait_for(request_task, timeout=_WAIT_SECONDS)

    try:
        assert adapters[0].generation == "old"
        assert [request.model_id for request in adapters[0].requests] == ["Old/Model"]
        assert registry.expected_revisions == [("audio_cpp", 1)]
        assert not publication.done()
    finally:
        await first_response.aclose()

    await asyncio.wait_for(publication, timeout=_WAIT_SECONDS)
    assert service.preferences_snapshot() == new_snapshot
    second_response = await service.synthesize_default(text="Next response")
    try:
        assert [
            (adapter.generation, adapter.requests[0].model_id) for adapter in adapters
        ] == [
            ("old", "Old/Model"),
            ("new", "New/Model"),
        ]
        assert registry.expected_revisions == [
            ("audio_cpp", 1),
            ("audio_cpp", 2),
        ]
    finally:
        await second_response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "provider_id",
        "configured_model",
        "configured_format",
        "expected_model",
        "expected_format",
        "expected_internal_id",
    ),
    (
        (
            "openai",
            "tts-1-hd",
            "opus",
            "tts-1-hd",
            "opus",
            "openai_official_tts-1-hd",
        ),
        (
            "elevenlabs",
            "eleven_multilingual_v2",
            "wav",
            "elevenlabs",
            "mp3",
            "elevenlabs_elevenlabs",
        ),
        (
            "kokoro",
            "kokoro",
            "mp3",
            "kokoro",
            "wav",
            "local_kokoro_default_onnx",
        ),
        (
            "chatterbox",
            "chatterbox",
            "mp3",
            "chatterbox",
            "wav",
            "local_chatterbox_default",
        ),
        (
            "higgs",
            "higgs-audio-v2",
            "wav",
            "higgs-audio-v2",
            "wav",
            "local_higgs_v2",
        ),
        (
            "alltalk",
            "alltalk",
            "mp3",
            "alltalk",
            "wav",
            "alltalk_default",
        ),
    ),
)
async def test_retained_provider_defaults_admit_the_legacy_bridge(
    monkeypatch: pytest.MonkeyPatch,
    provider_id: str,
    configured_model: str,
    configured_format: str,
    expected_model: str,
    expected_format: str,
    expected_internal_id: str,
) -> None:
    captured: list[tuple[str, OpenAISpeechRequest]] = []

    async def audio() -> AsyncIterator[bytes]:
        yield b"audio"

    def capture_generate(
        _host: LegacyBackendHost,
        internal_model_id: str,
        request: OpenAISpeechRequest,
        _progress_sink: ProgressSink | None,
    ) -> AsyncIterator[bytes]:
        captured.append((internal_model_id, request))
        return audio()

    monkeypatch.setattr(LegacyBackendHost, "generate", capture_generate)
    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs(
            {},
            manager_factory=lambda _provider, _config: pytest.fail(
                "request admission must stop at the legacy adapter boundary"
            ),
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=_snapshot(
            provider_id=provider_id,
            model_id=configured_model,
            voice_mode="exact",
            voice_id="Voice/Case",
            response_format=configured_format,
        ),
    )

    response = await service.synthesize_default(text="Character response")
    try:
        active = registry._slots[provider_id].active
        assert active is not None
        assert isinstance(active.adapter, LegacyTTSAdapter)
        assert captured == [
            (
                expected_internal_id,
                OpenAISpeechRequest(
                    model=expected_model,
                    input="Character response",
                    voice="voice/case",
                    response_format=expected_format,
                    speed=1.0,
                ),
            )
        ]
        assert all(
            slot.active is None
            for candidate_id, slot in registry._slots.items()
            if candidate_id != provider_id
        )
    finally:
        await response.aclose()
        await service.close()
        await service.wait_closed()
