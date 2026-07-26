from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Mapping
from typing import Any

import pytest

from Tests.TTS.adapter_fakes import FakeAdapter
from tldw_chatbook.TTS.adapter_registry import (
    ReconfigureResult,
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
    TTSProviderReconfiguringError,
    TTSProviderSpec,
    TTSProviderUnavailableError,
    TTSRequest,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import (
    LegacyBackendHost,
    LegacyTTSAdapter,
    legacy_provider_specs,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS import TTS_Generation as generation_module
from tldw_chatbook.TTS.TTS_Generation import TTSService

_WAIT_SECONDS = 1.0


async def _wait_bounded(awaitable: Awaitable[Any]) -> Any:
    """Join one test synchronization point with a finite deadline."""
    protected = (
        asyncio.shield(awaitable)
        if isinstance(awaitable, asyncio.Future)
        else awaitable
    )
    return await asyncio.wait_for(
        protected,
        timeout=_WAIT_SECONDS,
    )


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

    async def _pause_admission(self, request: TTSRequest) -> None:
        if self._pause_next_admission:
            self._pause_next_admission = False
            self.frozen_requests.append(request)
            self.admission_started.set()
            await self.allow_admission.wait()

    async def admit(
        self,
        request: TTSRequest,
        *,
        expected_configuration_revision: int | None = None,
    ) -> Any:
        await self._pause_admission(request)
        return await super().admit(
            request,
            expected_configuration_revision=expected_configuration_revision,
        )

    async def _admit_reserved(
        self,
        request: TTSRequest,
        reservation: Any,
        *,
        expected_configuration_revision: int | None = None,
    ) -> Any:
        await self._pause_admission(request)
        return await super()._admit_reserved(
            request,
            reservation,
            expected_configuration_revision=expected_configuration_revision,
        )


class _CountingRecordingRegistry(_RecordingRegistry):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.release_calls = 0

    async def _release(self, slot: Any, record: Any) -> None:
        self.release_calls += 1
        await super()._release(slot, record)


def _counting_native_registry(
    adapter: _CapturingAdapter,
    *,
    shutdown_timeout_seconds: float = 10.0,
) -> _CountingRecordingRegistry:
    return _CountingRecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=lambda _config: adapter,
                initial_config={"generation": "initial"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
        shutdown_timeout_seconds=shutdown_timeout_seconds,
    )


class _GateExitPauseService(TTSService):
    def __init__(
        self,
        registry: TTSAdapterRegistry,
        snapshot: TTSPreferencesSnapshot,
    ) -> None:
        self.operation_admitted = asyncio.Event()
        self.allow_admit_return = asyncio.Event()
        self._admit_return_paused = False
        super().__init__(
            registry,
            max_concurrent_operations=1,
            preferences_snapshot=snapshot,
        )

    async def _pause_admit_return(self, operation: Any) -> Any:
        if self._admit_return_paused:
            return operation
        self._admit_return_paused = True
        self.operation_admitted.set()
        await self.allow_admit_return.wait()
        return operation

    async def admit(
        self,
        request: TTSRequest,
        *,
        expected_configuration_revision: int | None = None,
    ) -> Any:
        operation = await super().admit(
            request,
            expected_configuration_revision=expected_configuration_revision,
        )
        return await self._pause_admit_return(operation)

    async def _admit_reserved(
        self,
        request: TTSRequest,
        reservation: Any,
        *,
        expected_configuration_revision: int | None = None,
    ) -> Any:
        operation = await super()._admit_reserved(
            request,
            reservation,
            expected_configuration_revision=expected_configuration_revision,
        )
        return await self._pause_admit_return(operation)


@pytest.mark.asyncio
async def test_cancellation_during_read_gate_exit_closes_claimed_operation() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    registry = _counting_native_registry(adapter)
    service = _GateExitPauseService(registry, _snapshot())
    generation = asyncio.create_task(
        service.synthesize_default(text="Character response")
    )
    await asyncio.wait_for(service.operation_admitted.wait(), timeout=_WAIT_SECONDS)

    try:
        async with service._request_admission._gate._condition:
            service.allow_admit_return.set()
            await asyncio.sleep(0)
            assert not generation.done()
            generation.cancel("cancelled during read-gate exit")
            await asyncio.sleep(0)
            assert not generation.done()

        with pytest.raises(asyncio.CancelledError) as cancellation:
            await generation
        assert cancellation.value.args == ("cancelled during read-gate exit",)
        assert adapter.synthesize_calls == 0
        assert service._admitted_operations == set()
        assert service._operation_limit._value == 1
        assert registry._total_leases() == 0
        assert registry.release_calls == 1
    finally:
        if not generation.done():
            generation.cancel()
            await asyncio.gather(generation, return_exceptions=True)
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_shutdown_during_read_gate_exit_cannot_consume_claimed_operation() -> (
    None
):
    adapter = _CapturingAdapter("audio_cpp")
    registry = _counting_native_registry(
        adapter,
        shutdown_timeout_seconds=0,
    )
    service = _GateExitPauseService(registry, _snapshot())
    generation = asyncio.create_task(
        service.synthesize_default(text="Character response")
    )
    await asyncio.wait_for(service.operation_admitted.wait(), timeout=_WAIT_SECONDS)
    close_task: asyncio.Task[None] | None = None

    try:
        async with service._request_admission._gate._condition:
            service.allow_admit_return.set()
            await asyncio.sleep(0)
            assert not generation.done()
            close_task = asyncio.create_task(service.close())
            await asyncio.wait_for(
                service._close_signal.wait(),
                timeout=_WAIT_SECONDS,
            )
            await asyncio.sleep(0)

        with pytest.raises(TTSRegistryClosedError, match="service is closed"):
            await generation
        assert adapter.synthesize_calls == 0
        assert service._admitted_operations == set()
        assert service._operation_limit._value == 1
        assert registry._total_leases() == 0
        assert registry.release_calls == 1
    finally:
        if not generation.done():
            generation.cancel()
        tasks: list[asyncio.Future[Any]] = [generation]
        if close_task is not None:
            tasks.append(close_task)
        await asyncio.gather(*tasks, return_exceptions=True)
        await service.close()
        await service.wait_closed()


class _CapacityObservedService(TTSService):
    def __init__(
        self,
        registry: TTSAdapterRegistry,
        snapshot: TTSPreferencesSnapshot,
    ) -> None:
        self.capacity_acquisitions = 0
        self.second_capacity_wait_started = asyncio.Event()
        super().__init__(
            registry,
            max_concurrent_operations=1,
            preferences_snapshot=snapshot,
        )

    async def _acquire_operation_slot(self) -> None:
        self.capacity_acquisitions += 1
        if self.capacity_acquisitions == 2:
            self.second_capacity_wait_started.set()
        await super()._acquire_operation_slot()


@pytest.mark.asyncio
async def test_saturated_capacity_waiter_does_not_block_waiting_writer() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    registry = _counting_native_registry(adapter)
    service = _CapacityObservedService(registry, _snapshot())
    first_response = await service.synthesize_default(text="First response")
    second_generation = asyncio.create_task(
        service.synthesize_default(text="Second response")
    )
    await asyncio.wait_for(
        service.second_capacity_wait_started.wait(),
        timeout=_WAIT_SECONDS,
    )
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()

    async def publish() -> None:
        async with service._request_admission._gate.write():
            writer_entered.set()
            await release_writer.wait()

    publication = asyncio.create_task(publish())
    try:
        for _ in range(3):
            await asyncio.sleep(0)
        assert writer_entered.is_set()
        assert adapter.response_close_calls == 0
        assert not second_generation.done()
    finally:
        second_generation.cancel()
        release_writer.set()
        await first_response.aclose()
        await asyncio.gather(
            second_generation,
            publication,
            return_exceptions=True,
        )
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_cancelled_capacity_reservation_does_not_leak_or_overrelease() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    registry = _counting_native_registry(adapter)
    service = _CapacityObservedService(registry, _snapshot())
    first_response = await service.synthesize_default(text="First response")
    waiting_generation = asyncio.create_task(
        service.synthesize_default(text="Waiting response")
    )
    await asyncio.wait_for(
        service.second_capacity_wait_started.wait(),
        timeout=_WAIT_SECONDS,
    )

    waiting_generation.cancel("cancelled while reserving capacity")
    with pytest.raises(asyncio.CancelledError) as cancellation:
        await waiting_generation
    assert cancellation.value.args == ("cancelled while reserving capacity",)
    assert service._operation_limit._value == 0
    assert registry._total_leases() == 1

    await first_response.aclose()
    next_response = await service.synthesize_default(text="Next response")
    await next_response.aclose()

    assert service._operation_limit._value == 1
    assert registry._total_leases() == 0
    assert registry.release_calls == 2
    await service.close()
    await service.wait_closed()


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
async def test_settings_publication_times_out_without_cancelling_old_speech() -> None:
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
                initial_config={"generation": "one"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    old_snapshot = _snapshot(model_id="Old/Model")
    new_snapshot = _snapshot(model_id="New/Model")
    service = TTSService(registry, preferences_snapshot=old_snapshot)
    response = await service.synthesize_default(text="Generation one")

    outcome_type = getattr(
        generation_module,
        "TTSSettingsPersistenceOutcome",
        None,
    )
    assert outcome_type is not None, "settings persistence outcome is missing"
    ticket = service.begin_preferences_publication(
        new_snapshot,
        {"audio_cpp": {"generation": "two"}},
        lambda: outcome_type(True, True, None),
        foreground_timeout_seconds=0,
    )
    foreground = await asyncio.shield(ticket.foreground)

    assert foreground.generation == ticket.generation
    assert foreground.persistence.file_replaced is True
    assert foreground.provider_statuses == {"audio_cpp": "pending"}
    assert service.preferences_snapshot() == new_snapshot
    assert registry.configuration_revision("audio_cpp") == 1
    with pytest.raises(TTSProviderReconfiguringError):
        await registry.acquire("audio_cpp")
    assert len(adapters) == 1
    assert adapters[0].close_calls == 0

    chunks = [chunk async for chunk in response.byte_stream]
    assert chunks == [b"audio"]
    await response.aclose()
    completion = await asyncio.shield(ticket.completion)

    assert completion.provider_statuses == {"audio_cpp": "applied"}
    assert registry.configuration_revision("audio_cpp") == 2
    assert adapters[0].close_calls == 1
    assert len(adapters) == 1

    replacement = await service.synthesize_default(text="Generation two")
    assert adapters[1].generation == "two"
    assert adapters[1].requests[0].model_id == "New/Model"
    await replacement.aclose()
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_pre_replacement_failure_changes_no_preferences_or_provider() -> None:
    adapter = _CapturingAdapter("audio_cpp", generation="one")
    old_snapshot = _snapshot(model_id="Model/One")
    service, registry = _native_service(adapter, old_snapshot)
    lease = await registry.acquire("audio_cpp")
    await lease.release()
    outcome_type = generation_module.TTSSettingsPersistenceOutcome

    ticket = service.begin_preferences_publication(
        _snapshot(model_id="Model/Two"),
        {"audio_cpp": {"generation": "two"}},
        lambda: outcome_type(False, False, "before_replace"),
        foreground_timeout_seconds=0,
    )
    foreground = await asyncio.shield(ticket.foreground)
    completion = await asyncio.shield(ticket.completion)

    assert foreground == completion
    assert foreground.published is False
    assert foreground.provider_statuses == {"audio_cpp": "unchanged"}
    assert service.preferences_snapshot() == old_snapshot
    assert service.preferences_generation() == 0
    assert registry.configuration_revision("audio_cpp") == 1
    assert adapter.close_calls == 0

    response = await service.synthesize_default(text="Still generation one")
    assert adapter.requests[-1].model_id == "Model/One"
    await response.aclose()
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_first_publication_cannot_collide_with_compatibility_reconfigure() -> (
    None
):
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
                initial_config={"generation": "initial"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    old_snapshot = _snapshot(model_id="Model/Initial")
    saved_snapshot = _snapshot(model_id="Model/Saved")
    service = TTSService(registry, preferences_snapshot=old_snapshot)

    try:
        assert (
            await service.reconfigure_provider(
                "audio_cpp",
                {"generation": "compatibility"},
            )
            is ReconfigureResult.CHANGED
        )
        assert registry.configuration_revision("audio_cpp") == 2

        ticket = service.begin_preferences_publication(
            saved_snapshot,
            {"audio_cpp": {"generation": "saved"}},
            lambda: generation_module.TTSSettingsPersistenceOutcome(
                True,
                True,
                None,
            ),
            foreground_timeout_seconds=0,
        )
        foreground = await asyncio.shield(ticket.foreground)
        completion = await asyncio.shield(ticket.completion)

        assert foreground.provider_statuses == {"audio_cpp": "applied"}
        assert completion.provider_statuses == {"audio_cpp": "applied"}
        assert service.preferences_snapshot() == saved_snapshot
        assert service.preferences_generation() == ticket.generation
        assert registry.configuration_generation("audio_cpp") == ticket.generation
        assert registry.configuration_revision("audio_cpp") == 3

        response = await service.synthesize_default(text="Saved generation")
        assert adapters[0].generation == "saved"
        assert adapters[0].requests[0].model_id == "Model/Saved"
        await response.aclose()
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_compatibility_reconfigure_cannot_supersede_pending_publication() -> None:
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
                initial_config={"generation": "initial"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    saved_snapshot = _snapshot(model_id="Model/Saved")
    service = TTSService(
        registry,
        preferences_snapshot=_snapshot(model_id="Model/Initial"),
    )
    response = await service.synthesize_default(text="Active speech")

    try:
        publication = service.begin_preferences_publication(
            saved_snapshot,
            {"audio_cpp": {"generation": "saved"}},
            lambda: generation_module.TTSSettingsPersistenceOutcome(
                True,
                True,
                None,
            ),
            foreground_timeout_seconds=0,
        )
        foreground = await asyncio.shield(publication.foreground)
        assert foreground.provider_statuses == {"audio_cpp": "pending"}

        failed_publication = service.begin_preferences_publication(
            _snapshot(model_id="Model/Not-Replaced"),
            {},
            lambda: generation_module.TTSSettingsPersistenceOutcome(
                False,
                False,
                "before_replace",
            ),
            foreground_timeout_seconds=0,
        )
        failed_result = await asyncio.shield(failed_publication.completion)
        assert failed_result.published is False
        assert service.preferences_snapshot() == saved_snapshot

        compatibility = asyncio.create_task(
            service.reconfigure_provider(
                "audio_cpp",
                {"generation": "compatibility"},
            )
        )
        await asyncio.sleep(0)
        assert compatibility.done() is False

        await response.aclose()
        assert await compatibility is ReconfigureResult.CHANGED
        completion = await asyncio.shield(publication.completion)

        assert completion.provider_statuses == {"audio_cpp": "unavailable"}
        assert service.preferences_snapshot() == saved_snapshot
        with pytest.raises(TTSProviderUnavailableError):
            await registry.acquire("audio_cpp")
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_provider_id", ("missing", "audio.cpp"))
async def test_publication_rejects_noncanonical_preference_provider_synchronously(
    invalid_provider_id: str,
) -> None:
    adapter = _CapturingAdapter("audio_cpp", generation="initial")
    initial_snapshot = _snapshot(model_id="Model/Initial")
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=lambda _config: adapter,
                initial_config={"generation": "initial"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={"audio.cpp": "audio_cpp"},
    )
    service = TTSService(registry, preferences_snapshot=initial_snapshot)
    persistence_calls = 0

    def persistence() -> Any:
        nonlocal persistence_calls
        persistence_calls += 1
        return generation_module.TTSSettingsPersistenceOutcome(True, True, None)

    try:
        with pytest.raises(
            ValueError,
            match="preferences must use a canonical registered provider ID",
        ):
            service.begin_preferences_publication(
                _snapshot(
                    provider_id=invalid_provider_id,
                    model_id="Model/Invalid",
                ),
                {},
                persistence,
            )

        await asyncio.sleep(0)
        assert persistence_calls == 0
        assert service._settings_publication_tasks == set()
        assert service.preferences_snapshot() == initial_snapshot
        assert service.preferences_generation() == 0
        assert service._settings_generation == 0
        assert registry._generation_sequence == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_newer_settings_publication_supersedes_pending_handoff() -> None:
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
                initial_config={"generation": "one"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=_snapshot(model_id="Model/One"),
    )
    response = await service.synthesize_default(text="Generation one")
    outcome_type = generation_module.TTSSettingsPersistenceOutcome

    generation_two = service.begin_preferences_publication(
        _snapshot(model_id="Model/Two"),
        {"audio_cpp": {"generation": "two"}},
        lambda: outcome_type(True, True, None),
        foreground_timeout_seconds=0,
    )
    assert (await asyncio.shield(generation_two.foreground)).provider_statuses == {
        "audio_cpp": "pending"
    }
    generation_three = service.begin_preferences_publication(
        _snapshot(model_id="Model/Three"),
        {"audio_cpp": {"generation": "three"}},
        lambda: outcome_type(True, True, None),
        foreground_timeout_seconds=0,
    )
    assert (await asyncio.shield(generation_three.foreground)).provider_statuses == {
        "audio_cpp": "pending"
    }

    assert registry.configuration_revision("audio_cpp") == 1
    assert len(adapters) == 1
    await response.aclose()
    second, third = await asyncio.gather(
        asyncio.shield(generation_two.completion),
        asyncio.shield(generation_three.completion),
    )

    assert second.provider_statuses == {"audio_cpp": "superseded"}
    assert third.provider_statuses == {"audio_cpp": "applied"}
    assert service.preferences_snapshot().model_id == "Model/Three"
    assert service.preferences_generation() == generation_three.generation
    assert registry.configuration_revision("audio_cpp") == 2
    assert adapters[0].close_calls == 1
    assert len(adapters) == 1

    replacement = await service.synthesize_default(text="Generation three")
    assert adapters[1].generation == "three"
    await replacement.aclose()
    await service.close()
    await service.wait_closed()


class _OlderPublicationObserverFirstService(TTSService):
    """Force an older pending publication to classify its result first."""

    def __init__(
        self,
        registry: TTSAdapterRegistry,
        snapshot: TTSPreferencesSnapshot,
    ) -> None:
        self.older_generation: int | None = None
        self.newer_generation: int | None = None
        self.older_status_classified = asyncio.Event()
        super().__init__(registry, preferences_snapshot=snapshot)

    async def _reconfiguration_status(
        self,
        provider_id: str,
        ticket: Any,
    ) -> Any:
        if ticket.generation == self.newer_generation:
            await _wait_bounded(self.older_status_classified.wait())
        status = await super()._reconfiguration_status(provider_id, ticket)
        if ticket.generation == self.older_generation:
            self.older_status_classified.set()
        return status


class _ReconfigurationObservedRegistry(_RecordingRegistry):
    """Expose when a publication has registered its provider transition."""

    def __init__(self, **kwargs: Any) -> None:
        self.reconfiguration_begun = asyncio.Event()
        super().__init__(**kwargs)

    async def begin_reconfigure_provider(
        self,
        provider_id: str,
        config: Mapping[str, Any],
        *,
        generation: int | None = None,
    ) -> Any:
        ticket = await super().begin_reconfigure_provider(
            provider_id,
            config,
            generation=generation,
        )
        self.reconfiguration_begun.set()
        return ticket


@pytest.mark.asyncio
async def test_persisted_newer_handoff_supersedes_older_before_snapshot_publish() -> (
    None
):
    adapters: list[_CapturingAdapter] = []

    def factory(config: Mapping[str, Any]) -> _CapturingAdapter:
        adapter = _CapturingAdapter(
            "audio_cpp",
            generation=str(config["generation"]),
        )
        adapters.append(adapter)
        return adapter

    registry = _ReconfigurationObservedRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=factory,
                initial_config={"generation": "one"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    service = _OlderPublicationObserverFirstService(
        registry,
        _snapshot(model_id="Model/One"),
    )
    active_response = await service.synthesize_default(text="Generation one")
    outcome_type = generation_module.TTSSettingsPersistenceOutcome
    replacement: TTSAudioResponse | None = None

    try:
        older = service.begin_preferences_publication(
            _snapshot(model_id="Model/Two"),
            {"audio_cpp": {"generation": "two"}},
            lambda: outcome_type(True, True, None),
            foreground_timeout_seconds=0,
        )
        service.older_generation = older.generation
        assert (await _wait_bounded(older.foreground)).provider_statuses == {
            "audio_cpp": "pending"
        }
        registry.reconfiguration_begun.clear()

        newer = service.begin_preferences_publication(
            _snapshot(model_id="Model/Three"),
            {"audio_cpp": {"generation": "three"}},
            lambda: outcome_type(True, True, None),
            foreground_timeout_seconds=_WAIT_SECONDS,
        )
        service.newer_generation = newer.generation

        await _wait_bounded(registry.reconfiguration_begun.wait())
        assert registry._slots["audio_cpp"].pending_generation == newer.generation
        await _wait_bounded(active_response.aclose())

        older_result, newer_foreground, newer_result = await _wait_bounded(
            asyncio.gather(
                asyncio.shield(older.completion),
                asyncio.shield(newer.foreground),
                asyncio.shield(newer.completion),
            )
        )

        assert older_result.provider_statuses == {"audio_cpp": "superseded"}
        assert newer_foreground.provider_statuses == {"audio_cpp": "applied"}
        assert newer_result.provider_statuses == {"audio_cpp": "applied"}
        assert service.preferences_snapshot().model_id == "Model/Three"
        assert service.preferences_generation() == newer.generation
        assert registry.configuration_generation("audio_cpp") == newer.generation
        assert registry.configuration_revision("audio_cpp") == 2
        assert len(adapters) == 1
        assert adapters[0].generation == "one"
        assert adapters[0].close_calls == 1

        replacement = await _wait_bounded(
            service.synthesize_default(text="Generation three")
        )
        assert len(adapters) == 2
        assert adapters[0].close_calls == 1
        assert adapters[1].generation == "three"
        assert adapters[1].requests[0].model_id == "Model/Three"
        assert registry.expected_revisions[-1] == ("audio_cpp", 2)
    finally:
        await _wait_bounded(active_response.aclose())
        if replacement is not None:
            await _wait_bounded(replacement.aclose())
        await _wait_bounded(service.close())
        await _wait_bounded(service.wait_closed())

    assert [adapter.close_calls for adapter in adapters] == [1, 1]


@pytest.mark.asyncio
async def test_persistence_runs_off_loop_and_publications_remain_serialized() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    service, _registry = _native_service(
        adapter,
        _snapshot(model_id="Model/One"),
    )
    loop = asyncio.get_running_loop()
    first_started = asyncio.Event()
    release_first = threading.Event()
    second_started = asyncio.Event()
    release_second = threading.Event()
    second_publication_entered = asyncio.Event()
    outcome_type = generation_module.TTSSettingsPersistenceOutcome
    heartbeat_ticks = 0
    heartbeat_advanced = asyncio.Event()
    heartbeat_running = True
    publication_entries = 0
    publications: list[Any] = []
    original_run = service._run_preferences_publication

    async def observe_publication_entry(*args: Any, **kwargs: Any) -> Any:
        nonlocal publication_entries
        publication_entries += 1
        if publication_entries == 2:
            second_publication_entered.set()
        return await original_run(*args, **kwargs)

    service._run_preferences_publication = (  # type: ignore[method-assign]
        observe_publication_entry
    )

    def heartbeat() -> None:
        nonlocal heartbeat_ticks
        if not heartbeat_running:
            return
        heartbeat_ticks += 1
        heartbeat_advanced.set()
        loop.call_soon(heartbeat)

    def first_persistence() -> Any:
        loop.call_soon_threadsafe(first_started.set)
        release_first.wait()
        return outcome_type(True, True, None)

    def second_persistence() -> Any:
        loop.call_soon_threadsafe(second_started.set)
        release_second.wait()
        return outcome_type(True, True, None)

    loop.call_soon(heartbeat)
    try:
        first = service.begin_preferences_publication(
            _snapshot(model_id="Model/Two"),
            {},
            first_persistence,
            foreground_timeout_seconds=0,
        )
        publications.append(first)
        await _wait_bounded(first_started.wait())
        observed_ticks = heartbeat_ticks
        heartbeat_advanced.clear()
        await _wait_bounded(heartbeat_advanced.wait())
        assert heartbeat_ticks > observed_ticks

        async def wait_for_foreground() -> Any:
            return await asyncio.shield(first.foreground)

        initiating_waiter = asyncio.create_task(wait_for_foreground())
        initiating_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await _wait_bounded(initiating_waiter)
        assert first.completion.cancelled() is False

        second = service.begin_preferences_publication(
            _snapshot(model_id="Model/Three"),
            {},
            second_persistence,
            foreground_timeout_seconds=0,
        )
        publications.append(second)
        await _wait_bounded(second_publication_entered.wait())
        assert second_started.is_set() is False

        release_first.set()
        await _wait_bounded(first.completion)
        await _wait_bounded(second_started.wait())
        release_second.set()
        await _wait_bounded(second.completion)

        assert service.preferences_snapshot().model_id == "Model/Three"
    finally:
        heartbeat_running = False
        release_first.set()
        release_second.set()
        await _wait_bounded(
            asyncio.gather(
                *(publication.completion for publication in publications),
                return_exceptions=True,
            )
        )
        await _wait_bounded(service.close())
        await _wait_bounded(service.wait_closed())


@pytest.mark.asyncio
async def test_service_shutdown_joins_retained_settings_publication() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    service, _registry = _native_service(
        adapter,
        _snapshot(model_id="Model/One"),
    )
    loop = asyncio.get_running_loop()
    persistence_started = asyncio.Event()
    release_persistence = threading.Event()
    outcome_type = generation_module.TTSSettingsPersistenceOutcome
    ticket: Any = None
    wait_closed: asyncio.Task[None] | None = None

    def persistence() -> Any:
        loop.call_soon_threadsafe(persistence_started.set)
        release_persistence.wait()
        return outcome_type(True, True, None)

    try:
        ticket = service.begin_preferences_publication(
            _snapshot(model_id="Model/Two"),
            {},
            persistence,
            foreground_timeout_seconds=0,
        )
        await _wait_bounded(persistence_started.wait())

        await _wait_bounded(service.close())
        wait_closed_entered = asyncio.Event()

        async def join_service() -> None:
            wait_closed_entered.set()
            await service.wait_closed()

        wait_closed = asyncio.create_task(join_service())
        await _wait_bounded(wait_closed_entered.wait())

        assert wait_closed.done() is False
        assert ticket.completion.done() is False

        release_persistence.set()
        await _wait_bounded(ticket.completion)
        await _wait_bounded(wait_closed)
    finally:
        release_persistence.set()
        pending = [
            task
            for task in (
                ticket.completion if ticket is not None else None,
                wait_closed,
            )
            if task is not None
        ]
        if pending:
            await _wait_bounded(asyncio.gather(*pending, return_exceptions=True))
        if wait_closed is None:
            await _wait_bounded(service.wait_closed())


@pytest.mark.asyncio
async def test_failed_multi_provider_begin_seals_in_reverse_and_joins_started_work() -> (
    None
):
    events: list[str] = []
    alpha_started = asyncio.Event()
    allow_alpha = asyncio.Event()
    secret = "PRIVATE_TRANSITION_VALUE"

    class OrderedFailingRegistry(TTSAdapterRegistry):
        async def begin_reconfigure_provider(
            self,
            provider_id: str,
            config: Mapping[str, Any],
            *,
            generation: int | None = None,
        ) -> Any:
            events.append(f"begin-{provider_id}")
            if provider_id == "beta":
                if events[:2] != ["begin-alpha", "begin-beta"]:
                    raise RuntimeError("providers started out of canonical order")
                await _wait_bounded(alpha_started.wait())
                raise RuntimeError(secret)
            return await super().begin_reconfigure_provider(
                provider_id,
                config,
                generation=generation,
            )

        async def _reconfigure_retiring(
            self,
            slot: Any,
            new_config: dict[str, Any],
        ) -> Any:
            alpha_started.set()
            await _wait_bounded(allow_alpha.wait())
            return await super()._reconfigure_retiring(slot, new_config)

        async def seal_provider_unavailable(self, provider_id: str) -> None:
            events.append(f"seal-{provider_id}")
            await super().seal_provider_unavailable(provider_id)

    registry = OrderedFailingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor("alpha", "alpha", True),
                factory=lambda _config: _CapturingAdapter("alpha"),
                initial_config={"generation": "one"},
            ),
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor("beta", "beta", True),
                factory=lambda _config: _CapturingAdapter("beta"),
                initial_config={"generation": "one"},
            ),
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=_snapshot(
            provider_id="alpha",
            model_id="Model/One",
            voice_mode="exact",
            voice_id="voice",
        ),
    )
    outcome_type = generation_module.TTSSettingsPersistenceOutcome
    publication: Any = None
    try:
        publication = service.begin_preferences_publication(
            _snapshot(
                provider_id="alpha",
                model_id="Model/Two",
                voice_mode="exact",
                voice_id="voice",
            ),
            {
                "beta": {"generation": "two"},
                "alpha": {"generation": "two"},
            },
            lambda: outcome_type(True, True, None),
            foreground_timeout_seconds=0,
        )
        foreground = await _wait_bounded(publication.foreground)

        assert foreground.provider_statuses == {
            "alpha": "unavailable",
            "beta": "unavailable",
        }
        assert events[:4] == [
            "begin-alpha",
            "begin-beta",
            "seal-beta",
            "seal-alpha",
        ]
        assert publication.completion.done() is False

        allow_alpha.set()
        completion = await _wait_bounded(publication.completion)
        assert completion.provider_statuses == foreground.provider_statuses
        with pytest.raises(TTSProviderUnavailableError):
            await _wait_bounded(registry.acquire("alpha"))
        assert secret not in repr(completion)
    finally:
        allow_alpha.set()
        if publication is not None:
            await _wait_bounded(
                asyncio.gather(
                    publication.completion,
                    return_exceptions=True,
                )
            )
        await _wait_bounded(service.close())
        await _wait_bounded(service.wait_closed())


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
