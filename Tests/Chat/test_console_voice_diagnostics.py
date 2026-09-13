"""Content-free durable diagnostics for the speculative voice owner loop."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from Tests.Audio import test_native_duplex_stream as native_helpers
from Tests.Audio.fakes.native_duplex_helpers import emit_callback
from Tests.integration.test_speculative_voice_pipeline import (
    _AudioStream,
    _AttemptGateway,
    _CoordinatorEffects,
    _Transport,
    _frame,
    _prepared_attempt,
)
from tldw_chatbook.Audio.duplex_contracts import (
    DuplexMode,
    RenderBoundary,
    RenderSubmission,
    RouteKind,
)
from tldw_chatbook.Audio.voice_preprocessor import VoicePreprocessor
from tldw_chatbook.Chat.console_speculative_voice_session import (
    ConsoleSpeculativeVoiceSession,
    SpeculativeVoiceAttemptEffects,
    _AttemptLifecycle,
    _NativeStreamingStt,
    _RollingWindowStt,
    _native_fault_snapshot,
)
from tldw_chatbook.Chat.console_speculative_voice import AttemptDispatchPrepared
from tldw_chatbook.Chat.console_voice_attempts import VoiceAttemptDelta
from tldw_chatbook.Chat.console_voice_supervisor import VoiceDispatchSupervisor
from tldw_chatbook.Chat.console_voice_controls import ControlKind
from tldw_chatbook.Logging_Config import PrivateRotatingFileHandler
from tldw_chatbook.TTS.adapter_types import TTSAudioResponse
from tldw_chatbook.Utils.persistent_diagnostics import PersistentDiagnosticFilter


driver = native_helpers.driver
native_transport = native_helpers.native_transport
portaudio = native_helpers.portaudio


class _UnusedTranscript:
    async def close(self) -> None:
        return None


def _private_sink(path: Path) -> PrivateRotatingFileHandler:
    handler = PrivateRotatingFileHandler(
        path,
        maxBytes=320,
        backupCount=8,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    handler.addFilter(PersistentDiagnosticFilter())
    return handler


def _all_generations(path: Path) -> str:
    return "\n".join(
        candidate.read_text(encoding="utf-8")
        for candidate in sorted(path.parent.glob(f"{path.name}*"))
        if candidate.is_file()
    )


def _fault_records(persisted: str) -> list[str]:
    return [
        line for line in persisted.splitlines() if "event=audio_transport_fault" in line
    ]


class _Clock:
    def __init__(self, now_ns: int = 1_000_000_000) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance_ms(self, milliseconds: int) -> None:
        self.now_ns += milliseconds * 1_000_000


def _first_stage_effects(clock: _Clock) -> SpeculativeVoiceAttemptEffects:
    return SpeculativeVoiceAttemptEffects(
        submit_event=lambda _event: asyncio.sleep(0),
        prepare_attempt=lambda **_kwargs: asyncio.sleep(0),
        gateway=None,
        synthesizer=None,
        transport=SimpleNamespace(fence_output=lambda: None),
        promotion=lambda **_kwargs: None,
        promotion_owner=object(),
        dispatch_supervisor=VoiceDispatchSupervisor(),
        project_preview=lambda _preview: None,
        clear_preview=lambda: None,
        submit_accepted_voice_turn=lambda *_args: None,
        clock=clock,
    )


def test_first_stage_timings_are_attempt_local_deduplicated_and_content_free(
    monkeypatch,
) -> None:
    from tldw_chatbook.Chat import console_speculative_voice_session as module

    private = "VOICE-PRIVATE-SENTINEL-sk-not-a-real-key"
    records: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        module,
        "_persist_voice_event",
        lambda event, **fields: records.append((event, fields)),
    )
    clock = _Clock()
    effects = _first_stage_effects(clock)
    first = RenderSubmission(3, 4, 5)
    effects._attempts[1] = _AttemptLifecycle(
        "turn-private",
        1,
        private,
        started_ns=clock(),
        first_render_submission=first,
    )

    clock.advance_ms(10)
    effects._on_delta(VoiceAttemptDelta(1, ""))
    clock.advance_ms(110)
    effects._on_delta(VoiceAttemptDelta(1, private))
    effects._on_delta(VoiceAttemptDelta(1, "duplicate"))
    clock.advance_ms(230)
    effects._on_first_eligible_phrase(1)
    effects._on_first_eligible_phrase(1)
    clock.advance_ms(450)
    effects._on_first_synthesis_complete(1)
    effects._on_first_synthesis_complete(1)
    clock.advance_ms(100)
    # A device may schedule output into the future relative to owner-loop receipt
    # observation. Keep the mapped DAC latency even when it exceeds duration.
    for mismatch in (
        RenderBoundary(4, 4, 5, clock() + 110_000_000),
        RenderBoundary(3, 5, 5, clock() + 110_000_000),
        RenderBoundary(3, 4, 6, clock() + 110_000_000),
    ):
        effects.observe_render_receipt(1, mismatch)
    effects.observe_render_receipt(
        1,
        RenderBoundary(3, 4, 5, clock() + 110_000_000),
    )
    effects.observe_render_receipt(
        1,
        RenderBoundary(3, 4, 5, clock() + 110_000_000),
    )

    stages = [(fields["phase"], fields) for event, fields in records]
    assert [stage for stage, _fields in stages] == [
        "provider_delta",
        "eligible_phrase",
        "synthesis_complete",
        "render_receipt",
    ]
    assert [fields["duration_ms"] for _stage, fields in stages] == [120, 350, 800, 900]
    assert stages[-1][1]["latency_ms"] == 1_000
    assert all(event == "voice_first_stage" for event, _fields in records)
    assert private not in repr(records)

    effects.fence_attempt(1)
    clock.advance_ms(10)
    effects._on_delta(VoiceAttemptDelta(1, "late success"))
    effects._on_first_eligible_phrase(1)
    effects._on_first_synthesis_complete(1)
    effects.observe_render_receipt(1, RenderBoundary(3, 4, 5, clock()))
    assert len(records) == 4


@pytest.mark.asyncio
async def test_first_stage_callbacks_are_wired_through_real_attempt_and_sequencer(
    monkeypatch,
) -> None:
    from tldw_chatbook.Chat import console_speculative_voice_session as module

    records: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        module,
        "_persist_voice_event",
        lambda event, **fields: records.append((event, fields)),
    )

    class SubmissionTransport(_Transport):
        def queue_render(self, pcm16):
            super().queue_render(pcm16)
            return RenderSubmission(0, 0, len(self.rendered) - 1)

    clock = _Clock()
    effects: SpeculativeVoiceAttemptEffects

    class DelayedSynthesizer:
        async def synthesize_hands_free(self, *, text: str) -> TTSAudioResponse:
            assert text == "Hello there."
            clock.advance_ms(300)

            async def cleanup() -> None:
                clock.advance_ms(400)

            return TTSAudioResponse(
                provider_id="test",
                model_id="test",
                audio_format="pcm",
                content_type="audio/pcm",
                byte_stream=_AudioStream(),
                sample_rate=48_000,
                cleanup=cleanup,
                metadata={"channels": 1},
            )

    async def submit_event(event: object) -> None:
        if isinstance(event, AttemptDispatchPrepared):
            effects.start_prepared_attempt(event.attempt_epoch)

    effects = SpeculativeVoiceAttemptEffects(
        submit_event=submit_event,
        prepare_attempt=lambda **kwargs: asyncio.sleep(
            0, result=_prepared_attempt(kwargs["attempt_epoch"])
        ),
        gateway=_AttemptGateway(),
        synthesizer=DelayedSynthesizer(),
        transport=SubmissionTransport(),
        promotion=lambda **_kwargs: None,
        promotion_owner=object(),
        dispatch_supervisor=VoiceDispatchSupervisor(),
        project_preview=lambda _preview: None,
        clear_preview=lambda: None,
        submit_accepted_voice_turn=lambda *_args: None,
        clock=clock,
    )
    effects.dispatch_attempt(turn_id="turn", attempt_epoch=1, transcript="private")
    clock.advance_ms(200)
    try:
        async with asyncio.timeout(1):
            while effects.final_render_submission(1) is None:
                await asyncio.sleep(0)
        lifecycle = effects._attempts[1]
        assert lifecycle.first_render_submission == RenderSubmission(0, 0, 0)
        stages = [fields for event, fields in records if event == "voice_first_stage"]
        assert [fields["phase"] for fields in stages] == [
            "provider_delta",
            "eligible_phrase",
            "synthesis_complete",
        ]
        assert [fields["duration_ms"] for fields in stages] == [200, 200, 900]
    finally:
        await effects.close()


def test_first_stage_replacement_rejects_late_old_attempt_success(monkeypatch) -> None:
    from tldw_chatbook.Chat import console_speculative_voice_session as module

    records: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        module,
        "_persist_voice_event",
        lambda event, **fields: records.append((event, fields)),
    )
    clock = _Clock()
    effects = _first_stage_effects(clock)
    effects._attempts[1] = _AttemptLifecycle(
        "old",
        1,
        "old private",
        started_ns=clock(),
        first_render_submission=RenderSubmission(0, 1, 1),
    )
    clock.advance_ms(10)
    effects._on_delta(VoiceAttemptDelta(1, "old delta"))
    effects.fence_attempt(1)

    effects._attempts[2] = _AttemptLifecycle(
        "new",
        2,
        "new private",
        started_ns=clock(),
        first_render_submission=RenderSubmission(0, 2, 2),
    )
    clock.advance_ms(20)
    effects._on_first_eligible_phrase(1)
    effects._on_first_synthesis_complete(1)
    effects.observe_render_receipt(1, RenderBoundary(0, 1, 1, clock()))
    effects._on_delta(VoiceAttemptDelta(2, "new delta"))
    effects._on_delta(VoiceAttemptDelta(2, "new duplicate"))
    effects._on_first_eligible_phrase(2)
    effects._on_first_eligible_phrase(2)
    effects._on_first_synthesis_complete(2)
    effects._on_first_synthesis_complete(2)
    effects.observe_render_receipt(2, RenderBoundary(0, 2, 2, clock() + 10_000_000))

    assert [fields["phase"] for _event, fields in records] == [
        "provider_delta",
        "provider_delta",
        "eligible_phrase",
        "synthesis_complete",
        "render_receipt",
    ]
    assert [fields["duration_ms"] for _event, fields in records] == [10, 20, 20, 20, 20]


def test_first_stage_durable_records_exclude_text_pcm_secrets_and_ids(
    tmp_path: Path,
) -> None:
    private = "VOICE-PRIVATE-SENTINEL-sk-not-a-real-key content-id-739"
    path = tmp_path / "application.log"
    handler = _private_sink(path)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    clock = _Clock()
    effects = _first_stage_effects(clock)
    effects._attempts[1] = _AttemptLifecycle(
        private,
        1,
        private,
        started_ns=clock(),
        first_render_submission=RenderSubmission(0, 1, 2),
    )
    try:
        effects._on_delta(VoiceAttemptDelta(1, private))
        effects._on_first_eligible_phrase(1)
        effects._on_first_synthesis_complete(1)
        effects.observe_render_receipt(1, RenderBoundary(0, 1, 2, 10_000_000))
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()

    persisted = _all_generations(path)
    records = [
        line for line in persisted.splitlines() if "event=voice_first_stage" in line
    ]
    assert len(records) == 4
    assert private not in persisted
    assert "sk-not-a-real-key" not in persisted
    assert "content-id-739" not in persisted
    assert "\\x01\\x02" not in persisted


def test_native_fault_snapshot_omits_invalid_or_unavailable_measurements() -> None:
    transport = SimpleNamespace(
        native_counters={
            "callback_count": 7,
            "fatal_status_bits": True,
            "capture_overflows": -1,
            "invalid_frames": 2**64,
            "capture_occupancy": 65,
            "render_occupancy": "0",
        },
        buffer_capacities=SimpleNamespace(capture_frames=64, render_frames=64),
    )
    evidence = SimpleNamespace(observed_ns=11)

    assert _native_fault_snapshot(transport, evidence, observed_ns=10) == {
        "native_callback_count": 7
    }
    assert _native_fault_snapshot(SimpleNamespace(), evidence, observed_ns=10) == {}


def test_native_fault_snapshot_measures_callback_to_consumer_lag() -> None:
    transport = SimpleNamespace(
        native_counters={"callback_count": 1},
        buffer_capacities=SimpleNamespace(capture_frames=64, render_frames=64),
    )

    assert _native_fault_snapshot(
        transport,
        SimpleNamespace(observed_ns=1_000_000),
        observed_ns=4_500_000,
    ) == {"native_callback_count": 1, "lag_ms": 3}


@pytest.mark.asyncio
async def test_fault_snapshot_records_each_native_generation_at_drain_time(
    native_transport, driver, tmp_path: Path
) -> None:
    path = tmp_path / "application.log"
    handler = _private_sink(path)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    session = ConsoleSpeculativeVoiceSession(
        transport=native_transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(
            aec=None,
            vad=lambda _frame: False,
            **kwargs,
        ),
        transcript_factory=lambda *_args: _UnusedTranscript(),
        effects=_CoordinatorEffects(),
        initial_duplex_mode=DuplexMode.HALF_DUPLEX,
    )
    try:
        await session.coordinator.start()
        await native_transport.start()
        first_bridge = native_transport._stream.bridge
        for _ in range(50):
            emit_callback(driver, first_bridge, status=2)
        emit_callback(driver, first_bridge, status=4)
        assert await session.process_pending_audio()
        emit_callback(driver, first_bridge)
        assert await session.process_pending_audio()
        for index in range(65):
            emit_callback(
                driver,
                first_bridge,
                times=(
                    11.0 + index / 100,
                    11.01 + index / 100,
                    11.02 + index / 100,
                ),
            )
        assert await session.process_pending_audio()

        await native_transport.notify_route_changed(RouteKind.DUPLEX)
        assert await session.process_pending_audio()
        await native_transport.start()
        second_bridge = native_transport._stream.bridge
        for _ in range(50):
            emit_callback(driver, second_bridge, status=2)
        for index in range(65):
            emit_callback(
                driver,
                second_bridge,
                times=(
                    20.0 + index / 100,
                    20.01 + index / 100,
                    20.02 + index / 100,
                ),
            )
        assert await session.process_pending_audio()
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()

    records = _fault_records(_all_generations(path))
    assert len(records) == 2
    host_status = next(
        record for record in records if "status-output_underflow" in record
    )
    ring_loss = next(record for record in records if "buffer-overflow" in record)
    assert "result_type=status-output_underflow" in host_status
    assert "native_fatal_status_bits=4" in host_status
    assert "native_capture_overflows=0" in host_status
    assert "native_invalid_frames=0" in host_status
    assert "native_invalid_timing=0" in host_status
    assert "result_type=buffer-overflow" in ring_loss
    assert "native_fatal_status_bits=0" in ring_loss
    assert "native_capture_overflows=1" in ring_loss
    for record in records:
        assert "operation=drain_snapshot" in record
        assert "phase=capture" in record
        assert "native_callback_count=" in record
        assert "native_capture_occupancy=" in record
        assert "native_render_occupancy=" in record
        assert "lag_ms=" in record
        assert "fault_instant" not in record


@pytest.mark.asyncio
async def test_no_record_native_fault_omits_unavailable_lag(
    native_transport, driver, tmp_path: Path
) -> None:
    path = tmp_path / "application.log"
    handler = _private_sink(path)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    session = ConsoleSpeculativeVoiceSession(
        transport=native_transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(
            aec=None,
            vad=lambda _frame: False,
            **kwargs,
        ),
        transcript_factory=lambda *_args: _UnusedTranscript(),
        effects=_CoordinatorEffects(),
        initial_duplex_mode=DuplexMode.HALF_DUPLEX,
    )
    try:
        await session.coordinator.start()
        await native_transport.start()
        emit_callback(driver, native_transport._stream.bridge, frames=479)
        assert await session.process_pending_audio()
        assert not await session.process_pending_audio()
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()

    records = _fault_records(_all_generations(path))
    assert len(records) == 1
    assert "operation=drain_snapshot" in records[0]
    assert "result_type=missing-timing" in records[0]
    assert "native_invalid_frames=1" in records[0]
    assert "lag_ms=" not in records[0]


@pytest.mark.asyncio
async def test_stt_failure_categories_are_deduplicated_and_content_free(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetFailure

    private = "VOICE-PRIVATE-SENTINEL-sk-not-a-real-key transcript=b'\\x01\\x02'"

    class FailingService:
        def __init__(self, factory):
            self._factory = factory

        def transcribe_buffer(self, **_kwargs):
            raise self._factory()

    path = tmp_path / "application.log"
    handler = _private_sink(path)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    native = _NativeStreamingStt(
        object(),
        provider="parakeet-mlx",
        model=None,
        language="en",
        prepared_candidate=SimpleNamespace(
            process_audio=lambda _pcm: (_ for _ in ()).throw(
                ParakeetFailure("identity_mismatch")
            )
        ),
    )
    try:
        rolling = _RollingWindowStt(
            FailingService(lambda: ParakeetFailure("context_busy")),
            provider="parakeet-mlx",
            model=None,
            language="en",
        )
        for _ in range(2):
            with pytest.raises(ParakeetFailure):
                await rolling.transcribe_window(
                    pcm16=bytes(960), started_ns=0, ended_ns=10_000_000
                )

        imitation = _RollingWindowStt(
            FailingService(
                lambda: RuntimeError(f"parakeet_worker_stream_active {private}")
            ),
            provider="parakeet-mlx",
            model=None,
            language="en",
        )
        with pytest.raises(RuntimeError):
            await imitation.transcribe_window(
                pcm16=bytes(960), started_ns=0, ended_ns=10_000_000
            )

        later_lifecycle = _RollingWindowStt(
            FailingService(lambda: ParakeetFailure("context_busy")),
            provider="parakeet-mlx",
            model=None,
            language="en",
        )
        with pytest.raises(ParakeetFailure):
            await later_lifecycle.transcribe_window(
                pcm16=bytes(960), started_ns=0, ended_ns=10_000_000
            )

        failures = []
        native.start(lambda _result: None, failures.append, lambda _sequence: None)
        native.submit(_frame(0))
        await asyncio.wait_for(native._worker, 1)
        assert isinstance(failures[0], ParakeetFailure)
    finally:
        await native.close()
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()

    persisted = _all_generations(path)
    failures = [line for line in persisted.splitlines() if "event=stt_failed" in line]
    assert len(failures) == 4
    assert sum("error_category=context_busy" in line for line in failures) == 2
    assert sum("error_category=identity_mismatch" in line for line in failures) == 1
    assert sum("error_category=unknown_native" in line for line in failures) == 1
    assert "exception_type=RuntimeError" in persisted
    assert private not in persisted
    assert "sk-not-a-real-key" not in persisted


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["timeout", "disconnected"])
async def test_native_closed_service_preserves_terminal_failure_reason(reason) -> None:
    from tldw_chatbook.Audio.parakeet_voice_worker import (
        ParakeetRequestTimeout,
        ParakeetServiceUnavailable,
        ParakeetVoiceProcess,
    )

    failure = (
        ParakeetRequestTimeout()
        if reason == "timeout"
        else ParakeetServiceUnavailable("disconnected")
    )
    service = ParakeetVoiceProcess(model=None, language="en")
    service._closed = True
    adapter = _NativeStreamingStt(
        service,
        provider="parakeet-mlx",
        model=None,
        language="en",
        prepared_candidate=SimpleNamespace(
            process_audio=lambda _pcm: (_ for _ in ()).throw(failure)
        ),
    )
    observed = []
    try:
        adapter.start(lambda _result: None, observed.append, lambda _sequence: None)
        adapter.submit(_frame(0))
        await asyncio.wait_for(adapter._worker, 1)
        assert observed == [failure]
        assert observed[0].reason == reason
    finally:
        await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["enter", "exit"])
@pytest.mark.parametrize(
    "case",
    [
        "timeout",
        "disconnected",
        "context_busy",
        "identity_mismatch",
        "provider",
    ],
)
async def test_parakeet_context_projects_failure_after_checked_retirement(
    phase: str,
    case: str,
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Audio.parakeet_voice_worker import (
        ParakeetFailure,
        ParakeetRequestTimeout,
        ParakeetServiceUnavailable,
        ParakeetVoiceProcess,
    )
    from tldw_chatbook.Chat.console_speculative_voice_session import _SerialSttWorker

    private = "VOICE-PRIVATE-SENTINEL-sk-not-a-real-key"
    if case == "timeout":
        failure = ParakeetRequestTimeout()
    elif case == "disconnected":
        failure = ParakeetServiceUnavailable("disconnected")
    elif case in ("context_busy", "identity_mismatch"):
        failure = ParakeetFailure(case)
    else:
        failure = RuntimeError(f"parakeet_worker_timeout {private}")
    expected_reason = "unknown_native" if case == "provider" else case
    reached, release = threading.Event(), threading.Event()

    class Context:
        result = SimpleNamespace(text="")

        def __enter__(self):
            if phase == "enter":
                reached.set()
                assert release.wait(2), "test did not release context entry"
                raise failure
            return self

        def add_pcm16(self, _pcm):
            return None

        def __exit__(self, *_args):
            if phase == "exit":
                reached.set()
                assert release.wait(2), "test did not release checked exit"
                raise failure

    path = tmp_path / "application.log"
    handler = _private_sink(path)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    service = ParakeetVoiceProcess(model=None, language="en")
    worker = _SerialSttWorker()
    adapter = _NativeStreamingStt(
        service,
        provider="parakeet-mlx",
        model=None,
        language="en",
        serial_worker=worker,
        prepared_candidate=SimpleNamespace(
            model=SimpleNamespace(transcribe_stream=lambda **_kwargs: Context())
        ),
        quiet_seconds=0.01,
    )
    failures = []
    try:
        adapter.start(lambda _result: None, failures.append, lambda _sequence: None)
        adapter.submit(_frame(0))
        assert await asyncio.to_thread(reached.wait, 2), f"{phase} was not reached"
        assert worker._model_lock.locked()
        release.set()
        await asyncio.wait_for(adapter._worker, 2)
        assert len(failures) == 1
        if case in ("context_busy", "identity_mismatch", "provider"):
            assert failures[0] is not failure
            assert isinstance(failures[0], ParakeetServiceUnavailable)
        else:
            assert failures == [failure]
        assert failures[0].reason == expected_reason
        assert service.closed and service.reaped
        assert not worker._model_lock.locked()
        with pytest.raises(ParakeetServiceUnavailable, match="closed"):
            async with worker.lease(service):
                pytest.fail("retired service must not permit rolling reuse")
    finally:
        release.set()
        await adapter.close()
        await worker.close()
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()

    records = [
        line
        for line in _all_generations(path).splitlines()
        if "event=stt_failed" in line
    ]
    assert len(records) == 1
    assert f"error_category={expected_reason}" in records[0]
    assert private not in _all_generations(path)
