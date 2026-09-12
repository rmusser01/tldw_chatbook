"""Causal render completion and capture history, without audio hardware."""

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_speculative_voice import _coordinator, _start_attempt
from Tests.Audio import test_native_duplex_stream as native_helpers
from Tests.Audio.fakes.native_duplex_helpers import emit_callback
from Tests.integration.test_speculative_voice_pipeline import (
    _CoordinatorEffects,
    _HealthyAec,
    _TransitionIsolationMonitor,
    _Transport,
    _frame,
    _timed_aec_frame,
)
from tldw_chatbook.Audio.duplex_contracts import (
    CaptureDrainError,
    DeviceRouteChanged,
    RenderBoundary,
    RenderSubmission,
    RouteKind,
    DuplexMode,
)
from tldw_chatbook.Audio.native_duplex_stream import AudioShutdownUnconfirmed
from tldw_chatbook.Audio.voice_preprocessor import VoicePreprocessor
from tldw_chatbook.Chat.console_speculative_voice import (
    AttemptGenerationCompleted,
    AttemptPlaybackStarted,
    ManualInterruption,
    SpeculativeVoiceState,
)
from tldw_chatbook.Chat.console_speculative_voice_session import (
    ConsoleSpeculativeVoiceSession,
    SpeculativeVoiceAttemptEffects,
    _AttemptLifecycle,
)
from tldw_chatbook.Chat.console_voice_supervisor import VoiceDispatchSupervisor
from tldw_chatbook.Chat.voice_phrase_sequencer import PhraseSpeechSequencer
from tldw_chatbook.Chat.console_voice_controls import ControlKind

driver = native_helpers.driver
native_transport = native_helpers.native_transport
portaudio = native_helpers.portaudio


class Transcript:
    def __init__(self, turn_id, _callback):
        self.turn_id = turn_id
        self.frames = []

    def append_admitted_frame(self, frame):
        self.frames.append(frame)

    async def seal_through(self, _sequence):
        await asyncio.Future()

    async def close(self):
        pass


def production_effects(transport):
    return SpeculativeVoiceAttemptEffects(
        submit_event=lambda event: asyncio.sleep(0),
        prepare_attempt=None,
        gateway=None,
        synthesizer=None,
        transport=transport,
        promotion=lambda **kwargs: None,
        promotion_owner=object(),
        dispatch_supervisor=VoiceDispatchSupervisor(),
        project_preview=lambda preview: None,
        clear_preview=lambda: None,
        submit_accepted_voice_turn=lambda *args: None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["attempt", "tool_abort", "sequencer", "ready_tts"])
async def test_production_cancel_fences_native_before_provider_or_tts_cleanup(
    native_transport, driver, entry
):
    transport = native_transport
    await transport.start()
    bridge = transport._stream.bridge
    effects = production_effects(transport)
    observed = []

    def cleanup_entry():
        observed.append(emit_callback(driver, bridge)[0])

    async def tts_worker():
        try:
            await asyncio.Future()
        finally:
            cleanup_entry()

    speech = PhraseSpeechSequencer(epoch=1, synthesizer=None, sink=transport)
    speech._worker = asyncio.create_task(tts_worker())
    await asyncio.sleep(0)
    effects._attempts[1] = _AttemptLifecycle(
        "turn",
        1,
        "text",
        speech=speech,
        attempt=SimpleNamespace(invalidate=cleanup_entry),
    )
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(aec=None, **kwargs),
        transcript_factory=Transcript,
        effects=effects,
    )
    transport.queue_render(b"\x01\x00" * 480)
    try:
        if entry == "attempt":
            session._effects.fence_attempt(1)
            cleanup_entry()  # provider cancellation starts in this same turn
            await effects.abort_output(1)
        elif entry == "tool_abort":
            cleanup = effects.abort_output(1)
            cleanup_entry()  # abort's synchronous side must precede tool cleanup
            await cleanup
        elif entry == "ready_tts":

            async def ready_producer():
                await speech._queue_render(b"\x01\x00" * 480)
                cleanup_entry()

            # A TTS continuation is already ready before cancellation is
            # scheduled. It must not republish old speech under the new epoch.
            producer = asyncio.create_task(ready_producer())
            session._effects.fence_attempt(1)
            cleanup = asyncio.create_task(effects.abort_output(1))
            await asyncio.gather(producer, cleanup)
        else:
            await speech.cancel(1)
        assert observed and all(pcm == bytes(960) for pcm in observed)
        assert bridge.snapshot()["active"]  # attempt cancel leaves capture available
        assert transport.queue_render(b"\x02\x00" * 480) is not None
        # Late cleanup of this old attempt cannot invalidate its replacement.
        output_epoch = bridge.output_epoch
        callbacks_before = len(observed)
        session._effects.fence_attempt(1)
        await effects.abort_output(1)
        assert bridge.output_epoch == output_epoch
        replacement = (
            observed[callbacks_before]
            if len(observed) > callbacks_before
            else emit_callback(driver, bridge)[0]
        )
        assert replacement == b"\x02\x00" * 480
    finally:
        await speech.cancel(1)
        await transport.close()


@pytest.mark.asyncio
async def test_native_half_duplex_buffered_idle_speech_precedes_ready_terminal_seal(
    native_transport, driver
):
    transport = native_transport
    await transport.start()
    bridge = transport._stream.bridge
    emit_callback(driver, bridge, times=(10.00, 10.01, 10.02))
    transport.pop_capture()
    transport.acknowledge_capture(0, clock_generation=0, dsp_ok=True, vad_ok=True)
    offset = transport._device_to_monotonic_offset_ns
    session, _, scheduler, effects, turn_id, epoch = await receipt_session(
        transport, mode=DuplexMode.HALF_DUPLEX
    )
    session._effects.seal_transcript_through = effects.seal_transcript_through
    effects.submission = transport.queue_render(b"\x01\x00" * 480)
    try:
        await session.submit(AttemptGenerationCompleted(epoch, "obsolete answer"))
        # Speech captured 30ms before the final DAC end remains queued when the
        # receipt is discovered. Half-duplex playback gating remains enabled.
        for i in range(1, 6):
            emit_callback(
                driver,
                bridge,
                times=(10 + i * 0.01, 10.01 + i * 0.01, 10.02 + i * 0.01),
            )
        scheduler.now_ns = offset + 10_050_000_000
        await session.process_pending_audio()
        await session.coordinator.flush()
        assert effects.promotions == []
        assert session.coordinator.snapshot.turn_id == turn_id
        assert session.coordinator.snapshot.current_attempt_epoch is None
        assert session._transcripts[0].turn_id == turn_id
        assert session._transcripts[0].frames[0].sequence == 1
        assert transport.capture_count == 4
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "check", ["synchronous", "deadline", "cleanup_failure", "cancelled_observer"]
)
async def test_session_shutdown_deactivates_and_observes_native_independently(
    native_transport, portaudio, driver, check
):
    transport = native_transport
    await transport.start()
    stream = transport._stream
    now = [10.0]
    stream._clock = lambda: now[0]
    portaudio.close_gate = threading.Event()
    entered, release = asyncio.Event(), asyncio.Event()

    async def interrupt():
        entered.set()
        if check == "cleanup_failure":
            raise RuntimeError("transcription cleanup failed")
        await release.wait()

    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(aec=None, **kwargs),
        transcript_factory=Transcript,
        effects=_CoordinatorEffects(),
        interrupt_transcription=interrupt,
    )
    close = None
    try:
        session.fence_audio_admission()  # actual facade's synchronous seam
        if check == "synchronous":
            assert not stream.bridge.snapshot()["active"]
            assert stream._deadline == 12.0
            assert stream._closing.is_set()
            emit_callback(driver, stream.bridge)
            assert stream.bridge.snapshot()["capture_occupancy"] == 0
        close = session.fence_and_close(ControlKind.TEARDOWN)
        await entered.wait()
        if check == "cancelled_observer":
            close.cancel()
            await asyncio.sleep(0)
            close = session.fence_and_close(ControlKind.TEARDOWN)
        now[0] += 3
        with pytest.raises(AudioShutdownUnconfirmed):
            await asyncio.wait_for(asyncio.shield(close), 1)
        assert not stream.bridge.snapshot()["active"]
        with pytest.raises(AudioShutdownUnconfirmed):
            await session.fence_and_close(ControlKind.TEARDOWN)
    finally:
        release.set()
        portaudio.close_gate.set()
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        await transport.close()
        await native_helpers.eventually(lambda: stream.registry.owner is None)


@pytest.mark.asyncio
async def test_cancelled_playback_history_cannot_reach_idle_raw_stt():
    transport = _Transport()
    effects = _CoordinatorEffects()
    effects.assistant_rendering = False  # cancellation already fenced lifecycle
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(
            aec=None, vad=lambda _frame: True, **kwargs
        ),
        transcript_factory=Transcript,
        effects=effects,
    )
    await session.coordinator.start()
    try:
        transport.captures.extend(
            replace(_frame(i), assistant_rendering=True) for i in range(3)
        )
        for _ in range(3):
            await session.process_pending_audio()
        assert session._transcripts == []
        transport.captures.append(replace(_frame(3), assistant_rendering=False))
        await session.process_pending_audio()
        assert [f.sequence for f in session._transcripts[0].frames] == [3]
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_pending_replay_keeps_each_frames_recorded_playback_context():
    transport = _Transport()
    effects = _CoordinatorEffects()
    effects.assistant_rendering = True
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(
            aec=_HealthyAec(),
            isolation_monitor=_TransitionIsolationMonitor(),
            vad=lambda _frame: True,
            healthy_streak=1,
            **kwargs,
        ),
        transcript_factory=Transcript,
        effects=effects,
    )
    admitted = []
    original_submit = session.coordinator.submit

    async def record(event):
        if type(event).__name__ == "AdmittedSpeechFrame":
            admitted.append((event.sequence, event.assistant_rendering))
        return await original_submit(event)

    session.coordinator.submit = record
    await session.coordinator.start()
    try:
        for i in range(4):
            transport.references.append(_timed_aec_frame(i))
            transport.captures.append(
                replace(
                    _timed_aec_frame(i, render_reference_sequence=i),
                    assistant_rendering=True,
                )
            )
            await session.process_pending_audio()
        assert admitted == []
        effects.assistant_rendering = False
        transport.captures.append(
            replace(
                _timed_aec_frame(4, render_reference_sequence=3),
                assistant_rendering=False,
            )
        )
        await session.process_pending_audio()
        assert admitted == [(0, True), (1, True), (2, True), (3, True), (4, False)]
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_session_reports_categorical_unconfirmed_audio_shutdown():
    class UncertainTransport(_Transport):
        async def close(self):
            raise AudioShutdownUnconfirmed("audio_shutdown_unconfirmed")

    session = ConsoleSpeculativeVoiceSession(
        transport=UncertainTransport(),
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(aec=None, **kwargs),
        transcript_factory=Transcript,
        effects=_CoordinatorEffects(),
    )
    with pytest.raises(AudioShutdownUnconfirmed, match="audio_shutdown_unconfirmed"):
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_pump_reports_shutdown_uncertainty_through_runtime_failure_callback():
    class UncertainTransport(_Transport):
        async def close(self):
            raise AudioShutdownUnconfirmed("private native error")

    transport = UncertainTransport()
    transport.raise_on_capture = True
    errors = []
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(aec=None, **kwargs),
        transcript_factory=Transcript,
        effects=_CoordinatorEffects(),
        on_runtime_failure=errors.append,
    )
    await session.enter(capture_live=False)
    pump = session._pump_task
    await pump
    assert isinstance(errors[-1], AudioShutdownUnconfirmed)
    assert str(errors[-1]) == "audio_shutdown_unconfirmed"
    with pytest.raises(AudioShutdownUnconfirmed):
        await session.fence_and_close(ControlKind.TEARDOWN)


class ReceiptTransport(_Transport):
    def __init__(self):
        super().__init__()
        self.boundary = None
        self.native_counters = {"callback_count": 20}
        self.buffer_capacities = SimpleNamespace(render_frames=64)
        self.stream_latency_seconds = (0.01, 0.02)

    def render_boundary(self, submission):
        if isinstance(self.boundary, Exception):
            raise self.boundary
        return self.boundary


async def receipt_session(transport=None, *, mode=DuplexMode.FULL_DUPLEX):
    coordinator, scheduler, effects = await _coordinator(mode=mode)
    transport = transport or ReceiptTransport()
    effects.assistant_rendering = False
    effects.submission = None
    effects.final_render_submission = lambda epoch: effects.submission
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(
            aec=None, vad=lambda f: True, **kwargs
        ),
        transcript_factory=Transcript,
        effects=effects,
        clock=lambda: scheduler.now_ns,
    )
    session._coordinator = coordinator
    coordinator._effects = session._effects
    turn_id, epoch = await _start_attempt(coordinator, scheduler)
    await coordinator.submit(AttemptPlaybackStarted(epoch))
    effects.submission = RenderSubmission(0, 4, 9)
    return session, transport, scheduler, effects, turn_id, epoch


@pytest.mark.asyncio
@pytest.mark.parametrize("pump_before_capture", [False, True])
async def test_pump_installs_actual_boundary_before_post_boundary_capture(
    pump_before_capture,
):
    session, transport, scheduler, effects, turn_id, epoch = await receipt_session()
    try:
        transport.boundary = RenderBoundary(0, 4, 9, 1_000_000_000)
        scheduler.advance_ms(301)  # now 1,001,000,000 ns, beyond actual DAC
        if pump_before_capture:
            await session.process_pending_audio()
        transport.captures.append(
            replace(
                _frame(1),
                started_ns=1_000_000_001,
                ended_ns=1_010_000_001,
                assistant_rendering=False,
            )
        )
        await session.process_pending_audio()
        assert session.coordinator.snapshot.turn_id == turn_id
        assert session.coordinator.snapshot.current_attempt_epoch == epoch
        assert session.coordinator.snapshot.pending_next_turn_id is not None
        await session.submit(AttemptGenerationCompleted(epoch, "answer"))
        assert session.state is SpeculativeVoiceState.SEALING
        assert effects.promotions == []
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_complete", [False, True])
async def test_callback_cessation_uses_fixed_delivery_deadline(generation_complete):
    session, transport, scheduler, effects, turn_id, epoch = await receipt_session()
    try:
        if generation_complete:
            await session.submit(AttemptGenerationCompleted(epoch, "answer"))
        await session.process_pending_audio()  # arm at 700 ms
        # 30 startup callbacks + 64 render frames + 20 ms latency + 500 ms = 1460 ms
        transport.native_counters["callback_count"] = 50
        scheduler.advance_ms(1459)
        await session.process_pending_audio()
        assert session.coordinator.snapshot.current_attempt_epoch == epoch
        scheduler.advance_ms(1)
        await session.process_pending_audio()
        assert session.state is SpeculativeVoiceState.REBUILDING_AUDIO
        await session.submit(AttemptGenerationCompleted(epoch, "too late"))
        assert effects.promotions == []
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_pump_first", [False, True])
async def test_native_actual_receipt_routes_later_capture_in_both_pump_orders(
    native_transport, driver, terminal_pump_first
):
    transport = native_transport
    await transport.start()
    bridge = transport._stream.bridge
    emit_callback(driver, bridge, times=(10.00, 10.01, 10.02))
    capture = transport.pop_capture()
    transport.acknowledge_capture(0, clock_generation=0, dsp_ok=True, vad_ok=True)
    offset = transport._device_to_monotonic_offset_ns
    assert capture.started_ns == offset + 10_000_000_000
    session, _, scheduler, effects, turn_id, epoch = await receipt_session(transport)
    effects.submission = transport.queue_render(b"\x01\x00" * 480)
    assert effects.submission == RenderSubmission(0, 0, 0)
    # Keep production preroll enabled across the actual native DAC boundary.
    session._preprocessor = VoicePreprocessor(
        aec=None,
        vad=lambda f: f.sequence >= 5,
        on_admitted_frame=session._on_admitted_frame,
        on_processed=session._on_processed,
        on_classification_changed=session._on_classification_changed,
    )
    try:
        scheduler.now_ns = offset + 10_025_000_000
        for times in [
            (10.01, 10.02, 10.03),
            (10.02, 10.03, 10.04),
            (10.03, 10.04, 10.05),
            (10.04, 10.05, 10.06),
        ]:
            emit_callback(driver, bridge, times=times)
            await session.process_pending_audio()
        assert (
            session.coordinator.snapshot.terminal_boundary_ns == offset + 10_040_000_000
        )
        assert not session._playback_target.terminal_sent
        scheduler.now_ns = offset + 10_050_000_000
        if terminal_pump_first:
            await session.process_pending_audio()
        emit_callback(driver, bridge, times=(10.05, 10.06, 10.07))
        await session.process_pending_audio()
        assert session.coordinator.snapshot.turn_id == turn_id
        assert session.coordinator.snapshot.current_attempt_epoch == epoch
        assert session.coordinator.snapshot.pending_next_turn_id is not None
        assert session._playback_target.terminal_sent
        assert effects.promotions == []
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_successful_causal_seals_stop_audio_deadline_during_slow_promotion():
    from tldw_chatbook.Audio.duplex_contracts import DrainReceipt

    session, transport, scheduler, effects, turn_id, epoch = await receipt_session()
    effects.promotion_future = asyncio.get_running_loop().create_future()
    effects.drain_receipt = DrainReceipt(1_000_000_001, 0, 0, 0, 0)
    session._effects.drain_capture_through = effects.drain_capture_through
    session._effects.drain_pending_classification_through = (
        effects.drain_pending_classification_through
    )
    session._effects.seal_transcript_through = effects.seal_transcript_through
    try:
        transport.boundary = RenderBoundary(0, 4, 9, 1_000_000_000)
        await session.submit(AttemptGenerationCompleted(epoch, "answer"))
        scheduler.advance_ms(300)
        await session.process_pending_audio()
        await session.coordinator.flush()
        assert session.state is SpeculativeVoiceState.PROMOTING
        scheduler.advance_ms(500)
        await session.process_pending_audio()
        assert session.state is SpeculativeVoiceState.PROMOTING
        assert effects.rebuilds == []
    finally:
        effects.promotion_future.set_result(True)
        await session.coordinator.flush()
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        CaptureDrainError("invalidated"),
        RenderBoundary(0, 5, 9, 1_000_000_000),
        RenderBoundary(1, 4, 9, 1_000_000_000),
        RenderBoundary(0, 4, 10, 1_000_000_000),
    ],
)
async def test_uncertain_or_interleaved_receipt_cannot_promote_completed_generation(
    boundary,
):
    session, transport, scheduler, effects, turn_id, epoch = await receipt_session()
    try:
        await session.submit(AttemptGenerationCompleted(epoch, "answer"))
        transport.boundary = boundary
        await session.process_pending_audio()
        assert session.state is SpeculativeVoiceState.REBUILDING_AUDIO
        assert effects.promotions == []
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", [False, True])
async def test_cancel_or_route_reset_invalidates_pending_target_and_late_receipt(reset):
    session, transport, scheduler, effects, turn_id, epoch = await receipt_session()
    try:
        await session.process_pending_audio()
        assert session._playback_target is not None
        await session.submit(
            DeviceRouteChanged(0, RouteKind.DUPLEX) if reset else ManualInterruption()
        )
        assert session._playback_target is None
        transport.boundary = RenderBoundary(0, 4, 9, 1_000_000_000)
        scheduler.advance_ms(301)
        await session.process_pending_audio()
        assert session.coordinator.snapshot.terminal_boundary_ns is None
        assert effects.promotions == []
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_first_render_receipt_uses_original_valid_capture_after_delayed_drain():
    session, transport, scheduler, effects, turn_id, epoch = await receipt_session()
    observed = []
    effects.first_render_submission = lambda attempt_epoch: effects.submission
    effects.observe_render_receipt = lambda attempt_epoch, boundary: observed.append(
        (attempt_epoch, boundary)
    )
    matching = RenderBoundary(0, 4, 9, 2_010_000_000)
    try:
        # Queue admission alone, lost records, and a later submission cannot stand
        # in for the first callback that actually committed output.
        await session.process_pending_audio()
        for sequence in range(20):
            session._observe_first_render_receipt(_frame(sequence))
        session._observe_first_render_receipt(
            replace(
                _frame(20), committed_render=RenderBoundary(0, 4, 10, 2_020_000_000)
            )
        )
        session._observe_first_render_receipt(
            replace(_frame(21), discontinuity=True, committed_render=matching)
        )
        assert observed == []

        session._observe_first_render_receipt(
            replace(_frame(22), committed_render=matching)
        )
        assert observed == [(epoch, matching)]

        await session.submit(ManualInterruption())
        session._observe_first_render_receipt(
            replace(_frame(23), committed_render=matching)
        )
        assert observed == [(epoch, matching)]
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_output_abort_revokes_first_receipt_before_original_capture_delivery(
    native_transport, driver, monkeypatch
):
    from Tests.Chat.test_voice_phrase_sequencer import (
        _ChunkStream,
        _Synthesizer,
        _response,
        _wait_until,
    )
    from tldw_chatbook.Chat import console_speculative_voice_session as module

    records = []
    monkeypatch.setattr(
        module,
        "_persist_voice_event",
        lambda event, **fields: records.append((event, fields)),
    )
    transport = native_transport
    await transport.start()
    coordinator, scheduler, _coordinator_effects = await _coordinator()
    effects = production_effects(transport)
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(
            aec=None, vad=lambda _frame: False, **kwargs
        ),
        transcript_factory=Transcript,
        effects=effects,
        clock=lambda: scheduler.now_ns,
    )
    session._coordinator = coordinator
    coordinator._effects = session._effects
    turn_id, epoch = await _start_attempt(coordinator, scheduler)
    speech = PhraseSpeechSequencer(
        epoch=epoch,
        sink=transport,
        synthesizer=_Synthesizer(lambda _text: _response(_ChunkStream((bytes(960),)))),
        on_playback_started=effects._on_playback_started,
        on_failed=effects._on_tts_failed,
        on_first_eligible_phrase=effects._on_first_eligible_phrase,
        on_first_synthesis_complete=effects._on_first_synthesis_complete,
    )
    lifecycle = _AttemptLifecycle(
        turn_id,
        epoch,
        "private",
        started_ns=scheduler.now_ns,
        speech=speech,
    )
    effects._attempts[epoch] = lifecycle
    try:
        await speech.feed(epoch, "Queued output. ")
        await _wait_until(lambda: lifecycle.first_render_submission is not None)
        submission = lifecycle.first_render_submission
        emit_callback(driver, transport._stream.bridge, times=(10.0, 10.01, 10.02))

        cleanup = effects.abort_output(epoch)
        assert lifecycle.current  # Tool handoff keeps the attempt lifecycle live.
        assert effects.first_render_submission(epoch) is None
        with pytest.raises(CaptureDrainError):
            transport.render_boundary(submission)
        await session.process_pending_audio()
        await cleanup

        assert not any(
            event == "voice_first_stage" and fields.get("phase") == "render_receipt"
            for event, fields in records
        )
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_first", [False, True])
async def test_native_first_render_receipt_uses_paired_record_after_latest_advances(
    native_transport, driver, drop_first
):
    transport = native_transport
    await transport.start()
    session, _, _scheduler, effects, _turn_id, epoch = await receipt_session(transport)
    observed = []
    effects.first_render_submission = lambda attempt_epoch: effects.submission
    effects.observe_render_receipt = lambda attempt_epoch, boundary: observed.append(
        (attempt_epoch, boundary)
    )
    submissions = [transport.queue_render(bytes(960)) for _ in range(12)]
    assert all(submission is not None for submission in submissions)
    effects.submission = submissions[0]
    bridge = transport._stream.bridge
    try:
        for index in range(12):
            emit_callback(
                driver,
                bridge,
                times=(
                    10.0 + index / 100,
                    10.01 + index / 100,
                    10.02 + index / 100,
                ),
            )
        assert observed == []
        if drop_first:
            first_capture = transport.pop_capture()
            assert (
                first_capture.committed_render.submission_id
                == submissions[0].submission_id
            )
        else:
            await session.process_pending_audio()

        assert (
            transport.render_boundary(submissions[-1]).submission_id
            == submissions[-1].submission_id
        )
        assert transport.render_boundary(effects.submission) is None

        for _ in range(11):
            await session.process_pending_audio()

        if drop_first:
            assert observed == []
        else:
            assert len(observed) == 1
            assert observed[0][0] == epoch
            assert observed[0][1].submission_id == effects.submission.submission_id
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_known_dac_boundary_waits_for_elapsed_time_and_times_out_without_generation():
    session, transport, scheduler, effects, turn_id, epoch = await receipt_session()
    try:
        transport.boundary = RenderBoundary(0, 4, 9, 1_000_000_000)
        await session.process_pending_audio()
        assert session.coordinator.snapshot.terminal_boundary_ns == 1_000_000_000
        assert not session._playback_target.terminal_sent
        scheduler.advance_ms(300)
        await session.process_pending_audio()
        assert session._playback_target.terminal_sent
        assert session.state is SpeculativeVoiceState.SPEAKING
        scheduler.advance_ms(500)
        await session.process_pending_audio()
        assert session.state is SpeculativeVoiceState.REBUILDING_AUDIO
        assert effects.promotions == []
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_final_submission_armed_during_preprocessing_precedes_same_capture_admission():
    session, transport, scheduler, effects, turn_id, epoch = await receipt_session()
    effects.submission = None
    original = session._preprocessor.process_capture

    async def finish_synthesis(*args, **kwargs):
        await original(*args, **kwargs)
        effects.submission = RenderSubmission(0, 4, 9)

    session._preprocessor.process_capture = finish_synthesis
    try:
        transport.boundary = RenderBoundary(0, 4, 9, 1_000_000_000)
        transport.captures.append(
            replace(
                _frame(1),
                started_ns=1_000_000_001,
                ended_ns=1_010_000_001,
                assistant_rendering=False,
            )
        )
        await session.process_pending_audio()
        assert session.coordinator.snapshot.current_attempt_epoch == epoch
        assert session.coordinator.snapshot.pending_next_turn_id is not None
        assert effects.promotions == []
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
@pytest.mark.parametrize("preroll", [False, True])
async def test_pre_boundary_admission_finishes_before_terminal_transcript_seal(preroll):
    from tldw_chatbook.Audio.duplex_contracts import DrainReceipt

    session, transport, scheduler, effects, turn_id, epoch = await receipt_session()
    effects.submission = None
    effects.drain_receipt = DrainReceipt(1_010_000_000, 1, 1, 1, 0)
    session._effects.drain_capture_through = effects.drain_capture_through
    session._effects.seal_transcript_through = effects.seal_transcript_through
    original = session._preprocessor.process_capture

    if preroll:
        session._preprocessor._vad = lambda frame: frame.sequence >= 2
        transport.captures.append(
            replace(
                _frame(1),
                started_ns=989_999_999,
                ended_ns=999_999_999,
                assistant_rendering=False,
            )
        )
        await session.process_pending_audio()

    async def finish_synthesis(*args, **kwargs):
        await original(*args, **kwargs)
        effects.submission = RenderSubmission(0, 4, 9)

    session._preprocessor.process_capture = finish_synthesis
    try:
        await session.submit(AttemptGenerationCompleted(epoch, "obsolete answer"))
        scheduler.advance_ms(310)
        transport.boundary = RenderBoundary(0, 4, 9, 1_000_000_000)
        transport.captures.append(
            replace(
                _frame(2 if preroll else 1),
                started_ns=999_999_999,
                ended_ns=1_009_999_999,
                assistant_rendering=False,
            )
        )
        await session.process_pending_audio()
        await session.coordinator.flush()
        assert effects.promotions == []
        assert session.coordinator.snapshot.turn_id == turn_id
        assert session.coordinator.snapshot.current_attempt_epoch is None
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_native_capture_after_abort_preserves_playback_gate(
    native_transport, driver
):
    transport = native_transport
    effects = _CoordinatorEffects()
    effects.assistant_rendering = False
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(
            aec=None, vad=lambda frame: frame.sequence >= 2, **kwargs
        ),
        transcript_factory=Transcript,
        effects=effects,
    )
    await session.coordinator.start()
    await transport.start()
    bridge = transport._stream.bridge
    try:
        assert transport.queue_render(b"\x01\x00" * 480) == RenderSubmission(
            0, bridge.output_epoch, 0
        )
        emit_callback(driver, bridge, times=(10.00, 10.01, 10.02))
        emit_callback(driver, bridge, times=(10.01, 10.02, 10.03))
        emit_callback(driver, bridge, times=(10.02, 10.03, 10.04))
        await transport.abort_output()
        for _ in range(3):
            await session.process_pending_audio()
        assert session._transcripts == []
        emit_callback(driver, bridge, times=(10.03, 10.04, 10.05))
        await session.process_pending_audio()
        assert [f.sequence for f in session._transcripts[0].frames] == [3]
        assert session._transcripts[0].frames[0].assistant_rendering is False
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_first", [False, True])
@pytest.mark.parametrize("generation_complete", [False, True])
async def test_default_preroll_preserves_pcm_but_routes_by_actual_postboundary_speech(
    terminal_first, generation_complete
):
    session, transport, scheduler, effects, turn_id, epoch = await receipt_session()
    session._preprocessor = VoicePreprocessor(
        aec=None,
        vad=lambda frame: frame.sequence == 2,
        on_admitted_frame=session._on_admitted_frame,
        on_processed=session._on_processed,
        on_classification_changed=session._on_classification_changed,
    )
    try:
        if generation_complete:
            effects.drain_future = asyncio.get_running_loop().create_future()
            session._effects.drain_capture_through = effects.drain_capture_through
            await session.submit(AttemptGenerationCompleted(epoch, "completed answer"))
        transport.boundary = RenderBoundary(0, 4, 9, 1_000_000_000)
        # Exact-boundary silence, followed by positive speech ten milliseconds later.
        silent = replace(
            _frame(1),
            started_ns=1_000_000_000,
            ended_ns=1_010_000_000,
            pcm16=b"\x01\x00" * 480,
            assistant_rendering=False,
        )
        positive = replace(
            _frame(2),
            started_ns=1_010_000_000,
            ended_ns=1_020_000_000,
            pcm16=b"\x02\x00" * 480,
            assistant_rendering=False,
        )
        transport.captures.append(silent)
        await session.process_pending_audio()
        if terminal_first:
            scheduler.advance_ms(320)
            await session.process_pending_audio()
        transport.captures.append(positive)
        await session.process_pending_audio()
        if not terminal_first:
            scheduler.advance_ms(320)
            await session.process_pending_audio()
        assert session.coordinator.snapshot.current_attempt_epoch == epoch
        assert session.coordinator.snapshot.turn_id == turn_id
        new_turn = session.coordinator.snapshot.pending_next_turn_id
        assert new_turn is not None and new_turn != turn_id
        frames = session._transcripts[0].frames
        assert session._transcripts[0].turn_id == new_turn
        assert [f.speech_started_ns for f in frames] == [1_010_000_000, None]
        assert [
            (f.sequence, f.started_ns, f.ended_ns, f.pcm16, f.assistant_rendering)
            for f in frames
        ] == [
            (1, 1_000_000_000, 1_010_000_000, silent.pcm16, False),
            (2, 1_010_000_000, 1_020_000_000, positive.pcm16, False),
        ]
        assert effects.promotions == []
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_first", [False, True])
async def test_callback_cessation_while_queueing_cannot_promote_without_final_submission(
    generation_first,
):
    from Tests.Audio.fakes.fake_duplex_backend import FakeDuplexBackend, ManualClock
    from Tests.Chat.test_voice_phrase_sequencer import (
        _Synthesizer,
        _response,
        _ChunkStream,
    )
    from tldw_chatbook.Audio.duplex_transport import DuplexAudioTransport
    from tldw_chatbook.Chat.voice_phrase_sequencer import PhraseSpeechSequencer
    from tldw_chatbook.Chat.console_speculative_voice_session import (
        SpeculativeVoiceAttemptEffects,
    )

    coordinator, scheduler, effects = await _coordinator()
    transport = DuplexAudioTransport(backend=FakeDuplexBackend(), clock=ManualClock())
    await transport.start()
    posted = []

    def post(event):
        posted.append(asyncio.create_task(coordinator.submit(event)))

    # Exercise the production failure mapping with the real sequencer/reducer.
    attempt_effects = SimpleNamespace(_post=post)
    turn_id, epoch = await _start_attempt(coordinator, scheduler)
    sequencer = PhraseSpeechSequencer(
        epoch=epoch,
        sink=transport,
        synthesizer=_Synthesizer(lambda _: _response(_ChunkStream((bytes(960),) * 65))),
        on_playback_started=lambda epoch: post(AttemptPlaybackStarted(epoch)),
        on_failed=lambda epoch, code: SpeculativeVoiceAttemptEffects._on_tts_failed(
            attempt_effects, epoch, code
        ),
    )
    try:
        if generation_first:
            await coordinator.submit(
                AttemptGenerationCompleted(epoch, "completed answer")
            )
        await sequencer.feed(epoch, "Sixty five frames. ")
        await sequencer.finish(epoch)
        async with asyncio.timeout(2):
            while sequencer.failure_code is None:
                await asyncio.sleep(0.005)
        assert transport.buffer_occupancy.render_frames == 64
        assert sequencer.failure_code == "output_rejected"
        assert sequencer.final_submission is None
        await asyncio.gather(*posted)
        if not generation_first:
            await coordinator.submit(
                AttemptGenerationCompleted(epoch, "completed answer")
            )
        await coordinator.flush()
        assert effects.promotions == []
        assert effects.drain_calls == []
        assert effects.seal_calls == []
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
    finally:
        await sequencer.cancel(epoch)
        await transport.close()
        await coordinator.close()
