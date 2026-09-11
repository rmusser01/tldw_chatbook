"""Software-only contracts for the extracted audio owner and normalized speech."""

import asyncio
from collections import deque
from dataclasses import replace
import importlib
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_speculative_voice import _FakeEffects, _FakeScheduler
from Tests.Audio import test_native_duplex_stream as native_helpers
from Tests.Audio.fakes.native_duplex_helpers import emit_callback
from tldw_chatbook.Audio.duplex_contracts import (
    AecDelayEvidence,
    AudioFrame,
    DuplexMode,
    RenderSubmission,
)
from tldw_chatbook.Audio.rolling_transcript import TranscriptEngine, TranscriptRevision
from tldw_chatbook.Audio.voice_preprocessor import VoicePreprocessor
from tldw_chatbook.Audio.voice_process_types import (
    ControlKind,
    VoiceTerminalDisposition,
)
from tldw_chatbook.Audio.voice_turn_coordinator import (
    AttemptGenerationCompleted,
    AttemptPlaybackBoundaryKnown,
    AttemptPlaybackStarted,
    SpeculativeTurnCoordinator,
)

driver = native_helpers.driver
native_transport = native_helpers.native_transport
portaudio = native_helpers.portaudio


def extracted(name):
    try:
        return importlib.import_module(f"tldw_chatbook.Audio.{name}")
    except ModuleNotFoundError:
        pytest.fail(f"missing extracted Audio.{name}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"stt_device": None, "stt_compute_type": None, "stt_precision": None},
        {"stt_device": "cpu", "stt_compute_type": "float32", "stt_precision": "fp32"},
    ],
)
async def test_production_child_prepares_typed_local_stt_without_opening_audio(
    monkeypatch, options
):
    from Tests.Chat.test_console_voice_process import bootstrap, console
    from tldw_chatbook.Audio import (
        duplex_transport,
        parakeet_voice_worker,
        voice_transcription,
        voice_preprocessor,
    )
    from tldw_chatbook.Audio.voice_process_entry import _ProductionSession
    from tldw_chatbook.Audio.voice_process_protocol import Record
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    captured = []

    class LocalStt:
        cleanup_outcome = AttemptCleanupOutcome.CLEAN

        def __init__(self, **kwargs):
            captured.append(kwargs)

        def close(self):
            pass

    async def prepare(service, **kwargs):
        assert isinstance(service, LocalStt)
        return voice_transcription._UNPREPARED_STREAMING_CANDIDATE, False

    transport = Transport()
    monkeypatch.setattr(parakeet_voice_worker, "LocalVoiceSttProcess", LocalStt)
    monkeypatch.setattr(voice_transcription, "_prepare_streaming_candidate", prepare)
    monkeypatch.setattr(duplex_transport, "DuplexAudioTransport", lambda: transport)
    monkeypatch.setattr(
        voice_preprocessor, "create_webrtc_vad", lambda **kw: lambda *args: False
    )
    header = dict(bootstrap(console()).header, aec_enabled=False, **options)
    session = _ProductionSession(Record(header))
    session.bind_pipe(
        SimpleNamespace(generation=7, request_id="a" * 32), lambda _: None
    )
    assert await session.prepare() is False
    assert captured[0]["provider"] == "faster-whisper"
    actual = captured[0]["options"]
    assert (actual.device, actual.compute_type, actual.precision) == tuple(
        options.values()
    )
    assert transport.started == 0
    native, resources = session.begin_close()
    assert await native
    assert await resources is AttemptCleanupOutcome.CLEAN


def frame(sequence):
    return AudioFrame(
        sequence,
        sequence * 10_000_000,
        (sequence + 1) * 10_000_000,
        b"\x01\x00" * 480,
        assistant_rendering=False,
    )


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


class Transport:
    clock_generation = 0
    native_counters = {}

    def __init__(self):
        self.captures = deque()
        self.started = 0
        self.close_requested = 0
        self.closed = 0
        self.rendered = []

    async def start(self):
        self.started += 1

    def request_close(self):
        self.close_requested += 1

    async def close(self):
        self.closed += 1

    async def abort_output(self):
        pass

    def queue_render(self, pcm):
        self.rendered.append(pcm)
        return RenderSubmission(0, 0, len(self.rendered))

    def fence_output(self):
        self.rendered.clear()

    def pop_control_event(self):
        return None

    def pop_capture(self):
        return self.captures.popleft() if self.captures else None

    def pop_render_reference(self):
        return None

    def acknowledge_capture(self, *_args, **_kwargs):
        pass


class Transcript:
    def __init__(self, turn_id, publish):
        self.turn_id = turn_id
        self.publish = publish
        self.frames = []
        self.sealed = []
        self.closed = False

    def append_admitted_frame(self, item):
        self.frames.append(item)
        self.publish(self.revision())

    def revision(self):
        return TranscriptRevision(
            self.turn_id,
            len(self.frames),
            "",
            "available draft",
            self.frames[-1].ended_ns,
            "live",
        )

    async def seal_through(self, sequence):
        if sequence not in {frame.sequence for frame in self.frames}:
            raise ValueError("transcript does not own sequence")
        self.sealed.append(sequence)
        return self.revision()

    async def close(self):
        self.closed = True


class Effects(_FakeEffects):
    def __init__(self):
        super().__init__()
        self.fence_calls = 0
        self.submission = None

    def fence_all(self):
        self.fence_calls += 1

    def final_render_submission(self, _epoch):
        return self.submission

    def promote(self, **kwargs):
        super().promote(**kwargs)
        return VoiceTerminalDisposition.PROMOTED


def core(*, transport=None, effects=None, **kwargs):
    module = extracted("voice_process_core")
    scheduler = _FakeScheduler()
    transport = transport or Transport()
    effects = effects or Effects()
    session = module.ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **callbacks: VoicePreprocessor(
            aec=None, vad=lambda _frame: True, **callbacks
        ),
        transcript_factory=Transcript,
        effects=effects,
        coordinator_factory=lambda **options: SpeculativeTurnCoordinator(
            scheduler=scheduler, **options
        ),
        clock=lambda: scheduler.now_ns,
        **kwargs,
    )
    return session, transport, effects, scheduler


@pytest.mark.asyncio
async def test_transcription_preparation_is_separate_from_audio_start():
    calls = []

    async def prepare():
        calls.append("prepare")

    session, transport, _, _ = core(prepare_transcription=prepare)
    try:
        await session.prepare()
        await session.prepare()
        assert calls == ["prepare"]
        assert transport.started == 0
        assert session._pump_task is None
        await session.start_audio()
        assert transport.started == 1
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_close_during_blocked_prepare_prevents_audio_start():
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare():
        entered.set()
        await release.wait()

    session, transport, _, _ = core(prepare_transcription=prepare)
    starting = asyncio.create_task(session.start_audio())
    await entered.wait()
    await session.fence_and_close(ControlKind.TEARDOWN)
    release.set()
    with pytest.raises(RuntimeError, match="closed"):
        await starting
    assert transport.started == 0
    assert session._pump_task is None
    assert not session._prepared


@pytest.mark.asyncio
async def test_close_between_prepare_queue_and_callback_prevents_invocation():
    calls = 0

    async def prepare():
        nonlocal calls
        calls += 1

    session, transport, _, _ = core(prepare_transcription=prepare)
    preparing = asyncio.create_task(session.prepare())
    await asyncio.sleep(0)
    assert calls == 0
    await session.fence_and_close(ControlKind.TEARDOWN)
    with pytest.raises(RuntimeError, match="closed"):
        await preparing
    assert calls == 0
    assert not session._prepared
    assert transport.started == 0


@pytest.mark.asyncio
async def test_start_audio_owns_pump_and_drains_capture():
    session, transport, _, _ = core()
    transport.captures.append(frame(0))
    await session.start_audio()
    await eventually(lambda: bool(session._transcripts))
    assert [item.sequence for item in session._transcripts[0].frames] == [0]
    await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_concurrent_start_audio_shares_prepare_start_and_pump_owner():
    prepare_started, release_prepare = asyncio.Event(), asyncio.Event()
    prepare_calls = 0

    async def prepare():
        nonlocal prepare_calls
        prepare_calls += 1
        prepare_started.set()
        await release_prepare.wait()

    session, transport, _, _ = core(prepare_transcription=prepare)
    first = asyncio.create_task(session.start_audio())
    await prepare_started.wait()
    second = asyncio.create_task(session.start_audio())
    release_prepare.set()
    await asyncio.gather(first, second)
    assert prepare_calls == 1
    assert transport.started == 1
    assert session._pump_task is not None
    assert (
        len(
            [
                task
                for task in asyncio.all_tasks()
                if task.get_name() == "console-speculative-voice-audio"
            ]
        )
        == 1
    )
    await session.start_audio()
    assert transport.started == 1
    await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_concurrent_prepare_and_start_share_cancel_safe_owner():
    prepare_started, release_prepare = asyncio.Event(), asyncio.Event()
    prepare_calls = 0

    async def prepare():
        nonlocal prepare_calls
        prepare_calls += 1
        prepare_started.set()
        await release_prepare.wait()

    session, transport, _, _ = core(prepare_transcription=prepare)
    preparing = asyncio.create_task(session.prepare())
    await prepare_started.wait()
    starting = asyncio.create_task(session.start_audio())
    preparing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparing
    release_prepare.set()
    await starting
    assert prepare_calls == 1
    assert transport.started == 1
    await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_prepare_then_close_refuses_later_audio_start():
    session, transport, _, _ = core()
    await session.prepare()
    await session.fence_and_close(ControlKind.TEARDOWN)
    with pytest.raises(RuntimeError, match="closed"):
        await session.start_audio()
    assert transport.started == 0


@pytest.mark.asyncio
async def test_close_while_transport_starts_cannot_publish_ready_or_start_pump():
    started, release = asyncio.Event(), asyncio.Event()
    diagnostics = []

    class BlockingStartTransport(Transport):
        async def start(self):
            started.set()
            await release.wait()
            self.started += 1

    transport = BlockingStartTransport()
    session, _, _, _ = core(
        transport=transport,
        diagnostic_sink=lambda event, fields: diagnostics.append((event, fields)),
    )
    starting = asyncio.create_task(session.start_audio())
    await started.wait()
    await session.fence_and_close(ControlKind.TEARDOWN)
    release.set()
    with pytest.raises(RuntimeError, match="closed"):
        await starting
    assert session._pump_task is None
    assert not any(event == "session_ready" for event, _fields in diagnostics)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_start", [False, True])
async def test_enter_awaits_and_shares_blocked_start_audio_owner(fail_start):
    started, release = asyncio.Event(), asyncio.Event()

    class BlockingStartTransport(Transport):
        async def start(self):
            started.set()
            await release.wait()
            if fail_start:
                raise RuntimeError("start_failed")
            self.started += 1

    transport = BlockingStartTransport()
    session, _, _, _ = core(transport=transport)
    starting = asyncio.create_task(session.start_audio())
    await started.wait()
    entering = asyncio.create_task(session.enter(capture_live=False))
    await asyncio.sleep(0)
    assert not entering.done()
    release.set()
    if fail_start:
        with pytest.raises(RuntimeError, match="start_failed"):
            await starting
        with pytest.raises(RuntimeError, match="start_failed"):
            await entering
    else:
        await asyncio.gather(starting, entering)
        assert transport.started == 1
        assert session._pump_task is not None
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_admitted_frames_dispatch_then_resume_fences_same_turn():
    session, transport, effects, scheduler = core()
    try:
        assert session.coordinator.snapshot.duplex_mode is DuplexMode.HALF_DUPLEX
        transport.captures.append(frame(0))
        await session.process_pending_audio()
        await session.coordinator.flush()
        turn = session.coordinator.snapshot.turn_id
        scheduler.advance_ms(710)
        await session.coordinator.flush()
        assert effects.dispatches == [(turn, 1, "available draft")]
        transport.captures.append(frame(1))
        await session.process_pending_audio()
        assert effects.fenced_epochs == {1}
        assert session.coordinator.snapshot.turn_id == turn
        assert len(session._transcripts) == 1
        assert [item.sequence for item in session._transcripts[0].frames] == [0, 1]
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_first", [False, True])
async def test_exact_native_terminal_preserves_two_transcripts_and_seal_order(
    native_transport, driver, terminal_first
):
    await native_transport.start()
    bridge = native_transport._stream.bridge
    emit_callback(driver, bridge, times=(10.0, 10.01, 10.02))
    native_transport.pop_capture()
    native_transport.acknowledge_capture(
        0, clock_generation=0, dsp_ok=True, vad_ok=True
    )
    offset = native_transport._device_to_monotonic_offset_ns
    session, _, effects, scheduler = core(transport=native_transport)
    try:
        await session._accept_admitted_frame(frame(0), assistant_rendering=False)
        await session.coordinator.flush()
        scheduler.advance_ms(710)
        await session.coordinator.flush()
        turn, epoch, _ = effects.dispatches[0]
        effects.submission = native_transport.queue_render(b"\x01\x00" * 480)
        await session.submit(AttemptPlaybackStarted(epoch))
        session._preprocessor = VoicePreprocessor(
            aec=None,
            vad=lambda item: item.sequence >= 5,
            on_admitted_frame=session._on_admitted_frame,
            on_processed=session._on_processed,
            on_classification_changed=session._on_classification_changed,
        )
        for index in range(1, 5):
            emit_callback(
                driver,
                bridge,
                times=(10 + index * 0.01, 10.01 + index * 0.01, 10.02 + index * 0.01),
            )
            scheduler.now_ns = offset + 10_025_000_000
            await session.process_pending_audio()
        scheduler.now_ns = offset + 10_050_000_000
        if terminal_first:
            await session.process_pending_audio()
        emit_callback(driver, bridge, times=(10.05, 10.06, 10.07))
        await session.process_pending_audio()
        await session.coordinator.flush()
        assert session._playback_target.boundary.ended_ns == offset + 10_040_000_000
        assert session._playback_target.terminal_sent
        assert session.coordinator.snapshot.turn_id == turn
        assert len(session._transcripts) == 2
        assert (
            session._transcripts[1].turn_id
            == session.coordinator.snapshot.pending_next_turn_id
        )
        assert session._transcripts[0].sealed == []
        assert effects.promotions == []
        await session.submit(AttemptGenerationCompleted(epoch, "answer"))
        await session.coordinator.flush()
        assert session._transcripts[0].sealed == [0]
        assert len(effects.promotions) == 1
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_retired_transcript_state_stays_bounded_without_sequence_index():
    session, _, _, _ = core()
    for index in range(100):
        session._transcript_for(f"turn-{index}")
    await asyncio.sleep(0)
    assert len(session._transcripts) == 2
    assert not hasattr(session, "_transcript_turn_by_sequence")
    await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_capture", [False, True])
async def test_permanent_transport_fault_notifies_before_stalled_preservation(
    with_capture,
):
    effects = Effects()
    preserve_started, release_preserve = asyncio.Event(), asyncio.Event()
    runtime_failures = []

    async def preserve(**kwargs):
        assert effects.fence_calls == 1
        preserve_started.set()
        await release_preserve.wait()
        effects.drafts.append(kwargs)

    effects.preserve_draft = preserve
    session, transport, _, scheduler = core(
        effects=effects, on_runtime_failure=runtime_failures.append
    )
    try:
        transport.captures.append(frame(0))
        await session.process_pending_audio()
        await session.coordinator.flush()
        transport.native_counters = {"capture_overflows": 1}
        if with_capture:
            transport.captures.append(
                replace(
                    frame(1),
                    discontinuity=True,
                    delay_evidence=AecDelayEvidence(
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        True,
                        timing_discontinuity=True,
                        clock_drift=False,
                    ),
                )
            )
        assert await session.process_pending_audio()
        assert session._fenced
        assert effects.fence_calls == 1
        assert not await session.submit(AttemptGenerationCompleted(1, "late"))
        await preserve_started.wait()
        await eventually(lambda: transport.closed == 1)
        assert len(runtime_failures) == 1
        scheduler.advance_ms(2000)
        await session.coordinator.flush()
        assert effects.dispatches == []
        assert not await session.process_pending_audio()
        release_preserve.set()
        await eventually(lambda: bool(runtime_failures))
        await eventually(lambda: bool(effects.drafts))
        assert len(runtime_failures) == 1
        assert runtime_failures[0].code == "audio_transport_failed"
        assert len(effects.drafts) == 1
        assert effects.drafts[0]["transcript"] == "available draft"
        assert effects.fence_calls == 1
    finally:
        release_preserve.set()
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_aec_only_degradation_keeps_healthy_half_duplex_functional():
    session, transport, effects, scheduler = core()
    try:
        transport.captures.extend([frame(0), frame(1)])
        while await session.process_pending_audio():
            pass
        await session.coordinator.flush()
        scheduler.advance_ms(720)
        await session.coordinator.flush()
        assert len(effects.dispatches) == 1
        assert not session._fenced
        assert session.coordinator.snapshot.duplex_mode is DuplexMode.HALF_DUPLEX
    finally:
        await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_transport_failure_reports_even_when_draft_preservation_raises():
    effects = Effects()
    failures = []

    async def preserve(**_kwargs):
        raise RuntimeError("private draft persistence detail")

    effects.preserve_draft = preserve
    session, transport, _, _ = core(effects=effects, on_runtime_failure=failures.append)
    transport.captures.append(frame(0))
    await session.process_pending_audio()
    await session.coordinator.flush()
    transport.native_counters = {"capture_overflows": 1}
    assert await session.process_pending_audio()
    await eventually(lambda: bool(failures))
    assert [failure.code for failure in failures] == ["audio_transport_failed"]
    await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_transport_failure_preserves_pending_next_draft_not_prior_turn():
    effects = Effects()
    session, transport, _, scheduler = core(effects=effects)
    await session._accept_admitted_frame(frame(0), assistant_rendering=False)
    await session.coordinator.flush()
    prior_turn = session.coordinator.snapshot.turn_id
    scheduler.advance_ms(710)
    await session.coordinator.flush()
    epoch = effects.dispatches[0][1]
    await session.submit(AttemptPlaybackStarted(epoch))
    await session.submit(AttemptPlaybackBoundaryKnown(epoch, 5_000_000))
    await session._accept_admitted_frame(frame(1), assistant_rendering=True)
    pending_turn = session.coordinator.snapshot.pending_next_turn_id
    assert pending_turn is not None and pending_turn != prior_turn
    # The coordinator owns transcript classification; this unit targets the
    # core's choice between its two already-projected drafts.
    session._coordinator._pending_next_transcript_text = "pending draft"

    transport.native_counters = {"capture_overflows": 1}
    assert await session.process_pending_audio()
    await eventually(lambda: bool(effects.drafts))
    assert effects.drafts == [(pending_turn, "pending draft", "audio_transport_failed")]
    await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_post_fence_authority_rejected_but_claimed_promotion_survives():
    session, _, effects, _ = core()
    claimed = asyncio.Event()
    release = asyncio.Event()

    async def promote(**_kwargs):
        claimed.set()
        await release.wait()
        return VoiceTerminalDisposition.PROMOTED

    effects.promote = promote
    effects.submit_accepted_voice_turn = lambda *_args, **_kwargs: None
    in_flight = session._effects.promote(
        turn_id="turn",
        attempt_epoch=1,
        transcript="draft",
        assistant_text="answer",
        terminal_boundary_ns=None,
    )
    promotion = asyncio.create_task(in_flight)
    await claimed.wait()
    session._fence_callbacks()

    with pytest.raises(Exception, match="audio_transport_failed"):
        session._effects.promote(
            turn_id="turn",
            attempt_epoch=2,
            transcript="draft",
            assistant_text="answer",
            terminal_boundary_ns=None,
        )
    with pytest.raises(Exception, match="audio_transport_failed"):
        session._effects.submit_accepted_voice_turn("draft", object())
    release.set()
    assert await promotion is VoiceTerminalDisposition.PROMOTED
    await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_native_stt_capacity_is_terminal_with_bounded_inflight_work():
    module = extracted("voice_transcription")
    entered, release = threading.Event(), threading.Event()

    def process(_pcm):
        entered.set()
        release.wait(2)
        return {"partial": "available draft"}

    native = module._NativeStreamingStt(
        object(),
        provider="fake",
        model=None,
        language="en",
        prepared_candidate=SimpleNamespace(process_audio=process),
    )
    fallback_calls = []

    class Fallback:
        async def transcribe_window(self, **kwargs):
            fallback_calls.append(kwargs)
            raise AssertionError("capacity exhaustion must not replay incomplete PCM")

        async def abort(self):
            pass

    revisions = []
    engine = TranscriptEngine(
        turn_id="turn",
        live_adapter=native,
        fallback_adapter=Fallback(),
        on_revision=revisions.append,
    )
    try:
        engine.append_admitted_frame(frame(0))
        assert await asyncio.to_thread(entered.wait, 1)
        tasks_before = len(asyncio.all_tasks())
        for sequence in range(1, 1002):
            engine.append_admitted_frame(frame(sequence))
        await eventually(lambda: any(item.failure_code for item in revisions))
        assert len(asyncio.all_tasks()) <= tasks_before + 1
        assert native._frames.qsize() <= 1000
        assert fallback_calls == []
        assert [item.failure_code for item in revisions if item.failure_code] == [
            "backend_failed"
        ]
        assert native.failure.code == "transcript_capacity_exceeded"
    finally:
        release.set()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during_cleanup", [False, True])
async def test_normalized_pcm_cleanup_receipt_blocks_next_phrase(cancel_during_cleanup):
    module = extracted("voice_phrase_sequencer")
    types = extracted("voice_process_types")
    cleanup_started, cleanup_release = asyncio.Event(), asyncio.Event()
    calls, rendered = [], []

    async def cleanup():
        cleanup_started.set()
        await cleanup_release.wait()

    class Synthesizer:
        async def synthesize_hands_free(self, *, text):
            calls.append(text)

            async def frames():
                yield bytes(960)

            return types.NormalizedPcmStream(frames(), cleanup())

    class Sink:
        def queue_render(self, pcm):
            rendered.append(pcm)
            return RenderSubmission(0, 0, len(rendered))

        def fence_output(self):
            rendered.clear()

    sequencer = module.PhraseSpeechSequencer(
        epoch=1, synthesizer=Synthesizer(), sink=Sink()
    )
    try:
        await sequencer.feed(1, "First phrase. Second phrase. ")
        await sequencer.finish(1)
        await cleanup_started.wait()
        assert calls == ["First phrase."]
        assert rendered == [bytes(960)]
        assert sequencer.final_submission is None
        if cancel_during_cleanup:
            await sequencer.cancel(1)
            assert sequencer.supervised_cleanup_count == 1
            assert sequencer.synthesis_in_flight
        cleanup_release.set()
        if cancel_during_cleanup:
            await sequencer.wait_for_cleanup()
            assert calls == ["First phrase."]
            assert sequencer.final_submission is None
        else:
            await eventually(lambda: sequencer.final_submission is not None)
            assert calls == ["First phrase.", "Second phrase."]
    finally:
        cleanup_release.set()
        await sequencer.cancel(1)
        await sequencer.wait_for_cleanup()


@pytest.mark.asyncio
async def test_normalized_pcm_retries_temporary_ring_pressure():
    module = extracted("voice_phrase_sequencer")
    types = extracted("voice_process_types")
    accepted = []
    attempts = 0

    class Synthesizer:
        async def synthesize_hands_free(self, *, text):
            async def frames():
                yield b"\x01\x00" * 480

            return types.NormalizedPcmStream(frames(), asyncio.sleep(0))

    class Sink:
        def queue_render(self, pcm):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return None
            accepted.append(pcm)
            return RenderSubmission(0, 0, 1)

        def fence_output(self):
            pass

    sequencer = module.PhraseSpeechSequencer(
        epoch=1, synthesizer=Synthesizer(), sink=Sink()
    )
    await sequencer.feed(1, "Hello. ")
    await sequencer.finish(1)
    await eventually(lambda: sequencer.final_submission is not None)
    assert attempts == 3
    assert accepted == [b"\x01\x00" * 480]
    assert sequencer.failure_code is None


@pytest.mark.asyncio
async def test_normalized_pcm_rejects_non_ten_millisecond_frame_and_cleans_up():
    module = extracted("voice_phrase_sequencer")
    types = extracted("voice_process_types")
    cleaned = asyncio.Event()

    async def cleanup():
        cleaned.set()

    class Synthesizer:
        async def synthesize_hands_free(self, *, text):
            async def frames():
                yield bytes(959)

            return types.NormalizedPcmStream(frames(), cleanup())

    class Sink:
        def queue_render(self, pcm):
            raise AssertionError("invalid PCM must not reach the native sink")

        def fence_output(self):
            pass

    sequencer = module.PhraseSpeechSequencer(
        epoch=1, synthesizer=Synthesizer(), sink=Sink()
    )
    await sequencer.feed(1, "Hello. ")
    await sequencer.finish(1)
    await eventually(lambda: sequencer.failure_code is not None)
    assert sequencer.failure_code == "synthesis_failed"
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_cancel_during_iterator_close_retains_close_and_cleanup_receipts():
    module = extracted("voice_phrase_sequencer")
    types = extracted("voice_process_types")
    close_started, close_release = asyncio.Event(), asyncio.Event()
    cleanup_started, cleanup_release = asyncio.Event(), asyncio.Event()

    class Frames:
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return bytes(960)

        async def aclose(self):
            close_started.set()
            await close_release.wait()

    async def cleanup():
        cleanup_started.set()
        await cleanup_release.wait()

    class Synthesizer:
        async def synthesize_hands_free(self, *, text):
            return types.NormalizedPcmStream(Frames(), cleanup())

    sequencer = module.PhraseSpeechSequencer(
        epoch=1, synthesizer=Synthesizer(), sink=Transport()
    )
    await sequencer.feed(1, "Hello. ")
    await sequencer.finish(1)
    await asyncio.wait_for(close_started.wait(), 1)
    await sequencer.cancel(1)
    assert sequencer.supervised_cleanup_count == 1
    close_release.set()
    await asyncio.wait_for(cleanup_started.wait(), 1)
    waiting = asyncio.create_task(sequencer.wait_for_cleanup())
    await asyncio.sleep(0)
    assert not waiting.done()
    cleanup_release.set()
    await asyncio.wait_for(waiting, 1)


@pytest.mark.asyncio
async def test_iterator_close_error_still_waits_for_independent_cleanup():
    module = extracted("voice_phrase_sequencer")
    types = extracted("voice_process_types")
    cleanup_started, cleanup_release = asyncio.Event(), asyncio.Event()

    class Frames:
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return bytes(960)

        async def aclose(self):
            raise RuntimeError("iterator close failed")

    async def cleanup():
        cleanup_started.set()
        await cleanup_release.wait()

    class Synthesizer:
        async def synthesize_hands_free(self, *, text):
            return types.NormalizedPcmStream(Frames(), cleanup())

    sequencer = module.PhraseSpeechSequencer(
        epoch=1, synthesizer=Synthesizer(), sink=Transport()
    )
    await sequencer.feed(1, "Hello. ")
    await sequencer.finish(1)
    await asyncio.wait_for(cleanup_started.wait(), 1)
    assert sequencer.failure_code is None
    cleanup_release.set()
    await eventually(lambda: sequencer.failure_code == "synthesis_failed")
