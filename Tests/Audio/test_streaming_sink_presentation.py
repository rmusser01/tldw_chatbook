"""Device output, natural completion and bounded portrait observation contracts."""

import asyncio
import struct
import threading
from types import SimpleNamespace

import pytest

from Tests.Audio.test_streaming_sink import (
    BLOCK_MS,
    FRAMES,
    FakeStream,
)
from Tests.Audio.test_streaming_sink import (
    _reset_live_sink_registry as _reset_live_sink_registry,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.Audio.speech_presentation import (
    SpeechPresentation,
    SpeechPresentationOwner,
)
from tldw_chatbook.Audio.streaming_sink import (
    SinkDrained,
    SinkStopped,
    StreamingPcmSink,
)


class NotifyingPresentation(SpeechPresentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.changed = threading.Condition()

    def publish(self, snapshot):
        super().publish(snapshot)
        with self.changed:
            self.changed.notify_all()

    def wait_for(self, predicate):
        with self.changed:
            assert self.changed.wait_for(
                lambda: self.read() is not None and predicate(self.read()), timeout=2
            ), self.read()


class DelayedStream(FakeStream):
    """A native device acknowledges completion separately from buffer submission."""

    def __init__(self, *, finished_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.finished_callback = finished_callback
        self.auto_finish = False
        self.closed = False

    def finish_output(self):
        assert self.finished_callback is not None
        self.finished_callback()

    def close(self):
        self.closed = True


@pytest.fixture
def device_case():
    sinks = []

    def make(*, observed=False):
        holder = {}
        events = []
        clock = [10.0]
        owner = SpeechPresentationOwner(
            "device", object(), "session", 1, 7, "message", None
        )
        speech = NotifyingPresentation(owner, clock=lambda: clock[0])

        def factory(**kwargs):
            holder["stream"] = DelayedStream(**kwargs)
            return holder["stream"]

        kwargs = {"presentation": speech, "clock": lambda: clock[0]} if observed else {}
        sink = StreamingPcmSink(
            on_event=events.append,
            stream_factory=factory,
            blocksize_ms=BLOCK_MS,
            **kwargs,
        )
        sink.open(24000, 1)
        sinks.append(sink)
        return sink, holder["stream"], events, speech, clock

    yield make
    for sink in sinks:
        sink.stop()
        sink.settle()


@pytest.mark.parametrize("observed", [False, True])
def test_last_submitted_block_is_not_successful_drain_until_device_finishes(
    device_case, observed
):
    sink, stream, events, _speech, _clock = device_case(observed=observed)
    assert sink.feed(struct.pack(f"<{FRAMES}h", *([16384] * FRAMES)))
    sink.close()
    stream.tick()
    assert sink.state == "draining"
    assert sink.terminal_reason is None
    assert not stream.aborted and not stream.closed
    assert not any(isinstance(event, SinkDrained) for event in events)
    stream.finish_output()
    assert sink.settle()
    assert sink.terminal_reason == "drained" and stream.closed and not stream.aborted
    assert sum(isinstance(event, SinkDrained) for event in events) == 1
    stream.finish_output()
    assert sink.settle()
    assert sum(isinstance(event, SinkDrained) for event in events) == 1


def test_stop_wins_over_a_late_native_finished_acknowledgement(device_case):
    sink, stream, events, _speech, _clock = device_case()
    assert sink.feed(struct.pack(f"<{FRAMES}h", *([8192] * FRAMES)))
    sink.close()
    stream.tick()
    assert sink.state == "draining"
    sink.stop()
    stream.finish_output()
    assert sink.settle()
    assert stream.aborted and stream.closed and sink.terminal_reason == "stopped"
    assert sum(isinstance(event, SinkStopped) for event in events) == 1
    assert not any(isinstance(event, SinkDrained) for event in events)


def test_dac_delay_closes_mouth_until_the_scheduled_output_interval(device_case):
    sink, stream, _events, speech, clock = device_case(observed=True)
    assert sink.feed(struct.pack("<24000h", *([16384] * 24000)))
    stream.tick(time_info=SimpleNamespace(currentTime=100.0, outputBufferDacTime=100.2))
    assert sink.settle()
    clock[0] = 10.05
    sink._publish_presentation()
    assert speech.read().mouth == "rest"
    clock[0] = 10.21
    sink._publish_presentation()
    assert speech.read().mouth == "open"


def tick_at(stream, clock, now, delay=0.2):
    clock[0] = now
    current = 100.0 + now - 10.0
    stream.tick(
        time_info=SimpleNamespace(
            currentTime=current, outputBufferDacTime=current + delay
        )
    )


def test_real_notification_loop_publishes_loud_pause_loud_during_pending_device_tail(
    device_case,
):
    sink, stream, _events, speech, clock = device_case(observed=True)
    pcm = b"".join(
        struct.pack(f"<{FRAMES}h", *([value] * FRAMES))
        for value in (16384, 32, 32, 16384)
    )
    assert sink.feed(pcm)
    sink.close()
    for index in range(4):
        tick_at(stream, clock, 10.0 + index * 0.02)
    assert sink.terminal_reason is None
    for now, pose in ((10.21, "open"), (10.25, "rest"), (10.27, "open")):
        clock[0] = now
        speech.wait_for(
            lambda snap, now=now, pose=pose: (
                snap.observed_at >= now and snap.mouth == pose and snap.timing_valid
            )
        )
    clock[0] = 10.4
    speech.wait_for(
        lambda snap: (
            snap.observed_at >= 10.4 and snap.mouth == "rest" and not snap.timing_valid
        )
    )
    stream.finish_output()
    assert sink.settle()
    assert sink.terminal_reason == "drained"


def test_partial_block_zero_suffix_closes_without_a_full_underrun_event(device_case):
    from tldw_chatbook.Audio.streaming_sink import SinkUnderrun

    sink, stream, events, speech, clock = device_case(observed=True)
    assert sink.feed(
        struct.pack(f"<{FRAMES + FRAMES // 2}h", *([16384] * (FRAMES + FRAMES // 2)))
    )
    sink.close()
    tick_at(stream, clock, 10.0)
    tick_at(stream, clock, 10.02)
    clock[0] = 10.225
    speech.wait_for(lambda snap: snap.observed_at >= 10.225 and snap.mouth == "open")
    clock[0] = 10.235
    speech.wait_for(
        lambda snap: (
            snap.observed_at >= 10.235 and snap.mouth == "rest" and snap.timing_valid
        )
    )
    assert not any(isinstance(event, SinkUnderrun) for event in events)


def test_split_sample_is_withheld_until_its_second_byte_reaches_the_device(device_case):
    sink, stream, _events, _speech, clock = device_case(observed=True)
    assert sink.feed(struct.pack(f"<{FRAMES * 16}h", *([8192] * FRAMES * 16)))
    for index in range(16):
        tick_at(stream, clock, 10.0 + index * 0.02)
    assert sink.feed(b"\x00")
    tick_at(stream, clock, 10.32)
    assert stream.out[-1] == b"\x00" * FRAMES * 2
    remaining = b"\x40" + struct.pack(f"<{FRAMES - 1}h", *([16384] * (FRAMES - 1)))
    assert sink.feed(remaining)
    tick_at(stream, clock, 10.34)
    assert stream.out[-1] == struct.pack(f"<{FRAMES}h", *([16384] * FRAMES))


def test_invalid_dac_timing_keeps_audio_and_native_completion_working(device_case):
    sink, stream, _events, speech, _clock = device_case(observed=True)
    assert sink.feed(struct.pack(f"<{FRAMES}h", *([16384] * FRAMES)))
    sink.close()
    stream.tick(
        time_info=SimpleNamespace(currentTime=100.0, outputBufferDacTime=float("nan"))
    )
    speech.wait_for(lambda snap: snap.mouth == "rest" and not snap.timing_valid)
    assert stream.out[-1] == struct.pack(f"<{FRAMES}h", *([16384] * FRAMES))
    assert sink.terminal_reason is None
    stream.finish_output()
    assert sink.settle() and sink.terminal_reason == "drained"


def test_full_underrun_advances_device_time_without_skipping_resumed_source(
    device_case,
):
    sink, stream, _events, speech, clock = device_case(observed=True)
    initial = struct.pack(f"<{FRAMES * 16}h", *([16384] * FRAMES * 16))
    assert sink.feed(initial)
    speech.wait_for(lambda _: sink._observer.input_bytes == len(initial))
    for index in range(16):
        tick_at(stream, clock, 10.0 + index * 0.02)
    for index in range(16, 41):
        tick_at(stream, clock, 10.0 + index * 0.02)
    speech.wait_for(lambda snap: snap.mouth == "rest" and snap.timing_valid)
    assert sink.feed(struct.pack(f"<{FRAMES}h", *([16384] * FRAMES)))
    sink.close()
    speech.wait_for(lambda _: sink._observer.input_bytes == len(initial) + FRAMES * 2)
    tick_at(stream, clock, 10.82)
    clock[0] = 11.03
    speech.wait_for(
        lambda snap: (
            snap.observed_at >= 11.03 and snap.mouth == "open" and snap.timing_valid
        )
    )
    assert sink._observer.source_frames == FRAMES * 17


def test_rejected_and_retried_feed_only_analyzes_accepted_bytes(
    device_case, monkeypatch
):
    import tldw_chatbook.Audio.streaming_sink as module

    monkeypatch.setattr(module, "BUFFER_CAP_SECONDS", 1)
    sink, stream, _events, speech, clock = device_case(observed=True)
    original = struct.pack("<24000h", *([8192] * 24000))
    retry = struct.pack(f"<{FRAMES}h", *([32] * FRAMES))
    assert sink.feed(original)
    assert not sink.feed(retry)
    assert not sink.feed(retry)
    speech.wait_for(lambda _: sink._observer.input_bytes == len(original))
    assert (
        sink._observer.windows[-1].start_frame + sink._observer.windows[-1].frame_count
        == 24000
    )
    tick_at(stream, clock, 10.0)
    assert sink.feed(retry)
    speech.wait_for(lambda _: sink._observer.input_bytes == len(original) + len(retry))
    assert sink._observer.windows[-1].start_frame == 24000


@pytest.mark.parametrize(
    "current,dac", [(float("nan"), 100.2), (100.0, 99.9), (100.0, 101.1)]
)
def test_nonfinite_negative_or_excessive_output_horizon_disables_only_observation(
    device_case, current, dac
):
    sink, stream, _events, speech, _clock = device_case(observed=True)
    pcm = struct.pack(f"<{FRAMES}h", *([16384] * FRAMES))
    assert sink.feed(pcm)
    sink.close()
    stream.tick(time_info=SimpleNamespace(currentTime=current, outputBufferDacTime=dac))
    speech.wait_for(lambda snap: not snap.timing_valid and snap.mouth == "rest")
    assert stream.out[-1] == pcm and sink.terminal_reason is None
    stream.finish_output()
    assert sink.settle() and sink.terminal_reason == "drained"


def test_unexpected_synchronous_finish_during_open_does_not_evict_healthy_sink(
    device_case,
):
    healthy, old_stream, _events, _speech, _clock = device_case()

    def factory(**kwargs):
        stream = DelayedStream(**kwargs)
        stream.start = stream.finish_output
        return stream

    failed = StreamingPcmSink(on_event=lambda _: None, stream_factory=factory)
    try:
        failed.open(24000, 1)
        failed.settle()
        assert failed.state == "failed"
        assert healthy.terminal_reason is None and not old_stream.aborted
    finally:
        failed.stop()


def test_blocked_listener_keeps_handoff_bounded_and_stop_does_not_join(device_case):
    import numpy as np

    from tldw_chatbook.Audio.streaming_sink import SinkUnderrun

    sink, stream, events, speech, clock = device_case(observed=True)
    initial = struct.pack(f"<{FRAMES * 16}h", *([16384] * FRAMES * 16))
    assert sink.feed(initial)
    speech.wait_for(lambda _: sink._observer.input_bytes == len(initial))
    for index in range(16):
        tick_at(stream, clock, 10.0 + index * 0.02)
    clock[0] = 10.31
    speech.wait_for(lambda snap: snap.observed_at >= 10.31 and snap.mouth == "open")
    entered = threading.Event()
    release = threading.Event()

    def block_underrun(event):
        events.append(event)
        if isinstance(event, SinkUnderrun):
            entered.set()
            assert release.wait(timeout=5)

    sink._emit_cb = block_underrun
    out = np.zeros((FRAMES, 1), dtype=np.int16)
    try:
        clock[0] = 10.32
        sink._callback(
            out,
            FRAMES,
            SimpleNamespace(currentTime=100.32, outputBufferDacTime=100.52),
            None,
        )
        assert entered.wait(timeout=2)
        clock[0] = 10.5
        assert speech.read().mouth == "rest" and not speech.read().timing_valid
        for index in range(200):
            clock[0] = 10.52 + index * 0.02
            current = 100.52 + index * 0.02
            sink._callback(
                out,
                FRAMES,
                SimpleNamespace(currentTime=current, outputBufferDacTime=current + 0.2),
                None,
            )
        assert sink._notify_q.qsize() <= 2
        assert len(sink._observer.raw) <= sink._observer.raw_cap
        stopped = threading.Event()
        caller = threading.Thread(target=lambda: (sink.stop(), stopped.set()))
        caller.start()
        assert stopped.wait(timeout=0.5)
        caller.join(timeout=1)
        assert stream.aborted and speech.read().mouth == "rest"
    finally:
        release.set()
        assert sink.settle()


@pytest.mark.asyncio
async def test_missing_native_finish_acknowledgement_has_bounded_pump_failure(
    device_case, monkeypatch
):
    import tldw_chatbook.Audio.streaming_sink as module

    monkeypatch.setattr(module, "_DRAIN_WAIT_MARGIN_SECONDS", 0.02)
    sink, stream, _events, _speech, _clock = device_case()

    async def chunks():
        yield struct.pack(f"<{FRAMES}h", *([8192] * FRAMES))

    task = asyncio.create_task(module.pump(sink, chunks()))

    async def wait_for_close():
        while sink.state == "open":
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_close(), timeout=1)
    stream.tick()
    result = await asyncio.wait_for(task, timeout=1)
    assert result.outcome == "failed" and "drain wait" in result.reason
    assert stream.aborted and stream.closed


def test_observer_analysis_failure_cannot_change_audio_or_completion(
    device_case, monkeypatch
):
    import tldw_chatbook.Audio.streaming_sink as module

    def broken_analysis(*args):
        raise RuntimeError("observer analysis failed")

    monkeypatch.setattr(module.PcmEnvelope, "push", broken_analysis)
    sink, stream, _events, speech, clock = device_case(observed=True)
    pcm = struct.pack(f"<{FRAMES}h", *([16384] * FRAMES))
    assert sink.feed(pcm)
    sink.close()
    tick_at(stream, clock, 10.0)
    speech.wait_for(lambda snap: snap.mouth == "rest" and not snap.timing_valid)
    assert stream.out[-1] == pcm and sink.terminal_reason is None
    stream.finish_output()
    assert sink.settle() and sink.terminal_reason == "drained"


def test_production_backend_wrapper_requests_graceful_completion_before_native_ack(
    monkeypatch,
):
    import numpy as np

    import tldw_chatbook.Audio.streaming_sink as module

    class CallbackStop(Exception):
        pass

    holder = {}

    def output_stream(**kwargs):
        kwargs.pop("dtype")
        holder["stream"] = DelayedStream(**kwargs)
        return holder["stream"]

    backend = SimpleNamespace(OutputStream=output_stream, CallbackStop=CallbackStop)
    monkeypatch.setattr(module, "_import_sounddevice", lambda: backend)
    sink = module.StreamingPcmSink(on_event=lambda _: None)
    try:
        sink.open(24000, 1)
        sink.feed(struct.pack(f"<{FRAMES}h", *([8192] * FRAMES)))
        sink.close()
        stream = holder["stream"]
        out = np.zeros((FRAMES, 1), dtype=np.int16)
        with pytest.raises(CallbackStop):
            stream.callback(out, FRAMES, None, None)
        assert sink.state == "draining" and sink.terminal_reason is None
        assert not stream.aborted
        stream.finish_output()
        assert sink.settle()
        assert (
            sink.terminal_reason == "drained" and stream.closed and not stream.aborted
        )
    finally:
        sink.stop()


@pytest.mark.parametrize("invalid", [None, "not PCM", bytearray(b"xx")])
def test_invalid_pcm_admission_is_rejected_without_mutating_audio(device_case, invalid):
    sink, stream, _events, _speech, _clock = device_case()
    assert sink.feed(invalid) is False
    pcm = struct.pack(f"<{FRAMES}h", *([8192] * FRAMES))
    assert sink.feed(pcm)
    sink.close()
    stream.tick()
    assert stream.out[-1] == pcm


def test_hardware_underflow_flag_invalidates_observation_without_dropping_audio(
    device_case,
):
    import numpy as np

    sink, _stream, _events, speech, _clock = device_case(observed=True)
    pcm = struct.pack(f"<{FRAMES}h", *([16384] * FRAMES))
    assert sink.feed(pcm)
    sink.close()
    out = np.zeros((FRAMES, 1), dtype=np.int16)
    sink._callback(
        out,
        FRAMES,
        SimpleNamespace(currentTime=100.0, outputBufferDacTime=100.2),
        SimpleNamespace(output_underflow=True),
    )
    speech.wait_for(lambda snap: not snap.timing_valid)
    assert sink._observer.unsupported
    assert out.tobytes() == pcm and sink.terminal_reason is None


def test_malformed_final_sample_fails_without_claiming_started_or_drained(device_case):
    from tldw_chatbook.Audio.streaming_sink import SinkFailed, SinkStarted

    sink, stream, events, _speech, _clock = device_case()
    assert sink.feed(b"x")
    sink.close()
    stream.tick()
    assert sink.settle()
    assert sink.terminal_reason == "failed" and stream.aborted
    assert any(isinstance(event, SinkFailed) for event in events)
    assert not any(isinstance(event, (SinkStarted, SinkDrained)) for event in events)


def test_late_old_sink_cleanup_cannot_clear_newer_output_for_same_owner(device_case):
    old, old_stream, _events, speech, clock = device_case(observed=True)
    pcm = struct.pack(f"<{FRAMES}h", *([16384] * FRAMES))
    assert old.feed(pcm)
    old.close()
    tick_at(old_stream, clock, 10.0)
    clock[0] = 10.21
    speech.wait_for(lambda snap: snap.mouth == "open")
    previous_generation = speech.read().generation
    holder = {}

    def factory(**kwargs):
        holder["stream"] = DelayedStream(**kwargs)
        return holder["stream"]

    new = StreamingPcmSink(
        on_event=lambda _: None,
        presentation=speech,
        clock=lambda: clock[0],
        stream_factory=factory,
    )
    try:
        new.open(24000, 1)
        assert old.terminal_reason == "stopped"
        assert new.feed(pcm)
        new.close()
        tick_at(holder["stream"], clock, 10.22)
        clock[0] = 10.43
        speech.wait_for(
            lambda snap: snap.mouth == "open" and snap.generation > previous_generation
        )
        current = speech.read().generation
        old_stream.finish_output()
        old._end_presentation()
        assert speech.read().generation == current and speech.read().mouth == "open"
        assert new.terminal_reason is None
    finally:
        new.stop()
        assert new.settle()


def test_stop_fences_an_inflight_pose_even_when_clock_timestamp_is_equal(
    device_case, monkeypatch
):
    sink, stream, _events, speech, clock = device_case(observed=True)
    assert sink.feed(struct.pack(f"<{FRAMES}h", *([16384] * FRAMES)))
    sink.close()
    tick_at(stream, clock, 10.0)
    clock[0] = 10.21
    speech.wait_for(lambda snap: snap.mouth == "open")
    ready = threading.Event()
    release = threading.Event()
    original_publish = speech.publish

    def held_publish(snapshot):
        if (
            threading.current_thread().name == "held-publication"
            and snapshot.mouth == "open"
        ):
            ready.set()
            assert release.wait(timeout=3)
        original_publish(snapshot)

    monkeypatch.setattr(speech, "publish", held_publish)
    caller = threading.Thread(
        target=sink._publish_presentation, name="held-publication"
    )
    caller.start()
    try:
        assert ready.wait(timeout=2)
        sink.stop()
        assert speech.read().mouth == "rest"
        release.set()
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert speech.read().mouth == "rest" and not speech.read().timing_valid
    finally:
        release.set()
        caller.join(timeout=2)


def test_notification_and_explicit_mapping_checks_cannot_analyze_pcm_concurrently(
    device_case, monkeypatch
):
    import tldw_chatbook.Audio.streaming_sink as module

    sink, _stream, _events, _speech, _clock = device_case(observed=True)
    entered = threading.Event()
    release = threading.Event()
    collision = threading.Event()
    analyzing = threading.Lock()
    original_push = module.PcmEnvelope.push

    def held_push(envelope, pcm):
        if not analyzing.acquire(blocking=False):
            collision.set()
            return original_push(envelope, pcm)
        try:
            entered.set()
            assert release.wait(timeout=3)
            return original_push(envelope, pcm)
        finally:
            analyzing.release()

    monkeypatch.setattr(module.PcmEnvelope, "push", held_push)
    assert sink.feed(struct.pack("<24000h", *([8192] * 24000)))
    assert entered.wait(timeout=2)
    caller = threading.Thread(target=sink._publish_presentation)
    caller.start()
    try:
        assert not collision.wait(timeout=0.05)
    finally:
        release.set()
        caller.join(timeout=2)
        assert not caller.is_alive()
    assert not collision.is_set()
