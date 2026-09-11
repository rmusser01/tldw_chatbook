"""Hardware-free checks at the installed sounddevice PortAudio boundary."""

import asyncio
import threading
from types import SimpleNamespace
import weakref

import pytest

from Tests.Audio.fakes.native_duplex_helpers import (
    build_driver,
    driver_library,
    emit_callback,
    load_native,
)


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    return driver_library(build_driver(tmp_path_factory.mktemp("adapter-driver")))


@pytest.fixture
def portaudio(monkeypatch):
    import sounddevice as sd

    calls = []
    controls = SimpleNamespace(
        calls=calls,
        callback=None,
        userdata=None,
        stop_gate=None,
        close_gate=None,
        open_gate=None,
        start_gate=None,
        close_error=False,
        stop_error=False,
        cleaned=False,
    )
    real_lib = sd._lib
    ffi = sd._ffi

    class Library:
        def __getattr__(self, name):
            return getattr(real_lib, name)

        def Pa_OpenStream(self, ptr, inp, out, rate, block, flags, cb, userdata):
            calls.append("open")
            controls.callback, controls.userdata = cb, userdata
            assert (rate, block, inp.channelCount, out.channelCount) == (
                48000,
                480,
                1,
                1,
            )
            if controls.open_gate:
                controls.open_gate.wait()
            ptr[0] = ffi.cast("PaStream *", 1234)
            return 0

        def Pa_GetStreamInfo(self, ptr):
            return SimpleNamespace(
                sampleRate=48000, inputLatency=0.01, outputLatency=0.01
            )

        def Pa_StartStream(self, ptr):
            calls.append("start")
            if controls.start_gate:
                controls.start_gate.wait()
            return 0

        def Pa_StopStream(self, ptr):
            calls.append("stop")
            if controls.stop_gate:
                controls.stop_gate.wait()
            return -9999 if controls.stop_error else 0

        def Pa_CloseStream(self, ptr):
            calls.append("close")
            if controls.close_gate:
                controls.close_gate.wait()
            return -9999 if controls.close_error else 0

    monkeypatch.setattr(sd, "_lib", Library())

    def parameters(kind, device, channels, dtype, latency, extra, samplerate):
        params = ffi.new("PaStreamParameters *", dict(device=0, channelCount=1))
        return params, "int16", 2, 48000

    monkeypatch.setattr(sd, "_get_stream_parameters", parameters)

    # No native error-details call is needed to establish checked failure.
    def check(value, *args):
        if value < 0:
            raise RuntimeError("checked native failure")
        return value

    monkeypatch.setattr(sd, "_check", check)
    controls.sd = sd
    return controls


def adapter(portaudio, **kwargs):
    from tldw_chatbook.Audio.native_duplex_stream import (
        NativeDuplexStream,
        NativeStreamRegistry,
    )

    registry = kwargs.pop("registry", NativeStreamRegistry())
    return NativeDuplexStream(
        generation=7,
        capture_capacity=64,
        render_capacity=64,
        native=load_native(),
        sounddevice=portaudio.sd,
        registry=registry,
        **kwargs,
    )


async def eventually(predicate):
    for _ in range(2000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("owner did not settle")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["guard", "constructor", "start"])
async def test_failed_worker_dispatch_releases_provably_unopened_owner(
    portaudio, monkeypatch, failure
):
    from tldw_chatbook.Audio import native_duplex_stream

    registry = native_duplex_stream.NativeStreamRegistry()
    monkeypatch.setattr(native_duplex_stream, "NATIVE_STREAM_REGISTRY", registry)
    monkeypatch.setattr(registry, "install_shutdown_guard", lambda sd: None)
    stream = adapter(portaudio, registry=registry)

    def fail(*args, **kwargs):
        raise RuntimeError("dispatch failed")

    with monkeypatch.context() as patch:
        if failure == "guard":
            patch.setattr(registry, "install_shutdown_guard", fail)
        elif failure == "constructor":
            patch.setattr(native_duplex_stream.threading, "Thread", fail)
        else:
            patch.setattr(native_duplex_stream.threading.Thread, "start", fail)
        with pytest.raises(RuntimeError, match="dispatch failed"):
            stream.begin()

    assert portaudio.calls == []
    assert registry.owner is None
    replacement = adapter(portaudio, registry=registry)
    replacement.begin()
    await replacement.wait_started()
    replacement.request_close()
    await replacement.wait_closed()


@pytest.mark.asyncio
async def test_dispatch_error_after_native_entry_retains_owner_until_checked_close(
    portaudio, monkeypatch
):
    from tldw_chatbook.Audio import native_duplex_stream

    portaudio.close_gate = threading.Event()
    entered = threading.Event()
    real_start = threading.Thread.start
    stream = adapter(portaudio)
    real_calibrate = stream._calibrate

    def calibrate():
        entered.set()
        real_calibrate()

    monkeypatch.setattr(stream, "_calibrate", calibrate)

    def start_then_fail(thread):
        real_start(thread)
        assert entered.wait(1)
        raise RuntimeError("dispatch observation interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(native_duplex_stream.threading.Thread, "start", start_then_fail)
        with pytest.raises(RuntimeError, match="dispatch observation interrupted"):
            stream.begin()
    assert stream.registry.owner is stream
    try:
        await eventually(lambda: "close" in portaudio.calls)
        assert stream.registry.owner is stream
    finally:
        portaudio.close_gate.set()
        stream.request_close()
        await stream.wait_closed()
    assert stream.registry.owner is None


def test_worker_reaching_entry_after_failed_dispatch_cannot_open_native(
    portaudio, monkeypatch
):
    pending = []
    real_start = threading.Thread.start
    stream = adapter(portaudio)

    def delayed_start(thread):
        pending.append(thread)
        raise RuntimeError("dispatch observation interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", delayed_start)
        with pytest.raises(RuntimeError, match="dispatch observation interrupted"):
            stream.begin()
    assert stream.registry.owner is None
    real_start(pending[0])
    pending[0].join(1)
    assert not pending[0].is_alive()
    assert portaudio.calls == []


@pytest.mark.asyncio
async def test_portaudio_receives_native_pointers_and_progresses_with_gil_held(
    portaudio, driver
):
    stream = adapter(portaudio)
    stream.begin()
    await stream.wait_started()
    ffi = portaudio.sd._ffi
    assert (
        int(ffi.cast("uintptr_t", portaudio.callback)) == stream.bridge.callback_address
    )
    assert (
        int(ffi.cast("uintptr_t", portaudio.userdata)) == stream.bridge.userdata_address
    )
    assert (
        driver.progress_with_gil_held(
            int(ffi.cast("uintptr_t", portaudio.callback)),
            int(ffi.cast("uintptr_t", portaudio.userdata)),
            8,
        )
        == 8
    )
    assert stream.bridge.snapshot()["capture_occupancy"] == 8
    stream.request_close()
    await stream.wait_closed()
    assert portaudio.calls == ["open", "start", "stop", "close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", ["stop", "close", "open", "start"])
async def test_cancelled_observer_retains_owner_and_late_checked_close_releases(
    portaudio, driver, blocked
):
    gate = threading.Event()
    setattr(portaudio, blocked + "_gate", gate)
    stream = adapter(portaudio)
    registry = stream.registry
    stream.begin()
    await eventually(
        lambda: (
            blocked in portaudio.calls
            if blocked in ("open", "start")
            else "start" in portaudio.calls
        )
    )
    stream.request_close()
    observer = asyncio.create_task(stream.wait_closed())
    await asyncio.sleep(0)
    observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observer
    reference = weakref.ref(stream)
    del stream
    assert reference() is not None
    with pytest.raises(RuntimeError, match="shutdown unconfirmed"):
        adapter(portaudio, registry=registry).begin()
    assert emit_callback(driver, reference().bridge)[0] == bytes(960)
    if blocked == "stop":
        assert "close" not in portaudio.calls
    gate.set()
    await eventually(lambda: registry.owner is None)
    await eventually(lambda: reference() is None)
    assert portaudio.calls.count("open") == 1  # Never automatically reopen.


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", ["stop", "close"])
async def test_deadline_is_two_seconds_from_first_fence_without_freeing_owner(
    portaudio, blocked
):
    now = [10.0]
    gate = threading.Event()
    setattr(portaudio, blocked + "_gate", gate)
    stream = adapter(portaudio, clock=lambda: now[0])
    stream.begin()
    await stream.wait_started()
    stream.request_close()
    await eventually(lambda: blocked in portaudio.calls)
    now[0] = 12.0
    with pytest.raises(RuntimeError, match="shutdown unconfirmed"):
        await stream.wait_closed()
    assert stream.registry.owner is stream
    if blocked == "stop":
        assert "close" not in portaudio.calls
    gate.set()
    await eventually(lambda: stream.registry.owner is None)


@pytest.mark.asyncio
async def test_failed_close_with_null_wrapper_pointer_remains_quarantined(portaudio):
    portaudio.close_error = True
    stream = adapter(portaudio)
    stream.begin()
    await stream.wait_started()
    stream.request_close()
    with pytest.raises(RuntimeError, match="shutdown unconfirmed"):
        await stream.wait_closed()
    assert stream.raw_stream._ptr == portaudio.sd._ffi.NULL
    assert stream.native_handle != portaudio.sd._ffi.NULL
    assert stream.registry.owner is stream
    assert portaudio.calls.count("close") == 1


@pytest.mark.asyncio
async def test_stop_error_still_allows_one_checked_close(portaudio):
    portaudio.stop_error = True
    stream = adapter(portaudio)
    stream.begin()
    await stream.wait_started()
    stream.request_close()
    await stream.wait_closed()
    assert stream.registry.owner is None
    assert stream.stop_failed


def test_stale_abi_rejected_before_any_device_open(portaudio):
    from tldw_chatbook.Audio.native_duplex_stream import NativeDuplexStream

    with pytest.raises(RuntimeError, match="native duplex"):
        NativeDuplexStream(
            generation=0,
            native=SimpleNamespace(DUPLEX_ABI_VERSION=0),
            sounddevice=portaudio.sd,
        )
    assert portaudio.calls == []


def test_unsupported_native_atomics_report_categorical_unavailability(portaudio):
    from tldw_chatbook.Audio.native_duplex_stream import (
        NativeDuplexStream,
        NativeDuplexUnavailable,
    )

    def unavailable(**kwargs):
        raise RuntimeError("unsupported atomic implementation")

    native = SimpleNamespace(DUPLEX_ABI_VERSION=1, NativeDuplexBridge=unavailable)
    with pytest.raises(NativeDuplexUnavailable):
        NativeDuplexStream(generation=0, native=native, sounddevice=portaudio.sd)
    assert portaudio.calls == []


def test_late_owner_completion_never_calls_the_closed_observer_loop(portaudio):
    portaudio.close_gate = threading.Event()
    stream = adapter(portaudio)
    stream.begin()

    async def observe():
        await stream.wait_started()
        stream.request_close()
        await eventually(lambda: "close" in portaudio.calls)
        waiter = asyncio.create_task(stream.wait_closed())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    asyncio.run(observe())
    # The observer loop has been closed before the checked native call returns.
    portaudio.close_gate.set()
    assert stream._closed.result(timeout=1)
    assert stream.registry.owner is None


def test_shutdown_guard_preserves_clean_handler_and_retains_uncertain_owner():
    from tldw_chatbook.Audio.native_duplex_stream import NativeStreamRegistry

    registry = NativeStreamRegistry()
    calls = []
    registry.shutdown(lambda: calls.append("terminate"))
    assert calls == ["terminate"]
    registry = NativeStreamRegistry()
    owner = SimpleNamespace(
        bridge=SimpleNamespace(
            deactivate=lambda: calls.append("deactivate"),
            retain_for_process_lifetime=lambda: calls.append("retain"),
        )
    )
    registry.claim(owner)
    registry.shutdown(lambda: calls.append("terminate"))
    assert calls == ["terminate", "deactivate", "retain"]


def test_shutdown_guard_replaces_registered_sounddevice_hook_only_once(monkeypatch):
    from tldw_chatbook.Audio import native_duplex_stream

    registrations = []
    monkeypatch.setattr(
        native_duplex_stream.atexit,
        "unregister",
        lambda fn: registrations.append(("remove", fn)),
    )
    monkeypatch.setattr(
        native_duplex_stream.atexit,
        "register",
        lambda fn, arg: registrations.append(("add", fn, arg)),
    )
    registry = native_duplex_stream.NativeStreamRegistry()

    def handler():
        return None

    sd = SimpleNamespace(_exit_handler=handler)
    registry.install_shutdown_guard(sd)
    registry.install_shutdown_guard(sd)
    assert registrations == [("remove", handler), ("add", registry.shutdown, handler)]


@pytest.fixture
def native_transport(portaudio, monkeypatch):
    from tldw_chatbook.Audio import native_duplex_stream
    from tldw_chatbook.Audio.duplex_transport import DuplexAudioTransport

    registry = native_duplex_stream.NativeStreamRegistry()
    monkeypatch.setattr(native_duplex_stream, "NATIVE_STREAM_REGISTRY", registry)
    monkeypatch.setattr(registry, "install_shutdown_guard", lambda sd: None)
    return DuplexAudioTransport()


@pytest.mark.asyncio
async def test_permanent_session_close_cannot_start_capture_later(
    native_transport, portaudio
):
    native_transport.request_close()
    try:
        with pytest.raises(RuntimeError, match="closed"):
            await native_transport.start()
        assert portaudio.calls == []
    finally:
        await native_transport.close()


@pytest.mark.asyncio
async def test_native_transport_actual_receipts_references_and_historical_playback(
    native_transport, driver
):
    from tldw_chatbook.Audio.duplex_contracts import RenderSubmission, CaptureDrainError
    from tldw_chatbook.Audio.voice_preprocessor import VoicePreprocessor
    from Tests.Audio.test_duplex_transport import HealthyAec, AdmittingIsolationMonitor

    t = native_transport
    await t.start()
    bridge = t._stream.bridge
    first = t.queue_render(b"\x01\x00" * 480)
    assert first == RenderSubmission(0, bridge.output_epoch, 0)
    assert t.render_boundary(first) is None
    emit_callback(driver, bridge, times=(10.0, 10.01, 10.02))
    capture = t.pop_capture()
    reference = t.pop_render_reference()
    assert reference.sequence == 0
    assert capture.assistant_rendering is False  # DAC playback is still in future.
    cancelled = t.queue_render(bytes(960))
    t.queue_render(bytes(960))
    await t.abort_output()
    with pytest.raises(CaptureDrainError):
        t.render_boundary(cancelled)
    replacement = t.queue_render(b"\x01\x00" * 480)
    assert replacement.submission_id == 3
    emit_callback(driver, bridge, times=(10.01, 10.02, 10.03))
    second = t.pop_capture()
    next_reference = t.pop_render_reference()
    assert next_reference.sequence == 1
    assert second.render_reference_sequence == 1
    boundary = t.render_boundary(replacement)
    assert boundary.ended_ns == next_reference.ended_ns
    assert not hasattr(replacement, "ended_ns")
    # Real preprocessor continuity survives cancelled submission IDs 1/2.
    preprocessor = VoicePreprocessor(
        aec=HealthyAec(), isolation_monitor=AdmittingIsolationMonitor()
    )
    await preprocessor.process_capture(
        capture, render_frames=(reference,), assistant_rendering=True
    )
    await preprocessor.process_capture(
        second, render_frames=(next_reference,), assistant_rendering=True
    )
    assert second.discontinuity is False
    from tldw_chatbook.Audio.duplex_contracts import AecHealth

    assert preprocessor.health is not AecHealth.DEGRADED
    await t.abort_output()
    emit_callback(driver, bridge, times=(10.02, 10.03, 10.04))
    cancelled_tail_capture = t.pop_capture()
    cancelled_tail_reference = t.pop_render_reference()
    assert cancelled_tail_capture is not None
    assert cancelled_tail_reference.sequence == 2
    assert cancelled_tail_reference.pcm16 == bytes(960)
    assert cancelled_tail_capture.assistant_rendering is True
    emit_callback(driver, bridge, times=(10.03, 10.04, 10.05))
    t.pop_capture()
    assert t.pop_render_reference().sequence == 3
    emit_callback(driver, bridge, times=(10.04, 10.05, 10.06))
    assert t.pop_capture().assistant_rendering is False
    assert t.pop_render_reference().sequence == 4
    await t.close()


@pytest.mark.asyncio
async def test_native_capture_pairs_only_the_exact_submitted_render_receipt(
    native_transport, driver
):
    from tldw_chatbook.Audio.duplex_contracts import RenderBoundary

    transport = native_transport
    await transport.start()
    try:
        bridge = transport._stream.bridge
        submission = transport.queue_render(bytes(960))
        emit_callback(driver, bridge, times=(10.0, 10.01, 10.02))
        capture = transport.pop_capture()
        reference = transport.pop_render_reference()
        expected = RenderBoundary(
            submission.generation,
            submission.output_epoch,
            submission.submission_id,
            reference.ended_ns,
        )
        assert capture.committed_render == expected
        assert reference.committed_render == expected
        assert reference.pcm16 == bytes(960)  # Submitted silence is still a commit.

        emit_callback(driver, bridge, times=(10.01, 10.02, 10.03))
        silent_capture = transport.pop_capture()
        silent_reference = transport.pop_render_reference()
        assert silent_capture.committed_render is None
        assert silent_reference.committed_render is None
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_native_phrase_tail_keeps_complete_dsp_references(
    native_transport, driver
):
    from tldw_chatbook.Audio.acoustic_isolation import AcousticIsolationMonitor
    from tldw_chatbook.Audio.duplex_contracts import AcousticDemotionReason
    from Tests.Audio.test_acoustic_isolation import QUIET, RENDER, pcm16

    transport = native_transport
    await transport.start()
    try:
        bridge = transport._stream.bridge
        submissions = [transport.queue_render(pcm16(RENDER)) for _ in range(4)]
        assert all(submission is not None for submission in submissions)

        monitor = AcousticIsolationMonitor()
        captures = []
        references = []
        reasons = []
        last_actual_boundary = None
        for index in range(6):
            emit_callback(
                driver,
                bridge,
                pcm16=pcm16(QUIET),
                times=(
                    10.0 + index / 100,
                    10.01 + index / 100,
                    10.02 + index / 100,
                ),
            )
            capture = transport.pop_capture()
            assert capture is not None
            captures.append(capture)
            drained = []
            while reference := transport.pop_render_reference():
                drained.append(reference)
            references.extend(drained)
            observation = monitor.observe(
                capture=capture,
                render_frames=drained,
                assistant_rendering=bool(capture.assistant_rendering),
                near_end_speech=False,
                native_processor_ok=True,
            )
            reasons.append(observation.safety.demotion_reason)
            if index == 3:
                last_actual_boundary = transport.render_boundary(submissions[-1])

        assert [frame.render_reference_sequence for frame in captures] == list(range(6))
        assert [frame.sequence for frame in references] == list(range(6))
        assert [frame.pcm16 for frame in references[4:]] == [bytes(960), bytes(960)]
        assert all(frame.assistant_rendering for frame in captures[4:])
        assert not any(
            reason
            in {
                AcousticDemotionReason.RENDER_REFERENCE_GAP,
                AcousticDemotionReason.MISSING_RENDER_REFERENCE,
            }
            for reason in reasons
        )
        assert transport.render_boundary(submissions[-1]) == last_actual_boundary
        assert transport.native_counters["capture_overflows"] == 0
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_native_preprocessor_consumes_silent_dsp_ticks_once_and_in_order(
    native_transport, driver
):
    from tldw_chatbook.Audio.acoustic_isolation import AcousticIsolationMonitor
    from tldw_chatbook.Audio.voice_preprocessor import VoicePreprocessor
    from Tests.Audio.test_acoustic_isolation import QUIET, RENDER, pcm16

    class RecordingAec:
        def __init__(self, real):
            self.real = real
            self.render_pcm = []

        def analyze_render(self, pcm, *, delay_ms):
            self.render_pcm.append(pcm)
            return self.real.analyze_render(pcm, delay_ms=delay_ms)

        def process_capture(self, pcm, *, delay_ms):
            return self.real.process_capture(pcm, delay_ms=delay_ms)

        def metrics(self):
            return self.real.metrics()

        def reset(self):
            return self.real.reset()

    admitted = []
    preprocessor = VoicePreprocessor.from_native(
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        isolation_monitor=AcousticIsolationMonitor(),
    )
    assert preprocessor._aec is not None
    recording_aec = RecordingAec(preprocessor._aec)
    preprocessor._aec = recording_aec

    transport = native_transport
    await transport.start()
    try:
        bridge = transport._stream.bridge
        for _ in range(4):
            assert transport.queue_render(pcm16(RENDER)) is not None

        captures = []
        for index in range(6):
            emit_callback(
                driver,
                bridge,
                pcm16=pcm16(QUIET),
                times=(
                    10.0 + index / 100,
                    10.01 + index / 100,
                    10.02 + index / 100,
                ),
            )
            capture = transport.pop_capture()
            reference = transport.pop_render_reference()
            assert capture is not None and reference is not None
            captures.append(capture)
            await preprocessor.process_capture(
                capture,
                render_frames=(reference,),
                assistant_rendering=bool(capture.assistant_rendering),
            )

        assert transport.queue_render(pcm16(RENDER)) is not None
        emit_callback(
            driver,
            bridge,
            pcm16=pcm16(QUIET),
            times=(10.06, 10.07, 10.08),
        )
        resumed_capture = transport.pop_capture()
        resumed_reference = transport.pop_render_reference()
        assert resumed_capture is not None and resumed_reference is not None
        captures.append(resumed_capture)
        await preprocessor.process_capture(
            resumed_capture,
            render_frames=(resumed_reference,),
            assistant_rendering=bool(resumed_capture.assistant_rendering),
        )

        assert recording_aec.render_pcm == [pcm16(RENDER)] * 4 + [bytes(960)] * 2 + [
            pcm16(RENDER)
        ]
        assert [frame.render_reference_sequence for frame in captures] == list(range(7))
        playback_sequences = {
            frame.sequence for frame in captures if frame.assistant_rendering
        }
        assert not playback_sequences.intersection(frame.sequence for frame in admitted)
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_native_drain_uses_callback_observation_and_overlays_dropped_fault(
    native_transport, driver
):
    t = native_transport
    await t.start()
    bridge = t._stream.bridge
    emit_callback(driver, bridge)
    recorded = bridge.monotonic_ns() + t._stream.native_to_python_offset_ns
    t._clock = lambda: recorded + 10_000_000_000
    frame = t.pop_capture()
    assert frame.delay_evidence.observed_ns <= recorded
    for i in range(65):
        emit_callback(
            driver, bridge, times=(10.01 + i * 0.01, 10.03 + i * 0.01, 10.05 + i * 0.01)
        )
    assert t.capture_count == 64
    assert t.capture_overflows == 1
    assert t.buffer_occupancy.capture_frames == 64
    t.acknowledge_capture(999, clock_generation=0, dsp_ok=False, vad_ok=False)
    faulted = t.pop_capture()
    assert not faulted.delay_evidence.occupancy_bounded
    assert faulted.discontinuity
    assert faulted.assistant_rendering is True  # Unknown cannot become legacy/idle.
    await t.close()
    assert t.capture_overflows == 1  # Cumulative diagnostics survive route detach.


@pytest.mark.asyncio
async def test_native_completion_survives_capture_pop(native_transport, driver):
    t = native_transport
    await t.start()
    submission = t.queue_render(bytes(960))
    emit_callback(driver, t._stream.bridge)
    capture = t.pop_capture()
    reference = t.pop_render_reference()
    t.acknowledge_capture(
        capture.sequence, clock_generation=0, dsp_ok=True, vad_ok=True
    )
    assert t.render_boundary(submission).ended_ns == reference.ended_ns
    await t.close()
    from tldw_chatbook.Audio.duplex_contracts import CaptureDrainError

    with pytest.raises(CaptureDrainError):
        t.render_boundary(submission)


@pytest.mark.asyncio
async def test_route_reset_deadline_does_not_wait_for_blocked_open_lock(
    native_transport, portaudio
):
    from tldw_chatbook.Audio.duplex_contracts import RouteKind
    from tldw_chatbook.Audio.native_duplex_stream import AudioShutdownUnconfirmed

    portaudio.open_gate = threading.Event()
    starting = asyncio.create_task(native_transport.start())
    await eventually(lambda: "open" in portaudio.calls)
    stream = native_transport._stream
    now = [10.0]
    stream._clock = lambda: now[0]
    closing = native_transport.notify_route_changed(RouteKind.DUPLEX)
    now[0] = 12.0
    try:
        with pytest.raises(AudioShutdownUnconfirmed):
            await asyncio.wait_for(asyncio.shield(closing), 0.1)
    finally:
        portaudio.open_gate.set()
        await asyncio.gather(starting, closing, return_exceptions=True)
        await eventually(lambda: stream.registry.owner is None)


@pytest.mark.asyncio
async def test_new_transport_is_quarantined_until_late_checked_close(
    native_transport, portaudio
):
    from tldw_chatbook.Audio.duplex_transport import DuplexAudioTransport
    from tldw_chatbook.Audio.native_duplex_stream import AudioShutdownUnconfirmed

    await native_transport.start()
    stream = native_transport._stream
    portaudio.close_gate = threading.Event()
    now = [10.0]
    stream._clock = lambda: now[0]
    observer = asyncio.create_task(native_transport.close())
    await eventually(lambda: "close" in portaudio.calls)
    now[0] = 12.0
    with pytest.raises(AudioShutdownUnconfirmed):
        await observer
    replacement = DuplexAudioTransport()
    with pytest.raises(AudioShutdownUnconfirmed):
        await replacement.start()
    assert portaudio.calls.count("open") == 1
    portaudio.close_gate.set()
    await eventually(lambda: stream.registry.owner is None)
    assert not replacement.is_open
    assert portaudio.calls.count("open") == 1
    # A deliberate later user start can use the now-confirmed clean runtime.
    await replacement.start()
    await replacement.close()


@pytest.mark.asyncio
async def test_playback_gap_is_proven_idle_despite_native_first_to_last_envelope(
    native_transport, driver
):
    t = native_transport
    await t.start()
    bridge = t._stream.bridge
    first_pcm = b"\x01\x00" * 480
    resumed_pcm = b"\x02\x00" * 480
    first_submission = t.queue_render(first_pcm)
    emit_callback(driver, bridge, times=(10.0, 10.01, 10.02))
    t.pop_capture()
    references = [t.pop_render_reference()]
    first_boundary = t.render_boundary(first_submission)
    emit_callback(driver, bridge, times=(10.01, 10.02, 10.03))
    t.pop_capture()
    references.append(t.pop_render_reference())
    emit_callback(driver, bridge, times=(10.02, 10.03, 10.04))
    assert t.pop_capture().assistant_rendering
    references.append(t.pop_render_reference())
    assert t.render_boundary(first_submission) == first_boundary
    resumed_submission = t.queue_render(resumed_pcm)
    emit_callback(driver, bridge, times=(10.03, 10.04, 10.05))
    assert t.pop_capture().assistant_rendering is False
    references.append(t.pop_render_reference())
    assert [reference.sequence for reference in references] == list(range(4))
    assert [reference.pcm16 for reference in references] == [
        first_pcm,
        bytes(960),
        bytes(960),
        resumed_pcm,
    ]
    assert t.render_boundary(resumed_submission) is not None
    await t.close()


@pytest.mark.asyncio
async def test_native_idle_reference_overflow_remains_fail_closed(
    native_transport, driver
):
    t = native_transport
    await t.start()
    try:
        bridge = t._stream.bridge
        for index in range(65):
            emit_callback(
                driver,
                bridge,
                times=(
                    10.0 + index / 100,
                    10.01 + index / 100,
                    10.02 + index / 100,
                ),
            )
            capture = t.pop_capture()
            assert capture is not None

        assert capture.discontinuity
        assert capture.delay_evidence is not None
        assert not capture.delay_evidence.occupancy_bounded
        assert t.render_reference_overflows == 1
        assert t.native_counters["capture_overflows"] == 0
    finally:
        await t.close()


@pytest.mark.asyncio
async def test_calibration_uses_minimum_bracket_midpoint_and_rejects_slow_brackets(
    portaudio,
):
    native = load_native()

    class ClockBridge:
        def __init__(self, **kwargs):
            self.real = native.NativeDuplexBridge(**kwargs)

        def __getattr__(self, name):
            return getattr(self.real, name)

        def monotonic_ns(self):
            return 1_000_000

    module = SimpleNamespace(DUPLEX_ABI_VERSION=1, NativeDuplexBridge=ClockBridge)
    from tldw_chatbook.Audio.native_duplex_stream import (
        NativeDuplexStream,
        NativeStreamRegistry,
        NativeDuplexUnavailable,
    )

    values = iter(
        [5_000_000, 6_100_000, 5_000_000, 5_000_100] + [5_000_000, 5_100_000] * 6
    )
    stream = NativeDuplexStream(
        generation=0,
        native=module,
        sounddevice=portaudio.sd,
        registry=NativeStreamRegistry(),
        transport_clock=lambda: next(values),
    )
    stream.begin()
    await stream.wait_started()
    assert stream.native_to_python_offset_ns == 4_000_050
    stream.request_close()
    await stream.wait_closed()
    portaudio.calls.clear()
    values = iter([0, 1_000_001] * 8)
    stream = NativeDuplexStream(
        generation=0,
        native=module,
        sounddevice=portaudio.sd,
        registry=NativeStreamRegistry(),
        transport_clock=lambda: next(values),
    )
    stream.begin()
    with pytest.raises(NativeDuplexUnavailable):
        await stream.wait_started()
    await stream.wait_closed()
    assert portaudio.calls == []


@pytest.mark.asyncio
async def test_native_status_remains_fatal_after_ack_error_and_clean_record(
    native_transport, driver
):
    t = native_transport
    await t.start()
    bridge = t._stream.bridge
    # Startup discards must not consume the capture sequence or poison faults.
    for _ in range(50):
        emit_callback(driver, bridge, status=2)
    assert t.pop_capture() is None
    assert t.native_counters["fatal_status_bits"] == 0
    emit_callback(driver, bridge, status=4)
    t.acknowledge_capture(1, clock_generation=0, dsp_ok=False, vad_ok=False)
    frame = t.pop_capture()
    assert frame.sequence == 0
    assert frame.delay_evidence.status_flags == ("output_underflow",)
    emit_callback(driver, bridge, times=(10.01, 10.03, 10.05))
    frame = t.pop_capture()
    assert frame.delay_evidence.status_flags == ("output_underflow",)
    assert frame.assistant_rendering is True
    await t.close()


@pytest.mark.asyncio
async def test_priming_then_timestamp_fault_keeps_native_fatal_latches_clear(
    native_transport, driver
):
    t = native_transport
    await t.start()
    bridge = t._stream.bridge
    emit_callback(driver, bridge, status=2)
    emit_callback(driver, bridge)
    assert t.pop_capture().sequence == 0
    emit_callback(driver, bridge, times=(10.05, 10.07, 10.09))
    frame = t.pop_capture()
    assert frame.discontinuity
    assert frame.delay_evidence.status_flags == ()
    assert frame.delay_evidence.occupancy_bounded
    assert t.native_counters["startup_discarded"] == 1
    assert t.native_counters["fatal_status_bits"] == 0
    await t.close()
