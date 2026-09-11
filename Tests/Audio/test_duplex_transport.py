"""Deterministic tests for the app-owned duplex callback bridge."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import gc
import subprocess
import sys
from types import SimpleNamespace

import pytest

from Tests.Audio.fakes.fake_duplex_backend import FakeDuplexBackend, ManualClock
from tldw_chatbook.Audio.duplex_contracts import (
    AcousticIsolationObservation,
    AcousticSafetyPath,
    AcousticSafetySnapshot,
    AecHealth,
    AudioFrame,
    CaptureDrainError,
    DeviceRouteChanged,
    DuplexMode,
    NearEndDisposition,
    RouteKind,
)
from tldw_chatbook.Audio.duplex_transport import (
    FRAME_BYTES,
    FRAME_DURATION_NS,
    FRAME_SAMPLES,
    PROCESSING_SAMPLE_RATE,
    DuplexAudioTransport,
)
from tldw_chatbook.Audio.voice_preprocessor import VoicePreprocessor

SILENCE = bytes(FRAME_BYTES)
TONE = b"\x01\x00" * FRAME_SAMPLES


class HealthyAec:
    def analyze_render(self, _pcm16: bytes, *, delay_ms: int) -> None:
        assert delay_ms == 20

    def process_capture(self, pcm16: bytes, *, delay_ms: int) -> bytes:
        assert delay_ms == 20
        return pcm16

    def metrics(self) -> dict[str, float]:
        return {
            "erle_db": 6.0,
            "delay_ms": 20.0,
            "delay_estimate_available": 1.0,
            "delay_estimate_refined": 1.0,
            "delay_age_blocks": 5.0,
            "clock_drift": 0.0,
        }

    def reset(self) -> None:
        return None


class AdmittingIsolationMonitor:
    """Keep transport tests focused on transport rather than acoustic policy."""

    def observe(
        self,
        *,
        capture: AudioFrame,
        render_frames: object,
        assistant_rendering: bool,
        near_end_speech: bool,
        native_processor_ok: bool,
    ) -> AcousticIsolationObservation:
        del render_frames, assistant_rendering
        assert native_processor_ok is True
        return AcousticIsolationObservation(
            AcousticSafetySnapshot(
                AcousticSafetyPath.ACOUSTIC_ISOLATION,
                DuplexMode.FULL_DUPLEX,
                capture.clock_generation,
                True,
                None,
            ),
            NearEndDisposition.ADMIT if near_end_speech else None,
        )

    def reset_for_route(self, _clock_generation: int) -> None:
        return None


def test_import_does_not_load_audio_or_native_dependencies() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import tldw_chatbook.Audio.duplex_transport; "
                "assert 'sounddevice' not in sys.modules; "
                "assert 'tldw_voice_aec' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr


@pytest.mark.asyncio
async def test_opens_one_48khz_mono_ten_millisecond_session() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())

    await transport.start()
    await transport.start()

    assert backend.open_calls == [
        {
            "sample_rate": PROCESSING_SAMPLE_RATE,
            "channels": 1,
            "frame_samples": FRAME_SAMPLES,
        }
    ]
    assert backend.stream.started is True
    assert transport.stream_latency_seconds == (0.01, 0.01)


@pytest.mark.asyncio
async def test_explicit_device_pair_is_bound_to_the_opened_stream() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())

    await transport.start(device_pair=(7, 8))

    assert backend.open_calls == [
        {
            "sample_rate": PROCESSING_SAMPLE_RATE,
            "channels": 1,
            "frame_samples": FRAME_SAMPLES,
            "device_pair": (7, 8),
        }
    ]


@pytest.mark.asyncio
async def test_default_start_omits_device_pair_for_legacy_backends() -> None:
    fake = FakeDuplexBackend()

    class LegacyBackend:
        def open_stream(
            self,
            *,
            sample_rate: int,
            channels: int,
            frame_samples: int,
            callback: object,
        ) -> object:
            return fake.open_stream(
                sample_rate=sample_rate,
                channels=channels,
                frame_samples=frame_samples,
                callback=callback,  # type: ignore[arg-type]
            )

    transport = DuplexAudioTransport(
        backend=LegacyBackend(),  # type: ignore[arg-type]
        clock=ManualClock(),
    )

    await transport.start()

    assert fake.open_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_error", "close_error", "expected_failures"),
    [
        (None, None, 0),
        (RuntimeError("stop failed"), None, 1),
        (None, RuntimeError("close failed"), 1),
        (RuntimeError("stop failed"), RuntimeError("close failed"), 2),
    ],
)
async def test_teardown_attempts_stop_and_close_and_records_each_failure(
    stop_error: Exception | None,
    close_error: Exception | None,
    expected_failures: int,
) -> None:
    backend = FakeDuplexBackend(
        stop_error=stop_error,
        close_error=close_error,
    )
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()

    await transport.close()

    assert backend.stream.stop_count == 1
    assert backend.stream.close_count == 1
    assert transport.teardown_failures == expected_failures


@pytest.mark.asyncio
async def test_start_error_is_preserved_when_its_teardown_also_fails() -> None:
    backend = FakeDuplexBackend(
        start_failures=1,
        stop_error=RuntimeError("stop failed"),
        close_error=RuntimeError("close failed"),
    )
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())

    with pytest.raises(RuntimeError, match="fake stream start failure"):
        await transport.start()

    assert backend.stream.stop_count == 1
    assert backend.stream.close_count == 1
    assert transport.teardown_failures == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking_phase", ["open", "start", "stop", "close"])
async def test_native_lifecycle_work_does_not_block_event_loop(
    blocking_phase: str,
) -> None:
    delay_options = {f"{blocking_phase}_delay_seconds": 0.05}
    backend = FakeDuplexBackend(**delay_options)
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    if blocking_phase in {"stop", "close"}:
        await transport.start()

    operation = (
        transport.start if blocking_phase in {"open", "start"} else transport.close
    )
    lifecycle_task = asyncio.create_task(operation())
    heartbeats = 0
    while not lifecycle_task.done():
        heartbeats += 1
        await asyncio.sleep(0.005)
    await lifecycle_task

    assert heartbeats >= 3


@pytest.mark.asyncio
async def test_concurrent_start_close_and_route_change_are_serialized() -> None:
    backend = FakeDuplexBackend(
        open_delay_seconds=0.05,
        start_delay_seconds=0.05,
    )
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    start_task = asyncio.create_task(transport.start())
    while backend.open_count == 0:
        await asyncio.sleep(0)
    assert start_task.done() is False

    close_task = asyncio.create_task(transport.close())
    route_task = transport.notify_route_changed(RouteKind.INPUT)
    await asyncio.gather(start_task, close_task, route_task)

    assert backend.open_count == 1
    assert backend.stream.stop_count == 1
    assert backend.stream.close_count == 1
    assert transport.clock_generation == 2
    assert transport.is_open is False


@pytest.mark.asyncio
async def test_route_callback_is_fenced_before_blocking_teardown_finishes() -> None:
    backend = FakeDuplexBackend(stop_delay_seconds=0.05)
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()
    old_callback = backend.stream.callback

    route_task = transport.notify_route_changed(RouteKind.INPUT)
    while backend.stream.stop_count == 0:
        await asyncio.sleep(0)

    assert route_task.done() is False
    assert old_callback(SILENCE, SimpleNamespace(), SimpleNamespace(), False) == SILENCE
    assert transport.stale_callbacks == 0
    await route_task

    assert old_callback(SILENCE, SimpleNamespace(), SimpleNamespace(), False) == SILENCE
    assert transport.stale_callbacks == 1


@pytest.mark.asyncio
async def test_route_invocation_fences_before_returned_awaitable_runs() -> None:
    backend = FakeDuplexBackend(stop_delay_seconds=0.05)
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()
    old_callback = backend.stream.callback
    old_generation = transport.clock_generation

    pending_teardown = transport.notify_route_changed(RouteKind.INPUT)

    assert transport.clock_generation == old_generation + 1
    assert transport.is_open is False
    assert transport.queue_render(TONE) is None
    assert backend.stream.stop_count == 0
    assert backend.stream.close_count == 0
    event = transport.pop_control_event()
    assert event == DeviceRouteChanged(old_generation, RouteKind.INPUT)
    assert old_callback(SILENCE, SimpleNamespace(), SimpleNamespace(), False) == SILENCE
    assert transport.stale_callbacks == 0

    await pending_teardown

    assert backend.stream.stop_count == 1
    assert backend.stream.close_count == 1
    assert old_callback(SILENCE, SimpleNamespace(), SimpleNamespace(), False) == SILENCE
    assert transport.stale_callbacks == 1


@pytest.mark.asyncio
async def test_dropped_route_awaitable_still_owns_native_teardown(
    recwarn: pytest.WarningsRecorder,
) -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()

    transport.notify_route_changed(RouteKind.INPUT)
    gc.collect()
    for _ in range(100):
        if backend.stream.close_count:
            break
        await asyncio.sleep(0.001)

    assert backend.stream.stop_count == 1
    assert backend.stream.close_count == 1
    assert not [
        warning
        for warning in recwarn
        if issubclass(warning.category, RuntimeWarning)
        and "never awaited" in str(warning.message)
    ]
    current = asyncio.current_task()
    assert not [
        task for task in asyncio.all_tasks() if task is not current and not task.done()
    ]


@pytest.mark.asyncio
async def test_immediate_route_waiter_cancellation_cannot_cancel_teardown() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()

    pending_teardown = transport.notify_route_changed(RouteKind.OUTPUT)
    assert isinstance(pending_teardown, asyncio.Future)
    pending_teardown.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_teardown
    for _ in range(100):
        if backend.stream.close_count:
            break
        await asyncio.sleep(0.001)

    assert backend.stream.stop_count == 1
    assert backend.stream.close_count == 1
    assert transport.is_open is False
    current = asyncio.current_task()
    assert not [
        task for task in asyncio.all_tasks() if task is not current and not task.done()
    ]


@pytest.mark.asyncio
async def test_notify_then_immediate_start_waits_for_owned_teardown() -> None:
    backend = FakeDuplexBackend(stop_delay_seconds=0.05)
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()
    old_stream = backend.stream

    pending_teardown = transport.notify_route_changed(RouteKind.DUPLEX)
    await transport.start()

    assert old_stream.stop_count == 1
    assert old_stream.close_count == 1
    assert backend.open_count == 2
    assert backend.stream is not old_stream
    await pending_teardown


def test_closed_route_owner_cannot_livelock_independently_scheduled_start() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio

from Tests.Audio.fakes.fake_duplex_backend import FakeDuplexBackend, ManualClock
from tldw_chatbook.Audio.duplex_contracts import RouteKind
from tldw_chatbook.Audio.duplex_transport import DuplexAudioTransport

async def main():
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    route_waiter = transport.notify_route_changed(RouteKind.INPUT)
    start_task = asyncio.create_task(transport.start())
    await asyncio.gather(route_waiter, start_task)
    assert backend.open_count == 1
    assert transport.is_open is True
    await transport.close()

asyncio.run(main())
""",
        ],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr


def test_start_prunes_multiple_completed_owners_and_retrieves_errors() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import gc

from Tests.Audio.fakes.fake_duplex_backend import FakeDuplexBackend, ManualClock
from tldw_chatbook.Audio.duplex_transport import DuplexAudioTransport

async def main():
    loop = asyncio.get_running_loop()
    unhandled = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    completed = [asyncio.create_task(asyncio.sleep(0)) for _ in range(3)]
    await asyncio.gather(*completed)
    transport._teardown_owners.update(completed)
    await transport.start()
    assert not transport._teardown_owners
    await transport.close()

    async def fail():
        raise RuntimeError("owned teardown failed")

    failed = asyncio.create_task(fail())
    await asyncio.sleep(0)
    transport._teardown_owners.add(failed)
    try:
        await transport.start()
    except RuntimeError as error:
        assert str(error) == "owned teardown failed"
    else:
        raise AssertionError("completed owner error was swallowed")
    assert failed not in transport._teardown_owners
    del failed
    gc.collect()
    await asyncio.sleep(0)
    assert not unhandled, unhandled

asyncio.run(main())
""",
        ],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr


@pytest.mark.asyncio
async def test_cancelled_start_keeps_pending_route_teardown_owned() -> None:
    backend = FakeDuplexBackend(stop_delay_seconds=0.05)
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()
    old_stream = backend.stream

    route_waiter = transport.notify_route_changed(RouteKind.DUPLEX)
    while old_stream.stop_count == 0:
        await asyncio.sleep(0)
    start_waiter = asyncio.create_task(transport.start())
    await asyncio.sleep(0)
    start_waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await start_waiter
    assert transport._teardown_owners
    await route_waiter

    assert old_stream.stop_count == 1
    assert old_stream.close_count == 1
    assert backend.open_count == 1
    assert not transport._teardown_owners


@pytest.mark.asyncio
async def test_route_invocation_fences_stream_while_native_start_is_blocked() -> None:
    backend = FakeDuplexBackend(start_delay_seconds=0.05)
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    start_task = asyncio.create_task(transport.start())
    while not backend.streams or not backend.stream.start_count:
        await asyncio.sleep(0)
    old_callback = backend.stream.callback
    old_generation = transport.clock_generation

    pending_teardown = transport.notify_route_changed(RouteKind.OUTPUT)

    assert transport.clock_generation == old_generation + 1
    assert transport.is_open is False
    old_callback(SILENCE, SimpleNamespace(), SimpleNamespace(), False)
    assert transport.capture_count == 0
    assert transport.stale_callbacks == 0

    await asyncio.gather(start_task, pending_teardown)

    assert backend.stream.stop_count == 1
    assert backend.stream.close_count == 1
    assert transport.is_open is False
    assert transport.queue_render(TONE) is None
    old_callback(SILENCE, SimpleNamespace(), SimpleNamespace(), False)
    assert transport.stale_callbacks == 1


@pytest.mark.asyncio
async def test_cancelled_route_wait_still_tears_down_its_fenced_stream() -> None:
    backend = FakeDuplexBackend(start_delay_seconds=0.05)
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    start_task = asyncio.create_task(transport.start())
    while not backend.streams or not backend.stream.start_count:
        await asyncio.sleep(0)

    route_task = transport.notify_route_changed(RouteKind.DUPLEX)
    await asyncio.sleep(0)
    route_task.cancel()

    await start_task
    with pytest.raises(asyncio.CancelledError):
        await route_task

    assert backend.stream.stop_count == 1
    assert backend.stream.close_count == 1
    assert transport.is_open is False
    current = asyncio.current_task()
    assert not [
        task for task in asyncio.all_tasks() if task is not current and not task.done()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_phase", ["open", "start", "teardown"])
async def test_lifecycle_cancellation_leaves_no_stream_or_owned_task(
    cancel_phase: str,
) -> None:
    delay_options = {
        "open": {"open_delay_seconds": 0.05},
        "start": {"start_delay_seconds": 0.05},
        "teardown": {"stop_delay_seconds": 0.05},
    }[cancel_phase]
    backend = FakeDuplexBackend(**delay_options)
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    if cancel_phase == "teardown":
        await transport.start()
        lifecycle_task = asyncio.create_task(transport.close())
        while backend.stream.stop_count == 0:
            await asyncio.sleep(0)
    else:
        lifecycle_task = asyncio.create_task(transport.start())
        while backend.open_count == 0 or (
            cancel_phase == "start"
            and (not backend.streams or backend.stream.start_count == 0)
        ):
            await asyncio.sleep(0)

    lifecycle_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await lifecycle_task

    assert backend.stream.stop_count == 1
    assert backend.stream.close_count == 1
    assert transport.is_open is False
    current = asyncio.current_task()
    assert not [
        task for task in asyncio.all_tasks() if task is not current and not task.done()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "actual_latency",
    [None, (0.01,), (-0.01, 0.01), (float("nan"), 0.01)],
)
async def test_invalid_actual_stream_latency_fails_before_start(
    actual_latency: object,
) -> None:
    backend = FakeDuplexBackend(stream_latency=actual_latency)
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())

    with pytest.raises(ValueError, match="latency"):
        await transport.start()

    assert backend.stream.started is False
    assert backend.stream.closed is True
    assert transport.stream_latency_seconds is None


@pytest.mark.asyncio
async def test_capture_sequences_increase_on_the_shared_generation_clock() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    await transport.start()

    queued_render = transport.queue_render(TONE)
    clock.advance(FRAME_DURATION_NS)
    backend.stream.emit_capture(SILENCE)
    first = transport.pop_capture()
    scheduled_render = transport.pop_render_reference()
    clock.advance(FRAME_DURATION_NS)
    backend.stream.emit_capture(SILENCE)
    second = transport.pop_capture()

    assert queued_render is not None
    assert scheduled_render is not None
    assert first is not None and second is not None
    assert (first.sequence, second.sequence) == (0, 1)
    assert first.clock_generation == scheduled_render.clock_generation
    assert first.delay_evidence is scheduled_render.delay_evidence
    assert first.render_reference_sequence == scheduled_render.sequence
    assert second.started_ns == first.ended_ns


@pytest.mark.asyncio
async def test_callback_rings_are_bounded_and_report_overflow() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=ManualClock(),
        capture_capacity=1,
        render_capacity=1,
    )
    await transport.start()

    assert transport.queue_render(TONE) is not None
    assert transport.queue_render(TONE) is None
    assert transport.render_overflows == 1

    backend.stream.emit_capture(SILENCE)
    backend.stream.emit_capture(SILENCE)

    assert transport.capture_count == 1
    assert transport.capture_overflows == 1


@pytest.mark.asyncio
async def test_content_free_buffer_snapshot_covers_every_callback_ring() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=ManualClock(),
        capture_capacity=2,
        render_capacity=3,
        control_capacity=4,
    )
    await transport.start()

    assert transport.queue_render(TONE) is not None
    backend.stream.emit_capture(SILENCE)

    assert asdict(transport.buffer_occupancy) == {
        "capture_frames": 1,
        "render_frames": 0,
        "render_reference_frames": 1,
        "control_events": 0,
    }
    assert asdict(transport.buffer_capacities) == {
        "capture_frames": 2,
        "render_frames": 3,
        "render_reference_frames": 3,
        "control_events": 4,
    }


@pytest.mark.asyncio
async def test_dropped_capture_taints_later_frame_and_blocks_admission() -> None:
    admitted: list[AudioFrame] = []
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=ManualClock(),
        capture_capacity=1,
    )
    preprocessor = VoicePreprocessor(
        aec=HealthyAec(),
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        healthy_streak=1,
        isolation_monitor=AdmittingIsolationMonitor(),
    )
    await transport.start()

    backend.stream.emit_capture(SILENCE)  # seq 0 is retained.
    first_render = transport.pop_render_reference()
    backend.stream.emit_capture(SILENCE)  # seq 1 is dropped.
    first = transport.pop_capture()
    backend.stream.emit_capture(SILENCE)  # seq 2 is retained but tainted.
    after_drop = transport.pop_capture()
    assert first is not None and first_render is not None and after_drop is not None

    await preprocessor.process_capture(
        first,
        render_frames=(first_render,),
        assistant_rendering=True,
    )
    result = await preprocessor.process_capture(after_drop, assistant_rendering=True)

    assert (first.sequence, after_drop.sequence) == (0, 2)
    assert after_drop.discontinuity is True
    assert result is None
    assert [captured.sequence for captured in admitted] == [0]
    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX


@pytest.mark.asyncio
async def test_render_callbacks_consume_scheduled_pcm_in_order() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()
    second_tone = b"\x02\x00" * FRAME_SAMPLES

    transport.queue_render(TONE)
    transport.queue_render(second_tone)

    assert backend.stream.emit_capture(SILENCE) == TONE
    assert backend.stream.emit_capture(SILENCE) == second_tone
    assert backend.stream.emit_capture(SILENCE) == SILENCE
    assert [
        transport.pop_render_reference().pcm16,  # type: ignore[union-attr]
        transport.pop_render_reference().pcm16,  # type: ignore[union-attr]
        transport.pop_render_reference().pcm16,  # type: ignore[union-attr]
    ] == [TONE, second_tone, SILENCE]
    assert transport.pop_render_reference() is None


@pytest.mark.asyncio
async def test_abort_preserves_submission_identity_without_false_aec_gap() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    preprocessor = VoicePreprocessor(
        aec=HealthyAec(),
        vad=lambda _frame: True,
        healthy_streak=1,
    )
    await transport.start()
    assert transport.queue_render(TONE).submission_id == 0  # type: ignore[union-attr]
    backend.stream.emit_capture(SILENCE)
    first_capture = transport.pop_capture()
    first_render = transport.pop_render_reference()
    assert first_capture is not None and first_render is not None
    await preprocessor.process_capture(
        first_capture,
        render_frames=(first_render,),
        assistant_rendering=True,
    )
    assert preprocessor.health is AecHealth.HEALTHY

    cancelled = transport.queue_render(TONE)
    assert cancelled.submission_id == 1  # type: ignore[union-attr]
    await transport.abort_output()
    with pytest.raises(CaptureDrainError, match="invalidated"):
        transport.render_boundary(cancelled)  # type: ignore[arg-type]
    clock.advance(FRAME_DURATION_NS)
    assert backend.stream.emit_capture(SILENCE) == SILENCE
    cancelled_capture = transport.pop_capture()
    cancelled_reference = transport.pop_render_reference()
    assert cancelled_capture is not None and cancelled_reference is not None
    assert cancelled_reference.sequence == 1
    assert cancelled_reference.pcm16 == SILENCE
    await preprocessor.process_capture(
        cancelled_capture,
        render_frames=(cancelled_reference,),
        assistant_rendering=True,
    )

    replacement = transport.queue_render(TONE)
    assert replacement is not None
    assert replacement.submission_id == 2
    clock.advance(FRAME_DURATION_NS)
    backend.stream.emit_capture(SILENCE)
    second_capture = transport.pop_capture()
    second_render = transport.pop_render_reference()
    assert second_capture is not None and second_render is not None

    await preprocessor.process_capture(
        second_capture,
        render_frames=(second_render,),
        assistant_rendering=True,
    )

    assert second_render.sequence == 2
    assert preprocessor.health is AecHealth.HEALTHY
    assert preprocessor.mode is DuplexMode.FULL_DUPLEX


@pytest.mark.asyncio
async def test_hardware_times_map_into_one_monotonic_delay_evidence() -> None:
    clock = ManualClock(now_ns=5_000_000_000)
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    await transport.start()
    transport.queue_render(TONE)

    backend.stream.emit_capture(
        SILENCE,
        input_adc_time=10.000,
        current_time=10.020,
        output_dac_time=10.040,
    )
    capture = transport.pop_capture()
    render = transport.pop_render_reference()

    assert capture is not None and render is not None
    timing = capture.delay_evidence
    assert timing is not None
    assert timing is render.delay_evidence
    assert timing.observed_ns == 5_000_000_000
    assert timing.capture_adc_ns == 4_980_000_000
    assert timing.render_dac_ns == 5_020_000_000
    assert timing.delay_ms == 40
    assert timing.capture_occupancy_frames == 0
    assert timing.render_occupancy_frames == 0
    assert timing.occupancy_bounded is True
    assert timing.timing_discontinuity is False
    assert timing.clock_drift is False


@pytest.mark.asyncio
async def test_callback_preserves_named_status_and_marks_timing_unreliable() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()

    for _ in range(50):
        backend.stream.emit_capture(SILENCE)
        assert transport.pop_capture() is not None

    backend.stream.emit_capture(
        SILENCE,
        status=SimpleNamespace(input_overflow=True, output_underflow=True),
    )
    capture = transport.pop_capture()

    assert capture is not None and capture.delay_evidence is not None
    assert capture.delay_evidence.status_flags == (
        "input_overflow",
        "output_underflow",
    )
    assert capture.delay_evidence.timing_discontinuity is True

    backend.stream.emit_capture(SILENCE)
    later = transport.pop_capture()
    assert later is not None and later.delay_evidence is not None
    assert later.delay_evidence.status_flags == ()
    assert later.delay_evidence.timing_discontinuity is True
    assert later.discontinuity is True


@pytest.mark.asyncio
async def test_one_flagged_callback_inside_startup_preroll_resets_cadence() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()

    for expected_sequence in range(2):
        backend.stream.emit_capture(SILENCE)
        capture = transport.pop_capture()
        assert capture is not None
        assert capture.sequence == expected_sequence

    backend.stream.emit_capture(
        SILENCE,
        status=SimpleNamespace(input_underflow=True),
    )
    assert transport.pop_capture() is None

    backend.stream.emit_capture(SILENCE)
    recovered = transport.pop_capture()

    assert recovered is not None and recovered.delay_evidence is not None
    assert recovered.sequence == 2
    assert recovered.discontinuity is False
    assert recovered.delay_evidence.status_flags == ()
    assert recovered.delay_evidence.timing_discontinuity is False


@pytest.mark.asyncio
async def test_single_flagged_startup_callback_is_discarded_before_timeline() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()

    backend.stream.emit_capture(
        SILENCE,
        status=SimpleNamespace(input_overflow=True),
    )

    assert transport.pop_capture() is None
    assert transport.capture_overflows == 0

    backend.stream.emit_capture(SILENCE)
    first = transport.pop_capture()

    assert first is not None and first.delay_evidence is not None
    assert first.sequence == 0
    assert first.discontinuity is False
    assert first.delay_evidence.status_flags == ()
    assert first.delay_evidence.timing_discontinuity is False


@pytest.mark.asyncio
async def test_persistent_flagged_callbacks_exceed_startup_preroll_fail_closed() -> (
    None
):
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()

    for _ in range(50):
        backend.stream.emit_capture(
            SILENCE,
            status=SimpleNamespace(input_overflow=True),
        )
        assert transport.pop_capture() is None

    backend.stream.emit_capture(
        SILENCE,
        status=SimpleNamespace(input_overflow=True),
    )
    failed = transport.pop_capture()

    assert failed is not None and failed.delay_evidence is not None
    assert failed.discontinuity is True
    assert failed.delay_evidence.timing_discontinuity is True
    with pytest.raises(CaptureDrainError, match="discontinuity"):
        await transport.drain_capture_through(failed.started_ns)


@pytest.mark.asyncio
async def test_callback_detects_gradual_cumulative_device_clock_drift() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    await transport.start()
    for index in range(5):
        current_time = (index + 1) * 0.01
        backend.stream.emit_capture(
            SILENCE,
            input_adc_time=current_time - 0.01,
            current_time=current_time,
            output_dac_time=current_time + 0.01,
        )
        baseline = transport.pop_capture()
        assert baseline is not None and baseline.delay_evidence is not None
        assert baseline.delay_evidence.clock_drift is False
        clock.advance(FRAME_DURATION_NS)

    drifted = None
    for index in range(5, 255):
        current_time = (index + 1) * 0.01
        backend.stream.emit_capture(
            SILENCE,
            input_adc_time=current_time - 0.01,
            current_time=current_time,
            output_dac_time=current_time + 0.01,
        )
        captured = transport.pop_capture()
        assert captured is not None and captured.delay_evidence is not None
        if captured.delay_evidence.clock_drift:
            drifted = captured
        clock.advance(10_100_000)

    assert drifted is not None
    assert drifted.delay_evidence is not None
    assert drifted.delay_evidence.clock_drift is True
    assert drifted.discontinuity is True


@pytest.mark.asyncio
async def test_single_callback_scheduling_jitter_does_not_claim_clock_drift() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    await transport.start()
    for index in range(5):
        current_time = (index + 1) * 0.01
        backend.stream.emit_capture(
            SILENCE,
            input_adc_time=current_time - 0.01,
            current_time=current_time,
            output_dac_time=current_time + 0.01,
        )
        assert transport.pop_capture() is not None
        clock.advance(FRAME_DURATION_NS)

    clock.advance(25_000_000)
    backend.stream.emit_capture(
        SILENCE,
        input_adc_time=0.05,
        current_time=0.06,
        output_dac_time=0.07,
    )
    jittered = transport.pop_capture()
    assert jittered is not None and jittered.delay_evidence is not None
    assert jittered.delay_evidence.clock_drift is False

    backend.stream.emit_capture(
        SILENCE,
        input_adc_time=0.06,
        current_time=0.07,
        output_dac_time=0.08,
    )
    recovered = transport.pop_capture()
    assert recovered is not None and recovered.delay_evidence is not None
    assert recovered.delay_evidence.clock_drift is False


@pytest.mark.asyncio
async def test_callback_detects_a_clock_consistent_missed_device_frame() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    await transport.start()
    backend.stream.emit_capture(
        SILENCE,
        input_adc_time=0.00,
        current_time=0.01,
        output_dac_time=0.02,
    )
    assert transport.pop_capture() is not None

    clock.advance(30_000_000)
    backend.stream.emit_capture(
        SILENCE,
        input_adc_time=0.03,
        current_time=0.04,
        output_dac_time=0.05,
    )
    missed = transport.pop_capture()

    assert missed is not None and missed.delay_evidence is not None
    assert missed.delay_evidence.clock_drift is False
    assert missed.delay_evidence.timing_discontinuity is True
    assert missed.discontinuity is True

    clock.advance(FRAME_DURATION_NS)
    backend.stream.emit_capture(
        SILENCE,
        input_adc_time=0.04,
        current_time=0.05,
        output_dac_time=0.06,
    )
    later = transport.pop_capture()
    assert later is not None and later.delay_evidence is not None
    assert later.delay_evidence.clock_drift is False
    assert later.delay_evidence.timing_discontinuity is True
    assert later.discontinuity is True


@pytest.mark.asyncio
@pytest.mark.parametrize("ring", ["capture", "reference"])
@pytest.mark.parametrize("later_failure", [None, "acknowledgement", "format"])
async def test_overflow_cause_survives_until_consumer_observes_fault(
    ring, later_failure
) -> None:
    """A dropped overflow frame must not reappear as a clock-only failure."""
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=clock,
        capture_capacity=2 if ring == "capture" else 4,
        render_capacity=4 if ring == "capture" else 2,
    )
    await transport.start()
    try:
        for _ in range(3):
            backend.stream.emit_capture(SILENCE)
            clock.advance(FRAME_DURATION_NS)
            if ring == "reference":
                transport.pop_capture()
        assert (
            transport.capture_overflows
            if ring == "capture"
            else transport.render_reference_overflows
        ) == 1
        while transport.pop_capture() is not None:
            pass
        while transport.pop_render_reference() is not None:
            pass

        if later_failure == "acknowledgement":
            transport.acknowledge_capture(
                0,
                clock_generation=transport.clock_generation,
                dsp_ok=False,
                vad_ok=False,
            )
        elif later_failure == "format":
            backend.stream.emit_capture(b"malformed")
            clock.advance(FRAME_DURATION_NS)

        # All device timestamps are continuous and the rings are empty now.
        backend.stream.emit_capture(SILENCE)
        observed = transport.pop_capture()
        assert observed is not None and observed.discontinuity
        assert observed.delay_evidence is not None
        assert not observed.delay_evidence.occupancy_bounded

        generation = transport.clock_generation
        await transport.close()
        await transport.start()
        assert transport.clock_generation > generation
        backend.stream.emit_capture(SILENCE)
        restarted = transport.pop_capture()
        assert restarted is not None and not restarted.discontinuity
        assert restarted.delay_evidence.occupancy_bounded
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_callback_never_schedules_or_mutates_async_wait_state() -> None:
    wait_calls = 0

    async def injected_wait() -> None:
        nonlocal wait_calls
        wait_calls += 1
        await asyncio.sleep(0)

    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=ManualClock(),
        state_wait=injected_wait,
    )
    await transport.start()

    class BombLoop:
        def is_running(self) -> bool:
            return True

        def call_soon_threadsafe(self, *_args: object) -> None:
            raise AssertionError("device callback scheduled event-loop work")

    transport._event_loop = BombLoop()  # type: ignore[attr-defined]
    backend.stream.emit_capture(SILENCE)

    assert wait_calls == 0


@pytest.mark.asyncio
async def test_render_reference_ring_overflow_marks_capture_incomplete() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=ManualClock(),
        render_capacity=1,
    )
    await transport.start()
    transport.queue_render(TONE)
    backend.stream.emit_capture(SILENCE)
    assert transport.queue_render(TONE) is not None

    backend.stream.emit_capture(SILENCE)
    first = transport.pop_capture()
    second = transport.pop_capture()

    assert first is not None and second is not None
    assert second.discontinuity is True
    assert second.delay_evidence is not None
    assert second.delay_evidence.occupancy_bounded is False
    assert transport.render_reference_overflows == 1


@pytest.mark.asyncio
async def test_route_change_fences_old_generation_before_content_free_event() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()
    old_callback = backend.stream.callback
    old_generation = transport.clock_generation

    await transport.notify_route_changed(RouteKind.DUPLEX)
    event = transport.pop_control_event()
    stale_output = old_callback(
        SILENCE,
        SimpleNamespace(),
        SimpleNamespace(),
        False,
    )

    assert isinstance(event, DeviceRouteChanged)
    assert asdict(event) == {
        "old_clock_generation": old_generation,
        "route_kind": RouteKind.DUPLEX,
    }
    assert transport.clock_generation == old_generation + 1
    assert stale_output == SILENCE
    assert transport.stale_callbacks == 1
    assert transport.capture_count == 0
    assert transport.queue_render(TONE) is None


@pytest.mark.asyncio
async def test_route_rebuild_uses_new_generation_while_old_callback_stays_fenced() -> (
    None
):
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()
    old_callback = backend.stream.callback

    await transport.notify_route_changed(RouteKind.OUTPUT)
    await transport.start()
    old_callback(SILENCE, SimpleNamespace(), SimpleNamespace(), False)
    backend.stream.emit_capture(SILENCE)
    frame = transport.pop_capture()

    assert len(backend.streams) == 2
    assert transport.stale_callbacks == 1
    assert frame is not None
    assert frame.clock_generation == transport.clock_generation
    assert transport.capture_count == 0


@pytest.mark.asyncio
async def test_route_rebuild_clears_old_failure_for_new_generation_drain() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()
    await transport.notify_route_changed(RouteKind.DUPLEX)

    await transport.start()
    backend.stream.emit_capture(SILENCE)
    frame = transport.pop_capture()
    assert frame is not None
    transport.acknowledge_capture(
        frame.sequence,
        clock_generation=frame.clock_generation,
        dsp_ok=True,
        vad_ok=True,
    )

    receipt = await transport.drain_capture_through(frame.started_ns)
    assert receipt.clock_generation == transport.clock_generation
    assert receipt.capture_sequence == frame.sequence


@pytest.mark.asyncio
async def test_drain_waits_for_later_input_and_returns_all_acknowledged_watermarks() -> (
    None
):
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    await transport.start()

    backend.stream.emit_capture(SILENCE)
    first = transport.pop_capture()
    assert first is not None
    transport.acknowledge_capture(
        first.sequence,
        clock_generation=first.clock_generation,
        dsp_ok=True,
        vad_ok=True,
    )
    boundary = first.ended_ns
    pending = asyncio.create_task(transport.drain_capture_through(boundary))
    await asyncio.sleep(0)
    assert pending.done() is False

    clock.advance(FRAME_DURATION_NS)
    backend.stream.emit_capture(SILENCE)
    second = transport.pop_capture()
    assert second is not None
    transport.acknowledge_capture(
        second.sequence,
        clock_generation=second.clock_generation,
        dsp_ok=True,
        vad_ok=True,
    )

    receipt = await asyncio.wait_for(pending, timeout=0.1)
    assert receipt.capture_watermark_ns > boundary
    assert (
        receipt.capture_sequence,
        receipt.dsp_sequence,
        receipt.vad_sequence,
    ) == (second.sequence, second.sequence, second.sequence)
    assert receipt.clock_generation == transport.clock_generation


@pytest.mark.asyncio
async def test_drain_fails_closed_after_capture_overflow() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=clock,
        capture_capacity=1,
    )
    await transport.start()
    backend.stream.emit_capture(SILENCE)
    clock.advance(FRAME_DURATION_NS)
    backend.stream.emit_capture(SILENCE)

    with pytest.raises(CaptureDrainError, match="overflow"):
        await transport.drain_capture_through(clock.now_ns - 1)


@pytest.mark.asyncio
async def test_drain_fails_closed_on_processing_acknowledgement_failure() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()
    backend.stream.emit_capture(SILENCE)
    frame = transport.pop_capture()
    assert frame is not None
    transport.acknowledge_capture(
        frame.sequence,
        clock_generation=frame.clock_generation,
        dsp_ok=True,
        vad_ok=False,
    )

    with pytest.raises(CaptureDrainError, match="acknowledgement"):
        await transport.drain_capture_through(frame.started_ns)


@pytest.mark.asyncio
async def test_full_duplex_native_failure_cannot_certify_capture_drain() -> None:
    class FailingAec(HealthyAec):
        def __init__(self) -> None:
            self.raise_on_capture = False

        def process_capture(self, pcm16: bytes, *, delay_ms: int) -> bytes:
            if self.raise_on_capture:
                raise RuntimeError("native failure")
            return super().process_capture(pcm16, delay_ms=delay_ms)

    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    aec = FailingAec()
    preprocessor = VoicePreprocessor(
        aec=aec,
        vad=lambda _frame: True,
        healthy_streak=1,
        isolation_monitor=AdmittingIsolationMonitor(),
        on_processed=lambda sequence, generation, dsp_ok, vad_ok: (
            transport.acknowledge_capture(
                sequence,
                clock_generation=generation,
                dsp_ok=dsp_ok,
                vad_ok=vad_ok,
            )
        ),
    )
    await transport.start()

    backend.stream.emit_capture(SILENCE)
    first = transport.pop_capture()
    first_render = transport.pop_render_reference()
    assert first is not None
    assert first_render is not None
    await preprocessor.process_capture(
        first,
        render_frames=(first_render,),
        assistant_rendering=True,
    )
    assert preprocessor.health is AecHealth.HEALTHY

    aec.raise_on_capture = True
    backend.stream.emit_capture(SILENCE)
    failed = transport.pop_capture()
    failed_render = transport.pop_render_reference()
    assert failed is not None
    assert failed_render is not None
    assert (
        await preprocessor.process_capture(
            failed,
            render_frames=(failed_render,),
            assistant_rendering=True,
        )
        is None
    )

    with pytest.raises(CaptureDrainError, match="acknowledgement"):
        await transport.drain_capture_through(failed.started_ns)


@pytest.mark.asyncio
async def test_sticky_timing_fault_cannot_be_overridden_by_success_ack() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()
    for _ in range(50):
        backend.stream.emit_capture(SILENCE)
        baseline = transport.pop_capture()
        assert baseline is not None
        transport.acknowledge_capture(
            baseline.sequence,
            clock_generation=baseline.clock_generation,
            dsp_ok=True,
            vad_ok=True,
        )
    backend.stream.emit_capture(
        SILENCE,
        status=SimpleNamespace(input_overflow=True),
    )
    failed = transport.pop_capture()
    assert failed is not None

    transport.acknowledge_capture(
        failed.sequence,
        clock_generation=failed.clock_generation,
        dsp_ok=True,
        vad_ok=True,
    )

    with pytest.raises(CaptureDrainError, match="discontinuity"):
        await transport.drain_capture_through(failed.started_ns)


@pytest.mark.asyncio
async def test_drain_fails_closed_on_processing_sequence_gap() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()
    backend.stream.emit_capture(SILENCE)
    frame = transport.pop_capture()
    assert frame is not None
    transport.acknowledge_capture(
        frame.sequence + 1,
        clock_generation=frame.clock_generation,
        dsp_ok=True,
        vad_ok=True,
    )

    with pytest.raises(CaptureDrainError, match="sequence gap"):
        await transport.drain_capture_through(frame.started_ns)


@pytest.mark.asyncio
async def test_drain_fails_closed_when_route_resets_while_waiting() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    await transport.start()

    pending = asyncio.create_task(transport.drain_capture_through(100_000_000))
    await asyncio.sleep(0)
    await transport.notify_route_changed(RouteKind.INPUT)

    with pytest.raises(CaptureDrainError, match="route reset"):
        await asyncio.wait_for(pending, timeout=0.1)


@pytest.mark.asyncio
async def test_old_frame_and_ack_cannot_poison_rebuilt_generation() -> None:
    admitted: list[AudioFrame] = []
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    preprocessor = VoicePreprocessor(
        aec=HealthyAec(),
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        isolation_monitor=AdmittingIsolationMonitor(),
        on_processed=lambda sequence, generation, dsp_ok, vad_ok: (
            transport.acknowledge_capture(
                sequence,
                clock_generation=generation,
                dsp_ok=dsp_ok,
                vad_ok=vad_ok,
            )
        ),
        healthy_streak=1,
    )
    await transport.start()
    backend.stream.emit_capture(SILENCE)
    old_frame = transport.pop_capture()
    old_render = transport.pop_render_reference()
    assert old_frame is not None
    assert old_render is not None

    await transport.notify_route_changed(RouteKind.DUPLEX)
    await transport.start()
    new_generation = transport.clock_generation
    preprocessor.reset_for_device_route(new_generation)
    backend.stream.emit_capture(SILENCE)
    current = transport.pop_capture()
    current_render = transport.pop_render_reference()
    assert current is not None
    assert current_render is not None

    assert (
        await preprocessor.process_capture(old_frame, assistant_rendering=True) is None
    )
    transport.acknowledge_capture(
        old_frame.sequence,
        clock_generation=old_frame.clock_generation,
        dsp_ok=False,
        vad_ok=False,
    )
    await preprocessor.process_capture(
        current,
        render_frames=(current_render,),
        assistant_rendering=True,
    )

    backend.stream.emit_capture(SILENCE)
    later = transport.pop_capture()
    later_render = transport.pop_render_reference()
    assert later is not None
    assert later_render is not None
    await preprocessor.process_capture(
        later,
        render_frames=(later_render,),
        assistant_rendering=True,
    )
    receipt = await transport.drain_capture_through(current.started_ns)

    assert receipt.clock_generation == new_generation
    assert preprocessor.health is AecHealth.HEALTHY
    assert admitted
    assert all(frame.clock_generation == new_generation for frame in admitted)


@pytest.mark.asyncio
async def test_drain_fails_across_a_truncated_timestamp_gap() -> None:
    backend = FakeDuplexBackend()
    clock = ManualClock()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=clock,
        capture_capacity=1,
    )
    await transport.start()

    backend.stream.emit_capture(
        SILENCE,
        input_adc_time=0.00,
        current_time=0.01,
        output_dac_time=0.02,
    )
    first = transport.pop_capture()
    assert first is not None
    transport.acknowledge_capture(
        first.sequence,
        clock_generation=first.clock_generation,
        dsp_ok=True,
        vad_ok=True,
    )

    for index in range(1, 17):
        clock.advance(11_000_000 if index == 1 else FRAME_DURATION_NS)
        input_time = 0.011 + (index - 1) * 0.01
        backend.stream.emit_capture(
            SILENCE,
            input_adc_time=input_time,
            current_time=input_time + 0.01,
            output_dac_time=input_time + 0.02,
        )
        captured = transport.pop_capture()
        assert captured is not None
        transport.acknowledge_capture(
            captured.sequence,
            clock_generation=captured.clock_generation,
            dsp_ok=True,
            vad_ok=True,
        )

    with pytest.raises(CaptureDrainError, match="history"):
        await transport.drain_capture_through(first.ended_ns + 500_000)


@pytest.mark.asyncio
async def test_failed_stream_start_fences_callback_and_allows_retry() -> None:
    backend = FakeDuplexBackend(start_failures=1)
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())

    with pytest.raises(RuntimeError, match="start failure"):
        await transport.start()

    failed_stream = backend.stream
    failed_generation = transport.clock_generation
    assert transport.queue_render(TONE) is None
    assert failed_stream.emit_capture(SILENCE) == SILENCE
    assert failed_stream.closed is True

    await transport.start()

    assert transport.clock_generation == failed_generation
    assert backend.stream is not failed_stream
    assert backend.stream.started is True
    assert transport.queue_render(TONE) is not None


@pytest.mark.asyncio
async def test_synchronous_start_callback_failure_is_never_erased() -> None:
    backend = FakeDuplexBackend(synchronous_start_capture=b"invalid")
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())

    await transport.start()

    with pytest.raises(CaptureDrainError, match="format discontinuity"):
        await asyncio.wait_for(
            transport.drain_capture_through(0),
            timeout=0.05,
        )


@pytest.mark.asyncio
async def test_start_time_invalid_frame_taints_every_later_capture() -> None:
    backend = FakeDuplexBackend(synchronous_start_capture=b"invalid")
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=HealthyAec(),
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        healthy_streak=1,
    )
    await transport.start()

    backend.stream.emit_capture(SILENCE)
    later = transport.pop_capture()

    assert later is not None and later.delay_evidence is not None
    assert later.discontinuity is True
    assert later.delay_evidence.timing_discontinuity is True
    result = await preprocessor.process_capture(later, assistant_rendering=True)
    assert result is None
    assert admitted == []
    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX
