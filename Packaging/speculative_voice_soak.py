"""Bounded, content-free lifecycle soaks for speculative duplex voice."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import asdict
import math
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from tldw_chatbook.Audio.duplex_contracts import DuplexBufferOccupancy
from tldw_chatbook.Audio.duplex_transport import (
    FRAME_BYTES,
    FRAME_DURATION_NS,
    FRAME_SAMPLES,
    DuplexAudioTransport,
)
from tldw_chatbook.Chat.console_voice_attempts import AttemptCleanupManager
from tldw_chatbook.Chat.console_voice_supervisor import VoiceDispatchSupervisor


_SILENCE = bytes(FRAME_BYTES)
_TONE = b"\x01\x00" * FRAME_SAMPLES
_MAX_DURATION_SECONDS = 4 * 60 * 60
_MAX_TASK_DELTA = 16
_NATIVE_PROCESS_PROGRAM = (
    "import tldw_voice_aec;"
    "p=tldw_voice_aec.AecProcessor(sample_rate=48000,channels=1);"
    "f=bytes(960);"
    "p.analyze_render(f,delay_ms=20);"
    "assert len(p.process_capture(f,delay_ms=20))==960"
)


class _SoakClock:
    def __init__(self) -> None:
        self.now_ns = FRAME_DURATION_NS

    def __call__(self) -> int:
        return self.now_ns

    def advance(self) -> None:
        self.now_ns += FRAME_DURATION_NS


class _SoakStream:
    def __init__(
        self, callback: Callable[[bytes, object, object, bool], bytes]
    ) -> None:
        self.callback = callback
        self.latency = (0.01, 0.01)
        self.started = False
        self.stopped = False
        self.closed = False
        self.frame_index = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def emit(self) -> bytes:
        self.frame_index += 1
        current = self.frame_index * 0.01
        return self.callback(
            _SILENCE,
            SimpleNamespace(
                inputBufferAdcTime=current - 0.01,
                currentTime=current,
                outputBufferDacTime=current + 0.01,
            ),
            False,
            False,
        )


class _SoakBackend:
    def __init__(self) -> None:
        self.streams: list[_SoakStream] = []

    @property
    def stream(self) -> _SoakStream:
        return self.streams[-1]

    def open_stream(
        self,
        *,
        sample_rate: int,
        channels: int,
        frame_samples: int,
        callback: Callable[[bytes, object, object, bool], bytes],
    ) -> _SoakStream:
        if (sample_rate, channels, frame_samples) != (48_000, 1, FRAME_SAMPLES):
            raise ValueError("unexpected soak transport format")
        stream = _SoakStream(callback)
        self.streams.append(stream)
        return stream

    def handle_report(self) -> dict[str, int]:
        return {
            "opened": len(self.streams),
            "stopped": sum(stream.stopped for stream in self.streams),
            "closed": sum(stream.closed for stream in self.streams),
            "leaked": sum(
                stream.started and not stream.closed for stream in self.streams
            ),
        }


class _CancellationSoakAttempt:
    """Provider-boundary fake driven through the real cleanup manager."""

    def __init__(self, epoch: int) -> None:
        self.attempt_epoch = epoch
        self._release = asyncio.Event()
        self._provider_task = asyncio.create_task(self._release.wait())
        self._force_close_task: asyncio.Task[None] | None = None
        self._fenced = False
        self.detached = False

    @property
    def provider_cleanup_task(self) -> asyncio.Task[None]:
        return self._provider_task

    @property
    def tts_cleanup_task(self) -> None:
        return None

    def request_cancellation(self) -> None:
        self._fenced = True

    def start_force_close_transport(self) -> asyncio.Task[None]:
        if self._force_close_task is None:
            self._force_close_task = asyncio.create_task(asyncio.sleep(0))
        return self._force_close_task

    def detach_content(self) -> None:
        self.detached = True
        self._fenced = True

    def callback_accepted(self) -> bool:
        return not self._fenced

    def release(self) -> None:
        self._release.set()


def _validate_duration(duration_seconds: float, cycle_interval_seconds: float) -> None:
    for name, value in (
        ("duration", duration_seconds),
        ("cycle interval", cycle_interval_seconds),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 < duration_seconds <= _MAX_DURATION_SECONDS:
        raise ValueError("duration must be positive and no more than four hours")
    if not 0 <= cycle_interval_seconds <= 1.0:
        raise ValueError("cycle interval must be between zero and one second")


def _task_count() -> int:
    return sum(not task.done() for task in asyncio.all_tasks())


def _max_occupancy(
    current: DuplexBufferOccupancy,
    maximum: DuplexBufferOccupancy,
) -> DuplexBufferOccupancy:
    return DuplexBufferOccupancy(
        capture_frames=max(current.capture_frames, maximum.capture_frames),
        render_frames=max(current.render_frames, maximum.render_frames),
        render_reference_frames=max(
            current.render_reference_frames,
            maximum.render_reference_frames,
        ),
        control_events=max(current.control_events, maximum.control_events),
    )


def run_native_process_probe(*, timeout_seconds: float = 5.0) -> dict[str, object]:
    """Load and use the compiled AEC in a real child, then prove it was reaped."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 30.0
    ):
        raise ValueError("native process timeout must be between zero and 30 seconds")
    forced = False
    try:
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter and program
            (sys.executable, "-c", _NATIVE_PROCESS_PROGRAM),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception as exc:
        return {
            "started": False,
            "error_class": type(exc).__name__,
            "forced_termination": False,
            "reaped": True,
            "passed": False,
        }
    try:
        exit_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        forced = True
        try:
            process.terminate()
        except OSError:
            pass
        try:
            exit_code = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                exit_code = process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                exit_code = -1
    reaped = process.poll() is not None
    return {
        "started": True,
        "exit_code": int(exit_code),
        "forced_termination": forced,
        "reaped": reaped,
        "passed": exit_code == 0 and not forced and reaped,
    }


async def run_duplex_soak(
    *,
    duration_seconds: float,
    cycle_interval_seconds: float = 0.01,
    route_change_every: int = 1_000,
    native_probe: Callable[[], Mapping[str, object]] = run_native_process_probe,
) -> dict[str, object]:
    """Exercise callback fencing, bounded rings, and handle closure for real time."""

    _validate_duration(duration_seconds, cycle_interval_seconds)
    if type(route_change_every) is not int or route_change_every <= 0:
        raise ValueError("route-change interval must be a positive integer")
    process_report = dict(native_probe())
    backend = _SoakBackend()
    clock = _SoakClock()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=clock,
        capture_capacity=8,
        render_capacity=8,
        control_capacity=2,
    )
    baseline_tasks = _task_count()
    maximum_tasks = baseline_tasks
    maximum_occupancy = DuplexBufferOccupancy(0, 0, 0, 0)
    post_fence_callbacks_accepted = 0
    iterations = 0
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    deadline = started_at + duration_seconds
    await transport.start()
    try:
        while loop.time() < deadline or iterations == 0:
            if transport.queue_render(_TONE) is None:
                raise RuntimeError("soak render admission unexpectedly closed")
            maximum_occupancy = _max_occupancy(
                transport.buffer_occupancy,
                maximum_occupancy,
            )
            stream = backend.stream
            stream.emit()
            maximum_occupancy = _max_occupancy(
                transport.buffer_occupancy,
                maximum_occupancy,
            )
            transport.pop_capture()
            transport.pop_render_reference()
            iterations += 1
            if iterations % route_change_every == 0:
                await transport.close()
                fenced = transport.buffer_occupancy
                if stream.emit() != _SILENCE or transport.buffer_occupancy != fenced:
                    post_fence_callbacks_accepted += 1
                transport.pop_control_event()
                await transport.start()
            maximum_tasks = max(maximum_tasks, _task_count())
            clock.advance()
            await asyncio.sleep(cycle_interval_seconds)
    except BaseException:
        await transport.close()
        raise

    final_stream = backend.stream
    await transport.close()
    fenced = transport.buffer_occupancy
    if final_stream.emit() != _SILENCE or transport.buffer_occupancy != fenced:
        post_fence_callbacks_accepted += 1
    transport.pop_control_event()
    await asyncio.sleep(0)
    elapsed_seconds = loop.time() - started_at
    final_occupancy = transport.buffer_occupancy
    capacities = transport.buffer_capacities
    handles = backend.handle_report()
    final_tasks = _task_count()
    overflow_count = (
        transport.capture_overflows
        + transport.render_overflows
        + transport.render_reference_overflows
    )
    bounded = all(
        current <= limit
        for current, limit in zip(
            asdict(maximum_occupancy).values(),
            asdict(capacities).values(),
            strict=True,
        )
    )
    passed = all(
        (
            elapsed_seconds >= duration_seconds,
            process_report.get("passed") is True,
            bounded,
            not any(asdict(final_occupancy).values()),
            overflow_count == 0,
            post_fence_callbacks_accepted == 0,
            handles["leaked"] == 0,
            final_tasks <= baseline_tasks,
            maximum_tasks - baseline_tasks <= _MAX_TASK_DELTA,
        )
    )
    return {
        "requested_duration_seconds": duration_seconds,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "iterations": iterations,
        "native_process": process_report,
        "audio_buffers": {
            "capacities": asdict(capacities),
            "maximum": asdict(maximum_occupancy),
            "final": asdict(final_occupancy),
            "overflow_count": overflow_count,
        },
        "tasks": {
            "baseline": baseline_tasks,
            "maximum": maximum_tasks,
            "final": final_tasks,
            "maximum_delta_limit": _MAX_TASK_DELTA,
        },
        "device_handles": handles,
        "post_fence_callbacks_accepted": post_fence_callbacks_accepted,
        "passed": passed,
    }


async def _never_observe_exit(_task: asyncio.Task[Any], _timeout: float) -> bool:
    await asyncio.sleep(0)
    return False


async def run_cancellation_soak(
    *,
    duration_seconds: float,
    cycle_interval_seconds: float = 0.01,
    native_probe: Callable[[], Mapping[str, object]] = run_native_process_probe,
) -> dict[str, object]:
    """Drive cancellation-resistant work through the real bounded supervisor."""

    _validate_duration(duration_seconds, cycle_interval_seconds)
    process_report = dict(native_probe())
    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=_never_observe_exit)
    baseline_tasks = _task_count()
    maximum_tasks = baseline_tasks
    maximum_orphans = 0
    post_fence_callbacks_accepted = 0
    cancellations = 0
    epoch = 0
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    deadline = started_at + duration_seconds
    while loop.time() < deadline or cancellations == 0:
        attempts = (
            _CancellationSoakAttempt(epoch),
            _CancellationSoakAttempt(epoch + 1),
        )
        epoch += 2
        try:
            outcomes = tuple(
                manager.cancel(attempt)  # type: ignore[arg-type]
                for attempt in attempts
            )
            await asyncio.gather(*outcomes)
            cancellations += len(attempts)
            maximum_orphans = max(maximum_orphans, supervisor.orphan_count)
            maximum_tasks = max(maximum_tasks, _task_count())
            post_fence_callbacks_accepted += sum(
                attempt.callback_accepted() for attempt in attempts
            )
        finally:
            for attempt in attempts:
                attempt.release()
            await asyncio.gather(
                *(attempt.provider_cleanup_task for attempt in attempts),
                return_exceptions=True,
            )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        maximum_tasks = max(maximum_tasks, _task_count())
        await asyncio.sleep(cycle_interval_seconds)

    await asyncio.sleep(0)
    elapsed_seconds = loop.time() - started_at
    final_tasks = _task_count()
    final_orphans = supervisor.orphan_count
    passed = all(
        (
            elapsed_seconds >= duration_seconds,
            process_report.get("passed") is True,
            maximum_orphans <= 2,
            final_orphans == 0,
            manager.obsolete_cleanup_count == 0,
            post_fence_callbacks_accepted == 0,
            final_tasks <= baseline_tasks,
            maximum_tasks - baseline_tasks <= _MAX_TASK_DELTA,
        )
    )
    return {
        "requested_duration_seconds": duration_seconds,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "iterations": cancellations // 2,
        "cancellations": cancellations,
        "native_process": process_report,
        "orphans": {"maximum": maximum_orphans, "final": final_orphans, "limit": 2},
        "obsolete_cleanups_final": manager.obsolete_cleanup_count,
        "tasks": {
            "baseline": baseline_tasks,
            "maximum": maximum_tasks,
            "final": final_tasks,
            "maximum_delta_limit": _MAX_TASK_DELTA,
        },
        "post_fence_callbacks_accepted": post_fence_callbacks_accepted,
        "passed": passed,
    }


__all__ = [
    "run_cancellation_soak",
    "run_duplex_soak",
    "run_native_process_probe",
]
