"""App-owned full-duplex callback bridge with one fenced clock domain."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import replace
import math
import threading
import time
from typing import Protocol, TypeVar, cast

from .duplex_contracts import (
    AecDelayEvidence,
    AudioFrame,
    CaptureDrainError,
    DeviceRouteChanged,
    DrainReceipt,
    DuplexBufferOccupancy,
    RenderBoundary,
    RenderSubmission,
    RouteKind,
)
from .native_duplex_stream import NativeDuplexStream

PROCESSING_SAMPLE_RATE = 48_000
PROCESSING_CHANNELS = 1
FRAME_DURATION_NS = 10_000_000
FRAME_SAMPLES = 480
FRAME_BYTES = FRAME_SAMPLES * 2
_SILENCE = bytes(FRAME_BYTES)
_NANOSECONDS_PER_SECOND = 1_000_000_000
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_TIMING_DRIFT_TOLERANCE_NS = 20_000_000
_DRIFT_BASELINE_SAMPLES = 5
_DRIFT_REQUIRED_OBSERVATIONS = 3
_FRAME_CADENCE_JITTER_TOLERANCE_NS = 2_000_000
# CoreAudio may prime a newly opened full-duplex route with a short burst of
# status-flagged callbacks. Bound that exception to 500 ms; later faults remain fatal.
_STARTUP_PREROLL_CALLBACKS = 500_000_000 // FRAME_DURATION_NS
_STATUS_FLAG_NAMES = (
    "input_underflow",
    "input_overflow",
    "output_underflow",
    "output_overflow",
    "priming_output",
)

_StateWait = Callable[[], Awaitable[None]]
_T = TypeVar("_T")


async def _poll_state() -> None:
    """Yield outside the device callback before re-checking drain state."""

    await asyncio.sleep(0.001)


async def _run_owned_thread(
    operation: Callable[[], _T],
) -> tuple[_T | None, Exception | None, asyncio.CancelledError | None]:
    """Run one native operation to completion without orphaning it on cancellation."""

    task = asyncio.create_task(asyncio.to_thread(operation))
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
        except Exception:
            break
    try:
        return task.result(), None, cancellation
    except Exception as error:
        return None, error, cancellation


def _time_value(time_info: object, name: str) -> float:
    if isinstance(time_info, dict):
        value = time_info[name]
    else:
        value = getattr(time_info, name)
    seconds = float(value)
    if not math.isfinite(seconds):
        raise ValueError(f"non-finite callback time {name}")
    return seconds


def _status_flags(status: object) -> tuple[str, ...]:
    if isinstance(status, tuple):
        return status
    flags = tuple(
        name for name in _STATUS_FLAG_NAMES if bool(getattr(status, name, False))
    )
    if flags or not bool(status):
        return flags
    return ("callback_status",)


def _validated_stream_latency(value: object) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("duplex latency must contain input and output seconds")
    if any(isinstance(item, bool) for item in value):
        raise TypeError("duplex latency values must be numeric")
    latency = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) and item >= 0.0 for item in latency):
        raise ValueError("duplex latency values must be finite and non-negative")
    return latency


class _DuplexStream(Protocol):
    latency: object

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class _DuplexBackend(Protocol):
    def open_stream(
        self,
        *,
        sample_rate: int,
        channels: int,
        frame_samples: int,
        callback: Callable[[bytes, object, object, bool], bytes],
        device_pair: tuple[int, int] | None = None,
    ) -> _DuplexStream: ...


class DuplexAudioTransport:
    """Own one bounded render/capture session and its monotonic generation.

    Production callbacks run entirely in the native bridge. Python drains paired
    records on demand; only explicitly injected test backends enter Python.
    """

    def __init__(
        self,
        *,
        backend: _DuplexBackend | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
        capture_capacity: int = 64,
        render_capacity: int = 64,
        control_capacity: int = 8,
        state_wait: _StateWait | None = None,
    ) -> None:
        if min(capture_capacity, render_capacity, control_capacity) <= 0:
            raise ValueError("duplex ring capacities must be positive")
        self._backend = backend
        self._clock = clock
        self._capture_capacity = capture_capacity
        self._render_capacity = render_capacity
        self._control_capacity = control_capacity
        self._lock = threading.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._teardown_owners: set[asyncio.Task[None]] = set()
        self._teardown_pending_generations: set[int] = set()
        self._capture_ring: deque[AudioFrame] = deque()
        self._render_ring: deque[tuple[RenderSubmission, bytes]] = deque()
        self._render_reference_ring: deque[AudioFrame] = deque()
        self._control_ring: deque[DeviceRouteChanged] = deque()
        self._capture_history: deque[AudioFrame] = deque(
            maxlen=max(16, capture_capacity * 4)
        )
        self._history_truncated_through_ns = -1
        self._stream: _DuplexStream | None = None
        self._stream_latency_seconds: tuple[float, float] | None = None
        self._clock_generation = 0
        self._next_capture_sequence = 0
        self._next_render_sequence = 0
        self._output_epoch = 0
        self._latest_render_boundary: RenderBoundary | None = None
        self._playback_intervals: deque[tuple[int, int]] = deque()
        self._playback_history_lost_through_ns = -1
        self._native_startup_discarded = 0
        self._native_faults: dict[str, int] = {}
        self._last_capture_ended_ns = -1
        self._last_capture_started_ns = -1
        self._last_render_reference_sequence: int | None = None
        self._drift_offset_samples: deque[int] = deque(maxlen=_DRIFT_BASELINE_SAMPLES)
        self._drift_baseline_offset_ns: int | None = None
        self._drift_violation_count = 0
        self._last_input_adc_ns: int | None = None
        self._last_output_dac_ns: int | None = None
        self._device_to_monotonic_offset_ns: int | None = None
        self._last_dsp_sequence = -1
        self._last_vad_sequence = -1
        self._generation_sequence_floor = 0
        self._capture_overflows = 0
        self._render_overflows = 0
        self._render_reference_overflows = 0
        self._control_overflows = 0
        self._stale_callbacks = 0
        self._startup_callbacks_seen = 0
        self._teardown_failures = 0
        self._drain_failure: str | None = None
        self._generation_overflowed = False
        self._render_admission_open = False
        self._close_requested = False
        self._state_wait = state_wait or _poll_state

    @property
    def clock_generation(self) -> int:
        with self._lock:
            return self._clock_generation

    @property
    def capture_count(self) -> int:
        with self._lock:
            if isinstance(self._stream, NativeDuplexStream):
                return self._stream.bridge.snapshot()["capture_occupancy"]
            return len(self._capture_ring)

    @property
    def buffer_occupancy(self) -> DuplexBufferOccupancy:
        """Return content-free current occupancy for every callback ring."""

        with self._lock:
            snapshot = self._native_snapshot_locked()
            return DuplexBufferOccupancy(
                capture_frames=snapshot.get(
                    "capture_occupancy", len(self._capture_ring)
                ),
                render_frames=snapshot.get("render_occupancy", len(self._render_ring)),
                render_reference_frames=len(self._render_reference_ring),
                control_events=len(self._control_ring),
            )

    @property
    def buffer_capacities(self) -> DuplexBufferOccupancy:
        """Return the configured hard bounds for every callback ring."""

        return DuplexBufferOccupancy(
            capture_frames=self._capture_capacity,
            render_frames=self._render_capacity,
            render_reference_frames=self._render_capacity,
            control_events=self._control_capacity,
        )

    @property
    def capture_overflows(self) -> int:
        with self._lock:
            return self._capture_overflows + self._native_snapshot_locked().get(
                "capture_overflows", 0
            )

    @property
    def render_overflows(self) -> int:
        with self._lock:
            return self._render_overflows

    @property
    def render_reference_overflows(self) -> int:
        with self._lock:
            return self._render_reference_overflows

    @property
    def control_overflows(self) -> int:
        """Return the number of bounded control-ring overwrites."""

        with self._lock:
            return self._control_overflows

    @property
    def stale_callbacks(self) -> int:
        """Return callbacks rejected after their stream generation was fenced."""

        with self._lock:
            return self._stale_callbacks

    @property
    def teardown_failures(self) -> int:
        """Return cumulative native stop/close failures."""

        with self._lock:
            return self._teardown_failures

    @property
    def is_open(self) -> bool:
        """Return whether the current route owns a live device stream."""

        with self._lock:
            return self._stream is not None

    @property
    def stream_latency_seconds(self) -> tuple[float, float] | None:
        """Return content-free actual input/output latency for the live stream."""

        with self._lock:
            return self._stream_latency_seconds

    async def start(self, *, device_pair: tuple[int, int] | None = None) -> None:
        """Open one 48 kHz mono raw session for the current generation."""

        if device_pair is not None and (
            type(device_pair) is not tuple
            or len(device_pair) != 2
            or any(type(index) is not int for index in device_pair)
        ):
            raise TypeError("duplex device pair must contain two integer indices")
        if device_pair is not None and min(device_pair) < 0:
            raise ValueError("duplex device indices must be non-negative")

        await self._wait_for_teardown_owners()
        async with self._lifecycle_lock:
            await self._start_locked(device_pair)

    async def _wait_for_teardown_owners(self) -> None:
        """Keep a replacement open behind every already-owned route teardown."""

        while self._teardown_owners:
            owners = tuple(self._teardown_owners)
            completed = tuple(owner for owner in owners if owner.done())
            for owner in completed:
                self._release_teardown_owner(owner)
            for owner in completed:
                owner.result()
            for owner in owners:
                if owner.done():
                    continue
                try:
                    await asyncio.shield(owner)
                finally:
                    if owner.done():
                        self._release_teardown_owner(owner)
                owner.result()

    async def _start_locked(self, device_pair: tuple[int, int] | None) -> None:
        """Open and start a stream while the lifecycle lock is held."""

        with self._lock:
            if self._close_requested:
                raise RuntimeError("duplex audio session is closed")
            if self._stream is not None:
                return
            generation = self._clock_generation
        backend = self._backend
        if backend is None:
            await self._start_native_locked(generation, device_pair)
            return

        def callback(
            pcm16: bytes,
            time_info: object,
            status: object,
            discontinuity: bool = False,
        ) -> bytes:
            return self._audio_callback(
                generation,
                pcm16,
                time_info,
                status,
                discontinuity,
            )

        def open_stream() -> _DuplexStream:
            if device_pair is None:
                return backend.open_stream(
                    sample_rate=PROCESSING_SAMPLE_RATE,
                    channels=PROCESSING_CHANNELS,
                    frame_samples=FRAME_SAMPLES,
                    callback=callback,
                )
            return backend.open_stream(
                sample_rate=PROCESSING_SAMPLE_RATE,
                channels=PROCESSING_CHANNELS,
                frame_samples=FRAME_SAMPLES,
                callback=callback,
                device_pair=device_pair,
            )

        opened, open_error, cancellation = await _run_owned_thread(open_stream)
        if open_error is not None:
            if cancellation is not None:
                raise cancellation from open_error
            raise open_error
        stream = cast(_DuplexStream, opened)
        if cancellation is not None:
            teardown_cancellation = await self._teardown_stream(stream)
            raise cancellation from teardown_cancellation

        try:
            stream_latency = _validated_stream_latency(stream.latency)
        except (AttributeError, TypeError, ValueError):
            teardown_cancellation = await self._teardown_stream(stream)
            if teardown_cancellation is not None:
                raise teardown_cancellation
            raise ValueError("duplex stream reported invalid actual latency") from None

        with self._lock:
            stale_open = (
                generation != self._clock_generation or self._stream is not None
            )
            if not stale_open:
                self._stream = stream
                self._stream_latency_seconds = stream_latency
                self._drain_failure = None
        if stale_open:
            teardown_cancellation = await self._teardown_stream(stream)
            if teardown_cancellation is not None:
                raise teardown_cancellation
            return

        _, start_error, cancellation = await _run_owned_thread(stream.start)
        if start_error is not None or cancellation is not None:
            owns_teardown = False
            with self._lock:
                if self._stream is stream:
                    self._teardown_pending_generations.add(generation)
                    self._stream = None
                    self._stream_latency_seconds = None
                    self._render_admission_open = False
                    self._fence_generation_locked("audio stream start failed")
                    owns_teardown = True
            if owns_teardown:
                teardown_cancellation = await self._teardown_stream(
                    stream,
                    generation=generation,
                )
                cancellation = cancellation or teardown_cancellation
            if cancellation is not None:
                raise cancellation from start_error
            if start_error is not None:
                raise start_error
            raise RuntimeError("duplex stream start failed without an error")
        with self._lock:
            if (
                generation == self._clock_generation
                and self._stream is stream
                and not self._close_requested
            ):
                self._render_admission_open = True

    async def close(self) -> None:
        """Close the live session and fence its callback generation."""

        await self.notify_route_changed(RouteKind.DUPLEX)

    async def _start_native_locked(
        self, generation: int, device_pair: tuple[int, int] | None
    ) -> None:
        stream = NativeDuplexStream(
            generation=generation,
            capture_capacity=self._capture_capacity,
            render_capacity=self._render_capacity,
            device_pair=device_pair,
            transport_clock=self._clock,
        )
        with self._lock:
            if self._close_requested:
                raise RuntimeError("duplex audio session is closed")
            # Publish ownership atomically with dispatch, so a concurrent facade
            # fence cannot miss an owner that is about to activate capture.
            stream.begin()
            self._stream = stream
            self._native_faults.clear()
        try:
            await stream.wait_started()
        except BaseException:
            stream.request_close()
            with self._lock:
                if self._stream is stream:
                    self._stream = None
                    self._fence_generation_locked("audio stream start failed")
            await stream.wait_closed()
            raise
        with self._lock:
            if (
                generation == self._clock_generation
                and self._stream is stream
                and not self._close_requested
            ):
                self._stream_latency_seconds = stream.latency
                self._drain_failure = None
                self._render_admission_open = True
                stream.bridge.set_render_admission(True)

    async def abort_output(self) -> None:
        """Discard queued render PCM without blocking the callback."""

        self.fence_output()

    def fence_output(self) -> None:
        """Invalidate queued PCM synchronously while keeping capture/render open."""

        with self._lock:
            self._render_ring.clear()
            self._latest_render_boundary = None
            self._output_epoch += 1
            if isinstance(self._stream, NativeDuplexStream):
                self._output_epoch = self._stream.bridge.abort_output()

    def queue_render(self, pcm16: bytes) -> RenderSubmission | None:
        """Schedule one normalized ten-millisecond render frame.

        Returns a submission identity. Only ``pop_render_reference`` returns
        a frame proven to have been scheduled to the device and suitable for AEC
        analysis. A full ring or fenced route rejects rather than overwriting PCM.
        """

        if len(pcm16) != FRAME_BYTES:
            raise ValueError("render PCM must be one 48 kHz mono 10 ms frame")
        copied = bytes(pcm16)
        with self._lock:
            if not self._render_admission_open:
                return None
            if isinstance(self._stream, NativeDuplexStream):
                self._output_epoch = self._stream.bridge.output_epoch
                submission = RenderSubmission(
                    self._clock_generation,
                    self._output_epoch,
                    self._next_render_sequence,
                )
                if not self._stream.bridge.queue_render(
                    copied, submission.submission_id, submission.output_epoch
                ):
                    self._render_overflows += 1
                    return None
                self._next_render_sequence += 1
                return submission
            if len(self._render_ring) >= self._render_capacity:
                self._render_overflows += 1
                return None
            submission = RenderSubmission(
                self._clock_generation, self._output_epoch, self._next_render_sequence
            )
            self._next_render_sequence += 1
            self._render_ring.append((submission, copied))
            return submission

    def render_boundary(self, submission: RenderSubmission) -> RenderBoundary | None:
        """Read the latest actual DAC receipt; obsolete or faulted output fails."""
        with self._lock:
            snapshot = self._native_snapshot_locked()
            epoch = snapshot.get("output_epoch", self._output_epoch)
            if (
                submission.generation != self._clock_generation
                or submission.output_epoch != epoch
                or self._drain_failure is not None
                or self._native_fatal_locked()
            ):
                raise CaptureDrainError("render submission invalidated")
            if isinstance(self._stream, NativeDuplexStream):
                receipt = self._stream.bridge.latest_render()
                if (
                    receipt is not None
                    and self._device_to_monotonic_offset_ns is not None
                ):
                    self._latest_render_boundary = RenderBoundary(
                        receipt["generation"],
                        receipt["output_epoch"],
                        receipt["submission_id"],
                        round(receipt["output_dac_end_time"] * _NANOSECONDS_PER_SECOND)
                        + self._device_to_monotonic_offset_ns,
                    )
            boundary = self._latest_render_boundary
            if boundary is not None and (
                boundary.generation,
                boundary.output_epoch,
                boundary.submission_id,
            ) == (
                submission.generation,
                submission.output_epoch,
                submission.submission_id,
            ):
                return boundary
            return None

    def pop_capture(self) -> AudioFrame | None:
        """Pop the oldest callback-stamped capture frame for DSP."""

        with self._lock:
            if isinstance(self._stream, NativeDuplexStream):
                record = self._stream.bridge.pop_capture()
                self._native_snapshot_locked()
                if record is None:
                    return None
                return self._ingest_native_locked(record)
            if not self._capture_ring:
                return None
            return self._capture_ring.popleft()

    def pop_render_reference(self) -> AudioFrame | None:
        """Pop the oldest render frame that was actually scheduled to hardware."""

        with self._lock:
            if not self._render_reference_ring:
                return None
            return self._render_reference_ring.popleft()

    def pop_control_event(self) -> DeviceRouteChanged | None:
        """Pop the oldest content-free route control event."""

        with self._lock:
            if not self._control_ring:
                return None
            return self._control_ring.popleft()

    def fence_audio_admission(self) -> None:
        """Silence queued output immediately from any thread, without teardown."""
        with self._lock:
            self._render_admission_open = False
            self._render_ring.clear()
            self._output_epoch += 1
            self._latest_render_boundary = None
            if isinstance(self._stream, NativeDuplexStream):
                self._stream.bridge.set_render_admission(False)

    def request_close(self) -> None:
        """Deactivate capture and start native checked close before async cleanup."""
        with self._lock:
            self._close_requested = True
            self._render_admission_open = False
            self._render_ring.clear()
            self._latest_render_boundary = None
            if isinstance(self._stream, NativeDuplexStream):
                self._stream.request_close()

    def notify_route_changed(self, route_kind: RouteKind) -> asyncio.Task[None]:
        """Fence synchronously and return a cancellable native-teardown waiter.

        The returned task does not own the teardown. Dropping or cancelling it
        cannot prevent the transport-retained owner from stopping and closing
        the detached stream.
        """

        if not isinstance(route_kind, RouteKind):
            raise TypeError("route_kind must be a content-free RouteKind")
        with self._lock:
            old_stream = self._stream
            if isinstance(old_stream, NativeDuplexStream):
                old_stream.request_close()
            self._stream = None
            self._render_admission_open = False
            old_generation = self._clock_generation
            if old_stream is not None:
                self._teardown_pending_generations.add(old_generation)
            self._fence_generation_locked("device route reset")
            event = DeviceRouteChanged(old_generation, route_kind)
            if len(self._control_ring) >= self._control_capacity:
                self._control_ring.popleft()
                self._control_overflows += 1
            self._control_ring.append(event)

        predecessors = tuple(self._teardown_owners)

        async def teardown_under_lifecycle_lock() -> None:
            if isinstance(old_stream, NativeDuplexStream):
                # Its daemon already serializes open/start/stop/close. Waiting
                # for the startup asyncio lock could outlive the close deadline.
                await self._teardown_stream(old_stream, generation=old_generation)
                return
            for predecessor in predecessors:
                await asyncio.shield(predecessor)
            async with self._lifecycle_lock:
                if old_stream is not None:
                    cancellation = await self._teardown_stream(
                        old_stream,
                        generation=old_generation,
                    )
                    if cancellation is not None:
                        raise cancellation

        owner = asyncio.create_task(teardown_under_lifecycle_lock())
        self._teardown_owners.add(owner)
        owner.add_done_callback(self._release_teardown_owner)

        async def wait_for_teardown() -> None:
            cancellation: asyncio.CancelledError | None = None
            while not owner.done():
                try:
                    await asyncio.shield(owner)
                except asyncio.CancelledError as error:
                    if cancellation is None:
                        cancellation = error
                except Exception:
                    break
            try:
                owner.result()
            except Exception as error:
                if cancellation is not None:
                    raise cancellation from error
                raise
            if cancellation is not None:
                raise cancellation

        return asyncio.create_task(wait_for_teardown())

    def _release_teardown_owner(self, owner: asyncio.Task[None]) -> None:
        """Release one completed owner while consuming any terminal exception."""

        self._teardown_owners.discard(owner)
        if not owner.cancelled():
            owner.exception()

    def acknowledge_capture(
        self,
        sequence: int,
        *,
        clock_generation: int,
        dsp_ok: bool,
        vad_ok: bool,
    ) -> None:
        """Advance downstream causal acknowledgements or record failure."""

        with self._lock:
            if clock_generation < self._clock_generation:
                return
            if clock_generation > self._clock_generation:
                self._drain_failure = "capture acknowledgement generation mismatch"
                return
            if not dsp_ok or not vad_ok:
                self._drain_failure = "capture processing acknowledgement failed"
            else:
                if sequence != self._last_dsp_sequence + 1:
                    self._drain_failure = "capture processing sequence gap"
                else:
                    self._last_dsp_sequence = sequence
                if sequence != self._last_vad_sequence + 1:
                    self._drain_failure = "capture VAD sequence gap"
                else:
                    self._last_vad_sequence = sequence

    async def drain_capture_through(self, render_boundary_ns: int) -> DrainReceipt:
        """Prove capture, DSP, and VAD completion through a render boundary."""

        with self._lock:
            generation = self._clock_generation
        while True:
            with self._lock:
                self._native_snapshot_locked()
                if generation != self._clock_generation:
                    raise CaptureDrainError("device route reset during capture drain")
                if self._drain_failure is not None:
                    raise CaptureDrainError(self._drain_failure)
                if (
                    self._history_truncated_through_ns >= 0
                    and render_boundary_ns <= self._history_truncated_through_ns
                ):
                    raise CaptureDrainError("capture history overflow")
                if self._last_capture_ended_ns > render_boundary_ns:
                    target_sequence: int | None = None
                    for frame in self._capture_history:
                        if frame.started_ns <= render_boundary_ns:
                            target_sequence = frame.sequence
                    if target_sequence is None:
                        if self._history_truncated_through_ns >= 0:
                            raise CaptureDrainError(
                                "capture history has a truncated timestamp gap"
                            )
                        target_sequence = self._generation_sequence_floor - 1
                    if (
                        self._last_dsp_sequence >= target_sequence
                        and self._last_vad_sequence >= target_sequence
                    ):
                        return DrainReceipt(
                            capture_watermark_ns=self._last_capture_ended_ns,
                            capture_sequence=target_sequence,
                            dsp_sequence=self._last_dsp_sequence,
                            vad_sequence=self._last_vad_sequence,
                            clock_generation=generation,
                        )
            await self._state_wait()

    @property
    def native_counters(self) -> dict[str, int]:
        """Expose every native fault, priming and occupancy counter without PCM."""
        with self._lock:
            return dict(self._native_snapshot_locked())

    def _native_snapshot_locked(self) -> dict[str, int]:
        if isinstance(self._stream, NativeDuplexStream):
            snapshot = self._stream.bridge.snapshot()
            for key in (
                "capture_overflows",
                "invalid_frames",
                "invalid_timing",
                "fatal_status_bits",
            ):
                self._native_faults[key] = (
                    self._native_faults.get(key, 0) | snapshot[key]
                    if key == "fatal_status_bits"
                    else max(self._native_faults.get(key, 0), snapshot[key])
                )
            self._generation_overflowed |= bool(snapshot["capture_overflows"])
            if self._native_fatal_locked() and self._drain_failure is None:
                self._drain_failure = "native audio callback fault"
            return snapshot
        return {}

    def _native_fatal_locked(self) -> bool:
        return any(self._native_faults.values())

    def _historical_playback_locked(self, capture_start: int, *, valid: bool) -> bool:
        """Classify from bounded actual DAC intervals, including cancelled output."""
        if not valid or capture_start < self._playback_history_lost_through_ns:
            return True  # Ambiguity always takes the closed playback path.
        while (
            self._playback_intervals and self._playback_intervals[0][1] <= capture_start
        ):
            self._playback_intervals.popleft()
        return any(
            start < capture_start + FRAME_DURATION_NS and end > capture_start
            for start, end in self._playback_intervals
        )

    def _ingest_native_locked(self, record: dict[str, object]) -> AudioFrame:
        stream = self._stream
        assert isinstance(stream, NativeDuplexStream)
        if record["generation"] != self._clock_generation:
            raise CaptureDrainError("native capture generation mismatch")
        if record["startup_discarded_before"] != self._native_startup_discarded:
            self._last_input_adc_ns = None
            self._last_output_dac_ns = None
            self._native_startup_discarded = record["startup_discarded_before"]
        bits = self._native_faults.get("fatal_status_bits", 0)
        flags = tuple(
            name for index, name in enumerate(_STATUS_FLAG_NAMES) if bits & (1 << index)
        )
        if bits & ~31:
            flags += ("callback_status",)
        for key in ("invalid_frames", "invalid_timing"):
            if self._native_faults.get(key):
                flags += ("native_" + key,)
        timing = self._map_timing_locked(
            dict(
                inputBufferAdcTime=record["input_adc_time"],
                currentTime=record["current_time"],
                outputBufferDacTime=record["output_dac_time"],
            ),
            flags,
            observed_ns=record["observed_ns"] + stream.native_to_python_offset_ns,
            format_discontinuity=self._drain_failure is not None,
            capture_occupancy_frames=record["capture_occupancy"],
            render_occupancy_frames=record["render_occupancy"],
            occupancy_bounded=not self._generation_overflowed,
        )
        return self._ingest_record_locked(
            pcm16=record["pcm16"],
            timing=timing,
            capture_sequence=self._generation_sequence_floor
            + record["capture_sequence"],
            output_pcm16=record["output_pcm16"],
            submission=None
            if record["submission_id"] is None
            else RenderSubmission(
                record["generation"], record["output_epoch"], record["submission_id"]
            ),
            context_valid=bool(record["playback_context_valid"])
            and not self._native_fatal_locked(),
        )

    def _ingest_record_locked(
        self,
        *,
        pcm16: bytes,
        timing: AecDelayEvidence,
        capture_sequence: int,
        output_pcm16: bytes,
        submission: RenderSubmission | None,
        context_valid: bool,
    ) -> AudioFrame:
        discontinuity = (
            timing.timing_discontinuity
            or timing.clock_drift
            or not timing.occupancy_bounded
            or not context_valid
        )
        started_ns = timing.capture_adc_ns
        if started_ns <= self._last_capture_started_ns:
            started_ns = self._last_capture_started_ns + FRAME_DURATION_NS
            discontinuity = True
        if capture_sequence != self._next_capture_sequence:
            discontinuity = True
        committed_render = (
            None
            if submission is None
            else RenderBoundary(
                submission.generation,
                submission.output_epoch,
                submission.submission_id,
                timing.render_dac_ns + FRAME_DURATION_NS,
            )
        )
        reference = AudioFrame(
            sequence=capture_sequence,
            started_ns=timing.render_dac_ns,
            ended_ns=timing.render_dac_ns + FRAME_DURATION_NS,
            pcm16=output_pcm16,
            clock_generation=self._clock_generation,
            delay_evidence=timing,
            committed_render=committed_render,
        )
        self._last_render_reference_sequence = capture_sequence
        if len(self._render_reference_ring) >= self._render_capacity:
            self._render_reference_overflows += 1
            self._generation_overflowed = True
            self._drain_failure = "render reference ring overflow"
            discontinuity = True
            timing = replace(timing, occupancy_bounded=False)
            if isinstance(self._stream, NativeDuplexStream):
                self._stream.bridge.set_render_admission(False)
        else:
            self._render_reference_ring.append(reference)
        if submission is not None:
            if len(self._playback_intervals) >= self._render_capacity:
                _, lost_end = self._playback_intervals.popleft()
                self._playback_history_lost_through_ns = max(
                    lost_end, self._playback_history_lost_through_ns
                )
            self._playback_intervals.append((reference.started_ns, reference.ended_ns))
            assert committed_render is not None
            self._latest_render_boundary = committed_render
        valid = context_valid and not discontinuity
        assistant_rendering = self._historical_playback_locked(started_ns, valid=valid)
        if started_ns < self._playback_history_lost_through_ns:
            discontinuity = True
        frame = AudioFrame(
            sequence=capture_sequence,
            started_ns=started_ns,
            ended_ns=started_ns + FRAME_DURATION_NS,
            pcm16=pcm16,
            clock_generation=self._clock_generation,
            discontinuity=discontinuity,
            delay_evidence=timing,
            render_reference_sequence=self._last_render_reference_sequence,
            assistant_rendering=assistant_rendering,
            committed_render=committed_render,
        )
        self._next_capture_sequence = capture_sequence + 1
        self._last_capture_started_ns = frame.started_ns
        self._last_capture_ended_ns = frame.ended_ns
        if len(self._capture_history) == self._capture_history.maxlen:
            self._history_truncated_through_ns = self._capture_history[0].ended_ns
        self._capture_history.append(frame)
        if discontinuity and self._drain_failure is None:
            self._drain_failure = "capture timing discontinuity"
        return frame

    def _audio_callback(
        self,
        generation: int,
        pcm16: bytes,
        time_info: object,
        status: object,
        discontinuity: bool,
    ) -> bytes:
        copied = bytes(pcm16)
        with self._lock:
            if (
                generation != self._clock_generation
                or self._stream is None
                or self._close_requested
            ):
                if generation not in self._teardown_pending_generations:
                    self._stale_callbacks += 1
                return _SILENCE
            if len(copied) != FRAME_BYTES:
                self._drain_failure = "capture frame format discontinuity"
                return _SILENCE
            self._startup_callbacks_seen += 1
            if (
                self._startup_callbacks_seen <= _STARTUP_PREROLL_CALLBACKS
                and _status_flags(status)
            ):
                self._last_input_adc_ns = None
                self._last_output_dac_ns = None
                return _SILENCE
            prior_failure = self._drain_failure is not None
            queued_render = (
                self._render_ring.popleft()
                if self._render_admission_open and self._render_ring
                else None
            )
            reference_overflow = (
                len(self._render_reference_ring) >= self._render_capacity
            )
            capture_overflow = len(self._capture_ring) >= self._capture_capacity
            # Keep data loss distinct from subsequent timing/processing errors.
            # The overflowing capture itself may never reach the consumer.
            self._generation_overflowed |= capture_overflow or reference_overflow
            timing = self._map_timing_locked(
                time_info,
                status,
                format_discontinuity=discontinuity or prior_failure,
                capture_occupancy_frames=len(self._capture_ring),
                render_occupancy_frames=len(self._render_reference_ring),
                occupancy_bounded=not self._generation_overflowed,
                observed_ns=self._clock(),
            )
            frame = self._ingest_record_locked(
                pcm16=copied,
                timing=timing,
                capture_sequence=self._next_capture_sequence,
                output_pcm16=queued_render[1] if queued_render else _SILENCE,
                submission=queued_render[0] if queued_render else None,
                context_valid=not self._generation_overflowed,
            )
            if capture_overflow:
                self._capture_overflows += 1
                self._drain_failure = "capture ring overflow"
            else:
                self._capture_ring.append(frame)
            return queued_render[1] if queued_render is not None else _SILENCE

    def _map_timing_locked(
        self,
        time_info: object,
        status: object,
        *,
        format_discontinuity: bool,
        capture_occupancy_frames: int,
        render_occupancy_frames: int,
        occupancy_bounded: bool,
        observed_ns: int,
    ) -> AecDelayEvidence:
        status_flags = _status_flags(status)
        timing_discontinuity = format_discontinuity or bool(status_flags)
        clock_drift = False
        try:
            input_adc_seconds = _time_value(time_info, "inputBufferAdcTime")
            current_seconds = _time_value(time_info, "currentTime")
            output_dac_seconds = _time_value(time_info, "outputBufferDacTime")
            device_current_ns = round(current_seconds * _NANOSECONDS_PER_SECOND)
            if self._device_to_monotonic_offset_ns is None:
                self._device_to_monotonic_offset_ns = observed_ns - device_current_ns
            offset_ns = self._device_to_monotonic_offset_ns
            capture_adc_ns = (
                round(input_adc_seconds * _NANOSECONDS_PER_SECOND) + offset_ns
            )
            render_dac_ns = (
                round(output_dac_seconds * _NANOSECONDS_PER_SECOND) + offset_ns
            )
            if min(capture_adc_ns, render_dac_ns) < 0:
                raise ValueError("mapped device timestamp is negative")
            if render_dac_ns < capture_adc_ns:
                timing_discontinuity = True
            callback_offset_ns = observed_ns - device_current_ns
            if self._drift_baseline_offset_ns is None:
                self._drift_offset_samples.append(callback_offset_ns)
                if len(self._drift_offset_samples) == _DRIFT_BASELINE_SAMPLES:
                    ordered_offsets = sorted(self._drift_offset_samples)
                    self._drift_baseline_offset_ns = ordered_offsets[
                        _DRIFT_BASELINE_SAMPLES // 2
                    ]
            elif (
                abs(callback_offset_ns - self._drift_baseline_offset_ns)
                > _TIMING_DRIFT_TOLERANCE_NS
            ):
                self._drift_violation_count += 1
                clock_drift = (
                    self._drift_violation_count >= _DRIFT_REQUIRED_OBSERVATIONS
                )
            else:
                self._drift_violation_count = 0
            if (
                self._last_input_adc_ns is not None
                and abs((capture_adc_ns - self._last_input_adc_ns) - FRAME_DURATION_NS)
                > _FRAME_CADENCE_JITTER_TOLERANCE_NS
            ):
                timing_discontinuity = True
            if (
                self._last_output_dac_ns is not None
                and abs((render_dac_ns - self._last_output_dac_ns) - FRAME_DURATION_NS)
                > _FRAME_CADENCE_JITTER_TOLERANCE_NS
            ):
                timing_discontinuity = True
            self._last_input_adc_ns = capture_adc_ns
            self._last_output_dac_ns = render_dac_ns
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            timing_discontinuity = True
            capture_adc_ns = max(
                0,
                observed_ns - FRAME_DURATION_NS,
                self._last_capture_ended_ns,
            )
            render_dac_ns = observed_ns
        delay_ms = round(
            max(0, render_dac_ns - capture_adc_ns) / _NANOSECONDS_PER_MILLISECOND
        )
        if delay_ms > 1_000:
            delay_ms = 1_000
            timing_discontinuity = True
        return AecDelayEvidence(
            observed_ns=observed_ns,
            capture_adc_ns=capture_adc_ns,
            render_dac_ns=render_dac_ns,
            delay_ms=delay_ms,
            capture_occupancy_frames=capture_occupancy_frames,
            render_occupancy_frames=render_occupancy_frames,
            occupancy_bounded=occupancy_bounded,
            timing_discontinuity=timing_discontinuity,
            clock_drift=clock_drift,
            status_flags=status_flags,
        )

    def _fence_generation_locked(self, failure: str) -> None:
        self._clock_generation += 1
        self._generation_overflowed = False
        self._native_startup_discarded = 0
        self._output_epoch = 0
        self._latest_render_boundary = None
        self._playback_intervals.clear()
        self._playback_history_lost_through_ns = -1
        self._capture_ring.clear()
        self._render_ring.clear()
        self._render_reference_ring.clear()
        self._capture_history.clear()
        self._history_truncated_through_ns = -1
        self._next_render_sequence = 0
        self._last_render_reference_sequence = None
        self._stream_latency_seconds = None
        self._last_capture_started_ns = -1
        self._last_capture_ended_ns = -1
        self._drift_offset_samples.clear()
        self._drift_baseline_offset_ns = None
        self._drift_violation_count = 0
        self._last_input_adc_ns = None
        self._last_output_dac_ns = None
        self._device_to_monotonic_offset_ns = None
        self._startup_callbacks_seen = 0
        self._generation_sequence_floor = self._next_capture_sequence
        self._last_dsp_sequence = self._generation_sequence_floor - 1
        self._last_vad_sequence = self._generation_sequence_floor - 1
        self._drain_failure = failure

    async def _teardown_stream(
        self,
        stream: _DuplexStream,
        *,
        generation: int | None = None,
    ) -> asyncio.CancelledError | None:
        """Attempt stop and close, recording failures and delayed cancellation."""

        cancellation: asyncio.CancelledError | None = None
        if isinstance(stream, NativeDuplexStream):
            stream.request_close()
            try:
                await stream.wait_closed()
            except Exception:
                with self._lock:
                    self._teardown_failures += 1
                raise
            finally:
                with self._lock:
                    self._capture_overflows += stream.bridge.snapshot()[
                        "capture_overflows"
                    ]
                    if generation is not None:
                        self._teardown_pending_generations.discard(generation)
            if stream.stop_failed:
                with self._lock:
                    self._teardown_failures += 1
            return None
        try:
            for operation in (stream.stop, stream.close):
                _, error, operation_cancellation = await _run_owned_thread(operation)
                cancellation = cancellation or operation_cancellation
                if error is not None:
                    with self._lock:
                        self._teardown_failures += 1
        finally:
            if generation is not None:
                with self._lock:
                    self._teardown_pending_generations.discard(generation)
        return cancellation
