"""Deterministic callback backend for duplex transport tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import time
from types import SimpleNamespace


class ManualClock:
    """Small injected monotonic clock whose time advances only in tests."""

    def __init__(self, now_ns: int = 10_000_000) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, nanoseconds: int) -> None:
        self.now_ns += nanoseconds


@dataclass(slots=True)
class FakeDuplexStream:
    """One fake raw stream retaining its callback for stale-route probes."""

    callback: Callable[[bytes, object, object, bool], bytes]
    latency: object = (0.01, 0.01)
    start_error: Exception | None = None
    synchronous_start_capture: bytes | None = None
    started: bool = False
    stopped: bool = False
    closed: bool = False
    frame_index: int = 0
    scripted_capture: Callable[[bytes, int], bytes] | None = None
    start_delay_seconds: float = 0.0
    stop_delay_seconds: float = 0.0
    close_delay_seconds: float = 0.0
    stop_error: Exception | None = None
    close_error: Exception | None = None
    start_count: int = 0
    stop_count: int = 0
    close_count: int = 0
    _render_history: deque[bytes] | None = None

    def start(self) -> None:
        self.start_count += 1
        time.sleep(self.start_delay_seconds)
        if self.start_error is not None:
            error = self.start_error
            self.start_error = None
            raise error
        if self.synchronous_start_capture is not None:
            self.emit_capture(self.synchronous_start_capture)
        self.started = True

    def __post_init__(self) -> None:
        self._render_history = deque()

    def stop(self) -> None:
        self.stop_count += 1
        time.sleep(self.stop_delay_seconds)
        if self.stop_error is not None:
            raise self.stop_error
        self.started = False
        self.stopped = True

    def close(self) -> None:
        self.close_count += 1
        time.sleep(self.close_delay_seconds)
        if self.close_error is not None:
            raise self.close_error
        self.started = False
        self.closed = True
        assert self._render_history is not None
        self._render_history.clear()

    def emit_capture(
        self,
        pcm16: bytes,
        *,
        input_adc_time: float | None = None,
        current_time: float | None = None,
        output_dac_time: float | None = None,
        status: object | None = None,
        discontinuity: bool = False,
    ) -> bytes:
        frame_index = self.frame_index
        self.frame_index += 1
        current_time = (
            (frame_index + 1) * 0.01 if current_time is None else current_time
        )
        input_adc_time = (
            current_time - 0.01 if input_adc_time is None else input_adc_time
        )
        output_dac_time = (
            current_time + 0.01 if output_dac_time is None else output_dac_time
        )
        time_info = SimpleNamespace(
            inputBufferAdcTime=input_adc_time,
            currentTime=current_time,
            outputBufferDacTime=output_dac_time,
        )
        return self.callback(
            pcm16,
            time_info,
            False if status is None else status,
            discontinuity,
        )

    def emit_scripted_capture(self, *, delay_frames: int = 1) -> bytes:
        """Feed capture derived from prior callback output and return new output."""

        if delay_frames <= 0:
            raise ValueError("scripted capture delay must be positive")
        assert self._render_history is not None
        delayed = (
            self._render_history[-delay_frames]
            if len(self._render_history) >= delay_frames
            else bytes(960)
        )
        capture = (
            delayed
            if self.scripted_capture is None
            else self.scripted_capture(delayed, self.frame_index)
        )
        rendered = self.emit_capture(capture)
        self._render_history.append(rendered)
        while len(self._render_history) > delay_frames:
            self._render_history.popleft()
        return rendered


class FakeDuplexBackend:
    """Records the single-session format requested by the transport."""

    def __init__(
        self,
        *,
        start_failures: int = 0,
        synchronous_start_capture: bytes | None = None,
        stream_latency: object = (0.01, 0.01),
        scripted_capture: Callable[[bytes, int], bytes] | None = None,
        open_delay_seconds: float = 0.0,
        start_delay_seconds: float = 0.0,
        stop_delay_seconds: float = 0.0,
        close_delay_seconds: float = 0.0,
        stop_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.open_calls: list[dict[str, object]] = []
        self.streams: list[FakeDuplexStream] = []
        self.start_failures = start_failures
        self.synchronous_start_capture = synchronous_start_capture
        self.stream_latency = stream_latency
        self.scripted_capture = scripted_capture
        self.open_delay_seconds = open_delay_seconds
        self.start_delay_seconds = start_delay_seconds
        self.stop_delay_seconds = stop_delay_seconds
        self.close_delay_seconds = close_delay_seconds
        self.stop_error = stop_error
        self.close_error = close_error

    @property
    def open_count(self) -> int:
        return len(self.open_calls)

    @property
    def close_count(self) -> int:
        return sum(stream.closed for stream in self.streams)

    @property
    def stream(self) -> FakeDuplexStream:
        return self.streams[-1]

    def open_stream(
        self,
        *,
        sample_rate: int,
        channels: int,
        frame_samples: int,
        callback: Callable[[bytes, object, object, bool], bytes],
        device_pair: tuple[int, int] | None = None,
    ) -> FakeDuplexStream:
        open_call: dict[str, object] = {
            "sample_rate": sample_rate,
            "channels": channels,
            "frame_samples": frame_samples,
        }
        if device_pair is not None:
            open_call["device_pair"] = device_pair
        self.open_calls.append(open_call)
        time.sleep(self.open_delay_seconds)
        start_error = None
        if self.start_failures:
            self.start_failures -= 1
            start_error = RuntimeError("fake stream start failure")
        stream = FakeDuplexStream(
            callback,
            latency=self.stream_latency,
            start_error=start_error,
            synchronous_start_capture=self.synchronous_start_capture,
            scripted_capture=self.scripted_capture,
            start_delay_seconds=self.start_delay_seconds,
            stop_delay_seconds=self.stop_delay_seconds,
            close_delay_seconds=self.close_delay_seconds,
            stop_error=self.stop_error,
            close_error=self.close_error,
        )
        self.streams.append(stream)
        return stream
