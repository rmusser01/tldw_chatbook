"""Pure, bounded audio energy and disposable speech presentation (ADR-242)."""

from __future__ import annotations

import math
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

MouthPose = Literal["rest", "small", "open"]
TransportState = Literal["preparing", "playing", "paused", "stopped", "failed"]
_WINDOW_HZ = 50
_FULL_SCALE = 32768
_GATE_CLOSE_DBFS = -45.0
_GATE_OPEN_DBFS = -42.0
_GATE_CLOSE_WINDOWS = 2
_ATTACK_MS = 40.0
_RELEASE_MS = 100.0
_OPEN_DBFS = -24.0
_SMALL_DBFS = -27.0
_FRESH_SECONDS = 0.1
_TRANSPORT_STATES = ("preparing", "playing", "paused", "stopped", "failed")


@dataclass(frozen=True, slots=True)
class SpeechPresentationOwner:
    """One host-issued request identity; opaque authority is never serialized."""

    token: str
    scope_token: object = field(repr=False)
    session_id: str | None
    session_epoch: int | None
    character_id: int | None
    message_id: str | None
    utterance_token: int | None


@dataclass(frozen=True, slots=True)
class SpeechPresentationSnapshot:
    """Latest output-timed level and confirmed transport, without PCM/history."""

    owner: SpeechPresentationOwner
    generation: int
    state: TransportState
    observed_at: float
    level_dbfs: float | None
    mouth: MouthPose
    timing_valid: bool


@dataclass(frozen=True, slots=True)
class EnvelopeWindow:
    """One source-frame interval with RMS energy across all channel samples."""

    start_frame: int
    frame_count: int
    dbfs: float


class PcmEnvelope:
    """Accumulate little-endian PCM16 in nominal 20 ms source-frame windows.

    Call push only for accepted audio, not rejected/backpressure attempts. Carry
    at most one incomplete PCM frame and one scalar-energy window between calls.
    """

    def __init__(self, *, sample_rate: int, channels: int) -> None:
        if type(sample_rate) is not int or sample_rate < _WINDOW_HZ:
            raise ValueError("speech_sample_rate_invalid")
        if type(channels) is not int or channels < 1:
            raise ValueError("speech_channels_invalid")
        self._rate = sample_rate
        self._channels = channels
        self._frame_bytes = channels * 2
        self._pending = b""
        self._window_index = 0
        self._completed_frames = 0
        self._window_frames = 0
        self._energy = 0
        self._finished = False

    def _emit_window(self) -> EnvelopeWindow:
        mean_square = self._energy / (self._window_frames * self._channels)
        dbfs = (
            10.0 * math.log10(mean_square / (_FULL_SCALE * _FULL_SCALE))
            if self._energy
            else float("-inf")
        )
        result = EnvelopeWindow(self._completed_frames, self._window_frames, dbfs)
        self._completed_frames += self._window_frames
        self._window_frames = 0
        self._energy = 0
        self._window_index += 1
        return result

    def push(self, pcm: bytes) -> tuple[EnvelopeWindow, ...]:
        """Consume accepted PCM bytes, preserving partial frames across chunks.

        Raises:
            ValueError: For non-bytes input or input after finish.
        """
        if self._finished or type(pcm) is not bytes:
            raise ValueError("speech_pcm_input_invalid")
        data = self._pending + pcm
        aligned = len(data) - len(data) % self._frame_bytes
        self._pending = data[aligned:]
        windows = []
        offset = 0
        while offset < aligned:
            boundary = (self._window_index + 1) * self._rate // _WINDOW_HZ
            needed = boundary - self._completed_frames - self._window_frames
            frame_count = min(needed, (aligned - offset) // self._frame_bytes)
            end = offset + frame_count * self._frame_bytes
            self._energy += sum(
                value * value
                for (value,) in struct.iter_unpack("<h", memoryview(data)[offset:end])
            )
            self._window_frames += frame_count
            offset = end
            if self._completed_frames + self._window_frames == boundary:
                windows.append(self._emit_window())
        return tuple(windows)

    def finish(self) -> tuple[EnvelopeWindow, ...]:
        """Flush a short complete final window; reject a partial channel frame."""
        if self._finished:
            return ()
        self._finished = True
        if self._pending:
            self._pending = b""
            self._energy = 0
            self._window_frames = 0
            raise ValueError("speech_pcm_final_frame_incomplete")
        return (self._emit_window(),) if self._window_frames else ()


class SpeechEnvelopeFilter:
    """Two-window raw silence gate followed by finite dBFS mouth hysteresis."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Close the gate and discard smoothing, including a prior silent window."""
        self._smoothed: float | None = None
        self._quiet_windows = 0
        self._pose: MouthPose = "rest"

    def update(self, dbfs: float, *, duration_ms: float = 20.0) -> MouthPose:
        """Select a pose using actual window duration; zero/invalid energy closes.

        Raises:
            ValueError: For an invalid window duration or nonnumeric level.
        """
        if (
            type(duration_ms) not in (float, int)
            or not math.isfinite(duration_ms)
            or duration_ms <= 0
        ):
            raise ValueError("speech_window_duration_invalid")
        if type(dbfs) not in (float, int):
            raise ValueError("speech_level_invalid")
        if not math.isfinite(dbfs):
            self.reset()
            return "rest"
        level = min(0.0, max(-120.0, dbfs))
        if self._smoothed is None:
            if level < _GATE_OPEN_DBFS:
                return "rest"
            self._smoothed = level
            self._pose = "open" if level >= _OPEN_DBFS else "small"
            return self._pose
        # Counting raw windows avoids a third-window delay at fractional rates.
        self._quiet_windows = self._quiet_windows + 1 if level < _GATE_CLOSE_DBFS else 0
        if self._quiet_windows >= _GATE_CLOSE_WINDOWS:
            self.reset()
            return "rest"
        tau = _ATTACK_MS if level > self._smoothed else _RELEASE_MS
        self._smoothed += (level - self._smoothed) * (
            1.0 - math.exp(-duration_ms / tau)
        )
        if self._pose == "open":
            if self._smoothed < _SMALL_DBFS:
                self._pose = "small"
        elif self._smoothed >= _OPEN_DBFS:
            self._pose = "open"
        return self._pose


class SpeechPresentation:
    """Thread-safe single-value observation for one exact host-issued owner.

    Readers cannot block a producer with UI work; this channel owns no listener,
    waveform queue, thread or timer. An owner's terminal state cannot be reopened.
    """

    __slots__ = ("_clock", "_latest", "_lock", "_owner")

    def __init__(
        self,
        owner: SpeechPresentationOwner,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._owner = owner
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: SpeechPresentationSnapshot | None = None

    @property
    def owner(self) -> SpeechPresentationOwner:
        """The original host-issued identity, never retargeted on context changes."""
        return self._owner

    def _now(self) -> float | None:
        try:
            now = self._clock()
            return now if math.isfinite(now) else None
        except Exception:  # noqa: BLE001 - observer failures must not fail audio.
            return None

    @staticmethod
    def _closed(snapshot: SpeechPresentationSnapshot) -> SpeechPresentationSnapshot:
        return replace(snapshot, mouth="rest", level_dbfs=None, timing_valid=False)

    def publish(self, snapshot: SpeechPresentationSnapshot) -> None:
        """Replace the latest observation, refusing stale or unrelated publishers."""
        if (
            not isinstance(snapshot, SpeechPresentationSnapshot)
            or snapshot.owner is not self.owner
            or type(snapshot.generation) is not int
            or snapshot.generation < 0
            or snapshot.state not in _TRANSPORT_STATES
            or snapshot.mouth not in ("rest", "small", "open")
        ):
            return
        now = self._now()
        try:
            valid_time = (
                math.isfinite(snapshot.observed_at)
                and now is not None
                and snapshot.observed_at <= now
            )
            silent = snapshot.level_dbfs is None or snapshot.level_dbfs == float("-inf")
            valid_level = silent or math.isfinite(snapshot.level_dbfs)
        except (TypeError, ValueError, OverflowError):
            return
        if not valid_time:
            snapshot = replace(
                self._closed(snapshot), observed_at=now if now is not None else 0.0
            )
        if not valid_level or not snapshot.timing_valid or snapshot.state != "playing":
            snapshot = self._closed(snapshot)
        elif silent or snapshot.mouth == "rest":
            snapshot = replace(snapshot, mouth="rest", level_dbfs=None)
        with self._lock:
            old = self._latest
            if old is not None and (
                old.state in ("stopped", "failed")
                or snapshot.generation < old.generation
                or (
                    snapshot.generation == old.generation
                    and snapshot.observed_at < old.observed_at
                )
                or (old.state == "paused" and snapshot.state == "playing")
                or (
                    old.state in ("playing", "paused") and snapshot.state == "preparing"
                )
            ):
                return
            self._latest = snapshot

    def read(self) -> SpeechPresentationSnapshot | None:
        """Return current output or a closed, timing-invalid stale fallback."""
        with self._lock:
            snapshot = self._latest
        if snapshot is None:
            return None
        now = self._now()
        if now is None or not 0 <= now - snapshot.observed_at <= _FRESH_SECONDS:
            return self._closed(snapshot)
        return snapshot

    def set_transport(self, state: TransportState) -> None:
        """Project confirmed transport without retaining an old mouth on resume."""
        if state not in _TRANSPORT_STATES:
            return
        now = self._now()
        with self._lock:
            old = self._latest
            if old is not None and (
                old.state in ("stopped", "failed")
                or old.state == state
                or (old.state in ("playing", "paused") and state == "preparing")
            ):
                return
            self._latest = SpeechPresentationSnapshot(
                self.owner,
                0 if old is None else old.generation,
                state,
                now if now is not None else 0.0,
                None,
                "rest",
                False,
            )
