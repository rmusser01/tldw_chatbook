"""Bounded transient ffplay statistics from the qualified Windows build."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .controlled_wav_profile import PADDING_FRAMES, RATE

_STATS = re.compile(
    rb"^\s*(\d+(?:\.\d+)?)\s+(?:M-A|A-V):\s*[+-]?(?:\d+(?:\.\d+)?|nan)\s+fd=\s*\d+\s+aq="
)
FRESH_S = 0.5
STARTUP_S = 30.0
MAX_SKEW_S = 0.250
TERMINAL_PAD_S = 0.100
MIN_SPAN_S = 0.030


class StatsReader:
    """Parse a single generation's CR/LF records with bounded partial storage."""

    def __init__(self):
        self._pending = bytearray()
        self._discarding = False

    def feed(self, data: bytes) -> list[float]:
        values = []
        for byte in data:
            if byte in (10, 13):
                match = None if self._discarding else _STATS.match(self._pending)
                if match:
                    value = float(match[1])
                    if math.isfinite(value):
                        values.append(value)
                self._pending.clear()
                self._discarding = False
            elif not self._discarding:
                if len(self._pending) >= 1024:
                    self._pending.clear()
                    self._discarding = True
                else:
                    self._pending.append(byte)
        return values


@dataclass(frozen=True)
class ProgressObservation:
    frame: int
    observed_at_s: float
    generation: int


class ProgressClock:
    """Observe absolute media time. Wall time can veto, never supply a bookmark."""

    def __init__(
        self,
        start_frame: int,
        content_frames: int,
        generation: int,
        launched_at_s: float,
    ):
        self.start = start_frame
        self.content = content_frames
        self.generation = generation
        self.launched_at = launched_at_s
        self.first: tuple[float, float] | None = None
        self.last_physical_frame: int | None = None
        self.last: ProgressObservation | None = None
        self.advanced_span_s = 0.0
        self.clock_skew_s: float | None = None

    def observe(
        self,
        seconds: float,
        stamp: float,
        generation: int,
        *,
        now_s: float | None = None,
    ) -> None:
        if (
            generation != self.generation
            or (now_s is not None and not 0 <= now_s - stamp <= FRESH_S)
            or not math.isfinite(stamp)
            or stamp < self.launched_at
            or not math.isfinite(seconds)
            or not 0 <= seconds <= (self.content + PADDING_FRAMES) / RATE
        ):
            return
        frame = round(seconds * RATE)
        if self.first is None:
            if abs(seconds - self.start / RATE) > MAX_SKEW_S or self.expired(stamp):
                return
            self.first = (seconds, stamp)
        else:
            if (
                stamp < self.last.observed_at_s
                or frame <= self.last_physical_frame
                or abs(
                    (frame - self.last_physical_frame) / RATE
                    - (stamp - self.last.observed_at_s)
                )
                > MAX_SKEW_S
                or self.expired(stamp)
            ):
                return
        first_source, first_stamp = self.first
        span = stamp - first_stamp
        skew = abs((seconds - first_source) - span)
        if skew > MAX_SKEW_S:
            return
        self.last_physical_frame = frame
        self.last = ProgressObservation(
            max(self.start, min(self.content, frame)), stamp, generation
        )
        self.advanced_span_s = span
        self.clock_skew_s = skew

    def position(self, now: float) -> ProgressObservation | None:
        if self.last and 0 <= now - self.last.observed_at_s <= FRESH_S:
            return self.last
        return None

    def expired(self, now: float) -> bool:
        if self.last is None:
            return now - self.launched_at > STARTUP_S
        return now - self.last.observed_at_s > FRESH_S

    def natural_candidate(self, exited_at: float) -> bool:
        return (
            self.position(exited_at) is not None
            and self.advanced_span_s >= MIN_SPAN_S
            and self.last_physical_frame >= self.content + round(TERMINAL_PAD_S * RATE)
            and exited_at - self.launched_at + MAX_SKEW_S
            >= (self.content + PADDING_FRAMES - self.start) / RATE
        )
