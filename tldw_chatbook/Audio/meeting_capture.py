"""Meeting capture: mic + system audio mixed into one dictation stream.

Textual-free. numpy is required (the recorder already requires it).

`EnergyRing` answers "who was talking in this window" from per-source RMS
history (spec §4). `MeetingCapture` (Task 5) duck-types the recorder
surface `LazyLiveDictationService` uses.
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np

#: -60 dBFS expressed as int16 RMS. Digital silence must not yield a zero
#: adaptive floor that any dither would exceed.
ABS_MIN_RMS: float = 32768 * 10 ** (-60 / 20)
SHARE_YOU: float = 0.7
SHARE_OTHERS: float = 0.3
FLOOR_MULTIPLIER: float = 3.0


def rms_int16(pcm: bytes) -> float:
    """RMS of little-endian int16 PCM; 0.0 for an empty buffer."""
    if len(pcm) < 2:
        return 0.0
    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


@dataclass
class SpeechRun:
    """A span of VAD-detected speech in audio seconds; ``end_s`` None while open."""

    start_s: float
    end_s: float | None = None


class EnergyRing:
    """Per-source RMS history in fixed buckets, bounded by a time horizon."""

    def __init__(
        self,
        bucket_s: float = 0.1,
        horizon_s: float = 600.0,
        floor_window_s: float = 30.0,
    ) -> None:
        self.bucket_s = bucket_s
        self.horizon_s = horizon_s
        self.floor_window_s = floor_window_s
        self._times: Deque[float] = deque()
        self._mic: Deque[float] = deque()
        self._sys: Deque[float] = deque()

    def add(self, position_s: float, mic_rms: float, sys_rms: float) -> None:
        bucket = math.floor(position_s / self.bucket_s) * self.bucket_s
        if self._times and bucket <= self._times[-1]:
            # Same bucket: keep the louder reading for each source.
            self._mic[-1] = max(self._mic[-1], mic_rms)
            self._sys[-1] = max(self._sys[-1], sys_rms)
        else:
            self._times.append(bucket)
            self._mic.append(mic_rms)
            self._sys.append(sys_rms)
        while self._times and self._times[0] < bucket - self.horizon_s:
            self._times.popleft()
            self._mic.popleft()
            self._sys.popleft()

    def _bucket(self, t: float) -> float:
        return math.floor(t / self.bucket_s) * self.bucket_s

    def _slice(self, start_s: float, end_s: float, *, inclusive_end: bool = True) -> Tuple[list, list]:
        """Buckets with start <= time <= end (or < end when inclusive_end is False)."""
        times = list(self._times)
        lo = bisect.bisect_left(times, self._bucket(start_s))
        hi = (
            bisect.bisect_right(times, end_s)
            if inclusive_end
            else bisect.bisect_left(times, self._bucket(end_s))
        )
        return list(self._mic)[lo:hi], list(self._sys)[lo:hi]

    def floor(self, source: str, before_s: float) -> float:
        """Adaptive noise floor from the 30 s strictly before ``before_s``: 3x its 10th percentile, never below ABS_MIN_RMS."""
        mic, sys_ = self._slice(before_s - self.floor_window_s, before_s, inclusive_end=False)
        values = mic if source == "mic" else sys_
        if not values:
            return ABS_MIN_RMS
        p10 = float(np.percentile(np.asarray(values, dtype=np.float64), 10))
        return max(FLOOR_MULTIPLIER * p10, ABS_MIN_RMS)

    def dominant_source(self, start_s: float, end_s: float) -> str:
        """Label a window ``you`` / ``others`` / ``both`` (spec §4)."""
        mic, sys_ = self._slice(start_s, end_s)
        mic_floor = self.floor("mic", start_s)
        sys_floor = self.floor("sys", start_s)
        mic_active = sum(v for v in mic if v > mic_floor)
        sys_active = sum(v for v in sys_ if v > sys_floor)
        total = mic_active + sys_active
        if total <= 0.0:
            # ponytail: energy-share heuristic; the Diarizer seam (meeting_session.py)
            # replaces this whole method in phase 2.
            return "you" if sum(mic) > sum(sys_) else "others"
        share_you = mic_active / total
        if share_you >= SHARE_YOU:
            return "you"
        if share_you <= SHARE_OTHERS:
            return "others"
        return "both"
