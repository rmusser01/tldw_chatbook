"""Pure visible-time portrait motion from the packaged design tokens (ADR-242)."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

from tldw_chatbook.Audio.speech_presentation import (
    MouthPose,
    SpeechPresentationSnapshot,
)
from tldw_chatbook.Character_Chat.portrait_performance import FrameRole


@dataclass(frozen=True, slots=True)
class PerformancePose:
    """One role and a vertical offset inside its fixed fitted canvas."""

    role: FrameRole
    offset_px: float


@dataclass(frozen=True, slots=True)
class PerformanceMotionValues:
    """Validated motion values, shared by live and disposable preview playback."""

    blink_min_s: float
    blink_max_s: float
    blink_hold_s: float
    idle_cycle_s: float
    idle_offset_px: float

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(v)
                for v in (
                    self.blink_min_s,
                    self.blink_max_s,
                    self.blink_hold_s,
                    self.idle_cycle_s,
                    self.idle_offset_px,
                )
            )
            or not 0 < self.blink_hold_s < self.blink_min_s <= self.blink_max_s
            or self.idle_cycle_s <= 0
            or not 0 <= self.idle_offset_px <= 1
        ):
            raise ValueError("portrait_motion_values_invalid")


@lru_cache(maxsize=1)
def load_performance_motion_values() -> PerformanceMotionValues:
    """Read only packaged scalar motion tokens, once during worker preparation."""
    path = files("tldw_chatbook").joinpath("css/core/_variables.tcss")
    with path.open("r", encoding="utf-8") as source:
        text = source.read(256 * 1024 + 1)
    if len(text) > 256 * 1024:
        raise ValueError("portrait_motion_tokens_invalid")
    names = (
        ("portrait-blink-min", "s"),
        ("portrait-blink-max", "s"),
        ("duration-fast", "s"),
        ("portrait-idle-cycle", "s"),
        ("portrait-idle-offset-px", ""),
    )
    values = []
    for name, unit in names:
        match = re.search(
            rf"^\$ds-{name}:\s*([0-9]+(?:\.[0-9]+)?){unit};",
            text,
            re.MULTILINE,
        )
        if match is None:
            raise ValueError("portrait_motion_tokens_invalid")
        values.append(float(match[1]))
    return PerformanceMotionValues(*values)


class PortraitPerformanceClock:
    """Blink independently of mouth changes, advancing only in visible time."""

    def __init__(
        self,
        *,
        motion: PerformanceMotionValues,
        interval: Callable[[], float],
    ) -> None:
        self.motion = motion
        self._interval = interval
        self._next_blink: float | None = None
        self._blink_until: float | None = None

    def _next_interval(self) -> float:
        value = self._interval()
        if (
            not math.isfinite(value)
            or not self.motion.blink_min_s <= value <= self.motion.blink_max_s
        ):
            raise ValueError("portrait_motion_interval_invalid")
        return value

    def sample(self, mouth: MouthPose, *, now: float) -> PerformancePose:
        """Select a pose without decoding, I/O or restarting the blink for speech."""
        if not math.isfinite(now) or now < 0:
            raise ValueError("portrait_motion_clock_invalid")
        if self._next_blink is None:
            self._next_blink = now + self._next_interval()
        if self._blink_until is not None and now >= self._blink_until:
            self._blink_until = None
            self._next_blink = now + self._next_interval()
        if self._blink_until is None and now >= self._next_blink:
            self._blink_until = now + self.motion.blink_hold_s
        roles: dict[MouthPose, FrameRole] = (
            {"rest": "blink_rest", "small": "blink_small", "open": "blink_open"}
            if self._blink_until is not None
            else {"rest": "rest", "small": "small", "open": "open"}
        )
        return PerformancePose(
            roles[mouth],
            math.sin(now * math.tau / self.motion.idle_cycle_s)
            * self.motion.idle_offset_px,
        )


def portrait_radio_status(snapshot: SpeechPresentationSnapshot | None) -> str:
    """Describe owned transport, independently of text streaming or timing support."""
    return (
        {
            "preparing": "Preparing voice",
            "playing": "Transmitting",
            "paused": "Paused",
            "stopped": "Ready",
            "failed": "Voice unavailable",
        }.get(snapshot.state, "Voice unavailable")
        if snapshot is not None
        else "Ready"
    )


def portrait_radio_level(
    snapshot: SpeechPresentationSnapshot | None, *, now: float
) -> float | None:
    """Return a measured display fraction, with silence empty and stale timing hidden."""
    from tldw_chatbook.Audio.speech_presentation import _FRESH_SECONDS

    if (
        snapshot is None
        or snapshot.state != "playing"
        or not snapshot.timing_valid
        or not 0 <= now - snapshot.observed_at <= _FRESH_SECONDS
    ):
        return None
    if snapshot.level_dbfs is None:
        return 0.0
    if not math.isfinite(snapshot.level_dbfs):
        return None
    return max(0.0, min(1.0, (snapshot.level_dbfs + 60) / 60))
