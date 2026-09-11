"""Bounded, content-free diagnostics for speculative Console voice.

The public methods accept numbers and closed enums only.  Transcript text,
response text, PCM, request bodies, and capture payloads therefore have no
ingress into this metrics surface.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, fields
from enum import Enum


_MAX_RECENT_SAMPLES = 256
_MAX_COUNTER = 2**63 - 1


class VoiceLatencyKind(str, Enum):
    """Latency intervals measured by the speculative pipeline."""

    EOS_TO_DISPATCH = "eos_to_dispatch"
    BARGE_TO_AUDIBLE_STOP = "barge_to_audible_stop"
    REPLACEMENT_DISPATCH = "replacement_dispatch"
    FIRST_AUDIO = "first_audio"


class VoiceBackendMode(str, Enum):
    """Incremental transcription backend selected for a turn."""

    NATIVE_LIVE = "native_live"
    ROLLING_WINDOW = "rolling_window"


class VoiceAecState(str, Enum):
    """User-relevant echo-cancellation health state."""

    WARMING = "warming"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    HALF_DUPLEX = "half_duplex"


class VoiceProviderResultClass(str, Enum):
    """Content-free terminal class for provider work."""

    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    CANCELLATION_UNCONFIRMED = "cancellation_unconfirmed"
    FAILED = "failed"


class VoiceUsageOutcome(str, Enum):
    """Whether billable work belonged to the promoted or a discarded attempt."""

    WINNING = "winning"
    DISCARDED = "discarded"


def _bounded_count(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value > _MAX_COUNTER:
        raise ValueError(f"{name} is outside the safe range")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class VoiceUsageCounters:
    """Provider-neutral counts retained without request or response content."""

    calls: int = 0
    stt_audio_ms: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    tts_characters: int = 0
    tts_audio_ms: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            _bounded_count(getattr(self, item.name), name=item.name)

    def plus(self, other: "VoiceUsageCounters") -> "VoiceUsageCounters":
        """Return checked component-wise totals."""

        if type(other) is not VoiceUsageCounters:
            raise TypeError("usage must be VoiceUsageCounters")
        values: dict[str, int] = {}
        for item in fields(self):
            value = getattr(self, item.name) + getattr(other, item.name)
            values[item.name] = _bounded_count(value, name=item.name)
        return VoiceUsageCounters(**values)


@dataclass(frozen=True, slots=True)
class VoiceMetricsSnapshot:
    """Immutable bounded projection suitable for diagnostics or tests."""

    latencies_ms: dict[VoiceLatencyKind, tuple[float, ...]]
    backend_modes: tuple[VoiceBackendMode, ...]
    aec_states: tuple[VoiceAecState, ...]
    erle_db: tuple[float, ...]
    underruns: int
    restarts: int
    conservative_entries: int
    duplicated_stt_audio_ms: int
    provider_results: tuple[VoiceProviderResultClass, ...]
    winning_usage: VoiceUsageCounters
    discarded_usage: VoiceUsageCounters


class VoiceMetrics:
    """Small in-memory collector with a closed, content-free input surface."""

    def __init__(self, *, max_recent_samples: int = _MAX_RECENT_SAMPLES) -> None:
        if type(max_recent_samples) is not int or not 1 <= max_recent_samples <= 4096:
            raise ValueError("max_recent_samples must be between 1 and 4096")
        self._latencies = {
            kind: deque(maxlen=max_recent_samples) for kind in VoiceLatencyKind
        }
        self._backend_modes: deque[VoiceBackendMode] = deque(maxlen=max_recent_samples)
        self._aec_states: deque[VoiceAecState] = deque(maxlen=max_recent_samples)
        self._erle_db: deque[float] = deque(maxlen=max_recent_samples)
        self._provider_results: deque[VoiceProviderResultClass] = deque(
            maxlen=max_recent_samples
        )
        self._underruns = 0
        self._restarts = 0
        self._conservative_entries = 0
        self._duplicated_stt_audio_ms = 0
        self._winning_usage = VoiceUsageCounters()
        self._discarded_usage = VoiceUsageCounters()

    def observe_latency(self, kind: VoiceLatencyKind, duration_ms: float) -> None:
        if type(kind) is not VoiceLatencyKind:
            raise TypeError("kind must be VoiceLatencyKind")
        self._latencies[kind].append(
            _finite_nonnegative(duration_ms, name="duration_ms")
        )

    def observe_backend_mode(self, mode: VoiceBackendMode) -> None:
        if type(mode) is not VoiceBackendMode:
            raise TypeError("mode must be VoiceBackendMode")
        self._backend_modes.append(mode)

    def observe_aec(
        self,
        state: VoiceAecState,
        *,
        erle_db: float | None = None,
    ) -> None:
        if type(state) is not VoiceAecState:
            raise TypeError("state must be VoiceAecState")
        normalized_erle: float | None = None
        if erle_db is not None:
            if type(erle_db) not in (int, float) or not math.isfinite(float(erle_db)):
                raise ValueError("erle_db must be finite")
            normalized_erle = float(erle_db)
        self._aec_states.append(state)
        if normalized_erle is not None:
            self._erle_db.append(normalized_erle)

    def increment_underruns(self, count: int = 1) -> None:
        self._underruns = self._add_count(self._underruns, count, "underruns")

    def increment_restarts(self, count: int = 1) -> None:
        self._restarts = self._add_count(self._restarts, count, "restarts")

    def increment_conservative_entries(self, count: int = 1) -> None:
        self._conservative_entries = self._add_count(
            self._conservative_entries, count, "conservative_entries"
        )

    def add_duplicated_stt_audio_ms(self, duration_ms: int) -> None:
        self._duplicated_stt_audio_ms = self._add_count(
            self._duplicated_stt_audio_ms,
            duration_ms,
            "duplicated_stt_audio_ms",
        )

    @staticmethod
    def _add_count(current: int, increment: int, name: str) -> int:
        increment = _bounded_count(increment, name=name)
        return _bounded_count(current + increment, name=name)

    def record_provider_result(self, result: VoiceProviderResultClass) -> None:
        if type(result) is not VoiceProviderResultClass:
            raise TypeError("result must be VoiceProviderResultClass")
        self._provider_results.append(result)

    def record_usage(
        self,
        outcome: VoiceUsageOutcome,
        usage: VoiceUsageCounters,
    ) -> None:
        if type(outcome) is not VoiceUsageOutcome:
            raise TypeError("outcome must be VoiceUsageOutcome")
        if type(usage) is not VoiceUsageCounters:
            raise TypeError("usage must be VoiceUsageCounters")
        if outcome is VoiceUsageOutcome.WINNING:
            self._winning_usage = self._winning_usage.plus(usage)
        else:
            self._discarded_usage = self._discarded_usage.plus(usage)

    def snapshot(self) -> VoiceMetricsSnapshot:
        """Return a detached bounded copy; raw content cannot be represented."""

        return VoiceMetricsSnapshot(
            latencies_ms={
                kind: tuple(samples) for kind, samples in self._latencies.items()
            },
            backend_modes=tuple(self._backend_modes),
            aec_states=tuple(self._aec_states),
            erle_db=tuple(self._erle_db),
            underruns=self._underruns,
            restarts=self._restarts,
            conservative_entries=self._conservative_entries,
            duplicated_stt_audio_ms=self._duplicated_stt_audio_ms,
            provider_results=tuple(self._provider_results),
            winning_usage=self._winning_usage,
            discarded_usage=self._discarded_usage,
        )
