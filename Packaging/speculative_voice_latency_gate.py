"""Deterministic clock-boundary latency gate for speculative voice."""

from __future__ import annotations

import asyncio
import heapq
import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable

from tldw_chatbook.Audio.duplex_contracts import DrainReceipt, DuplexMode
from tldw_chatbook.Audio.rolling_transcript import TranscriptRevision
from tldw_chatbook.Chat.console_speculative_voice import (
    AdmittedSpeechFrame,
    AttemptOutputDelta,
    AttemptPlaybackStarted,
    SpeculativeTurnCoordinator,
)
from tldw_chatbook.Chat.console_voice_attempts import AttemptCleanupOutcome


LATENCY_TRIALS = 40


@dataclass(slots=True)
class _Handle:
    deadline_ns: int
    order: int
    callback: Callable[[], None]
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


class _Scheduler:
    def __init__(self) -> None:
        self.now_ns = 10_000_000
        self._order = 0
        self._pending: list[tuple[int, int, _Handle]] = []

    def call_at_ns(self, deadline_ns: int, callback: Callable[[], None]) -> _Handle:
        handle = _Handle(deadline_ns, self._order, callback)
        self._order += 1
        heapq.heappush(self._pending, (deadline_ns, handle.order, handle))
        return handle

    def advance_ms(self, duration_ms: int) -> None:
        self.now_ns += duration_ms * 1_000_000
        while self._pending and self._pending[0][0] <= self.now_ns:
            _deadline, _order, handle = heapq.heappop(self._pending)
            if not handle.cancelled:
                handle.callback()


class _Effects:
    def __init__(self, scheduler: _Scheduler) -> None:
        self.scheduler = scheduler
        self.dispatches: list[tuple[int, int, str]] = []
        self.aborts: list[tuple[int, int]] = []

    def dispatch_attempt(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
    ) -> None:
        del turn_id
        self.dispatches.append((self.scheduler.now_ns, attempt_epoch, transcript))

    def fence_attempt(self, _attempt_epoch: int) -> None:
        return None

    def cancel_attempt(
        self,
        _attempt_epoch: int,
    ) -> asyncio.Future[AttemptCleanupOutcome]:
        future = asyncio.get_running_loop().create_future()
        future.set_result(AttemptCleanupOutcome.CLEAN)
        return future

    def abort_output(self, attempt_epoch: int) -> None:
        self.aborts.append((self.scheduler.now_ns, attempt_epoch))

    def clear_preview(self, _attempt_epoch: int) -> None:
        return None

    def publish_preview(self, _attempt_epoch: int, _delta: str) -> None:
        return None

    def preserve_draft(self, **_kwargs: Any) -> None:
        return None

    def promote(self, **_kwargs: Any) -> None:
        return None

    def classify_spoken_command(self, _transcript: str) -> bool:
        return False

    def handle_spoken_command(self, **_kwargs: Any) -> None:
        return None

    def rebuild_audio(self, _old: int, _epoch: int) -> None:
        return None

    def voice_dispatch_quarantined(self) -> bool:
        return False

    async def drain_capture_through(self, boundary_ns: int) -> DrainReceipt:
        return DrainReceipt(boundary_ns + 1, 0, 0, 0, 0)

    async def seal_transcript_through(
        self,
        _sequence: int,
    ) -> TranscriptRevision:
        raise AssertionError("promotion is outside these latency boundaries")

    def submit_accepted_voice_turn(self, *_args: Any) -> None:
        return None


def _speech(sequence: int, ended_ns: int) -> AdmittedSpeechFrame:
    return AdmittedSpeechFrame(
        sequence,
        ended_ns - 10_000_000,
        ended_ns,
        0,
    )


def _revision(
    turn_id: str,
    revision_id: int,
    text: str,
    covered_through_ns: int,
) -> TranscriptRevision:
    return TranscriptRevision(
        turn_id=turn_id,
        revision_id=revision_id,
        stable_text="",
        revisable_text=text,
        covered_through_ns=covered_through_ns,
        mode="live",
    )


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def _summary(samples: list[float]) -> dict[str, object]:
    rounded_samples = [round(sample, 6) for sample in samples]
    return {
        "samples_ms": rounded_samples,
        "median_ms": round(statistics.median(rounded_samples), 6),
        "p95_ms": round(_percentile(rounded_samples, 0.95), 6),
    }


async def _one_trial() -> tuple[float, float, float, float]:
    scheduler = _Scheduler()
    effects = _Effects(scheduler)
    coordinator = SpeculativeTurnCoordinator(
        effects=effects,
        scheduler=scheduler,
        response_eagerness_ms=700,
        initial_duplex_mode=DuplexMode.FULL_DUPLEX,
    )
    await coordinator.start()
    try:
        first_eos_ns = scheduler.now_ns
        await coordinator.submit(_speech(0, first_eos_ns))
        turn_id = coordinator.snapshot.turn_id
        if turn_id is None:
            raise RuntimeError("latency trial did not create a logical turn")
        await coordinator.submit(_revision(turn_id, 1, "first", first_eos_ns))
        scheduler.advance_ms(700)
        await coordinator.flush()
        first_dispatch_ns, first_epoch, _text = effects.dispatches[-1]

        scheduler.advance_ms(250)
        await coordinator.submit(AttemptOutputDelta(first_epoch, "Ready!"))
        scheduler.advance_ms(100)
        await coordinator.submit(AttemptPlaybackStarted(first_epoch))
        first_audio_ns = scheduler.now_ns

        added_eos_ns = scheduler.now_ns
        await coordinator.submit(_speech(1, added_eos_ns))
        audible_stop_ns = effects.aborts[-1][0]
        await coordinator.submit(
            _revision(turn_id, 2, "first additional", added_eos_ns)
        )
        scheduler.advance_ms(700)
        await coordinator.flush()
        replacement_dispatch_ns, replacement_epoch, replacement = effects.dispatches[-1]
        if replacement_epoch == first_epoch or replacement != "first additional":
            raise RuntimeError("latency trial did not replace the stale attempt")

        return (
            (first_dispatch_ns - first_eos_ns) / 1_000_000,
            (audible_stop_ns - added_eos_ns) / 1_000_000,
            (replacement_dispatch_ns - added_eos_ns) / 1_000_000,
            (first_audio_ns - first_eos_ns) / 1_000_000,
        )
    finally:
        await coordinator.close()


async def measure_speculative_voice_latency(
    *,
    trials: int = LATENCY_TRIALS,
) -> dict[str, object]:
    """Measure deterministic coordinator latency at declared clock boundaries."""

    if type(trials) is not int or trials < 1:
        raise ValueError("latency trials must be a positive integer")
    samples = [await _one_trial() for _ in range(trials)]
    dispatch = _summary([sample[0] for sample in samples])
    audible_stop = _summary([sample[1] for sample in samples])
    replacement = _summary([sample[2] for sample in samples])
    first_audio = _summary([sample[3] for sample in samples])
    thresholds = {
        "eos_to_dispatch_p95_ms_max": 850.0,
        "barge_to_audible_stop_p95_ms_max": 150.0,
        "added_eos_to_replacement_dispatch_p95_ms_max": 850.0,
        "eos_to_first_assistant_audio_median_ms_max": 1_500.0,
        "eos_to_first_assistant_audio_p95_ms_max": 2_500.0,
    }
    passed = bool(
        dispatch["p95_ms"] <= thresholds["eos_to_dispatch_p95_ms_max"]
        and audible_stop["p95_ms"] <= thresholds["barge_to_audible_stop_p95_ms_max"]
        and replacement["p95_ms"]
        <= thresholds["added_eos_to_replacement_dispatch_p95_ms_max"]
        and first_audio["median_ms"]
        <= thresholds["eos_to_first_assistant_audio_median_ms_max"]
        and first_audio["p95_ms"]
        <= thresholds["eos_to_first_assistant_audio_p95_ms_max"]
    )
    return {
        "trial_count": trials,
        "clock_boundaries": {
            "eos": "post_aec_admitted_speech_end",
            "dispatch": "attempt_dispatch_handoff",
            "audible_stop": "transport_abort_fence",
            "first_assistant_audio": "transport_playback_started",
        },
        "thresholds": thresholds,
        "eos_to_dispatch": dispatch,
        "barge_to_audible_stop": audible_stop,
        "added_eos_to_replacement_dispatch": replacement,
        "eos_to_first_assistant_audio": first_audio,
        "passed": passed,
    }


def run_speculative_voice_latency_gate(
    *,
    trials: int = LATENCY_TRIALS,
) -> dict[str, object]:
    """Run the deterministic latency gate outside an event loop."""

    return asyncio.run(measure_speculative_voice_latency(trials=trials))


__all__ = [
    "LATENCY_TRIALS",
    "measure_speculative_voice_latency",
    "run_speculative_voice_latency_gate",
]
