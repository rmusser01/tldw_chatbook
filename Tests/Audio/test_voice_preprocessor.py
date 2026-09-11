"""Tests for normalized, fail-closed pre-AEC speech admission."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import replace
import inspect
import math
import random
from typing import Any

import pytest

from tldw_chatbook.Audio.duplex_contracts import (
    AcousticDemotionReason,
    AcousticIsolationObservation,
    AcousticSafetyPath,
    AcousticSafetySnapshot,
    AecDelayEvidence,
    AecHealth,
    AudioFrame,
    DuplexMode,
    NearEndDisposition,
)
from tldw_chatbook.Audio.acoustic_isolation import AcousticIsolationMonitor
from tldw_chatbook.Audio import voice_preprocessor
from tldw_chatbook.Audio.voice_preprocessor import (
    FRAME_BYTES,
    FRAME_DURATION_NS,
    FRAME_SAMPLES,
    VoicePreprocessor,
    normalize_pcm16_frames,
)


def pcm16(value: int, samples: int = FRAME_SAMPLES) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * samples


def evidence(
    *,
    delay_ms: int = 40,
    occupancy_bounded: bool = True,
    timing_discontinuity: bool = False,
    clock_drift: bool = False,
) -> AecDelayEvidence:
    return AecDelayEvidence(
        observed_ns=100_000_000,
        capture_adc_ns=80_000_000,
        render_dac_ns=120_000_000,
        delay_ms=delay_ms,
        capture_occupancy_frames=0,
        render_occupancy_frames=0,
        occupancy_bounded=occupancy_bounded,
        timing_discontinuity=timing_discontinuity,
        clock_drift=clock_drift,
        status_flags=(),
    )


def frame(
    sequence: int,
    value: int = 0,
    *,
    generation: int = 0,
    delay_evidence: AecDelayEvidence | None = None,
    render_reference_sequence: int | None = None,
) -> AudioFrame:
    started_ns = sequence * FRAME_DURATION_NS
    return AudioFrame(
        sequence=sequence,
        started_ns=started_ns,
        ended_ns=started_ns + FRAME_DURATION_NS,
        pcm16=pcm16(value),
        clock_generation=generation,
        delay_evidence=evidence() if delay_evidence is None else delay_evidence,
        render_reference_sequence=render_reference_sequence,
    )


class FakeAec:
    def __init__(
        self,
        observations: Iterable[dict[str, float]] = (),
        *,
        health: AecHealth | None = None,
        cleaned_value: int = 7,
    ) -> None:
        self.observations = deque(observations)
        self.health = health
        self.cleaned_value = cleaned_value
        self.events: list[tuple[str, int]] = []
        self.reset_count = 0
        self.raise_on_capture = False
        self.raise_on_metrics = False
        self.metrics_calls = 0
        self.cleaned_frames: deque[bytes] = deque()

    def analyze_render(self, pcm16_bytes: bytes, *, delay_ms: int) -> None:
        assert len(pcm16_bytes) == FRAME_BYTES
        self.events.append(("render", delay_ms))

    def process_capture(self, pcm16_bytes: bytes, *, delay_ms: int) -> bytes:
        assert len(pcm16_bytes) == FRAME_BYTES
        self.events.append(("capture", delay_ms))
        if self.raise_on_capture:
            raise RuntimeError("native failure")
        if self.cleaned_frames:
            return self.cleaned_frames.popleft()
        return pcm16(self.cleaned_value)

    def metrics(self) -> dict[str, float]:
        self.metrics_calls += 1
        if self.raise_on_metrics:
            raise RuntimeError("metrics failure")
        if self.observations:
            return self.observations.popleft()
        return dict(UNHEALTHY)

    def reset(self) -> None:
        self.reset_count += 1


HEALTHY = {
    "erle_db": 6.0,
    "delay_ms": 40.0,
    "delay_estimate_available": 1.0,
    "delay_estimate_refined": 1.0,
    "delay_age_blocks": 5.0,
    "clock_drift": 0.0,
}
UNHEALTHY = {**HEALTHY, "erle_db": 4.9}
DELAY_UNAVAILABLE = {
    **HEALTHY,
    "delay_estimate_available": 0.0,
    "delay_estimate_refined": 0.0,
}


def test_aec_metric_snapshot_validator_reads_one_exact_snapshot() -> None:
    aec = FakeAec([HEALTHY], health=AecHealth.HEALTHY)

    metrics, health = voice_preprocessor.read_aec_metric_snapshot(aec)

    assert metrics == HEALTHY
    assert health is AecHealth.HEALTHY
    assert aec.metrics_calls == 1


@pytest.mark.parametrize("erle_db", [-100.0, 100.0])
def test_aec_metric_snapshot_validator_accepts_public_erle_boundaries(
    erle_db: float,
) -> None:
    aec = FakeAec([{**HEALTHY, "erle_db": erle_db}])

    metrics, _health = voice_preprocessor.read_aec_metric_snapshot(aec)

    assert metrics["erle_db"] == erle_db


@pytest.mark.parametrize("erle_db", [-100.000001, 100.000001])
def test_aec_metric_snapshot_validator_rejects_erle_outside_public_range(
    erle_db: float,
) -> None:
    aec = FakeAec([{**HEALTHY, "erle_db": erle_db}])

    with pytest.raises(ValueError, match="ERLE"):
        voice_preprocessor.read_aec_metric_snapshot(aec)


@pytest.mark.parametrize(
    "invalid_metrics",
    [
        {**HEALTHY, "unexpected": 1.0},
        {key: value for key, value in HEALTHY.items() if key != "erle_db"},
        {**HEALTHY, "erle_db": True},
        {**HEALTHY, "delay_ms": math.inf},
        {
            **HEALTHY,
            "delay_estimate_available": 0.0,
            "delay_estimate_refined": 1.0,
        },
    ],
)
def test_aec_metric_snapshot_validator_rejects_malformed_metrics(
    invalid_metrics: dict[str, float],
) -> None:
    aec = FakeAec([invalid_metrics])

    with pytest.raises((KeyError, TypeError, ValueError)):
        voice_preprocessor.read_aec_metric_snapshot(aec)


def test_aec_metric_snapshot_validator_rejects_invalid_explicit_health() -> None:
    aec = FakeAec([HEALTHY])
    aec.health = "healthy"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="health"):
        voice_preprocessor.read_aec_metric_snapshot(aec)


def test_preprocessor_delegates_metric_reads_to_shared_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aec = FakeAec([HEALTHY])
    preprocessor = VoicePreprocessor(aec=aec)
    calls: list[object] = []

    def validate(candidate: object) -> tuple[dict[str, float], AecHealth]:
        calls.append(candidate)
        return HEALTHY, AecHealth.HEALTHY

    monkeypatch.setattr(voice_preprocessor, "read_aec_metric_snapshot", validate)

    assert preprocessor._read_operational_metrics() == (  # noqa: SLF001
        HEALTHY,
        AecHealth.HEALTHY,
    )
    assert calls == [aec]


def varied_pcm16(seed: int, *, amplitude: int = 4_000) -> bytes:
    generator = random.Random(seed)
    samples: list[int] = []
    for _index in range(FRAME_SAMPLES // 2):
        value = generator.randint(1, amplitude)
        sign = 1 if generator.getrandbits(1) else -1
        samples.extend((sign * value, -sign * value))
    return b"".join(value.to_bytes(2, "little", signed=True) for value in samples)


def timed_frame(
    sequence: int,
    pcm: bytes,
    *,
    generation: int = 0,
    delay_ms: int = 40,
) -> AudioFrame:
    capture_adc_ns = 1_000_000_000 + sequence * FRAME_DURATION_NS
    return AudioFrame(
        sequence=sequence,
        started_ns=capture_adc_ns,
        ended_ns=capture_adc_ns + FRAME_DURATION_NS,
        pcm16=pcm,
        clock_generation=generation,
        delay_evidence=AecDelayEvidence(
            observed_ns=capture_adc_ns,
            capture_adc_ns=capture_adc_ns,
            render_dac_ns=capture_adc_ns + delay_ms * 1_000_000,
            delay_ms=delay_ms,
            capture_occupancy_frames=0,
            render_occupancy_frames=0,
            occupancy_bounded=True,
            timing_discontinuity=False,
            clock_drift=False,
        ),
        render_reference_sequence=sequence,
    )


def timed_render_frame(sequence: int, pcm: bytes, *, generation: int = 0) -> AudioFrame:
    return timed_frame(sequence, pcm, generation=generation)


class ScriptedIsolationMonitor:
    def __init__(
        self,
        dispositions: Iterable[NearEndDisposition | None] = (),
        *,
        path: AcousticSafetyPath = AcousticSafetyPath.AEC,
    ) -> None:
        self.dispositions = deque(dispositions)
        self.path = path
        self.calls: list[dict[str, Any]] = []
        self.reset_generations: list[int] = []

    def observe(self, **kwargs: Any) -> AcousticIsolationObservation:
        self.calls.append(kwargs)
        disposition = self.dispositions.popleft() if self.dispositions else None
        open_path = self.path in {
            AcousticSafetyPath.AEC,
            AcousticSafetyPath.ACOUSTIC_ISOLATION,
        }
        if not kwargs["native_processor_ok"] or disposition is NearEndDisposition.FENCE:
            return AcousticIsolationObservation(
                AcousticSafetySnapshot(
                    AcousticSafetyPath.HALF_DUPLEX,
                    DuplexMode.HALF_DUPLEX,
                    kwargs["capture"].clock_generation,
                    False,
                    AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE
                    if not kwargs["native_processor_ok"]
                    else AcousticDemotionReason.CORRELATED_RENDER,
                ),
                NearEndDisposition.FENCE if kwargs["near_end_speech"] else None,
            )
        return AcousticIsolationObservation(
            AcousticSafetySnapshot(
                self.path,
                DuplexMode.FULL_DUPLEX if open_path else DuplexMode.HALF_DUPLEX,
                kwargs["capture"].clock_generation,
                open_path,
                None,
            ),
            disposition,
        )

    def reset_for_route(self, clock_generation: int) -> None:
        self.reset_generations.append(clock_generation)


class ClosedIsolationMonitor(ScriptedIsolationMonitor):
    def __init__(self, reason: AcousticDemotionReason) -> None:
        super().__init__()
        self.reason = reason

    def observe(self, **kwargs: Any) -> AcousticIsolationObservation:
        self.calls.append(kwargs)
        return AcousticIsolationObservation(
            AcousticSafetySnapshot(
                AcousticSafetyPath.HALF_DUPLEX,
                DuplexMode.HALF_DUPLEX,
                kwargs["capture"].clock_generation,
                False,
                self.reason,
            ),
            None,
        )


class MutableIsolationMonitor(ScriptedIsolationMonitor):
    def __init__(self) -> None:
        super().__init__()
        self.reason: AcousticDemotionReason | None = None

    def observe(self, **kwargs: Any) -> AcousticIsolationObservation:
        if self.reason is None:
            return super().observe(**kwargs)
        self.calls.append(kwargs)
        return AcousticIsolationObservation(
            AcousticSafetySnapshot(
                AcousticSafetyPath.HALF_DUPLEX,
                DuplexMode.HALF_DUPLEX,
                kwargs["capture"].clock_generation,
                False,
                self.reason,
            ),
            None,
        )


def test_normalizes_16khz_mono_to_one_48khz_ten_millisecond_frame() -> None:
    normalized = normalize_pcm16_frames(
        pcm16(1234, samples=160),
        sample_rate=16_000,
        channels=1,
    )

    assert len(normalized) == 1
    assert len(normalized[0]) == FRAME_BYTES
    assert normalized[0] == pcm16(1234)


def test_normalizes_48khz_stereo_to_mono_and_preserves_ten_ms_chunks() -> None:
    stereo_sample = pcm16(1000, samples=1) + pcm16(-500, samples=1)
    normalized = normalize_pcm16_frames(
        stereo_sample * (FRAME_SAMPLES * 2),
        sample_rate=48_000,
        channels=2,
    )

    assert len(normalized) == 2
    assert normalized == (pcm16(250), pcm16(250))


def test_normalizer_rejects_partial_ten_millisecond_input() -> None:
    with pytest.raises(ValueError, match="ten-millisecond"):
        normalize_pcm16_frames(
            pcm16(0, samples=100),
            sample_rate=48_000,
            channels=1,
        )


def test_normalizer_rejects_downsampling_instead_of_aliasing() -> None:
    high_frequency = pcm16(20_000, samples=1) + pcm16(-20_000, samples=1)

    with pytest.raises(ValueError, match="downsampling"):
        normalize_pcm16_frames(
            high_frequency * 960,
            sample_rate=96_000,
            channels=1,
        )


def test_upsampling_preserves_multi_frame_boundary_continuity() -> None:
    source = b"".join(pcm16(index - 240, samples=1) for index in range(480))

    normalized = normalize_pcm16_frames(
        source,
        sample_rate=24_000,
        channels=1,
    )

    assert len(normalized) == 2
    assert all(len(chunk) == FRAME_BYTES for chunk in normalized)
    before_boundary = int.from_bytes(normalized[0][-2:], "little", signed=True)
    after_boundary = int.from_bytes(normalized[1][:2], "little", signed=True)
    assert 0 <= after_boundary - before_boundary <= 1


@pytest.mark.asyncio
async def test_render_is_analyzed_in_order_before_capture_and_post_aec_vad() -> None:
    aec = FakeAec([HEALTHY])
    events = aec.events
    admitted: list[AudioFrame] = []

    def vad(cleaned: AudioFrame) -> bool:
        events.append(("vad", int.from_bytes(cleaned.pcm16[:2], "little")))
        return True

    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=ScriptedIsolationMonitor([NearEndDisposition.ADMIT]),
        vad=vad,
        on_admitted_frame=admitted.append,
        healthy_streak=1,
    )

    shared_evidence = evidence(delay_ms=30)
    cleaned = await preprocessor.process_capture(
        frame(
            2,
            delay_evidence=shared_evidence,
            render_reference_sequence=1,
        ),
        render_frames=(frame(0, 1), frame(1, 2)),
        assistant_rendering=True,
    )

    assert [name for name, _value in events] == ["render", "render", "capture", "vad"]
    assert cleaned is not None
    assert admitted == [cleaned]
    assert admitted[0].pcm16 == pcm16(7)
    assert events[:3] == [("render", 30), ("render", 30), ("capture", 30)]


def test_process_capture_has_no_caller_supplied_delay_override() -> None:
    assert (
        "delay_ms"
        not in inspect.signature(VoicePreprocessor.process_capture).parameters
    )


@pytest.mark.asyncio
async def test_health_opens_after_bounded_streak_and_closes_on_first_failure() -> None:
    aec = FakeAec([HEALTHY, HEALTHY, UNHEALTHY])
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=ScriptedIsolationMonitor([NearEndDisposition.ADMIT] * 3),
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        healthy_streak=2,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=True)
    assert preprocessor.health is AecHealth.WARMING
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX
    assert admitted == []

    await preprocessor.process_capture(frame(1), assistant_rendering=True)
    assert preprocessor.health is AecHealth.HEALTHY
    assert len(admitted) == 1

    await preprocessor.process_capture(frame(2), assistant_rendering=True)
    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX
    assert len(admitted) == 1


@pytest.mark.asyncio
async def test_capture_sequence_gap_degrades_before_admission() -> None:
    admitted: list[AudioFrame] = []
    acknowledgements: list[tuple[int, int, bool, bool]] = []
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY, HEALTHY]),
        isolation_monitor=ScriptedIsolationMonitor([NearEndDisposition.ADMIT]),
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        on_processed=lambda sequence, generation, dsp_ok, vad_ok: (
            acknowledgements.append((sequence, generation, dsp_ok, vad_ok))
        ),
        healthy_streak=1,
    )

    first = await preprocessor.process_capture(frame(0), assistant_rendering=True)
    after_gap = await preprocessor.process_capture(frame(2), assistant_rendering=True)

    assert first is not None
    assert after_gap is None
    assert [captured.sequence for captured in admitted] == [0]
    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX
    assert (
        preprocessor.safety.demotion_reason
        is AcousticDemotionReason.CAPTURE_SEQUENCE_GAP
    )
    assert acknowledgements == [(0, 0, True, True), (2, 0, False, False)]


@pytest.mark.asyncio
async def test_capture_generation_change_requires_explicit_route_reset() -> None:
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY, HEALTHY, HEALTHY]),
        isolation_monitor=ScriptedIsolationMonitor([NearEndDisposition.ADMIT] * 3),
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        healthy_streak=1,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=True)
    before_reset = await preprocessor.process_capture(
        frame(1, generation=1),
        assistant_rendering=True,
    )

    assert before_reset is None
    assert [captured.sequence for captured in admitted] == [0]
    assert preprocessor.health is AecHealth.DEGRADED

    preprocessor.reset_for_device_route(1)
    after_reset = await preprocessor.process_capture(
        frame(1, generation=1),
        assistant_rendering=True,
    )

    assert after_reset is not None
    assert [captured.sequence for captured in admitted] == [0, 1]
    assert preprocessor.health is AecHealth.HEALTHY


@pytest.mark.asyncio
async def test_explicit_degraded_health_closes_admission() -> None:
    aec = FakeAec([UNHEALTHY], health=AecHealth.DEGRADED)
    observed: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=ScriptedIsolationMonitor([NearEndDisposition.ADMIT]),
        vad=lambda _frame: True,
        on_admitted_frame=observed.append,
    )

    await preprocessor.process_capture(frame(1), assistant_rendering=True)

    assert observed == []
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX


@pytest.mark.asyncio
async def test_warming_has_a_bounded_recoverable_half_duplex_transition() -> None:
    aec = FakeAec([UNHEALTHY, UNHEALTHY, HEALTHY])
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=ScriptedIsolationMonitor([NearEndDisposition.ADMIT]),
        healthy_streak=1,
        warming_observation_limit=2,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=True)
    assert preprocessor.health is AecHealth.WARMING
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX

    await preprocessor.process_capture(frame(1), assistant_rendering=True)
    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX

    await preprocessor.process_capture(frame(2), assistant_rendering=True)
    assert preprocessor.health is AecHealth.HEALTHY
    assert preprocessor.mode is DuplexMode.FULL_DUPLEX


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {**HEALTHY, "delay_estimate_available": 0.0},
        {**HEALTHY, "delay_estimate_refined": 0.0},
        {**HEALTHY, "delay_age_blocks": 26.0},
        {**HEALTHY, "clock_drift": 1.0},
        {**HEALTHY, "erle_db": math.nan},
        {**HEALTHY, "delay_estimate_available": 0.5},
    ],
)
async def test_missing_malformed_or_stale_native_health_fails_closed(
    metrics: dict[str, float],
) -> None:
    preprocessor = VoicePreprocessor(
        aec=FakeAec([metrics]),
        healthy_streak=1,
        warming_observation_limit=1,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=True)

    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timing", "reason"),
    [
        (None, AcousticDemotionReason.MISSING_TIMING),
        (
            evidence(occupancy_bounded=False),
            AcousticDemotionReason.UNBOUNDED_TIMING,
        ),
        (
            evidence(timing_discontinuity=True),
            AcousticDemotionReason.TIMING_DISCONTINUITY,
        ),
        (
            evidence(clock_drift=True),
            AcousticDemotionReason.TIMING_DISCONTINUITY,
        ),
    ],
)
async def test_missing_or_unreliable_transport_delay_evidence_fails_closed(
    timing: AecDelayEvidence | None,
    reason: AcousticDemotionReason,
) -> None:
    preprocessor = VoicePreprocessor(aec=FakeAec([HEALTHY]), healthy_streak=1)
    capture = frame(0, delay_evidence=timing)
    if timing is None:
        capture = AudioFrame(
            sequence=0,
            started_ns=0,
            ended_ns=FRAME_DURATION_NS,
            pcm16=pcm16(0),
        )

    cleaned = await preprocessor.process_capture(capture, assistant_rendering=True)

    assert cleaned is None
    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.safety.demotion_reason is reason


@pytest.mark.asyncio
async def test_unusable_timing_runs_honest_half_duplex_during_and_after_playback() -> (
    None
):
    vad_frames: list[AudioFrame] = []
    admitted: list[AudioFrame] = []
    acknowledgements: list[tuple[int, int, bool, bool]] = []
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY, HEALTHY]),
        vad=lambda captured: vad_frames.append(captured) or True,
        on_admitted_frame=admitted.append,
        on_processed=lambda sequence, generation, dsp_ok, vad_ok: (
            acknowledgements.append((sequence, generation, dsp_ok, vad_ok))
        ),
        healthy_streak=1,
    )
    during = AudioFrame(
        sequence=0,
        started_ns=0,
        ended_ns=FRAME_DURATION_NS,
        pcm16=pcm16(11),
    )
    after = AudioFrame(
        sequence=1,
        started_ns=FRAME_DURATION_NS,
        ended_ns=2 * FRAME_DURATION_NS,
        pcm16=pcm16(12),
    )

    during_result = await preprocessor.process_capture(
        during,
        assistant_rendering=True,
    )
    after_result = await preprocessor.process_capture(
        after,
        assistant_rendering=False,
    )

    assert during_result is None
    assert after_result == after
    assert vad_frames == [after]
    assert admitted == [after]
    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX
    assert acknowledgements == [(0, 0, True, True), (1, 0, True, True)]


@pytest.mark.asyncio
async def test_render_reference_gap_zero_then_two_degrades_before_capture() -> None:
    aec = FakeAec([HEALTHY])
    preprocessor = VoicePreprocessor(aec=aec, healthy_streak=1)

    cleaned = await preprocessor.process_capture(
        frame(0, render_reference_sequence=2),
        render_frames=(frame(0, 1), frame(2, 2)),
        assistant_rendering=True,
    )

    assert cleaned is None
    assert aec.events == [("render", 40)]
    assert preprocessor.health is AecHealth.DEGRADED


@pytest.mark.asyncio
async def test_first_paired_render_uses_capture_timeline_after_route_reset() -> None:
    aec = FakeAec([HEALTHY])
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=ScriptedIsolationMonitor(),
        healthy_streak=1,
        clock_generation=3,
    )
    capture = replace(
        timed_frame(17, pcm16(7), generation=3),
        render_reference_sequence=17,
    )
    render = timed_render_frame(17, pcm16(8), generation=3)

    await preprocessor.process_capture(
        capture,
        render_frames=(render,),
        assistant_rendering=True,
    )

    assert aec.events[0] == ("render", 40)
    assert preprocessor.health is AecHealth.HEALTHY


@pytest.mark.asyncio
async def test_omitted_current_render_reference_degrades_before_capture() -> None:
    aec = FakeAec([HEALTHY])
    preprocessor = VoicePreprocessor(aec=aec, healthy_streak=1)

    cleaned = await preprocessor.process_capture(
        frame(0, render_reference_sequence=0),
        render_frames=(),
        assistant_rendering=True,
    )

    assert cleaned is None
    assert aec.events == []
    assert preprocessor.health is AecHealth.DEGRADED


@pytest.mark.asyncio
async def test_absent_native_aec_is_honest_half_duplex() -> None:
    vad_frames: list[AudioFrame] = []
    admitted: list[AudioFrame] = []

    def vad(cleaned: AudioFrame) -> bool:
        vad_frames.append(cleaned)
        return True

    preprocessor = VoicePreprocessor(
        aec=None,
        vad=vad,
        on_admitted_frame=admitted.append,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=True)
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX
    assert vad_frames == []
    assert admitted == []

    await preprocessor.process_capture(frame(1), assistant_rendering=False)
    assert vad_frames == [frame(1)]
    assert admitted == [frame(1)]


@pytest.mark.asyncio
async def test_idle_vad_replays_bounded_rejected_audio_before_speech() -> None:
    verdicts = iter([False, False, True])
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=None,
        vad=lambda _frame: next(verdicts),
        vad_preroll_ms=20,
        on_admitted_frame=admitted.append,
    )

    for sequence in range(3):
        await preprocessor.process_capture(
            frame(sequence, sequence + 1),
            assistant_rendering=False,
        )

    assert [captured.sequence for captured in admitted] == [0, 1, 2]
    assert [captured.speech_started_ns for captured in admitted] == [
        20_000_000,
        20_000_000,
        None,
    ]
    assert [captured.started_ns for captured in admitted] == [0, 10_000_000, 20_000_000]
    assert [captured.pcm16 for captured in admitted] == [
        frame(i, i + 1).pcm16 for i in range(3)
    ]


@pytest.mark.asyncio
async def test_idle_vad_preroll_keeps_only_configured_tail() -> None:
    verdicts = iter([False] * 5 + [True])
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=None,
        vad=lambda _frame: next(verdicts),
        vad_preroll_ms=20,
        on_admitted_frame=admitted.append,
    )

    for sequence in range(6):
        await preprocessor.process_capture(
            frame(sequence, sequence + 1),
            assistant_rendering=False,
        )

    assert [captured.sequence for captured in admitted] == [3, 4, 5]


@pytest.mark.asyncio
async def test_playback_and_route_reset_clear_idle_vad_preroll() -> None:
    verdicts = iter([False, True, False, True])
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=None,
        vad=lambda _frame: next(verdicts),
        vad_preroll_ms=240,
        on_admitted_frame=admitted.append,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=False)
    await preprocessor.process_capture(frame(1), assistant_rendering=True)
    await preprocessor.process_capture(frame(2), assistant_rendering=False)
    await preprocessor.process_capture(frame(3), assistant_rendering=False)
    preprocessor.reset_for_device_route(1)
    await preprocessor.process_capture(
        frame(0, generation=1),
        assistant_rendering=False,
    )

    assert [
        (captured.sequence, captured.clock_generation) for captured in admitted
    ] == [
        (2, 0),
        (0, 1),
    ]


@pytest.mark.parametrize("bad_preroll", [True, -1, 1.5, "240"])
def test_vad_preroll_requires_an_exact_nonnegative_integer(bad_preroll: object) -> None:
    with pytest.raises(ValueError, match="pre-roll"):
        VoicePreprocessor(aec=None, vad_preroll_ms=bad_preroll)  # type: ignore[arg-type]


def test_webrtc_vad_uses_exact_pipeline_frame_contract(monkeypatch) -> None:
    calls: list[tuple[bytes, int]] = []

    class FakeDetector:
        def __init__(self, mode: int) -> None:
            assert mode == 2

        def is_speech(self, payload: bytes, sample_rate: int) -> bool:
            calls.append((payload, sample_rate))
            return True

    monkeypatch.setattr(
        voice_preprocessor,
        "import_module",
        lambda name: (
            type("WebRtcVad", (), {"Vad": FakeDetector})
            if name == "webrtcvad"
            else None
        ),
    )
    detect = voice_preprocessor.create_webrtc_vad(aggressiveness=2)
    candidate = frame(0, 19)

    assert detect(candidate) is True
    assert calls == [(candidate.pcm16, 48_000)]


@pytest.mark.asyncio
async def test_discontinuity_closes_admission_and_device_reset_requires_rewarming() -> (
    None
):
    aec = FakeAec([HEALTHY, HEALTHY])
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=ScriptedIsolationMonitor([NearEndDisposition.ADMIT] * 2),
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        healthy_streak=1,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=True)
    assert len(admitted) == 1
    await preprocessor.process_capture(
        frame(1),
        assistant_rendering=True,
        discontinuity=True,
    )
    assert preprocessor.health is AecHealth.DEGRADED
    assert len(admitted) == 1

    with pytest.raises(ValueError, match="advance"):
        preprocessor.reset_for_device_route(0)

    preprocessor.reset_for_device_route(1)
    assert aec.reset_count == 1
    assert preprocessor.health is AecHealth.WARMING
    assert preprocessor.active_clock_generation == 1


@pytest.mark.asyncio
async def test_native_failure_during_active_full_duplex_fails_acknowledgement() -> None:
    aec = FakeAec([HEALTHY])
    admitted: list[AudioFrame] = []
    acknowledgements: list[tuple[int, int, bool, bool]] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=ScriptedIsolationMonitor([NearEndDisposition.ADMIT]),
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        on_processed=lambda sequence, generation, dsp_ok, vad_ok: (
            acknowledgements.append((sequence, generation, dsp_ok, vad_ok))
        ),
        healthy_streak=1,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=True)
    assert preprocessor.health is AecHealth.HEALTHY
    aec.raise_on_capture = True

    cleaned = await preprocessor.process_capture(frame(1), assistant_rendering=True)

    assert cleaned is None
    assert [captured.sequence for captured in admitted] == [0]
    assert preprocessor.health is AecHealth.DEGRADED
    assert acknowledgements == [(0, 0, True, True), (1, 0, False, False)]


@pytest.mark.asyncio
async def test_post_aec_health_failure_marks_vad_acknowledgement_failed() -> None:
    aec = FakeAec([HEALTHY, UNHEALTHY])
    vad_frames: list[AudioFrame] = []
    acknowledgements: list[tuple[int, int, bool, bool]] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        vad=lambda captured: vad_frames.append(captured) or True,
        on_processed=lambda sequence, generation, dsp_ok, vad_ok: (
            acknowledgements.append((sequence, generation, dsp_ok, vad_ok))
        ),
        healthy_streak=1,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=True)
    failed = await preprocessor.process_capture(frame(1), assistant_rendering=True)

    assert failed is None
    assert [captured.sequence for captured in vad_frames] == [0, 1]
    assert preprocessor.health is AecHealth.DEGRADED
    assert acknowledgements == [(0, 0, True, True), (1, 0, True, True)]


@pytest.mark.asyncio
async def test_native_processing_failure_uses_raw_half_duplex_after_render() -> None:
    aec = FakeAec([HEALTHY, HEALTHY])
    aec.raise_on_capture = True
    vad_frames: list[AudioFrame] = []
    admitted: list[AudioFrame] = []
    acknowledgements: list[tuple[int, int, bool, bool]] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=ScriptedIsolationMonitor([NearEndDisposition.ADMIT] * 5),
        vad=lambda captured: vad_frames.append(captured) or True,
        on_admitted_frame=admitted.append,
        on_processed=lambda sequence, generation, dsp_ok, vad_ok: (
            acknowledgements.append((sequence, generation, dsp_ok, vad_ok))
        ),
        healthy_streak=1,
    )

    during = await preprocessor.process_capture(frame(0, 11), assistant_rendering=True)
    after = await preprocessor.process_capture(frame(1, 12), assistant_rendering=False)

    assert during is None
    assert after == frame(1, 12)
    assert vad_frames == [frame(1, 12)]
    assert admitted == [frame(1, 12)]
    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX
    assert acknowledgements == [(0, 0, True, True), (1, 0, True, True)]


@pytest.mark.asyncio
async def test_successful_unhealthy_aec_vads_cleaned_but_admits_raw_capture() -> None:
    vad_frames: list[AudioFrame] = []
    admitted: list[AudioFrame] = []
    capture = frame(0, 1_000)
    preprocessor = VoicePreprocessor(
        aec=FakeAec([UNHEALTHY], cleaned_value=7),
        isolation_monitor=ScriptedIsolationMonitor(),
        vad=lambda cleaned: vad_frames.append(cleaned) or True,
        on_admitted_frame=admitted.append,
    )

    result = await preprocessor.process_capture(
        capture,
        assistant_rendering=False,
    )

    assert result == capture
    assert admitted == [capture]
    assert vad_frames == [replace(capture, pcm16=pcm16(7))]
    assert preprocessor.health is AecHealth.WARMING
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX


@pytest.mark.asyncio
async def test_explicit_backend_warming_closes_and_rewarms_full_duplex() -> None:
    aec = FakeAec([HEALTHY] * 5)
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=ScriptedIsolationMonitor([NearEndDisposition.ADMIT] * 5),
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        healthy_streak=2,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=True)
    await preprocessor.process_capture(frame(1), assistant_rendering=True)
    assert preprocessor.health is AecHealth.HEALTHY
    assert [captured.sequence for captured in admitted] == [1]

    aec.health = AecHealth.WARMING
    await preprocessor.process_capture(frame(2), assistant_rendering=True)
    assert preprocessor.health is AecHealth.WARMING
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX
    assert [captured.sequence for captured in admitted] == [1]

    aec.health = AecHealth.HEALTHY
    await preprocessor.process_capture(frame(3), assistant_rendering=True)
    assert preprocessor.health is AecHealth.WARMING
    await preprocessor.process_capture(frame(4), assistant_rendering=True)
    assert preprocessor.health is AecHealth.HEALTHY
    assert preprocessor.mode is DuplexMode.FULL_DUPLEX
    assert [captured.sequence for captured in admitted] == [1, 4]


@pytest.mark.asyncio
async def test_isolation_path_opens_while_native_aec_health_remains_warming() -> None:
    aec = FakeAec([DELAY_UNAVAILABLE] * 100, cleaned_value=0)
    monitor = AcousticIsolationMonitor(window_frames=100, required_windows=1)
    preprocessor = VoicePreprocessor(
        aec=aec,
        vad=lambda _frame: False,
        isolation_monitor=monitor,
    )
    for sequence in range(100):
        await preprocessor.process_capture(
            timed_frame(sequence, varied_pcm16(10_000 + sequence)),
            render_frames=(
                timed_render_frame(sequence, varied_pcm16(20_000 + sequence)),
            ),
            assistant_rendering=True,
        )

    assert preprocessor.health is AecHealth.WARMING
    assert preprocessor.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
    assert preprocessor.mode is DuplexMode.FULL_DUPLEX
    assert aec.metrics_calls == 100


@pytest.mark.asyncio
async def test_isolation_monitor_binds_to_nonzero_initial_route_generation() -> None:
    preprocessor = VoicePreprocessor(
        aec=FakeAec([DELAY_UNAVAILABLE] * 100, cleaned_value=0),
        vad=lambda _frame: False,
        isolation_monitor=AcousticIsolationMonitor(
            window_frames=100,
            required_windows=1,
        ),
        clock_generation=3,
        warming_observation_limit=1,
    )

    for sequence in range(100):
        await preprocessor.process_capture(
            timed_frame(sequence, varied_pcm16(31_000 + sequence), generation=3),
            render_frames=(
                timed_render_frame(
                    sequence,
                    varied_pcm16(32_000 + sequence),
                    generation=3,
                ),
            ),
            assistant_rendering=True,
        )

    assert preprocessor.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
    assert preprocessor.safety.route_generation == 3
    assert preprocessor.health is AecHealth.DEGRADED


@pytest.mark.asyncio
@pytest.mark.parametrize("saturated_source", ["capture", "render"])
async def test_healthy_aec_cannot_override_real_monitor_saturation(
    saturated_source: str,
) -> None:
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY], cleaned_value=0),
        vad=lambda _frame: False,
        isolation_monitor=AcousticIsolationMonitor(
            window_frames=100, required_windows=1
        ),
        healthy_streak=1,
    )
    raw_capture = pcm16(32_767) if saturated_source == "capture" else varied_pcm16(41)
    raw_render = pcm16(32_767) if saturated_source == "render" else varied_pcm16(42)

    await preprocessor.process_capture(
        timed_frame(0, raw_capture),
        render_frames=(timed_render_frame(0, raw_render),),
        assistant_rendering=True,
    )

    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert preprocessor.safety.demotion_reason is AcousticDemotionReason.SATURATION


@pytest.mark.asyncio
async def test_hard_monitor_demotion_requires_a_fresh_healthy_streak() -> None:
    monitor = MutableIsolationMonitor()
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY] * 7, cleaned_value=0),
        vad=lambda _frame: False,
        isolation_monitor=monitor,
        healthy_streak=3,
    )

    for sequence in range(3):
        await preprocessor.process_capture(frame(sequence), assistant_rendering=True)
    assert preprocessor.health is AecHealth.HEALTHY
    assert preprocessor.safety.path is AcousticSafetyPath.AEC

    monitor.reason = AcousticDemotionReason.SATURATION
    await preprocessor.process_capture(frame(3), assistant_rendering=True)
    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX

    monitor.reason = None
    await preprocessor.process_capture(frame(4), assistant_rendering=False)
    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX
    await preprocessor.process_capture(frame(5), assistant_rendering=False)
    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX
    await preprocessor.process_capture(frame(6), assistant_rendering=False)
    assert preprocessor.health is AecHealth.HEALTHY
    assert preprocessor.safety.path is AcousticSafetyPath.AEC


@pytest.mark.asyncio
async def test_healthy_aec_cannot_override_real_monitor_reference_gap() -> None:
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY, HEALTHY], cleaned_value=0),
        vad=lambda _frame: False,
        isolation_monitor=AcousticIsolationMonitor(
            window_frames=100, required_windows=1
        ),
        healthy_streak=1,
    )
    await preprocessor.process_capture(
        timed_frame(0, varied_pcm16(51)),
        render_frames=(timed_render_frame(0, varied_pcm16(52)),),
        assistant_rendering=True,
    )

    await preprocessor.process_capture(
        replace(
            timed_frame(1, varied_pcm16(53)),
            render_reference_sequence=2,
        ),
        render_frames=(
            timed_render_frame(1, varied_pcm16(54)),
            timed_render_frame(2, varied_pcm16(55)),
        ),
        assistant_rendering=True,
    )

    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert (
        preprocessor.safety.demotion_reason
        is AcousticDemotionReason.RENDER_REFERENCE_GAP
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        AcousticDemotionReason.INAUDIBLE_RENDER,
        AcousticDemotionReason.CORRELATED_RENDER,
    ],
)
async def test_healthy_aec_can_override_only_nonproof_isolation_reasons(
    reason: AcousticDemotionReason,
) -> None:
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY], cleaned_value=0),
        vad=lambda _frame: False,
        isolation_monitor=ClosedIsolationMonitor(reason),
        healthy_streak=1,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=True)

    assert preprocessor.health is AecHealth.HEALTHY
    assert preprocessor.safety.path is AcousticSafetyPath.AEC
    assert preprocessor.safety.demotion_reason is None


@pytest.mark.asyncio
async def test_healthy_aec_closes_for_unallowlisted_monitor_reason() -> None:
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY], cleaned_value=0),
        vad=lambda _frame: False,
        isolation_monitor=ClosedIsolationMonitor(
            AcousticDemotionReason.AMBIGUOUS_CORRELATION
        ),
        healthy_streak=1,
    )

    await preprocessor.process_capture(frame(0), assistant_rendering=True)

    assert preprocessor.health is AecHealth.DEGRADED
    assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert (
        preprocessor.safety.demotion_reason
        is AcousticDemotionReason.AMBIGUOUS_CORRELATION
    )


@pytest.mark.asyncio
async def test_render_dominant_echo_discards_pending_frames_and_revokes_isolation() -> (
    None
):
    aec = FakeAec([DELAY_UNAVAILABLE] * 105, cleaned_value=0)
    aec.cleaned_frames.extend(
        [pcm16(0)] * 100 + [varied_pcm16(30_000 + index) for index in range(5)]
    )
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        vad=lambda cleaned: cleaned.sequence >= 100,
        on_admitted_frame=admitted.append,
        isolation_monitor=AcousticIsolationMonitor(
            window_frames=100,
            required_windows=1,
        ),
    )
    renders = [varied_pcm16(20_000 + sequence) for sequence in range(105)]
    for sequence in range(100):
        await preprocessor.process_capture(
            timed_frame(sequence, varied_pcm16(10_000 + sequence)),
            render_frames=(timed_render_frame(sequence, renders[sequence]),),
            assistant_rendering=True,
        )
    assert preprocessor.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION

    for sequence in range(100, 105):
        await preprocessor.process_capture(
            timed_frame(sequence, renders[sequence - 4]),
            render_frames=(timed_render_frame(sequence, renders[sequence]),),
            assistant_rendering=True,
        )

    assert admitted == []
    assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert (
        preprocessor.safety.demotion_reason is AcousticDemotionReason.CORRELATED_RENDER
    )


@pytest.mark.asyncio
async def test_playback_speech_replays_five_cleaned_frames_then_admits_immediately() -> (
    None
):
    dispositions = [NearEndDisposition.PENDING] * 4 + [NearEndDisposition.ADMIT] * 2
    monitor = ScriptedIsolationMonitor(dispositions)
    aec = FakeAec([HEALTHY] * 7)
    aec.cleaned_frames.extend(pcm16(100 + index) for index in range(7))
    admitted: list[AudioFrame] = []
    acknowledgements: list[tuple[int, int, bool, bool]] = []
    vad_sequences: list[int] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=monitor,
        vad=lambda cleaned: vad_sequences.append(cleaned.sequence) or True,
        on_admitted_frame=admitted.append,
        on_processed=lambda *values: acknowledgements.append(values),
        healthy_streak=1,
    )

    for sequence in range(4):
        await preprocessor.process_capture(frame(sequence), assistant_rendering=True)
        assert admitted == []
    await preprocessor.process_capture(frame(4), assistant_rendering=True)

    assert [item.sequence for item in admitted] == [0, 1, 2, 3, 4]
    assert [
        int.from_bytes(item.pcm16[:2], "little", signed=True) for item in admitted
    ] == [
        100,
        101,
        102,
        103,
        104,
    ]
    await preprocessor.process_capture(frame(5), assistant_rendering=True)
    assert [item.sequence for item in admitted] == [0, 1, 2, 3, 4, 5]
    assert vad_sequences == list(range(6))
    assert acknowledgements == [(index, 0, True, True) for index in range(6)]

    monitor.dispositions.extend(
        [None, NearEndDisposition.PENDING, NearEndDisposition.PENDING]
    )
    preprocessor._vad = lambda cleaned: (  # type: ignore[method-assign]
        vad_sequences.append(cleaned.sequence) or cleaned.sequence != 6
    )
    await preprocessor.process_capture(frame(6), assistant_rendering=True)
    await preprocessor.process_capture(frame(7), assistant_rendering=True)
    assert [item.sequence for item in admitted] == [0, 1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_first_positive_post_render_frame_replays_contiguous_pending_speech() -> (
    None
):
    monitor = ScriptedIsolationMonitor([NearEndDisposition.PENDING] * 4 + [None])
    aec = FakeAec([HEALTHY] * 5)
    aec.cleaned_frames.extend(pcm16(100 + sequence) for sequence in range(5))
    events: list[tuple[str, int | None]] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=monitor,
        vad=lambda _frame: True,
        on_admitted_frame=lambda admitted: events.append(("admit", admitted.sequence)),
        on_classification_changed=lambda watermark: events.append(
            (
                "classification",
                None if watermark is None else watermark.through_sequence,
            )
        ),
        healthy_streak=1,
    )

    for sequence in range(4):
        await preprocessor.process_capture(frame(sequence), assistant_rendering=True)
    await preprocessor.process_capture(frame(4), assistant_rendering=False)

    assert [event for event in events if event[0] == "admit"] == [
        ("admit", sequence) for sequence in range(5)
    ]
    assert events[-6:] == [
        *(("admit", sequence) for sequence in range(5)),
        ("classification", None),
    ]
    assert preprocessor.pending_classification is None


@pytest.mark.asyncio
async def test_post_render_vad_negative_clears_pending_without_replay() -> None:
    monitor = ScriptedIsolationMonitor([NearEndDisposition.PENDING] * 4 + [None])
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY] * 5),
        isolation_monitor=monitor,
        vad=lambda cleaned: cleaned.sequence < 4,
        on_admitted_frame=admitted.append,
        healthy_streak=1,
    )

    for sequence in range(4):
        await preprocessor.process_capture(frame(sequence), assistant_rendering=True)
    await preprocessor.process_capture(frame(4), assistant_rendering=False)

    assert admitted == []
    assert preprocessor.pending_classification is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["timing-gap", "discontinuity", "route-reset"])
async def test_post_render_discontinuity_or_reset_drops_pending_speech(
    boundary: str,
) -> None:
    monitor = ScriptedIsolationMonitor([NearEndDisposition.PENDING] * 4 + [None])
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY] * 5),
        isolation_monitor=monitor,
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        healthy_streak=1,
    )

    for sequence in range(4):
        await preprocessor.process_capture(frame(sequence), assistant_rendering=True)

    capture = frame(4)
    kwargs: dict[str, bool] = {}
    if boundary == "timing-gap":
        capture = replace(
            capture,
            started_ns=capture.started_ns + FRAME_DURATION_NS,
            ended_ns=capture.ended_ns + FRAME_DURATION_NS,
        )
    elif boundary == "discontinuity":
        kwargs["discontinuity"] = True
    else:
        preprocessor.reset_for_device_route(1)
        capture = frame(0, generation=1)

    await preprocessor.process_capture(
        capture,
        assistant_rendering=False,
        **kwargs,
    )

    assert [item.sequence for item in admitted] == [capture.sequence]
    assert admitted[0].clock_generation == capture.clock_generation
    assert preprocessor.pending_classification is None
    if boundary != "route-reset":
        assert preprocessor.health is AecHealth.DEGRADED
        assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX
        assert (
            preprocessor.safety.demotion_reason
            is AcousticDemotionReason.TIMING_DISCONTINUITY
        )


@pytest.mark.asyncio
async def test_pending_watermark_precedes_ack_and_releases_after_ordered_replay() -> (
    None
):
    monitor = ScriptedIsolationMonitor(
        [NearEndDisposition.PENDING] * 4 + [NearEndDisposition.ADMIT]
    )
    events: list[tuple[str, object]] = []
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY] * 5),
        isolation_monitor=monitor,
        vad=lambda _frame: True,
        on_admitted_frame=lambda admitted: events.append(("admit", admitted.sequence)),
        on_processed=lambda sequence, *_rest: events.append(("ack", sequence)),
        on_classification_changed=lambda watermark: events.append(
            ("classification", watermark)
        ),
        healthy_streak=1,
    )

    for sequence in range(4):
        await preprocessor.process_capture(frame(sequence), assistant_rendering=True)
        watermark = preprocessor.pending_classification
        assert watermark is not None
        assert watermark.clock_generation == 0
        assert watermark.first_sequence == 0
        assert watermark.first_started_ns == 0
        assert watermark.through_sequence == sequence
        assert events[-2][0] == "classification"
        assert events[-1] == ("ack", sequence)

    await preprocessor.process_capture(frame(4), assistant_rendering=True)

    assert preprocessor.pending_classification is None
    release_index = events.index(("classification", None))
    assert events[release_index - 5 : release_index] == [
        ("admit", sequence) for sequence in range(5)
    ]
    assert events[release_index + 1] == ("ack", 4)


@pytest.mark.asyncio
async def test_fence_drops_all_pending_cleaned_frames_before_callback() -> None:
    monitor = ScriptedIsolationMonitor(
        [NearEndDisposition.PENDING] * 4
        + [NearEndDisposition.FENCE, NearEndDisposition.ADMIT]
    )
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY] * 6),
        isolation_monitor=monitor,
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        healthy_streak=1,
    )

    for sequence in range(5):
        await preprocessor.process_capture(frame(sequence), assistant_rendering=True)

    assert admitted == []
    assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX
    assert preprocessor.health is AecHealth.DEGRADED

    await preprocessor.process_capture(frame(5), assistant_rendering=True)

    assert [item.sequence for item in admitted] == [5]


@pytest.mark.asyncio
@pytest.mark.parametrize("resolution", ["vad-negative", "fence", "failure", "reset"])
async def test_pending_classification_releases_on_every_closed_boundary(
    resolution: str,
) -> None:
    monitor = ScriptedIsolationMonitor(
        [NearEndDisposition.PENDING, NearEndDisposition.FENCE]
    )
    aec = FakeAec([HEALTHY, HEALTHY])
    classification: list[object | None] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=monitor,
        vad=lambda cleaned: resolution != "vad-negative" or cleaned.sequence == 0,
        on_classification_changed=classification.append,
        healthy_streak=1,
    )
    await preprocessor.process_capture(frame(0), assistant_rendering=True)
    assert preprocessor.pending_classification is not None

    if resolution == "failure":
        aec.raise_on_capture = True
        await preprocessor.process_capture(frame(1), assistant_rendering=True)
    elif resolution == "reset":
        preprocessor.reset_for_device_route(1)
    else:
        await preprocessor.process_capture(frame(1), assistant_rendering=True)

    assert preprocessor.pending_classification is None
    assert classification[-1] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_pcm", [b"bad", None])
async def test_malformed_capture_closes_and_drops_pending_before_raising(
    malformed_pcm: object,
) -> None:
    monitor = ScriptedIsolationMonitor(
        [
            NearEndDisposition.PENDING,
            NearEndDisposition.ADMIT,
            NearEndDisposition.ADMIT,
        ]
    )
    admitted: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=FakeAec([HEALTHY, HEALTHY]),
        isolation_monitor=monitor,
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        healthy_streak=1,
    )
    await preprocessor.process_capture(frame(0), assistant_rendering=True)
    assert preprocessor.safety.path is AcousticSafetyPath.AEC
    assert admitted == []

    malformed = replace(frame(1), pcm16=malformed_pcm)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capture PCM"):
        await preprocessor.process_capture(malformed, assistant_rendering=True)

    assert admitted == []
    assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert (
        preprocessor.safety.demotion_reason
        is AcousticDemotionReason.INVALID_CAPTURE_PCM
    )
    assert monitor.calls[-1]["native_processor_ok"] is False

    await preprocessor.process_capture(frame(1), assistant_rendering=True)

    assert [item.sequence for item in admitted] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_metrics",
    [
        {},
        {**HEALTHY, "erle_db": math.inf},
        {**HEALTHY, "delay_estimate_available": 2.0},
        {**HEALTHY, "delay_estimate_refined": -1.0},
        {
            **HEALTHY,
            "delay_estimate_available": 0.0,
            "delay_estimate_refined": 1.0,
        },
        {**HEALTHY, "delay_ms": -1.0},
        {**HEALTHY, "delay_age_blocks": -1.0},
        {**HEALTHY, "clock_drift": 2.0},
    ],
)
async def test_invalid_metrics_are_not_native_operational_for_isolation(
    invalid_metrics: dict[str, float],
) -> None:
    monitor = ScriptedIsolationMonitor(
        [NearEndDisposition.PENDING],
        path=AcousticSafetyPath.ACOUSTIC_ISOLATION,
    )
    admitted: list[AudioFrame] = []
    vad_frames: list[AudioFrame] = []
    aec = FakeAec([invalid_metrics])
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=monitor,
        vad=lambda cleaned: vad_frames.append(cleaned) or True,
        on_admitted_frame=admitted.append,
        healthy_streak=1,
    )

    result = await preprocessor.process_capture(frame(0), assistant_rendering=True)

    assert result is None
    assert admitted == []
    assert vad_frames == []
    assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert (
        preprocessor.safety.demotion_reason
        is AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE
    )
    assert monitor.calls[-1]["native_processor_ok"] is False
    assert aec.metrics_calls == 1


@pytest.mark.asyncio
async def test_exact_length_non_bytes_native_output_fails_closed() -> None:
    monitor = ScriptedIsolationMonitor(path=AcousticSafetyPath.ACOUSTIC_ISOLATION)
    aec = FakeAec([HEALTHY])
    aec.cleaned_frames.append("x" * FRAME_BYTES)  # type: ignore[arg-type]
    vad_frames: list[AudioFrame] = []
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=monitor,
        vad=lambda cleaned: vad_frames.append(cleaned) or True,
        healthy_streak=1,
    )

    result = await preprocessor.process_capture(frame(0), assistant_rendering=True)

    assert result is None
    assert vad_frames == []
    assert preprocessor.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert monitor.calls[-1]["native_processor_ok"] is False


@pytest.mark.asyncio
async def test_render_iterable_is_consumed_once_and_fails_on_frame_151() -> None:
    monitor = ScriptedIsolationMonitor(path=AcousticSafetyPath.ACOUSTIC_ISOLATION)
    aec = FakeAec([HEALTHY])
    preprocessor = VoicePreprocessor(aec=aec, isolation_monitor=monitor)
    consumed: list[int] = []

    def render_frames() -> Iterable[AudioFrame]:
        for sequence in range(151):
            consumed.append(sequence)
            yield frame(sequence)

    result = await preprocessor.process_capture(
        frame(0),
        render_frames=render_frames(),
        assistant_rendering=True,
    )

    assert result is None
    assert consumed == list(range(151))
    assert len(aec.events) == 150
    assert (
        preprocessor.safety.demotion_reason
        is AcousticDemotionReason.RENDER_REFERENCE_OVERFLOW
    )
    assert monitor.calls[-1]["native_processor_ok"] is False
    assert tuple(monitor.calls[-1]["render_frames"]) == tuple(
        frame(i) for i in range(150)
    )


@pytest.mark.asyncio
async def test_route_reset_clears_pending_replay_and_rewarms_both_proofs() -> None:
    monitor = ScriptedIsolationMonitor([NearEndDisposition.PENDING])
    admitted: list[AudioFrame] = []
    aec = FakeAec([HEALTHY, HEALTHY])
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=monitor,
        vad=lambda _frame: True,
        on_admitted_frame=admitted.append,
        healthy_streak=1,
    )
    await preprocessor.process_capture(frame(0), assistant_rendering=True)

    preprocessor.reset_for_device_route(1)
    monitor.dispositions.append(NearEndDisposition.ADMIT)
    await preprocessor.process_capture(
        frame(0, generation=1),
        assistant_rendering=True,
    )

    assert admitted == [replace(frame(0, generation=1), pcm16=pcm16(7))]
    assert monitor.reset_generations == [1]
    assert preprocessor.health is AecHealth.HEALTHY


@pytest.mark.parametrize("failing_reset", ["aec", "monitor"])
@pytest.mark.asyncio
async def test_route_reset_publishes_closed_new_generation_before_native_reset(
    failing_reset: str,
) -> None:
    monitor = ScriptedIsolationMonitor([NearEndDisposition.PENDING])
    aec = FakeAec([HEALTHY])
    preprocessor = VoicePreprocessor(
        aec=aec,
        isolation_monitor=monitor,
        vad=lambda _frame: True,
        healthy_streak=1,
    )
    await preprocessor.process_capture(frame(0), assistant_rendering=True)
    observed_state: list[tuple[object, ...]] = []

    def snapshot() -> tuple[object, ...]:
        return (
            preprocessor.active_clock_generation,
            preprocessor.safety.path,
            preprocessor.safety.mode,
            preprocessor.safety.route_generation,
            preprocessor.health,
            len(preprocessor._pending_replay),
        )

    def fail_aec_reset() -> None:
        observed_state.append(snapshot())
        raise RuntimeError("AEC reset failed")

    def fail_monitor_reset(_clock_generation: int) -> None:
        observed_state.append(snapshot())
        raise RuntimeError("monitor reset failed")

    if failing_reset == "aec":
        aec.reset = fail_aec_reset  # type: ignore[method-assign]
    else:
        monitor.reset_for_route = fail_monitor_reset  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="reset failed"):
        preprocessor.reset_for_device_route(1)

    expected = (
        1,
        AcousticSafetyPath.WARMING,
        DuplexMode.HALF_DUPLEX,
        1,
        AecHealth.WARMING,
        0,
    )
    assert observed_state == [expected]
    assert snapshot() == expected
