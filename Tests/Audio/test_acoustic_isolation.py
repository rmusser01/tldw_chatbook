"""Deterministic tests for content-free acoustic-isolation proof."""

from __future__ import annotations

from array import array
from dataclasses import replace
import math
import random
import subprocess
import sys

import pytest

from tldw_chatbook.Audio.acoustic_isolation import (
    ISOLATION_FRAME_BYTES,
    ISOLATION_FLOOR_DB,
    ISOLATION_MAX_CORRELATION,
    ISOLATION_MAX_LAG_MS,
    ISOLATION_MAX_LEAKAGE_DB,
    ISOLATION_METRIC_SAMPLES_MAX,
    ISOLATION_NEAR_END_CONFIRM_FRAMES,
    ISOLATION_NEAR_END_RENDER_DOMINANCE_CORRELATION,
    ISOLATION_RENDER_RMS_MIN,
    ISOLATION_REQUIRED_WINDOWS,
    ISOLATION_SAMPLE_STRIDE,
    ISOLATION_WINDOW_FRAMES,
    AcousticIsolationMonitor,
    _event_render_dominant,
    _pcm16_samples,
)
from tldw_chatbook.Audio.duplex_contracts import (
    AcousticDemotionReason,
    AcousticIsolationObservation,
    AcousticSafetyPath,
    AcousticSafetySnapshot,
    AecDelayEvidence,
    AudioFrame,
    DuplexMode,
    NearEndDisposition,
)


FRAME_NS = 10_000_000
CAPTURE_BASE_NS = 4_980_000_000
SAMPLES_PER_FRAME = 480
DOWNSAMPLED_SAMPLES_PER_FRAME = SAMPLES_PER_FRAME // ISOLATION_SAMPLE_STRIDE
CALIBRATION_EVENT_FRAMES = 5
CALIBRATION_MAX_LAG_FRAMES = 50
QUIET = (20, 20, -20, -20, 20, 20, -20, -20, 0, 0)
RENDER = (4_000, -4_000, 4_000, -4_000, 4_000, -4_000, 4_000, -4_000, 4_000, -4_000)


def pcm16(downsampled: tuple[int, ...]) -> bytes:
    samples = array(
        "h",
        (
            downsampled[index % len(downsampled)]
            for index in range(SAMPLES_PER_FRAME // ISOLATION_SAMPLE_STRIDE)
            for _repeat in range(ISOLATION_SAMPLE_STRIDE)
        ),
    )
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def varied_pcm16(sequence: int, *, amplitude: int = 4_000) -> bytes:
    state = sequence + 1
    values: list[int] = []
    for _index in range(SAMPLES_PER_FRAME // (2 * ISOLATION_SAMPLE_STRIDE)):
        state = (1_103_515_245 * state + 12_345) & 0x7FFF_FFFF
        value = amplitude * (1 + state % 7) // 7
        values.extend((value, -value))
    return pcm16(tuple(values))


def low_autocorrelation_pcm16(sequence: int) -> bytes:
    residue = sequence % 127
    symbol = 1 if residue == 0 or pow(residue, 63, 127) == 1 else -1
    return pcm16(tuple(symbol * sample for sample in RENDER))


def zero_mean_noise(sequence: int, *, amplitude: int = 4_000) -> tuple[int, ...]:
    generator = random.Random(sequence)
    values: list[int] = []
    for _index in range(SAMPLES_PER_FRAME // (2 * ISOLATION_SAMPLE_STRIDE)):
        value = generator.randint(1, amplitude)
        sign = 1 if generator.getrandbits(1) else -1
        values.extend((sign * value, -sign * value))
    generator.shuffle(values)
    return tuple(values)


def ar_like_vectors(
    seed: int,
    rho: float,
    frame_count: int,
) -> list[tuple[float, ...]]:
    generator = random.Random(seed)
    value = 0.0
    samples: list[float] = []
    for _index in range(frame_count * DOWNSAMPLED_SAMPLES_PER_FRAME):
        value = rho * value + generator.uniform(-1.0, 1.0)
        samples.append(value)

    vectors = []
    for frame_index in range(frame_count):
        start = frame_index * DOWNSAMPLED_SAMPLES_PER_FRAME
        raw = samples[start : start + DOWNSAMPLED_SAMPLES_PER_FRAME]
        mean = sum(raw) / len(raw)
        vectors.append(tuple(sample - mean for sample in raw))
    return vectors


def calibration_event(
    render: list[tuple[float, ...]],
    capture: list[tuple[float, ...]],
    *,
    delay_ms: int = 450,
) -> tuple[
    tuple[tuple[int, tuple[float, ...], int, int], ...],
    dict[int, tuple[tuple[float, ...], int]],
]:
    capture_base_ns = 1_000_000_000
    captures = tuple(
        (
            CALIBRATION_MAX_LAG_FRAMES + index,
            vector,
            delay_ms,
            capture_base_ns + index * FRAME_NS,
        )
        for index, vector in enumerate(capture)
    )
    return captures, {
        sequence: (
            vector,
            capture_base_ns
            + (sequence - CALIBRATION_MAX_LAG_FRAMES) * FRAME_NS
            + delay_ms * 1_000_000,
        )
        for sequence, vector in enumerate(render)
    }


def event_rms(vectors: list[tuple[float, ...]]) -> float:
    samples = [sample for vector in vectors for sample in vector]
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def mixed_event(
    render: list[tuple[float, ...]],
    speech: list[tuple[float, ...]],
    *,
    lag_frames: int,
    echo_to_speech_ratio: float,
) -> list[tuple[float, ...]]:
    echo = [
        render[CALIBRATION_MAX_LAG_FRAMES + index - lag_frames]
        for index in range(CALIBRATION_EVENT_FRAMES)
    ]
    echo_rms = event_rms(echo)
    speech_rms = event_rms(speech)
    return [
        tuple(
            echo_to_speech_ratio * render_sample / echo_rms + speech_sample / speech_rms
            for render_sample, speech_sample in zip(
                echo[index],
                speech[index],
                strict=True,
            )
        )
        for index in range(CALIBRATION_EVENT_FRAMES)
    ]


def timing(
    sequence: int,
    *,
    delay_ms: int = 40,
    time_offset_ns: int = 0,
    occupancy_bounded: bool = True,
    timing_discontinuity: bool = False,
    clock_drift: bool = False,
    status_flags: tuple[str, ...] = (),
) -> AecDelayEvidence:
    capture_adc_ns = CAPTURE_BASE_NS + sequence * FRAME_NS + time_offset_ns
    render_dac_ns = capture_adc_ns + delay_ms * 1_000_000
    return AecDelayEvidence(
        observed_ns=capture_adc_ns,
        capture_adc_ns=capture_adc_ns,
        render_dac_ns=render_dac_ns,
        delay_ms=delay_ms,
        capture_occupancy_frames=0,
        render_occupancy_frames=0,
        occupancy_bounded=occupancy_bounded,
        timing_discontinuity=timing_discontinuity,
        clock_drift=clock_drift,
        status_flags=status_flags,
    )


def frame(
    sequence: int,
    samples: tuple[int, ...] = QUIET,
    *,
    generation: int = 0,
    reference_sequence: int | None = None,
    evidence: AecDelayEvidence | None = None,
    discontinuity: bool = False,
    time_offset_ns: int = 0,
) -> AudioFrame:
    started_ns = CAPTURE_BASE_NS + sequence * FRAME_NS + time_offset_ns
    return AudioFrame(
        sequence=sequence,
        started_ns=started_ns,
        ended_ns=started_ns + FRAME_NS,
        pcm16=pcm16(samples),
        clock_generation=generation,
        discontinuity=discontinuity,
        delay_evidence=(
            timing(sequence, time_offset_ns=time_offset_ns)
            if evidence is None
            else evidence
        ),
        render_reference_sequence=(
            sequence if reference_sequence is None else reference_sequence
        ),
    )


def render_frame(
    sequence: int,
    samples: tuple[int, ...] = RENDER,
    *,
    generation: int = 0,
    pcm: bytes | None = None,
    time_offset_ns: int = 0,
    delay_ms: int = 40,
    timing_sequence: int | None = None,
) -> AudioFrame:
    evidence = timing(
        sequence if timing_sequence is None else timing_sequence,
        delay_ms=delay_ms,
        time_offset_ns=time_offset_ns,
    )
    started_ns = evidence.render_dac_ns
    return AudioFrame(
        sequence=sequence,
        started_ns=started_ns,
        ended_ns=started_ns + FRAME_NS,
        pcm16=pcm16(samples) if pcm is None else pcm,
        clock_generation=generation,
        delay_evidence=evidence,
    )


def observe_window(
    monitor: AcousticIsolationMonitor,
    *,
    start: int,
    capture_pcm: bytes | None = None,
    generation: int = 0,
    near_end_speech: bool = False,
) -> AcousticSafetySnapshot:
    snapshot = None
    for sequence in range(start, start + 10):
        capture = frame(sequence, generation=generation)
        if capture_pcm is not None:
            capture = AudioFrame(
                sequence=capture.sequence,
                started_ns=capture.started_ns,
                ended_ns=capture.ended_ns,
                pcm16=capture_pcm,
                clock_generation=generation,
                delay_evidence=capture.delay_evidence,
                render_reference_sequence=sequence,
            )
        observation = monitor.observe(
            capture=capture,
            render_frames=(render_frame(sequence, generation=generation),),
            assistant_rendering=True,
            near_end_speech=near_end_speech and sequence == start + 9,
            native_processor_ok=True,
        )
        snapshot = observation.safety
    assert snapshot is not None
    return snapshot


def small_monitor() -> AcousticIsolationMonitor:
    return AcousticIsolationMonitor(window_frames=10, required_windows=2)


def opened_monitor() -> AcousticIsolationMonitor:
    monitor = small_monitor()
    observe_window(monitor, start=0)
    observe_window(monitor, start=10)
    return monitor


def opened_noise_monitor() -> tuple[
    AcousticIsolationMonitor, dict[int, tuple[int, ...]]
]:
    monitor = small_monitor()
    rendered: dict[int, tuple[int, ...]] = {}
    for sequence in range(20):
        rendered[sequence] = zero_mean_noise(sequence)
        monitor.observe(
            capture=frame(sequence),
            render_frames=(render_frame(sequence, rendered[sequence]),),
            assistant_rendering=True,
            near_end_speech=False,
            native_processor_ok=True,
        )
    return monitor, rendered


def test_production_policy_constants_are_fixed() -> None:
    assert (
        ISOLATION_FRAME_BYTES,
        ISOLATION_WINDOW_FRAMES,
        ISOLATION_REQUIRED_WINDOWS,
        ISOLATION_SAMPLE_STRIDE,
        ISOLATION_MAX_LAG_MS,
        ISOLATION_RENDER_RMS_MIN,
        ISOLATION_MAX_CORRELATION,
        ISOLATION_MAX_LEAKAGE_DB,
        ISOLATION_FLOOR_DB,
        ISOLATION_METRIC_SAMPLES_MAX,
        ISOLATION_NEAR_END_CONFIRM_FRAMES,
        ISOLATION_NEAR_END_RENDER_DOMINANCE_CORRELATION,
    ) == (960, 100, 5, 8, 500, 512.0, 0.12, -30.0, -120.0, 3_600, 5, 0.85)


def test_observe_always_returns_content_free_observation_wrapper() -> None:
    observation = AcousticIsolationMonitor().observe(
        capture=frame(0),
        render_frames=(),
        assistant_rendering=False,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert isinstance(observation, AcousticIsolationObservation)
    assert observation.near_end_disposition is None
    assert observation.safety.path is AcousticSafetyPath.WARMING


def test_transport_scheduled_reference_timing_can_qualify_isolation() -> None:
    monitor = AcousticIsolationMonitor(window_frames=10, required_windows=1)
    first_capture = frame(0)
    first_render = render_frame(0)

    assert first_capture.delay_evidence is not None
    assert first_capture.delay_evidence.capture_adc_ns == 4_980_000_000
    assert first_render.delay_evidence is not None
    assert first_render.delay_evidence.render_dac_ns == 5_020_000_000
    assert first_render.started_ns == 5_020_000_000

    for sequence in range(10):
        observation = monitor.observe(
            capture=frame(sequence),
            render_frames=(render_frame(sequence),),
            assistant_rendering=True,
            near_end_speech=False,
            native_processor_ok=True,
        )

    assert observation.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
    assert observation.safety.demotion_reason is None


@pytest.mark.parametrize(
    ("offset_error_ns", "expected_reason"),
    [
        (500_000, AcousticDemotionReason.AMBIGUOUS_CORRELATION),
        (500_001, AcousticDemotionReason.MISSING_RENDER_REFERENCE),
    ],
)
def test_scheduled_reference_allows_only_rounded_delay_tolerance(
    offset_error_ns: int,
    expected_reason: AcousticDemotionReason,
) -> None:
    monitor = AcousticIsolationMonitor(window_frames=1, required_windows=1)
    capture = frame(0)
    render = render_frame(0)
    assert render.delay_evidence is not None
    evidence = replace(
        render.delay_evidence,
        render_dac_ns=render.delay_evidence.render_dac_ns + offset_error_ns,
    )
    render = replace(
        render,
        started_ns=evidence.render_dac_ns,
        ended_ns=evidence.render_dac_ns + FRAME_NS,
        delay_evidence=evidence,
    )

    observation = monitor.observe(
        capture=capture,
        render_frames=(render,),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert observation.safety.demotion_reason is expected_reason


def test_frame_local_dc_cannot_hide_exactly_copied_ac_leakage() -> None:
    monitor = AcousticIsolationMonitor(window_frames=10, required_windows=1)
    ac = tuple(1_000 if index % 2 else -1_000 for index in range(60))
    observation = None

    for sequence in range(10):
        dc = 10_000 if sequence % 2 else -10_000
        render = tuple(dc + value for value in ac)
        observation = monitor.observe(
            capture=frame(sequence, ac),
            render_frames=(render_frame(sequence, render),),
            assistant_rendering=True,
            near_end_speech=False,
            native_processor_ok=True,
        )

    assert observation is not None
    assert observation.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert observation.safety.demotion_reason == "correlated-render"


@pytest.mark.parametrize("rho", [0.8, 0.9, 0.95])
def test_seed_635_independent_colored_speech_is_admitted_through_public_api(
    rho: float,
) -> None:
    rendered = [
        tuple(round(sample * 1_000) for sample in vector)
        for vector in ar_like_vectors(635, rho, 14)
    ]
    captured = [
        tuple(round(sample * 1_000) for sample in vector)
        for vector in ar_like_vectors(10_635, rho, CALIBRATION_EVENT_FRAMES)
    ]
    monitor = AcousticIsolationMonitor(window_frames=9, required_windows=1)
    for sequence in range(9):
        opened = monitor.observe(
            capture=frame(sequence, (0,) * DOWNSAMPLED_SAMPLES_PER_FRAME),
            render_frames=(render_frame(sequence, rendered[sequence]),),
            assistant_rendering=True,
            near_end_speech=False,
            native_processor_ok=True,
        )
    assert opened.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION

    dispositions = []
    for sequence in range(9, 14):
        observation = monitor.observe(
            capture=frame(sequence, captured[sequence - 9]),
            render_frames=(render_frame(sequence, rendered[sequence]),),
            assistant_rendering=True,
            near_end_speech=True,
            native_processor_ok=True,
        )
        dispositions.append(observation.near_end_disposition)

    assert dispositions == [NearEndDisposition.PENDING] * 4 + [NearEndDisposition.ADMIT]
    assert observation.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_calibrated_independent_corpus_has_zero_false_fences() -> None:
    false_fences = 0
    total = 0
    for rho in (0.0, 0.8, 0.9, 0.95):
        for event in range(5_000):
            seed = 1_900_000 + event * 29 + int(rho * 100)
            render = ar_like_vectors(
                seed,
                rho,
                CALIBRATION_MAX_LAG_FRAMES + CALIBRATION_EVENT_FRAMES,
            )
            capture = ar_like_vectors(
                seed + 90_000_049,
                rho,
                CALIBRATION_EVENT_FRAMES,
            )
            captures, render_by_sequence = calibration_event(render, capture)
            result = _event_render_dominant(captures, render_by_sequence)
            assert result is not None
            false_fences += result
            total += 1

    assert total == 20_000
    assert false_fences == 0


def test_calibrated_pure_echo_corpus_fences_every_lag_sign_and_colour() -> None:
    missed = 0
    total = 0
    for rho in (0.0, 0.8, 0.9, 0.95):
        for lag_frames in range(CALIBRATION_MAX_LAG_FRAMES + 1):
            render = ar_like_vectors(
                900_000 + lag_frames + int(rho * 100),
                rho,
                CALIBRATION_MAX_LAG_FRAMES + CALIBRATION_EVENT_FRAMES,
            )
            for sign in (-1.0, 1.0):
                capture = [
                    tuple(
                        sign * sample
                        for sample in render[
                            CALIBRATION_MAX_LAG_FRAMES + index - lag_frames
                        ]
                    )
                    for index in range(CALIBRATION_EVENT_FRAMES)
                ]
                captures, render_by_sequence = calibration_event(
                    render,
                    capture,
                    delay_ms=lag_frames * 10,
                )
                missed += not bool(_event_render_dominant(captures, render_by_sequence))
                total += 1

    assert total == 408
    assert missed == 0


def test_calibrated_mixtures_separate_render_and_near_end_dominance() -> None:
    render_dominant_fences = 0
    equal_energy_admissions = 0
    total = 0
    for rho in (0.0, 0.8, 0.9, 0.95):
        for event in range(5_000):
            seed = 2_600_000 + event * 37 + int(rho * 100)
            render = ar_like_vectors(
                seed,
                rho,
                CALIBRATION_MAX_LAG_FRAMES + CALIBRATION_EVENT_FRAMES,
            )
            speech = ar_like_vectors(
                seed + 130_000_081,
                rho,
                CALIBRATION_EVENT_FRAMES,
            )
            lag_frames = event % (CALIBRATION_MAX_LAG_FRAMES + 1)
            render_dominant = mixed_event(
                render,
                speech,
                lag_frames=lag_frames,
                echo_to_speech_ratio=2.0,
            )
            equal_energy = mixed_event(
                render,
                speech,
                lag_frames=lag_frames,
                echo_to_speech_ratio=1.0,
            )
            captures, render_by_sequence = calibration_event(
                render,
                render_dominant,
                delay_ms=lag_frames * 10,
            )
            render_dominant_result = _event_render_dominant(
                captures,
                render_by_sequence,
            )
            captures, render_by_sequence = calibration_event(
                render,
                equal_energy,
                delay_ms=lag_frames * 10,
            )
            equal_energy_result = _event_render_dominant(
                captures,
                render_by_sequence,
            )
            assert render_dominant_result is not None
            assert equal_energy_result is not None
            render_dominant_fences += render_dominant_result
            equal_energy_admissions += not equal_energy_result
            total += 1

    assert total == 20_000
    assert render_dominant_fences >= 19_800
    assert equal_energy_admissions >= 19_800


def test_event_render_dominance_fails_closed_for_ambiguous_vectors() -> None:
    render = ar_like_vectors(
        42,
        0.9,
        CALIBRATION_MAX_LAG_FRAMES + CALIBRATION_EVENT_FRAMES,
    )
    zero_capture = [
        (0.0,) * DOWNSAMPLED_SAMPLES_PER_FRAME
        for _index in range(CALIBRATION_EVENT_FRAMES)
    ]
    captures, render_by_sequence = calibration_event(render, zero_capture)
    assert _event_render_dominant(captures, render_by_sequence) is None
    assert _event_render_dominant(captures, {}) is None

    non_finite = list(zero_capture)
    non_finite[0] = (math.inf,) + non_finite[0][1:]
    captures, render_by_sequence = calibration_event(render, non_finite)
    assert _event_render_dominant(captures, render_by_sequence) is None


def test_independent_near_end_event_is_admitted_after_exactly_five_frames() -> None:
    monitor, rendered = opened_noise_monitor()
    dispositions: list[NearEndDisposition | None] = []

    for sequence in range(20, 26):
        rendered[sequence] = zero_mean_noise(sequence)
        observation = monitor.observe(
            capture=frame(sequence, zero_mean_noise(sequence + 10_000)),
            render_frames=(render_frame(sequence, rendered[sequence]),),
            assistant_rendering=True,
            near_end_speech=True,
            native_processor_ok=True,
        )
        dispositions.append(observation.near_end_disposition)
        assert observation.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION

    assert dispositions == [
        NearEndDisposition.PENDING,
        NearEndDisposition.PENDING,
        NearEndDisposition.PENDING,
        NearEndDisposition.PENDING,
        NearEndDisposition.ADMIT,
        NearEndDisposition.ADMIT,
    ]


@pytest.mark.parametrize("delay_frames", [4, 7])
def test_stable_same_lag_echo_fences_after_exactly_five_frames(
    delay_frames: int,
) -> None:
    monitor, rendered = opened_noise_monitor()
    dispositions: list[NearEndDisposition | None] = []
    observation = None

    for sequence in range(20, 25):
        rendered[sequence] = zero_mean_noise(sequence)
        capture = rendered[sequence - delay_frames]
        observation = monitor.observe(
            capture=frame(sequence, capture),
            render_frames=(render_frame(sequence, rendered[sequence]),),
            assistant_rendering=True,
            near_end_speech=True,
            native_processor_ok=True,
        )
        dispositions.append(observation.near_end_disposition)

    assert observation is not None
    assert dispositions == [NearEndDisposition.PENDING] * 4 + [NearEndDisposition.FENCE]
    assert observation.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert observation.safety.demotion_reason == "correlated-render"


def test_future_scheduled_lag_zero_render_is_not_a_causal_echo_candidate() -> None:
    monitor, rendered = opened_noise_monitor()
    dispositions: list[NearEndDisposition | None] = []

    for sequence in range(20, 25):
        rendered[sequence] = zero_mean_noise(sequence)
        observation = monitor.observe(
            capture=frame(sequence, rendered[sequence]),
            render_frames=(render_frame(sequence, rendered[sequence]),),
            assistant_rendering=True,
            near_end_speech=True,
            native_processor_ok=True,
        )
        dispositions.append(observation.near_end_disposition)

    assert dispositions == [NearEndDisposition.PENDING] * 4 + [NearEndDisposition.ADMIT]
    assert observation.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_render_at_capture_adc_time_is_a_causal_echo_candidate() -> None:
    monitor, rendered = opened_noise_monitor()
    dispositions: list[NearEndDisposition | None] = []

    for sequence in range(20, 25):
        rendered[sequence] = zero_mean_noise(sequence)
        observation = monitor.observe(
            capture=frame(sequence, rendered[sequence - 4]),
            render_frames=(render_frame(sequence, rendered[sequence]),),
            assistant_rendering=True,
            near_end_speech=True,
            native_processor_ok=True,
        )
        dispositions.append(observation.near_end_disposition)

    assert dispositions == [NearEndDisposition.PENDING] * 4 + [NearEndDisposition.FENCE]
    assert observation.safety.path is AcousticSafetyPath.HALF_DUPLEX


def test_high_correlation_below_leakage_floor_is_admitted() -> None:
    monitor, rendered = opened_noise_monitor()
    dispositions: list[NearEndDisposition | None] = []

    for sequence in range(20, 25):
        rendered[sequence] = zero_mean_noise(sequence)
        capture = tuple(round(sample * 0.03) for sample in rendered[sequence - 4])
        observation = monitor.observe(
            capture=frame(sequence, capture),
            render_frames=(render_frame(sequence, rendered[sequence]),),
            assistant_rendering=True,
            near_end_speech=True,
            native_processor_ok=True,
        )
        dispositions.append(observation.near_end_disposition)

    assert dispositions == [NearEndDisposition.PENDING] * 4 + [NearEndDisposition.ADMIT]
    assert observation.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_one_opposite_frame_dot_sign_prevents_render_dominance_fence() -> None:
    monitor, rendered = opened_noise_monitor()
    dispositions: list[NearEndDisposition | None] = []

    for sequence in range(20, 25):
        rendered[sequence] = zero_mean_noise(sequence)
        gain = -0.01 if sequence == 24 else 1.0
        capture = tuple(round(sample * gain) for sample in rendered[sequence - 4])
        observation = monitor.observe(
            capture=frame(sequence, capture),
            render_frames=(render_frame(sequence, rendered[sequence]),),
            assistant_rendering=True,
            near_end_speech=True,
            native_processor_ok=True,
        )
        dispositions.append(observation.near_end_disposition)

    assert dispositions == [NearEndDisposition.PENDING] * 4 + [NearEndDisposition.ADMIT]
    assert observation.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_vad_negative_break_restarts_pending_event_without_revoking_proof() -> None:
    monitor, rendered = opened_noise_monitor()
    first_event = []
    for sequence in range(20, 22):
        rendered[sequence] = zero_mean_noise(sequence)
        first_event.append(
            monitor.observe(
                capture=frame(sequence, zero_mean_noise(sequence + 10_000)),
                render_frames=(render_frame(sequence, rendered[sequence]),),
                assistant_rendering=True,
                near_end_speech=True,
                native_processor_ok=True,
            )
        )
    rendered[22] = zero_mean_noise(22)
    ended = monitor.observe(
        capture=frame(22),
        render_frames=(render_frame(22, rendered[22]),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )
    rendered[23] = zero_mean_noise(23)
    restarted = monitor.observe(
        capture=frame(23, zero_mean_noise(10_023)),
        render_frames=(render_frame(23, rendered[23]),),
        assistant_rendering=True,
        near_end_speech=True,
        native_processor_ok=True,
    )

    assert [item.near_end_disposition for item in first_event] == [
        NearEndDisposition.PENDING,
        NearEndDisposition.PENDING,
    ]
    assert ended.near_end_disposition is None
    assert ended.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
    assert restarted.near_end_disposition is NearEndDisposition.PENDING
    assert restarted.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_vad_negative_frame_ends_admitted_event_and_restarts_pending() -> None:
    monitor, rendered = opened_noise_monitor()
    admitted = None
    for sequence in range(20, 25):
        rendered[sequence] = zero_mean_noise(sequence)
        admitted = monitor.observe(
            capture=frame(sequence, zero_mean_noise(sequence + 10_000)),
            render_frames=(render_frame(sequence, rendered[sequence]),),
            assistant_rendering=True,
            near_end_speech=True,
            native_processor_ok=True,
        )
    assert admitted is not None
    assert admitted.near_end_disposition is NearEndDisposition.ADMIT

    rendered[25] = zero_mean_noise(25)
    ended = monitor.observe(
        capture=frame(25),
        render_frames=(render_frame(25, rendered[25]),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )
    rendered[26] = zero_mean_noise(26)
    restarted = monitor.observe(
        capture=frame(26, zero_mean_noise(10_026)),
        render_frames=(render_frame(26, rendered[26]),),
        assistant_rendering=True,
        near_end_speech=True,
        native_processor_ok=True,
    )

    assert ended.near_end_disposition is None
    assert ended.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
    assert restarted.near_end_disposition is NearEndDisposition.PENDING
    assert restarted.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_audible_render_and_uncorrelated_capture_opens_on_fifth_window() -> None:
    monitor = AcousticIsolationMonitor()

    for window_index in range(ISOLATION_REQUIRED_WINDOWS):
        snapshot = None
        start = window_index * ISOLATION_WINDOW_FRAMES
        for sequence in range(start, start + ISOLATION_WINDOW_FRAMES):
            snapshot = monitor.observe(
                capture=frame(sequence),
                render_frames=(render_frame(sequence),),
                assistant_rendering=True,
                near_end_speech=False,
                native_processor_ok=True,
            ).safety
        assert snapshot is not None
        expected_mode = (
            DuplexMode.FULL_DUPLEX
            if window_index == ISOLATION_REQUIRED_WINDOWS - 1
            else DuplexMode.HALF_DUPLEX
        )
        assert snapshot.mode is expected_mode

    assert snapshot.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
    assert snapshot.admission_open is True


@pytest.mark.parametrize("delay_frames", [4, 7])
def test_copied_or_delayed_render_never_opens(delay_frames: int) -> None:
    monitor = small_monitor()
    rendered: dict[int, bytes] = {}
    snapshot = None

    for sequence in range(30):
        rendered[sequence] = varied_pcm16(sequence)
        source = rendered.get(sequence - delay_frames, varied_pcm16(sequence + 1_000))
        capture = frame(sequence)
        capture = AudioFrame(
            sequence=capture.sequence,
            started_ns=capture.started_ns,
            ended_ns=capture.ended_ns,
            pcm16=source,
            clock_generation=0,
            delay_evidence=capture.delay_evidence,
            render_reference_sequence=sequence,
        )
        snapshot = monitor.observe(
            capture=capture,
            render_frames=(render_frame(sequence, pcm=rendered[sequence]),),
            assistant_rendering=True,
            near_end_speech=False,
            native_processor_ok=True,
        ).safety

    assert snapshot is not None
    assert snapshot.mode is DuplexMode.HALF_DUPLEX
    assert snapshot.admission_open is False


def test_lag_scan_does_not_round_past_reported_latency_bound() -> None:
    monitor = AcousticIsolationMonitor(window_frames=127, required_windows=1)
    rendered: dict[int, bytes] = {}
    snapshot = None

    for sequence in range(137):
        rendered[sequence] = low_autocorrelation_pcm16(sequence)
        capture_pcm = pcm16(QUIET) if sequence < 10 else rendered[sequence - 10]
        capture = replace(
            frame(sequence, evidence=timing(sequence, delay_ms=41)),
            pcm16=capture_pcm,
        )
        snapshot = monitor.observe(
            capture=capture,
            render_frames=(
                render_frame(sequence, pcm=rendered[sequence], delay_ms=41),
            ),
            assistant_rendering=True,
            near_end_speech=sequence < 10,
            native_processor_ok=True,
        ).safety

    assert snapshot is not None
    assert snapshot.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_route_generation_change_clears_all_accumulated_evidence() -> None:
    monitor = small_monitor()
    before_reset = observe_window(monitor, start=0)
    assert before_reset.path is AcousticSafetyPath.WARMING
    assert len(monitor.correlation_samples) == 1

    monitor.reset_for_route(1)

    assert monitor.correlation_samples == ()
    assert monitor.leakage_db_samples == ()
    after_one = observe_window(monitor, start=10, generation=1)
    after_two = observe_window(monitor, start=20, generation=1)
    assert after_one.path is AcousticSafetyPath.WARMING
    assert after_two.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
    assert after_two.route_generation == 1


@pytest.mark.parametrize("clock_generation", [True, 1.5])
def test_route_reset_requires_exact_nonnegative_integer_generation(
    clock_generation: object,
) -> None:
    with pytest.raises(TypeError):
        AcousticIsolationMonitor().reset_for_route(  # type: ignore[arg-type]
            clock_generation
        )


def test_near_end_vad_during_warmup_restarts_clean_window_count() -> None:
    monitor = small_monitor()

    observe_window(monitor, start=0)
    contaminated = observe_window(monitor, start=10, near_end_speech=True)
    after_one_clean = observe_window(monitor, start=20)
    opened = observe_window(monitor, start=30)

    assert contaminated.path is AcousticSafetyPath.WARMING
    assert after_one_clean.path is AcousticSafetyPath.WARMING
    assert opened.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_near_end_vad_after_opening_is_pending_without_contaminating_stats() -> None:
    monitor = small_monitor()
    observe_window(monitor, start=0)
    opened = observe_window(monitor, start=10)
    before = (monitor.correlation_samples, monitor.leakage_db_samples)

    speech = observe_window(monitor, start=20, near_end_speech=True)

    assert opened.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
    assert speech.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
    assert speech.admission_open is True
    assert (monitor.correlation_samples, monitor.leakage_db_samples) == before


def test_render_pause_preserves_proof_and_starts_clean_sequence_boundary() -> None:
    monitor = small_monitor()
    observe_window(monitor, start=0)
    observe_window(monitor, start=10)

    paused = monitor.observe(
        capture=frame(20),
        render_frames=(render_frame(20, pcm=bytes(ISOLATION_FRAME_BYTES)),),
        assistant_rendering=False,
        near_end_speech=True,
        native_processor_ok=True,
    )
    resumed = observe_window(monitor, start=21)

    assert paused.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
    assert resumed.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_render_pause_preserves_completed_warmup_windows_only() -> None:
    monitor = small_monitor()
    first = observe_window(monitor, start=0)

    paused = monitor.observe(
        capture=frame(10),
        render_frames=(render_frame(10, pcm=bytes(ISOLATION_FRAME_BYTES)),),
        assistant_rendering=False,
        near_end_speech=False,
        native_processor_ok=True,
    )
    opened = observe_window(monitor, start=11)

    assert first.path is AcousticSafetyPath.WARMING
    assert paused.safety.path is AcousticSafetyPath.WARMING
    assert opened.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_idle_silence_preserves_render_timeline_and_real_safety_fault() -> None:
    monitor = AcousticIsolationMonitor(window_frames=10, required_windows=1)
    faulted = monitor.observe(
        capture=frame(0),
        render_frames=(),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    idle = monitor.observe(
        capture=frame(1),
        render_frames=(render_frame(1, pcm=bytes(ISOLATION_FRAME_BYTES)),),
        assistant_rendering=False,
        near_end_speech=False,
        native_processor_ok=True,
    )
    resumed = monitor.observe(
        capture=frame(2),
        render_frames=(render_frame(2),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert (
        faulted.safety.demotion_reason
        is AcousticDemotionReason.MISSING_RENDER_REFERENCE
    )
    assert (
        idle.safety.demotion_reason is AcousticDemotionReason.MISSING_RENDER_REFERENCE
    )
    assert (
        resumed.safety.demotion_reason
        is AcousticDemotionReason.MISSING_RENDER_REFERENCE
    )


def test_brief_render_pause_retains_plausible_delayed_echo_for_fencing() -> None:
    monitor, rendered = opened_noise_monitor()
    monitor.observe(
        capture=frame(20),
        render_frames=(render_frame(20, pcm=bytes(ISOLATION_FRAME_BYTES)),),
        assistant_rendering=False,
        near_end_speech=False,
        native_processor_ok=True,
    )

    dispositions: list[NearEndDisposition | None] = []
    for sequence in range(21, 26):
        reference_sequence = sequence
        rendered[reference_sequence] = zero_mean_noise(reference_sequence)
        observation = monitor.observe(
            capture=frame(
                sequence,
                rendered[reference_sequence - 10],
                reference_sequence=reference_sequence,
                evidence=timing(sequence, delay_ms=50),
            ),
            render_frames=(
                render_frame(
                    reference_sequence,
                    rendered[reference_sequence],
                    delay_ms=50,
                    timing_sequence=sequence,
                ),
            ),
            assistant_rendering=True,
            near_end_speech=True,
            native_processor_ok=True,
        )
        dispositions.append(observation.near_end_disposition)

    assert dispositions == [NearEndDisposition.PENDING] * 4 + [NearEndDisposition.FENCE]
    assert observation.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert (
        observation.safety.demotion_reason is AcousticDemotionReason.CORRELATED_RENDER
    )


def test_long_render_pause_excludes_stale_delayed_echo_from_current_event() -> None:
    monitor, rendered = opened_noise_monitor()
    monitor.observe(
        capture=frame(20),
        render_frames=(render_frame(20, pcm=bytes(ISOLATION_FRAME_BYTES)),),
        assistant_rendering=False,
        near_end_speech=False,
        native_processor_ok=True,
    )
    rendered[20] = zero_mean_noise(20)

    time_offset_ns = 1_000_000_000
    for sequence in range(21, 26):
        reference_sequence = sequence
        rendered[reference_sequence] = zero_mean_noise(reference_sequence)
        monitor.observe(
            capture=frame(
                sequence,
                reference_sequence=reference_sequence,
                evidence=timing(
                    sequence,
                    delay_ms=50,
                    time_offset_ns=time_offset_ns,
                ),
                time_offset_ns=time_offset_ns,
            ),
            render_frames=(
                render_frame(
                    reference_sequence,
                    rendered[reference_sequence],
                    time_offset_ns=time_offset_ns,
                    delay_ms=50,
                    timing_sequence=sequence,
                ),
            ),
            assistant_rendering=True,
            near_end_speech=False,
            native_processor_ok=True,
        )

    dispositions: list[NearEndDisposition | None] = []
    for sequence in range(26, 31):
        reference_sequence = sequence
        rendered[reference_sequence] = zero_mean_noise(reference_sequence)
        observation = monitor.observe(
            capture=frame(
                sequence,
                rendered[reference_sequence - 10],
                reference_sequence=reference_sequence,
                evidence=timing(
                    sequence,
                    delay_ms=50,
                    time_offset_ns=time_offset_ns,
                ),
                time_offset_ns=time_offset_ns,
            ),
            render_frames=(
                render_frame(
                    reference_sequence,
                    rendered[reference_sequence],
                    time_offset_ns=time_offset_ns,
                    delay_ms=50,
                    timing_sequence=sequence,
                ),
            ),
            assistant_rendering=True,
            near_end_speech=True,
            native_processor_ok=True,
        )
        dispositions.append(observation.near_end_disposition)

    assert dispositions == [NearEndDisposition.PENDING] * 4 + [NearEndDisposition.ADMIT]
    assert observation.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_capture_overflow_during_render_pause_demotes_existing_proof() -> None:
    monitor = small_monitor()
    observe_window(monitor, start=0)
    observe_window(monitor, start=10)

    snapshot = monitor.observe(
        capture=frame(30, evidence=timing(30, status_flags=("input-overflow",))),
        render_frames=(),
        assistant_rendering=False,
        near_end_speech=True,
        native_processor_ok=True,
    )

    assert snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert snapshot.safety.admission_open is False


def test_one_correlated_window_after_opening_demotes_immediately() -> None:
    monitor = small_monitor()
    observe_window(monitor, start=0)
    observe_window(monitor, start=10)

    demoted = observe_window(monitor, start=20, capture_pcm=pcm16(RENDER))

    assert demoted.path is AcousticSafetyPath.HALF_DUPLEX
    assert demoted.mode is DuplexMode.HALF_DUPLEX
    assert demoted.admission_open is False
    assert demoted.demotion_reason == "correlated-render"


def test_one_saturated_frame_after_opening_demotes_immediately() -> None:
    monitor = small_monitor()
    observe_window(monitor, start=0)
    observe_window(monitor, start=10)
    saturated = list(QUIET)
    saturated[0] = 32_767

    snapshot = monitor.observe(
        capture=frame(20, tuple(saturated)),
        render_frames=(render_frame(20),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert snapshot.safety.admission_open is False
    assert snapshot.safety.demotion_reason == "saturation"


def test_saturation_between_downsampled_points_is_still_unsafe() -> None:
    monitor = AcousticIsolationMonitor(window_frames=1, required_windows=1)
    raw = array(
        "h",
        [QUIET[index % len(QUIET)] for index in range(SAMPLES_PER_FRAME)],
    )
    raw[1] = 32_767
    if sys.byteorder != "little":
        raw.byteswap()
    capture = frame(0)
    capture = AudioFrame(
        sequence=capture.sequence,
        started_ns=capture.started_ns,
        ended_ns=capture.ended_ns,
        pcm16=raw.tobytes(),
        clock_generation=0,
        delay_evidence=capture.delay_evidence,
        render_reference_sequence=0,
    )

    snapshot = monitor.observe(
        capture=capture,
        render_frames=(render_frame(0),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert snapshot.safety.demotion_reason == "saturation"


def test_correlated_post_aec_vad_frame_fences_isolation_admission() -> None:
    monitor = small_monitor()
    observe_window(monitor, start=0)
    observe_window(monitor, start=10)

    fenced = None
    for sequence in range(20, 25):
        fenced = monitor.observe(
            capture=frame(sequence, RENDER),
            render_frames=(render_frame(sequence),),
            assistant_rendering=True,
            near_end_speech=True,
            native_processor_ok=True,
        )

    assert fenced is not None
    assert fenced.near_end_disposition is NearEndDisposition.FENCE
    assert fenced.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert fenced.safety.admission_open is False
    assert fenced.safety.demotion_reason == "correlated-render"


def test_missing_render_reference_cannot_return_isolation_proof() -> None:
    monitor = AcousticIsolationMonitor(window_frames=1, required_windows=1)

    snapshot = monitor.observe(
        capture=frame(0),
        render_frames=(),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert snapshot.safety.admission_open is False


def test_missing_timing_or_reference_sequence_cannot_return_isolation_proof() -> None:
    monitor = AcousticIsolationMonitor(window_frames=1, required_windows=1)
    missing_timing = AudioFrame(
        sequence=0,
        started_ns=1_000_000_000,
        ended_ns=1_000_000_000 + FRAME_NS,
        pcm16=pcm16(QUIET),
        render_reference_sequence=0,
    )
    timing_snapshot = monitor.observe(
        capture=missing_timing,
        render_frames=(render_frame(0),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    monitor.reset_for_route(0)
    missing_reference = AudioFrame(
        sequence=0,
        started_ns=1_000_000_000,
        ended_ns=1_000_000_000 + FRAME_NS,
        pcm16=pcm16(QUIET),
        delay_evidence=timing(0),
    )
    reference_snapshot = monitor.observe(
        capture=missing_reference,
        render_frames=(render_frame(0),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert timing_snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert reference_snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX


def test_zero_variance_render_is_ambiguous_and_fails_closed() -> None:
    monitor = AcousticIsolationMonitor(window_frames=1, required_windows=1)

    snapshot = monitor.observe(
        capture=frame(0),
        render_frames=(render_frame(0, (4_000,) * 10),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert snapshot.safety.demotion_reason == "ambiguous-correlation"


@pytest.mark.parametrize(
    "bad_timing",
    [
        timing(0, occupancy_bounded=False),
        timing(0, timing_discontinuity=True),
        timing(0, clock_drift=True),
        timing(0, status_flags=("input-overflow",)),
    ],
)
def test_unbounded_or_discontinuous_timing_cannot_return_isolation_proof(
    bad_timing: AecDelayEvidence,
) -> None:
    monitor = AcousticIsolationMonitor(window_frames=1, required_windows=1)

    snapshot = monitor.observe(
        capture=frame(0, evidence=bad_timing),
        render_frames=(render_frame(0),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert snapshot.safety.admission_open is False


def test_audio_frame_discontinuity_cannot_return_isolation_proof() -> None:
    monitor = AcousticIsolationMonitor(window_frames=1, required_windows=1)

    snapshot = monitor.observe(
        capture=frame(0, discontinuity=True),
        render_frames=(render_frame(0),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert snapshot.safety.admission_open is False


def test_bounded_timestamp_jitter_keeps_sequence_aligned_evidence_eligible() -> None:
    monitor = AcousticIsolationMonitor(window_frames=5, required_windows=1)
    monitor.observe(
        capture=frame(0),
        render_frames=(render_frame(0),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )
    jitter_ns = 1_000
    capture = frame(1)
    capture = AudioFrame(
        sequence=1,
        started_ns=capture.started_ns + jitter_ns,
        ended_ns=capture.ended_ns + jitter_ns,
        pcm16=capture.pcm16,
        delay_evidence=capture.delay_evidence,
        render_reference_sequence=1,
    )
    render = render_frame(1)
    render = AudioFrame(
        sequence=1,
        started_ns=render.started_ns + jitter_ns,
        ended_ns=render.ended_ns + jitter_ns,
        pcm16=render.pcm16,
        delay_evidence=render.delay_evidence,
    )

    snapshot = monitor.observe(
        capture=capture,
        render_frames=(render,),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )
    for sequence in range(2, 5):
        snapshot = monitor.observe(
            capture=frame(sequence),
            render_frames=(render_frame(sequence),),
            assistant_rendering=True,
            near_end_speech=False,
            native_processor_ok=True,
        )

    assert snapshot.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_capture_or_render_sequence_gap_cannot_return_isolation_proof() -> None:
    capture_gap = AcousticIsolationMonitor(window_frames=2, required_windows=1)
    capture_gap.observe(
        capture=frame(0),
        render_frames=(render_frame(0),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )
    capture_snapshot = capture_gap.observe(
        capture=frame(2),
        render_frames=(render_frame(2),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    render_gap = AcousticIsolationMonitor(window_frames=2, required_windows=1)
    render_gap.observe(
        capture=frame(0),
        render_frames=(render_frame(0),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )
    render_snapshot = render_gap.observe(
        capture=frame(1),
        render_frames=(render_frame(2),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert capture_snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert render_snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX


def test_route_mismatch_or_processor_failure_cannot_return_isolation_proof() -> None:
    route_mismatch = AcousticIsolationMonitor(window_frames=1, required_windows=1)
    mismatch = route_mismatch.observe(
        capture=frame(0, generation=1),
        render_frames=(render_frame(0, generation=0),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )
    processor_failure = AcousticIsolationMonitor(window_frames=1, required_windows=1)
    failed = processor_failure.observe(
        capture=frame(0),
        render_frames=(render_frame(0),),
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=False,
    )

    assert mismatch.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert failed.safety.path is AcousticSafetyPath.HALF_DUPLEX


@pytest.mark.parametrize(
    ("target", "size"),
    [("capture", 958), ("capture", 962), ("render", 958), ("render", 962)],
)
def test_noncanonical_pcm_size_fails_closed(target: str, size: int) -> None:
    monitor = opened_monitor()
    capture = frame(20)
    render = render_frame(20)
    if target == "capture":
        capture = replace(capture, pcm16=bytes(size))
    else:
        render = replace(render, pcm16=bytes(size))

    observation = monitor.observe(
        capture=capture,
        render_frames=(render,),
        assistant_rendering=True,
        near_end_speech=True,
        native_processor_ok=True,
    )

    assert observation.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert observation.near_end_disposition is NearEndDisposition.FENCE


@pytest.mark.parametrize("target", ["capture", "render"])
def test_non_bytes_like_pcm_fails_closed_before_decoding(target: str) -> None:
    monitor = opened_monitor()
    capture = frame(20)
    render = render_frame(20)
    if target == "capture":
        capture = replace(capture, pcm16=object())  # type: ignore[arg-type]
    else:
        render = replace(render, pcm16=object())  # type: ignore[arg-type]

    observation = monitor.observe(
        capture=capture,
        render_frames=(render,),
        assistant_rendering=True,
        near_end_speech=True,
        native_processor_ok=True,
    )

    assert observation.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert observation.near_end_disposition is NearEndDisposition.FENCE


def test_shaped_memoryview_uses_byte_count_for_pcm_bound() -> None:
    shaped = memoryview(bytes(ISOLATION_FRAME_BYTES * 2)).cast(
        "B",
        shape=[ISOLATION_FRAME_BYTES, 2],
    )
    assert len(shaped) == ISOLATION_FRAME_BYTES
    assert shaped.nbytes == ISOLATION_FRAME_BYTES * 2
    capture = replace(frame(20), pcm16=shaped)  # type: ignore[arg-type]

    observation = opened_monitor().observe(
        capture=capture,
        render_frames=(render_frame(20),),
        assistant_rendering=True,
        near_end_speech=True,
        native_processor_ok=True,
    )

    assert observation.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert (
        observation.safety.demotion_reason is AcousticDemotionReason.INVALID_CAPTURE_PCM
    )
    assert observation.near_end_disposition is NearEndDisposition.FENCE


def test_exact_sized_contiguous_shaped_memoryview_normalizes_to_bytes() -> None:
    shaped = memoryview(bytes(ISOLATION_FRAME_BYTES)).cast(
        "B",
        shape=[SAMPLES_PER_FRAME, 2],
    )

    samples = _pcm16_samples(shaped)

    assert samples is not None
    assert len(samples) == SAMPLES_PER_FRAME


def test_exact_sized_contiguous_nonbyte_memoryview_normalizes_to_bytes() -> None:
    words = array("h", [0]) * SAMPLES_PER_FRAME
    view = memoryview(words)
    assert view.format != "B"
    assert view.nbytes == ISOLATION_FRAME_BYTES

    samples = _pcm16_samples(view)

    assert samples is not None
    assert len(samples) == SAMPLES_PER_FRAME


def test_noncontiguous_memoryview_fails_closed_without_raising() -> None:
    view = memoryview(bytes(ISOLATION_FRAME_BYTES * 2))[::2]
    assert view.nbytes == ISOLATION_FRAME_BYTES
    assert view.contiguous is False

    assert _pcm16_samples(view) is None


def test_bounded_bytes_like_pcm_inputs_are_accepted() -> None:
    monitor = AcousticIsolationMonitor(
        window_frames=5,
        required_windows=1,
    )
    for sequence in range(5):
        capture = replace(  # type: ignore[arg-type]
            frame(sequence),
            pcm16=bytearray(pcm16(QUIET)),
        )
        render = replace(  # type: ignore[arg-type]
            render_frame(sequence),
            pcm16=memoryview(pcm16(RENDER)),
        )
        observation = monitor.observe(
            capture=capture,
            render_frames=(render,),
            assistant_rendering=True,
            near_end_speech=False,
            native_processor_ok=True,
        )

    assert observation.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION


def test_render_iteration_stops_at_first_frame_over_cap() -> None:
    monitor = AcousticIsolationMonitor(window_frames=10, required_windows=1)
    limit = monitor._render_history.maxlen
    assert limit is not None

    class RenderProbe:
        def __init__(self) -> None:
            self.calls = 0
            self.overread = False

        def __iter__(self) -> RenderProbe:
            return self

        def __next__(self) -> AudioFrame:
            if self.calls == limit + 1:
                self.overread = True
                raise AssertionError("render iterator consumed beyond overflow probe")
            sequence = self.calls
            self.calls += 1
            return render_frame(sequence)

    probe = RenderProbe()
    observation = monitor.observe(
        capture=frame(limit),
        render_frames=probe,
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=True,
    )

    assert probe.calls == limit + 1
    assert probe.overread is False
    assert observation.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert observation.safety.demotion_reason == "render-reference-overflow"


@pytest.mark.parametrize(
    ("condition", "expected_reason"),
    [
        ("missing-render", AcousticDemotionReason.MISSING_RENDER_REFERENCE),
        ("missing-timing", AcousticDemotionReason.MISSING_TIMING),
        ("capture-sequence-gap", AcousticDemotionReason.CAPTURE_SEQUENCE_GAP),
        ("render-sequence-gap", AcousticDemotionReason.RENDER_SEQUENCE_GAP),
        ("reference-sequence-gap", AcousticDemotionReason.RENDER_REFERENCE_GAP),
        ("route-mismatch", AcousticDemotionReason.ROUTE_MISMATCH),
        ("native-failure", AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE),
    ],
)
def test_unsafe_observation_revokes_open_isolation_proof(
    condition: str,
    expected_reason: AcousticDemotionReason,
) -> None:
    monitor = opened_monitor()
    capture = frame(20)
    render_frames = (render_frame(20),)
    native_processor_ok = True

    if condition == "missing-render":
        render_frames = ()
    elif condition == "missing-timing":
        capture = replace(capture, delay_evidence=None)
    elif condition == "capture-sequence-gap":
        capture = frame(21)
        render_frames = (render_frame(21),)
    elif condition == "render-sequence-gap":
        render_frames = (render_frame(21),)
    elif condition == "reference-sequence-gap":
        capture = replace(capture, render_reference_sequence=21)
    elif condition == "route-mismatch":
        capture = frame(20, generation=1)
    elif condition == "native-failure":
        native_processor_ok = False

    snapshot = monitor.observe(
        capture=capture,
        render_frames=render_frames,
        assistant_rendering=True,
        near_end_speech=False,
        native_processor_ok=native_processor_ok,
    )

    assert snapshot.safety.path is AcousticSafetyPath.HALF_DUPLEX
    assert snapshot.safety.mode is DuplexMode.HALF_DUPLEX
    assert snapshot.safety.admission_open is False
    assert snapshot.safety.demotion_reason is expected_reason


def test_metric_samples_are_finite_bounded_and_canonicalized_once() -> None:
    monitor = AcousticIsolationMonitor(window_frames=1, required_windows=1)

    for sequence in range(ISOLATION_METRIC_SAMPLES_MAX + 5):
        monitor.observe(
            capture=frame(sequence),
            render_frames=(render_frame(sequence),),
            assistant_rendering=True,
            near_end_speech=False,
            native_processor_ok=True,
        )

    correlations = monitor.correlation_samples
    leakage = monitor.leakage_db_samples
    assert len(correlations) == ISOLATION_METRIC_SAMPLES_MAX
    assert len(leakage) == ISOLATION_METRIC_SAMPLES_MAX
    assert all(
        math.isfinite(value) and value == round(value, 6) for value in correlations
    )
    assert all(math.isfinite(value) and value == round(value, 6) for value in leakage)
    assert correlations == monitor.correlation_samples
    assert leakage == monitor.leakage_db_samples


def test_metric_samples_store_one_six_decimal_canonical_value() -> None:
    monitor = AcousticIsolationMonitor(window_frames=1, required_windows=1)
    render = (4_000, -3_000, 2_000, -1_000, 500, -500, 1_000, -2_000, 3_000, -4_000)
    dot_product = sum(
        render_sample * capture_sample
        for render_sample, capture_sample in zip(render, QUIET, strict=True)
    )
    render_energy = sum(sample * sample for sample in render)
    capture_energy = sum(sample * sample for sample in QUIET)
    raw_correlation = abs(dot_product) / math.sqrt(render_energy * capture_energy)
    raw_leakage_db = 20.0 * math.log10(abs(dot_product) / render_energy)

    for sequence in range(5):
        monitor.observe(
            capture=frame(sequence),
            render_frames=(render_frame(sequence, render),),
            assistant_rendering=True,
            near_end_speech=False,
            native_processor_ok=True,
        )

    assert monitor.correlation_samples == (round(raw_correlation, 6),)
    assert monitor.leakage_db_samples == (round(raw_leakage_db, 6),)
    assert monitor.correlation_samples[0] != raw_correlation
    assert monitor.leakage_db_samples[0] != raw_leakage_db


def test_monitor_imports_without_optional_dsp_dependencies() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import tldw_chatbook.Audio.acoustic_isolation; "
                "assert 'numpy' not in sys.modules; "
                "assert 'scipy' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr
