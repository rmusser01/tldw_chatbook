"""Dependency-free acoustic-isolation safety monitor."""

from __future__ import annotations

from array import array
from collections import deque
import math
import statistics
from typing import Iterable

from .duplex_contracts import (
    AcousticDemotionReason,
    AcousticIsolationObservation,
    AcousticSafetyPath,
    AcousticSafetySnapshot,
    AudioFrame,
    DuplexMode,
    NearEndDisposition,
)

ISOLATION_FRAME_BYTES = 960
ISOLATION_WINDOW_FRAMES = 100
ISOLATION_REQUIRED_WINDOWS = 5
ISOLATION_SAMPLE_STRIDE = 8
ISOLATION_MAX_LAG_MS = 500
ISOLATION_RENDER_RMS_MIN = 512.0
ISOLATION_MAX_CORRELATION = 0.12
ISOLATION_MAX_LEAKAGE_DB = -30.0
ISOLATION_FLOOR_DB = -120.0
ISOLATION_METRIC_SAMPLES_MAX = 3_600
ISOLATION_NEAR_END_CONFIRM_FRAMES = 5
ISOLATION_NEAR_END_RENDER_DOMINANCE_CORRELATION = 0.85

_FRAME_DURATION_NS = 10_000_000
_FRAME_DURATION_MS = 10
_LATENCY_GUARD_MS = 50
_SCHEDULED_REFERENCE_TOLERANCE_NS = 500_000
_METRIC_DECIMAL_PLACES = 6
_NATIVE_PCM_NEEDS_BYTESWAP = array("H", [1]).tobytes() != b"\x01\x00"

_CaptureVector = tuple[int, tuple[float, ...], int, int]
_RenderEvidence = tuple[tuple[float, ...], int]
_RenderVector = tuple[int, tuple[float, ...], int]


class AcousticIsolationMonitor:
    """Accumulate bounded render/capture evidence for acoustic isolation."""

    def __init__(
        self,
        *,
        window_frames: int = ISOLATION_WINDOW_FRAMES,
        required_windows: int = ISOLATION_REQUIRED_WINDOWS,
    ) -> None:
        if min(window_frames, required_windows) <= 0:
            raise ValueError("isolation window bounds must be positive")
        self._window_frames = window_frames
        self._required_windows = required_windows
        self._clock_generation = 0
        self._capture_window: list[_CaptureVector] = []
        self._near_end_frames: list[_CaptureVector] = []
        self._near_end_admitted = False
        self._render_history: deque[_RenderVector] = deque(
            maxlen=window_frames + ISOLATION_MAX_LAG_MS // _FRAME_DURATION_MS,
        )
        self._correlations: deque[float] = deque(
            maxlen=ISOLATION_METRIC_SAMPLES_MAX,
        )
        self._leakage_db: deque[float] = deque(
            maxlen=ISOLATION_METRIC_SAMPLES_MAX,
        )
        self._last_capture_sequence: int | None = None
        self._last_render_sequence: int | None = None
        self._last_reference_sequence: int | None = None
        self._clean_windows = 0
        self._opened = False
        self._demotion_reason: AcousticDemotionReason | None = None

    def observe(
        self,
        *,
        capture: AudioFrame,
        render_frames: Iterable[AudioFrame],
        assistant_rendering: bool,
        near_end_speech: bool,
        native_processor_ok: bool,
    ) -> AcousticIsolationObservation:
        """Observe one processed capture boundary."""

        if not native_processor_ok:
            return self._fail_closed(
                AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE,
                clear_history=True,
                near_end_speech=near_end_speech,
            )
        if capture.clock_generation != self._clock_generation:
            return self._fail_closed(
                AcousticDemotionReason.ROUTE_MISMATCH,
                clear_history=True,
                near_end_speech=near_end_speech,
            )
        timing_issue = _timing_issue(capture)
        if timing_issue is not None:
            return self._fail_closed(
                timing_issue,
                clear_history=True,
                near_end_speech=near_end_speech,
            )
        capture_pcm = _pcm16_samples(capture.pcm16)
        if capture_pcm is None:
            return self._fail_closed(
                AcousticDemotionReason.INVALID_CAPTURE_PCM,
                near_end_speech=near_end_speech,
            )
        if _is_saturated(capture_pcm):
            return self._fail_closed(
                AcousticDemotionReason.SATURATION,
                near_end_speech=near_end_speech,
            )
        if assistant_rendering and (
            self._last_capture_sequence is not None
            and capture.sequence != self._last_capture_sequence + 1
        ):
            self._last_capture_sequence = capture.sequence
            return self._fail_closed(
                AcousticDemotionReason.CAPTURE_SEQUENCE_GAP,
                clear_history=True,
                near_end_speech=near_end_speech,
            )
        if assistant_rendering:
            self._last_capture_sequence = capture.sequence
        render_issue = self._append_render_frames(render_frames)
        if render_issue is not None:
            return self._fail_closed(
                render_issue,
                clear_history=True,
                near_end_speech=near_end_speech,
            )
        timing = capture.delay_evidence
        assert timing is not None
        self._prune_render_history(timing.capture_adc_ns, timing.delay_ms)
        if not assistant_rendering:
            self._capture_window.clear()
            self._reset_near_end_event()
            self._last_capture_sequence = None
            self._last_reference_sequence = None
            disposition = NearEndDisposition.ADMIT if near_end_speech else None
            return self._observation(disposition)

        reference_sequence = capture.render_reference_sequence
        if reference_sequence is None:
            return self._fail_closed(
                AcousticDemotionReason.MISSING_RENDER_REFERENCE,
                near_end_speech=near_end_speech,
            )
        if (
            self._last_reference_sequence is not None
            and reference_sequence != self._last_reference_sequence + 1
        ):
            self._last_reference_sequence = reference_sequence
            return self._fail_closed(
                AcousticDemotionReason.RENDER_REFERENCE_GAP,
                clear_history=True,
                near_end_speech=near_end_speech,
            )
        self._last_reference_sequence = reference_sequence
        render_by_sequence = _render_index(self._render_history)
        if (
            _scheduled_render(
                render_by_sequence,
                reference_sequence,
                timing.capture_adc_ns,
                timing.delay_ms,
            )
            is None
        ):
            return self._fail_closed(
                AcousticDemotionReason.MISSING_RENDER_REFERENCE,
                near_end_speech=near_end_speech,
            )

        capture_samples = _center(list(capture_pcm[::ISOLATION_SAMPLE_STRIDE]))
        if capture_samples is None:
            return self._fail_closed(
                AcousticDemotionReason.INVALID_CAPTURE_PCM,
                near_end_speech=near_end_speech,
            )
        capture_vector = (
            reference_sequence,
            capture_samples,
            timing.delay_ms,
            timing.capture_adc_ns,
        )

        if near_end_speech:
            self._capture_window.clear()
            if not self._opened:
                self._clean_windows = 0
            if self._near_end_admitted:
                return self._observation(NearEndDisposition.ADMIT)
            self._near_end_frames.append(capture_vector)
            if len(self._near_end_frames) < ISOLATION_NEAR_END_CONFIRM_FRAMES:
                return self._observation(NearEndDisposition.PENDING)
            render_dominant = _event_render_dominant(
                tuple(self._near_end_frames),
                render_by_sequence,
            )
            if render_dominant is None:
                return self._fail_closed(
                    AcousticDemotionReason.AMBIGUOUS_CORRELATION,
                    near_end_speech=True,
                )
            if render_dominant:
                return self._fail_closed(
                    AcousticDemotionReason.CORRELATED_RENDER,
                    near_end_speech=True,
                )
            self._near_end_frames.clear()
            self._near_end_admitted = True
            return self._observation(NearEndDisposition.ADMIT)

        self._reset_near_end_event()
        self._capture_window.append(capture_vector)
        if len(self._capture_window) < self._window_frames:
            return self._observation()
        return self._close_window()

    def reset_for_route(self, clock_generation: int) -> None:
        """Clear evidence after a device-clock generation change."""

        if type(clock_generation) is not int:
            raise TypeError("clock generation must be an integer")
        if clock_generation < 0:
            raise ValueError("clock generation must be non-negative")
        self._clock_generation = clock_generation
        self._capture_window.clear()
        self._reset_near_end_event()
        self._render_history.clear()
        self._correlations.clear()
        self._leakage_db.clear()
        self._last_capture_sequence = None
        self._last_render_sequence = None
        self._last_reference_sequence = None
        self._clean_windows = 0
        self._opened = False
        self._demotion_reason = None

    @property
    def correlation_samples(self) -> tuple[float, ...]:
        """Return bounded per-window normalized correlation observations."""

        return tuple(self._correlations)

    @property
    def leakage_db_samples(self) -> tuple[float, ...]:
        """Return bounded per-window correlated leakage observations."""

        return tuple(self._leakage_db)

    def _append_render_frames(
        self,
        render_frames: Iterable[AudioFrame],
    ) -> AcousticDemotionReason | None:
        try:
            iterator = iter(render_frames)
            limit = self._render_history.maxlen
            assert limit is not None
            for index in range(limit + 1):
                try:
                    render = next(iterator)
                except StopIteration:
                    return None
                if index == limit:
                    return AcousticDemotionReason.RENDER_REFERENCE_OVERFLOW
                if render.clock_generation != self._clock_generation:
                    return AcousticDemotionReason.ROUTE_MISMATCH
                timing_issue = _timing_issue(render)
                if timing_issue is not None:
                    return timing_issue
                if (
                    self._last_render_sequence is not None
                    and render.sequence != self._last_render_sequence + 1
                ):
                    self._last_render_sequence = render.sequence
                    return AcousticDemotionReason.RENDER_SEQUENCE_GAP
                pcm = _pcm16_samples(render.pcm16)
                if pcm is None:
                    return AcousticDemotionReason.INVALID_RENDER_PCM
                if _is_saturated(pcm):
                    return AcousticDemotionReason.SATURATION
                samples = _center(list(pcm[::ISOLATION_SAMPLE_STRIDE]))
                if samples is None:
                    return AcousticDemotionReason.INVALID_RENDER_PCM
                self._last_render_sequence = render.sequence
                timing = render.delay_evidence
                assert timing is not None
                self._render_history.append(
                    (render.sequence, samples, timing.render_dac_ns)
                )
        except Exception:
            return AcousticDemotionReason.RENDER_REFERENCE_FAILURE
        return None

    def _prune_render_history(
        self,
        capture_adc_ns: int,
        delay_ms: int,
    ) -> None:
        capture_ranges = [
            (pending_adc_ns, pending_delay_ms)
            for _reference, _samples, pending_delay_ms, pending_adc_ns in (
                self._capture_window + self._near_end_frames
            )
        ]
        capture_ranges.append((capture_adc_ns, delay_ms))
        oldest_plausible_dac_ns = min(
            adc_ns - _lag_limit_ns(candidate_delay_ms)
            for adc_ns, candidate_delay_ms in capture_ranges
        )
        retained = (
            render
            for render in self._render_history
            if render[2] >= oldest_plausible_dac_ns
        )
        self._render_history = deque(retained, maxlen=self._render_history.maxlen)

    def _close_window(self) -> AcousticIsolationObservation:
        metrics = self._analyze(tuple(self._capture_window))
        self._capture_window.clear()
        if metrics is None:
            return self._fail_closed(AcousticDemotionReason.AMBIGUOUS_CORRELATION)
        render_rms, correlation, leakage_db = metrics
        self._correlations.append(round(correlation, _METRIC_DECIMAL_PLACES))
        self._leakage_db.append(round(leakage_db, _METRIC_DECIMAL_PLACES))
        if render_rms < ISOLATION_RENDER_RMS_MIN:
            return self._fail_closed(AcousticDemotionReason.INAUDIBLE_RENDER)
        if (
            correlation > ISOLATION_MAX_CORRELATION
            or leakage_db > ISOLATION_MAX_LEAKAGE_DB
        ):
            return self._fail_closed(AcousticDemotionReason.CORRELATED_RENDER)

        self._clean_windows += 1
        self._demotion_reason = None
        if self._clean_windows >= self._required_windows:
            self._opened = True
        return self._observation()

    def _analyze(
        self,
        captures: tuple[_CaptureVector, ...],
    ) -> tuple[float, float, float] | None:
        render_by_sequence = _render_index(self._render_history)
        current_render: list[float] = []
        for reference_sequence, _capture, delay_ms, capture_adc_ns in captures:
            render = _scheduled_render(
                render_by_sequence,
                reference_sequence,
                capture_adc_ns,
                delay_ms,
            )
            if render is None:
                return None
            current_render.extend(render)
        render_rms = math.sqrt(
            statistics.fmean(value * value for value in current_render),
        )

        max_delay_ms = max(delay_ms for _ref, _samples, delay_ms, _adc in captures)
        lag_limit_ms = min(
            ISOLATION_MAX_LAG_MS,
            max_delay_ms + _LATENCY_GUARD_MS,
        )
        lag_limit_frames = lag_limit_ms // _FRAME_DURATION_MS
        correlations: list[float] = []
        leakages: list[float] = []
        for lag_frames in range(lag_limit_frames + 1):
            capture_values: list[float] = []
            render_values: list[float] = []
            for reference_sequence, capture, delay_ms, capture_adc_ns in captures:
                render = _causal_render(
                    render_by_sequence,
                    reference_sequence - lag_frames,
                    capture_adc_ns,
                    delay_ms,
                )
                if render is None:
                    continue
                if len(render) != len(capture):
                    return None
                capture_values.extend(capture)
                render_values.extend(render)
            if len(capture_values) < 2 or len(render_values) < 2:
                continue
            render_energy = sum(value * value for value in render_values)
            capture_energy = sum(value * value for value in capture_values)
            if render_energy <= 0:
                continue
            dot_product = sum(
                render * capture
                for render, capture in zip(render_values, capture_values, strict=True)
            )
            if capture_energy == 0:
                correlation = 0.0
            else:
                correlation = abs(dot_product) / math.sqrt(
                    render_energy * capture_energy,
                )
            gain = abs(dot_product) / render_energy
            leakage_db = (
                ISOLATION_FLOOR_DB
                if gain == 0
                else max(ISOLATION_FLOOR_DB, 20.0 * math.log10(gain))
            )
            if not math.isfinite(correlation) or not math.isfinite(leakage_db):
                return None
            correlations.append(correlation)
            leakages.append(leakage_db)
        if not correlations:
            return None
        return render_rms, max(correlations), max(leakages)

    def _fail_closed(
        self,
        reason: AcousticDemotionReason,
        *,
        clear_history: bool = False,
        near_end_speech: bool = False,
    ) -> AcousticIsolationObservation:
        self._capture_window.clear()
        self._reset_near_end_event()
        if clear_history:
            self._render_history.clear()
            self._last_render_sequence = None
            self._last_reference_sequence = None
        self._clean_windows = 0
        self._opened = False
        self._demotion_reason = reason
        disposition = NearEndDisposition.FENCE if near_end_speech else None
        return self._observation(disposition)

    def _reset_near_end_event(self) -> None:
        self._near_end_frames.clear()
        self._near_end_admitted = False

    def _observation(
        self,
        disposition: NearEndDisposition | None = None,
    ) -> AcousticIsolationObservation:
        return AcousticIsolationObservation(self._snapshot(), disposition)

    def _snapshot(self) -> AcousticSafetySnapshot:
        if self._opened:
            return AcousticSafetySnapshot(
                AcousticSafetyPath.ACOUSTIC_ISOLATION,
                DuplexMode.FULL_DUPLEX,
                self._clock_generation,
                True,
                None,
            )
        if self._demotion_reason is not None:
            return AcousticSafetySnapshot(
                AcousticSafetyPath.HALF_DUPLEX,
                DuplexMode.HALF_DUPLEX,
                self._clock_generation,
                False,
                self._demotion_reason,
            )
        return AcousticSafetySnapshot(
            AcousticSafetyPath.WARMING,
            DuplexMode.HALF_DUPLEX,
            self._clock_generation,
            False,
            None,
        )


def _timing_issue(frame: AudioFrame) -> AcousticDemotionReason | None:
    evidence = frame.delay_evidence
    if evidence is None:
        return AcousticDemotionReason.MISSING_TIMING
    if frame.duration_ns != _FRAME_DURATION_NS:
        return AcousticDemotionReason.UNBOUNDED_TIMING
    if not evidence.occupancy_bounded:
        return AcousticDemotionReason.UNBOUNDED_TIMING
    if (
        frame.discontinuity
        or evidence.timing_discontinuity
        or evidence.clock_drift
        or evidence.status_flags
    ):
        return AcousticDemotionReason.TIMING_DISCONTINUITY
    return None


def _pcm16_samples(pcm16: object) -> tuple[int, ...] | None:
    if not isinstance(pcm16, (bytes, bytearray, memoryview)):
        return None
    try:
        view = memoryview(pcm16)
        if view.nbytes != ISOLATION_FRAME_BYTES or not view.c_contiguous:
            return None
        byte_view = view.cast("B")
        if byte_view.format != "B" or byte_view.ndim != 1:
            return None
        samples = array("h")
        samples.frombytes(byte_view)
    except (BufferError, TypeError, ValueError):
        return None
    if _NATIVE_PCM_NEEDS_BYTESWAP:
        samples.byteswap()
    return tuple(samples)


def _is_saturated(samples: tuple[int, ...]) -> bool:
    return any(sample in {-32_768, 32_767} for sample in samples)


def _center(values: list[int]) -> tuple[float, ...] | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return tuple(value - mean for value in values)


def _lag_limit_ns(delay_ms: int) -> int:
    return min(ISOLATION_MAX_LAG_MS, delay_ms + _LATENCY_GUARD_MS) * 1_000_000


def _render_index(
    history: Iterable[_RenderVector],
) -> dict[int, _RenderEvidence]:
    return {
        sequence: (samples, render_dac_ns)
        for sequence, samples, render_dac_ns in history
    }


def _scheduled_render(
    render_by_sequence: dict[int, _RenderEvidence],
    sequence: int,
    capture_adc_ns: int,
    delay_ms: int,
) -> tuple[float, ...] | None:
    evidence = render_by_sequence.get(sequence)
    if evidence is None:
        return None
    samples, render_dac_ns = evidence
    scheduled_offset_ns = render_dac_ns - capture_adc_ns
    if (
        abs(scheduled_offset_ns - delay_ms * 1_000_000)
        > _SCHEDULED_REFERENCE_TOLERANCE_NS
    ):
        return None
    return samples


def _causal_render(
    render_by_sequence: dict[int, _RenderEvidence],
    sequence: int,
    capture_adc_ns: int,
    delay_ms: int,
) -> tuple[float, ...] | None:
    evidence = render_by_sequence.get(sequence)
    if evidence is None:
        return None
    samples, render_dac_ns = evidence
    age_ns = capture_adc_ns - render_dac_ns
    if not 0 <= age_ns <= _lag_limit_ns(delay_ms):
        return None
    return samples


def _event_render_dominant(
    captures: tuple[_CaptureVector, ...],
    render_by_sequence: dict[int, _RenderEvidence],
) -> bool | None:
    if len(captures) != ISOLATION_NEAR_END_CONFIRM_FRAMES:
        return None

    capture_energies = []
    for reference_sequence, capture, delay_ms, capture_adc_ns in captures:
        current_render = _scheduled_render(
            render_by_sequence,
            reference_sequence,
            capture_adc_ns,
            delay_ms,
        )
        if current_render is None or len(current_render) != len(capture):
            return None
        capture_energy = sum(value * value for value in capture)
        render_energy = sum(value * value for value in current_render)
        if (
            capture_energy <= 0
            or render_energy <= 0
            or not math.isfinite(capture_energy)
            or not math.isfinite(render_energy)
        ):
            return None
        capture_energies.append(capture_energy)

    max_delay_ms = max(delay_ms for _ref, _samples, delay_ms, _adc in captures)
    lag_limit_ms = min(ISOLATION_MAX_LAG_MS, max_delay_ms + _LATENCY_GUARD_MS)
    complete_lag_found = False
    render_energy_cache: dict[int, float] = {}
    for lag_frames in range(lag_limit_ms // _FRAME_DURATION_MS + 1):
        sign = 0
        same_sign = True
        complete = True
        dot_product = 0.0
        render_energy = 0.0
        capture_energy = 0.0
        for index, (
            reference_sequence,
            capture,
            delay_ms,
            capture_adc_ns,
        ) in enumerate(captures):
            render_sequence = reference_sequence - lag_frames
            render = _causal_render(
                render_by_sequence,
                render_sequence,
                capture_adc_ns,
                delay_ms,
            )
            if render is None:
                complete = False
                break
            if len(render) != len(capture):
                return None
            frame_render_energy = render_energy_cache.get(render_sequence)
            if frame_render_energy is None:
                frame_render_energy = sum(value * value for value in render)
                render_energy_cache[render_sequence] = frame_render_energy
            if frame_render_energy <= 0 or not math.isfinite(frame_render_energy):
                complete = False
                break
            frame_dot_product = sum(
                render_sample * capture_sample
                for render_sample, capture_sample in zip(
                    render,
                    capture,
                    strict=True,
                )
            )
            if not math.isfinite(frame_dot_product):
                return None
            current_sign = (
                1 if frame_dot_product > 0 else -1 if frame_dot_product < 0 else 0
            )
            if current_sign == 0 or (sign != 0 and current_sign != sign):
                same_sign = False
            if sign == 0:
                sign = current_sign
            dot_product += frame_dot_product
            render_energy += frame_render_energy
            capture_energy += capture_energies[index]

        if not complete:
            continue
        complete_lag_found = True
        if not same_sign:
            continue
        correlation = dot_product / math.sqrt(render_energy * capture_energy)
        gain = dot_product / render_energy
        if not math.isfinite(correlation) or not math.isfinite(gain):
            return None
        if (
            abs(correlation) > ISOLATION_NEAR_END_RENDER_DOMINANCE_CORRELATION
            and _leakage_db(gain) > ISOLATION_MAX_LEAKAGE_DB
        ):
            return True
    return False if complete_lag_found else None


def _leakage_db(gain: float) -> float:
    return (
        ISOLATION_FLOOR_DB
        if gain == 0
        else max(ISOLATION_FLOOR_DB, 20.0 * math.log10(abs(gain)))
    )
