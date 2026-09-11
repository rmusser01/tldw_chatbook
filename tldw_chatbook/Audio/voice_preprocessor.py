"""48 kHz frame normalization and fail-closed AEC/VAD admission."""

from __future__ import annotations

from array import array
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from importlib import import_module
import math
import sys

from .aec_backend import AecProcessor, create_aec_processor
from .acoustic_isolation import AcousticIsolationMonitor
from .duplex_contracts import (
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

PROCESSING_SAMPLE_RATE = 48_000
PROCESSING_CHANNELS = 1
FRAME_DURATION_NS = 10_000_000
FRAME_SAMPLES = 480
FRAME_BYTES = FRAME_SAMPLES * 2
_MAX_RENDER_REFERENCE_FRAMES = 150
_PENDING_REPLAY_FRAMES = 5
_DEFAULT_VAD_PREROLL_MS = 240
_METRIC_NAMES = (
    "erle_db",
    "delay_ms",
    "delay_estimate_available",
    "delay_estimate_refined",
    "delay_age_blocks",
    "clock_drift",
)
_AEC_OVERRIDEABLE_ISOLATION_REASONS = frozenset(
    {
        None,
        AcousticDemotionReason.CORRELATED_RENDER,
        AcousticDemotionReason.INAUDIBLE_RENDER,
    }
)

_Vad = Callable[[AudioFrame], bool]
_FrameConsumer = Callable[[AudioFrame], None]
_ProcessedConsumer = Callable[[int, int, bool, bool], None]


@dataclass(frozen=True, slots=True)
class PendingClassificationWatermark:
    """Content-free bounds for one unresolved playback speech event."""

    clock_generation: int
    first_sequence: int
    first_started_ns: int
    through_sequence: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int
            for value in (
                self.clock_generation,
                self.first_sequence,
                self.first_started_ns,
                self.through_sequence,
            )
        ):
            raise TypeError("pending classification bounds must be integers")
        if min(self.clock_generation, self.first_sequence, self.first_started_ns) < 0:
            raise ValueError("pending classification bounds must be non-negative")
        if self.through_sequence < self.first_sequence:
            raise ValueError("pending classification sequence bounds are invalid")


_ClassificationConsumer = Callable[[PendingClassificationWatermark | None], None]


def create_webrtc_vad(*, aggressiveness: int = 2) -> _Vad:
    """Build the qualified 48 kHz/10 ms WebRTC speech detector lazily."""

    if type(aggressiveness) is not int or not 0 <= aggressiveness <= 3:
        raise ValueError("VAD aggressiveness must be an integer from zero to three")
    detector = import_module("webrtcvad").Vad(aggressiveness)

    def detect(frame: AudioFrame) -> bool:
        return bool(detector.is_speech(frame.pcm16, PROCESSING_SAMPLE_RATE))

    return detect


def read_aec_metric_snapshot(
    aec: AecProcessor,
) -> tuple[dict[str, float], AecHealth | None]:
    """Read and strictly validate one native AEC metric snapshot."""

    explicit_health = getattr(aec, "health", None)
    if explicit_health is not None and not isinstance(explicit_health, AecHealth):
        raise TypeError("native AEC health state is invalid")
    metrics = aec.metrics()
    if not isinstance(metrics, Mapping) or set(metrics) != set(_METRIC_NAMES):
        raise TypeError("native AEC metrics must have the exact required mapping")
    values: dict[str, float] = {}
    for name in _METRIC_NAMES:
        raw_value = metrics[name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"native AEC metric {name} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"native AEC metric {name} must be finite")
        values[name] = value
    if not -100.0 <= values["erle_db"] <= 100.0:
        raise ValueError("native AEC ERLE is outside the supported range")
    available = values["delay_estimate_available"]
    refined = values["delay_estimate_refined"]
    if available not in {0.0, 1.0} or refined not in {0.0, 1.0}:
        raise ValueError("native AEC delay flags must be binary")
    if refined > available:
        raise ValueError("refined delay requires an available estimate")
    if not 0.0 <= values["delay_ms"] <= 1_000.0:
        raise ValueError("native AEC delay is outside the supported range")
    if values["delay_age_blocks"] < 0.0:
        raise ValueError("native AEC delay age must be non-negative")
    if values["clock_drift"] not in {0.0, 1.0}:
        raise ValueError("native AEC drift flag must be binary")
    if explicit_health is AecHealth.DEGRADED:
        raise RuntimeError("native AEC reported degraded health")
    return values, explicit_health


class _AcousticProcessingFailure(RuntimeError):
    def __init__(self, reason: AcousticDemotionReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


def normalize_pcm16_frames(
    pcm16: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> tuple[bytes, ...]:
    """Downmix and resample whole PCM16 input into 48 kHz mono 10 ms frames."""

    if sample_rate <= 0 or channels <= 0:
        raise ValueError("sample rate and channels must be positive")
    if sample_rate > PROCESSING_SAMPLE_RATE:
        raise ValueError("downsampling is unsupported without a band-limited resampler")
    bytes_per_source_frame = channels * 2
    if not pcm16 or len(pcm16) % bytes_per_source_frame:
        raise ValueError("PCM16 input must contain complete interleaved samples")

    samples = array("h")
    samples.frombytes(pcm16)
    if sys.byteorder != "little":
        samples.byteswap()
    source_frames = len(samples) // channels
    mono = [
        sum(samples[index : index + channels]) // channels
        for index in range(0, len(samples), channels)
    ]

    target_numerator = source_frames * PROCESSING_SAMPLE_RATE
    if target_numerator % sample_rate:
        raise ValueError("PCM duration must map to whole output samples")
    target_samples = target_numerator // sample_rate
    if target_samples == 0 or target_samples % FRAME_SAMPLES:
        raise ValueError("PCM input must contain whole ten-millisecond frames")

    if sample_rate == PROCESSING_SAMPLE_RATE:
        normalized = mono
    else:
        normalized = []
        for output_index in range(target_samples):
            source_position = output_index * sample_rate
            left_index, fraction = divmod(
                source_position,
                PROCESSING_SAMPLE_RATE,
            )
            if left_index >= source_frames - 1:
                value = mono[-1]
            else:
                left = mono[left_index]
                right = mono[left_index + 1]
                value = round(
                    (left * (PROCESSING_SAMPLE_RATE - fraction) + right * fraction)
                    / PROCESSING_SAMPLE_RATE
                )
            normalized.append(max(-32_768, min(32_767, value)))

    encoded = array("h", normalized)
    if sys.byteorder != "little":
        encoded.byteswap()
    output = encoded.tobytes()
    return tuple(
        output[offset : offset + FRAME_BYTES]
        for offset in range(0, len(output), FRAME_BYTES)
    )


class VoicePreprocessor:
    """Run ordered AEC, health hysteresis, VAD, and admission policy."""

    def __init__(
        self,
        *,
        aec: AecProcessor | None,
        vad: _Vad | None = None,
        on_admitted_frame: _FrameConsumer | None = None,
        on_processed: _ProcessedConsumer | None = None,
        healthy_streak: int = 3,
        minimum_erle_db: float = 5.0,
        maximum_delay_age_blocks: int = 25,
        warming_observation_limit: int = 1_500,
        clock_generation: int = 0,
        isolation_monitor: AcousticIsolationMonitor | None = None,
        on_classification_changed: _ClassificationConsumer | None = None,
        vad_preroll_ms: int = _DEFAULT_VAD_PREROLL_MS,
    ) -> None:
        if (
            min(healthy_streak, maximum_delay_age_blocks, warming_observation_limit)
            <= 0
        ):
            raise ValueError("AEC health bounds must be positive")
        if not math.isfinite(minimum_erle_db):
            raise ValueError("minimum ERLE must be finite")
        if clock_generation < 0:
            raise ValueError("clock generation must be non-negative")
        if type(vad_preroll_ms) is not int or vad_preroll_ms < 0:
            raise ValueError("VAD pre-roll must be a non-negative integer")
        self._aec = aec
        self._vad = vad or _energy_vad
        self._on_admitted_frame = on_admitted_frame or _ignore_frame
        self._on_processed = on_processed or _ignore_processed
        self._on_classification_changed = (
            on_classification_changed or _ignore_classification
        )
        self._healthy_streak_required = healthy_streak
        self._minimum_erle_db = minimum_erle_db
        self._maximum_delay_age_blocks = maximum_delay_age_blocks
        self._warming_observation_limit = warming_observation_limit
        self._healthy_observations = 0
        self._warming_observations = 0
        self._health = AecHealth.WARMING if aec is not None else AecHealth.DEGRADED
        self._last_render_sequence = -1
        self._render_generation: int | None = None
        self._last_capture_sequence: int | None = None
        self._active_clock_generation = clock_generation
        self._capture_continuity_failed = False
        self._capture_failure_reason: AcousticDemotionReason | None = None
        self._isolation_monitor = (
            isolation_monitor
            if isolation_monitor is not None
            else AcousticIsolationMonitor()
            if aec is not None
            else None
        )
        if self._isolation_monitor is not None and clock_generation:
            self._isolation_monitor.reset_for_route(clock_generation)
        initial_path = (
            AcousticSafetyPath.WARMING
            if aec is not None
            else AcousticSafetyPath.HALF_DUPLEX
        )
        self._safety = AcousticSafetySnapshot(
            initial_path,
            DuplexMode.HALF_DUPLEX,
            clock_generation,
            False,
            None,
        )
        self._pending_replay: deque[AudioFrame] = deque(maxlen=_PENDING_REPLAY_FRAMES)
        self._pending_classification: PendingClassificationWatermark | None = None
        self._vad_preroll: deque[AudioFrame] = deque(
            maxlen=round(vad_preroll_ms / (FRAME_DURATION_NS / 1_000_000)),
        )

    @classmethod
    def from_native(cls, **kwargs: object) -> VoicePreprocessor:
        """Build from the optional companion, selecting half duplex if absent."""

        return cls(aec=create_aec_processor(), **kwargs)  # type: ignore[arg-type]

    @property
    def health(self) -> AecHealth:
        return self._health

    @property
    def mode(self) -> DuplexMode:
        return self._safety.mode

    @property
    def safety(self) -> AcousticSafetySnapshot:
        """Return the current closed acoustic-admission decision."""

        return self._safety

    @property
    def active_clock_generation(self) -> int:
        return self._active_clock_generation

    @property
    def pending_classification(self) -> PendingClassificationWatermark | None:
        """Return content-free bounds for unresolved playback speech."""

        return self._pending_classification

    async def process_capture(
        self,
        capture_frame: AudioFrame,
        *,
        render_frames: Iterable[AudioFrame] = (),
        assistant_rendering: bool,
        discontinuity: bool = False,
    ) -> AudioFrame | None:
        """Process one capture frame, admitting speech only through the safe gate."""

        try:
            capture_view = memoryview(capture_frame.pcm16)
            valid_capture_pcm = (
                capture_view.c_contiguous and capture_view.nbytes == FRAME_BYTES
            )
        except (TypeError, ValueError):
            valid_capture_pcm = False
        if not valid_capture_pcm:
            self._close_native_path(
                capture_frame,
                render_frames=(),
                assistant_rendering=assistant_rendering,
                reason_override=AcousticDemotionReason.INVALID_CAPTURE_PCM,
            )
            raise ValueError("capture PCM must be one 48 kHz mono 10 ms frame")
        discontinuity = discontinuity or capture_frame.discontinuity
        continuity = self._capture_continuity(capture_frame)
        if continuity is None:
            return None
        started_full_duplex = self._safety.admission_open
        pending_timing_discontinuity = (
            continuity
            and not assistant_rendering
            and bool(self._pending_replay)
            and capture_frame.clock_generation
            == self._pending_replay[-1].clock_generation
            and capture_frame.sequence == self._pending_replay[-1].sequence + 1
            and capture_frame.started_ns != self._pending_replay[-1].ended_ns
        )

        if not continuity or discontinuity or pending_timing_discontinuity:
            return self._handle_aec_failure(
                capture_frame,
                render_frames=(),
                assistant_rendering=assistant_rendering,
                started_full_duplex=started_full_duplex,
                reason=(
                    AcousticDemotionReason.TIMING_DISCONTINUITY
                    if discontinuity or pending_timing_discontinuity
                    else self._capture_failure_reason
                    or AcousticDemotionReason.CAPTURE_SEQUENCE_GAP
                ),
            )

        if self._aec is None:
            self._clear_pending_replay()
            self._publish_safety(AcousticSafetyPath.HALF_DUPLEX)
            if assistant_rendering:
                self._vad_preroll.clear()
                self._acknowledge(capture_frame, dsp_ok=True, vad_ok=True)
                return None
            return self._run_vad_and_admit(capture_frame, admit=True)

        timing = capture_frame.delay_evidence
        timing_failure = _timing_failure_reason(
            timing,
            duration_ns=capture_frame.duration_ns,
        )
        if timing_failure is not None:
            return self._handle_aec_failure(
                capture_frame,
                render_frames=(),
                assistant_rendering=assistant_rendering,
                started_full_duplex=started_full_duplex,
                reason=timing_failure,
            )
        assert timing is not None
        delay_ms = timing.delay_ms

        collected_render: list[AudioFrame] = []
        try:
            try:
                iterator = iter(render_frames)
            except Exception as exc:
                raise _AcousticProcessingFailure(
                    AcousticDemotionReason.RENDER_REFERENCE_FAILURE
                ) from exc
            for index in range(_MAX_RENDER_REFERENCE_FRAMES + 1):
                try:
                    render_frame = next(iterator)
                except StopIteration:
                    break
                except Exception as exc:
                    raise _AcousticProcessingFailure(
                        AcousticDemotionReason.RENDER_REFERENCE_FAILURE
                    ) from exc
                if index == _MAX_RENDER_REFERENCE_FRAMES:
                    raise _AcousticProcessingFailure(
                        AcousticDemotionReason.RENDER_REFERENCE_OVERFLOW
                    )
                collected_render.append(render_frame)
                if not _is_exact_pcm_frame(render_frame.pcm16):
                    raise _AcousticProcessingFailure(
                        AcousticDemotionReason.INVALID_RENDER_PCM
                    )
                render_timing_failure = _timing_failure_reason(
                    render_frame.delay_evidence,
                    duration_ns=render_frame.duration_ns,
                )
                if render_timing_failure is not None:
                    raise _AcousticProcessingFailure(render_timing_failure)
                if render_frame.clock_generation != capture_frame.clock_generation:
                    raise _AcousticProcessingFailure(
                        AcousticDemotionReason.ROUTE_MISMATCH
                    )
                if self._render_generation is None:
                    expected_sequence = render_frame.sequence
                elif self._render_generation == render_frame.clock_generation:
                    expected_sequence = self._last_render_sequence + 1
                else:
                    raise _AcousticProcessingFailure(
                        AcousticDemotionReason.ROUTE_MISMATCH
                    )
                if render_frame.sequence != expected_sequence:
                    raise _AcousticProcessingFailure(
                        AcousticDemotionReason.RENDER_SEQUENCE_GAP
                    )
                self._aec.analyze_render(render_frame.pcm16, delay_ms=delay_ms)
                self._render_generation = render_frame.clock_generation
                self._last_render_sequence = render_frame.sequence
            required_render_sequence = capture_frame.render_reference_sequence
            if required_render_sequence is not None and (
                self._render_generation != capture_frame.clock_generation
                or self._last_render_sequence != required_render_sequence
            ):
                raise _AcousticProcessingFailure(
                    AcousticDemotionReason.MISSING_RENDER_REFERENCE
                )
            cleaned_output = self._aec.process_capture(
                capture_frame.pcm16,
                delay_ms=delay_ms,
            )
            cleaned_view = memoryview(cleaned_output)
            if not cleaned_view.c_contiguous or cleaned_view.nbytes != FRAME_BYTES:
                raise RuntimeError("AEC returned an invalid capture frame")
            cleaned_pcm = cleaned_view.cast("B").tobytes()
            metrics, explicit_health = self._read_operational_metrics()
        except _AcousticProcessingFailure as exc:
            return self._handle_aec_failure(
                capture_frame,
                render_frames=tuple(collected_render),
                assistant_rendering=assistant_rendering,
                started_full_duplex=started_full_duplex,
                reason=exc.reason,
            )
        except Exception:
            return self._handle_aec_failure(
                capture_frame,
                render_frames=tuple(collected_render),
                assistant_rendering=assistant_rendering,
                started_full_duplex=started_full_duplex,
                reason=AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE,
            )

        cleaned_frame = replace(capture_frame, pcm16=cleaned_pcm)
        try:
            speech = self._vad(cleaned_frame)
            vad_ok = True
        except Exception:
            speech = False
            vad_ok = False

        monitor = self._isolation_monitor
        assert monitor is not None
        try:
            observation = monitor.observe(
                capture=capture_frame,
                render_frames=tuple(collected_render),
                assistant_rendering=assistant_rendering,
                near_end_speech=speech if vad_ok else False,
                native_processor_ok=True,
            )
        except Exception:
            return self._handle_aec_failure(
                capture_frame,
                render_frames=tuple(collected_render),
                assistant_rendering=assistant_rendering,
                started_full_duplex=started_full_duplex,
                reason=AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE,
            )

        self._observe_health(metrics, explicit_health=explicit_health)
        if observation.near_end_disposition is NearEndDisposition.FENCE:
            self._mark_degraded()
            self._clear_pending_replay()
            self._publish_safety(
                AcousticSafetyPath.HALF_DUPLEX,
                observation.safety.demotion_reason,
            )
        else:
            self._select_effective_safety(observation)

        if not assistant_rendering:
            admitted_frame = (
                cleaned_frame if self._health is AecHealth.HEALTHY else capture_frame
            )
            replay_pending = self._pending_replay_is_contiguous_with(
                capture_frame,
                observation=observation,
                vad_ok=vad_ok,
                speech=speech,
            )
            if replay_pending:
                self._vad_preroll.clear()
                while self._pending_replay:
                    self._on_admitted_frame(self._pending_replay.popleft())
                self._on_admitted_frame(admitted_frame)
                self._clear_pending_replay()
            else:
                self._clear_pending_replay()
                self._admit_idle_vad(admitted_frame, vad_ok=vad_ok, speech=speech)
            self._acknowledge(capture_frame, dsp_ok=True, vad_ok=vad_ok)
            return admitted_frame

        self._vad_preroll.clear()
        if not vad_ok or not speech:
            self._clear_pending_replay()
        elif not self._safety.admission_open:
            self._clear_pending_replay()
        else:
            self._admit_playback_frame(cleaned_frame, observation)
        self._acknowledge(capture_frame, dsp_ok=True, vad_ok=vad_ok)
        if not vad_ok or not speech or not self._safety.admission_open:
            return None
        return cleaned_frame

    def reset_for_device_route(self, clock_generation: int) -> None:
        """Bind a rebuilt route generation and close admission until rewarming."""

        if clock_generation <= self._active_clock_generation:
            raise ValueError("route reset must advance the clock generation")

        self._health = (
            AecHealth.WARMING if self._aec is not None else AecHealth.DEGRADED
        )
        self._healthy_observations = 0
        self._warming_observations = 0
        self._last_render_sequence = -1
        self._render_generation = None
        self._last_capture_sequence = None
        self._active_clock_generation = clock_generation
        self._capture_continuity_failed = False
        self._capture_failure_reason = None
        self._clear_pending_replay()
        self._vad_preroll.clear()
        self._publish_safety(
            AcousticSafetyPath.WARMING
            if self._aec is not None
            else AcousticSafetyPath.HALF_DUPLEX
        )
        if self._aec is not None:
            self._aec.reset()
        if self._isolation_monitor is not None:
            self._isolation_monitor.reset_for_route(clock_generation)

    def _read_operational_metrics(
        self,
    ) -> tuple[dict[str, float], AecHealth | None]:
        assert self._aec is not None
        return read_aec_metric_snapshot(self._aec)

    def _observe_health(
        self,
        metrics: Mapping[str, float],
        *,
        explicit_health: AecHealth | None,
    ) -> None:
        if explicit_health is AecHealth.WARMING:
            self._health = AecHealth.WARMING
            self._healthy_observations = 0
            self._warming_observations += 1
            if self._warming_observations >= self._warming_observation_limit:
                self._mark_degraded()
            return
        healthy = self._metrics_are_healthy(metrics)

        if not healthy:
            if self._health is AecHealth.HEALTHY:
                self._mark_degraded()
            else:
                self._healthy_observations = 0
                if self._health is AecHealth.WARMING:
                    self._warming_observations += 1
                    if self._warming_observations >= self._warming_observation_limit:
                        self._mark_degraded()
            return
        self._healthy_observations += 1
        if self._healthy_observations >= self._healthy_streak_required:
            self._health = AecHealth.HEALTHY
            self._warming_observations = 0

    def _metrics_are_healthy(self, values: Mapping[str, float]) -> bool:
        return (
            values["erle_db"] >= self._minimum_erle_db
            and 0.0 <= values["delay_ms"] <= 1_000.0
            and values["delay_estimate_available"] == 1.0
            and values["delay_estimate_refined"] == 1.0
            and 0.0 <= values["delay_age_blocks"] <= self._maximum_delay_age_blocks
            and values["clock_drift"] == 0.0
        )

    def _mark_degraded(self) -> None:
        self._health = AecHealth.DEGRADED
        self._healthy_observations = 0

    def _select_effective_safety(
        self,
        observation: AcousticIsolationObservation,
    ) -> None:
        reason = observation.safety.demotion_reason
        if reason not in _AEC_OVERRIDEABLE_ISOLATION_REASONS:
            self._mark_degraded()
            self._publish_safety(AcousticSafetyPath.HALF_DUPLEX, reason)
            return
        if self._health is AecHealth.HEALTHY:
            path = AcousticSafetyPath.AEC
        elif observation.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION:
            path = AcousticSafetyPath.ACOUSTIC_ISOLATION
        elif self._aec is not None and self._health is AecHealth.WARMING:
            path = AcousticSafetyPath.WARMING
        else:
            path = AcousticSafetyPath.HALF_DUPLEX
        effective_reason = (
            None
            if path in {AcousticSafetyPath.AEC, AcousticSafetyPath.ACOUSTIC_ISOLATION}
            else reason
        )
        self._publish_safety(path, effective_reason)

    def _pending_replay_is_contiguous_with(
        self,
        frame: AudioFrame,
        *,
        observation: AcousticIsolationObservation,
        vad_ok: bool,
        speech: bool,
    ) -> bool:
        if not vad_ok or not speech or not self._pending_replay:
            return False
        if (
            observation.near_end_disposition is NearEndDisposition.FENCE
            or observation.safety.demotion_reason
            not in _AEC_OVERRIDEABLE_ISOLATION_REASONS
        ):
            return False
        previous = self._pending_replay[-1]
        return (
            frame.clock_generation == previous.clock_generation
            and frame.sequence == previous.sequence + 1
            and frame.started_ns == previous.ended_ns
        )

    def _publish_safety(
        self,
        path: AcousticSafetyPath,
        reason: AcousticDemotionReason | None = None,
    ) -> None:
        opened = path in {
            AcousticSafetyPath.AEC,
            AcousticSafetyPath.ACOUSTIC_ISOLATION,
        }
        self._safety = AcousticSafetySnapshot(
            path,
            DuplexMode.FULL_DUPLEX if opened else DuplexMode.HALF_DUPLEX,
            self._active_clock_generation,
            opened,
            None if opened else reason,
        )

    def _admit_playback_frame(
        self,
        frame: AudioFrame,
        observation: AcousticIsolationObservation,
    ) -> None:
        disposition = observation.near_end_disposition
        if disposition is NearEndDisposition.PENDING:
            if len(self._pending_replay) >= _PENDING_REPLAY_FRAMES:
                self._clear_pending_replay()
                self._mark_degraded()
                self._publish_safety(
                    AcousticSafetyPath.HALF_DUPLEX,
                    AcousticDemotionReason.AMBIGUOUS_CORRELATION,
                )
                return
            self._pending_replay.append(frame)
            first = self._pending_replay[0]
            watermark = PendingClassificationWatermark(
                clock_generation=frame.clock_generation,
                first_sequence=first.sequence,
                first_started_ns=first.started_ns,
                through_sequence=frame.sequence,
            )
            self._pending_classification = watermark
            self._on_classification_changed(watermark)
            return
        if disposition is NearEndDisposition.ADMIT:
            if self._pending_replay:
                self._pending_replay.append(frame)
                while self._pending_replay:
                    self._on_admitted_frame(self._pending_replay.popleft())
                self._clear_pending_replay()
            else:
                self._on_admitted_frame(frame)
            return
        self._clear_pending_replay()

    def _capture_continuity(self, frame: AudioFrame) -> bool | None:
        if frame.clock_generation < self._active_clock_generation:
            return None
        if self._capture_continuity_failed:
            return False
        if frame.clock_generation > self._active_clock_generation:
            self._capture_continuity_failed = True
            self._capture_failure_reason = AcousticDemotionReason.ROUTE_MISMATCH
            return False
        if self._last_capture_sequence is None:
            self._last_capture_sequence = frame.sequence
            return True
        if frame.sequence != self._last_capture_sequence + 1:
            self._capture_continuity_failed = True
            self._capture_failure_reason = AcousticDemotionReason.CAPTURE_SEQUENCE_GAP
            return False
        self._last_capture_sequence = frame.sequence
        return True

    def _handle_aec_failure(
        self,
        frame: AudioFrame,
        *,
        render_frames: tuple[AudioFrame, ...],
        assistant_rendering: bool,
        started_full_duplex: bool,
        reason: AcousticDemotionReason,
    ) -> AudioFrame | None:
        self._close_native_path(
            frame,
            render_frames=render_frames,
            assistant_rendering=assistant_rendering,
            reason_override=reason,
        )
        if assistant_rendering:
            self._acknowledge(
                frame,
                dsp_ok=not started_full_duplex,
                vad_ok=not started_full_duplex,
            )
            return None
        return self._run_vad_and_admit(frame, admit=True)

    def _close_native_path(
        self,
        frame: AudioFrame,
        *,
        render_frames: tuple[AudioFrame, ...],
        assistant_rendering: bool,
        reason_override: AcousticDemotionReason | None = None,
    ) -> None:
        self._mark_degraded()
        self._clear_pending_replay()
        self._vad_preroll.clear()
        reason = reason_override or AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE
        if self._isolation_monitor is not None:
            try:
                observation = self._isolation_monitor.observe(
                    capture=frame,
                    render_frames=render_frames,
                    assistant_rendering=assistant_rendering,
                    near_end_speech=False,
                    native_processor_ok=False,
                )
                if reason_override is None:
                    reason = (
                        observation.safety.demotion_reason
                        or AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE
                    )
            except Exception:
                if reason_override is None:
                    reason = AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE
        self._publish_safety(AcousticSafetyPath.HALF_DUPLEX, reason)

    def _clear_pending_replay(self) -> None:
        self._pending_replay.clear()
        if self._pending_classification is None:
            return
        self._pending_classification = None
        self._on_classification_changed(None)

    def _acknowledge(
        self,
        frame: AudioFrame,
        *,
        dsp_ok: bool,
        vad_ok: bool,
    ) -> None:
        self._on_processed(
            frame.sequence,
            frame.clock_generation,
            dsp_ok,
            vad_ok,
        )

    def _run_vad_and_admit(
        self,
        frame: AudioFrame,
        *,
        admit: bool,
    ) -> AudioFrame:
        try:
            speech = self._vad(frame)
        except Exception:
            self._vad_preroll.clear()
            self._acknowledge(frame, dsp_ok=True, vad_ok=False)
            return frame
        self._acknowledge(frame, dsp_ok=True, vad_ok=True)
        if admit:
            self._admit_idle_vad(frame, vad_ok=True, speech=speech)
        return frame

    def _admit_idle_vad(
        self,
        frame: AudioFrame,
        *,
        vad_ok: bool,
        speech: bool,
    ) -> None:
        if not vad_ok:
            self._vad_preroll.clear()
            return
        if not speech:
            self._vad_preroll.append(frame)
            return
        while self._vad_preroll:
            self._on_admitted_frame(
                replace(self._vad_preroll.popleft(), speech_started_ns=frame.started_ns)
            )
        self._on_admitted_frame(frame)


def _energy_vad(frame: AudioFrame) -> bool:
    samples = array("h")
    samples.frombytes(frame.pcm16)
    if sys.byteorder != "little":
        samples.byteswap()
    return sum(abs(value) for value in samples) >= len(samples) * 250


def _timing_failure_reason(
    evidence: AecDelayEvidence | None,
    *,
    duration_ns: int,
) -> AcousticDemotionReason | None:
    if evidence is None:
        return AcousticDemotionReason.MISSING_TIMING
    if duration_ns != FRAME_DURATION_NS or evidence.occupancy_bounded is not True:
        return AcousticDemotionReason.UNBOUNDED_TIMING
    if evidence.timing_discontinuity or evidence.clock_drift or evidence.status_flags:
        return AcousticDemotionReason.TIMING_DISCONTINUITY
    if not 0 <= evidence.delay_ms <= 1_000:
        return AcousticDemotionReason.UNBOUNDED_TIMING
    return None


def _is_exact_pcm_frame(pcm16: object) -> bool:
    try:
        view = memoryview(pcm16)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return view.c_contiguous and view.nbytes == FRAME_BYTES


def _ignore_frame(_frame: AudioFrame) -> None:
    return None


def _ignore_processed(
    _sequence: int,
    _clock_generation: int,
    _dsp_ok: bool,
    _vad_ok: bool,
) -> None:
    return None


def _ignore_classification(
    _watermark: PendingClassificationWatermark | None,
) -> None:
    return None
