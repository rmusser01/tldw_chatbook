"""Dependency-free contracts for speculative duplex audio processing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class AecHealth(str, Enum):
    """Coarse echo-cancellation health exposed to pipeline consumers."""

    WARMING = "warming"
    HEALTHY = "healthy"
    DEGRADED = "degraded"


class DuplexMode(str, Enum):
    """Effective acoustic-interruption capability for the current route."""

    FULL_DUPLEX = "full-duplex"
    HALF_DUPLEX = "half-duplex"


class AcousticSafetyPath(str, Enum):
    """Proof path governing playback-period speech admission."""

    WARMING = "warming"
    AEC = "aec"
    ACOUSTIC_ISOLATION = "acoustic-isolation"
    HALF_DUPLEX = "half-duplex"


class AcousticDemotionReason(str, Enum):
    """Closed reasons that invalidate an acoustic-safety proof."""

    AMBIGUOUS_CORRELATION = "ambiguous-correlation"
    CAPTURE_SEQUENCE_GAP = "capture-sequence-gap"
    CORRELATED_RENDER = "correlated-render"
    INAUDIBLE_RENDER = "inaudible-render"
    INVALID_CAPTURE_PCM = "invalid-capture-pcm"
    INVALID_RENDER_PCM = "invalid-render-pcm"
    MISSING_RENDER_REFERENCE = "missing-render-reference"
    MISSING_TIMING = "missing-timing"
    NATIVE_PROCESSOR_FAILURE = "native-processor-failure"
    RENDER_REFERENCE_FAILURE = "render-reference-failure"
    RENDER_REFERENCE_GAP = "render-reference-gap"
    RENDER_REFERENCE_OVERFLOW = "render-reference-overflow"
    RENDER_SEQUENCE_GAP = "render-sequence-gap"
    ROUTE_MISMATCH = "route-mismatch"
    SATURATION = "saturation"
    TIMING_DISCONTINUITY = "timing-discontinuity"
    UNBOUNDED_TIMING = "unbounded-timing"


class NearEndDisposition(str, Enum):
    """Content-free classification of one playback-period speech event."""

    PENDING = "pending"
    ADMIT = "admit"
    FENCE = "fence"


@dataclass(frozen=True, slots=True)
class AcousticSafetySnapshot:
    """Content-free acoustic admission state for one device route."""

    path: AcousticSafetyPath
    mode: DuplexMode
    route_generation: int
    admission_open: bool
    demotion_reason: AcousticDemotionReason | None

    def __post_init__(self) -> None:
        if not isinstance(self.path, AcousticSafetyPath):
            raise TypeError("path must be an AcousticSafetyPath")
        if not isinstance(self.mode, DuplexMode):
            raise TypeError("mode must be a DuplexMode")
        if type(self.route_generation) is not int:
            raise TypeError("route generation must be an integer")
        if self.demotion_reason is not None and not isinstance(
            self.demotion_reason,
            AcousticDemotionReason,
        ):
            raise TypeError("demotion reason must be an AcousticDemotionReason")
        full_duplex_path = self.path in {
            AcousticSafetyPath.AEC,
            AcousticSafetyPath.ACOUSTIC_ISOLATION,
        }
        if (self.mode is DuplexMode.FULL_DUPLEX) is not full_duplex_path:
            raise ValueError("acoustic safety path and duplex mode are inconsistent")
        if self.admission_open is not full_duplex_path:
            raise ValueError("acoustic safety path and admission are inconsistent")
        if full_duplex_path and self.demotion_reason is not None:
            raise ValueError("open admission cannot carry a demotion reason")
        if self.route_generation < 0:
            raise ValueError("route generation must be non-negative")


@dataclass(frozen=True, slots=True)
class AcousticIsolationObservation:
    """Safety state plus classification of the current near-end event."""

    safety: AcousticSafetySnapshot
    near_end_disposition: NearEndDisposition | None

    def __post_init__(self) -> None:
        if not isinstance(self.safety, AcousticSafetySnapshot):
            raise TypeError("safety must be an AcousticSafetySnapshot")
        if self.near_end_disposition is not None and not isinstance(
            self.near_end_disposition,
            NearEndDisposition,
        ):
            raise TypeError("near_end_disposition must be a NearEndDisposition")
        if (
            self.near_end_disposition is NearEndDisposition.FENCE
            and self.safety.admission_open
        ):
            raise ValueError("fenced near-end audio cannot keep admission open")


class RouteKind(str, Enum):
    """Content-free portion of the device route that changed."""

    INPUT = "input"
    OUTPUT = "output"
    DUPLEX = "duplex"


class CaptureDrainError(RuntimeError):
    """Raised when a causal capture/AEC/VAD boundary cannot be proven."""


@dataclass(frozen=True, slots=True)
class AecDelayEvidence:
    """Content-free hardware timing used for one render/capture DSP step."""

    observed_ns: int
    capture_adc_ns: int
    render_dac_ns: int
    delay_ms: int
    capture_occupancy_frames: int
    render_occupancy_frames: int
    occupancy_bounded: bool
    timing_discontinuity: bool
    clock_drift: bool
    status_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(self.observed_ns, self.capture_adc_ns, self.render_dac_ns) < 0:
            raise ValueError("AEC hardware timestamps must be non-negative")
        if not 0 <= self.delay_ms <= 1_000:
            raise ValueError("AEC delay must be between 0 and 1000 milliseconds")
        if min(self.capture_occupancy_frames, self.render_occupancy_frames) < 0:
            raise ValueError("AEC ring occupancy must be non-negative")
        if any(not flag for flag in self.status_flags):
            raise ValueError("AEC status flag names must be nonempty")


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """One ordered PCM16 frame on the paired capture/DSP callback timeline.

    Capture frames identify their actual-output partner with
    ``render_reference_sequence``. Render-reference frames use that same callback
    sequence even when the device scheduled silence and no TTS submission committed.
    ``committed_render`` is present only when that callback submitted identified
    output to the device, including zero-valued PCM.
    """

    sequence: int
    started_ns: int
    ended_ns: int
    pcm16: bytes
    clock_generation: int = 0
    discontinuity: bool = False
    delay_evidence: AecDelayEvidence | None = None
    render_reference_sequence: int | None = None
    assistant_rendering: bool | None = None
    committed_render: RenderBoundary | None = None
    # Set on VAD-negative preroll to its triggering positive speech onset.
    # It may follow this context frame's ended_ns; PCM/seal timestamps stay intact.
    speech_started_ns: int | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("audio frame sequence must be non-negative")
        if self.ended_ns <= self.started_ns:
            raise ValueError("audio frame duration must be positive")
        if self.clock_generation < 0:
            raise ValueError("audio frame clock generation must be non-negative")
        if self.render_reference_sequence is not None and (
            self.render_reference_sequence < 0
        ):
            raise ValueError("render reference sequence must be non-negative")
        if self.committed_render is not None and not isinstance(
            self.committed_render, RenderBoundary
        ):
            raise TypeError("committed render must be a RenderBoundary or None")
        if self.speech_started_ns is not None:
            if type(self.speech_started_ns) is not int:
                raise TypeError("speech onset must be an integer or None")
            if self.speech_started_ns < self.started_ns:
                raise ValueError("speech onset cannot precede its context frame")

    @property
    def duration_ns(self) -> int:
        """Return the frame's monotonic duration in nanoseconds."""

        return self.ended_ns - self.started_ns


@dataclass(frozen=True, slots=True)
class DrainReceipt:
    """Acknowledged capture and processing watermarks through a render boundary."""

    capture_watermark_ns: int
    capture_sequence: int
    dsp_sequence: int
    vad_sequence: int
    clock_generation: int = 0


@dataclass(frozen=True, slots=True)
class RenderSubmission:
    """Queued output identity, without a speculative completion timestamp."""

    generation: int
    output_epoch: int
    submission_id: int


@dataclass(frozen=True, slots=True)
class RenderBoundary:
    """Actual callback DAC completion for one live output submission."""

    generation: int
    output_epoch: int
    submission_id: int
    ended_ns: int


@dataclass(frozen=True, slots=True)
class DeviceRouteChanged:
    """Content-free fence notification for an invalidated device clock."""

    old_clock_generation: int
    route_kind: RouteKind


@dataclass(frozen=True, slots=True)
class DuplexBufferOccupancy:
    """Content-free counts for every bounded callback-owned ring."""

    capture_frames: int
    render_frames: int
    render_reference_frames: int
    control_events: int

    def __post_init__(self) -> None:
        if (
            min(
                self.capture_frames,
                self.render_frames,
                self.render_reference_frames,
                self.control_events,
            )
            < 0
        ):
            raise ValueError("duplex buffer counts must be non-negative")


class DuplexAudioPort(Protocol):
    """Minimal asynchronous audio boundary required by duplex orchestration."""

    async def abort_output(self) -> None: ...

    async def drain_capture_through(
        self,
        render_boundary_ns: int,
    ) -> DrainReceipt: ...
