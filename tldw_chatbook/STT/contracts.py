"""Dependency-free, provider-neutral speech-to-text contracts.

The values in this module are intentionally limited to the Python standard
library. In particular, importing the contracts must not initialize an STT
runtime, application configuration, artifact acquisition, or legacy
transcription code.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.leases import ArtifactLeaseKey


# A generous but finite in-process PCM payload boundary. Longer recordings
# should use FileAudioSource or be split by the app-owned executor.
MAX_BUFFER_AUDIO_BYTES = 64 * 1024 * 1024

_VALID_SAMPLE_WIDTHS = frozenset({1, 2, 3, 4})
_LANGUAGE_PATTERN = re.compile(r"(?:auto|[a-z]{2,3}(?:-[a-z0-9]{1,8})*)")
_DETAIL_CODE_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*")
_MAX_DETAIL_CODE_LENGTH = 128


class TranscriptionTask(str, Enum):
    """Provider-neutral transcription operations."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class InputKind(str, Enum):
    """Supported request source kinds."""

    FILE = "file"
    BUFFER = "buffer"


class TimestampGranularity(str, Enum):
    """Timestamp detail produced or requested."""

    NONE = "none"
    SEGMENT = "segment"
    WORD = "word"


class CancellationGranularity(str, Enum):
    """The points at which a provider can observe cancellation."""

    NONE = "none"
    BEFORE_EXECUTION = "before_execution"
    SEGMENT_BOUNDARY = "segment_boundary"
    ACTIVE = "active"


class TranscriptionPhase(str, Enum):
    """Stable phases used for progress reporting."""

    QUEUED = "queued"
    PREPARING = "preparing"
    LOADING = "loading"
    TRANSCRIBING = "transcribing"
    POST_PROCESSING = "post-processing"
    SAVING = "saving"
    COMPLETE = "complete"


class LanguageInputMode(str, Enum):
    """How a provider interprets caller-supplied language."""

    ENFORCED = "enforced"
    ROUTING_ASSERTION = "routing_assertion"
    AUTOMATIC = "automatic"
    AUTOMATIC_ONLY = "automatic_only"


class ExecutionDevice(str, Enum):
    """Provider-neutral execution device selection."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"


class TranscriptionWarningCode(str, Enum):
    """Stable non-fatal warning codes."""

    REQUESTED_LANGUAGE_NOT_ENFORCED = "requested_language_not_enforced"


def _require_string(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_optional_string(
    value: object,
    field_name: str,
) -> None:
    if value is not None:
        _require_string(value, field_name)


def _require_enum(value: object, enum_type: type[Enum], field_name: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{field_name} must be a {enum_type.__name__}")


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")


def _require_finite_nonnegative(
    value: object,
    field_name: str,
    *,
    allow_none: bool = False,
) -> None:
    if allow_none and value is None:
        return
    if type(value) not in (int, float):
        raise TypeError(f"{field_name} must be a number")
    number = cast("int | float", value)
    if number < 0:
        raise ValueError(f"{field_name} must be finite and nonnegative")
    if type(number) is float and not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite and nonnegative")


def _require_language(
    value: object,
    field_name: str,
    *,
    allow_none: bool = False,
    allow_empty: bool = False,
) -> None:
    if allow_none and value is None:
        return
    _require_string(value, field_name, allow_empty=allow_empty)
    language = cast(str, value)
    if language == "" and allow_empty:
        return
    if not _LANGUAGE_PATTERN.fullmatch(language):
        raise ValueError(
            f"{field_name} must be 'auto' or a canonical lower-case language tag"
        )


def _is_artifact_lease_key(value: object) -> bool:
    """Check the exact lease key type without importing its package.

    Importing ``tldw_chatbook.Model_Artifacts.leases`` normally executes the
    package ``__init__`` first, which pulls acquisition and service modules
    into this otherwise dependency-free boundary. A genuine key can only
    exist after its defining module has already been loaded by the caller, so
    consulting ``sys.modules`` preserves exact type checking without causing
    that import graph.
    """

    lease_module = sys.modules.get("tldw_chatbook.Model_Artifacts.leases")
    lease_key_type = (
        getattr(lease_module, "ArtifactLeaseKey", None)
        if lease_module is not None
        else None
    )
    return lease_key_type is not None and type(value) is lease_key_type


@dataclass(frozen=True, slots=True)
class FileAudioSource:
    """A filesystem-backed audio source."""

    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path")


@dataclass(frozen=True, slots=True)
class BufferAudioSource:
    """A bounded, interleaved PCM audio buffer.

    ``audio`` must contain between 1 and ``MAX_BUFFER_AUDIO_BYTES`` bytes.
    It must end on a complete interleaved frame. Positive sample rates and
    channel counts, and sample widths of 1, 2, 3, or 4 bytes, are accepted.
    Provider capability checks remain a separate layer.
    """

    audio: bytes = field(repr=False)
    sample_rate: int
    channels: int = 1
    sample_width: int = 2

    def __post_init__(self) -> None:
        if type(self.audio) is not bytes:
            raise TypeError("audio must be bytes")
        if not self.audio:
            raise ValueError("audio must not be empty")
        if len(self.audio) > MAX_BUFFER_AUDIO_BYTES:
            raise ValueError(
                f"audio must not exceed {MAX_BUFFER_AUDIO_BYTES} bytes"
            )
        if type(self.sample_rate) is not int:
            raise TypeError("sample_rate must be an int")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if type(self.channels) is not int:
            raise TypeError("channels must be an int")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if type(self.sample_width) is not int:
            raise TypeError("sample_width must be an int")
        if self.sample_width not in _VALID_SAMPLE_WIDTHS:
            raise ValueError("sample_width must be one of 1, 2, 3, or 4")
        if len(self.audio) % (self.channels * self.sample_width) != 0:
            raise ValueError("audio must contain complete interleaved PCM frames")


@dataclass(frozen=True, slots=True)
class PrivacyRequirements:
    """Privacy constraints that routing and execution must honor."""

    allow_remote_processing: bool = False
    allow_disk_staging: bool = True

    def __post_init__(self) -> None:
        _require_bool(self.allow_remote_processing, "allow_remote_processing")
        _require_bool(self.allow_disk_staging, "allow_disk_staging")


@runtime_checkable
class CancellationToken(Protocol):
    """Cooperative cancellation observed by coordinators and providers."""

    def is_cancelled(self) -> bool:
        """Return whether the request has been cancelled."""

        ...


@runtime_checkable
class ProgressSink(Protocol):
    """Callback receiving immutable transcription progress events."""

    def __call__(self, event: TranscriptionProgress) -> None:
        """Consume one progress event."""

        ...


@dataclass(frozen=True, slots=True)
class TranscriptionProgress:
    """One stable provider-neutral progress event."""

    attempt_id: str
    batch_id: str | None
    job_id: str | None
    phase: TranscriptionPhase
    fraction: float | None = None
    detail_code: str | None = None

    def __post_init__(self) -> None:
        _require_string(self.attempt_id, "attempt_id")
        _require_optional_string(self.batch_id, "batch_id")
        _require_optional_string(self.job_id, "job_id")
        _require_enum(self.phase, TranscriptionPhase, "phase")
        if self.fraction is not None:
            _require_finite_nonnegative(self.fraction, "fraction")
            if self.fraction > 1:
                raise ValueError("fraction must be between 0 and 1")
        if self.detail_code is not None:
            _require_string(self.detail_code, "detail_code")
            if (
                len(self.detail_code) > _MAX_DETAIL_CODE_LENGTH
                or not _DETAIL_CODE_PATTERN.fullmatch(self.detail_code)
            ):
                raise ValueError(
                    "detail_code must be a stable lower-case code of at most "
                    f"{_MAX_DETAIL_CODE_LENGTH} characters"
                )


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    """A provider-neutral transcription request."""

    attempt_id: str
    source: FileAudioSource | BufferAudioSource
    batch_id: str | None = None
    job_id: str | None = None
    retry_of_attempt_id: str | None = None
    retry_of_job_id: str | None = None
    provider_id: str = "default"
    model_id: str | None = None
    language: str | None = None
    task: TranscriptionTask = TranscriptionTask.TRANSCRIBE
    precision: str | None = None
    device: ExecutionDevice = ExecutionDevice.AUTO
    timestamps: TimestampGranularity = TimestampGranularity.SEGMENT
    diarization: bool = False
    vad: bool = False
    privacy: PrivacyRequirements = field(default_factory=PrivacyRequirements)
    cancellation: CancellationToken | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    progress: ProgressSink | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        _require_string(self.attempt_id, "attempt_id")
        if type(self.source) not in (FileAudioSource, BufferAudioSource):
            raise TypeError("source must be a FileAudioSource or BufferAudioSource")
        _require_optional_string(self.batch_id, "batch_id")
        _require_optional_string(self.job_id, "job_id")
        _require_optional_string(
            self.retry_of_attempt_id,
            "retry_of_attempt_id",
        )
        _require_optional_string(self.retry_of_job_id, "retry_of_job_id")
        _require_string(self.provider_id, "provider_id")
        _require_optional_string(self.model_id, "model_id")
        _require_language(
            self.language,
            "language",
            allow_none=True,
            allow_empty=True,
        )
        _require_enum(self.task, TranscriptionTask, "task")
        _require_optional_string(self.precision, "precision")
        _require_enum(self.device, ExecutionDevice, "device")
        _require_enum(self.timestamps, TimestampGranularity, "timestamps")
        _require_bool(self.diarization, "diarization")
        _require_bool(self.vad, "vad")
        if type(self.privacy) is not PrivacyRequirements:
            raise TypeError("privacy must be a PrivacyRequirements")
        if self.cancellation is not None and (
            not isinstance(self.cancellation, CancellationToken)
            or not callable(self.cancellation.is_cancelled)
        ):
            raise TypeError("cancellation must implement CancellationToken")
        if self.progress is not None and (
            not isinstance(self.progress, ProgressSink)
            or not callable(self.progress)
            or not callable(getattr(self.progress, "__call__", None))
        ):
            raise TypeError("progress must implement ProgressSink")


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """One timestamped transcript segment."""

    start_seconds: float
    end_seconds: float
    text: str
    speaker: str | None = None

    def __post_init__(self) -> None:
        _require_finite_nonnegative(self.start_seconds, "start_seconds")
        _require_finite_nonnegative(self.end_seconds, "end_seconds")
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must not precede start_seconds")
        _require_string(self.text, "text", allow_empty=True)
        if self.speaker is not None:
            _require_string(self.speaker, "speaker", allow_empty=True)


@dataclass(frozen=True, slots=True)
class TranscriptionProvenance:
    """Complete identity and execution provenance for a transcript."""

    schema_version: int
    attempt_id: str
    batch_id: str | None
    job_id: str | None
    retry_of_attempt_id: str | None
    retry_of_job_id: str | None
    provider_id: str
    model_id: str
    artifact_root: ArtifactLeaseKey | None
    artifact_dependencies: tuple[ArtifactLeaseKey, ...]
    precision: str
    requested_device: ExecutionDevice
    effective_device: ExecutionDevice
    requested_language: str
    effective_language: str
    detected_language: str | None
    task: TranscriptionTask

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an int")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        _require_string(self.attempt_id, "attempt_id")
        _require_optional_string(self.batch_id, "batch_id")
        _require_optional_string(self.job_id, "job_id")
        _require_optional_string(
            self.retry_of_attempt_id,
            "retry_of_attempt_id",
        )
        _require_optional_string(self.retry_of_job_id, "retry_of_job_id")
        _require_string(self.provider_id, "provider_id")
        _require_string(self.model_id, "model_id")
        if self.artifact_root is not None and not _is_artifact_lease_key(
            self.artifact_root
        ):
            raise TypeError("artifact_root must be an ArtifactLeaseKey")
        if type(self.artifact_dependencies) is not tuple:
            raise TypeError("artifact_dependencies must be a tuple")
        if not all(
            _is_artifact_lease_key(dependency)
            for dependency in self.artifact_dependencies
        ):
            raise TypeError(
                "artifact_dependencies must contain only ArtifactLeaseKey values"
            )
        _require_string(self.precision, "precision")
        _require_enum(
            self.requested_device,
            ExecutionDevice,
            "requested_device",
        )
        _require_enum(
            self.effective_device,
            ExecutionDevice,
            "effective_device",
        )
        _require_language(self.requested_language, "requested_language")
        _require_language(self.effective_language, "effective_language")
        _require_language(
            self.detected_language,
            "detected_language",
            allow_none=True,
        )
        _require_enum(self.task, TranscriptionTask, "task")


@dataclass(frozen=True, slots=True)
class ProducedCapabilities:
    """Capabilities represented by one concrete result."""

    timestamps: TimestampGranularity
    punctuation: bool
    capitalization: bool
    vad: bool
    diarization: bool

    def __post_init__(self) -> None:
        _require_enum(self.timestamps, TimestampGranularity, "timestamps")
        _require_bool(self.punctuation, "punctuation")
        _require_bool(self.capitalization, "capitalization")
        _require_bool(self.vad, "vad")
        _require_bool(self.diarization, "diarization")


@dataclass(frozen=True, slots=True)
class PipelineCapabilities:
    """Capabilities added by the composed application pipeline."""

    timestamps: frozenset[TimestampGranularity] = frozenset()
    vad: bool = False
    diarization: bool = False
    requires_disk_staging_for_buffer: bool = False

    def __post_init__(self) -> None:
        if type(self.timestamps) is not frozenset:
            raise TypeError("timestamps must be a frozenset")
        if not all(
            type(granularity) is TimestampGranularity
            for granularity in self.timestamps
        ):
            raise TypeError(
                "timestamps must contain only TimestampGranularity values"
            )
        _require_bool(self.vad, "vad")
        _require_bool(self.diarization, "diarization")
        _require_bool(
            self.requires_disk_staging_for_buffer,
            "requires_disk_staging_for_buffer",
        )


@dataclass(frozen=True, slots=True)
class TranscriptionTimings:
    """Provider-neutral stage timings in seconds."""

    preparation_seconds: float | None = None
    model_load_seconds: float | None = None
    inference_seconds: float | None = None
    postprocess_seconds: float | None = None
    total_seconds: float | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("preparation_seconds", self.preparation_seconds),
            ("model_load_seconds", self.model_load_seconds),
            ("inference_seconds", self.inference_seconds),
            ("postprocess_seconds", self.postprocess_seconds),
            ("total_seconds", self.total_seconds),
        ):
            _require_finite_nonnegative(
                value,
                field_name,
                allow_none=True,
            )


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """A normalized transcription result and its complete provenance."""

    text: str
    segments: tuple[TranscriptionSegment, ...]
    provenance: TranscriptionProvenance
    produced_capabilities: ProducedCapabilities
    duration_seconds: float | None
    timings: TranscriptionTimings
    warnings: tuple[TranscriptionWarningCode, ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.text, "text", allow_empty=True)
        if type(self.segments) is not tuple:
            raise TypeError("segments must be a tuple")
        if not all(type(segment) is TranscriptionSegment for segment in self.segments):
            raise TypeError("segments must contain only TranscriptionSegment values")
        for previous, current in zip(self.segments, self.segments[1:]):
            if current.start_seconds < previous.start_seconds:
                raise ValueError("segments must be ordered by start_seconds")
        if type(self.provenance) is not TranscriptionProvenance:
            raise TypeError("provenance must be a TranscriptionProvenance")
        if type(self.produced_capabilities) is not ProducedCapabilities:
            raise TypeError(
                "produced_capabilities must be a ProducedCapabilities"
            )
        _require_finite_nonnegative(
            self.duration_seconds,
            "duration_seconds",
            allow_none=True,
        )
        if type(self.timings) is not TranscriptionTimings:
            raise TypeError("timings must be a TranscriptionTimings")
        if type(self.warnings) is not tuple:
            raise TypeError("warnings must be a tuple")
        if not all(
            type(warning) is TranscriptionWarningCode for warning in self.warnings
        ):
            raise TypeError(
                "warnings must contain only TranscriptionWarningCode values"
            )

        has_segments = bool(self.segments)
        has_timestamps = (
            self.produced_capabilities.timestamps is not TimestampGranularity.NONE
        )
        if has_segments != has_timestamps:
            raise ValueError(
                "segment presence must agree with produced timestamp granularity"
            )
        if (
            not self.produced_capabilities.diarization
            and any(segment.speaker is not None for segment in self.segments)
        ):
            raise ValueError(
                "speaker labels require produced diarization capability"
            )


__all__ = [
    "MAX_BUFFER_AUDIO_BYTES",
    "BufferAudioSource",
    "CancellationGranularity",
    "CancellationToken",
    "ExecutionDevice",
    "FileAudioSource",
    "InputKind",
    "LanguageInputMode",
    "PipelineCapabilities",
    "PrivacyRequirements",
    "ProducedCapabilities",
    "ProgressSink",
    "TimestampGranularity",
    "TranscriptionPhase",
    "TranscriptionProgress",
    "TranscriptionProvenance",
    "TranscriptionRequest",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionTask",
    "TranscriptionTimings",
    "TranscriptionWarningCode",
]
