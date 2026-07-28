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
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable


# A generous but finite in-process PCM payload boundary. Longer recordings
# should use FileAudioSource or be split by the app-owned executor.
MAX_BUFFER_AUDIO_BYTES = 64 * 1024 * 1024

_VALID_SAMPLE_WIDTHS = frozenset({1, 2, 3, 4})
_LANGUAGE_PATTERN = re.compile(r"(?:auto|[a-z]{2,3}(?:-[a-z0-9]{1,8})*)")
_DETAIL_CODE_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*")
_MAX_DETAIL_CODE_LENGTH = 128


class _ArtifactIdentity(Protocol):
    """Runtime-resolvable structural annotation for an artifact lease key."""

    @property
    def artifact_id(self) -> str:
        """Return the stable artifact identifier."""

        ...

    @property
    def revision(self) -> str:
        """Return the immutable artifact revision."""

        ...

    @property
    def variant(self) -> str:
        """Return the artifact variant."""

        ...


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


class DeviceFailureOrigin(str, Enum):
    """Stable origins used to decide whether a device retry is safe."""

    EXECUTION_PROVIDER_INITIALIZATION = "execution_provider_initialization"
    INFERENCE = "inference"
    ENGINE_CRASH = "engine_crash"


class TranscriptionWarningCode(str, Enum):
    """Stable non-fatal warning codes."""

    REQUESTED_LANGUAGE_NOT_ENFORCED = "requested_language_not_enforced"


class TranscriptionAction(str, Enum):
    """Coordinator-owned actions that may create a future request."""

    INSTALL_MODEL = "install_model"
    CHOOSE_INSTALLED_MODEL = "choose_installed_model"
    RETRY_SAME_CONFIGURATION = "retry_same_configuration"
    RETRY_WITH_FASTER_WHISPER = "retry_with_faster_whisper"
    CHANGE_LANGUAGE_TO_AUTO = "change_language_to_auto"


class TranscriptionFailureCode(str, Enum):
    """Stable provider-neutral transcription failure codes."""

    MODEL_NOT_INSTALLED = "model_not_installed"
    ARTIFACT_CORRUPT = "artifact_corrupt"
    ARTIFACT_INCOMPATIBLE = "artifact_incompatible"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_REMOVED = "provider_removed"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    INSUFFICIENT_DISK_SPACE = "insufficient_disk_space"
    INSUFFICIENT_MEMORY = "insufficient_memory"
    INFERENCE_FAILED = "inference_failed"
    ENGINE_CRASHED = "engine_crashed"
    CANCELLED = "cancelled"


TRANSCRIPTION_FAILURE_CONTRACT = MappingProxyType(
    {
        TranscriptionFailureCode.MODEL_NOT_INSTALLED: (
            "The selected speech-to-text model is not installed.",
            False,
        ),
        TranscriptionFailureCode.ARTIFACT_CORRUPT: (
            "The installed speech-to-text model failed integrity verification.",
            False,
        ),
        TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE: (
            "The installed speech-to-text model is incompatible with this runtime.",
            False,
        ),
        TranscriptionFailureCode.PROVIDER_UNAVAILABLE: (
            "The selected speech-to-text provider is unavailable.",
            True,
        ),
        TranscriptionFailureCode.PROVIDER_REMOVED: (
            "The selected speech-to-text provider is no longer supported.",
            False,
        ),
        TranscriptionFailureCode.UNSUPPORTED_LANGUAGE: (
            "The selected speech-to-text model does not support the requested language.",
            False,
        ),
        TranscriptionFailureCode.UNSUPPORTED_CAPABILITY: (
            "The selected speech-to-text model does not support the requested capability.",
            False,
        ),
        TranscriptionFailureCode.INSUFFICIENT_DISK_SPACE: (
            "There is not enough disk space to prepare this transcription.",
            False,
        ),
        TranscriptionFailureCode.INSUFFICIENT_MEMORY: (
            "There is not enough memory to run this transcription.",
            False,
        ),
        TranscriptionFailureCode.INFERENCE_FAILED: (
            "Speech-to-text inference failed.",
            False,
        ),
        TranscriptionFailureCode.ENGINE_CRASHED: (
            "The speech-to-text engine stopped unexpectedly.",
            True,
        ),
        TranscriptionFailureCode.CANCELLED: (
            "The transcription was cancelled.",
            True,
        ),
    }
)


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

    path: Path = field(repr=False)

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
            raise ValueError(f"audio must not exceed {MAX_BUFFER_AUDIO_BYTES} bytes")
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
            if len(
                self.detail_code
            ) > _MAX_DETAIL_CODE_LENGTH or not _DETAIL_CODE_PATTERN.fullmatch(
                self.detail_code
            ):
                raise ValueError(
                    "detail_code must be a stable lower-case code of at most "
                    f"{_MAX_DETAIL_CODE_LENGTH} characters"
                )

    def __repr__(self) -> str:
        """Render progress without exposing caller-controlled text or identities."""

        return (
            f"{type(self).__name__}(phase={self.phase.value!r}, "
            f"fraction={self.fraction!r})"
        )

    def __str__(self) -> str:
        """Render a concise progress label from validated stable values only."""

        rendered = self.phase.value
        if self.fraction is not None:
            rendered = f"{rendered}: {self.fraction:.0%}"
        return rendered


@dataclass(frozen=True, slots=True)
class TranscriptionFailure:
    """An immutable failure envelope with no caller-controlled explanation."""

    code: TranscriptionFailureCode
    attempt_id: str
    batch_id: str | None
    job_id: str | None
    phase: TranscriptionPhase
    provider_id: str
    model_id: str
    artifact_root: _ArtifactIdentity | None
    precision: str
    requested_device: ExecutionDevice
    effective_device: ExecutionDevice | None

    def __post_init__(self) -> None:
        _require_enum(self.code, TranscriptionFailureCode, "code")
        _require_string(self.attempt_id, "attempt_id")
        _require_optional_string(self.batch_id, "batch_id")
        _require_optional_string(self.job_id, "job_id")
        _require_enum(self.phase, TranscriptionPhase, "phase")
        _require_string(self.provider_id, "provider_id")
        _require_string(self.model_id, "model_id")
        if self.artifact_root is not None and not _is_artifact_lease_key(
            self.artifact_root
        ):
            raise TypeError("artifact_root must be an ArtifactLeaseKey")
        _require_string(self.precision, "precision")
        _require_enum(
            self.requested_device,
            ExecutionDevice,
            "requested_device",
        )
        if self.effective_device is not None:
            _require_enum(
                self.effective_device,
                ExecutionDevice,
                "effective_device",
            )

    @property
    def message(self) -> str:
        """Return the fixed sanitized message for this failure code."""

        return TRANSCRIPTION_FAILURE_CONTRACT[self.code][0]

    @property
    def retryable(self) -> bool:
        """Return default same-configuration retryability."""

        return TRANSCRIPTION_FAILURE_CONTRACT[self.code][1]

    def __repr__(self) -> str:
        """Render only stable classification fields, never identifiers."""

        return (
            f"{type(self).__name__}(code={self.code.value!r}, "
            f"phase={self.phase.value!r})"
        )

    def __str__(self) -> str:
        """Return the fixed sanitized message."""

        return self.message


@dataclass(frozen=True, slots=True)
class DeviceRetryPolicy:
    """Represent no retry or one recycled same-provider/model CPU retry."""

    retry_device: ExecutionDevice | None = None
    max_retries: int = 0
    requires_worker_recycling: bool = False
    same_provider_model_only: bool = False

    def __post_init__(self) -> None:
        if self.retry_device is not None:
            _require_enum(self.retry_device, ExecutionDevice, "retry_device")
        if type(self.max_retries) is not int:
            raise TypeError("max_retries must be an int")
        _require_bool(
            self.requires_worker_recycling,
            "requires_worker_recycling",
        )
        _require_bool(
            self.same_provider_model_only,
            "same_provider_model_only",
        )

        no_retry = (
            self.retry_device is None
            and self.max_retries == 0
            and not self.requires_worker_recycling
            and not self.same_provider_model_only
        )
        cpu_retry = (
            self.retry_device is ExecutionDevice.CPU
            and self.max_retries == 1
            and self.requires_worker_recycling
            and self.same_provider_model_only
        )
        if not (no_retry or cpu_retry):
            raise ValueError(
                "device retry policy must allow no retry or exactly one "
                "recycled same-provider/model CPU retry"
            )

    @classmethod
    def no_retry(cls) -> DeviceRetryPolicy:
        """Return the fail-closed policy."""

        return cls()

    @classmethod
    def for_failure(
        cls,
        *,
        requested_device: ExecutionDevice,
        failed_device: ExecutionDevice,
        origin: DeviceFailureOrigin,
        retry_device: ExecutionDevice,
        worker_will_recycle: bool,
    ) -> DeviceRetryPolicy:
        """Return the one safe device retry policy, otherwise fail closed."""

        _require_enum(
            requested_device,
            ExecutionDevice,
            "requested_device",
        )
        _require_enum(failed_device, ExecutionDevice, "failed_device")
        _require_enum(origin, DeviceFailureOrigin, "origin")
        _require_enum(retry_device, ExecutionDevice, "retry_device")
        _require_bool(worker_will_recycle, "worker_will_recycle")

        concrete_accelerators = frozenset(ExecutionDevice) - {
            ExecutionDevice.AUTO,
            ExecutionDevice.CPU,
        }
        requested_device_matches = requested_device in {
            ExecutionDevice.AUTO,
            failed_device,
        }
        safe_retry = (
            origin is DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION
            and failed_device in concrete_accelerators
            and requested_device_matches
            and retry_device is ExecutionDevice.CPU
            and worker_will_recycle
        )
        if not safe_retry:
            return cls.no_retry()
        return cls(
            retry_device=ExecutionDevice.CPU,
            max_retries=1,
            requires_worker_recycling=True,
            same_provider_model_only=True,
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
    timestamps: TimestampGranularity = TimestampGranularity.NONE
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
class ResolvedTranscriptionRequest:
    """A request resolved to one exact provider/model without executing it."""

    request: TranscriptionRequest
    provider_id: str
    model_id: str
    requested_language: str
    effective_language: str
    precision: str
    warning_codes: tuple[TranscriptionWarningCode, ...] = ()

    def __post_init__(self) -> None:
        if type(self.request) is not TranscriptionRequest:
            raise TypeError("request must be a TranscriptionRequest")
        for field_name, value in (
            ("provider_id", self.provider_id),
            ("model_id", self.model_id),
            ("precision", self.precision),
        ):
            if type(value) is not str:
                raise TypeError(f"{field_name} must be a string")
            if not value or value != value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string without "
                    "surrounding whitespace"
                )
        for field_name, value in (
            ("requested_language", self.requested_language),
            ("effective_language", self.effective_language),
        ):
            if type(value) is not str:
                raise TypeError(f"{field_name} must be a string")
            if value != "auto" and not _LANGUAGE_PATTERN.fullmatch(value):
                raise ValueError(
                    f"{field_name} must be 'auto' or a canonical lower-case "
                    "language tag"
                )
        if type(self.warning_codes) is not tuple or not all(
            type(warning) is TranscriptionWarningCode for warning in self.warning_codes
        ):
            raise TypeError(
                "warning_codes must be a tuple of TranscriptionWarningCode values"
            )


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """One timestamped transcript segment."""

    start_seconds: float
    end_seconds: float
    text: str = field(repr=False)
    speaker: str | None = field(default=None, repr=False)

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
    artifact_root: _ArtifactIdentity | None
    artifact_dependencies: tuple[_ArtifactIdentity, ...]
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
            type(granularity) is TimestampGranularity for granularity in self.timestamps
        ):
            raise TypeError("timestamps must contain only TimestampGranularity values")
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

    text: str = field(repr=False)
    segments: tuple[TranscriptionSegment, ...] = field(repr=False)
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
            raise TypeError("produced_capabilities must be a ProducedCapabilities")
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
        if has_segments and not has_timestamps:
            raise ValueError("segments require produced timestamp capability")
        if self.text and has_timestamps and not has_segments:
            raise ValueError("non-empty timestamped transcript requires segments")
        if not self.produced_capabilities.diarization and any(
            segment.speaker is not None for segment in self.segments
        ):
            raise ValueError("speaker labels require produced diarization capability")


__all__ = [
    "MAX_BUFFER_AUDIO_BYTES",
    "TRANSCRIPTION_FAILURE_CONTRACT",
    "BufferAudioSource",
    "CancellationGranularity",
    "CancellationToken",
    "DeviceFailureOrigin",
    "DeviceRetryPolicy",
    "ExecutionDevice",
    "FileAudioSource",
    "InputKind",
    "LanguageInputMode",
    "PipelineCapabilities",
    "PrivacyRequirements",
    "ProducedCapabilities",
    "ProgressSink",
    "ResolvedTranscriptionRequest",
    "TimestampGranularity",
    "TranscriptionAction",
    "TranscriptionFailure",
    "TranscriptionFailureCode",
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
