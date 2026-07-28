"""Validated JSON persistence for normalized speech-to-text provenance."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, cast

from tldw_chatbook.Model_Artifacts.leases import ArtifactLeaseKey

from .contracts import (
    ExecutionDevice,
    TimestampGranularity,
    TranscriptionFailureCode,
    TranscriptionResult,
    TranscriptionTask,
    TranscriptionWarningCode,
)


TRANSCRIPTION_PROVENANCE_SCHEMA_VERSION = 1
MAX_TRANSCRIPTION_PROVENANCE_BYTES = 64 * 1024

_LANGUAGE_PATTERN = re.compile(r"(?:auto|[a-z]{2,3}(?:-[a-z0-9]{1,8})*)")
_ARTIFACT_FIELDS = frozenset({"artifact_id", "revision", "variant"})
_CAPABILITY_FIELDS = frozenset(
    {"timestamps", "punctuation", "capitalization", "vad", "diarization"}
)
_FAILED_ATTEMPT_FIELDS = frozenset(
    {
        "attempt_id",
        "batch_id",
        "job_id",
        "provider_id",
        "model_id",
        "artifact_root",
        "artifact_dependencies",
        "precision",
        "requested_device",
        "effective_device",
        "requested_language",
        "effective_language",
        "detected_language",
        "task",
        "error_code",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "attempt_id",
        "batch_id",
        "job_id",
        "retry_of_attempt_id",
        "retry_of_job_id",
        "provider_id",
        "model_id",
        "artifact_root",
        "artifact_dependencies",
        "precision",
        "requested_device",
        "effective_device",
        "requested_language",
        "effective_language",
        "detected_language",
        "task",
        "produced_capabilities",
        "warnings",
        "failed_attempt",
    }
)


def _require_string(
    value: object,
    field_name: str,
    *,
    nullable: bool = False,
) -> str | None:
    if nullable and value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_language(
    value: object,
    field_name: str,
    *,
    nullable: bool = False,
) -> str | None:
    language = _require_string(value, field_name, nullable=nullable)
    if language is not None and not _LANGUAGE_PATTERN.fullmatch(language):
        raise ValueError(
            f"{field_name} must be 'auto' or a canonical lower-case language tag"
        )
    return language


def _require_identifier(value: object, field_name: str) -> str:
    identifier = cast(str, _require_string(value, field_name))
    posix_path = PurePosixPath(identifier)
    windows_path = PureWindowsPath(identifier)
    if (
        identifier != identifier.strip()
        or identifier.lower().startswith("file:")
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or "\\" in identifier
        or identifier.startswith("~")
        or any(ord(character) < 32 or ord(character) == 127 for character in identifier)
        or any(part in {".", "..", "~"} for part in identifier.split("/"))
    ):
        raise ValueError(f"{field_name} must be an identifier, not a local path")
    return identifier


def _require_exact_fields(
    value: object,
    expected: frozenset[str],
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{field_name} fields do not match schema; "
            f"missing={missing}, unknown={unknown}"
        )
    return value


def _artifact_to_document(artifact: ArtifactLeaseKey | None) -> dict[str, str] | None:
    if artifact is None:
        return None
    if type(artifact) is not ArtifactLeaseKey:
        raise TypeError("artifact identity must be an ArtifactLeaseKey")
    return {
        "artifact_id": _require_identifier(
            artifact.artifact_id,
            "artifact.artifact_id",
        ),
        "revision": _require_identifier(artifact.revision, "artifact.revision"),
        "variant": _require_identifier(artifact.variant, "artifact.variant"),
    }


def _validate_artifact_document(
    value: object,
    field_name: str,
    *,
    nullable: bool = False,
) -> dict[str, str] | None:
    if nullable and value is None:
        return None
    artifact = _require_exact_fields(value, _ARTIFACT_FIELDS, field_name)
    normalized: dict[str, str] = {
        name: _require_identifier(artifact[name], f"{field_name}.{name}")
        for name in ("artifact_id", "revision", "variant")
    }
    ArtifactLeaseKey(**normalized)
    return normalized


def _artifacts_to_documents(
    artifacts: tuple[ArtifactLeaseKey, ...],
) -> list[dict[str, str]]:
    if type(artifacts) is not tuple or not all(
        type(artifact) is ArtifactLeaseKey for artifact in artifacts
    ):
        raise TypeError("artifact_dependencies must contain ArtifactLeaseKey values")
    return [
        cast(dict[str, str], _artifact_to_document(artifact)) for artifact in artifacts
    ]


def _validate_artifact_documents(
    value: object,
    field_name: str,
) -> list[dict[str, str]]:
    if type(value) is not list:
        raise TypeError(f"{field_name} must be a list")
    return [
        cast(
            dict[str, str],
            _validate_artifact_document(item, f"{field_name}[{index}]"),
        )
        for index, item in enumerate(value)
    ]


@dataclass(frozen=True, slots=True)
class FailedTranscriptionAttempt:
    """Complete sanitized context for one failed STT attempt."""

    attempt_id: str
    batch_id: str | None
    job_id: str | None
    provider_id: str
    model_id: str
    artifact_root: ArtifactLeaseKey | None
    artifact_dependencies: tuple[ArtifactLeaseKey, ...]
    precision: str
    requested_device: ExecutionDevice
    effective_device: ExecutionDevice | None
    requested_language: str
    effective_language: str
    detected_language: str | None
    task: TranscriptionTask
    error_code: TranscriptionFailureCode

    def __post_init__(self) -> None:
        _require_identifier(self.attempt_id, "attempt_id")
        if self.batch_id is not None:
            _require_identifier(self.batch_id, "batch_id")
        if self.job_id is not None:
            _require_identifier(self.job_id, "job_id")
        _require_identifier(self.provider_id, "provider_id")
        _require_identifier(self.model_id, "model_id")
        _artifact_to_document(self.artifact_root)
        _artifacts_to_documents(self.artifact_dependencies)
        _require_identifier(self.precision, "precision")
        if type(self.requested_device) is not ExecutionDevice:
            raise TypeError("requested_device must be an ExecutionDevice")
        if self.effective_device is not None and (
            type(self.effective_device) is not ExecutionDevice
        ):
            raise TypeError("effective_device must be an ExecutionDevice or None")
        _require_language(self.requested_language, "requested_language")
        _require_language(self.effective_language, "effective_language")
        _require_language(
            self.detected_language,
            "detected_language",
            nullable=True,
        )
        if type(self.task) is not TranscriptionTask:
            raise TypeError("task must be a TranscriptionTask")
        if type(self.error_code) is not TranscriptionFailureCode:
            raise TypeError("error_code must be a TranscriptionFailureCode")


def _failed_attempt_to_document(
    attempt: FailedTranscriptionAttempt,
) -> dict[str, Any]:
    if type(attempt) is not FailedTranscriptionAttempt:
        raise TypeError("failed_attempt must be a FailedTranscriptionAttempt")
    return {
        "attempt_id": attempt.attempt_id,
        "batch_id": attempt.batch_id,
        "job_id": attempt.job_id,
        "provider_id": attempt.provider_id,
        "model_id": attempt.model_id,
        "artifact_root": _artifact_to_document(attempt.artifact_root),
        "artifact_dependencies": _artifacts_to_documents(attempt.artifact_dependencies),
        "precision": attempt.precision,
        "requested_device": attempt.requested_device.value,
        "effective_device": (
            attempt.effective_device.value
            if attempt.effective_device is not None
            else None
        ),
        "requested_language": attempt.requested_language,
        "effective_language": attempt.effective_language,
        "detected_language": attempt.detected_language,
        "task": attempt.task.value,
        "error_code": attempt.error_code.value,
    }


def _validate_failed_attempt_document(value: object) -> dict[str, Any]:
    attempt = _require_exact_fields(value, _FAILED_ATTEMPT_FIELDS, "failed_attempt")
    requested_device = _require_enum_value(
        attempt["requested_device"],
        ExecutionDevice,
        "failed_attempt.requested_device",
    )
    effective_raw = attempt["effective_device"]
    effective_device = (
        None
        if effective_raw is None
        else _require_enum_value(
            effective_raw,
            ExecutionDevice,
            "failed_attempt.effective_device",
        )
    )
    return {
        "attempt_id": _require_identifier(
            attempt["attempt_id"], "failed_attempt.attempt_id"
        ),
        "batch_id": (
            None
            if attempt["batch_id"] is None
            else _require_identifier(attempt["batch_id"], "failed_attempt.batch_id")
        ),
        "job_id": (
            None
            if attempt["job_id"] is None
            else _require_identifier(attempt["job_id"], "failed_attempt.job_id")
        ),
        "provider_id": _require_identifier(
            attempt["provider_id"], "failed_attempt.provider_id"
        ),
        "model_id": _require_identifier(
            attempt["model_id"], "failed_attempt.model_id"
        ),
        "artifact_root": _validate_artifact_document(
            attempt["artifact_root"],
            "failed_attempt.artifact_root",
            nullable=True,
        ),
        "artifact_dependencies": _validate_artifact_documents(
            attempt["artifact_dependencies"],
            "failed_attempt.artifact_dependencies",
        ),
        "precision": _require_identifier(
            attempt["precision"], "failed_attempt.precision"
        ),
        "requested_device": requested_device,
        "effective_device": effective_device,
        "requested_language": _require_language(
            attempt["requested_language"],
            "failed_attempt.requested_language",
        ),
        "effective_language": _require_language(
            attempt["effective_language"],
            "failed_attempt.effective_language",
        ),
        "detected_language": _require_language(
            attempt["detected_language"],
            "failed_attempt.detected_language",
            nullable=True,
        ),
        "task": _require_enum_value(
            attempt["task"],
            TranscriptionTask,
            "failed_attempt.task",
        ),
        "error_code": _require_enum_value(
            attempt["error_code"],
            TranscriptionFailureCode,
            "failed_attempt.error_code",
        ),
    }


def _require_enum_value(
    value: object,
    enum_type: type[Any],
    field_name: str,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    try:
        enum_type(value)
    except ValueError as error:
        raise ValueError(f"{field_name} is not supported") from error
    return value


def build_transcription_provenance_document(
    result: TranscriptionResult,
    *,
    failed_attempt: FailedTranscriptionAttempt | Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build a version-1 persistence document from a normalized result."""

    if type(result) is not TranscriptionResult:
        raise TypeError("result must be a TranscriptionResult")
    if result.provenance.schema_version != TRANSCRIPTION_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported transcription provenance schema version")
    provenance = result.provenance
    document = {
        "schema_version": TRANSCRIPTION_PROVENANCE_SCHEMA_VERSION,
        "attempt_id": provenance.attempt_id,
        "batch_id": provenance.batch_id,
        "job_id": provenance.job_id,
        "retry_of_attempt_id": provenance.retry_of_attempt_id,
        "retry_of_job_id": provenance.retry_of_job_id,
        "provider_id": provenance.provider_id,
        "model_id": provenance.model_id,
        "artifact_root": _artifact_to_document(provenance.artifact_root),
        "artifact_dependencies": _artifacts_to_documents(
            provenance.artifact_dependencies
        ),
        "precision": provenance.precision,
        "requested_device": provenance.requested_device.value,
        "effective_device": provenance.effective_device.value,
        "requested_language": provenance.requested_language,
        "effective_language": provenance.effective_language,
        "detected_language": provenance.detected_language,
        "task": provenance.task.value,
        "produced_capabilities": {
            "timestamps": result.produced_capabilities.timestamps.value,
            "punctuation": result.produced_capabilities.punctuation,
            "capitalization": result.produced_capabilities.capitalization,
            "vad": result.produced_capabilities.vad,
            "diarization": result.produced_capabilities.diarization,
        },
        "warnings": [warning.value for warning in result.warnings],
        "failed_attempt": (
            _failed_attempt_to_document(failed_attempt)
            if type(failed_attempt) is FailedTranscriptionAttempt
            else _validate_failed_attempt_document(failed_attempt)
            if failed_attempt is not None
            else None
        ),
    }
    return _validate_transcription_provenance_document(document)


def _validate_capabilities(value: object) -> dict[str, Any]:
    capabilities = _require_exact_fields(
        value,
        _CAPABILITY_FIELDS,
        "produced_capabilities",
    )
    normalized: dict[str, Any] = {
        "timestamps": _require_enum_value(
            capabilities["timestamps"],
            TimestampGranularity,
            "produced_capabilities.timestamps",
        )
    }
    for name in ("punctuation", "capitalization", "vad", "diarization"):
        capability = capabilities[name]
        if type(capability) is not bool:
            raise TypeError(f"produced_capabilities.{name} must be a bool")
        normalized[name] = capability
    return normalized


def _validate_warnings(value: object) -> list[str]:
    if type(value) is not list:
        raise TypeError("warnings must be a list")
    warnings = [
        _require_enum_value(item, TranscriptionWarningCode, f"warnings[{index}]")
        for index, item in enumerate(value)
    ]
    if len(warnings) != len(set(warnings)):
        raise ValueError("warnings must not contain duplicates")
    return warnings


def _validate_transcription_provenance_document(
    value: object,
) -> dict[str, Any]:
    document = _require_exact_fields(value, _PROVENANCE_FIELDS, "provenance")
    if type(document["schema_version"]) is not int:
        raise TypeError("schema_version must be an int")
    if document["schema_version"] != TRANSCRIPTION_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported transcription provenance schema version")
    failed_attempt = document["failed_attempt"]
    normalized: dict[str, Any] = {
        "schema_version": TRANSCRIPTION_PROVENANCE_SCHEMA_VERSION,
        "attempt_id": _require_identifier(document["attempt_id"], "attempt_id"),
        "batch_id": (
            None
            if document["batch_id"] is None
            else _require_identifier(document["batch_id"], "batch_id")
        ),
        "job_id": (
            None
            if document["job_id"] is None
            else _require_identifier(document["job_id"], "job_id")
        ),
        "retry_of_attempt_id": (
            None
            if document["retry_of_attempt_id"] is None
            else _require_identifier(
                document["retry_of_attempt_id"],
                "retry_of_attempt_id",
            )
        ),
        "retry_of_job_id": (
            None
            if document["retry_of_job_id"] is None
            else _require_identifier(document["retry_of_job_id"], "retry_of_job_id")
        ),
        "provider_id": _require_identifier(document["provider_id"], "provider_id"),
        "model_id": _require_identifier(document["model_id"], "model_id"),
        "artifact_root": _validate_artifact_document(
            document["artifact_root"],
            "artifact_root",
            nullable=True,
        ),
        "artifact_dependencies": _validate_artifact_documents(
            document["artifact_dependencies"],
            "artifact_dependencies",
        ),
        "precision": _require_identifier(document["precision"], "precision"),
        "requested_device": _require_enum_value(
            document["requested_device"],
            ExecutionDevice,
            "requested_device",
        ),
        "effective_device": _require_enum_value(
            document["effective_device"],
            ExecutionDevice,
            "effective_device",
        ),
        "requested_language": _require_language(
            document["requested_language"],
            "requested_language",
        ),
        "effective_language": _require_language(
            document["effective_language"],
            "effective_language",
        ),
        "detected_language": _require_language(
            document["detected_language"],
            "detected_language",
            nullable=True,
        ),
        "task": _require_enum_value(
            document["task"],
            TranscriptionTask,
            "task",
        ),
        "produced_capabilities": _validate_capabilities(
            document["produced_capabilities"]
        ),
        "warnings": _validate_warnings(document["warnings"]),
        "failed_attempt": (
            None
            if failed_attempt is None
            else _validate_failed_attempt_document(failed_attempt)
        ),
    }
    normalized_failed_attempt = normalized["failed_attempt"]
    if (
        normalized["retry_of_job_id"] is not None
        and normalized["retry_of_attempt_id"] is None
    ):
        raise ValueError("retry_of_job_id requires retry_of_attempt_id")
    if (normalized["retry_of_attempt_id"] is None) != (
        normalized_failed_attempt is None
    ):
        raise ValueError("retry_of_attempt_id and failed_attempt must be set together")
    if normalized["retry_of_attempt_id"] == normalized["attempt_id"]:
        raise ValueError("an attempt cannot retry itself")
    if (
        normalized["job_id"] is not None
        and normalized["retry_of_job_id"] == normalized["job_id"]
    ):
        raise ValueError("a job cannot retry itself")
    if normalized_failed_attempt is not None:
        if (
            normalized_failed_attempt["attempt_id"]
            != normalized["retry_of_attempt_id"]
        ):
            raise ValueError("failed_attempt does not match retry_of_attempt_id")
        if normalized_failed_attempt["job_id"] != normalized["retry_of_job_id"]:
            raise ValueError("failed_attempt does not match retry_of_job_id")
    return normalized


def _dump_bounded(document: Mapping[str, object]) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > MAX_TRANSCRIPTION_PROVENANCE_BYTES:
        raise ValueError("transcription provenance exceeds the size limit")
    return encoded


def _load_bounded(raw: str) -> object:
    if type(raw) is not str:
        raise TypeError("serialized provenance must be a string")
    if not raw:
        raise ValueError("serialized provenance must not be empty")
    if len(raw.encode("utf-8")) > MAX_TRANSCRIPTION_PROVENANCE_BYTES:
        raise ValueError("transcription provenance exceeds the size limit")
    return json.loads(raw)


def dump_transcription_provenance_document(value: object) -> str:
    """Validate and serialize one canonical provenance document."""

    return _dump_bounded(_validate_transcription_provenance_document(value))


def load_transcription_provenance_document(raw: str) -> dict[str, Any]:
    """Parse and validate one serialized provenance document."""

    return _validate_transcription_provenance_document(_load_bounded(raw))


def dump_failed_transcription_attempt(value: object) -> str:
    """Validate and serialize one sanitized failed-attempt snapshot."""

    document = (
        _failed_attempt_to_document(value)
        if type(value) is FailedTranscriptionAttempt
        else _validate_failed_attempt_document(value)
    )
    return _dump_bounded(_validate_failed_attempt_document(document))


def load_failed_transcription_attempt(raw: str) -> dict[str, Any]:
    """Parse and validate one serialized failed-attempt snapshot."""

    return _validate_failed_attempt_document(_load_bounded(raw))


__all__ = [
    "MAX_TRANSCRIPTION_PROVENANCE_BYTES",
    "TRANSCRIPTION_PROVENANCE_SCHEMA_VERSION",
    "FailedTranscriptionAttempt",
    "build_transcription_provenance_document",
    "dump_failed_transcription_attempt",
    "dump_transcription_provenance_document",
    "load_failed_transcription_attempt",
    "load_transcription_provenance_document",
]
