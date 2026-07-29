from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Model_Artifacts.leases import ArtifactLeaseKey
from tldw_chatbook.STT.contracts import (
    ExecutionDevice,
    ProducedCapabilities,
    TimestampGranularity,
    TranscriptionFailureCode,
    TranscriptionProvenance,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTask,
    TranscriptionTimings,
    TranscriptionWarningCode,
)
from tldw_chatbook.STT.persistence import (
    MAX_TRANSCRIPTION_PROVENANCE_BYTES,
    FailedTranscriptionAttempt,
    build_transcription_provenance_document,
    dump_failed_transcription_attempt,
    dump_transcription_provenance_document,
    load_failed_transcription_attempt,
    load_transcription_provenance_document,
)


ROOT_ARTIFACT = ArtifactLeaseKey("parakeet-v2", "revision-2", "int8")
VAD_ARTIFACT = ArtifactLeaseKey("silero-vad", "revision-1", "fp32")


def _result(**provenance_overrides: object) -> TranscriptionResult:
    provenance_values: dict[str, object] = {
        "schema_version": 1,
        "attempt_id": "attempt-2",
        "batch_id": "batch-1",
        "job_id": "job-2",
        "retry_of_attempt_id": "attempt-1",
        "retry_of_job_id": "job-1",
        "provider_id": "parakeet-onnx",
        "model_id": "parakeet-v2",
        "artifact_root": ROOT_ARTIFACT,
        "artifact_dependencies": (VAD_ARTIFACT,),
        "precision": "int8",
        "requested_device": ExecutionDevice.AUTO,
        "effective_device": ExecutionDevice.CPU,
        "requested_language": "en",
        "effective_language": "en",
        "detected_language": None,
        "task": TranscriptionTask.TRANSCRIBE,
    }
    provenance_values.update(provenance_overrides)
    return TranscriptionResult(
        text="hello",
        segments=(TranscriptionSegment(0.0, 1.0, "hello"),),
        provenance=TranscriptionProvenance(**provenance_values),  # type: ignore[arg-type]
        produced_capabilities=ProducedCapabilities(
            timestamps=TimestampGranularity.SEGMENT,
            punctuation=True,
            capitalization=True,
            vad=True,
            diarization=False,
        ),
        duration_seconds=1.0,
        timings=TranscriptionTimings(total_seconds=0.5),
        warnings=(TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,),
    )


def _failed_attempt(**overrides: object) -> FailedTranscriptionAttempt:
    values: dict[str, object] = {
        "attempt_id": "attempt-1",
        "batch_id": "batch-1",
        "job_id": "job-1",
        "provider_id": "parakeet-onnx",
        "model_id": "parakeet-v2",
        "artifact_root": ROOT_ARTIFACT,
        "artifact_dependencies": (VAD_ARTIFACT,),
        "precision": "int8",
        "requested_device": ExecutionDevice.AUTO,
        "effective_device": ExecutionDevice.CPU,
        "requested_language": "en",
        "effective_language": "en",
        "detected_language": None,
        "task": TranscriptionTask.TRANSCRIBE,
        "error_code": TranscriptionFailureCode.INFERENCE_FAILED,
    }
    values.update(overrides)
    return FailedTranscriptionAttempt(**values)  # type: ignore[arg-type]


def test_complete_provenance_document_round_trips_canonically() -> None:
    document = build_transcription_provenance_document(
        _result(),
        failed_attempt=_failed_attempt(),
    )

    assert document == {
        "schema_version": 1,
        "attempt_id": "attempt-2",
        "batch_id": "batch-1",
        "job_id": "job-2",
        "retry_of_attempt_id": "attempt-1",
        "retry_of_job_id": "job-1",
        "provider_id": "parakeet-onnx",
        "model_id": "parakeet-v2",
        "artifact_root": {
            "artifact_id": "parakeet-v2",
            "revision": "revision-2",
            "variant": "int8",
        },
        "artifact_dependencies": [
            {
                "artifact_id": "silero-vad",
                "revision": "revision-1",
                "variant": "fp32",
            }
        ],
        "precision": "int8",
        "requested_device": "auto",
        "effective_device": "cpu",
        "requested_language": "en",
        "effective_language": "en",
        "detected_language": None,
        "task": "transcribe",
        "produced_capabilities": {
            "timestamps": "segment",
            "punctuation": True,
            "capitalization": True,
            "vad": True,
            "diarization": False,
        },
        "warnings": ["requested_language_not_enforced"],
        "failed_attempt": {
            "attempt_id": "attempt-1",
            "batch_id": "batch-1",
            "job_id": "job-1",
            "provider_id": "parakeet-onnx",
            "model_id": "parakeet-v2",
            "artifact_root": {
                "artifact_id": "parakeet-v2",
                "revision": "revision-2",
                "variant": "int8",
            },
            "artifact_dependencies": [
                {
                    "artifact_id": "silero-vad",
                    "revision": "revision-1",
                    "variant": "fp32",
                }
            ],
            "precision": "int8",
            "requested_device": "auto",
            "effective_device": "cpu",
            "requested_language": "en",
            "effective_language": "en",
            "detected_language": None,
            "task": "transcribe",
            "error_code": "inference_failed",
        },
    }

    encoded = dump_transcription_provenance_document(document)
    assert encoded == json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert load_transcription_provenance_document(encoded) == document


def test_non_library_failed_attempt_allows_nullable_job_identity() -> None:
    failed_attempt = _failed_attempt(batch_id=None, job_id=None)

    encoded = dump_failed_transcription_attempt(failed_attempt)
    retry_document = build_transcription_provenance_document(
        _result(job_id=None, retry_of_job_id=None),
        failed_attempt=failed_attempt,
    )

    assert load_failed_transcription_attempt(encoded)["batch_id"] is None
    assert load_failed_transcription_attempt(encoded)["job_id"] is None
    assert retry_document["retry_of_job_id"] is None
    assert retry_document["failed_attempt"]["job_id"] is None


def test_failed_attempt_is_frozen_and_contains_no_free_form_error() -> None:
    failed_attempt = _failed_attempt()

    assert not hasattr(failed_attempt, "__dict__")
    assert not hasattr(failed_attempt, "error")
    assert not hasattr(failed_attempt, "exception")
    with pytest.raises(FrozenInstanceError):
        failed_attempt.model_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 2),
        (("requested_device",), "tpu"),
        (("task",), "summarize"),
        (("warnings",), ["raw warning"]),
        (("artifact_root", "revision"), ""),
        (("artifact_root", "revision"), "bad\x1frevision"),
        (("produced_capabilities", "vad"), 1),
        (("failed_attempt", "error_code"), "traceback"),
    ],
)
def test_loader_rejects_invalid_versioned_fields(
    path: tuple[str, ...],
    value: object,
) -> None:
    document = build_transcription_provenance_document(
        _result(),
        failed_attempt=_failed_attempt(),
    )
    target: object = document
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises((TypeError, ValueError)):
        load_transcription_provenance_document(json.dumps(document))


@pytest.mark.parametrize(
    ("target_field", "forbidden_field"),
    [
        (None, "local_audio_path"),
        ("failed_attempt", "raw_exception"),
        ("failed_attempt", "log_output"),
    ],
)
def test_loader_rejects_unknown_or_sensitive_fields(
    target_field: str | None,
    forbidden_field: str,
) -> None:
    document = build_transcription_provenance_document(
        _result(),
        failed_attempt=_failed_attempt(),
    )
    target = document if target_field is None else document[target_field]
    target[forbidden_field] = "/private/customer/audio.wav"  # type: ignore[index]

    with pytest.raises(ValueError, match="fields"):
        load_transcription_provenance_document(json.dumps(document))


def test_loader_does_not_mutate_caller_document() -> None:
    document = build_transcription_provenance_document(
        _result(),
        failed_attempt=_failed_attempt(),
    )
    original = copy.deepcopy(document)

    encoded = dump_transcription_provenance_document(document)

    assert document == original
    assert load_transcription_provenance_document(encoded) == original


@pytest.mark.parametrize(
    ("retry_of_attempt_id", "retry_of_job_id", "failed_attempt"),
    [
        ("attempt-1", "job-1", None),
        (None, "job-1", None),
        ("different-attempt", "job-1", _failed_attempt()),
        ("attempt-1", "different-job", _failed_attempt()),
        (None, None, _failed_attempt()),
    ],
)
def test_builder_rejects_incomplete_or_contradictory_retry_lineage(
    retry_of_attempt_id: str | None,
    retry_of_job_id: str | None,
    failed_attempt: FailedTranscriptionAttempt | None,
) -> None:
    with pytest.raises(ValueError, match="failed_attempt|retry"):
        build_transcription_provenance_document(
            _result(
                retry_of_attempt_id=retry_of_attempt_id,
                retry_of_job_id=retry_of_job_id,
            ),
            failed_attempt=failed_attempt,
        )


@pytest.mark.parametrize(
    "result",
    [
        _result(attempt_id="attempt-1"),
        _result(job_id="job-1"),
    ],
)
def test_builder_rejects_self_referential_retry_lineage(
    result: TranscriptionResult,
) -> None:
    with pytest.raises(ValueError, match="itself|self"):
        build_transcription_provenance_document(
            result,
            failed_attempt=_failed_attempt(),
        )


@pytest.mark.parametrize(
    ("target_field", "identity_field", "value"),
    [
        (None, "model_id", "/Users/alice/Secret/model.bin"),
        (None, "precision", "/Users/alice/Secret/precision.txt"),
        ("failed_attempt", "model_id", r"C:\Users\alice\Secret\model.bin"),
        ("failed_attempt", "precision", r"C:\Users\alice\Secret\precision.txt"),
        ("failed_attempt", "attempt_id", "../../private-attempt"),
    ],
)
def test_loader_rejects_local_paths_in_identity_fields(
    target_field: str | None,
    identity_field: str,
    value: str,
) -> None:
    document = build_transcription_provenance_document(
        _result(),
        failed_attempt=_failed_attempt(),
    )
    target = document if target_field is None else document[target_field]
    target[identity_field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match="identifier|path"):
        load_transcription_provenance_document(json.dumps(document))


def test_failed_attempt_dto_rejects_path_shaped_artifact_identity() -> None:
    with pytest.raises(ValueError, match="identifier|path"):
        _failed_attempt(
            artifact_root=ArtifactLeaseKey(
                "/Users/alice/Secret/model",
                "revision-2",
                "int8",
            )
        )


def test_serializer_rejects_oversized_provenance() -> None:
    document = build_transcription_provenance_document(
        _result(retry_of_attempt_id=None, retry_of_job_id=None)
    )
    document["model_id"] = "x" * MAX_TRANSCRIPTION_PROVENANCE_BYTES

    with pytest.raises(ValueError, match="size"):
        dump_transcription_provenance_document(document)


@pytest.mark.parametrize("raw", ["", "null", "[]", "{invalid"])
def test_loader_rejects_non_document_json(raw: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        load_transcription_provenance_document(raw)
