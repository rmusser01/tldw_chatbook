from __future__ import annotations

from dataclasses import replace

from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJobRegistry,
    _job_from_row,
)
from tldw_chatbook.Model_Artifacts.leases import ArtifactLeaseKey
from tldw_chatbook.STT.contracts import (
    ExecutionDevice,
    ProducedCapabilities,
    TimestampGranularity,
    TranscriptionFailureCode,
    TranscriptionProvenance,
    TranscriptionResult,
    TranscriptionTask,
    TranscriptionTimings,
)
from tldw_chatbook.STT.persistence import (
    FailedTranscriptionAttempt,
    build_transcription_provenance_document,
    dump_failed_transcription_attempt,
    load_failed_transcription_attempt,
)


def _failed_attempt(
    *,
    attempt_id: str = "attempt-1",
    job_id: str | None = "ingest-job-1",
) -> FailedTranscriptionAttempt:
    return FailedTranscriptionAttempt(
        attempt_id=attempt_id,
        batch_id="batch-1" if job_id is not None else None,
        job_id=job_id,
        provider_id="parakeet-onnx",
        model_id="parakeet-v2",
        artifact_root=ArtifactLeaseKey("parakeet-v2", "revision-2", "int8"),
        artifact_dependencies=(),
        precision="int8",
        requested_device=ExecutionDevice.AUTO,
        effective_device=ExecutionDevice.CPU,
        requested_language="en",
        effective_language="en",
        detected_language=None,
        task=TranscriptionTask.TRANSCRIBE,
        error_code=TranscriptionFailureCode.INFERENCE_FAILED,
    )


def _failed_document(
    *,
    attempt_id: str = "attempt-1",
    job_id: str | None = "ingest-job-1",
) -> dict[str, object]:
    return load_failed_transcription_attempt(
        dump_failed_transcription_attempt(
            _failed_attempt(attempt_id=attempt_id, job_id=job_id)
        )
    )


def _successful_retry_result(job_id: str) -> TranscriptionResult:
    return TranscriptionResult(
        text="retry succeeded",
        segments=(),
        provenance=TranscriptionProvenance(
            schema_version=1,
            attempt_id="attempt-2",
            batch_id="batch-1",
            job_id=job_id,
            retry_of_attempt_id="attempt-1",
            retry_of_job_id="ingest-job-1",
            provider_id="faster-whisper",
            model_id="large-v3",
            artifact_root=ArtifactLeaseKey(
                "faster-whisper-large-v3",
                "revision-3",
                "int8",
            ),
            artifact_dependencies=(),
            precision="int8",
            requested_device=ExecutionDevice.AUTO,
            effective_device=ExecutionDevice.CPU,
            requested_language="en",
            effective_language="en",
            detected_language=None,
            task=TranscriptionTask.TRANSCRIBE,
        ),
        produced_capabilities=ProducedCapabilities(
            timestamps=TimestampGranularity.NONE,
            punctuation=True,
            capitalization=True,
            vad=False,
            diarization=False,
        ),
        duration_seconds=1.0,
        timings=TranscriptionTimings(total_seconds=0.5),
    )


def test_mark_failed_validates_and_requeue_carries_separate_source_snapshot() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/audio.wav")
    failed_document = _failed_document()

    failed = registry.mark_failed(
        job.job_id,
        error="Speech-to-text inference failed.",
        stt_failure_provenance=failed_document,
    )
    retried = registry.requeue(failed.job_id)

    assert failed.stt_failure_provenance == failed_document
    assert failed.retry_of_job_id is None
    assert retried.retry_of_job_id == failed.job_id
    assert retried.stt_failure_provenance is None
    assert retried.retry_source_failure_provenance == failed_document


def test_retry_failure_does_not_overwrite_retry_source_snapshot() -> None:
    registry = LibraryIngestJobRegistry()
    original = registry.submit(source_path="/tmp/audio.wav")
    first_failure = registry.mark_failed(
        original.job_id,
        error="first failure",
        stt_failure_provenance=_failed_document(),
    )
    retried = registry.requeue(first_failure.job_id)
    second_document = _failed_document(
        attempt_id="attempt-2",
        job_id=retried.job_id,
    )

    second_failure = registry.mark_failed(
        retried.job_id,
        error="second failure",
        stt_failure_provenance=second_document,
    )

    assert second_failure.stt_failure_provenance == second_document
    assert second_failure.retry_source_failure_provenance == (
        first_failure.stt_failure_provenance
    )


def test_returned_retry_lineage_cannot_mutate_registry_snapshot() -> None:
    registry = LibraryIngestJobRegistry()
    original = registry.submit(source_path="/tmp/audio.wav")
    failed = registry.mark_failed(
        original.job_id,
        error="first failure",
        stt_failure_provenance=_failed_document(),
    )
    retried = registry.requeue(failed.job_id)

    retried.retry_source_failure_provenance["model_id"] = "tampered"
    visible = registry.jobs()[0]
    assert visible.retry_source_failure_provenance["model_id"] == "parakeet-v2"

    visible.retry_source_failure_provenance["artifact_root"]["revision"] = "tampered"
    assert (
        registry.jobs()[0].retry_source_failure_provenance["artifact_root"]["revision"]
        == "revision-2"
    )


def test_job_upsert_preserves_first_persisted_retry_source(tmp_path) -> None:
    store = LibraryIngestJobsDB(tmp_path / "jobs.sqlite")
    registry = LibraryIngestJobRegistry()
    registry.attach_store(store)
    original = registry.submit(source_path="/tmp/audio.wav")
    failed = registry.mark_failed(
        original.job_id,
        error="first failure",
        stt_failure_provenance=_failed_document(),
    )
    retried = registry.requeue(failed.job_id)

    store.upsert_job(
        replace(
            retried,
            retry_of_job_id="different-job",
            retry_source_failure_provenance=_failed_document(
                attempt_id="different-attempt",
                job_id="different-job",
            ),
        )
    )
    restored = _job_from_row(store.all_jobs()[-1])

    assert restored.retry_of_job_id == failed.job_id
    assert restored.retry_source_failure_provenance == failed.stt_failure_provenance
    store.close()


def test_invalid_failure_provenance_does_not_mutate_job() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/audio.wav")
    invalid = _failed_document()
    invalid["raw_exception"] = "private path traceback"

    try:
        registry.mark_failed(
            job.job_id,
            error="failed",
            stt_failure_provenance=invalid,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid STT failure provenance was accepted")

    current = registry.jobs()[0]
    assert current.state is IngestJobState.QUEUED
    assert current.stt_failure_provenance is None


def test_retry_snapshot_survives_failed_job_pruning_and_builds_success(
    tmp_path,
) -> None:
    store = LibraryIngestJobsDB(tmp_path / "jobs.sqlite")
    registry = LibraryIngestJobRegistry()
    registry.attach_store(store)
    original = registry.submit(source_path="/tmp/audio.wav")
    failed = registry.mark_failed(
        original.job_id,
        error="Speech-to-text inference failed.",
        stt_failure_provenance=_failed_document(),
    )
    retried = registry.requeue(failed.job_id)

    store.delete_job(failed.job_id)
    rows = store.all_jobs()
    assert [row["job_id"] for row in rows] == [retried.job_id]
    restored_retry = _job_from_row(rows[0])

    persisted = build_transcription_provenance_document(
        _successful_retry_result(restored_retry.job_id),
        failed_attempt=restored_retry.retry_source_failure_provenance,
    )

    assert restored_retry.retry_of_job_id == failed.job_id
    assert persisted["retry_of_job_id"] == failed.job_id
    assert persisted["failed_attempt"] == failed.stt_failure_provenance
    store.close()
