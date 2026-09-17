"""Parent-side STT diagnostics through the real mounted failure callback."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from loguru import logger
from textual.app import App

from tldw_chatbook.app import LibraryIngestQueueMixin
from tldw_chatbook.Library.library_ingest_jobs import IngestJobState, LibraryIngestJob
from tldw_chatbook.STT.contracts import TranscriptionFailureCode
from tldw_chatbook.STT.executor import ExecutorEvent, ExecutorFailure, WorkerPhase


class _FailureHost(LibraryIngestQueueMixin, App):
    def __init__(self):
        super().__init__()
        self._init_library_ingest_runtime_state()
        # Exercise accepted callbacks with new parse admission paused.
        self._ingest_maintenance_paused = True


@pytest.mark.parametrize(
    ("progress", "expected_phase"),
    [
        ({"phase": "loading", "message": "/private/native-secret"}, "loading"),
        ({"phase": WorkerPhase.TRANSCRIBING}, "transcribing"),
        (None, "unknown"),
        ("/private/progress-secret", "unknown"),
        (["/private/progress-secret"], "unknown"),
        ({"phase": {"native": "/private/phase-secret"}}, "unknown"),
    ],
)
def test_failure_callback_normalizes_progress_without_app_or_file_sink(
    monkeypatch, progress, expected_phase
):
    error_log = Mock()
    monkeypatch.setattr("tldw_chatbook.app.logger", SimpleNamespace(error=error_log))
    host = SimpleNamespace(
        _ingest_shutdown=False,
        _local_stt_terminal_matches=lambda *_: True,
        _claim_ingest_local_stt_job=lambda _: SimpleNamespace(progress=progress),
        _ingest_local_stt_jobs={"job-1": (7, "attempt-123")},
        library_ingest_jobs=SimpleNamespace(mark_failed=Mock()),
        _top_up_ingest_parse_pool=Mock(),
    )
    LibraryIngestQueueMixin._on_ingest_local_stt_failure(
        host,
        "job-1",
        ExecutorFailure(7, "attempt-123", TranscriptionFailureCode.INFERENCE_FAILED),
    )

    error_log.assert_called_once()
    template, *fields = error_log.call_args.args
    message = template.format(*fields)
    for expected in (
        "job_id=job-1",
        "attempt_id=attempt-123",
        "generation=7",
        f"phase={expected_phase}",
        "code=inference_failed",
    ):
        assert expected in message
    assert "/private/" not in message
    assert "secret" not in message
    host.library_ingest_jobs.mark_failed.assert_called_once()
    assert "job-1" not in host._ingest_local_stt_jobs


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [WorkerPhase.LOADING, WorkerPhase.TRANSCRIBING])
async def test_accepted_stt_failure_logs_context_once_without_private_data(
    tmp_path, phase
):
    host = _FailureHost()
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/private/source-secret.wav",
        title="private title",
        state=IngestJobState.PARSING,
        ingest_options={
            "audio_video": {"transcription_model_dir": "/private/model-secret"}
        },
    )
    log_path = tmp_path / "application.log"
    sink = logger.add(log_path, level="ERROR", format="{message}")
    try:
        async with host.run_test():
            host.library_ingest_jobs.restore([job], next_id=2)
            host._ingest_local_stt_jobs[job.job_id] = (7, "attempt-123")
            host._on_ingest_local_stt_event(
                job.job_id, ExecutorEvent(7, "attempt-123", phase)
            )
            failure = ExecutorFailure(
                7, "attempt-123", TranscriptionFailureCode.INFERENCE_FAILED
            )
            host._on_ingest_local_stt_failure(job.job_id, failure)
            host._on_ingest_local_stt_failure(job.job_id, failure)
            assert (
                host.library_ingest_jobs.get_job(job.job_id).state
                is IngestJobState.FAILED
            )
    finally:
        logger.remove(sink)

    log = log_path.read_text()
    assert log.count("Library local STT failed") == 1
    for expected in (
        "ingest-job-1",
        "attempt-123",
        "generation=7",
        f"phase={phase.value}",
        "code=inference_failed",
    ):
        assert expected in log
    for private in (job.source_path, "source-secret", "model-secret", "private title"):
        assert private not in log


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["cancelled", "stale", "shutdown"])
async def test_cancelled_or_rejected_stt_failure_does_not_log_an_error(
    tmp_path, terminal
):
    host = _FailureHost()
    job = LibraryIngestJob(
        "ingest-job-1", "/private/source.wav", state=IngestJobState.PARSING
    )
    log_path = tmp_path / "application.log"
    sink = logger.add(log_path, level="ERROR", format="{message}")
    try:
        async with host.run_test():
            host.library_ingest_jobs.restore([job], next_id=2)
            host._ingest_local_stt_jobs[job.job_id] = (7, "attempt-123")
            host._ingest_shutdown = terminal == "shutdown"
            host._on_ingest_local_stt_failure(
                job.job_id,
                ExecutorFailure(
                    6 if terminal == "stale" else 7,
                    "attempt-123",
                    TranscriptionFailureCode.CANCELLED
                    if terminal == "cancelled"
                    else TranscriptionFailureCode.INFERENCE_FAILED,
                ),
            )
            expected = (
                IngestJobState.CANCELLED
                if terminal == "cancelled"
                else IngestJobState.PARSING
            )
            assert host.library_ingest_jobs.get_job(job.job_id).state is expected
    finally:
        logger.remove(sink)
    assert "Library local STT failed" not in log_path.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [None, "/private/phase-secret", [], {}])
async def test_stt_failure_logs_unknown_for_unrecognized_phase(tmp_path, phase):
    host = _FailureHost()
    job = LibraryIngestJob(
        "ingest-job-1",
        "/private/source.wav",
        state=IngestJobState.PARSING,
        progress={"phase": phase},
    )
    log_path = tmp_path / "application.log"
    sink = logger.add(log_path, level="ERROR", format="{message}")
    try:
        async with host.run_test():
            host.library_ingest_jobs.restore([job], next_id=2)
            host._ingest_local_stt_jobs[job.job_id] = (7, "attempt-123")
            host._on_ingest_local_stt_failure(
                job.job_id,
                ExecutorFailure(
                    7, "attempt-123", TranscriptionFailureCode.ENGINE_CRASHED
                ),
            )
            assert (
                host.library_ingest_jobs.get_job(job.job_id).state
                is IngestJobState.FAILED
            )
    finally:
        logger.remove(sink)
    log = log_path.read_text()
    assert "phase=unknown" in log
    assert "phase-secret" not in log
