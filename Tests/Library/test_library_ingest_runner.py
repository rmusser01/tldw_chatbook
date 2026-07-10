"""App-level Library ingest queue-runner contracts (L3b Task 2).

Drives the real ``LibraryIngestQueueMixin`` (mixed into ``TldwCli`` in
``app.py``) through a minimal Textual ``App`` test-harness -- mirroring the
``LibraryHarness`` pattern in ``Tests/UI/test_library_shell.py`` -- against a
real file-backed ``MediaDatabase`` and real small ``.txt`` files. The full
``TldwCli`` is never booted; only the mixin + a real registry + a real
``media_db`` attribute are exercised, matching the runner's actual
production seam (``ingest_local_file`` / ``detect_file_type``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
from textual.app import App

from tldw_chatbook.app import LibraryIngestQueueMixin
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
)

# Bounded polling: never wait unboundedly for a background worker.
_POLL_ATTEMPTS = 300
_POLL_INTERVAL = 0.02


class _IngestRunnerHarness(LibraryIngestQueueMixin, App):
    """Minimal headless App hosting the ingest registry + queue-runner."""

    def __init__(self, media_db: Optional[MediaDatabase]) -> None:
        super().__init__()
        self.library_ingest_jobs = LibraryIngestJobRegistry()
        self.media_db = media_db


def _make_db(tmp_path: Path, name: str = "library_ingest.db") -> MediaDatabase:
    return MediaDatabase(tmp_path / name, client_id="l3b-runner-test")


def _write_text_file(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


async def _wait_for_job_state(
    app: _IngestRunnerHarness,
    pilot,
    job_id: str,
    state: IngestJobState,
    *,
    attempts: int = _POLL_ATTEMPTS,
) -> LibraryIngestJob:
    for _ in range(attempts):
        job = next((j for j in app.library_ingest_jobs.jobs() if j.job_id == job_id), None)
        if job is not None and job.state == state:
            return job
        await pilot.pause(_POLL_INTERVAL)
    all_jobs = app.library_ingest_jobs.jobs()
    raise AssertionError(f"job {job_id} never reached {state}. Jobs: {all_jobs}")


async def _wait_for_runner_idle(app: _IngestRunnerHarness, pilot, *, attempts: int = _POLL_ATTEMPTS) -> None:
    for _ in range(attempts):
        if not app.library_ingest_jobs.runner_active:
            return
        await pilot.pause(_POLL_INTERVAL)
    raise AssertionError("runner_active never returned to False")


@pytest.mark.asyncio
async def test_submit_reaches_done_with_real_media_id(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note-a.txt", "Tides are driven by the moon's gravity.")
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source), title="Note A")
        assert job.state == IngestJobState.QUEUED

        done = await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)

        assert done.media_id is not None
        row = db.get_media_by_id(done.media_id)
        assert row is not None
        assert row["title"] == "Note A"
        assert "moon's gravity" in row["content"]

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_two_submissions_run_serially_fifo(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    source1 = _write_text_file(tmp_path, "note-1.txt", "First document body.")
    source2 = _write_text_file(tmp_path, "note-2.txt", "Second document body.")
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        job1 = app.submit_library_ingest_job(source_path=str(source1))
        job2 = app.submit_library_ingest_job(source_path=str(source2))
        assert job1.job_id != job2.job_id

        max_running_seen = 0
        for _ in range(_POLL_ATTEMPTS):
            counts = app.library_ingest_jobs.counts()
            max_running_seen = max(max_running_seen, counts["running"])
            jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
            if (
                jobs_by_id[job1.job_id].state == IngestJobState.DONE
                and jobs_by_id[job2.job_id].state == IngestJobState.DONE
            ):
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"jobs never both completed: {app.library_ingest_jobs.jobs()}"
            )

        assert max_running_seen <= 1, "two jobs were RUNNING simultaneously"

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_failing_job_does_not_block_next_queued_job(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    missing = tmp_path / "does-not-exist.txt"
    ok_source = _write_text_file(tmp_path, "note-ok.txt", "This file exists just fine.")
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        failing_job = app.submit_library_ingest_job(source_path=str(missing))
        ok_job = app.submit_library_ingest_job(source_path=str(ok_source))

        failed = await _wait_for_job_state(app, pilot, failing_job.job_id, IngestJobState.FAILED)
        assert failed.error != ""
        assert len(failed.error) <= 200
        assert "\n" not in failed.error

        done = await _wait_for_job_state(app, pilot, ok_job.job_id, IngestJobState.DONE)
        assert done.media_id is not None

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_retry_of_failed_job_succeeds_once_file_exists(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    target = tmp_path / "arrives-later.txt"
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        failing_job = app.submit_library_ingest_job(source_path=str(target))
        failed = await _wait_for_job_state(app, pilot, failing_job.job_id, IngestJobState.FAILED)
        await _wait_for_runner_idle(app, pilot)

        # The file now exists at the same source_path the failed job used.
        target.write_text("Arrived just in time.", encoding="utf-8")

        requeued = app.retry_library_ingest_job(failed.job_id)
        assert requeued is not None
        assert requeued.job_id != failed.job_id
        assert requeued.state == IngestJobState.QUEUED

        done = await _wait_for_job_state(app, pilot, requeued.job_id, IngestJobState.DONE)
        assert done.media_id is not None
        row = db.get_media_by_id(done.media_id)
        assert row is not None
        assert "Arrived just in time" in row["content"]

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_submit_with_no_media_db_fails_immediately_without_starting_runner(
    tmp_path: Path,
) -> None:
    source = _write_text_file(tmp_path, "note.txt", "Irrelevant content.")
    app = _IngestRunnerHarness(None)

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))

        assert job.state == IngestJobState.FAILED
        assert job.error == "Media database is unavailable."

        # No runner should ever have started for this failure.
        await pilot.pause(_POLL_INTERVAL)
        await pilot.pause(_POLL_INTERVAL)
        assert app.library_ingest_jobs.runner_active is False


@pytest.mark.asyncio
async def test_listener_fires_on_every_state_change(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note-listener.txt", "Listener smoke content.")
    app = _IngestRunnerHarness(db)

    calls: list[int] = []
    app.library_ingest_jobs.add_listener(lambda: calls.append(1))

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        assert len(calls) == 1  # submit

        await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
        await _wait_for_runner_idle(app, pilot)

        # submit -> mark_running -> mark_done == 3 notifications.
        assert len(calls) == 3
