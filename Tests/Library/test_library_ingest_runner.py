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


# --- Task 2 review fix regression tests -------------------------------------
#
# The review found a check-then-exit race in the queue-runner's old exit
# path: it called `next_queued()` and, only in a *separate later*
# `call_from_thread` (inside `finally`), cleared `runner_active`. A
# submission landing on the UI thread in the gap between those two calls
# would append a QUEUED job while `runner_active` was still (stale-)`True`,
# so `_start_library_ingest_queue_if_idle` would never start a new runner --
# stranding the job. The fix collapses the check-and-clear into one atomic
# UI-thread call, `_claim_next_ingest_job_or_release`, and adds a
# crash-recovery safety net (`_release_ingest_runner_after_crash`) for the
# case where something bypasses that atomic exit entirely.
#
# The tests below combine:
#   (i)/(ii)  direct, synchronous calls to `_claim_next_ingest_job_or_release`
#             proving its atomic claim-or-release contract in isolation;
#   (iii)     an end-to-end test that forces the runner to hit its
#             crash-recovery `finally` path (not a per-job failure -- a
#             failure in the claim step itself, outside all per-job
#             isolation) with a second job still queued, proving the
#             runner notices and restarts itself instead of stranding the
#             queue -- this is the test that fails (times out) against the
#             pre-fix implementation;
#   (iv)      a coarse end-to-end stress smoke across five rapid
#             submissions (some landing while the runner is genuinely
#             mid-flight) as a gross-stranding catch-all.


@pytest.mark.asyncio
async def test_claim_next_job_returns_job_and_keeps_runner_active(tmp_path: Path) -> None:
    """(i) Direct-call contract: a queued job is returned and
    ``runner_active`` is left untouched (``True``), so the runner keeps
    looping instead of exiting.
    """
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note-claim.txt", "Body for the claim contract test.")
    app = _IngestRunnerHarness(db)

    async with app.run_test():
        # Simulate the runner already being active (as it always is by the
        # time anything calls the claim method for real) without actually
        # starting the background worker thread -- this keeps the test
        # fully synchronous and deterministic. Submitting straight through
        # the registry (bypassing `submit_library_ingest_job`) avoids
        # triggering `_start_library_ingest_queue_if_idle` for the same
        # reason.
        app.library_ingest_jobs.runner_active = True
        job = app.library_ingest_jobs.submit(source_path=str(source))

        claimed = app._claim_next_ingest_job_or_release()

        assert claimed is not None
        assert claimed.job_id == job.job_id
        assert app.library_ingest_jobs.runner_active is True


@pytest.mark.asyncio
async def test_claim_next_job_returns_none_and_clears_runner_active_when_empty(
    tmp_path: Path,
) -> None:
    """(ii) Direct-call contract: with no queued jobs, ``None`` is
    returned and ``runner_active`` is cleared in that same call -- the
    exact atomicity the exit-race fix depends on.
    """
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(db)

    async with app.run_test():
        app.library_ingest_jobs.runner_active = True

        claimed = app._claim_next_ingest_job_or_release()

        assert claimed is None
        assert app.library_ingest_jobs.runner_active is False


@pytest.mark.asyncio
async def test_finally_restarts_runner_after_unexpected_claim_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(iii) Regression test for the crash-recovery path -- the strongest
    deterministic proxy for the exit race available without depending on
    real thread-scheduling timing.

    Simulates a genuinely unexpected ("catastrophic") failure in the
    runner's own claim step, as opposed to a per-job ingest failure (which
    is already isolated by the inner try/except and covered by
    ``test_failing_job_does_not_block_next_queued_job``): the *first ever*
    call to ``next_queued()`` -- made from inside
    ``_claim_next_ingest_job_or_release``, at the top of the runner's
    loop, outside any per-job try/except -- raises once; every subsequent
    call behaves normally. The patch is installed before either
    submission, and no ``await`` happens between the two submissions, so
    the scheduled worker task cannot have started running yet by the time
    both jobs are queued -- the ordering (both jobs queued *before* the
    one-shot raise fires) is therefore deterministic, not a timing gamble.

    Pre-fix, the runner's ``finally`` unconditionally set
    ``runner_active = False`` via a plain ``setattr`` and exited for
    good -- it never rechecked the queue or restarted itself, so both
    jobs submitted before the crash would be stranded ``QUEUED`` forever
    (the ``_wait_for_job_state`` call below times out and raises
    ``AssertionError`` pre-fix, within its bounded ~6s poll budget).
    Post-fix, ``_release_ingest_runner_after_crash`` notices the queue is
    still non-empty and restarts the runner, so both jobs still reach
    ``DONE``.
    """
    db = _make_db(tmp_path)
    source1 = _write_text_file(tmp_path, "note-crash-1.txt", "First body, present when the crash hits.")
    source2 = _write_text_file(tmp_path, "note-crash-2.txt", "Second body, queued behind the crash.")
    app = _IngestRunnerHarness(db)

    real_next_queued = app.library_ingest_jobs.next_queued
    call_state = {"raised": False}

    def _flaky_next_queued():
        if not call_state["raised"]:
            call_state["raised"] = True
            raise RuntimeError("simulated catastrophic queue-runner failure")
        return real_next_queued()

    async with app.run_test() as pilot:
        monkeypatch.setattr(app.library_ingest_jobs, "next_queued", _flaky_next_queued)

        job1 = app.submit_library_ingest_job(source_path=str(source1))
        job2 = app.submit_library_ingest_job(source_path=str(source2))

        done1 = await _wait_for_job_state(app, pilot, job1.job_id, IngestJobState.DONE)
        assert done1.media_id is not None
        done2 = await _wait_for_job_state(app, pilot, job2.job_id, IngestJobState.DONE)
        assert done2.media_id is not None

        assert call_state["raised"] is True, "the simulated crash never fired -- test is not exercising the recovery path"

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_five_rapid_submissions_all_complete_no_stranding(tmp_path: Path) -> None:
    """(iv) End-to-end stress smoke: five jobs submitted in rapid
    succession -- some landing while the runner is genuinely mid-flight
    processing an earlier job -- must ALL reach DONE.

    This is a coarse, gross-stranding catch-all: if any submission landed
    in the (pre-fix) exit-race gap and got stuck behind a stale
    ``runner_active``, at least one job here would never leave QUEUED and
    the wait loop below would time out.
    """
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        sources = [
            _write_text_file(tmp_path, f"note-stress-{i}.txt", f"Stress body number {i}.")
            for i in range(5)
        ]

        jobs = [
            app.submit_library_ingest_job(source_path=str(sources[0])),
            app.submit_library_ingest_job(source_path=str(sources[1])),
        ]
        # Give the runner a chance to actually start draining before the
        # remaining submissions land -- the scenario most likely to hit
        # the exit-race gap is a submission arriving while the runner is
        # mid-loop, possibly right as it decides whether to exit.
        await pilot.pause(_POLL_INTERVAL)
        jobs.append(app.submit_library_ingest_job(source_path=str(sources[2])))
        await pilot.pause(_POLL_INTERVAL)
        jobs.append(app.submit_library_ingest_job(source_path=str(sources[3])))
        jobs.append(app.submit_library_ingest_job(source_path=str(sources[4])))

        for job in jobs:
            done = await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
            assert done.media_id is not None

        await _wait_for_runner_idle(app, pilot)
