"""App-level Library ingest coordinator + writer contracts (F3 Task 4).

Drives the real ``LibraryIngestQueueMixin`` (mixed into ``TldwCli`` in
``app.py``) through a minimal Textual ``App`` test-harness -- mirroring the
``LibraryHarness`` pattern in ``Tests/UI/test_library_shell.py`` -- against a
real file-backed ``MediaDatabase`` and real small ``.txt`` files. The full
``TldwCli`` is never booted; only the mixin + a real registry + a real
``media_db`` attribute are exercised.

F3 splits the old single-worker queue-runner into a parse-pool coordinator
(UI thread) and a narrowed single-writer worker. Real parsing/persistence
still runs through the production seam (``run_parse_job``/
``persist_parsed_media``), but the *pool* itself is faked (see
``_FakeIngestParsePool`` below) so these pilots stay fast and deterministic
without spawning real OS processes -- a real ``multiprocessing.get_context
("spawn").Pool`` is already covered end-to-end by Task 2's
``Tests/Local_Ingestion/test_ingest_parse_worker.py::
test_run_parse_job_through_real_spawn_pool`` (marked ``integration``); this
file's job is proving the *coordinator's* wiring (top-up, completion
handling, claim-or-release, broken-pool recovery, shutdown), not re-proving
that a spawned process can run ``run_parse_job``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import patch

import pytest
from textual.app import App

import tldw_chatbook.app as _app_module
from tldw_chatbook.app import LibraryIngestQueueMixin
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Bounded polling: never wait unboundedly for a background worker.
_POLL_ATTEMPTS = 300
_POLL_INTERVAL = 0.02
# Gated-fake pattern (plan's Global Constraints): bound every fake-pool join
# at 30.0s so a stuck background thread can never hang a test run.
_FAKE_POOL_JOIN_TIMEOUT = 30.0


class _FakeIngestParsePool:
    """Test-only, in-process stand-in for a real spawn ``Pool`` (F3 seam).

    Monkeypatched in via ``_IngestRunnerHarness._create_ingest_parse_pool``.
    Two modes:

    - ``auto_run=True`` (default): every ``apply_async`` call spawns a
      plain background ``threading.Thread`` that runs ``func`` (typically
      the real ``run_parse_job``, against a real small file -- fully
      in-process, no real subprocess) and invokes ``callback``/
      ``error_callback`` from that same thread. A background thread (never
      the caller's thread) is required: the coordinator's callbacks call
      ``App.call_from_thread``, which raises ``RuntimeError`` if invoked
      from the app's own (UI) thread.
    - ``auto_run=False`` ("manual" mode): ``apply_async`` only *records*
      the call (in ``self.calls``) and does nothing further -- the test
      drives completion explicitly via ``trigger_success``/
      ``trigger_error`` (also always on a background thread, for the same
      reason). Used for backpressure/broken-pool pilots that need to hold
      a job in ``PARSING`` under direct test control.
    """

    def __init__(self, *, auto_run: bool = True) -> None:
        self.auto_run = auto_run
        self.calls: list[dict[str, Any]] = []
        self.terminated = False
        # Thread ident `terminate()` was invoked on -- the quit-deadlock
        # pilots assert teardown runs OFF the app's event-loop thread.
        self.terminate_thread_ident: Optional[int] = None
        self._threads: list[threading.Thread] = []

    def apply_async(
        self,
        func: Callable[..., Any],
        args: tuple = (),
        kwds: Optional[dict] = None,
        callback: Optional[Callable[[Any], None]] = None,
        error_callback: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        record = {
            "func": func,
            "args": args,
            "kwds": kwds or {},
            "callback": callback,
            "error_callback": error_callback,
        }
        self.calls.append(record)
        if self.auto_run:
            self._spawn(self._run_one, record)

    def _run_one(self, record: dict[str, Any]) -> None:
        try:
            result = record["func"](*record["args"], **record["kwds"])
        except Exception as exc:  # noqa: BLE001 - mirrors a real Pool's error_callback path
            if record["error_callback"] is not None:
                record["error_callback"](exc)
            return
        if record["callback"] is not None:
            record["callback"](result)

    def trigger_success(self, index: int, result: Any) -> None:
        """Manually complete the ``index``-th ``apply_async`` call (manual mode)."""
        callback = self.calls[index]["callback"]
        if callback is not None:
            self._spawn(callback, result)

    def trigger_error(self, index: int, exc: BaseException) -> None:
        """Manually fail the ``index``-th ``apply_async`` call (manual mode)."""
        error_callback = self.calls[index]["error_callback"]
        if error_callback is not None:
            self._spawn(error_callback, exc)

    def _spawn(self, target: Callable[..., Any], *args: Any) -> None:
        thread = threading.Thread(target=target, args=args, daemon=True)
        self._threads.append(thread)
        thread.start()

    def terminate(self) -> None:
        self.terminated = True
        self.terminate_thread_ident = threading.get_ident()

    def join(self) -> None:
        for thread in self._threads:
            thread.join(timeout=_FAKE_POOL_JOIN_TIMEOUT)

    def close(self) -> None:
        pass


class _IngestRunnerHarness(LibraryIngestQueueMixin, App):
    """Minimal headless App hosting the ingest registry + coordinator + writer.

    Defaults to an auto-run ``_FakeIngestParsePool`` (real ``run_parse_job``/
    ``persist_parsed_media``, fake pool) so pilots never spawn real OS
    processes. Tests needing manual control over completion timing (backpressure,
    broken-pool) pass their own ``pool_factory``; tests needing a specific
    worker cap pass ``worker_count``.
    """

    def __init__(
        self,
        media_db: Optional[MediaDatabase],
        *,
        pool_factory: Optional[Callable[[], Any]] = None,
        worker_count: Optional[int] = None,
        heavy_lane: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.library_ingest_jobs = LibraryIngestJobRegistry()
        self.media_db = media_db
        self._ingest_parse_pool = None
        self._ingest_parse_pool_generation = 0
        self._ingest_parse_jobs_by_generation: dict[int, set[str]] = {}
        self._ingest_parse_pool_stop_event: Optional[threading.Event] = None
        self._ingest_parsed_payloads: dict[str, dict] = {}
        self._ingest_shutdown = False
        self._pool_factory = pool_factory or (lambda: _FakeIngestParsePool())
        self._pool_create_count = 0
        self._worker_count_override = worker_count
        self._heavy_lane_override = heavy_lane

    def _create_ingest_parse_pool(self):
        self._pool_create_count += 1
        return self._pool_factory()

    def _ingest_parse_worker_count(self) -> int:
        if self._worker_count_override is not None:
            return self._worker_count_override
        return super()._ingest_parse_worker_count()

    def _ingest_heavy_lane_max_workers(self) -> int:
        if self._heavy_lane_override is not None:
            return self._heavy_lane_override
        return super()._ingest_heavy_lane_max_workers()


def _make_db(tmp_path: Path, name: str = "library_ingest.db") -> MediaDatabase:
    return MediaDatabase(tmp_path / name, client_id="f3-runner-test")


def _write_text_file(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _exit_ingest_worker_abruptly() -> None:
    """Picklable spawn-pool target that simulates a hard worker crash."""
    os._exit(17)


async def _wait_for_job_state(
    app: _IngestRunnerHarness,
    pilot,
    job_id: str,
    state: IngestJobState,
    *,
    attempts: int = _POLL_ATTEMPTS,
) -> LibraryIngestJob:
    for _ in range(attempts):
        job = next(
            (j for j in app.library_ingest_jobs.jobs() if j.job_id == job_id), None
        )
        if job is not None and job.state == state:
            return job
        await pilot.pause(_POLL_INTERVAL)
    all_jobs = app.library_ingest_jobs.jobs()
    raise AssertionError(f"job {job_id} never reached {state}. Jobs: {all_jobs}")


async def _wait_for_runner_idle(
    app: _IngestRunnerHarness, pilot, *, attempts: int = _POLL_ATTEMPTS
) -> None:
    for _ in range(attempts):
        if not app.library_ingest_jobs.runner_active:
            return
        await pilot.pause(_POLL_INTERVAL)
    raise AssertionError("runner_active never returned to False")


@pytest.mark.asyncio
async def test_submit_reaches_done_with_real_media_id(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(
        tmp_path, "note-a.txt", "Tides are driven by the moon's gravity."
    )
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
async def test_submitting_a_directory_queues_one_job_per_file(
    tmp_path: Path,
) -> None:
    """A folder must expand into per-file jobs, not one job for the folder.

    Pre-flight happily reports "4 plain text files" for a directory and lets
    the user start it, but the runner used to classify the directory itself
    and fail the whole submission with "Unsupported file type: ." -- so the
    batch import the UI advertises never worked at all (task-675).
    """
    db = _make_db(tmp_path)
    folder = tmp_path / "batch"
    folder.mkdir()
    for index in range(3):
        _write_text_file(folder, f"doc-{index}.txt", f"Body of document {index}.")
    app = _IngestRunnerHarness(db, worker_count=2)

    async with app.run_test() as pilot:
        app.submit_library_ingest_job(source_path=str(folder))

        jobs = app.library_ingest_jobs.jobs()
        assert len(jobs) == 3, f"expected one job per file, got {len(jobs)}"
        assert {Path(job.source_path).name for job in jobs} == {
            "doc-0.txt",
            "doc-1.txt",
            "doc-2.txt",
        }

        for job in jobs:
            await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_directory_unsupported_file_fails_alone(tmp_path: Path) -> None:
    """One unsupported file in a folder fails on its own row.

    The supported siblings must still reach DONE rather than being taken
    down with it.
    """
    db = _make_db(tmp_path)
    folder = tmp_path / "mixed"
    folder.mkdir()
    _write_text_file(folder, "good.txt", "A perfectly ingestible document.")
    (folder / "cover.jpg").write_bytes(b"not really a jpeg")
    app = _IngestRunnerHarness(db, worker_count=2)

    async with app.run_test() as pilot:
        app.submit_library_ingest_job(source_path=str(folder))

        jobs = {Path(j.source_path).name: j for j in app.library_ingest_jobs.jobs()}
        assert set(jobs) == {"good.txt", "cover.jpg"}

        await _wait_for_job_state(
            app, pilot, jobs["good.txt"].job_id, IngestJobState.DONE
        )
        await _wait_for_job_state(
            app, pilot, jobs["cover.jpg"].job_id, IngestJobState.FAILED
        )

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_directory_submission_honours_scan_limit(tmp_path: Path) -> None:
    """The folder expansion respects the configured directory scan limit."""
    db = _make_db(tmp_path)
    folder = tmp_path / "many"
    folder.mkdir()
    for index in range(6):
        _write_text_file(folder, f"doc-{index}.txt", f"Body {index}.")
    app = _IngestRunnerHarness(db, worker_count=2)

    from tldw_chatbook.app import get_cli_setting as _real_get_cli_setting

    def _limited(*args: Any, **kwargs: Any) -> Any:
        if args and args[0] == "library.ingest_directory_scan_limit":
            return 2
        return _real_get_cli_setting(*args, **kwargs)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_limited):
        async with app.run_test() as pilot:
            app.submit_library_ingest_job(source_path=str(folder))
            assert len(app.library_ingest_jobs.jobs()) == 2

            for job in app.library_ingest_jobs.jobs():
                await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
            await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_two_files_queued_both_reach_done_with_real_media_rows(
    tmp_path: Path,
) -> None:
    """(F3 pilot) Two files queued -> both reach DONE with real media rows,
    and the write stage (SQLite single-writer) never has two jobs WRITING
    at the same time -- even though both may PARSE concurrently (forced to
    N=2 here so both jobs' parses genuinely overlap)."""
    db = _make_db(tmp_path)
    source1 = _write_text_file(tmp_path, "note-1.txt", "First document body.")
    source2 = _write_text_file(tmp_path, "note-2.txt", "Second document body.")
    app = _IngestRunnerHarness(db, worker_count=2)

    async with app.run_test() as pilot:
        job1 = app.submit_library_ingest_job(source_path=str(source1))
        job2 = app.submit_library_ingest_job(source_path=str(source2))
        assert job1.job_id != job2.job_id

        max_writing_seen = 0
        for _ in range(_POLL_ATTEMPTS):
            counts = app.library_ingest_jobs.counts()
            max_writing_seen = max(max_writing_seen, counts["writing"])
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

        assert max_writing_seen <= 1, "two jobs were WRITING simultaneously"

        jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
        done1, done2 = jobs_by_id[job1.job_id], jobs_by_id[job2.job_id]
        assert done1.media_id is not None
        assert done2.media_id is not None
        assert db.get_media_by_id(done1.media_id) is not None
        assert db.get_media_by_id(done2.media_id) is not None

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

        failed = await _wait_for_job_state(
            app, pilot, failing_job.job_id, IngestJobState.FAILED
        )
        assert failed.error != ""
        assert len(failed.error) <= 200
        assert "\n" not in failed.error

        done = await _wait_for_job_state(app, pilot, ok_job.job_id, IngestJobState.DONE)
        assert done.media_id is not None

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_retry_of_failed_job_succeeds_once_transient_error_clears(
    tmp_path: Path, monkeypatch
) -> None:
    """(M4 re-anchor, fix batch F1b; F3 re-anchor) A requeued job can reach
    DONE once whatever caused the first failure is gone. F3: the pool
    worker entry point (``run_parse_job``) never raises across the process
    boundary -- it always returns a structured result -- so a *transient*
    per-job parse failure is simulated the same way a real worker would
    report one: a structured ``{"ok": False, ...}`` result, not a raised
    exception (which the coordinator's ``error_callback`` path would
    instead treat as a POOL-level failure -- see the broken-pool pilot
    below -- since ``run_parse_job`` is contractually never supposed to
    raise)."""
    db = _make_db(tmp_path)
    target = _write_text_file(tmp_path, "arrives-later.txt", "Arrived just in time.")
    app = _IngestRunnerHarness(db)

    import tldw_chatbook.app as app_module

    real_run_parse_job = app_module.run_parse_job
    call_count = {"n": 0}

    def _flaky_run_parse_job(file_path, options):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"ok": False, "error": "transient parse hiccup", "permanent": False}
        return real_run_parse_job(file_path, options)

    monkeypatch.setattr(app_module, "run_parse_job", _flaky_run_parse_job)

    async with app.run_test() as pilot:
        failing_job = app.submit_library_ingest_job(source_path=str(target))
        failed = await _wait_for_job_state(
            app, pilot, failing_job.job_id, IngestJobState.FAILED
        )
        assert failed.permanent is False
        assert failed.error == "transient parse hiccup"
        await _wait_for_runner_idle(app, pilot)

        requeued = app.retry_library_ingest_job(failed.job_id)
        assert requeued is not None
        assert requeued.job_id != failed.job_id
        assert requeued.state == IngestJobState.QUEUED

        done = await _wait_for_job_state(
            app, pilot, requeued.job_id, IngestJobState.DONE
        )
        assert done.media_id is not None
        row = db.get_media_by_id(done.media_id)
        assert row is not None
        assert "Arrived just in time" in row["content"]

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_missing_file_failure_is_permanent_and_refuses_retry(
    tmp_path: Path,
) -> None:
    """(M4, fix batch F1b) A ``FileNotFoundError`` from the parse worker
    fails the exact same way on every attempt -- classified ``permanent``
    inside ``run_parse_job`` (F3: the real exception type is only visible
    inside the worker) -- and ``retry_library_ingest_job`` must refuse it
    (defense in depth, on top of the canvas withholding the Retry button
    entirely for a permanent row)."""
    db = _make_db(tmp_path)
    missing = tmp_path / "does-not-exist.txt"
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        failing_job = app.submit_library_ingest_job(source_path=str(missing))
        failed = await _wait_for_job_state(
            app, pilot, failing_job.job_id, IngestJobState.FAILED
        )
        await _wait_for_runner_idle(app, pilot)

        assert failed.permanent is True
        assert app.retry_library_ingest_job(failed.job_id) is None


@pytest.mark.asyncio
async def test_unsupported_file_type_failure_is_permanent_and_refuses_retry(
    tmp_path: Path,
) -> None:
    """(M4) An unsupported extension is a validation-class failure too --
    classified ``permanent`` inside the parse worker, Retry refused."""
    db = _make_db(tmp_path)
    unsupported = _write_text_file(tmp_path, "note.xyz", "irrelevant content")
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        failing_job = app.submit_library_ingest_job(source_path=str(unsupported))
        failed = await _wait_for_job_state(
            app, pilot, failing_job.job_id, IngestJobState.FAILED
        )
        await _wait_for_runner_idle(app, pilot)

        assert failed.permanent is True
        assert "Unsupported file type" in failed.error
        assert app.retry_library_ingest_job(failed.job_id) is None


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

        # No runner should ever have started for this failure, and the
        # parse pool must never even have been created.
        await pilot.pause(_POLL_INTERVAL)
        await pilot.pause(_POLL_INTERVAL)
        assert app.library_ingest_jobs.runner_active is False
        assert app._pool_create_count == 0


@pytest.mark.asyncio
async def test_listener_fires_on_every_state_change(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note-listener.txt", "Listener smoke content.")
    app = _IngestRunnerHarness(db)

    calls: list[int] = []
    app.library_ingest_jobs.add_listener(lambda: calls.append(1))

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        # (F3) submit_library_ingest_job's own top-up call synchronously
        # claims the job into PARSING before returning -- submit(1) +
        # mark_parsing(1) land inside this one call.
        assert len(calls) == 2

        await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
        await _wait_for_runner_idle(app, pilot)

        # (F3) submit -> mark_parsing -> mark_writing -> mark_done ==
        # 4 real, separate transitions (parsing and writing are now
        # distinct pipeline stages, not one aliased call).
        assert len(calls) == 4


# --- F3 Task 4: coordinator + writer pilots ---------------------------------


@pytest.mark.asyncio
async def test_broken_pool_fails_all_parsing_jobs_and_rebuilds_on_next_submit(
    tmp_path: Path,
) -> None:
    """(F3 pilot) A pool-level failure (e.g. a worker process died) must
    fail EVERY currently-``PARSING`` job as retryable and drop the pool --
    the next submission lazily rebuilds a fresh one."""
    db = _make_db(tmp_path)
    source1 = _write_text_file(tmp_path, "note-1.txt", "First body.")
    source2 = _write_text_file(tmp_path, "note-2.txt", "Second body.")

    pools: list[_FakeIngestParsePool] = []

    def _pool_factory() -> _FakeIngestParsePool:
        pool = _FakeIngestParsePool(auto_run=False)
        pools.append(pool)
        return pool

    app = _IngestRunnerHarness(db, pool_factory=_pool_factory, worker_count=2)

    async with app.run_test() as pilot:
        job1 = app.submit_library_ingest_job(source_path=str(source1))
        job2 = app.submit_library_ingest_job(source_path=str(source2))
        await pilot.pause()

        jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
        assert jobs_by_id[job1.job_id].state == IngestJobState.PARSING
        assert jobs_by_id[job2.job_id].state == IngestJobState.PARSING
        assert len(pools) == 1
        first_pool = pools[0]
        assert len(first_pool.calls) == 2

        # Simulate the pool dying: fire error_callback for ONE of the two
        # in-flight calls -- this must fail BOTH currently-PARSING jobs
        # (not just the one tied to this specific callback), since neither
        # can be trusted to ever complete on a broken pool.
        first_pool.trigger_error(0, RuntimeError("simulated worker death"))

        for _ in range(_POLL_ATTEMPTS):
            jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
            if (
                jobs_by_id[job1.job_id].state == IngestJobState.FAILED
                and jobs_by_id[job2.job_id].state == IngestJobState.FAILED
            ):
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"both jobs never reached FAILED: {app.library_ingest_jobs.jobs()}"
            )

        jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
        assert jobs_by_id[job1.job_id].permanent is False
        assert jobs_by_id[job2.job_id].permanent is False
        assert app._ingest_parse_pool is None

        # Retry -- the next submission must rebuild a fresh pool.
        requeued = app.retry_library_ingest_job(job1.job_id)
        assert requeued is not None
        assert len(pools) == 2, (
            "a fresh pool must be created lazily on the next submission"
        )


@pytest.mark.asyncio
async def test_stale_pool_generation_error_does_not_fail_replacement_job(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "stale-error.txt", "Generation isolation.")
    pools: list[_FakeIngestParsePool] = []

    def _pool_factory() -> _FakeIngestParsePool:
        pool = _FakeIngestParsePool(auto_run=False)
        pools.append(pool)
        return pool

    app = _IngestRunnerHarness(db, pool_factory=_pool_factory, worker_count=1)

    async with app.run_test() as pilot:
        original = app.submit_library_ingest_job(source_path=str(source))
        await pilot.pause()
        generation_a = pools[0]
        generation_a.trigger_error(0, RuntimeError("generation A died"))
        await _wait_for_job_state(app, pilot, original.job_id, IngestJobState.FAILED)

        replacement = app.retry_library_ingest_job(original.job_id)
        assert replacement is not None
        await _wait_for_job_state(
            app, pilot, replacement.job_id, IngestJobState.PARSING
        )
        assert len(pools) == 2

        generation_a.trigger_error(0, RuntimeError("late generation A error"))
        await pilot.pause(_POLL_INTERVAL)
        current = next(
            job
            for job in app.library_ingest_jobs.jobs()
            if job.job_id == replacement.job_id
        )
        assert current.state == IngestJobState.PARSING
        assert app._ingest_parse_pool is pools[1]


@pytest.mark.asyncio
async def test_stale_pool_generation_success_does_not_store_payload(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "stale-success.txt", "Generation isolation.")
    pools: list[_FakeIngestParsePool] = []

    def _pool_factory() -> _FakeIngestParsePool:
        pool = _FakeIngestParsePool(auto_run=False)
        pools.append(pool)
        return pool

    app = _IngestRunnerHarness(db, pool_factory=_pool_factory, worker_count=1)

    async with app.run_test() as pilot:
        original = app.submit_library_ingest_job(source_path=str(source))
        await pilot.pause()
        generation_a = pools[0]
        generation_a.trigger_error(0, RuntimeError("generation A died"))
        await _wait_for_job_state(app, pilot, original.job_id, IngestJobState.FAILED)

        replacement = app.retry_library_ingest_job(original.job_id)
        assert replacement is not None
        await _wait_for_job_state(
            app, pilot, replacement.job_id, IngestJobState.PARSING
        )

        generation_a.trigger_success(
            0,
            {"ok": True, "payload": {"file_type": "plaintext", "content": "stale"}},
        )
        await pilot.pause(_POLL_INTERVAL)

        assert original.job_id not in app._ingest_parsed_payloads
        assert replacement.job_id not in app._ingest_parsed_payloads
        assert app.library_ingest_jobs.runner_active is False
        current = next(
            job
            for job in app.library_ingest_jobs.jobs()
            if job.job_id == replacement.job_id
        )
        assert current.state == IngestJobState.PARSING


@pytest.mark.asyncio
async def test_real_pool_worker_exit_is_reported_for_owning_generation(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "hard-exit.txt", "Worker sentinel coverage.")
    app: _IngestRunnerHarness
    app = _IngestRunnerHarness(
        db,
        pool_factory=lambda: LibraryIngestQueueMixin._create_ingest_parse_pool(app),
        worker_count=1,
    )
    pool = None
    teardown_threads: list[threading.Thread] = []
    real_terminate = app._terminate_ingest_parse_pool_off_thread

    def _capture_production_teardown(target_pool: Any) -> threading.Thread:
        thread = real_terminate(target_pool)
        teardown_threads.append(thread)
        return thread

    app._terminate_ingest_parse_pool_off_thread = _capture_production_teardown

    try:
        async with app.run_test() as pilot:
            job = app.library_ingest_jobs.submit(source_path=str(source))
            claimed = app.library_ingest_jobs.mark_parsing(job.job_id)
            assert claimed is not None

            pool = app._ensure_ingest_parse_pool()
            generation = app._ingest_parse_pool_generation
            app._ingest_parse_jobs_by_generation[generation].add(job.job_id)
            pool.apply_async(_exit_ingest_worker_abruptly)

            failed = await _wait_for_job_state(
                app,
                pilot,
                job.job_id,
                IngestJobState.FAILED,
                attempts=500,
            )
            assert failed.permanent is False
            assert app._ingest_parse_pool is None
            assert generation not in app._ingest_parse_jobs_by_generation
    finally:
        if pool is not None:
            if teardown_threads:
                cleanup = teardown_threads[0]
            else:
                pool.terminate()
                cleanup = threading.Thread(target=pool.join, daemon=True)
                cleanup.start()
            cleanup.join(timeout=10.0)
            assert not cleanup.is_alive(), "real Pool cleanup exceeded 10 seconds"


def test_current_generation_sentinel_failure_retires_idle_pool(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool, worker_count=1)
    app._ensure_ingest_parse_pool()
    generation = app._ingest_parse_pool_generation
    assert app._ingest_parse_jobs_by_generation[generation] == set()

    app._handle_broken_ingest_parse_pool(
        generation,
        None,
        RuntimeError("idle worker exited"),
    )

    assert app._ingest_parse_pool is None
    assert generation not in app._ingest_parse_jobs_by_generation
    for _ in range(100):
        if pool.terminated:
            break
        threading.Event().wait(0.01)
    assert pool.terminated is True


def test_ensure_pool_initializes_generation_state_for_existing_host(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool, worker_count=1)
    del app._ingest_parse_pool_generation
    del app._ingest_parse_jobs_by_generation
    del app._ingest_parse_pool_stop_event

    assert app._ensure_ingest_parse_pool() is pool
    assert app._ingest_parse_pool_generation == 1
    assert app._ingest_parse_jobs_by_generation == {1: set()}
    assert isinstance(app._ingest_parse_pool_stop_event, threading.Event)

    teardown = app._shutdown_ingest_parse_pool()
    assert teardown is not None
    teardown.join(timeout=_FAKE_POOL_JOIN_TIMEOUT)
    assert not teardown.is_alive()


@pytest.mark.asyncio
async def test_submit_cap_backpressure_second_job_stays_queued_until_first_completes(
    tmp_path: Path,
) -> None:
    """(F3 pilot) The pool-size cap IS the backpressure: with N=1, a second
    submission must stay QUEUED until the first job's parse completes."""
    db = _make_db(tmp_path)
    source1 = _write_text_file(tmp_path, "note-1.txt", "First body.")
    source2 = _write_text_file(tmp_path, "note-2.txt", "Second body.")

    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool, worker_count=1)

    async with app.run_test() as pilot:
        job1 = app.submit_library_ingest_job(source_path=str(source1))
        job2 = app.submit_library_ingest_job(source_path=str(source2))
        await pilot.pause()

        jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
        assert jobs_by_id[job1.job_id].state == IngestJobState.PARSING
        assert jobs_by_id[job2.job_id].state == IngestJobState.QUEUED
        assert len(pool.calls) == 1

        # Manually complete job1's parse -- this must top up the pool and
        # promote job2 to PARSING.
        payload = {
            "media_type": "plaintext",
            "file_type": "plaintext",
            "title": "note-1",
            "author": "Unknown",
            "content": "First body.",
            "keywords": [],
            "url": f"file://{source1.absolute()}",
            "analysis_content": "",
            "chunks": None,
            "chunk_options": None,
            "metadata": {},
            "file_path": str(source1),
        }
        pool.trigger_success(0, {"ok": True, "payload": payload})

        for _ in range(_POLL_ATTEMPTS):
            jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
            if jobs_by_id[job2.job_id].state == IngestJobState.PARSING:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"job2 never promoted to PARSING: {app.library_ingest_jobs.jobs()}"
            )

        assert len(pool.calls) == 2

        done1 = await _wait_for_job_state(app, pilot, job1.job_id, IngestJobState.DONE)
        assert done1.media_id is not None
        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_heavy_lane_caps_transcriptions_while_documents_fill_pool(
    tmp_path: Path,
) -> None:
    """(F3 pilot, heavy-lane cap) With worker_count=3 and heavy_lane=1, only
    one audio/video parse may run at a time -- a second transcription is
    skipped ahead of by queued documents, which fill the remaining pool
    slots. Completing the in-flight transcription frees the heavy slot and
    promotes the skipped one."""
    db = _make_db(tmp_path)
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(
        db, pool_factory=lambda: pool, worker_count=3, heavy_lane=1
    )

    async with app.run_test() as pilot:
        paths = {}
        for name in ("a1.mp3", "a2.mp3", "d1.txt", "d2.txt", "d3.txt"):
            p = tmp_path / name
            p.write_text("x", encoding="utf-8")
            paths[name] = app.submit_library_ingest_job(source_path=str(p))
        await pilot.pause()

        # pool holds exactly 3: audio1 (heavy) + doc1 + doc2. audio2 is
        # skipped (heavy lane full); doc3 waits (pool full).
        assert len(pool.calls) == 3
        states = {j.job_id: j.state for j in app.library_ingest_jobs.jobs()}
        assert states[paths["a1.mp3"].job_id] == IngestJobState.PARSING
        assert states[paths["d1.txt"].job_id] == IngestJobState.PARSING
        assert states[paths["d2.txt"].job_id] == IngestJobState.PARSING
        assert states[paths["a2.mp3"].job_id] == IngestJobState.QUEUED
        assert states[paths["d3.txt"].job_id] == IngestJobState.QUEUED

        # completing audio1 frees the heavy slot -> audio2 is admitted next.
        pool.trigger_success(0, {"ok": True, "payload": {}})
        await _wait_for_job_state(
            app, pilot, paths["a2.mp3"].job_id, IngestJobState.PARSING
        )
        assert len(pool.calls) == 4


@pytest.mark.asyncio
async def test_retried_transcription_still_obeys_the_heavy_lane_cap(
    tmp_path: Path,
) -> None:
    """(task 160) The heavy-lane cap must hold on the retry path too. A
    requeued audio job carries its ``detected_type`` forward, so retrying a
    failed transcription while another transcription is already PARSING must
    leave the retry QUEUED -- not dispatch a second concurrent transcription
    (which the dropped-``detected_type`` bug allowed via the Home Retry
    control)."""
    db = _make_db(tmp_path)
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(
        db, pool_factory=lambda: pool, worker_count=3, heavy_lane=1
    )

    async with app.run_test() as pilot:
        a1_path = tmp_path / "a1.mp3"
        a1_path.write_text("x", encoding="utf-8")
        a2_path = tmp_path / "a2.mp3"
        a2_path.write_text("x", encoding="utf-8")
        a1 = app.submit_library_ingest_job(source_path=str(a1_path))
        a2 = app.submit_library_ingest_job(source_path=str(a2_path))
        await pilot.pause()

        # Only one transcription parses at a time: a1 PARSING, a2 blocked.
        assert len(pool.calls) == 1
        states = {j.job_id: j.state for j in app.library_ingest_jobs.jobs()}
        assert states[a1.job_id] == IngestJobState.PARSING
        assert states[a2.job_id] == IngestJobState.QUEUED

        # Fail a1 (per-job structured failure, like the retry tests) -> its
        # heavy slot frees and a2 is admitted to PARSING.
        pool.trigger_success(0, {"ok": False, "error": "boom", "permanent": False})
        await _wait_for_job_state(app, pilot, a2.job_id, IngestJobState.PARSING)
        await _wait_for_job_state(app, pilot, a1.job_id, IngestJobState.FAILED)
        assert len(pool.calls) == 2

        # Retry a1 while a2 is still PARSING: the heavy lane is full, so the
        # requeued a1 must stay QUEUED (its detected_type='audio' is skipped),
        # NOT be dispatched as a second concurrent transcription.
        requeued = app.retry_library_ingest_job(a1.job_id)
        assert requeued is not None
        await pilot.pause()
        assert len(pool.calls) == 2
        states_after = {j.job_id: j.state for j in app.library_ingest_jobs.jobs()}
        assert states_after[requeued.job_id] == IngestJobState.QUEUED
        assert states_after[a2.job_id] == IngestJobState.PARSING


@pytest.mark.asyncio
async def test_shutdown_flag_stops_late_parse_completion_callbacks(
    tmp_path: Path,
) -> None:
    """(F3 pilot) Once ``_ingest_shutdown`` is set, a parse completion (or
    pool-level error) that lands afterward -- e.g. already in flight when
    the app started closing -- must be a pure no-op: no registry mutation,
    no pool top-up, no pool drop."""
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note.txt", "irrelevant")
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool)

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        await pilot.pause()

        current = next(
            j for j in app.library_ingest_jobs.jobs() if j.job_id == job.job_id
        )
        assert current.state == IngestJobState.PARSING

        app._ingest_shutdown = True

        # Late success completion: must be a no-op.
        app._on_ingest_parse_complete(
            app._ingest_parse_pool_generation,
            job.job_id,
            {"ok": True, "payload": {"file_type": "plaintext"}},
        )
        current = next(
            j for j in app.library_ingest_jobs.jobs() if j.job_id == job.job_id
        )
        assert current.state == IngestJobState.PARSING
        assert job.job_id not in app._ingest_parsed_payloads
        assert app.library_ingest_jobs.runner_active is False

        # Late pool-level error: also a no-op -- the pool is not dropped.
        app._handle_broken_ingest_parse_pool(
            app._ingest_parse_pool_generation,
            job.job_id,
            RuntimeError("late failure"),
        )
        current = next(
            j for j in app.library_ingest_jobs.jobs() if j.job_id == job.job_id
        )
        assert current.state == IngestJobState.PARSING
        assert app._ingest_parse_pool is not None


# --- F3 Task 4 review fixes: quit-deadlock guard + payload-sparing ----------
#
# The Task 4 review found a quit-time deadlock race: Textual's
# `call_from_thread` blocks the calling thread on the marshaled call's
# result, and CPython's `Pool._terminate_pool` does an unbounded
# `result_handler.join()`. If a parse completed right as the user quit, the
# pool's result-handler thread could park inside `call_from_thread` while
# `on_unmount` (the loop thread) parked inside `pool.terminate()` waiting
# for that same result-handler thread -- mutual deadlock, app hangs on
# quit. The deadlock itself is race-timed, so these tests pin the two
# OBSERVABLE contracts of the fix instead: (a) the pool-side callbacks
# check `_ingest_shutdown` BEFORE marshaling (never entering
# `call_from_thread` at all once the flag is up), and (b) quit-path
# terminate/join runs on a detached daemon thread, never the caller's
# (loop) thread, so the loop stays free to drain any in-flight marshaled
# call.


def test_pool_callbacks_short_circuit_without_marshaling_when_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(Quit-deadlock guard, layer a) `_ingest_pool_callback`/
    `_ingest_pool_error_callback` -- which run on the pool's result-handler
    thread -- must return WITHOUT calling `call_from_thread` once
    `_ingest_shutdown` is set. The pre-fix lambdas marshaled
    unconditionally (the shutdown check only ran later, inside the
    already-marshaled UI-thread body -- too late to prevent the
    result-handler thread from blocking)."""
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(db)

    marshaled: list[tuple] = []
    monkeypatch.setattr(
        app, "call_from_thread", lambda *args, **kwargs: marshaled.append(args)
    )

    app._ingest_shutdown = True
    app._ingest_pool_callback(1, "ingest-job-1", {"ok": True, "payload": {}})
    app._ingest_pool_error_callback(
        1, "ingest-job-1", RuntimeError("late pool failure")
    )
    assert marshaled == []

    # Positive control: with the flag down, both callbacks marshal.
    app._ingest_shutdown = False
    app._ingest_pool_callback(1, "ingest-job-1", {"ok": True, "payload": {}})
    app._ingest_pool_error_callback(1, "ingest-job-1", RuntimeError("pool failure"))
    assert len(marshaled) == 2


def test_shutdown_terminates_pool_off_the_caller_thread(tmp_path: Path) -> None:
    """(Quit-deadlock guard, layer b) `_shutdown_ingest_parse_pool` must
    set the shutdown flag, detach the pool reference, and run
    `terminate()`/`join()` on a DIFFERENT thread than the caller's (in
    production the caller is `on_unmount`, i.e. the app's event-loop
    thread -- exactly the thread that must never block on the pool's
    result-handler join)."""
    db = _make_db(tmp_path)
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool)
    app._ingest_parse_pool = pool

    caller_ident = threading.get_ident()
    teardown_thread = app._shutdown_ingest_parse_pool()

    # Synchronous effects, guaranteed before the method returns: flag up,
    # pool reference detached (nothing can submit to it anymore).
    assert app._ingest_shutdown is True
    assert app._ingest_parse_pool is None

    assert teardown_thread is not None
    assert teardown_thread.daemon is True
    teardown_thread.join(timeout=_FAKE_POOL_JOIN_TIMEOUT)
    assert not teardown_thread.is_alive()
    assert pool.terminated is True
    assert pool.terminate_thread_ident is not None
    assert pool.terminate_thread_ident != caller_ident


def test_shutdown_with_no_pool_still_sets_flag_and_returns_none(tmp_path: Path) -> None:
    """`_shutdown_ingest_parse_pool` with no pool ever created: the flag
    still goes up (late callbacks must no-op regardless), no thread is
    spawned."""
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(db)

    assert app._shutdown_ingest_parse_pool() is None
    assert app._ingest_shutdown is True


@pytest.mark.asyncio
async def test_broken_pool_spares_payload_ready_job_and_writer_drains_it(
    tmp_path: Path,
) -> None:
    """(Task 4 review fix) A job whose parse already COMPLETED (payload
    sitting in `_ingest_parsed_payloads`, job still `PARSING` because the
    writer hasn't claimed it yet) needs nothing further from the pool --
    a pool-level failure must NOT fail it and throw the finished parse
    away. Only jobs still genuinely mid-parse fail (retryable); the
    handler wakes the writer so the surviving payload drains to DONE."""
    db = _make_db(tmp_path)
    source_a = _write_text_file(tmp_path, "note-a.txt", "Payload-ready body.")
    source_b = _write_text_file(tmp_path, "note-b.txt", "Mid-parse body.")
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool, worker_count=2)

    async with app.run_test() as pilot:
        job_a = app.submit_library_ingest_job(source_path=str(source_a))
        job_b = app.submit_library_ingest_job(source_path=str(source_b))
        await pilot.pause()

        jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
        assert jobs_by_id[job_a.job_id].state == IngestJobState.PARSING
        assert jobs_by_id[job_b.job_id].state == IngestJobState.PARSING

        # Make job A payload-ready WITHOUT routing through
        # `_on_ingest_parse_complete` (which would wake the writer and let
        # it claim A out of PARSING before the pool breaks): stash the
        # payload directly -- exactly the state a real completion leaves
        # behind in the window before the writer's claim lands.
        payload_a = {
            "media_type": "plaintext",
            "file_type": "plaintext",
            "title": "note-a",
            "author": "Unknown",
            "content": "Payload-ready body.",
            "keywords": [],
            "url": f"file://{source_a.absolute()}",
            "analysis_content": "",
            "chunks": None,
            "chunk_options": None,
            "metadata": {},
            "file_path": str(source_a),
        }
        app._ingest_parsed_payloads[job_a.job_id] = payload_a

        # The pool dies while B is still genuinely mid-parse.
        pool.trigger_error(0, RuntimeError("simulated worker death"))

        failed_b = await _wait_for_job_state(
            app, pilot, job_b.job_id, IngestJobState.FAILED
        )
        assert failed_b.permanent is False

        # A must survive: never failed, and the handler's writer wake
        # drains its already-finished parse to DONE with a real media row.
        done_a = await _wait_for_job_state(
            app, pilot, job_a.job_id, IngestJobState.DONE
        )
        assert done_a.media_id is not None
        assert db.get_media_by_id(done_a.media_id) is not None

        assert app._ingest_parse_pool is None
        await _wait_for_runner_idle(app, pilot)


# --- Live-QA crash fix: Textual's fileno-less stderr vs. the resource tracker
#
# Served-TUI QA found the app dying on the FIRST ingest submission: under
# Textual (app mode / textual-serve), `sys.stderr` is replaced by a capture
# object whose `fileno()` returns -1 WITHOUT raising. CPython 3.12's
# `multiprocessing.resource_tracker._launch` appends `sys.stderr.fileno()`
# to the fds it passes to `util.spawnv_passfds` (its `except Exception`
# guard never fires because -1 is returned, not raised), and
# `spawnv_passfds` rejects the list with `ValueError: bad value(s) in
# fds_to_keep` -- so the very first `get_context("spawn").Pool(...)` (which
# ensure-runs the process-global resource tracker) exploded, propagated up
# the top-up path on the UI thread, and crashed the app.


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a FRESH interpreter (mirrors the Task 2 import-weight
    helper). Fresh matters here: the multiprocessing resource tracker is
    process-global and starts exactly once, so only a brand-new process is
    guaranteed to exercise its launch path (an earlier in-process test that
    touched multiprocessing would have already started it, silently turning
    the repro into a no-op)."""
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(_REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )


class _TextualLikeStderr:
    """Mimics Textual's stderr capture object: ``fileno()`` returns -1
    WITHOUT raising (the exact shape that defeats the resource tracker's
    ``except Exception`` guard)."""

    def fileno(self) -> int:
        return -1

    def write(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


@pytest.mark.integration
def test_create_pool_survives_filenoless_stderr_real_spawn(tmp_path: Path) -> None:
    """Real repro, fresh interpreter: with ``sys.stderr`` swapped for a
    Textual-shaped capture object (fileno() == -1) BEFORE any
    multiprocessing use, the real ``_create_ingest_parse_pool`` must still
    construct a working spawn Pool (this is where the resource tracker
    launches) and round-trip a trivial ``apply_async``. RED pre-fix: the
    subprocess died with ``ValueError: bad value(s) in fds_to_keep``."""
    result = _run_isolated_python(
        tmp_path,
        """
        import sys


        class _TextualLikeStderr:
            def fileno(self):
                return -1

            def write(self, *args, **kwargs):
                return 0

            def flush(self):
                pass

            def isatty(self):
                return False


        if __name__ == "__main__":
            sys.stderr = _TextualLikeStderr()

            from tldw_chatbook.app import LibraryIngestQueueMixin

            mixin = LibraryIngestQueueMixin()
            # Instance shadow: one worker keeps the spawn cost bounded.
            mixin._ingest_parse_worker_count = lambda: 1

            pool = mixin._create_ingest_parse_pool()
            try:
                result = pool.apply_async(pow, (2, 3)).get(timeout=120)
                assert result == 8, result
            finally:
                pool.terminate()
                pool.join()
            print("POOL_OK")
        """,
    )
    assert result.returncode == 0, (
        f"pool creation under fileno-less stderr failed:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "POOL_OK" in result.stdout


def test_create_pool_redirects_to_real_stderr_when_fileno_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process anchor for the helper's swap behavior (the raw repro only
    triggers in a fresh interpreter -- see ``_run_isolated_python``'s
    docstring): with ``sys.stderr``'s fileno() invalid, the real mixin
    ``_create_ingest_parse_pool`` must have a valid, fd-backed stderr in
    effect AT POOL-CONSTRUCTION TIME (the resource tracker snapshots
    ``sys.stderr.fileno()`` during construction). Pool construction itself
    is faked (recording, not spawning) so this stays fast and
    deterministic."""
    import tldw_chatbook.app as app_module

    recorded: dict[str, int] = {}

    class _RecordingPool:
        def __init__(self, processes=None):
            try:
                recorded["fd_during_construction"] = sys.stderr.fileno()
            except Exception:
                recorded["fd_during_construction"] = -1

    class _RecordingContext:
        def Pool(self, processes=None):
            return _RecordingPool(processes)

    class _RecordingMultiprocessing:
        @staticmethod
        def get_context(method: str):
            assert method == "spawn"
            return _RecordingContext()

    monkeypatch.setattr(app_module, "multiprocessing", _RecordingMultiprocessing())
    monkeypatch.setattr(sys, "stderr", _TextualLikeStderr())

    mixin = LibraryIngestQueueMixin()
    mixin._ingest_parse_worker_count = lambda: 1  # instance shadow: skip config read
    pool = mixin._create_ingest_parse_pool()

    assert isinstance(pool, _RecordingPool)
    assert recorded["fd_during_construction"] >= 0


def test_create_pool_leaves_stderr_alone_when_fileno_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: with a valid ``sys.stderr`` fileno, no redirect happens --
    the stream object seen during construction is the ambient one."""
    import tldw_chatbook.app as app_module

    recorded: dict[str, Any] = {}

    class _RecordingPool:
        def __init__(self, processes=None):
            recorded["stderr_during_construction"] = sys.stderr

    class _RecordingContext:
        def Pool(self, processes=None):
            return _RecordingPool(processes)

    class _RecordingMultiprocessing:
        @staticmethod
        def get_context(method: str):
            assert method == "spawn"
            return _RecordingContext()

    monkeypatch.setattr(app_module, "multiprocessing", _RecordingMultiprocessing())

    ambient_stderr = sys.stderr
    assert ambient_stderr.fileno() >= 0  # pytest's capture stream is fd-backed

    mixin = LibraryIngestQueueMixin()
    mixin._ingest_parse_worker_count = lambda: 1
    mixin._create_ingest_parse_pool()

    assert recorded["stderr_during_construction"] is ambient_stderr


@pytest.mark.asyncio
async def test_pool_creation_failure_fails_job_retryable_and_app_survives(
    tmp_path: Path,
) -> None:
    """(Containment) Pool creation raising must never crash the app: the
    triggering job lands FAILED retryable with the pool message, no
    exception escapes ``submit_library_ingest_job``, ``_ingest_parse_pool``
    stays ``None``, and a subsequent submit retries pool creation (and
    succeeds once the pool can be built again)."""
    db = _make_db(tmp_path)
    source1 = _write_text_file(tmp_path, "note-1.txt", "First body.")
    source2 = _write_text_file(tmp_path, "note-2.txt", "Second body.")

    boom = {"raise": True}

    def _flaky_factory():
        if boom["raise"]:
            raise RuntimeError("spawn machinery exploded")
        return _FakeIngestParsePool()

    app = _IngestRunnerHarness(db, pool_factory=_flaky_factory)

    async with app.run_test() as pilot:
        # Must not raise, despite pool creation exploding underneath.
        job1 = app.submit_library_ingest_job(source_path=str(source1))

        failed = await _wait_for_job_state(
            app, pilot, job1.job_id, IngestJobState.FAILED
        )
        assert failed.permanent is False
        assert failed.error.startswith("Parse pool could not start:")
        assert "spawn machinery exploded" in failed.error
        assert app._ingest_parse_pool is None
        assert app._pool_create_count == 1

        # The failure is retryable through the normal seam, and the next
        # submission retries pool creation from scratch.
        boom["raise"] = False
        job2 = app.submit_library_ingest_job(source_path=str(source2))
        done2 = await _wait_for_job_state(app, pilot, job2.job_id, IngestJobState.DONE)
        assert done2.media_id is not None
        assert app._pool_create_count == 2

        await _wait_for_runner_idle(app, pilot)


def test_top_up_abandons_pass_when_mark_parsing_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(Whole-branch review, Minor 2) The defensive `mark_parsing`-returned-
    ``None`` branch must TERMINATE the top-up pass: ``next_queued()``
    always returns the OLDEST queued job, so a skip-and-``continue`` would
    be handed the exact same unclaimable job straight back -- an infinite
    loop on the UI thread. The branch is unreachable while UI-thread
    atomicity holds (see the surrounding docstring); forced here by
    stubbing ``mark_parsing``. The test completing at all (within the
    suite timeout) is the core assertion."""
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(db)
    app.library_ingest_jobs.submit(source_path="/tmp/whatever.txt")
    monkeypatch.setattr(
        app.library_ingest_jobs, "mark_parsing", lambda *args, **kwargs: None
    )

    app._top_up_ingest_parse_pool()  # must return, not loop forever

    # The pass was abandoned before ever reaching pool creation, and the
    # unclaimable job is left QUEUED (a later pass re-attempts it).
    assert app._pool_create_count == 0
    assert app.library_ingest_jobs.counts()["queued"] == 1


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
# F3 re-anchor: the writer's claim now targets *payload-ready* jobs (an
# entry in `_ingest_parsed_payloads`) instead of *queued* ones -- the tests
# below are re-anchored to that new claim source but keep the same
# atomicity-proof structure:
#   (i)/(ii)  direct, synchronous calls to `_claim_next_ingest_job_or_release`
#             proving its atomic claim-or-release contract in isolation;
#   (iii)     an end-to-end test that forces the writer to hit its
#             crash-recovery `finally` path (not a per-job failure -- a
#             failure in the claim step itself, outside all per-job
#             isolation) with a second job's payload still pending,
#             proving the writer notices and restarts itself instead of
#             stranding the payload;
#   (iv)      a coarse end-to-end stress smoke across five rapid
#             submissions (some landing while the pool/writer are
#             genuinely mid-flight) as a gross-stranding catch-all.


@pytest.mark.asyncio
async def test_claim_next_job_returns_payload_ready_job_and_keeps_runner_active(
    tmp_path: Path,
) -> None:
    """(i) Direct-call contract: a payload-ready job is returned (job +
    payload, job now WRITING, payload popped) and ``runner_active`` is left
    untouched (``True``), so the writer keeps looping instead of exiting.
    """
    db = _make_db(tmp_path)
    source = _write_text_file(
        tmp_path, "note-claim.txt", "Body for the claim contract test."
    )
    app = _IngestRunnerHarness(db)

    async with app.run_test():
        # Simulate the writer already being active (as it always is by the
        # time anything calls the claim method for real) without actually
        # starting the background worker thread -- this keeps the test
        # fully synchronous and deterministic. Driving the registry
        # directly (bypassing submit/the pool) avoids triggering
        # `_start_library_ingest_queue_if_idle` for the same reason.
        app.library_ingest_jobs.runner_active = True
        job = app.library_ingest_jobs.submit(source_path=str(source))
        app.library_ingest_jobs.mark_parsing(job.job_id)
        fake_payload = {"file_type": "plaintext", "content": "irrelevant"}
        app._ingest_parsed_payloads[job.job_id] = fake_payload

        claimed = app._claim_next_ingest_job_or_release()

        assert claimed is not None
        claimed_job, claimed_payload = claimed
        assert claimed_job.job_id == job.job_id
        assert claimed_job.state == IngestJobState.WRITING
        assert claimed_payload == fake_payload
        assert job.job_id not in app._ingest_parsed_payloads
        assert app.library_ingest_jobs.runner_active is True


@pytest.mark.asyncio
async def test_claim_next_job_returns_none_and_clears_runner_active_when_empty(
    tmp_path: Path,
) -> None:
    """(ii) Direct-call contract: with no payload-ready jobs, ``None`` is
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
async def test_shutdown_refuses_to_claim_ready_payload(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "shutdown-claim.txt", "Do not persist me.")
    app = _IngestRunnerHarness(db)

    async with app.run_test():
        app.library_ingest_jobs.runner_active = True
        job = app.library_ingest_jobs.submit(source_path=str(source))
        app.library_ingest_jobs.mark_parsing(job.job_id)
        app._ingest_parsed_payloads[job.job_id] = {
            "file_type": "plaintext",
            "content": "ready before shutdown",
        }
        app._ingest_shutdown = True

        claimed = app._claim_next_ingest_job_or_release()

        assert claimed is None
        assert app.library_ingest_jobs.runner_active is False
        assert job.job_id in app._ingest_parsed_payloads
        current = next(
            current
            for current in app.library_ingest_jobs.jobs()
            if current.job_id == job.job_id
        )
        assert current.state == IngestJobState.PARSING


@pytest.mark.asyncio
async def test_finally_restarts_writer_after_unexpected_claim_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(iii) Regression test for the crash-recovery path -- the strongest
    deterministic proxy for the exit race available without depending on
    real thread-scheduling timing.

    Simulates a genuinely unexpected ("catastrophic") failure in the
    writer's own claim step, as opposed to a per-job write failure (already
    isolated by the inner try/except): the *first ever* call to
    ``LibraryIngestJobRegistry.jobs()`` -- made from inside
    ``_claim_next_ingest_job_or_release``, the only call site that reaches
    it in this harness's writer path -- raises once; every subsequent call
    behaves normally. The patch is installed before either submission.
    Polling below deliberately uses ``real_jobs`` (the pre-patch bound
    method) rather than going back through ``app.library_ingest_jobs.jobs``
    -- otherwise the test's own polling would be equally likely to be the
    call that trips the one-shot raise, instead of the writer's claim.

    Pre-fix (L3b), the writer's ``finally`` unconditionally set
    ``runner_active = False`` and exited for good -- it never rechecked for
    pending work and restarted itself. Post-fix,
    ``_release_ingest_runner_after_crash`` notices a payload is still
    waiting (the exception fires before the claim ever pops anything out of
    ``_ingest_parsed_payloads``) and restarts the writer, so both jobs
    still reach ``DONE``.
    """
    db = _make_db(tmp_path)
    source1 = _write_text_file(
        tmp_path, "note-crash-1.txt", "First body, present when the crash hits."
    )
    source2 = _write_text_file(
        tmp_path, "note-crash-2.txt", "Second body, queued behind the crash."
    )
    app = _IngestRunnerHarness(db, worker_count=2)

    real_jobs = app.library_ingest_jobs.jobs
    call_state = {"raised": False}

    def _flaky_jobs():
        if not call_state["raised"]:
            call_state["raised"] = True
            raise RuntimeError("simulated catastrophic writer failure")
        return real_jobs()

    async def _wait_for_done_via_real_jobs(job_id: str) -> LibraryIngestJob:
        for _ in range(_POLL_ATTEMPTS):
            job = next((j for j in real_jobs() if j.job_id == job_id), None)
            if job is not None and job.state == IngestJobState.DONE:
                return job
            await pilot.pause(_POLL_INTERVAL)
        raise AssertionError(f"job {job_id} never reached DONE: {real_jobs()}")

    async with app.run_test() as pilot:
        monkeypatch.setattr(app.library_ingest_jobs, "jobs", _flaky_jobs)

        job1 = app.submit_library_ingest_job(source_path=str(source1))
        job2 = app.submit_library_ingest_job(source_path=str(source2))

        done1 = await _wait_for_done_via_real_jobs(job1.job_id)
        assert done1.media_id is not None
        done2 = await _wait_for_done_via_real_jobs(job2.job_id)
        assert done2.media_id is not None

        assert call_state["raised"] is True, (
            "the simulated crash never fired -- test is not exercising the recovery path"
        )

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_five_rapid_submissions_all_complete_no_stranding(tmp_path: Path) -> None:
    """(iv) End-to-end stress smoke: five jobs submitted in rapid
    succession -- some landing while the pool/writer are genuinely
    mid-flight -- must ALL reach DONE.

    This is a coarse, gross-stranding catch-all: if any submission landed
    in the (pre-fix) exit-race gap and got stuck behind a stale
    ``runner_active``, at least one job here would never leave QUEUED/
    PARSING and the wait loop below would time out.
    """
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(db, worker_count=2)

    async with app.run_test() as pilot:
        sources = [
            _write_text_file(
                tmp_path, f"note-stress-{i}.txt", f"Stress body number {i}."
            )
            for i in range(5)
        ]

        jobs = [
            app.submit_library_ingest_job(source_path=str(sources[0])),
            app.submit_library_ingest_job(source_path=str(sources[1])),
        ]
        # Give the pool/writer a chance to actually start draining before
        # the remaining submissions land -- the scenario most likely to hit
        # the exit-race gap is a submission arriving while the writer is
        # mid-loop, possibly right as it decides whether to exit.
        await pilot.pause(_POLL_INTERVAL)
        jobs.append(app.submit_library_ingest_job(source_path=str(sources[2])))
        await pilot.pause(_POLL_INTERVAL)
        jobs.append(app.submit_library_ingest_job(source_path=str(sources[3])))
        jobs.append(app.submit_library_ingest_job(source_path=str(sources[4])))

        for job in jobs:
            done = await _wait_for_job_state(
                app, pilot, job.job_id, IngestJobState.DONE
            )
            assert done.media_id is not None

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_reingest_of_unchanged_file_still_resolves_media_id(
    tmp_path: Path,
) -> None:
    """Re-ingesting an already-present, unchanged file keeps Open usable.

    ``add_media_with_keywords`` takes its update path for a URL that already
    exists with identical content and returns ``media_id=None``; the writer
    must resolve the id via ``get_media_by_url`` (using the parsed payload's
    own ``url`` field) so the done job still carries a real ``media_id``
    (and the canvas keeps its Open action).
    """
    db = _make_db(tmp_path)
    source = _write_text_file(
        tmp_path, "note-b.txt", "Spring tides align sun and moon."
    )
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        first = app.submit_library_ingest_job(source_path=str(source))
        first_done = await _wait_for_job_state(
            app, pilot, first.job_id, IngestJobState.DONE
        )
        assert first_done.media_id is not None
        await _wait_for_runner_idle(app, pilot)

        second = app.submit_library_ingest_job(source_path=str(source))
        second_done = await _wait_for_job_state(
            app, pilot, second.job_id, IngestJobState.DONE
        )

        assert second_done.media_id == first_done.media_id
        await _wait_for_runner_idle(app, pilot)


# --- remote poller (task-684.2) ---------------------------------------------


class _FakeServerMediaService:
    """Records batch lookups and replays scripted status responses."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = responses
        self.batch_calls: list[str] = []

    async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
        self.batch_calls.append(batch_id)
        if self._responses:
            return self._responses.pop(0)
        return {"batch_id": batch_id, "jobs": []}


def _queued_server_job(app, *, remote_job_id: str, batch_id: str = "batch-1"):
    job = app.library_ingest_jobs.submit(source_path="/tmp/a.mp3", origin="server")
    return app.library_ingest_jobs.attach_remote(
        job.job_id, remote_job_id=remote_job_id, batch_id=batch_id
    )


@pytest.mark.asyncio
async def test_remote_poll_settles_a_server_job_then_stops(tmp_path: Path) -> None:
    """The poller applies a terminal status and then stops polling.

    A finished batch that kept being fetched would hit the server forever for an
    answer that cannot change.
    """
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    service = _FakeServerMediaService(
        [{"batch_id": "batch-1", "jobs": [{"id": 11, "status": "completed"}]}]
    )
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        app.poll_remote_ingest_jobs()

        for _ in range(_POLL_ATTEMPTS):
            if app.library_ingest_jobs.jobs()[0].state == IngestJobState.DONE:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError("the server job never settled")

        calls_after_settle = len(service.batch_calls)
        for _ in range(5):
            await pilot.pause(_POLL_INTERVAL)
        assert len(service.batch_calls) == calls_after_settle, (
            "poller kept fetching a settled batch"
        )


@pytest.mark.asyncio
async def test_remote_poll_does_not_start_without_outstanding_batches(
    tmp_path: Path,
) -> None:
    """A library with only local jobs must never touch the server."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    service = _FakeServerMediaService([])
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        app.library_ingest_jobs.submit(source_path="/tmp/local.txt")
        app.poll_remote_ingest_jobs()
        await pilot.pause(_POLL_INTERVAL)

    assert service.batch_calls == []


@pytest.mark.asyncio
async def test_remote_poll_survives_a_server_error(tmp_path: Path) -> None:
    """A transient failure must not kill the poller or the job."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01

    class _FlakyService:
        def __init__(self) -> None:
            self.calls = 0

        async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("connection reset")
            return {"batch_id": batch_id, "jobs": [{"id": 11, "status": "completed"}]}

    service = _FlakyService()
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        app.poll_remote_ingest_jobs()

        for _ in range(_POLL_ATTEMPTS):
            if app.library_ingest_jobs.jobs()[0].state == IngestJobState.DONE:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError("the poller did not recover from the error")

    assert service.calls >= 2


@pytest.mark.asyncio
async def test_remote_poll_stops_on_shutdown(tmp_path: Path) -> None:
    """The quit flag ends the loop rather than leaving it fetching."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    service = _FakeServerMediaService([])
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        app.poll_remote_ingest_jobs()
        await pilot.pause(_POLL_INTERVAL)

        app._ingest_shutdown = True
        await pilot.pause(_POLL_INTERVAL)
        calls_at_shutdown = len(service.batch_calls)
        for _ in range(5):
            await pilot.pause(_POLL_INTERVAL)

    assert len(service.batch_calls) == calls_at_shutdown, "poller ignored shutdown"


@pytest.mark.asyncio
async def test_remote_poll_without_a_server_service_is_a_noop(tmp_path: Path) -> None:
    """No configured server backend means nothing to poll, not a crash."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = None

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        app.poll_remote_ingest_jobs()
        await pilot.pause(_POLL_INTERVAL)

        assert app.library_ingest_jobs.jobs()[0].state == IngestJobState.QUEUED


@pytest.mark.asyncio
async def test_cancel_remote_batch_asks_the_server_and_resumes_polling(
    tmp_path: Path,
) -> None:
    """Cancelling asks the server; the local job is NOT pre-emptively marked.

    The request is asynchronous and may be refused, so the queue must not claim
    an outcome the server has not confirmed. The poller records the real state.
    """
    cancelled_batches: list[str] = []

    class _CancellableService:
        async def cancel_media_ingest_jobs_batch(
            self, *, batch_id: str | None = None, session_id: str | None = None,
            reason: str | None = None,
        ):
            # Keyword-only, mirroring the real client/service signature --
            # a fake with a positional parameter would happily accept a
            # positional call that fails against the real one.
            cancelled_batches.append(batch_id)
            return {"batch_id": batch_id, "cancelled": 1}

        async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
            return {
                "batch_id": batch_id,
                "jobs": [{"id": 11, "status": "cancelled",
                          "cancellation_reason": "user asked"}],
            }

    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    app.server_media_reading_service = _CancellableService()

    async with app.run_test() as pilot:
        job = _queued_server_job(app, remote_job_id="11")
        app.cancel_remote_ingest_batch("batch-1")

        # The request alone must not move the local job.
        await pilot.pause(_POLL_INTERVAL)
        assert cancelled_batches == ["batch-1"]

        for _ in range(_POLL_ATTEMPTS):
            if app.library_ingest_jobs.jobs()[0].state == IngestJobState.CANCELLED:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError("the poller never recorded the cancellation")

        assert "user asked" in app.library_ingest_jobs.jobs()[0].error


@pytest.mark.asyncio
async def test_cancel_remote_batch_without_a_server_seam_is_quiet(
    tmp_path: Path,
) -> None:
    """No server backend means no cancel request, and no crash."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = None

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        app.cancel_remote_ingest_batch("batch-1")
        await pilot.pause(_POLL_INTERVAL)

        assert app.library_ingest_jobs.jobs()[0].state == IngestJobState.QUEUED


@pytest.mark.asyncio
async def test_cancel_remote_batch_ignores_an_empty_batch_id(tmp_path: Path) -> None:
    calls: list[str] = []

    class _Service:
        async def cancel_media_ingest_jobs_batch(
            self, *, batch_id: str | None = None, session_id: str | None = None,
            reason: str | None = None,
        ):
            calls.append(batch_id)

        async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
            return {"batch_id": batch_id, "jobs": []}

    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = _Service()

    async with app.run_test() as pilot:
        app.cancel_remote_ingest_batch("")
        await pilot.pause(_POLL_INTERVAL)

    assert calls == []


# --- backend routing (task-684.1 slice 2) -----------------------------------


class _RecordingServerService:
    """Captures submissions and returns a scripted batch/job id pair."""

    def __init__(self, *, fail: bool = False) -> None:
        self.submissions: list[dict] = []
        self.fail = fail

    async def submit_ingest_jobs(self, **kwargs):
        self.submissions.append(kwargs)
        if self.fail:
            raise RuntimeError("server said no")
        return {"batch_id": "batch-7", "jobs": [{"id": 42, "source": "/tmp/a.mp3"}]}

    async def submit_media_ingest_jobs(self, **kwargs):
        return await self.submit_ingest_jobs(**kwargs)

    async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
        return {"batch_id": batch_id, "jobs": []}


def _server_ingest_preference():
    """Patch the ingest backend preference to "server" for a test's duration.

    Sending an ingest to a server is an explicit opt-in stored under
    ``[library.ingest] backend``; the browse scope deliberately does not decide
    it. The opt-in alone is not enough, though -- runtime policy requires the
    runtime to be in server mode as well -- so tests that want a server route
    must ALSO call ``_use_server_runtime`` (see ``_resolve_ingest_backend``).
    """
    real = _app_module.get_cli_setting

    def _fake(*args, **kwargs):
        if args[:2] == ("library.ingest", "backend"):
            return "server"
        return real(*args, **kwargs)

    return patch("tldw_chatbook.app.get_cli_setting", side_effect=_fake)


def _use_server_runtime(app) -> None:
    """Put the harness's runtime policy in server mode.

    Required alongside the opt-in: ``media.ingestion_jobs.launch.server`` is
    declared ``required_source="server"``, so the service refuses the launch in
    local mode -- verified live, where it failed with "requires server mode".
    """
    from types import SimpleNamespace

    app.runtime_policy = SimpleNamespace(
        state=SimpleNamespace(active_source="server")
    )


@pytest.mark.asyncio
async def test_local_backend_still_ingests_locally(tmp_path: Path) -> None:
    """The default path must be untouched by server routing."""
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note.txt", "Local body.")
    app = _IngestRunnerHarness(db)
    service = _RecordingServerService()
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        assert job.origin == "local"
        done = await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
        assert done.media_id is not None
        await _wait_for_runner_idle(app, pilot)

    assert service.submissions == [], "local ingest must not contact the server"


@pytest.mark.asyncio
async def test_server_backend_submits_remotely_and_attaches_ids(
    tmp_path: Path,
) -> None:
    """A server-backend submit goes to the server and records its ids."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    service = _RecordingServerService()
    app.server_media_reading_service = service
    source = _write_text_file(tmp_path, "note.txt", "Body.")

    _use_server_runtime(app)
    with _server_ingest_preference():
      async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source), title="A title")
        assert job.origin == "server"

        for _ in range(_POLL_ATTEMPTS):
            current = app.library_ingest_jobs.get_job(job.job_id)
            if current is not None and current.batch_id:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError("the remote ids were never attached")

    assert len(service.submissions) == 1
    submitted = service.submissions[0]
    # "document" not "plaintext": the live server accepts only video/audio/
    # document/pdf/ebook, enforced by a runtime validator not its OpenAPI spec.
    assert submitted["media_type"] == "document"
    assert submitted["file_paths"] == [str(source)]
    assert submitted["title"] == "A title"

    current = app.library_ingest_jobs.get_job(job.job_id)
    assert current.batch_id == "batch-7"
    assert current.remote_job_id == "42"


@pytest.mark.asyncio
async def test_server_backend_without_a_service_fails_the_job_clearly(
    tmp_path: Path,
) -> None:
    """Choosing a server with none configured must explain itself, not hang."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = None
    source = _write_text_file(tmp_path, "note.txt", "Body.")

    _use_server_runtime(app)
    with _server_ingest_preference():
      async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        failed = await _wait_for_job_state(
            app, pilot, job.job_id, IngestJobState.FAILED
        )

    assert "server" in failed.error.lower()


@pytest.mark.asyncio
async def test_server_backend_refuses_a_source_it_cannot_send(tmp_path: Path) -> None:
    """A plain web page belongs to the clipper, and says so rather than failing late."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = _RecordingServerService()

    _use_server_runtime(app)
    with _server_ingest_preference():
      async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path="https://example.com/a-post")
        failed = await _wait_for_job_state(
            app, pilot, job.job_id, IngestJobState.FAILED
        )

    assert "clip" in failed.error.lower()


@pytest.mark.asyncio
async def test_server_submit_failure_marks_the_job_failed(tmp_path: Path) -> None:
    """A refused submission must surface, not sit queued forever."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = _RecordingServerService(fail=True)
    source = _write_text_file(tmp_path, "note.txt", "Body.")

    _use_server_runtime(app)
    with _server_ingest_preference():
      async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        failed = await _wait_for_job_state(
            app, pilot, job.job_id, IngestJobState.FAILED
        )

    assert failed.error


@pytest.mark.asyncio
async def test_an_unrecognised_backend_falls_back_to_local(tmp_path: Path) -> None:
    """Anything that is not exactly "server" must mean local.

    Local is the backend that always works, so a typo'd or newly-added value
    must not silently start shipping the user's files to a server.

    Uses a raw stand-in rather than ``MediaRuntimeState``: that dataclass
    normalises in ``__post_init__``, so a test going through it would exercise
    the dataclass's guarantee instead of this fallback, and would pass even if
    the fallback were inverted.
    """
    from types import SimpleNamespace

    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note.txt", "Body.")
    app = _IngestRunnerHarness(db)
    service = _RecordingServerService()
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        for value in ("", "  ", "remote", "Server-ish", "cloud", None):
            with patch(
                "tldw_chatbook.app.get_cli_setting",
                side_effect=lambda *a, v=value, **k: (
                    v if a[:2] == ("library.ingest", "backend") else None
                ),
            ):
                assert app._resolve_ingest_backend() == "local", repr(value)

        # And end-to-end: with no preference set, ingest stays local.
        job = app.submit_library_ingest_job(source_path=str(source))
        assert job.origin == "local"
        await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
        await _wait_for_runner_idle(app, pilot)

    assert service.submissions == []


@pytest.mark.asyncio
async def test_server_backend_is_matched_case_insensitively(tmp_path: Path) -> None:
    """A raw "Server"/" SERVER " value should still route remotely."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    _use_server_runtime(app)
    async with app.run_test():
        for value in ("server", "Server", " SERVER "):
            with patch(
                "tldw_chatbook.app.get_cli_setting",
                side_effect=lambda *a, v=value, **k: (
                    v if a[:2] == ("library.ingest", "backend") else None
                ),
            ):
                assert app._resolve_ingest_backend() == "server", repr(value)


@pytest.mark.asyncio
async def test_media_runtime_state_already_normalises_the_backend() -> None:
    """Defence in depth: the shared dataclass narrows the value first.

    Documented as its own test so the two layers are not confused -- this one
    is about ``MediaRuntimeState``'s guarantee, not the ingest fallback above.
    """
    from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState

    assert MediaRuntimeState(runtime_backend="remote").runtime_backend == "local"
    assert MediaRuntimeState(runtime_backend="SERVER").runtime_backend == "server"


@pytest.mark.asyncio
async def test_browse_scope_alone_never_sends_files_to_a_server(
    tmp_path: Path,
) -> None:
    """Browsing in server scope must not silently upload a local file.

    ``build_library_ingest_state``'s own contract says ingest "always targets
    the local media store regardless of browsing scope". Letting the browse
    scope decide would mean a user who switched scope to look at server media
    then imported a file would have it leave their machine without ever asking
    for that -- so the ingest target is its own explicit preference, defaulting
    to local.
    """
    from types import SimpleNamespace

    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note.txt", "Body.")
    app = _IngestRunnerHarness(db)
    service = _RecordingServerService()
    app.server_media_reading_service = service
    # Browsing server-side, with no ingest preference expressed.
    app.media_runtime_state = SimpleNamespace(runtime_backend="server")

    async with app.run_test() as pilot:
        assert app._resolve_ingest_backend() == "local"
        job = app.submit_library_ingest_job(source_path=str(source))
        assert job.origin == "local"
        await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
        await _wait_for_runner_idle(app, pilot)

    assert service.submissions == [], "browse scope must not route ingest remotely"


@pytest.mark.asyncio
async def test_an_explicit_server_preference_routes_remotely(tmp_path: Path) -> None:
    """Opting in -- and only opting in -- sends the ingest to the server."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    service = _RecordingServerService()
    app.server_media_reading_service = service
    _use_server_runtime(app)
    source = _write_text_file(tmp_path, "note.txt", "Body.")

    with patch(
        "tldw_chatbook.app.get_cli_setting",
        side_effect=lambda *a, **k: (
            "server" if a and a[0] == "library.ingest" and a[1:2] == ("backend",) else None
        ),
    ):
        async with app.run_test() as pilot:
            assert app._resolve_ingest_backend() == "server"
            job = app.submit_library_ingest_job(source_path=str(source))
            assert job.origin == "server"
            await pilot.pause(_POLL_INTERVAL)

    assert len(service.submissions) == 1


@pytest.mark.asyncio
async def test_remote_poll_follows_pagination(tmp_path: Path) -> None:
    """A batch larger than one page must be reconciled in full.

    The live server's MediaIngestJobListResponse carries has_more/next_offset
    (confirmed from its OpenAPI spec). Reading only the first page would leave
    later jobs unreconciled -- and because they stay unsettled, the poller would
    keep fetching that batch forever, which is exactly what the stop condition
    exists to prevent.
    """
    class _PagedService:
        def __init__(self) -> None:
            self.offsets: list[int] = []

        async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100,
                                         offset: int = 0):
            self.offsets.append(offset)
            if offset == 0:
                return {
                    "batch_id": batch_id,
                    "jobs": [{"id": 11, "status": "completed"}],
                    "has_more": True,
                    "next_offset": 1,
                }
            return {
                "batch_id": batch_id,
                "jobs": [{"id": 12, "status": "completed"}],
                "has_more": False,
            }

    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    service = _PagedService()
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        _queued_server_job(app, remote_job_id="12")
        app.poll_remote_ingest_jobs()

        for _ in range(_POLL_ATTEMPTS):
            states = {j.state for j in app.library_ingest_jobs.jobs()}
            if states == {IngestJobState.DONE}:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"second page never reconciled; states={states}, offsets={service.offsets}"
            )

    assert 1 in service.offsets, "the poller never asked for the second page"


def test_the_real_service_accepts_the_offset_the_poller_sends() -> None:
    """Pin the poller's paging against the REAL service, not a fake.

    ``test_remote_poll_follows_pagination`` passed while pagination was dead in
    production: its fake declared ``offset`` because I wrote it to match my call
    site, but ``ServerMediaReadingService.list_media_ingest_jobs`` took only
    ``limit``. Every real call raised ``TypeError`` and fell into a one-page
    fallback, so the poller never read past page one.

    A fake can agree with a wrong assumption; the real signature cannot. Same
    failure mode as the keyword-only ``cancel`` call whose fake took it
    positionally (task-684.2).
    """
    from tldw_chatbook.app import _accepts_keyword
    from tldw_chatbook.Media.server_media_reading_service import (
        ServerMediaReadingService,
    )

    for method_name in ("list_media_ingest_jobs", "list_ingest_jobs"):
        method = getattr(ServerMediaReadingService, method_name)
        assert _accepts_keyword(method, "offset"), (
            f"{method_name} takes no offset, so the poller cannot page"
        )


def test_accepts_keyword_does_not_mistake_an_internal_typeerror_for_absence() -> None:
    """The signature check must not be fooled the way ``except TypeError`` was.

    Catching ``TypeError`` around the call conflated "no such parameter" with
    "the callable raised TypeError internally", silently degrading a real bug
    into a missing feature.
    """
    from tldw_chatbook.app import _accepts_keyword

    async def raises_internally(batch_id: str, *, offset: int = 0):
        raise TypeError("something inside blew up")

    assert _accepts_keyword(raises_internally, "offset") is True

    async def no_offset(batch_id: str, *, limit: int = 100):
        return {}

    assert _accepts_keyword(no_offset, "offset") is False

    async def takes_kwargs(batch_id: str, **kwargs):
        return {}

    assert _accepts_keyword(takes_kwargs, "offset") is True


@pytest.mark.asyncio
async def test_a_service_without_offset_reads_one_page_instead_of_looping(
    tmp_path: Path,
) -> None:
    """An offset-less service must stop, not re-read page one to the cap.

    ``has_more`` stays true on a first page, so continuing the loop without a
    way to advance would re-fetch the same page ``REMOTE_INGEST_MAX_PAGES``
    times per pass, every pass.
    """
    class _UnpagedService:
        def __init__(self) -> None:
            self.calls = 0

        async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
            self.calls += 1
            return {
                "batch_id": batch_id,
                "jobs": [{"id": 11, "status": "completed"}],
                "has_more": True,
                "next_offset": 1,
            }

    app = _IngestRunnerHarness(_make_db(tmp_path))
    service = _UnpagedService()
    app.server_media_reading_service = service

    async with app.run_test():
        _queued_server_job(app, remote_job_id="11")
        await app._reconcile_remote_batch(service, "batch-1")

    assert service.calls == 1, (
        f"re-read page one {service.calls} times with no way to advance"
    )
    assert {j.state for j in app.library_ingest_jobs.jobs()} == {IngestJobState.DONE}


@pytest.mark.asyncio
async def test_opting_in_without_server_mode_still_ingests_locally(
    tmp_path: Path,
) -> None:
    """The opt-in alone must not route remotely; policy requires server mode.

    Runtime policy declares ``media.ingestion_jobs.launch.server`` as
    ``required_source="server"``, so the service refuses the launch in local
    mode. Verified live: it failed with "media.ingestion_jobs.launch.server
    requires server mode". Falling back to a local ingest keeps the file where
    it is and lets the canvas explain the precondition, rather than handing the
    user a failed job.
    """
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note.txt", "Body.")
    app = _IngestRunnerHarness(db)
    service = _RecordingServerService()
    app.server_media_reading_service = service
    # Opted in, but the runtime is still local.

    with _server_ingest_preference():
        async with app.run_test() as pilot:
            assert app._resolve_ingest_backend() == "local"
            job = app.submit_library_ingest_job(source_path=str(source))
            assert job.origin == "local"
            await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
            await _wait_for_runner_idle(app, pilot)

    assert service.submissions == []
