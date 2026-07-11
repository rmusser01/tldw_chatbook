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

import pytest
from textual.app import App

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
    ) -> None:
        super().__init__()
        self.library_ingest_jobs = LibraryIngestJobRegistry()
        self.media_db = media_db
        self._ingest_parse_pool = None
        self._ingest_parsed_payloads: dict[str, dict] = {}
        self._ingest_shutdown = False
        self._pool_factory = pool_factory or (lambda: _FakeIngestParsePool())
        self._pool_create_count = 0
        self._worker_count_override = worker_count

    def _create_ingest_parse_pool(self):
        self._pool_create_count += 1
        return self._pool_factory()

    def _ingest_parse_worker_count(self) -> int:
        if self._worker_count_override is not None:
            return self._worker_count_override
        return super()._ingest_parse_worker_count()


def _make_db(tmp_path: Path, name: str = "library_ingest.db") -> MediaDatabase:
    return MediaDatabase(tmp_path / name, client_id="f3-runner-test")


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
async def test_two_files_queued_both_reach_done_with_real_media_rows(tmp_path: Path) -> None:
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

        failed = await _wait_for_job_state(app, pilot, failing_job.job_id, IngestJobState.FAILED)
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
        failed = await _wait_for_job_state(app, pilot, failing_job.job_id, IngestJobState.FAILED)
        assert failed.permanent is False
        assert failed.error == "transient parse hiccup"
        await _wait_for_runner_idle(app, pilot)

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
async def test_missing_file_failure_is_permanent_and_refuses_retry(tmp_path: Path) -> None:
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
        failed = await _wait_for_job_state(app, pilot, failing_job.job_id, IngestJobState.FAILED)
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
        failed = await _wait_for_job_state(app, pilot, failing_job.job_id, IngestJobState.FAILED)
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
        assert len(pools) == 2, "a fresh pool must be created lazily on the next submission"


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
async def test_shutdown_flag_stops_late_parse_completion_callbacks(tmp_path: Path) -> None:
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

        current = next(j for j in app.library_ingest_jobs.jobs() if j.job_id == job.job_id)
        assert current.state == IngestJobState.PARSING

        app._ingest_shutdown = True

        # Late success completion: must be a no-op.
        app._on_ingest_parse_complete(
            job.job_id, {"ok": True, "payload": {"file_type": "plaintext"}}
        )
        current = next(j for j in app.library_ingest_jobs.jobs() if j.job_id == job.job_id)
        assert current.state == IngestJobState.PARSING
        assert job.job_id not in app._ingest_parsed_payloads
        assert app.library_ingest_jobs.runner_active is False

        # Late pool-level error: also a no-op -- the pool is not dropped.
        app._handle_broken_ingest_parse_pool(RuntimeError("late failure"))
        current = next(j for j in app.library_ingest_jobs.jobs() if j.job_id == job.job_id)
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
    app._ingest_pool_callback("ingest-job-1", {"ok": True, "payload": {}})
    app._ingest_pool_error_callback(RuntimeError("late pool failure"))
    assert marshaled == []

    # Positive control: with the flag down, both callbacks marshal.
    app._ingest_shutdown = False
    app._ingest_pool_callback("ingest-job-1", {"ok": True, "payload": {}})
    app._ingest_pool_error_callback(RuntimeError("pool failure"))
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

        failed_b = await _wait_for_job_state(app, pilot, job_b.job_id, IngestJobState.FAILED)
        assert failed_b.permanent is False

        # A must survive: never failed, and the handler's writer wake
        # drains its already-finished parse to DONE with a real media row.
        done_a = await _wait_for_job_state(app, pilot, job_a.job_id, IngestJobState.DONE)
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

        failed = await _wait_for_job_state(app, pilot, job1.job_id, IngestJobState.FAILED)
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
    source = _write_text_file(tmp_path, "note-claim.txt", "Body for the claim contract test.")
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
    source1 = _write_text_file(tmp_path, "note-crash-1.txt", "First body, present when the crash hits.")
    source2 = _write_text_file(tmp_path, "note-crash-2.txt", "Second body, queued behind the crash.")
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

        assert call_state["raised"] is True, "the simulated crash never fired -- test is not exercising the recovery path"

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
            _write_text_file(tmp_path, f"note-stress-{i}.txt", f"Stress body number {i}.")
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
            done = await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
            assert done.media_id is not None

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_reingest_of_unchanged_file_still_resolves_media_id(tmp_path: Path) -> None:
    """Re-ingesting an already-present, unchanged file keeps Open usable.

    ``add_media_with_keywords`` takes its update path for a URL that already
    exists with identical content and returns ``media_id=None``; the writer
    must resolve the id via ``get_media_by_url`` (using the parsed payload's
    own ``url`` field) so the done job still carries a real ``media_id``
    (and the canvas keeps its Open action).
    """
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note-b.txt", "Spring tides align sun and moon.")
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        first = app.submit_library_ingest_job(source_path=str(source))
        first_done = await _wait_for_job_state(app, pilot, first.job_id, IngestJobState.DONE)
        assert first_done.media_id is not None
        await _wait_for_runner_idle(app, pilot)

        second = app.submit_library_ingest_job(source_path=str(source))
        second_done = await _wait_for_job_state(app, pilot, second.job_id, IngestJobState.DONE)

        assert second_done.media_id == first_done.media_id
        await _wait_for_runner_idle(app, pilot)
