"""Ingestion pause preserves pending bytes and admitted publication."""

import asyncio
import inspect
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import tldw_chatbook.app as app_module
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.app import LibraryIngestQueueMixin
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


class IngestHost(LibraryIngestQueueMixin):
    def __init__(self):
        self._init_library_ingest_runtime_state()
        self.media_db = None


@pytest.mark.asyncio
async def test_ingest_pause_keeps_payloads_and_queued_jobs_until_resume():
    app = IngestHost()
    queued = app.library_ingest_jobs.submit(source_path="queued.txt")
    pending = app.library_ingest_jobs.submit(source_path="parsed.txt")
    app.library_ingest_jobs.mark_parsing(pending.job_id)
    payload = {"content": "must survive maintenance"}
    app._ingest_parsed_payloads[pending.job_id] = payload
    app._ingest_maintenance_close_admission()
    with pytest.raises(RuntimeError, match="ingest_maintenance_paused"):
        app.submit_library_ingest_job(source_path="new.txt")
    app._top_up_ingest_parse_pool()
    assert app._ingest_parse_pool is None
    assert not await app._ingest_maintenance_drain(time.monotonic())
    assert app._ingest_parsed_payloads[pending.job_id] is payload
    assert app.library_ingest_jobs.get_job(queued.job_id).state.value == "queued"
    calls = []
    app._top_up_ingest_parse_pool = lambda: calls.append("top_up")
    app._start_library_ingest_queue_if_idle = lambda: calls.append("writer")
    app._ingest_maintenance_resume()
    assert calls == ["writer", "top_up"]
    assert not app._ingest_shutdown


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["parse", "stt", "writer"])
async def test_ingest_drain_waits_for_admitted_work_without_cancelling(kind):
    app = IngestHost()
    if kind == "parse":
        app._ingest_parse_jobs_by_generation[1] = {"pending"}
        finish = lambda: app._ingest_parse_jobs_by_generation[1].clear()
    elif kind == "stt":
        app._ingest_local_stt_jobs["pending"] = (1, "attempt")
        finish = app._ingest_local_stt_jobs.clear
    else:
        app.library_ingest_jobs.runner_active = True
        finish = lambda: setattr(app.library_ingest_jobs, "runner_active", False)
    app._ingest_maintenance_close_admission()
    waiter = asyncio.create_task(app._ingest_maintenance_drain(time.monotonic() + 2))
    await asyncio.sleep(0)
    assert not waiter.done()
    assert not app._ingest_shutdown
    finish()
    assert await waiter


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_finished_writer_retires_its_native_connection(
    tmp_path, local_root, fail
):
    app = IngestHost()
    app.media_db = MediaDatabase(tmp_path / "media.db", client_id="backup-test")
    app.call_from_thread = lambda callback, *args, **kwargs: callback(*args, **kwargs)

    def claim():
        app.media_db.get_connection().execute("SELECT 1").fetchone()
        if fail:
            raise ValueError("claim failed")

    app._claim_next_ingest_job_or_release = claim
    writer = inspect.unwrap(LibraryIngestQueueMixin._run_library_ingest_queue)
    try:
        if fail:
            with pytest.raises(ValueError, match="claim failed"):
                await asyncio.to_thread(writer, app)
        else:
            await asyncio.to_thread(writer, app)
        assert not worker_leases(app.media_db)
        assert app.media_db.get_connection().execute("SELECT 1").fetchone()[0] == 1
    finally:
        app.media_db.close_connection()


@pytest.mark.asyncio
async def test_drain_retains_writer_while_native_close_is_still_running(
    tmp_path, local_root
):
    app = IngestHost()
    db = MediaDatabase(tmp_path / "closing.db", client_id="backup-test")
    app.media_db = db
    app.call_from_thread = lambda callback, *args, **kwargs: callback(*args, **kwargs)
    app._claim_next_ingest_job_or_release = lambda: None
    entered, release = threading.Event(), threading.Event()
    close = db.close_connection

    def slow_close():
        entered.set()
        assert release.wait(3)
        close()

    db.close_connection = slow_close
    writer = inspect.unwrap(LibraryIngestQueueMixin._run_library_ingest_queue)
    running = asyncio.create_task(asyncio.to_thread(writer, app))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        app._ingest_maintenance_close_admission()
        assert not await app._ingest_maintenance_drain(time.monotonic())
        release.set()
        await running
        assert await app._ingest_maintenance_drain(time.monotonic() + 1)
    finally:
        release.set()
        await running
        close()


@pytest.mark.asyncio
async def test_remote_poll_settles_current_batch_without_an_extra_sleep(monkeypatch):
    app = IngestHost()
    app.server_media_reading_service = object()
    monkeypatch.setattr(
        app_module, "pending_remote_batches", lambda registry: ("batch",)
    )
    entered, release = asyncio.Event(), asyncio.Event()
    finished = []

    async def reconcile(service, batch):
        entered.set()
        await release.wait()
        finished.append(batch)

    app._reconcile_remote_batch = reconcile
    run = inspect.unwrap(LibraryIngestQueueMixin._run_remote_ingest_poll)
    runner = asyncio.create_task(run(app))
    try:
        await entered.wait()
        app._ingest_maintenance_close_admission()
        assert not runner.done()
        release.set()
        await asyncio.wait_for(runner, 1)
        assert finished == ["batch"]
    finally:
        release.set()
        await asyncio.gather(runner, return_exceptions=True)


_PILOT = r"""
import asyncio, time
from pathlib import Path
from Tests.network_guard import install
install()
from Tests.Library.test_library_ingest_runner import _IngestRunnerHarness, _FakeIngestParsePool
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB

async def main():
    root = Path.home()
    db = MediaDatabase(root / "media.db", client_id="backup-pilot")
    store = LibraryIngestJobsDB(root / "jobs.db")
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool, worker_count=1)
    app.library_ingest_jobs.attach_store(store)
    first = root / "first.txt"
    second = root / "second.txt"
    first.write_text("Committed before capture.")
    second.write_text("Queued across capture.")
    try:
        async with app.run_test() as pilot:
            a = app.submit_library_ingest_job(source_path=str(first), ingest_options={"generic":{"generate_embeddings":False}})
            b = app.submit_library_ingest_job(source_path=str(second), ingest_options={"generic":{"generate_embeddings":False}})
            assert len(pool.calls) == 1
            app._ingest_maintenance_close_admission()
            assert not await app._ingest_maintenance_drain(time.monotonic())
            pool._spawn(pool._run_one, pool.calls[0])
            assert await app._ingest_maintenance_drain(time.monotonic() + 10)
            done = app.library_ingest_jobs.get_job(a.job_id)
            assert done.state.value == "done", done
            assert "Committed before capture." in db.get_media_by_id(done.media_id)["content"]
            assert len(pool.calls) == 1
            assert app.library_ingest_jobs.get_job(b.job_id).state.value == "queued"
            assert any(row["source_path"] == str(second) for row in store.all_jobs())
            assert not pool.terminated
            assert not worker_leases(db)
            app._ingest_maintenance_resume()
            assert len(pool.calls) == 2
            app._ingest_maintenance_close_admission()
            pool._spawn(pool._run_one, pool.calls[1])
            assert await app._ingest_maintenance_drain(time.monotonic() + 10)
            assert app.library_ingest_jobs.get_job(b.job_id).state.value == "done"
        print("INGEST_CAPTURE_BOUNDARY_OK")
    finally:
        app._shutdown_ingest_parse_pool()
        db.close_connection()
        store.close()
asyncio.run(main())
"""


def test_live_ingest_completes_admitted_publication_and_resumes_queue(tmp_path):
    root = tmp_path.resolve()
    for directory in ("home", "config", "data"):
        (root / directory).mkdir(mode=0o700)
    env = os.environ.copy()
    env.update(
        HOME=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_DATA_HOME=str(root / "data"),
        TLDW_CONFIG_PATH=str(root / "config/config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", _PILOT],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-1000:]
    assert "INGEST_CAPTURE_BOUNDARY_OK" in result.stdout
