"""Scheduler stop configuration and file reads belong to its owned worker."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@private_profile_test
async def test_default_stop_resolution_and_real_transitions_run_off_loop(
    monkeypatch, request
):
    from tldw_chatbook import emergency_stop

    scheduler = SchedulerLoop(SimpleNamespace(), {})
    origin = threading.get_ident()
    resolve, read = (
        emergency_stop.default_emergency_stop_path,
        emergency_stop.is_emergency_stopped,
    )
    resolution_threads, read_threads = [], []

    def observed_resolve():
        resolution_threads.append(threading.get_ident())
        return resolve()

    def observed_read(path):
        read_threads.append(threading.get_ident())
        return read(path)

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", observed_resolve)
    monkeypatch.setattr(emergency_stop, "is_emergency_stopped", observed_read)
    path = await asyncio.to_thread(resolve)
    try:
        assert not await scheduler._emergency_stopped()
        emergency_stop.set_emergency_stop(path, reason="test stop")
        assert await scheduler._emergency_stopped()
        emergency_stop.clear_emergency_stop(path)
        assert not await scheduler._emergency_stopped()
        assert len(resolution_threads) == len(read_threads) == 3
        assert all(thread != origin for thread in resolution_threads + read_threads)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["resolve", "read"])
async def test_stop_resolution_or_read_failure_remains_fail_safe(
    monkeypatch, tmp_path, failure
):
    from tldw_chatbook import emergency_stop

    scheduler = SchedulerLoop(SimpleNamespace(), {})
    origin, threads = threading.get_ident(), []

    def resolve():
        threads.append(threading.get_ident())
        if failure == "resolve":
            raise OSError("unavailable selection")
        return tmp_path / "stop.json"

    def read(path):
        threads.append(threading.get_ident())
        raise OSError("unavailable sentinel")

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", resolve)
    monkeypatch.setattr(emergency_stop, "is_emergency_stopped", read)
    assert await scheduler._emergency_stopped()
    assert threads and all(thread != origin for thread in threads)


@pytest.mark.asyncio
async def test_explicit_stop_path_uses_fresh_worker_read_without_default_lookup(
    monkeypatch, tmp_path
):
    from tldw_chatbook import emergency_stop

    path = tmp_path / "explicit-stop.json"
    scheduler = SchedulerLoop(SimpleNamespace(), {}, emergency_stop_path=path)
    monkeypatch.setattr(
        emergency_stop,
        "default_emergency_stop_path",
        lambda: pytest.fail("explicit owner must not resolve a default path"),
    )
    assert not await scheduler._emergency_stopped()
    emergency_stop.set_emergency_stop(path)
    assert await scheduler._emergency_stopped()
    emergency_stop.clear_emergency_stop(path)
    assert not await scheduler._emergency_stopped()


@pytest.mark.asyncio
async def test_cancelled_stop_lookup_retains_worker_until_maintenance_drain(
    monkeypatch, tmp_path
):
    from tldw_chatbook import emergency_stop

    scheduler = SchedulerLoop(SimpleNamespace(), {})
    entered, release = threading.Event(), threading.Event()

    def resolve():
        entered.set()
        assert release.wait(5)
        return tmp_path / "stop.json"

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", resolve)
    pending = asyncio.create_task(scheduler._emergency_stopped())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        scheduler._maintenance_close_admission()
        assert not await scheduler._maintenance_drain(time.monotonic() + 0.05)
        release.set()
        assert await scheduler._maintenance_drain(time.monotonic() + 5)
        assert not scheduler._maintenance_db_tasks
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
