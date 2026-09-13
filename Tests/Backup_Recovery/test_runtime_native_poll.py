"""Native pause probes must leave the UI responsive and settle before exit."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import runtime_maintenance as maintenance
from tldw_chatbook.Backup_Recovery import storage_admission as storage


@pytest.mark.asyncio
async def test_monitor_remains_responsive_during_one_blocked_native_probe(monkeypatch):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = []

    def probe():
        calls.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        try:
            release.wait(3)
            return False
        finally:
            finished.set()

    monkeypatch.setattr(storage, "_local_pause_requested", probe)
    monitoring = asyncio.create_task(maintenance.monitor_app(SimpleNamespace()))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.sleep(0.15)
        assert not finished.is_set(), "native polling blocked the event loop"
        assert len(calls) == 1, "the monitor started overlapping native probes"
        assert calls[0] != threading.get_ident()
    finally:
        release.set()
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)
    assert finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (False, True))
async def test_monitor_cancellation_waits_for_native_probe_release(monkeypatch, failure):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()

    def probe():
        loop.call_soon_threadsafe(entered.set)
        try:
            release.wait(3)
            if failure:
                raise OSError("synthetic native failure")
            return False
        finally:
            finished.set()

    monkeypatch.setattr(storage, "_local_pause_requested", probe)
    monitoring = asyncio.create_task(maintenance.monitor_app(SimpleNamespace()))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        for _ in range(2):
            monitoring.cancel()
            await asyncio.sleep(0)
            assert not monitoring.done(), "cancellation abandoned the native probe"
            assert not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(monitoring, 5)
        assert finished.is_set()
    finally:
        release.set()
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", (OSError, ValueError, RuntimeError))
async def test_probe_failure_is_mapped_on_monitor_task_and_retried(monkeypatch, error_type):
    loop = asyncio.get_running_loop()
    retried = asyncio.Event()
    assignments = []
    calls = 0

    class App:
        def __setattr__(self, name, value):
            assignments.append((name, value, threading.get_ident(), asyncio.current_task()))

    def probe():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error_type("synthetic private path must not enter UI state")
        loop.call_soon_threadsafe(retried.set)
        return False

    monkeypatch.setattr(storage, "_local_pause_requested", probe)
    monitoring = asyncio.create_task(maintenance.monitor_app(App()))
    try:
        await asyncio.wait_for(retried.wait(), 5)
        assert assignments == [
            ("_backup_maintenance_error", "admission_state_unavailable",
             threading.get_ident(), monitoring)
        ]
    finally:
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)


@pytest.mark.asyncio
async def test_requested_pause_keeps_runtime_coordination_on_monitor_task(monkeypatch):
    resumed = asyncio.Event()
    events = []

    def record(name):
        events.append((name, threading.get_ident(), asyncio.current_task()))

    class Pause:
        def drain(self, deadline):
            record("drain")
            return True

        def retire_startup(self, runtime):
            record("retire_startup")

    class Runtime:
        def __init__(self, app):
            record("construct")
            self.pause = None
            self.closed = []

        async def settle_producers(self, deadline):
            record("settle_producers")

        def retire_local_caches(self):
            record("retire_local_caches")
            self.pause = Pause()

        async def resume(self):
            record("resume")
            self.pause = None
            resumed.set()

    monkeypatch.setattr(storage, "_local_pause_requested", lambda: True)
    monkeypatch.setattr(maintenance, "RuntimeMaintenance", Runtime)
    app = SimpleNamespace()
    monitoring = asyncio.create_task(maintenance.monitor_app(app))
    try:
        await asyncio.wait_for(resumed.wait(), 5)
        assert events == [
            (name, threading.get_ident(), monitoring)
            for name in (
                "construct", "settle_producers", "retire_local_caches",
                "drain", "retire_startup", "resume",
            )
        ]
        assert app._backup_runtime_maintenance is None
    finally:
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)
