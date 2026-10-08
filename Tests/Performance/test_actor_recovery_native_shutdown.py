"""PYTEST_DONT_REWRITE: original Actor Pack worker cancellation and native custody."""

import asyncio
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
import os
from pathlib import Path
import sqlite3
import sys
import threading

import pytest

from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shutdown_route", ["workers", "app_owner", "app_owner_recancel"]
)
@private_profile_test
async def test_original_actor_recovery_worker_retires_before_shutdown_returns(
    request, tmp_path, shutdown_route
):
    from tldw_chatbook import app as app_source
    from tldw_chatbook.Actor_Packs.persona_coordinator import (
        PersonaActorPackCoordinator,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    creators = OriginalPrivateAppCreators(
        app_source, Path(os.environ["XDG_DATA_HOME"]).parent
    )
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    runtime, database = app.console_runtime, app.chachanotes_db
    creator = database._local.conn
    callback = PersonaActorPackCoordinator.ensure_recovered
    entered, release = threading.Event(), threading.Event()
    held, errors = {}, []
    monitor = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for owner, name in (
        (PersonaActorPackCoordinator, "ensure_recovered"),
        (app_source.TldwCli, "ensure_actor_pack_recovery"),
        (app_source.TldwCli, "boot_worker_starters"),
        (app_source.TldwCli, "_cancel_and_settle_workers"),
        (_WorkItem, "run"),
    ):
        original = getattr(owner, name)
        monitor._pin(original)
        monitor.slots.append((owner, name, original))
    tool = None
    worker = shutdown = None

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    async def consume_native_result():
        future = held["future"]
        try:
            await asyncio.wrap_future(future)
        except asyncio.CancelledError as error:
            # Closing App admission rejects publication after the original
            # callback has returned and retired its connection.
            assert shutdown_route in {"app_owner", "app_owner_recancel"}
            assert future.done() and not future.cancelled()
            assert future.exception() is error

    def hold(code, offset, value):
        if held:
            return
        frame = ancestor = None
        try:
            frame = sys._getframe(1)
            if frame.f_locals.get("self") is not app.persona_actor_pack_coordinator:
                return
            assert code is callback.__code__ and frame.f_code is code
            assert frame.f_globals is callback.__globals__
            ancestor = frame.f_back
            while (
                ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__
            ):
                ancestor = ancestor.f_back
            assert ancestor is not None
            item = ancestor.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            connection = database._local.conn
            participant = database._maintenance_participant
            with storage._lock:
                lease = participant.connections[connection]
                assert lease in storage._live_leases
                assert lease.resource_thread is threading.current_thread()
                assert not closed(connection) and connection is not creator
            held.update(
                future=item.future,
                connection=connection,
                participant=participant,
                lease=lease,
            )
            entered.set()
            assert release.wait(10), "Original Actor Pack callback was never released"
        except BaseException as error:
            errors.append(type(error).__name__)
            entered.set()
        finally:
            frame = ancestor = None

    try:
        for candidate in range(5, 0, -1):
            if candidate == sys.monitoring.DEBUGGER_ID:
                continue
            try:
                sys.monitoring.use_tool_id(candidate, "actor-recovery-native-shutdown")
            except ValueError:
                continue
            tool = candidate
            break
        assert tool is not None and sys.monitoring.get_events(tool) == 0
        assert (
            sys.monitoring.register_callback(
                tool, sys.monitoring.events.PY_RETURN, hold
            )
            is None
        )
        assert sys.monitoring.get_local_events(tool, callback.__code__) == 0
        sys.monitoring.set_local_events(
            tool, callback.__code__, sys.monitoring.events.PY_RETURN
        )
        with app._context():
            worker = app.boot_worker_starters()["actor_pack_recovery"]()
            assert worker.node is app and worker.group == "actor_pack_recovery"
            assert await asyncio.to_thread(entered.wait, 10)
            assert held and not errors, errors
            original_task = worker._task
            shutdown = asyncio.create_task(
                app._cancel_and_settle_workers("original Actor Pack native control")
                if shutdown_route == "workers"
                else app._shutdown_app_owned_lifecycles()
            )
            await asyncio.wait({shutdown}, timeout=0.35)
            held["shutdown_returned_while_native_held"] = shutdown.done()
            held["original_task_terminal_while_native_held"] = original_task.done()
            if shutdown_route == "app_owner_recancel":
                for _ in range(2):
                    shutdown.cancel()
                    await asyncio.sleep(0)
                    assert not shutdown.done() and not original_task.done()
            assert held["future"].running() and not held["future"].done()
            assert not closed(held["connection"])
            release.set()
            await consume_native_result()
            if shutdown_route == "app_owner_recancel":
                with pytest.raises(asyncio.CancelledError):
                    await shutdown
            else:
                await shutdown
            await asyncio.gather(original_task, return_exceptions=True)
        assert not errors, errors
        assert closed(held["connection"])
        if shutdown_route == "workers":
            assert not closed(creator)
        with storage._lock:
            assert held["connection"] not in held["participant"].connections
            assert held["lease"] not in storage._live_leases
    finally:
        release.set()
        if "future" in held:
            await consume_native_result()
        if shutdown is not None:
            await asyncio.gather(shutdown, return_exceptions=True)
        if worker is not None:
            await asyncio.gather(worker._task, return_exceptions=True)
        if tool is not None:
            sys.monitoring.set_local_events(tool, callback.__code__, 0)
            assert (
                sys.monitoring.register_callback(
                    tool, sys.monitoring.events.PY_RETURN, None
                )
                is hold
            )
            assert sys.monitoring.get_events(tool) == 0
            sys.monitoring.free_tool_id(tool)
        receipt = monitor.close()
        assert receipt["original_source_current"]
        await runtime.dispose()
        creators.retire()
    assert not held[
        "shutdown_returned_while_native_held"
    ], "App worker shutdown returned with the original Actor Pack database callback still running"
    assert not held["original_task_terminal_while_native_held"]


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_actor_recovery_closed_owner_never_dispatches_late_callback():
    from types import SimpleNamespace
    from tldw_chatbook.app import TldwCli

    app = SimpleNamespace(
        _actor_pack_recovery_closed=True, _actor_pack_recovery_reads=set()
    )
    calls = []
    with pytest.raises(asyncio.CancelledError):
        await TldwCli._run_actor_pack_recovery_owned(app, lambda: calls.append(True))
    assert not calls and not app._actor_pack_recovery_reads
