"""Startup recovery retains its original ledger callback through shutdown."""

import asyncio
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
import inspect
import sqlite3
import sys
import threading
from types import CodeType, SimpleNamespace

import pytest

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["controller", "runtime"])
@private_profile_test
async def test_original_fleet_recovery_is_joined_before_shutdown(
    request, tmp_path, owner
):
    from Tests.Chat.test_console_fleet_wake import _controller_rig
    from Tests.conftest import _close_database_instance
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB import base_db
    from tldw_chatbook.DB.automatic_work import AutomaticWorkLedger

    notes, app, database, store, _session, _gateway, _bridge, controller = (
        _controller_rig(tmp_path)
    )
    runtime = ConsoleRuntime(app, canvas_enabled_reader=lambda: False)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    wake = controller.fleet_wake
    ledger = database.automatic_work
    reader = AutomaticWorkLedger.recover
    monitor = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for subject, name in (
        (ConsoleFleetWakeCoordinator, "recover"),
        (ConsoleFleetWakeCoordinator, "start_recovery"),
        (ConsoleFleetWakeCoordinator, "dispose"),
        (ConsoleChatController, "shutdown"),
        (ConsoleRuntime, "dispose"),
        (ConsoleRuntime, "_dispose_owned"),
        (AutomaticWorkLedger, "recover"),
        (base_db, "run_owned_db_call"),
        (_WorkItem, "run"),
    ):
        monitor._pin(getattr(subject, name))
        monitor.slots.append((subject, name, inspect.getattr_static(subject, name)))
    invoke_code = next(
        code
        for code in base_db.run_owned_db_call.__code__.co_consts
        if isinstance(code, CodeType) and code.co_name == "invoke"
    )
    later = tuple(
        getattr(ConsoleFleetWakeCoordinator, name)
        for name in ("seed_from_marks", "seed_progress_hints", "_notify_ui")
    )
    entered, release = threading.Event(), threading.Event()
    held, errors, late_calls = {}, [], []
    recovery = shutdown = None
    retained = True
    retired = False

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def observe_start(code, _offset):
        frame = sys._getframe(1)
        try:
            if frame.f_locals.get("self") is wake and wake._disposed:
                late_calls.append(code.co_name)
        finally:
            del frame

    def hold(code, _offset, value):
        frame = ancestor = invocation = None
        try:
            frame = sys._getframe(1)
            if frame.f_locals.get("self") is not ledger or held:
                return
            assert code is reader.__code__ and frame.f_globals is reader.__globals__
            assert value == 0
            ancestor = frame.f_back
            while (
                ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__
            ):
                if ancestor.f_code is invoke_code:
                    invocation = ancestor
                ancestor = ancestor.f_back
            assert invocation is not None and ancestor is not None
            assert invocation.f_globals is base_db.run_owned_db_call.__globals__
            assert invocation.f_locals["database"] is database
            operation = invocation.f_locals["operation"]
            assert operation.__self__ is ledger and operation.__func__ is reader
            assert invocation.f_locals["args"] == ()
            assert invocation.f_locals["kwargs"] == {"current_owner_id": wake._owner_id}
            item = ancestor.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            connection = database._thread_local.conn
            participant = database._maintenance_participant
            with storage._lock:
                lease = participant.connections[connection]
                assert lease in storage._live_leases and not closed(connection)
                assert lease.resource_thread is threading.current_thread()
            held.update(
                future=item.future,
                connection=connection,
                participant=participant,
                lease=lease,
            )
            entered.set()
            assert release.wait(10), "Original recovery callback was not released"
        except BaseException as error:
            errors.append(type(error).__name__)
            entered.set()
        finally:
            del frame, ancestor, invocation

    for candidate in range(5, 0, -1):
        if candidate == sys.monitoring.DEBUGGER_ID:
            continue
        try:
            sys.monitoring.use_tool_id(candidate, "fleet-recovery-native")
        except ValueError:
            continue
        tool = candidate
        break
    else:
        raise AssertionError("No local monitoring slot")
    assert sys.monitoring.get_events(tool) == 0
    assert (
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, hold)
        is None
    )
    assert (
        sys.monitoring.register_callback(
            tool, sys.monitoring.events.PY_START, observe_start
        )
        is None
    )
    sys.monitoring.set_local_events(
        tool, reader.__code__, sys.monitoring.events.PY_RETURN
    )
    for callback in later:
        monitor._pin(callback)
        monitor.slots.append((ConsoleFleetWakeCoordinator, callback.__name__, callback))
        sys.monitoring.set_local_events(
            tool, callback.__code__, sys.monitoring.events.PY_START
        )
    try:
        wake.start_recovery()
        recovery = wake._recovery_task
        assert type(recovery) is asyncio.Task
        assert await asyncio.to_thread(entered.wait, 10)
        assert held and not errors, errors
        operation = (
            controller.shutdown()
            if owner == "controller"
            else runtime.dispose(timeout_seconds=0)
        )
        shutdown = asyncio.create_task(operation)
        await asyncio.wait({shutdown}, timeout=0.05)
        retained = not shutdown.done()
        assert wake._disposed
        for _ in range(2):
            shutdown.cancel()
            await asyncio.sleep(0.01)
            retained = retained and not shutdown.done()
        assert not held["future"].done() and not closed(held["connection"])
    finally:
        release.set()
        try:
            if held:
                outcome = await asyncio.gather(
                    asyncio.wrap_future(held["future"]), return_exceptions=True
                )
            results = await asyncio.gather(
                *(task for task in (recovery, shutdown) if task is not None),
                return_exceptions=True,
            )
            if held:
                with storage._lock:
                    retired = (
                        closed(held["connection"])
                        and held["lease"] not in storage._live_leases
                        and held["connection"] not in held["participant"].connections
                    )
            ready = wake._recovery_ready
            after_dispose = tuple(late_calls)
        finally:
            sys.monitoring.set_local_events(tool, reader.__code__, 0)
            for callback in later:
                sys.monitoring.set_local_events(tool, callback.__code__, 0)
            assert (
                sys.monitoring.register_callback(
                    tool, sys.monitoring.events.PY_RETURN, None
                )
                is hold
            )
            assert (
                sys.monitoring.register_callback(
                    tool, sys.monitoring.events.PY_START, None
                )
                is observe_start
            )
            sys.monitoring.free_tool_id(tool)
            receipt = monitor.close()
            # Only fixture cleanup follows the captured production-retirement verdict.
            await runtime.dispose()
            _close_database_instance(notes)
            database.close()
    assert receipt["original_source_current"] and not errors, (receipt, errors)
    assert outcome == [0] and results[0] is None, (outcome, results)
    assert retired, "Original recovery worker connection or lease survived completion"
    assert (
        retained
    ), "Shutdown returned while the original recovery callback was native-live"
    assert isinstance(results[1], asyncio.CancelledError), results
    assert not ready and not after_dispose, (ready, after_dispose)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deferred", [False, True], ids=["new-request", "deferred-request"]
)
async def test_disposed_fleet_never_schedules_recovery(deferred):
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator

    wake = ConsoleFleetWakeCoordinator(SimpleNamespace(_buddy_sink=None))
    if deferred:
        # The original public request runs without a loop, then later captures one.
        await asyncio.to_thread(wake.start_recovery)
        assert wake._recovery_requested and wake._recovery_task is None
    wake.dispose()
    try:
        if deferred:
            wake.capture_loop_if_running()
        else:
            wake.start_recovery()
        assert wake._recovery_task is None
    finally:
        if wake._recovery_task is not None:
            await wake._recovery_task


@pytest.mark.asyncio
@private_profile_test
async def test_original_recovery_cannot_publish_after_progress_disposal(
    request, tmp_path
):
    from Tests.Chat.test_console_fleet_wake import _controller_rig
    from Tests.conftest import _close_database_instance
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator

    notes, _app, database, _store, _session, _gateway, _bridge, controller = (
        _controller_rig(tmp_path)
    )
    wake = controller.fleet_wake
    original = ConsoleFleetWakeCoordinator.seed_progress_hints
    monitor = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for name in ("recover", "seed_progress_hints", "dispose"):
        monitor._pin(getattr(ConsoleFleetWakeCoordinator, name))
        monitor.slots.append(
            (
                ConsoleFleetWakeCoordinator,
                name,
                inspect.getattr_static(ConsoleFleetWakeCoordinator, name),
            )
        )
    entered, disposed = threading.Event(), threading.Event()
    errors = []

    def dispose_after_progress():
        if not entered.wait(10):
            errors.append("progress not reached")
            disposed.set()
            return
        wake.dispose()
        disposed.set()

    def hold(code, _offset, _value):
        frame = sys._getframe(1)
        try:
            if frame.f_locals.get("self") is not wake:
                return
            assert code is original.__code__ and frame.f_globals is original.__globals__
            entered.set()
            assert disposed.wait(10), "Off-thread disposal did not finish"
        finally:
            del frame

    for candidate in range(5, 0, -1):
        if candidate == sys.monitoring.DEBUGGER_ID:
            continue
        try:
            sys.monitoring.use_tool_id(candidate, "recovery-final-publication")
        except ValueError:
            continue
        tool = candidate
        break
    else:
        raise AssertionError("No local monitoring slot")
    assert sys.monitoring.get_events(tool) == 0
    assert (
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, hold)
        is None
    )
    sys.monitoring.set_local_events(
        tool, original.__code__, sys.monitoring.events.PY_RETURN
    )
    closer = threading.Thread(target=dispose_after_progress)
    closer.start()
    try:
        wake.start_recovery()
        await wake.wait_for_recovery()
        assert entered.is_set() and disposed.is_set() and not errors, errors
        assert wake._disposed
        assert (
            not wake._recovery_ready
        ), "Recovery published readiness after off-thread disposal"
    finally:
        entered.set()
        await asyncio.to_thread(closer.join, 10)
        sys.monitoring.set_local_events(tool, original.__code__, 0)
        assert (
            sys.monitoring.register_callback(
                tool, sys.monitoring.events.PY_RETURN, None
            )
            is hold
        )
        sys.monitoring.free_tool_id(tool)
        receipt = monitor.close()
        await controller.shutdown()
        _close_database_instance(notes)
        database.close()
    assert not closer.is_alive() and receipt["original_source_current"]
