"""Scheduled maintenance must outlive the actual native callback it cancels."""

import asyncio
import sqlite3
import sys
import threading

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cancel", "dispose", "repeated_dispose_cancel"])
@private_profile_test
async def test_scheduled_maintenance_retains_original_native_callback(request, action):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_trace_legacy import LegacyTraceNormalizer
    from tldw_chatbook.Chat.console_trace_maintenance import LegacyTraceMaintenance
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(
        config.get_user_data_dir() / "maintenance-lifetime.sqlite",
        client_id="maintenance-lifetime",
    )
    participant = _repository_participant(database)
    runtime = ConsoleRuntime(app=None)
    entered, release = threading.Event(), threading.Event()
    captured = {}
    callback = LegacyTraceMaintenance.run_batch
    callback_code = callback.__code__
    caller_thread = threading.current_thread()

    def hold_return(code, offset, result):
        if code is not callback_code or captured:
            return
        frame = sys._getframe(1)
        if frame.f_locals.get("self").db is not database:
            return
        assert threading.current_thread() is not caller_thread
        assert result.admitted
        connection = database._local.conn
        with storage._lock:
            lease = participant.connections[connection]
            operations = tuple(
                item for item in storage._operations if item.participant is participant
            )
            assert lease in storage._live_leases and operations
        captured.update(connection=connection, lease=lease, operations=operations)
        entered.set()
        assert release.wait(
            20
        ), "test did not release the original maintenance callback"

    def retired():
        if not captured:
            return False
        try:
            sqlite3.Connection.in_transaction.__get__(captured["connection"])
        except sqlite3.ProgrammingError:
            closed = True
        else:
            closed = False
        with storage._lock:
            return (
                closed
                and captured["lease"] not in storage._live_leases
                and captured["connection"] not in participant.connections
                and all(
                    item not in storage._operations for item in captured["operations"]
                )
            )

    monitoring = sys.monitoring
    tool = next(value for value in range(1, 6) if monitoring.get_tool(value) is None)
    closing = None
    monitoring.use_tool_id(tool, "trace-maintenance-retirement")
    try:
        monitoring.register_callback(tool, monitoring.events.PY_RETURN, hold_return)
        monitoring.set_local_events(tool, callback_code, monitoring.events.PY_RETURN)
        runtime._schedule_legacy_trace_maintenance(
            database, lambda: LegacyTraceNormalizer(database)
        )
        maintenance = runtime._legacy_trace_maintenance_task
        # Preserve the production five-second readiness delay and real SQL.
        assert await asyncio.to_thread(entered.wait, 15)
        assert LegacyTraceMaintenance.run_batch is callback
        assert callback.__code__ is callback_code
        assert not retired()
        if action == "cancel":
            closing = maintenance
            closing.cancel()
        else:
            closing = asyncio.create_task(runtime.dispose())
        if action == "repeated_dispose_cancel":
            for _ in range(200):
                if maintenance.cancelling():
                    break
                await asyncio.sleep(0.005)
            assert maintenance.cancelling()
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
        done, _ = await asyncio.wait({closing}, timeout=0.1)
        assert (
            not done
        ), "maintenance owner finished while its original native callback was live"
        assert not retired()
        release.set()
        await asyncio.gather(closing, return_exceptions=True)
        assert retired(), "maintenance owner returned before native retirement"
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await runtime.dispose()
        for _ in range(400):
            if not captured or retired():
                break
            await asyncio.sleep(0.005)
        monitoring.set_local_events(tool, callback_code, 0)
        monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
        monitoring.free_tool_id(tool)
        database.close_connection()
        assert not captured or retired()
