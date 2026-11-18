"""One original trace checkpoint owns its adjacent native DB accesses."""

from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager, nullcontext

import pytest

from Tests.private_profile import private_profile_test


@contextmanager
def _observe(code, event, callback):
    monitoring = sys.monitoring
    slot = next(
        value for value in range(6)
        if value != monitoring.DEBUGGER_ID and monitoring.get_tool(value) is None
    )
    monitoring.use_tool_id(slot, "trace-checkpoint-operation")

    def observe(actual, *_args):
        if actual is code:
            callback(sys._getframe(1))

    monitoring.register_callback(slot, event, observe)
    monitoring.set_local_events(slot, code, event)
    try:
        yield
    finally:
        monitoring.set_local_events(slot, code, 0)
        monitoring.register_callback(slot, event, None)
        monitoring.free_tool_id(slot)


@contextmanager
def _record_admissions():
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    origins = {
        storage_admission._repository_operation.__wrapped__.__code__: "repository operation",
        connect_private_sqlite.__code__: "native connection",
    }
    acquisitions = []

    def record(frame):
        while frame is not None and frame.f_code not in origins:
            frame = frame.f_back
        acquisitions.append(origins[frame.f_code] if frame is not None else "unexpected")

    with _observe(storage_admission._acquire_storage.__code__, sys.monitoring.events.PY_START, record):
        yield acquisitions


@private_profile_test
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_trace_checkpoint_one_admission_retains_policy_and_native_ownership(
    tmp_path, request, borrowed, fail,
):
    from tldw_chatbook.Backup_Recovery.participants import core_operation_active
    from tldw_chatbook.Chat.console_trace_service import (
        trace_critical_write_checkpoint_policy,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(tmp_path / "trace-checkpoint.db", "trace-checkpoint")
    connection = database.get_connection()
    connection.execute("CREATE TABLE checkpoint_probe (value INTEGER)")
    if borrowed:
        connection.execute("PRAGMA wal_autocheckpoint=37")
    else:
        database.close_connection()
    retiring = []
    entered = None

    def retiring_handle(frame):
        if frame.f_locals["self"] is database:
            handle = frame.f_locals["conn"]
            retiring.append((
                handle,
                handle.execute("PRAGMA wal_autocheckpoint").fetchone()[0],
                core_operation_active(database),
            ))

    try:
        with (
            _record_admissions() as acquisitions,
            _observe(
                CharactersRAGDB._close_connection_handle.__code__,
                sys.monitoring.events.PY_START,
                retiring_handle,
            ),
            pytest.raises(ValueError, match="original trace write failed")
            if fail else nullcontext(),
        ):
            with trace_critical_write_checkpoint_policy(database):
                entered = database.get_connection()
                assert entered.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 0
                assert core_operation_active(database)
                with database.transaction(immediate=True) as cursor:
                    assert database.get_connection() is entered
                    cursor.execute("INSERT INTO checkpoint_probe VALUES (1)")
                    if fail:
                        raise ValueError("original trace write failed")

        assert acquisitions == (
            ["repository operation"] if borrowed else ["repository operation", "native connection"]
        )
        assert not core_operation_active(database)
        assert entered is not None
        if borrowed:
            assert entered is connection
            assert retiring == []
            assert database.registered_connection_count() == 1
            assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 37
        else:
            assert retiring == [(entered, 1000, False)]
            assert database.registered_connection_count() == 0
            with pytest.raises(sqlite3.ProgrammingError):
                entered.execute("SELECT 1")
        with database.transaction() as cursor:
            assert cursor.execute("SELECT COUNT(*) FROM checkpoint_probe").fetchone()[0] == (
                0 if fail else 1
            )
    finally:
        database.close_connection()


@private_profile_test
@pytest.mark.bootstrap_profile
def test_original_trace_dispatch_and_response_own_separate_operations_and_connections(
    tmp_path, request,
):
    from Tests.Chat.test_console_trace_call_lifecycle import (
        _identity,
        _unavailable_request,
    )
    from tldw_chatbook.Chat.console_trace_models import TraceCallState
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.Chat.console_trace_service import (
        ConsoleTraceCallBoundary,
        ConsoleTraceService,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(tmp_path / "trace-boundaries.db", "trace-boundaries")
    repository = ConsoleTraceRepository()
    service = ConsoleTraceService(repository)
    try:
        identity = _identity(database, repository)
        provenance, bundle, admission = _unavailable_request(identity)
        boundary = ConsoleTraceCallBoundary(
            service, database, identity, admission,
            lambda: "2026-11-18T00:00:00Z",
        )
        boundary.reserve()
        database.close_connection()
        with _record_admissions() as acquisitions:
            started = boundary.mark_dispatch_started(bundle, provenance)
            assert started.state is TraceCallState.DISPATCH_STARTED
            assert acquisitions == ["repository operation", "native connection"]
            assert database.registered_connection_count() == 0
            acquisitions.clear()
            assert boundary.mark_response_started() is True
            assert acquisitions == ["repository operation", "native connection"]
            assert database.registered_connection_count() == 0
        with database.transaction() as cursor:
            durable = repository.get_call(cursor, started.call_id)
            assert durable.state is TraceCallState.RESPONSE_STARTED
            assert durable.surface_node_id == started.surface_node_id
            assert durable.request_header_id == started.request_header_id
    finally:
        database.close_connection()


@private_profile_test
@pytest.mark.bootstrap_profile
def test_trace_checkpoint_closed_participant_refuses_before_policy_change(
    tmp_path, request,
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.participants import repository_participant
    from tldw_chatbook.Chat.console_trace_service import (
        trace_critical_write_checkpoint_policy,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(tmp_path / "trace-refusal.db", "trace-refusal")
    connection = database.get_connection()
    connection.execute("PRAGMA wal_autocheckpoint=37")
    participant = repository_participant(database)
    participant.close_admission()
    try:
        with pytest.raises(RecoveryRequired):
            with trace_critical_write_checkpoint_policy(database):
                pytest.fail("Closed native owner admitted a trace write")
        assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 37
        assert database.registered_connection_count() == 1
    finally:
        participant.resume()
        database.close_connection()


@pytest.mark.parametrize("failure", [None, "body", "restore"])
def test_custom_trace_database_keeps_original_checkpoint_protocol(failure):
    from tldw_chatbook.Chat.console_trace_service import (
        trace_critical_write_checkpoint_policy,
    )

    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA wal_autocheckpoint=37")

    class CustomConnection:
        def execute(self, sql):
            if failure == "restore" and sql == "PRAGMA wal_autocheckpoint=37":
                raise RuntimeError("original policy restore failed")
            return connection.execute(sql)

    class CustomDatabase:
        # Deliberately no core repository or is_memory_db contract.
        def get_connection(self):
            return CustomConnection()

    try:
        with (
            pytest.raises(RuntimeError, match="original policy restore failed")
            if failure == "restore" else
            pytest.raises(ValueError, match="original body failed")
            if failure == "body" else nullcontext()
        ):
            with trace_critical_write_checkpoint_policy(CustomDatabase()):
                assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 0
                if failure == "body":
                    raise ValueError("original body failed")
        assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == (
            0 if failure == "restore" else 37
        )
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        connection.close()
