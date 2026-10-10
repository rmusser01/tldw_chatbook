"""Original scope existence callbacks retire their own native connections."""

from concurrent.futures import ThreadPoolExecutor
import inspect
import sqlite3
import sys
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events as scope

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("kind", ["note", "media"])
@pytest.mark.parametrize("borrowed", [False, True], ids=["cold", "borrowed"])
@pytest.mark.parametrize("fail_sql", [False, True], ids=["success", "sql-error"])
def test_original_scope_existence_read_retires_only_its_native_handle(
    tmp_path, kind, borrowed, fail_sql
):
    database_type = CharactersRAGDB if kind == "note" else MediaDatabase
    database = database_type(tmp_path / "scope.sqlite", "scope-retirement")
    if fail_sql:
        # Exercise a real SQLite failure after acquiring the actual handle.
        table = "notes" if kind == "note" else "Media"
        database.get_connection().execute(f"DROP TABLE {table}")
    database.close_connection()
    app = SimpleNamespace(chachanotes_db=database, media_db=database)
    getter = inspect.unwrap(database_type.get_connection)
    captured = {}

    def observe(code, _offset, connection):
        frame = ancestor = None
        try:
            frame = sys._getframe(1)
            if frame.f_locals.get("self") is not database:
                return
            ancestor = frame.f_back
            while ancestor is not None:
                if ancestor.f_code is scope._existing_ids_sync.__code__:
                    break
                ancestor = ancestor.f_back
            if ancestor is None:
                return
            assert code is getter.__code__
            assert frame.f_globals is getter.__globals__
            assert ancestor.f_locals["app"] is app
            assert isinstance(connection, sqlite3.Connection)
            participant = database._maintenance_participant
            with storage._lock:
                lease = participant.connections[connection]
                assert lease in storage._live_leases
                assert lease.resource_thread is threading.current_thread()
            captured.update(connection=connection, participant=participant, lease=lease)
        finally:
            del frame, ancestor

    for tool in range(5, 0, -1):
        if tool == sys.monitoring.DEBUGGER_ID:
            continue
        try:
            sys.monitoring.use_tool_id(tool, "scope-native-retirement")
        except ValueError:
            continue
        break
    else:
        raise AssertionError("No local monitoring slot")
    assert sys.monitoring.get_events(tool) == 0
    assert (
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, observe)
        is None
    )
    sys.monitoring.set_local_events(
        tool, getter.__code__, sys.monitoring.events.PY_RETURN
    )

    def read():
        previous = database.get_connection() if borrowed else None
        if previous is not None:
            previous.execute("BEGIN")
        try:
            if fail_sql:
                with pytest.raises(scope._ScopeExistenceReadError):
                    scope._existing_ids_sync(app, kind, frozenset({"missing"}))
            else:
                assert (
                    scope._existing_ids_sync(app, kind, frozenset({"missing"}))
                    == frozenset()
                )
            assert captured, "Original SQLite getter was not observed"
            connection = captured["connection"]
            if borrowed:
                assert connection is previous and connection.in_transaction
                assert connection.execute("SELECT 1").fetchone()[0] == 1
                with storage._lock:
                    assert captured["lease"] in storage._live_leases
            else:
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    connection.execute("SELECT 1")
                with storage._lock:
                    assert connection not in captured["participant"].connections
                    assert captured["lease"] not in storage._live_leases
        finally:
            # Assertions above precede this exact fixture owner's cleanup.
            if previous is not None and previous.in_transaction:
                previous.rollback()
            database.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(read).result(timeout=15)
    finally:
        sys.monitoring.set_local_events(tool, getter.__code__, 0)
        assert (
            sys.monitoring.register_callback(
                tool, sys.monitoring.events.PY_RETURN, None
            )
            is observe
        )
        sys.monitoring.free_tool_id(tool)
        database.close_connection()
