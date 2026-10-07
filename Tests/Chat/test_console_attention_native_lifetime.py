"""The original attention callback retires only its newly acquired database."""

from concurrent.futures import ThreadPoolExecutor
import sqlite3
import sys
import threading
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.parametrize("action", ["owned", "borrowed", "query_error"])
@private_profile_test
async def test_original_attention_callback_retires_its_owned_connection(
    request, action
):
    borrowed = action == "borrowed"
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.conversation_local_marks_service import (
        ConversationLocalMarksService,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(
        config.get_user_data_dir() / "attention.sqlite", client_id="attention-native"
    )
    database.close_connection()
    participant = _repository_participant(database)
    service = ConversationLocalMarksService(database)
    runtime = ConsoleRuntime(SimpleNamespace(conversation_local_marks_service=service))
    original = ConversationLocalMarksService.list_console_unseen_marks
    code = original.__code__
    captured = {}

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def observe_original_return(observed_code, _offset, result):
        frame = sys._getframe(1)
        if observed_code is not code or frame.f_locals.get("self") is not service:
            return
        assert result == ()
        connection = database._local.conn
        with storage._lock:
            lease = participant.connections[connection]
            assert lease in storage._live_leases
            assert lease.resource_thread is threading.current_thread()
        assert not closed(connection)
        captured.update(connection=connection, lease=lease)
        if action == "query_error":
            raise sqlite3.OperationalError("injected attention result failure")

    def run_original():
        previous = None
        if borrowed:
            previous = database.get_connection()
            previous.execute("BEGIN")
        try:
            result = runtime.recompute_console_attention()
            assert result is False
            connection = captured["connection"]
            with storage._lock:
                after = (
                    closed(connection),
                    connection in participant.connections,
                    captured["lease"] in storage._live_leases,
                    database._connection_quiescence.is_registered(connection),
                )
            if borrowed:
                assert connection is previous
                assert previous.in_transaction
                assert previous.execute("SELECT 1").fetchone()[0] == 1
            return after
        finally:
            if previous is not None and not closed(previous):
                previous.rollback()
            database.close_connection()
            if captured:
                assert closed(captured["connection"])

    monitoring = sys.monitoring
    tool = next(value for value in range(1, 6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(tool, "original-attention-native-lifetime")
    try:
        monitoring.register_callback(
            tool, monitoring.events.PY_RETURN, observe_original_return
        )
        monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
        with ThreadPoolExecutor(max_workers=1) as executor:
            after_product = executor.submit(run_original).result(timeout=30)
    finally:
        monitoring.set_local_events(tool, code, 0)
        monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
        monitoring.free_tool_id(tool)
        await runtime.dispose()
        database.close_connection()
    assert ConversationLocalMarksService.list_console_unseen_marks is original
    assert original.__code__ is code
    assert captured
    with storage._lock:
        assert not participant.connections
        assert captured["lease"] not in storage._live_leases
    expected = (False, True, True, True) if borrowed else (True, False, False, False)
    assert after_product == expected, (
        "original attention callback retained its newly acquired native handle",
        after_product,
    )
