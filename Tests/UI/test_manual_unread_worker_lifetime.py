"""Actual unread-row producer must own only its new worker SQL cache."""

import asyncio
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.parametrize("borrowed", [False, True])
@private_profile_test
def test_manual_unread_rows_retire_only_their_new_worker_connection(
    request, tmp_path, borrowed
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.conversation_local_marks_service import (
        ConversationLocalMarksService,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
    from Tests.UI._hidden_sql_worker_calls import OriginalNotesWorkerCalls

    database = CharactersRAGDB(tmp_path / "unread.sqlite", client_id="worker-lifetime")
    marks = ConversationLocalMarksService(database)
    conversation_id = database.add_conversation({"title": "Unread lifetime"})
    marks.mark_unread(conversation_id)
    state = {
        "key": (marks, marks.manual_revision, ("profile", "source")),
        "values": {},
        "pending": {conversation_id},
    }
    renders = []
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(
            conversation_local_marks_service=marks, chachanotes_db=database
        ),
        _manual_unread_cache=state,
        _console_switcher_authority=lambda: ("profile", "source"),
        _sync_console_workspace_context=lambda: renders.append(True),
    )
    participant = database._maintenance_participant
    connections = []
    actors = []
    observation = OriginalNotesWorkerCalls(database, storage)
    observation.start()

    def existing_worker_connection():
        connection = database.get_connection()
        connections.append(connection)
        actors.append(threading.current_thread())

    def inspect_worker_after_call():
        actor = threading.current_thread()
        actors.append(actor)
        with storage._lock:
            held = [
                (connection, lease)
                for connection, lease in participant.connections.items()
                if lease.resource_thread is actor
            ]
        connections.extend(connection for connection, _ in held)
        # Inspect the actual retained native objects on their owning actor,
        # before the test's explicit cleanup. Windows' monotonic clock can
        # give adjacent callbacks the same timestamp.
        with observation.lock:
            selected = {
                row["connection_object_id"]
                for row in observation.rows.values()
                if row["thread_object_id"] == id(actor)
            }
            native = [
                ref
                for ref in observation.references
                if isinstance(ref, sqlite3.Connection) and id(ref) in selected
            ]
        closed = []
        for connection in native:
            if connection not in connections:
                connections.append(connection)
            try:
                sqlite3.Connection.in_transaction.__get__(connection)
                closed.append(False)
            except sqlite3.ProgrammingError as error:
                assert "closed" in str(error).lower()
                closed.append(True)
        return held, closed, time.monotonic()

    async def exercise():
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unread-owner")
        loop.set_default_executor(executor)
        if borrowed:
            await asyncio.to_thread(existing_worker_connection)
        try:
            await ConsoleWorkspaceController._load_manual_unread_rows(
                screen, marks, state, (conversation_id,)
            )
            assert state["values"] == {conversation_id: True}
            assert not state["pending"] and renders == [True]
            return await asyncio.to_thread(inspect_worker_after_call)
        finally:
            # This test owns this dedicated actor and every recorded connection.
            # Retain and physically inspect before its original explicit close.
            await asyncio.to_thread(database.close_connection)

    try:
        # private_profile_test calls synchronous bodies from its running loop.
        # Give this test a distinct owned loop and single worker instead of
        # changing pytest's executor or nesting asyncio.run on its loop.
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="unread-loop-owner"
        ) as owned_loop:
            held, physically_closed, inspected_at = owned_loop.submit(
                asyncio.run, exercise()
            ).result()
        observation.stop()
        receipt = observation.receipt()
        assert receipt["overflow"] == receipt["live_original_frames"] == 0
        assert receipt["original_bindings_and_codes_unchanged"]
        worker_rows = [row for row in receipt["rows"] if not row["is_initial_thread"]]
        assert len(worker_rows) == 1 and worker_rows[0]["registration_returns"] == 1
        if borrowed:
            assert len(held) == 1 and held[0][0] is connections[0]
            assert physically_closed and not any(physically_closed)
        else:
            assert (
                not held
            ), "unread-row producer retained its newly opened worker SQL lease"
            assert physically_closed and all(physically_closed)
            assert any(
                event["closed_descriptor_observed"]
                and event["returned_at"] <= inspected_at
                for event in worker_rows[0]["close_events"]
            ), "producer did not physically close before the test's explicit cleanup"
        assert actors and all(actor is actors[0] for actor in actors)
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError):
                sqlite3.Connection.in_transaction.__get__(connection)
            assert connection not in participant.connections
    finally:
        if observation.active:
            observation.stop()
        database.close_connection()
