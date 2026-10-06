"""Finite chat-request attribution and source reads retain only real borrowers."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("reader", ["attribution", "source"])
@pytest.mark.parametrize("borrowed", [False, True], ids=["cold", "warm-borrower"])
@pytest.mark.parametrize("fail_sql", [False, True], ids=["success", "sql-error"])
def test_chat_create_read_retires_only_its_worker_connection(
    tmp_path, monkeypatch, reader, borrowed, fail_sql
):
    """Read real parent/child rows without stranding a completed worker handle."""
    db = AgentRunsDB(tmp_path / "runs.sqlite", client_id="chat-create-owner")
    store = ConsoleChatStore()
    session = store.ensure_session()
    session.persisted_conversation_id = "conversation"
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=None)
    controller = ConsoleChatController(
        store=store, provider_gateway=None, agent_bridge=bridge
    )
    try:
        parent = db.create_run(conversation_id="conversation", agent_kind="primary")
        child = db.create_run(
            conversation_id="conversation", agent_kind="subagent", parent_run_id=parent
        )
        db.close()
        controller._chat_create_session_grants[session.id] = {"fork_chat"}
        observation = controller._capture_chat_creation_source(
            {
                "session_id": session.id,
                "source_run_id": child,
                "source_parent_run_id": parent,
                "source_agent_kind": "subagent",
            }
        )
        original = db.get_run
        handles = []

        def read_row(run_id):
            row = original(run_id)
            connection = db._thread_local.conn
            handles.append(connection)
            if fail_sql:
                connection.execute("SELECT * FROM missing_chat_request_table")
            return row

        monkeypatch.setattr(db, "get_run", read_row)

        def read():
            previous = db._held_connection() if borrowed else None
            if previous is not None:
                previous.execute("BEGIN")
            try:
                with use_run_id(parent if reader == "attribution" else child):
                    if reader == "attribution":
                        result = controller.request_chat_create_confirm(
                            {"tool": "fork_chat"}, session_id=session.id
                        )
                        assert result == {
                            "allow": not fail_sql,
                            "remember": not fail_sql,
                        }
                    else:
                        result = controller._read_chat_creation_source(observation)
                        if fail_sql:
                            assert result is None
                        else:
                            assert result.row["parent_run_id"] == parent
                            assert result.row["agent_kind"] == "subagent"
                            assert result.parent["agent_kind"] == "primary"
                            assert result.runs_db is db
                assert len(handles) == (2 if reader == "source" and not fail_sql else 1)
                if borrowed:
                    assert all(connection is previous for connection in handles)
                    assert previous.in_transaction
                    assert previous.execute("SELECT 1").fetchone()[0] == 1
                else:
                    for connection in handles:
                        with pytest.raises(
                            sqlite3.ProgrammingError, match="closed database"
                        ):
                            connection.execute("SELECT 1")
            finally:
                db.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(read).result(timeout=5)
    finally:
        controller.begin_shutdown()
        db.close()
