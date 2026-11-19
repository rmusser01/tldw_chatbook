"""Finite run-log metadata reads retire cold handles, not native borrowers."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.Agents import run_log
from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize(
    "reader", ["run_log_available", "load_run_log_page", "load_run_log_text"]
)
@pytest.mark.parametrize("borrowed", [False, True], ids=["cold", "warm-borrower"])
@pytest.mark.parametrize("fail_sql", [False, True], ids=["success", "sql-error"])
def test_run_log_read_retires_only_its_worker_metadata_connection(
    tmp_path, monkeypatch, reader, borrowed, fail_sql
):
    """A completed log reader must not strand its newly opened worker handle."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="log-owner")
    try:
        primary = db.create_run(conversation_id="conv", agent_kind="primary")
        child = db.create_run(
            conversation_id="conv", agent_kind="subagent", parent_run_id=primary
        )
        db.close()
        monkeypatch.setattr(run_log, "resolve_log_root", lambda: tmp_path)
        writer = RunLogWriter()
        writer.bind(primary)
        writer.append(
            run_id=child, kind="subagent", type="model", content="child result"
        )
        bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
        original = db.get_run_metadata
        handles = []

        def metadata(run_id):
            record = original(run_id)
            connection = db._thread_local.conn
            handles.append(connection)
            if fail_sql:
                connection.execute("SELECT * FROM missing_run_log_table")
            return record

        monkeypatch.setattr(db, "get_run_metadata", metadata)

        def read():
            previous = db._held_connection() if borrowed else None
            if previous is not None:
                previous.execute("BEGIN")
            try:
                if fail_sql:
                    with pytest.raises(sqlite3.OperationalError, match="no such table"):
                        getattr(bridge, reader)(child)
                else:
                    result = getattr(bridge, reader)(child)
                    if reader == "run_log_available":
                        assert result is True
                    elif reader == "load_run_log_page":
                        assert result.slices[0].record.content == "child result"
                        assert result.slices[0].record.run_id == child
                    else:
                        assert "child result" in result
                assert len(handles) == 1
                if borrowed:
                    assert handles[0] is previous
                    assert previous.in_transaction
                    assert previous.execute("SELECT 1").fetchone()[0] == 1
                else:
                    with pytest.raises(
                        sqlite3.ProgrammingError, match="closed database"
                    ):
                        handles[0].execute("SELECT 1")
            finally:
                # Even RED retires the captured test handle on its owning thread.
                db.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(read).result(timeout=5)
    finally:
        db.close()
