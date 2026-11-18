"""Console's newest-primary consumers do not hydrate unrelated saved primaries."""

import json

import pytest

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


def test_newest_primary_read_preserves_mixed_history_without_old_step_hydration():
    db = AgentRunsDB(":memory:", client_id="bounded-history")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    latest = "tied-primary-z"
    legacy = [{"kind": "model", "summary": f"step-{index}"} for index in range(4)]
    appended = [{"kind": "model", "summary": f"step-{index}"} for index in range(4, 8)]
    runs = [
        (f"old-primary-{index:03}", "primary", "done", None, "2000-01-01", json.dumps(legacy))
        for index in range(128)
    ]
    runs.extend([
        ("tied-primary-a", "primary", "done", None, "2001-01-01", json.dumps(legacy)),
        (latest, "primary", "stopped", None, "2001-01-01", json.dumps(legacy)),
        ("newer-superseded", "primary", "superseded", None, "2002-01-01", "[]"),
        ("current-child", "subagent", "done", latest, "2003-01-01", json.dumps(legacy)),
        ("old-child", "subagent", "done", "old-primary-000", "2004-01-01", "[]"),
        # A newer unrelated kind must not consume the primary limit or hydrate.
        ("unrelated-local-command", "local_command", "done", None, "2005-01-01", "not-json"),
    ])
    try:
        with db.connection() as conn:
            conn.executemany(
                "INSERT INTO agent_runs "
                "(id, conversation_id, agent_kind, status, parent_run_id, created_at, updated_at, steps) "
                "VALUES (?, 'conversation', ?, ?, ?, ?, '2005-01-01', ?)",
                runs,
            )
            conn.executemany(
                "INSERT INTO agent_run_steps (run_id, seq, payload, created_at) "
                "VALUES (?, ?, ?, '2005-01-01')",
                [
                    (run_id, index, json.dumps(step))
                    for run_id, kind, _status, _parent, _created, _legacy in runs
                    if kind == "primary"
                    for index, step in enumerate(appended)
                ],
            )
            statements = []
            conn.set_trace_callback(statements.append)
            try:
                assert bridge._previous_primary_run_id("conversation") == latest
                snapshot = bridge.historical_snapshot("conversation")
            finally:
                conn.set_trace_callback(None)

        assert snapshot.status == "stopped"
        assert snapshot.step == 8
        assert [step.text for step in snapshot.steps] == [f"step-{index}" for index in range(3, 8)]
        assert [child.run_id for child in snapshot.subagents] == ["current-child"]
        assert snapshot.subagents[0].detail == "step-3"
        primary_queries = [
            query for query in statements
            if "FROM agent_runs WHERE" in query and "agent_kind = 'primary'" in query
        ]
        assert len(primary_queries) == 2
        assert all("ORDER BY created_at DESC, id DESC LIMIT 1" in query for query in primary_queries)
        step_queries = [query for query in statements if "FROM agent_run_steps" in query]
        assert step_queries, "the selected run must retain its original full step hydration"
        assert any(latest in query for query in step_queries)
        assert not any(
            forbidden in query
            for query in step_queries
            for forbidden in ("old-primary-", "tied-primary-a", "newer-superseded", "unrelated-local-command")
        )
    finally:
        db.close()


def test_custom_history_database_keeps_original_list_contract():
    class LegacyDatabase:
        def __init__(self):
            self.calls = []

        def list_runs(self, conversation_id, *, include_superseded, agent_kind):
            self.calls.append((conversation_id, include_superseded, agent_kind))
            return [{"id": "legacy-primary", "status": "done", "steps": []}] if agent_kind == "primary" else []

    database = LegacyDatabase()
    # Exercise these two readers' legacy DB contract without constructing
    # unrelated service owners around a deliberately minimal custom reader.
    bridge = object.__new__(ConsoleAgentBridge)
    bridge._db = database
    assert bridge._previous_primary_run_id("conversation") == "legacy-primary"
    assert bridge._derive_historical_snapshot("conversation").status == "done"
    assert database.calls == [
        ("conversation", False, "primary"),
        ("conversation", False, "primary"),
        ("conversation", False, "subagent"),
    ]
