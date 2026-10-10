"""Change Review markers read anchors and snapshots, not full agent histories."""

import sqlite3
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.parametrize("storage_kind", ["memory", "file"])
@private_profile_test
def test_marker_projection_does_not_read_unrelated_step_payloads(
    request, storage_kind, tmp_path
):
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    database = AgentRunsDB(
        ":memory:" if storage_kind == "memory" else tmp_path / "marker-runs.sqlite"
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=database,
        store=None,
        provider_gateway=None,
        registry=ToolCatalogRegistry(),
    )
    try:
        for run_id, kind, conversation, anchor in (
            ("a", "primary", "chat", "assistant-a"),
            ("b", "primary", "chat", None),
            ("c", "primary", "chat", "assistant-c"),
            ("child", "subagent", "chat", "child-anchor"),
            ("superseded", "primary", "chat", "old-anchor"),
            ("foreign", "primary", "other-chat", "foreign-anchor"),
        ):
            database.create_run(
                run_id=run_id,
                conversation_id=conversation,
                agent_kind=kind,
                assistant_message_id=anchor,
                budget={"max_steps": 500},
            )
            database.append_steps(
                run_id,
                [{"index": 0, "kind": "model", "summary": "unused" * 1000}],
            )
            with database.transaction() as connection:
                connection.execute(
                    "UPDATE agent_runs SET created_at = ?, steps = ? WHERE id = ?",
                    ("2026-10-07T00:00:00Z", '[{"legacy": "unused payload"}]', run_id),
                )
            database.record_change_snapshot(
                run_id=run_id,
                root="/example/project",
                baseline_sha="before",
                end_sha="after",
                files_changed=1,
                adds=2,
            )
        database.set_status("superseded", "superseded")
        reads = []

        def observe_read(action, table, column, _database, _trigger):
            if action == sqlite3.SQLITE_READ:
                reads.append((table, column))
            return sqlite3.SQLITE_OK

        connection = database._held_connection()
        connection.set_authorizer(observe_read)
        try:
            projected = bridge.change_review_marker_messages("chat")
        finally:
            connection.set_authorizer(None)
        assert [anchor for anchor, _ in projected] == [
            "assistant-a",
            None,
            "assistant-c",
        ]
        assert [
            message.change_review_run_id
            for _anchor, block in projected
            for message in block
        ] == ["a", "b", "c"]
        assert (
            len({message.content for _, block in projected for message in block}) == 1
        )
        assert ("agent_runs", "assistant_message_id") in reads
        assert any(table == "change_snapshots" for table, _ in reads)
        assert not [
            (table, column)
            for table, column in reads
            if table == "agent_run_steps"
            or (
                table == "agent_runs"
                and column in {"steps", "budget", "task", "result"}
            )
        ], "marker projection fetched unrelated run-history payloads"
    finally:
        database.close()


@pytest.mark.parametrize("adapter_kind", ["legacy", "subclass", "instance", "class"])
def test_marker_projection_keeps_legacy_database_adapter_contract(
    adapter_kind, monkeypatch
):
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    calls = []

    def list_runs(conversation_id, *, include_superseded):
        calls.append((conversation_id, include_superseded))
        return [
            {"id": "new", "agent_kind": "primary", "assistant_message_id": "new-a"},
            {
                "id": "child",
                "agent_kind": "subagent",
                "assistant_message_id": "child-a",
            },
            {"id": "old", "agent_kind": "primary", "assistant_message_id": "old-a"},
        ]

    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    class CustomRuns(AgentRunsDB):
        def list_runs(self, conversation_id, *, include_superseded):
            return list_runs(conversation_id, include_superseded=include_superseded)

    if adapter_kind == "legacy":
        database = SimpleNamespace(list_runs=list_runs)
    elif adapter_kind == "subclass":
        database = CustomRuns(":memory:")
    else:
        database = AgentRunsDB(":memory:")
        if adapter_kind == "instance":
            database.list_runs = list_runs
        else:
            monkeypatch.setattr(
                AgentRunsDB,
                "list_runs",
                lambda self, conversation_id, *, include_superseded: list_runs(
                    conversation_id, include_superseded=include_superseded
                ),
            )
    database.change_snapshots_for_conversation = lambda _conversation: []
    try:
        bridge = ConsoleAgentBridge(
            agent_runs_db=database,
            store=None,
            provider_gateway=None,
            registry=ToolCatalogRegistry(),
        )
        assert bridge.change_review_marker_messages("chat") == [
            ("old-a", []),
            ("new-a", []),
        ]
        assert calls == [("chat", False)]
    finally:
        if isinstance(database, AgentRunsDB):
            database.close()
