"""Change Review markers read anchors and snapshots, not full agent histories."""

import sqlite3
from types import SimpleNamespace

import pytest

from Tests.agent_runs_owner import owned_agent_runs_db
from Tests.private_profile import private_profile_test


@pytest.mark.parametrize("storage_kind", ["memory", "file"])
@private_profile_test
def test_marker_projection_does_not_read_unrelated_step_payloads(
    request, storage_kind, tmp_path
):
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    database = owned_agent_runs_db(
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("callback_name", "mutation"),
    [
        (name, mutation)
        for name in (
            "change_review_marker_messages",
            "_change_review_marker_block",
            "_read_change_review_markers",
            "database.connection",
            "database.transaction",
            "database._execution_use",
            "database._require_execution_owner",
            "owner.require_current",
            "owner._use",
            "owner.close",
            "owner_global._identity",
            "owner_global._store_identity",
            "factory._stock_execution_owner",
        )
        for mutation in ("code", "defaults", "kwdefaults")
    ]
    + [("_read_change_review_markers", "kwitems")]
    + [
        (name, mutation)
        for name in ("database.connection", "database.transaction", "owner._use")
        for mutation in ("wrapped", "closure")
    ]
    + [
        (name, "instance")
        for name in ("owner._use", "owner.require_current", "owner.close")
    ]
    + [
        ("database", "owner_reference"),
        ("database", "read_only"),
        ("database", "custom_owner"),
        ("database", "db_path_descriptor"),
        ("owner", "owner_path_descriptor"),
    ],
)
@private_profile_test
async def test_marker_capture_retains_each_callback_metadata_pin(
    request, tmp_path, monkeypatch, callback_name, mutation
):
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module
    from tldw_chatbook.Chat.console_change_review_read import (
        MarkerReadObsolete,
        capture_marker_read,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB import AgentRuns_DB as database_module
    from tldw_chatbook.DB.console_execution_owner import (
        ConsoleExecutionOwner,
        ConsoleExecutionUnavailable,
    )

    database = owned_agent_runs_db(tmp_path / "marker-capture.sqlite")
    runtime = replacement_owner = None
    try:
        bridge = bridge_module.ConsoleAgentBridge(
            agent_runs_db=database,
            store=None,
            provider_gateway=None,
            registry=ToolCatalogRegistry(),
        )
        runtime = ConsoleRuntime(SimpleNamespace())
        runtime._agent_bridge = bridge
        coordinator = runtime._change_review_coordinator = object()
        retained = capture_marker_read(runtime, coordinator, bridge, "chat")
        assert retained is not None
        retained.require_current()
        owner = database._execution_owner
        with monkeypatch.context() as changed:
            if mutation == "owner_reference":
                database.close()
                assert owner._users == 0
                owner.close()
                replacement_owner = ConsoleExecutionOwner.acquire(database.db_path)
                changed.setattr(database, "_execution_owner", replacement_owner)
            elif mutation == "read_only":
                changed.setattr(database, "_read_only", True)
            elif mutation == "custom_owner":
                changed.setattr(database, "_execution_owner", object())
            elif mutation in {"db_path_descriptor", "owner_path_descriptor"}:

                def refuse_lookup(_self):
                    raise AssertionError("changed data descriptor executed")

                changed.setattr(
                    database_module.AgentRunsDB
                    if mutation == "db_path_descriptor"
                    else ConsoleExecutionOwner,
                    "db_path" if mutation == "db_path_descriptor" else "_path",
                    property(refuse_lookup),
                    raising=False,
                )
            elif mutation == "instance":
                name = callback_name.removeprefix("owner.")
                changed.setitem(vars(owner), name, getattr(owner, name))
            else:
                if callback_name.startswith("database."):
                    callback = getattr(
                        database_module.AgentRunsDB, callback_name.split(".")[1]
                    )
                elif callback_name.startswith("owner."):
                    callback = getattr(
                        ConsoleExecutionOwner, callback_name.split(".")[1]
                    )
                elif callback_name.startswith("owner_global."):
                    callback = ConsoleExecutionOwner.require_current.__globals__[
                        callback_name.split(".")[1]
                    ]
                elif callback_name.startswith("factory."):
                    callback = getattr(database_module, callback_name.split(".")[1])
                elif callback_name == "_read_change_review_markers":
                    callback = bridge_module._read_change_review_markers
                else:
                    callback = getattr(bridge_module.ConsoleAgentBridge, callback_name)
                if mutation == "code":
                    changed.setattr(callback, "__code__", callback.__code__.replace())
                elif mutation == "defaults":
                    changed.setattr(callback, "__defaults__", (object(),))
                elif mutation == "kwdefaults":
                    changed.setattr(
                        callback,
                        "__kwdefaults__",
                        {
                            **(callback.__kwdefaults__ or {}),
                            "capture_control": object(),
                        },
                    )
                elif mutation == "wrapped":
                    changed.setattr(callback, "__wrapped__", object())
                elif mutation == "closure":
                    cell = next(
                        cell
                        for cell in callback.__closure__
                        if cell.cell_contents is callback.__wrapped__
                    )
                    changed.setattr(cell, "cell_contents", object())
                else:
                    assert callback.__kwdefaults__ == {"require_current": None}
                    changed.setitem(
                        callback.__kwdefaults__, "require_current", object()
                    )
            fresh = capture_marker_read(runtime, coordinator, bridge, "chat")
            if mutation == "owner_reference":
                assert fresh is not None
                fresh.require_current()
            else:
                assert fresh is None
            with pytest.raises(MarkerReadObsolete, match="marker_source_changed"):
                retained.require_current()
        assert capture_marker_read(runtime, coordinator, bridge, "chat") is not None
        retained.require_current()
        if mutation == "owner_reference":
            import sys
            from contextlib import ExitStack

            observed = []
            code = database_module.AgentRunsDB._held_connection.__code__
            monitoring = sys.monitoring
            tool = next(i for i in range(1, 6) if monitoring.get_tool(i) is None)

            def started(observed_code, _offset):
                frame = sys._getframe(1)
                if observed_code is code and frame.f_locals.get("self") is database:
                    observed.append(True)

            monitoring.use_tool_id(tool, "expired-marker-owner")
            with ExitStack() as cleanup:
                cleanup.callback(monitoring.free_tool_id, tool)
                monitoring.register_callback(tool, monitoring.events.PY_START, started)
                cleanup.callback(
                    monitoring.register_callback, tool, monitoring.events.PY_START, None
                )
                monitoring.set_local_events(tool, code, monitoring.events.PY_START)
                cleanup.callback(monitoring.set_local_events, tool, code, 0)
                with pytest.raises(ConsoleExecutionUnavailable):
                    retained.read()
                assert observed == []
    finally:
        try:
            if runtime is not None:
                # The borrowed identity token owns no coordinator work to retire.
                runtime._change_review_coordinator = None
                await runtime.dispose()
        finally:
            try:
                database.close()
            finally:
                if replacement_owner is not None:
                    replacement_owner.close()
