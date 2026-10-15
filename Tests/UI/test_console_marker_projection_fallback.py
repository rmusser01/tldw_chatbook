"""Custom and memory marker projections retain the original caller route."""

import sys
import threading
from types import MethodType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("route", ["memory", "body", "defaults", "custom"])
async def test_marker_projection_declines_before_capture_and_preserves_live_abi(
    tmp_path, monkeypatch, route
):
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_change_review_read import capture_marker_read
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.UI.Console_Modules.transcript import (
        ConsoleChangeReviewProjection,
    )
    from tldw_chatbook.Workspaces.change_review_finalization import (
        ChangeReviewPublicationSignal,
    )

    database = AgentRunsDB(
        ":memory:" if route == "memory" else tmp_path / "markers.sqlite"
    )
    database.create_run(
        run_id="run",
        conversation_id="chat",
        agent_kind="primary",
        assistant_message_id="assistant",
    )
    database.record_change_snapshot(
        run_id="run",
        root="/project",
        baseline_sha="before",
        end_sha="after",
        files_changed=1,
        adds=2,
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=database,
        store=None,
        provider_gateway=None,
        registry=ToolCatalogRegistry(),
    )
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime._agent_bridge = bridge
    coordinator = runtime._change_review_coordinator = SimpleNamespace(
        publication_signal=ChangeReviewPublicationSignal()
    )
    projection = ConsoleChangeReviewProjection(
        runtime_accessor=lambda: runtime, conversation_id_accessor=lambda: "chat"
    )
    messages = [
        ConsoleChatMessage(
            id="assistant",
            persisted_message_id="assistant",
            role=ConsoleMessageRole.ASSISTANT,
            content="reply",
        )
    ]
    reader = AgentRunsDB.list_change_review_run_anchors
    if route != "memory":
        assert capture_marker_read(runtime, coordinator, bridge, "chat") is not None
    custom_calls = []
    if route == "body":
        monkeypatch.setattr(reader, "__code__", reader.__code__.replace())
    elif route == "defaults":
        monkeypatch.setattr(reader, "__defaults__", (None,))
    elif route == "custom":
        original = bridge.change_review_marker_messages

        def custom(self, conversation_id):
            custom_calls.append((self, conversation_id, threading.current_thread()))
            return original(conversation_id)

        monkeypatch.setattr(
            bridge, "change_review_marker_messages", MethodType(custom, bridge)
        )
    assert capture_marker_read(runtime, coordinator, bridge, "chat") is None
    observed = []
    code = reader.__code__

    def started(observed_code, _offset):
        frame = sys._getframe(1)
        if observed_code is code and frame.f_locals.get("self") is database:
            observed.append(threading.current_thread())

    monitoring = sys.monitoring
    tool = next(value for value in range(1, 6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(tool, "marker-fallback")
    try:
        monitoring.register_callback(tool, monitoring.events.PY_START, started)
        monitoring.set_local_events(tool, code, monitoring.events.PY_START)
        assert await projection.prepare()
        assert observed == [] and not projection._preparation_reads
        result = projection.project(messages)
        assert observed == [threading.current_thread()]
        assert [row.change_review_run_id for row in result[1:]] == ["run"]
        assert custom_calls == (
            [(bridge, "chat", threading.current_thread())] if route == "custom" else []
        )
    finally:
        monitoring.set_local_events(tool, code, 0)
        monitoring.register_callback(tool, monitoring.events.PY_START, None)
        monitoring.free_tool_id(tool)
        database.close()
