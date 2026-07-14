"""Agent rail section + [N Sub-Agents] badge render via a real App (run_test)."""
from __future__ import annotations

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow, build_console_conversation_browser_state,
)


def _all_rows(state):
    rows = []
    for section in state.sections:
        rows.extend(section.rows)
        for group in section.groups:
            rows.extend(group.rows)
    return rows


def test_conversation_row_carries_badge_count_for_render():
    row = ConsoleConversationBrowserInputRow(
        row_key="c1", conversation_id="c1", native_session_id=None, title="Alpha",
        scope_type="global", workspace_id=None, workspace_label="",
        updated_sort="2026-07-13T00:00:00Z")
    state = build_console_conversation_browser_state(
        rows=[row], active_workspace_id=None, subagent_counts={"c1": 2})
    assert _all_rows(state)[0].subagent_count == 2


def test_conversation_row_badge_label_is_escaped_and_present():
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        format_console_conversation_row_label,
    )
    label = format_console_conversation_row_label("Beta [x]", subagent_count=3)
    assert "3 Sub-Agents" in label
    assert "\\[x]" in label or "[x]" not in label.replace("Sub-Agents", "")


def test_no_badge_when_subagent_count_is_zero():
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        format_console_conversation_row_label,
    )
    label = format_console_conversation_row_label("Beta", subagent_count=0)
    assert "Sub-Agents" not in label
    assert label == "Beta"


@pytest.mark.asyncio
async def test_agent_rail_section_mounts():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        # Navigating to Console mounts the rail; the Agent header exists.
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        assert console.query_one("#console-rail-section-header-agent")
        assert console.query_one("#console-agent-section-status")
        assert console.query_one("#console-agent-section-steps")
        assert console.query_one("#console-agent-section-subagents")


def test_resume_rederives_subagent_data_from_durable_run_store(tmp_path):
    """A fresh bridge over the same durable AgentRunsDB file reproduces the
    same badge count + sub-agent listing after a "restart" -- the run store,
    not any in-memory/session state, is the source of truth for resume."""
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db_path = tmp_path / "agent_runs.db"
    db = AgentRunsDB(db_path, client_id="t")
    primary_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    sub_id = db.create_run(
        conversation_id="conv-1", agent_kind="subagent",
        task="research pricing", parent_run_id=primary_id,
    )
    db.set_status(sub_id, "done", result="done researching")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    assert bridge.subagent_count("conv-1") == 1

    # Simulate resume: a brand-new bridge/DB handle over the same file, with
    # no in-memory live-snapshot state carried over.
    fresh_db = AgentRunsDB(db_path, client_id="t")
    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=fresh_db, store=None, provider_gateway=None)

    assert fresh_bridge.subagent_count("conv-1") == 1
    runs = fresh_bridge.subagent_runs("conv-1")
    assert len(runs) == 1
    assert runs[0]["id"] == sub_id
    assert runs[0]["status"] == "done"
    record = fresh_bridge.subagent_run(sub_id)
    assert record is not None
    assert record["task"] == "research pricing"
    # No live-run activity was replayed -- the rail's "running" state does
    # not leak across a resume.
    assert fresh_bridge.live_snapshot("conv-1").status == "idle"
