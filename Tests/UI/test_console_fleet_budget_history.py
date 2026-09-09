"""Durable budget accounting reaches painted fleet rows and continuation detail."""

import pytest

from Tests.UI.test_console_fleet_panel import (
    _AGENT_SECTION_SIZE,
    _scroll_into_view,
    _setup_console,
)
from Tests.UI.test_console_parallel_runs import _assert_painted_at_own_region
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Agents.agent_models import AgentConfig
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
)


def _painted_text(host, widget):
    _assert_painted_at_own_region(host, widget)
    region = widget.region
    return " ".join(
        " ".join(
            strip.crop(region.x, region.right).text
            for strip in host.screen._compositor.render_strips()[
                region.y : region.bottom
            ]
        ).split()
    )


def _seed(db, budget, *, parent, ancestor=None):
    run_id = db.create_run(
        conversation_id="conv-A",
        agent_kind="subagent",
        task="Review logs",
        parent_run_id=parent,
        resumed_from_run_id=ancestor,
    )
    db.set_status(run_id, "done", budget_tokens=budget)
    return run_id


def _bridge(db):
    return ConsoleAgentBridge(
        agent_runs_db=db, store=ConsoleChatStore(), provider_gateway=None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget, label",
    [(1234, "1.2k budget tok"), (0, "0 budget tok"), (None, "Budget unavailable")],
)
async def test_reopened_historical_row_paints_budget_without_refeeding_cost_chip(
    tmp_path, budget, label
):
    path = tmp_path / "runs.db"
    db = AgentRunsDB(path, client_id="test")
    parent = db.create_run(conversation_id="conv-A", agent_kind="primary")
    _seed(db, budget, parent=parent)
    db.set_status(parent, "done")
    db.close()
    reopened = AgentRunsDB(path, client_id="reopened")
    try:
        host = ConsoleHarness(_build_test_app())
        async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
            console = await _setup_console(pilot, host, _bridge(reopened))
            section = console.query_one(
                "#console-agent-section-subagents", ConsoleInspectorSection
            )
            section.set_open(True)
            await pilot.pause()
            selector = "#console-inspector-section-agent-fleet-row-0-secondary"
            await _scroll_into_view(pilot, console, selector)
            assert label in _painted_text(host, console.query_one(selector))
            assert console._agent._console_agent_fleet_token_total() == 0
    finally:
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ancestor_budget, total",
    [
        (100, "Chain: 140 budget tok across 2 runs"),
        (None, "Chain: 40 budget tok recorded (partial; 1/2 runs)"),
    ],
)
async def test_continuation_detail_paints_complete_or_partial_ancestry(
    tmp_path, ancestor_budget, total
):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    try:
        parent = db.create_run(conversation_id="conv-A", agent_kind="primary")
        ancestor = _seed(db, ancestor_budget, parent=parent)
        selected = _seed(db, 40, parent=parent, ancestor=ancestor)
        _seed(db, 900, parent=parent, ancestor=ancestor)  # Excluded sibling fork.
        db.set_status(parent, "done")
        bridge = _bridge(db)
        fleet = FleetCoordinator(3, lambda: 0.0)
        service = AgentService(
            db=db,
            registry=ToolCatalogRegistry(),
            fleet_coordinator=fleet,
            chat_call=lambda **kwargs: {"choices": [{"message": {"content": "done"}}]},
        )
        service.run_turn(
            conversation_id="conv-A",
            messages=[],
            config=AgentConfig(model="test", system_prompt="Be helpful."),
            api_endpoint="llama_cpp",
        )
        handle = fleet.reserve("Review logs", None)
        fleet.attach_run(handle.handle_id, selected)
        fleet.finish(handle.handle_id, "done", total_tokens=40)
        bridge._fleet_services["conv-A"] = service
        host = ConsoleHarness(_build_test_app())
        async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
            console = await _setup_console(pilot, host, bridge)
            assert console._agent._console_agent_fleet_token_total() == 40
            console._console_agent_drilldown_run_id = selected
            # A new supervisor turn prunes the old handle. Its selected
            # child's durable detail must remain readable across that boundary.
            assert fleet.prune_terminal() == 1
            db.create_run(conversation_id="conv-A", agent_kind="primary")
            console._sync_console_agent_section()
            await pilot.pause()
            selector = "#console-agent-section-steps"
            await _scroll_into_view(pilot, console, selector)
            painted = _painted_text(host, console.query_one(selector))
            assert "This run: 40 budget tok" in painted
            assert total in painted
            assert console._agent._console_agent_fleet_token_total() == 0
    finally:
        db.close()
