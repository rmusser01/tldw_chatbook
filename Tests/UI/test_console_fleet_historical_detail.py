"""Historical fleet detail and status colors on the real painted Console."""

import pytest

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_console_fleet_budget_history import _bridge, _painted_text
from Tests.UI.test_console_fleet_panel import _scroll_into_view, _setup_console
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Console_Modules.agent import _fleet_row_from_record
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
)


class _StyledConsoleHarness(ConsoleHarness):
    CSS_PATH = BUNDLED_STYLESHEET


@pytest.mark.parametrize("size", [(180, 48), (120, 35)])
@pytest.mark.asyncio
async def test_historical_child_detail_elapsed_and_status_color_are_painted(
    tmp_path, size
):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    try:
        parent = db.create_run(conversation_id="conv-A", agent_kind="primary")
        db.append_steps(
            parent, [{"kind": "model", "summary": "unrelated parent output"}]
        )
        db.set_status(parent, "done")
        ids = []
        for state, detail in [
            ("stuck", "Needs direction"),
            ("cancelled", "Stopped by user"),
        ]:
            child = db.create_run(
                conversation_id="conv-A",
                agent_kind="subagent",
                parent_run_id=parent,
                task="Check logs",
            )
            db.append_steps(child, [{"kind": "error", "summary": detail}])
            db.set_status(child, state, budget_tokens=40)
            with db.transaction() as conn:
                conn.execute(
                    "UPDATE agent_runs SET created_at=?, updated_at=? WHERE id=?",
                    ("2026-09-07T10:00:00+00:00", "2026-09-07T10:01:04+00:00", child),
                )
            ids.append(child)
        app = _build_test_app()
        _configure_native_ready_console(app)
        host = _StyledConsoleHarness(app)
        async with host.run_test(size=size) as pilot:
            console = await _setup_console(pilot, host, _bridge(db))
            section = console.query_one(
                "#console-agent-section-subagents", ConsoleInspectorSection
            )
            section.set_open(True)
            await pilot.pause()
            colors = {}
            for index, row in enumerate(section.rows):
                selector = f"#console-inspector-section-agent-fleet-row-{index}"
                await _scroll_into_view(pilot, console, selector)
                primary = console.query_one(selector + "-primary")
                secondary = console.query_one(selector + "-secondary")
                primary.scroll_visible(animate=False, top=True)
                await pilot.pause(0.2)
                assert "~1m 4s" in _painted_text(host, primary)
                expected = (
                    "Needs direction" if row.status == "stuck" else "Stopped by user"
                )
                assert expected in _painted_text(host, secondary)
                assert "40 budget tok" in str(secondary.renderable)
                assert "unrelated parent" not in str(secondary.renderable)
                colors[row.status] = primary.styles.color
            assert colors["stuck"] != colors["cancelled"]
            variables = host.get_css_variables()
            from textual.color import Color

            assert colors["stuck"] == Color.parse(variables["warning"])
            assert colors["cancelled"] == secondary.styles.color
    finally:
        db.close()


@pytest.mark.parametrize(
    "start,end,status,visible",
    [
        ("2026-09-07T10:00:00", "2026-09-07T10:00:12+00:00", "done", True),
        ("bad", "2026-09-07T10:00:12", "done", False),
        (None, None, "done", False),
        ("2026-09-07T10:00:12", "2026-09-07T10:00:00", "done", False),
        ("2026-09-07T10:00:00", "2026-09-07T10:00:12", "running", False),
    ],
)
def test_historical_fallback_uses_child_detail_and_only_valid_terminal_elapsed(
    start, end, status, visible
):
    row = _fleet_row_from_record(
        {
            "id": "child",
            "task": "Check",
            "status": status,
            "created_at": start,
            "updated_at": end,
            "steps": [{"summary": "saved child detail"}],
            "budget_tokens": 0,
        }
    )
    assert ("~12s" in row.primary_text) is visible
    assert "saved child detail" in row.secondary_text
    assert "0 budget tok" in row.secondary_text


def test_historical_detail_applies_the_existing_display_cap_once():
    step = {"summary": "a long child answer " * 500}
    summary = ConsoleAgentBridge.historical_subagent_summary({"steps": [step]})
    assert summary.detail == ConsoleAgentBridge._summarize_persisted_step(step)
