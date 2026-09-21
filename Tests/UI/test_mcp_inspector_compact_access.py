"""The real MCP inspector keeps its test form reachable in compact terminals."""

import json

import pytest
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Static, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_mcp_compact_readability import _compact, _paint, _settle
from Tests.UI.test_mcp_workbench import ToolTestHubService
from Tests.UI.test_tool_profile_review_lifetime import _wait
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


def _visible(app, control):
    assert control in app.screen._compositor.visible_widgets, (
        app.size,
        control.region,
        [
            (p.id, p.region, p.scroll_y, p.max_scroll_y)
            for p in control.ancestors
            if isinstance(p, Widget)
        ],
    )
    region, clip = app.screen._compositor.visible_widgets[control]
    assert region.width > 0 and region.height > 0
    assert region.intersection(clip) == region, str(
        (
            control.id,
            region,
            clip,
            [
                (
                    type(parent).__name__,
                    parent.id,
                    parent.region,
                    str(parent.styles.height),
                    parent.styles.overflow_y,
                    parent.scroll_y,
                    parent.max_scroll_y,
                    parent.virtual_size,
                )
                for parent in control.ancestors
                if isinstance(parent, Widget)
            ],
        )
    )
    assert app.screen.get_widget_at(*region.center)[0] is control
    if isinstance(control, Button):
        assert _compact(str(control.label)) in _compact(_paint(app.screen, region))


async def _tab_to(pilot, target, *, reverse=False):
    if pilot.app.focused is target:
        return
    visited = []
    for _ in range(30):
        await pilot.press("shift+tab" if reverse else "tab")
        await _settle(pilot)
        visited.append(str(pilot.app.focused))
        if pilot.app.focused is target:
            return
    pytest.fail(
        f"Keyboard traversal never reached {target.id}; mounted={target.is_mounted}; visits={visited}"
    )


@pytest.mark.parametrize("tool_name", ["search", "fetch"])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_test_form_controls_remain_reachable_through_resize(
    request, theme, tool_name
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        await _wait(pilot, lambda: getattr(app, "_initial_screen_pushed", False))
        service = ToolTestHubService()
        service.gate_state = "ask"
        app.unified_mcp_service = service
        await app.handle_screen_navigation(NavigateToScreen("mcp"))
        await _wait(pilot, lambda: getattr(app.screen, "screen_name", None) == "mcp")
        workbench = app.screen.workbench
        await _wait(
            pilot,
            lambda: (
                not workbench.is_loading
                and not workbench._reloading
                and bool(workbench.query("#mcp-tools-table"))
            ),
        )
        workbench.set_mode("tools")
        await _settle(pilot)
        table = workbench.query_one("#mcp-tools-table", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index(f"local:docs::{tool_name}"))
        await pilot.press("enter")
        await _settle(pilot)
        inspector = app.screen.query_one(MCPInspector)
        assert inspector._current_tool.name == tool_name
        button = inspector.query_one("#mcp-inspector-test-tool", Button)
        await _tab_to(pilot, button)
        _visible(app, button)
        await pilot.press("enter")
        await _wait(pilot, lambda: bool(inspector.query("#mcp-inspector-test-form")))
        raw = tool_name == "fetch"
        field = (
            inspector.query_one("#mcp-schema-raw", TextArea)
            if raw
            else inspector.query_one("#mcp-inspector-test-form Input", Input)
        )
        assert app.focused is field
        _visible(app, field)
        arguments = {"query": "retained query"}
        if raw:
            await pilot.press("home", "shift+end", "backspace")
        await pilot.press(*(json.dumps(arguments) if raw else arguments["query"]))
        for size in [(170, 48), (100, 30), (80, 24)]:
            await pilot.resize_terminal(*size)
            await _settle(pilot)
            assert app.focused is field
            assert (
                json.loads(field.text) if raw else {"query": field.value}
            ) == arguments
            _visible(app, field)
            for control_id in ("run", "close"):
                control = inspector.query_one(
                    f"#mcp-inspector-test-{control_id}", Button
                )
                await _tab_to(pilot, control)
                _visible(app, control)
            await pilot.press("shift+tab")
            await _settle(pilot)
            assert app.focused is inspector.query_one("#mcp-inspector-test-run")
            await _tab_to(pilot, field, reverse=True)
            _visible(app, field)
        run = inspector.query_one("#mcp-inspector-test-run", Button)
        await _tab_to(pilot, run)
        assert not run.disabled
        _visible(app, run)
        preview = inspector.query_one("#mcp-inspector-test-preview", Static)
        await pilot.press("up")
        await _settle(pilot)
        _visible(app, preview)
        _visible(app, run)
        assert _compact(
            "Approves this one invocation only. The approval does not persist."
        ) in _compact(_paint(app.screen, preview.region))
        await pilot.press("enter")
        await _wait(pilot, lambda: bool(service.test_calls))
        assert service.test_calls == [("local:docs", tool_name, arguments)]
        assert service.decision_calls == ["approved"]
        close = inspector.query_one("#mcp-inspector-test-close", Button)
        await _tab_to(pilot, close)
        _visible(app, close)
        await pilot.press("enter")
        await _wait(pilot, lambda: not inspector.query("#mcp-inspector-test-panel"))
        assert inspector._current_tool.name == tool_name
