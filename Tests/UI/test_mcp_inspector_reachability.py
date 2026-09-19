"""Compact inspector controls must be reachable through real focus/scroll paths."""

from typing import ClassVar

import pytest
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_mcp_inspector import InspectorApp, _stale_snap, _tool
from tldw_chatbook.MCP.hub_test_execution import ToolTestAdmissionPreview
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.UI.MCP_Modules import mcp_inspector as inspector_module
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector


class ReachabilityApp(InspectorApp):
    CSS_PATH: ClassVar = list(APP_STYLESHEETS)

    def compose(self):
        with Horizontal(id="mcp-hub-grid"):
            yield Vertical(id="mcp-hub-rail")
            yield Vertical(id="mcp-hub-canvas")
            yield MCPInspector(id="mcp-hub-inspector")

    def on_resize(self, event):
        grid = self.query_one("#mcp-hub-grid")
        grid.set_class(event.size.width < 120, "mcp-compact")
        # Model the real destination's header/footer, using its available height.
        grid.styles.height = max(1, event.size.height - 8)


def assert_visible(app, control):
    visible = app.screen._compositor.visible_widgets
    inspector = app.query_one(MCPInspector)
    diagnostic = {
        "control": control.id,
        "region": control.region,
        "inspector": inspector.region,
        "overflow": inspector.styles.overflow_y,
        "scroll": inspector.scroll_y,
        "max_scroll": inspector.max_scroll_y,
    }
    assert control in visible, diagnostic
    region, clip = visible[control]
    assert region.intersection(clip) == region, diagnostic


def painted(app, control):
    region = control.region
    strips = app.screen._compositor.render_strips()
    return " ".join(
        strips[y].crop(region.x, region.right).text.strip()
        for y in range(region.y, region.bottom)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@pytest.mark.parametrize("raw", [False, True])
@private_profile_test
async def test_keyboard_form_actions_and_resize_keep_arguments(
    request, monkeypatch, theme, size, raw
):
    monkeypatch.setattr(
        inspector_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    app = ReachabilityApp()
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool(input_schema=None) if raw else _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        opener = app.query_one("#mcp-inspector-test-tool", Button)
        opener.focus()
        await pilot.pause()
        assert_visible(app, opener)
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        field = (
            app.query_one("#mcp-schema-raw", TextArea)
            if raw
            else app.query_one("#mcp-schema-field-0", Input)
        )
        assert field.has_focus
        assert_visible(app, field)
        if raw:
            field.text = '{"query": "retained draft"}'
        else:
            field.value = "retained draft"
        inspector.show_test_preview(
            ToolTestAdmissionPreview(
                nonce="preview-1",
                server_key=tool.server_key,
                tool_name=tool.name,
                definition_hash="definition",
                rendered_gate="ask",
                authority_fingerprint=None,
                safe_authority_label=None,
            )
        )
        run = app.query_one("#mcp-inspector-test-run", Button)
        await pilot.pause()
        # Tab really transfers focus out of the form (TextArea uses Tab for this).
        await pilot.press("tab")
        await pilot.pause()
        assert run.has_focus, (app.focused, field.text if raw else field.value)
        assert_visible(app, run)
        assert "Approve & run once" in painted(app, run)
        for resized in ((170, 48), (80, 24), size):
            await pilot.resize_terminal(*resized)
            await pilot.pause()
            assert run.has_focus
            assert_visible(app, run)
            assert "Approve & run once" in painted(app, run)
            assert "retained draft" in (field.text if raw else field.value)
        await pilot.press("enter")
        await pilot.pause()
        requests = [
            e for e in app.events if isinstance(e, MCPInspector.ToolTestRequested)
        ]
        assert len(requests) == 1
        assert requests[0].arguments == {"query": "retained draft"}
        close = app.query_one("#mcp-inspector-test-close", Button)
        close.focus()
        await pilot.pause()
        assert_visible(app, close)
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query("#mcp-inspector-test-panel")
        assert not opener.disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("panel", ["readiness", "permission", "advanced"])
@private_profile_test
async def test_other_inspector_actions_fit_beside_scrollbar(
    request, monkeypatch, theme, panel
):
    monkeypatch.setattr(
        inspector_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    app = ReachabilityApp()
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        inspector = app.query_one(MCPInspector)
        if panel == "readiness":
            await inspector.update_readiness(_stale_snap())
            expected = {"mcp-inspector-action-connect"}
        elif panel == "permission":
            tool = _tool()
            await inspector.show_permission(
                tool,
                EffectiveToolState(
                    state="ask", origin="tool_override", config_changed=True
                ),
                arg_rules=[{"rule_id": "rule-1", "args_json": '{"query":"example"}'}],
                session_approvals=[(tool.server_key, tool.name)],
            )
            expected = {
                "mcp-inspector-reallow",
                "mcp-inspector-arg-rule-remove-0",
                "mcp-inspector-session-approval-revoke-0",
            }
        else:
            opener = app.query_one("#mcp-inspector-advanced-reveal", Button)
            opener.focus()
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            expected = {"mcp-adv-run"}
        inspector.query_one("#mcp-inspector-message", Static).update(
            "Current server status contains several lines of diagnostic detail. " * 8
        )
        await pilot.pause()
        assert inspector.show_vertical_scrollbar
        exercised = set()
        for button in inspector.query(Button):
            if not button.display:
                continue
            button.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert_visible(app, button)
            assert " ".join(str(button.label).split()) in " ".join(
                painted(app, button).split()
            )
            exercised.add(button.id)
        assert expected <= exercised
