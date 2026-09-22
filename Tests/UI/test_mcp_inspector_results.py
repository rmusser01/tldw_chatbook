"""Current tool attempts replace earlier output, including its raw disclosure."""

import asyncio
import json
from typing import ClassVar

import pytest
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, Input, Static, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_mcp_inspector import InspectorApp, _tool
from tldw_chatbook.MCP.hub_test_execution import ToolTestAdmissionPreview
from tldw_chatbook.UI.MCP_Modules import mcp_inspector as inspector_module
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import PermissionProfileContext


class _ResultApp(InspectorApp):
    """Mount production styles and pane widths with the destination chrome budget."""

    CSS_PATH: ClassVar = list(APP_STYLESHEETS)

    def compose(self):
        with Horizontal(id="mcp-hub-grid"):
            yield Vertical(id="mcp-hub-rail")
            yield Vertical(id="mcp-hub-canvas")
            yield MCPInspector(id="mcp-hub-inspector")

    def on_resize(self, event):
        grid = self.query_one("#mcp-hub-grid")
        grid.set_class(event.size.width < 120, "mcp-compact")
        # Runtime test geometry models the real destination's header/footer.
        grid.styles.height = max(1, event.size.height - 8)


def _assert_visible(app, control):
    visible = app.screen._compositor.visible_widgets
    assert control in visible, (control.id, control.region)
    region, clip = visible[control]
    assert region.width > 0 and region.height > 0
    assert region.intersection(clip) == region, (control.id, region, clip)


def _painted(app, control):
    region = control.region
    strips = app.screen._compositor.render_strips()
    return " ".join(
        strips[y].crop(region.x, region.right).text.strip()
        for y in range(region.y, region.bottom)
    )


async def _settle(pilot):
    await pilot.pause()
    await pilot.wait_for_scheduled_animations()
    await pilot.pause()


def _preview(tool):
    return ToolTestAdmissionPreview(
        nonce="preview-current",
        server_key=tool.server_key,
        tool_name=tool.name,
        definition_hash="definition",
        rendered_gate="allow",
        authority_fingerprint=None,
        safe_authority_label=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_raw_response_keyboard_access_survives_resize(
    request, monkeypatch, theme, size
):
    monkeypatch.setattr(
        inspector_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    app = _ResultApp()
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await inspector.open_test_panel()
        await app.workers.wait_for_complete()
        await _settle(pilot)
        inspector.show_test_preview(_preview(tool))
        raw = json.dumps(
            ["[bold] literal [/] 中文", *[f"row {i}" for i in range(40)], "END"],
            ensure_ascii=False,
            indent=2,
        )
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            result=[{"text": "current"}],
            raw=raw,
        )
        await _settle(pilot)
        disclosure = app.query_one("#mcp-inspector-test-result-raw", Collapsible)
        title = disclosure.query_one("CollapsibleTitle")
        title.focus()
        await _settle(pilot)
        assert disclosure.collapsed
        _assert_visible(app, title)
        assert "Raw response" in " ".join(_painted(app, title).split())
        await pilot.press("enter")
        await _settle(pilot)
        assert not disclosure.collapsed
        scroll = app.query_one("#mcp-inspector-test-result-raw-scroll", VerticalScroll)
        scroll.focus()
        await _settle(pilot)
        assert scroll.has_focus
        _assert_visible(app, scroll)
        body = app.query_one("#mcp-inspector-test-result-raw-body", Static)
        assert "[bold] literal [/] 中文" in str(body.renderable)
        assert not body._render_markup
        await pilot.press("end")
        await _settle(pilot)
        assert scroll.scroll_y == scroll.max_scroll_y > 0
        assert "END" in _painted(app, scroll)
        for dimensions in ((170, 48), (80, 24), size):
            await pilot.resize_terminal(*dimensions)
            await _settle(pilot)
            assert scroll.has_focus
            _assert_visible(app, scroll)
        title.focus()
        await _settle(pilot)
        assert "Raw response" in " ".join(_painted(app, title).split())
        await pilot.press("enter")
        await _settle(pilot)
        assert disclosure.collapsed
        _assert_visible(app, title)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_form", [False, True])
@pytest.mark.parametrize(
    "context", [None, PermissionProfileContext("research", 7, "b" * 64, 3)]
)
@private_profile_test
async def test_invalid_attempt_replaces_previous_details_and_can_be_corrected(
    request, monkeypatch, raw_form, context
):
    monkeypatch.setattr(
        inspector_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    app = _ResultApp()
    async with app.run_test(size=(170, 48)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool(input_schema=None) if raw_form else _tool()
        await inspector.show_tool(tool, profile_context=context)
        await inspector.open_test_panel()
        await app.workers.wait_for_complete()
        await _settle(pilot)
        inspector.show_test_preview(_preview(tool))
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            result=[],
            raw='{"previous":true}',
            decision_note="Earlier approval context",
            profile_context=context,
        )
        disclosure = app.query_one("#mcp-inspector-test-result-raw", Collapsible)
        disclosure.collapsed = False
        field = (
            app.query_one("#mcp-schema-raw", TextArea)
            if raw_form
            else app.query_one("#mcp-schema-field-0", Input)
        )
        if raw_form:
            field.text = '{"query":'
        else:
            field.value = ""
        run = app.query_one("#mcp-inspector-test-run", Button)
        run.focus()
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert str(
            app.query_one("#mcp-inspector-test-result", Static).renderable
        ).startswith("Failed")
        assert not disclosure.display
        assert not str(
            app.query_one("#mcp-inspector-test-result-raw-body", Static).renderable
        )
        note = app.query_one("#mcp-inspector-test-result-note", Static)
        assert not note.display and not str(note.renderable)
        assert not run.disabled
        assert not [
            e for e in app.events if isinstance(e, MCPInspector.ToolTestRequested)
        ]
        assert field.text == '{"query":' if raw_form else field.value == ""
        if raw_form:
            field.text = '{"query":"corrected"}'
        else:
            field.value = "corrected"
        # A second real press waits for Textual's one-shot activation debounce.
        async with asyncio.timeout(5):
            while run.has_class("-active"):
                await pilot.pause(0.02)
        await pilot.press("enter")
        await _settle(pilot)
        requests = [
            e for e in app.events if isinstance(e, MCPInspector.ToolTestRequested)
        ]
        assert len(requests) == 1 and requests[0].arguments == {"query": "corrected"}
        assert requests[0].profile_context == context
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            result=["corrected"],
            raw='{"current":true}',
            profile_context=context,
        )
        await _settle(pilot)
        assert disclosure.display
        assert '"current":true' in str(
            app.query_one("#mcp-inspector-test-result-raw-body", Static).renderable
        )
        assert not note.display
        # Results belonging to another tool cannot replace this output.
        inspector.show_tool_result(
            server_key=tool.server_key, tool_name="other", ok=False, text="old failure"
        )
        assert str(
            app.query_one("#mcp-inspector-test-result", Static).renderable
        ).startswith("OK")
        await pilot.press("escape")
        await _settle(pilot)
        assert not app.query("#mcp-inspector-test-panel")
