"""MCP tool-switch names and states remain readable in their allocated pane."""

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_compact_readability import _settle
from Tests.UI.test_mcp_root_settings import _open
from Tests.UI.test_mcp_workbench import WorkbenchAppWithBundledCSS
from tldw_chatbook import config
from tldw_chatbook.UI.MCP_Modules import mcp_workbench
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_each_tool_switch_paints_complete_label_and_state(
    request, monkeypatch, enabled, theme, size
):
    """A one-row/clipped gate must fail even if its label property is complete."""
    original_get = config.get_cli_setting

    def get_setting(section, key=None, default=None):
        if (section == "tools" and key and key.endswith("_enabled")) or (
            section == "console" and key == "local_tools_enabled"
        ):
            return enabled
        return original_get(section, key, default)

    monkeypatch.setattr(config, "get_cli_setting", get_setting)
    monkeypatch.setattr(mcp_workbench, "get_cli_setting", get_setting)
    app = WorkbenchAppWithBundledCSS()
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        workbench, _ = await _open(pilot)
        await workbench._select_server_key(
            next(s.server_key for s in workbench._snapshots if s.source == "builtin")
        )
        workbench.set_mode("servers")
        await _settle(pilot)
        canvas = workbench.query_one(MCPServersMode)
        buttons = list(canvas.query("#mcp-detail-tool-gates Button"))
        assert len(buttons) >= 9
        reachable = [button for button in buttons if not button.disabled]
        reachable[0].focus()
        await _settle(pilot)
        for index, button in enumerate(reachable):
            assert app.focused is button
            region, clip = app.screen._compositor.visible_widgets[button]
            assert region.intersection(clip) == region
            painted = "".join(
                strip.crop(region.x, region.right).text
                for strip in app.screen._compositor.render_strips()[
                    region.y : region.bottom
                ]
            )
            assert "".join(button.label.plain.split()) in "".join(painted.split())
            if index + 1 < len(reachable):
                await pilot.press("tab")
                await _settle(pilot)

        dependent = canvas.query_one("#mcp-gate-web_deep_search_enabled", Button)
        assert dependent.disabled is (not enabled)
        for disabled in (button for button in buttons if button.disabled):
            disabled.scroll_visible(animate=False, immediate=True)
            await _settle(pilot)
            region, clip = app.screen._compositor.visible_widgets[disabled]
            assert region.intersection(clip) == region
            painted = "".join(
                strip.crop(region.x, region.right).text
                for strip in app.screen._compositor.render_strips()[
                    region.y : region.bottom
                ]
            )
            assert "".join(disabled.label.plain.split()) in "".join(painted.split())
