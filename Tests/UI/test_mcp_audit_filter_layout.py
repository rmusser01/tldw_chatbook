"""Audit filters retain readable values and keyboard access in real pane sizes."""

import pytest
from textual.widgets import DataTable, Input, Select

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_mcp_compact_readability import _compact, _open, _paint, _settle
from Tests.UI.test_mcp_permissions_handoff import _cursor_visible, _visible
from Tests.UI.test_mcp_workbench import _audit_record
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.UI.MCP_Modules.mcp_audit_mode import (
    _DECISION_OPTIONS,
    _INITIATOR_OPTIONS,
    MCPAuditMode,
)


async def open_audit(pilot, tmp_path):
    workbench = await _open(pilot, tmp_path / "permissions.json")
    workbench.set_mode("audit")
    canvas = workbench.query_one(MCPAuditMode)
    entries = [_audit_record(tool_name=f"search_{i:02}") for i in range(40)]
    workbench._last_audit_entries = entries
    await canvas.update_entries(entries)
    await _settle(pilot)
    return workbench, canvas


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (120, 40), (170, 48)])
@private_profile_test
async def test_filter_values_and_full_selector_labels_paint(
    request, tmp_path, theme, size
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        _, canvas = await open_audit(pilot, tmp_path)
        field = canvas.query_one("#mcp-audit-filter-text", Input)
        field.focus()
        await _settle(pilot)
        _visible(app, field)
        assert "Filter tool or server…" in _paint(app.screen, field.content_region)
        await pilot.press(*"search_17")
        await _settle(pilot)
        assert "search_17" in _paint(app.screen, field.content_region)
        assert canvas.query_one(DataTable).row_count == 1
        for name, prompt, options in [
            ("decision", "All decisions", _DECISION_OPTIONS),
            ("initiator", "All initiators", _INITIATOR_OPTIONS),
        ]:
            control = canvas.query_one(f"#mcp-audit-filter-{name}", Select)
            control.focus()
            for label, value in [(prompt, Select.NULL), *options]:
                control.value = value
                await _settle(pilot)
                assert app.focused is control
                _visible(app, control)
                assert _compact(label) in _compact(
                    _paint(app.screen, control.content_region)
                )


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_keyboard_filters_and_table_remain_visible_after_resize(
    request, tmp_path, theme
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, canvas = await open_audit(pilot, tmp_path)
        field = canvas.query_one("#mcp-audit-filter-text", Input)
        field.value = "search"
        controls = [
            field,
            canvas.query_one("#mcp-audit-filter-decision", Select),
            canvas.query_one("#mcp-audit-filter-initiator", Select),
            canvas.query_one("#mcp-audit-table", DataTable),
        ]
        for control in controls:
            control.focus()
            await _settle(pilot)
            if isinstance(control, DataTable):
                await pilot.press("ctrl+end")
            for size in [(80, 24), (100, 30), (120, 40), (170, 48)]:
                await pilot.resize_terminal(*size)
                await _settle(pilot)
                assert app.focused is control
                _visible(app, control)
                assert field.value == "search"
                if isinstance(control, DataTable):
                    assert control.cursor_row == 39
                    _cursor_visible(app, control)
        await pilot.resize_terminal(80, 24)
        field.focus()
        await _settle(pilot)
        for control in controls[1:]:
            await pilot.press("tab")
            await _settle(pilot)
            assert app.focused is control
            _visible(app, control)
        # Reflow must follow current focus instead of bringing a stale child back.
        rail = workbench.query_one("#mcp-rail-source", Select)
        rail.focus()
        await pilot.resize_terminal(100, 30)
        await _settle(pilot)
        assert app.focused is rail
