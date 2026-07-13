# Tests/UI/test_mcp_rail.py
from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Select

import tldw_chatbook
from tldw_chatbook.MCP.readiness import ReadinessSnapshot, ReadinessState
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCP_RAIL_ROW_PREFIX, MCPRail

_BUNDLED_CSS_PATH = str(Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss")


def _snap(key: str, label: str, state: ReadinessState = ReadinessState.READY) -> ReadinessSnapshot:
    return ReadinessSnapshot(
        server_key=key, label=label, source=key.split(":", 1)[0],
        state=state, reasons=(), message="",
    )


class RailApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield MCPRail(
            source="local",
            snapshots=[
                _snap("builtin:tldw_chatbook", "tldw_chatbook (built-in)"),
                _snap("local:docs", "docs", ReadinessState.NEEDS_SETUP),
            ],
            selected_server_key=None,
            scope_options=[("Personal", "personal")],
            scope_value="personal",
            scope_ref_options=[],
            scope_ref_value=None,
            id="mcp-rail",
        )

    def on_mcp_rail_server_selected(self, event: MCPRail.ServerSelected) -> None:
        self.events.append(event)

    def on_mcp_rail_source_changed(self, event: MCPRail.SourceChanged) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_rail_renders_all_servers_row_plus_one_row_per_snapshot():
    app = RailApp()
    async with app.run_test() as pilot:
        rows = list(app.query(f"Button.mcp-rail-row"))
        # "All servers" + builtin + docs
        assert len(rows) == 3
        labels = [str(row.label) for row in rows]
        assert any("All servers" in label for label in labels)
        assert any("docs" in label for label in labels)


@pytest.mark.asyncio
async def test_rail_row_click_posts_server_selected_with_key():
    app = RailApp()
    async with app.run_test() as pilot:
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # index 2 = "local:docs"
        await pilot.pause()
        selected = [e for e in app.events if isinstance(e, MCPRail.ServerSelected)]
        assert selected and selected[-1].server_key == "local:docs"


@pytest.mark.asyncio
async def test_rail_source_select_posts_source_changed():
    app = RailApp()
    async with app.run_test() as pilot:
        select = app.query_one("#mcp-rail-source", Select)
        select.value = "server"
        await pilot.pause()
        changed = [e for e in app.events if isinstance(e, MCPRail.SourceChanged)]
        assert changed and changed[-1].source == "server"


@pytest.mark.asyncio
async def test_rail_hides_scope_section_for_local_source():
    app = RailApp()
    async with app.run_test() as pilot:
        assert not list(app.query("#mcp-rail-scope"))


class RailScopeMismatchApp(App):
    """Reproduces a restored legacy scope ('team') outside Phase 1's Personal-only options."""

    def compose(self) -> ComposeResult:
        yield MCPRail(
            source="server",
            snapshots=[],
            selected_server_key=None,
            scope_options=[("Personal", "personal")],
            scope_value="team",
            scope_ref_options=[],
            scope_ref_value="21",
            id="mcp-rail",
        )


@pytest.mark.asyncio
async def test_rail_clamps_scope_value_not_in_options_instead_of_crashing():
    """Restoring a legacy scope not among Phase 1's options must not crash the Select.

    Regression test: MCPRail.compose() used to pass a restored `scope_value`
    (e.g. "team") straight through to the scope Select's `value=`, which
    raises `InvalidSelectValueError` when that value isn't among the
    Select's options (Phase 1 only offers Personal). The rail's DISPLAY
    should clamp to the first available option; state tracking of the true
    restored scope lives in the workbench, not the rail.
    """
    app = RailScopeMismatchApp()
    async with app.run_test() as pilot:
        select = app.query_one("#mcp-rail-scope-select", Select)
        assert select.value == "personal"
        ref_select = app.query_one("#mcp-rail-scope-ref", Select)
        assert ref_select.value is Select.BLANK


class RailScopeRefMismatchApp(App):
    """Reproduces a restored scope-ref value absent from real (non-empty) scope-ref options."""

    def compose(self) -> ComposeResult:
        yield MCPRail(
            source="server",
            snapshots=[],
            selected_server_key=None,
            scope_options=[("Personal", "personal"), ("Team", "team")],
            scope_value="team",
            scope_ref_options=[("Org 9", "9")],
            scope_ref_value="21",
            id="mcp-rail",
        )


@pytest.mark.asyncio
async def test_rail_clamps_scope_ref_value_not_in_options_to_no_selection():
    app = RailScopeRefMismatchApp()
    async with app.run_test() as pilot:
        select = app.query_one("#mcp-rail-scope-select", Select)
        assert select.value == "team"  # present among options — no clamp needed here
        ref_select = app.query_one("#mcp-rail-scope-ref", Select)
        assert ref_select.value is Select.NULL


class RailAppWithBundledCSS(App):
    """Mounts MCPRail under `#mcp-hub-rail` (the id the real MCP screen uses) and
    loads the actual bundled stylesheet, so `#mcp-hub-rail Button.mcp-rail-row.is-active`
    resolves exactly as it does in the live app.
    """

    CSS_PATH = _BUNDLED_CSS_PATH

    def compose(self) -> ComposeResult:
        yield MCPRail(
            source="local",
            snapshots=[
                _snap("builtin:tldw_chatbook", "tldw_chatbook (built-in)"),
                _snap("local:docs", "docs", ReadinessState.NEEDS_SETUP),
            ],
            selected_server_key=None,
            scope_options=[("Personal", "personal")],
            scope_value="personal",
            scope_ref_options=[],
            scope_ref_value=None,
            id="mcp-hub-rail",
        )


def _strip_text(strip) -> str:
    return "".join(segment.text for segment in strip)


@pytest.mark.asyncio
async def test_rail_active_row_label_is_not_blank_with_bundled_css():
    """Regression test for a blank selected rail row.

    The rail's own `#mcp-hub-rail Button.mcp-rail-row.is-active` rule used to
    only set `text-style: bold`, leaving the generic `.is-active` rule's
    `border: round $ds-action-focus` in effect. Button.mcp-rail-row is fixed
    at height 1 (see MCPRail.DEFAULT_CSS); a round border needs at least 2
    lines to render, so it consumed the row's only line and the label's
    content area collapsed to height 0 -- the selected row rendered as a
    blank bordered box. The fix adds `border: none` to that rule (mirroring
    the sibling `.mcp-mode-chip.is-active` rule), so the active row keeps
    height 1 and its label stays visible.
    """
    app = RailAppWithBundledCSS()
    async with app.run_test(size=(80, 30)) as pilot:
        active_row = app.query_one(f"#{MCP_RAIL_ROW_PREFIX}0", Button)
        assert "is-active" in active_row.classes
        # A round border needs >=2 lines; if one leaked back in, the row's
        # content (and thus its label) would be squeezed out again.
        assert active_row.styles.border.top[0] == ""
        assert active_row.size.height >= 1
        rendered_text = "".join(
            _strip_text(active_row.render_line(y)) for y in range(active_row.size.height)
        )
        assert "All servers" in rendered_text

        inactive_row = app.query_one(f"#{MCP_RAIL_ROW_PREFIX}2", Button)
        assert "is-active" not in inactive_row.classes
        inactive_text = "".join(
            _strip_text(inactive_row.render_line(y)) for y in range(inactive_row.size.height)
        )
        assert "docs" in inactive_text
