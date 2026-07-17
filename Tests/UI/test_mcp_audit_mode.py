# Tests/UI/test_mcp_audit_mode.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select

import tldw_chatbook
from tldw_chatbook.UI.MCP_Modules.mcp_audit_mode import MCPAuditMode

_CSS_ROOT = Path(tldw_chatbook.__file__).parent / "css"
_BUNDLED_STYLESHEET = _CSS_ROOT / "tldw_cli_modular.tcss"


def _entry(
    *,
    ts: str = "2026-07-16T21:22:00+00:00",
    server_key: str = "local:docs",
    tool_name: str = "search",
    initiator: str = "test",
    decision: str = "allowed",
    ok: bool = True,
    duration_ms: int = 42,
    error: str | None = None,
    arguments: dict[str, Any] | None = None,
    result_excerpt: str | None = None,
) -> dict[str, Any]:
    return {
        "ts": ts,
        "server_key": server_key,
        "tool_name": tool_name,
        "initiator": initiator,
        "decision": decision,
        "ok": ok,
        "duration_ms": duration_ms,
        "error": error,
        "arguments": arguments,
        "result_excerpt": result_excerpt,
    }


class AuditModeApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield MCPAuditMode(id="mcp-mode-canvas-audit")

    def on_mcp_audit_mode_entry_selected(self, event) -> None:
        self.events.append(event)

    def on_mcp_audit_mode_finding_selected(self, event) -> None:
        self.events.append(event)


def _row_texts(table: DataTable, row_index: int) -> list[str]:
    row = table.get_row_at(row_index)
    return [cell.plain if hasattr(cell, "plain") else str(cell) for cell in row]


# -- rendering ----------------------------------------------------------


@pytest.mark.asyncio
async def test_rows_render_in_given_newest_first_order_with_all_columns():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        entries = [
            _entry(ts="2026-07-16T21:22:02+00:00", tool_name="newest"),
            _entry(ts="2026-07-16T21:22:01+00:00", tool_name="middle"),
            _entry(ts="2026-07-16T21:22:00+00:00", tool_name="oldest"),
        ]
        await canvas.update_entries(entries)
        await pilot.pause()

        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.display is True
        assert table.row_count == 3
        # Rendered in the exact order given -- the widget never re-sorts.
        assert _row_texts(table, 0)[1] == "local:docs::newest"
        assert _row_texts(table, 1)[1] == "local:docs::middle"
        assert _row_texts(table, 2)[1] == "local:docs::oldest"
        assert app.query_one("#mcp-audit-empty").display is False


@pytest.mark.asyncio
async def test_when_column_formats_iso_timestamp():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries([_entry(ts="2026-07-16T21:22:05+00:00")])
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        assert _row_texts(table, 0)[0] == "2026-07-16 21:22:05"


@pytest.mark.asyncio
async def test_when_column_falls_back_gracefully_for_malformed_or_missing_ts():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries(
            [_entry(ts="not-a-timestamp", tool_name="a"), _entry(ts="", tool_name="b")]
        )
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        assert _row_texts(table, 0)[0] == "not-a-timestamp"
        assert _row_texts(table, 1)[0] == "—"


@pytest.mark.asyncio
async def test_duration_column_uses_shared_formatter():
    """The Duration column must use the SAME ms->s formatter Test Tool's
    result status line uses (`mcp_inspector.format_duration_ms()`), not a
    duplicate implementation."""
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries(
            [
                _entry(tool_name="ms", duration_ms=850),
                _entry(tool_name="secs", duration_ms=2500),
                _entry(tool_name="mins", duration_ms=90_000),
            ]
        )
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        durations = {_row_texts(table, i)[1].split("::")[-1]: _row_texts(table, i)[4] for i in range(3)}
        assert durations["ms"] == "850ms"
        assert durations["secs"] == "2.5s"
        assert durations["mins"] == "1m 30s"


@pytest.mark.asyncio
async def test_outcome_column_variants():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries(
            [
                _entry(tool_name="ok_call", decision="allowed", ok=True),
                _entry(tool_name="failed_call", decision="approved", ok=False, error="boom"),
                _entry(tool_name="blocked_call", decision="denied", ok=False, duration_ms=0),
                _entry(tool_name="timeout_call", decision="denied-timeout", ok=False, duration_ms=0),
                _entry(tool_name="downgraded_call", decision="downgraded", ok=False, duration_ms=0),
            ]
        )
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        outcomes = {
            _row_texts(table, i)[1].split("::")[-1]: _row_texts(table, i)[5] for i in range(table.row_count)
        }
        assert outcomes["ok_call"] == "OK"
        assert outcomes["failed_call"] == "Failed"
        assert outcomes["blocked_call"] == "Blocked"
        assert outcomes["timeout_call"] == "Blocked"
        assert outcomes["downgraded_call"] == "Downgraded"


@pytest.mark.asyncio
async def test_initiator_and_decision_columns_fall_back_to_em_dash_when_missing():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        entry = _entry()
        entry["initiator"] = None
        entry["decision"] = None
        await canvas.update_entries([entry])
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        assert _row_texts(table, 0)[2] == "—"
        assert _row_texts(table, 0)[3] == "—"


@pytest.mark.asyncio
async def test_row_keys_are_stable_synthetic_index():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries([_entry(tool_name="a"), _entry(tool_name="b")])
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        first_key, _ = table.coordinate_to_cell_key((0, 0))
        second_key, _ = table.coordinate_to_cell_key((1, 0))
        assert first_key.value == "0"
        assert second_key.value == "1"


# -- empty state ----------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_state_shown_when_no_entries_recorded():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries([])
        await pilot.pause()
        empty = app.query_one("#mcp-audit-empty")
        assert empty.display is True
        message = str(app.query_one("#mcp-audit-empty-message").renderable)
        assert message == "No tool executions recorded yet."
        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.display is False


@pytest.mark.asyncio
async def test_filter_narrowing_to_zero_rows_does_not_show_empty_state():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries([_entry(tool_name="search")])
        await pilot.pause()

        text_input = app.query_one("#mcp-audit-filter-text", Input)
        text_input.value = "does-not-match-anything"
        await pilot.pause()

        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.row_count == 0
        assert table.display is True
        assert app.query_one("#mcp-audit-empty").display is False


@pytest.mark.asyncio
async def test_going_from_empty_to_populated_hides_empty_state_again():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries([])
        await pilot.pause()
        assert app.query_one("#mcp-audit-empty").display is True

        await canvas.update_entries([_entry()])
        await pilot.pause()
        assert app.query_one("#mcp-audit-empty").display is False
        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.display is True
        assert table.row_count == 1


# -- filters ----------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_by_decision_select_narrows_rows():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries(
            [
                _entry(tool_name="a", decision="allowed"),
                _entry(tool_name="b", decision="denied"),
            ]
        )
        await pilot.pause()

        select = app.query_one("#mcp-audit-filter-decision", Select)
        select.value = "denied"
        await pilot.pause()

        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.row_count == 1
        assert _row_texts(table, 0)[1] == "local:docs::b"

        select.value = Select.NULL
        await pilot.pause()
        assert table.row_count == 2


@pytest.mark.asyncio
async def test_filter_by_initiator_select_narrows_rows():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries(
            [
                _entry(tool_name="a", initiator="test"),
                _entry(tool_name="b", initiator="agent"),
            ]
        )
        await pilot.pause()

        select = app.query_one("#mcp-audit-filter-initiator", Select)
        select.value = "agent"
        await pilot.pause()

        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.row_count == 1
        assert _row_texts(table, 0)[1] == "local:docs::b"


@pytest.mark.asyncio
async def test_filter_by_text_matches_tool_name_or_server_key():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries(
            [
                _entry(server_key="local:docs", tool_name="search"),
                _entry(server_key="server:weather", tool_name="forecast"),
            ]
        )
        await pilot.pause()

        text_input = app.query_one("#mcp-audit-filter-text", Input)
        text_input.value = "weather"
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.row_count == 1
        assert _row_texts(table, 0)[1] == "server:weather::forecast"

        text_input.value = "search"
        await pilot.pause()
        assert table.row_count == 1
        assert _row_texts(table, 0)[1] == "local:docs::search"


@pytest.mark.asyncio
async def test_filters_combine_with_and_semantics():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries(
            [
                _entry(tool_name="a", decision="allowed", initiator="test"),
                _entry(tool_name="b", decision="allowed", initiator="agent"),
                _entry(tool_name="c", decision="denied", initiator="agent"),
            ]
        )
        await pilot.pause()

        app.query_one("#mcp-audit-filter-decision", Select).value = "allowed"
        await pilot.pause()
        app.query_one("#mcp-audit-filter-initiator", Select).value = "agent"
        await pilot.pause()

        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.row_count == 1
        assert _row_texts(table, 0)[1] == "local:docs::b"


@pytest.mark.asyncio
async def test_select_options_cover_full_decision_and_initiator_vocabulary():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        decision_select = app.query_one("#mcp-audit-filter-decision", Select)
        initiator_select = app.query_one("#mcp-audit-filter-initiator", Select)
        decision_values = {value for _, value in decision_select._options if value is not Select.NULL}
        initiator_values = {value for _, value in initiator_select._options if value is not Select.NULL}
        assert decision_values == {"allowed", "approved", "denied", "denied-timeout", "downgraded"}
        assert initiator_values == {"test", "agent", "system"}
        assert decision_select.value is Select.NULL
        assert initiator_select.value is Select.NULL


@pytest.mark.asyncio
async def test_select_mount_posts_no_changed_event():
    """Both filter Selects are constructed with `value=Select.NULL`, the
    SAME value as the reactive var's own default -- Textual only fires a
    `Changed` echo when the constructor value DIFFERS from that default, so
    mounting these must never trigger a spurious re-filter."""
    app = AuditModeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        canvas = app.query_one(MCPAuditMode)
        assert canvas._filter_decision is None
        assert canvas._filter_initiator is None
        assert app.events == []


# -- cursor re-seat ----------------------------------------------------------


@pytest.mark.asyncio
async def test_cursor_key_is_preserved_across_a_filter_rebuild():
    """`DataTable.clear()` unconditionally resets the cursor to (0, 0) --
    a filter Input keystroke rebuilds the table on every change, so without
    a row-key-based restore, selecting row 1 and then typing an unrelated
    filter character would silently redirect the cursor back to row 0."""
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries(
            [_entry(tool_name="a"), _entry(tool_name="b"), _entry(tool_name="c")]
        )
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        table.focus()
        table.move_cursor(row=1)
        await pilot.pause()

        # A text filter that still matches every row -- the same three rows
        # re-render (same keys "0"/"1"/"2"), so the cursor should stay on
        # key "1" ("b").
        text_input = app.query_one("#mcp-audit-filter-text", Input)
        text_input.value = "local"
        await pilot.pause()

        row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
        assert row_key.value == "1"


# -- selection ----------------------------------------------------------


@pytest.mark.asyncio
async def test_row_selection_posts_entry_selected_with_synthetic_index():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries([_entry(tool_name="a"), _entry(tool_name="b")])
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        table.focus()
        table.move_cursor(row=1)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert isinstance(event, MCPAuditMode.EntrySelected)
        assert event.index == 1


# -- real bundled CSS (Phase 3/4 lesson: DEFAULT_CSS alone can still
# collapse to 0x0 under the real app stylesheet's global widget rules) ------


class AuditModeAppWithBundledCSS(App):
    """Mirrors `ToolsModeAppWithBundledCSS`/`PermissionsModeAppWithBundledCSS`
    -- loads the real generated bundle as CSS_PATH so the table and filter
    bar contest their actual CSS priority battle exactly as they do in the
    live app."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        yield MCPAuditMode(id="mcp-mode-canvas-audit")


@pytest.mark.asyncio
async def test_table_and_filter_bar_have_nonzero_geometry_with_bundled_css():
    app = AuditModeAppWithBundledCSS()
    async with app.run_test(size=(120, 40)) as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_entries(
            [_entry(server_key="local:docs", tool_name="search_docs")]
        )
        await pilot.pause()

        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.size.width > 0, "audit table collapsed to zero width under bundled CSS"
        assert table.size.height > 0, "audit table collapsed to zero height under bundled CSS"

        text_input = app.query_one("#mcp-audit-filter-text", Input)
        assert text_input.size.width > 0, "filter text Input collapsed to zero width under bundled CSS"
        assert text_input.size.height > 0, "filter text Input collapsed to zero height under bundled CSS"

        decision_select = app.query_one("#mcp-audit-filter-decision", Select)
        assert decision_select.size.width > 0, (
            "filter-decision Select collapsed to zero width under bundled CSS"
        )
        assert decision_select.size.height > 0, (
            "filter-decision Select collapsed to zero height under bundled CSS"
        )

        initiator_select = app.query_one("#mcp-audit-filter-initiator", Select)
        assert initiator_select.size.width > 0, (
            "filter-initiator Select collapsed to zero width under bundled CSS"
        )
        assert initiator_select.size.height > 0, (
            "filter-initiator Select collapsed to zero height under bundled CSS"
        )


# -- T8 (MCP Hub Phase 5): sub-view strip --------------------------------


@pytest.mark.asyncio
async def test_default_subview_is_executions():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#mcp-audit-executions-view").display is True
        assert app.query_one("#mcp-audit-findings-view").display is False
        exec_btn = app.query_one("#mcp-audit-subview-executions", Button)
        find_btn = app.query_one("#mcp-audit-subview-findings", Button)
        assert "is-active" in exec_btn.classes
        assert "is-active" not in find_btn.classes


@pytest.mark.asyncio
async def test_clicking_findings_button_switches_subview():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        assert app.query_one("#mcp-audit-executions-view").display is False
        assert app.query_one("#mcp-audit-findings-view").display is True
        exec_btn = app.query_one("#mcp-audit-subview-executions", Button)
        find_btn = app.query_one("#mcp-audit-subview-findings", Button)
        assert "is-active" not in exec_btn.classes
        assert "is-active" in find_btn.classes


@pytest.mark.asyncio
async def test_clicking_executions_button_switches_back():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-executions")
        await pilot.pause()
        assert app.query_one("#mcp-audit-executions-view").display is True
        assert app.query_one("#mcp-audit-findings-view").display is False


@pytest.mark.asyncio
async def test_subview_buttons_are_tooltipped():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#mcp-audit-subview-executions", Button).tooltip
        assert app.query_one("#mcp-audit-subview-findings", Button).tooltip


# -- T8 (MCP Hub Phase 5): Findings rendering ----------------------------


def _finding(
    *,
    severity: str = "high",
    finding_type: str = "orphaned_path_scope",
    object_kind: str = "path_scope",
    object_id: str = "5",
    message: str = "Needs review",
    remediation: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "severity": severity,
        "finding_type": finding_type,
        "object_kind": object_kind,
        "object_id": object_id,
        "message": message,
    }
    if remediation is not None:
        payload["remediation"] = remediation
    return payload


@pytest.mark.asyncio
async def test_findings_render_from_a_fake_payload():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_findings(
            [
                _finding(severity="high", finding_type="orphaned_path_scope", message="Needs review"),
                _finding(severity="low", finding_type="stale_binding", message="Check binding"),
            ],
            source="server",
        )
        await pilot.pause()

        table = app.query_one("#mcp-audit-findings-table", DataTable)
        assert table.display is True
        assert table.row_count == 2
        assert _row_texts(table, 0) == ["high", "orphaned_path_scope", "Needs review"]
        assert _row_texts(table, 1) == ["low", "stale_binding", "Check binding"]
        assert app.query_one("#mcp-audit-findings-empty").display is False


@pytest.mark.asyncio
async def test_findings_defensive_missing_fields_fall_back_to_em_dash():
    """Mirrors `hub_tool_catalog.server_tools_from_inventory()`'s own
    tolerant-of-missing-keys reads -- a raw finding straight off the wire
    may be missing any field."""
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_findings([{"message": "only message"}], source="server")
        await pilot.pause()
        table = app.query_one("#mcp-audit-findings-table", DataTable)
        assert _row_texts(table, 0) == ["—", "—", "only message"]


@pytest.mark.asyncio
async def test_findings_row_keys_are_stable_synthetic_index():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_findings(
            [_finding(message="a"), _finding(message="b")], source="server"
        )
        await pilot.pause()
        table = app.query_one("#mcp-audit-findings-table", DataTable)
        first_key, _ = table.coordinate_to_cell_key((0, 0))
        second_key, _ = table.coordinate_to_cell_key((1, 0))
        assert first_key.value == "0"
        assert second_key.value == "1"


@pytest.mark.asyncio
async def test_findings_cursor_key_is_preserved_across_a_resync_with_same_rows():
    """`DataTable.clear()` unconditionally resets the cursor to (0, 0) --
    the next `_sync_children()` pass calls `update_findings()` again even
    when the underlying list is unchanged, so without a row-key-based
    restore the cursor would silently jump back to row 0 on every
    background resync."""
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        findings = [_finding(message="a"), _finding(message="b"), _finding(message="c")]
        await canvas.update_findings(findings, source="server")
        await pilot.pause()
        table = app.query_one("#mcp-audit-findings-table", DataTable)
        table.focus()
        table.move_cursor(row=1)
        await pilot.pause()

        await canvas.update_findings(list(findings), source="server")
        await pilot.pause()

        row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
        assert row_key.value == "1"


# -- T8 (MCP Hub Phase 5): Findings empty states -------------------------


@pytest.mark.asyncio
async def test_local_source_shows_server_only_copy():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_findings(None, source="local")
        await pilot.pause()

        table = app.query_one("#mcp-audit-findings-table", DataTable)
        assert table.display is False
        empty = app.query_one("#mcp-audit-findings-empty")
        assert empty.display is True
        message = str(app.query_one("#mcp-audit-findings-empty-message").renderable)
        assert message == "Findings come from a tldw_server target."


@pytest.mark.asyncio
async def test_builtin_source_shows_server_only_copy():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_findings(None, source="builtin")
        await pilot.pause()
        message = str(app.query_one("#mcp-audit-findings-empty-message").renderable)
        assert message == "Findings come from a tldw_server target."


@pytest.mark.asyncio
async def test_mount_before_any_update_findings_call_shows_server_only_copy():
    """Default `_findings_source` mirrors `MCPWorkbench._source`'s own
    default ("local") -- a fresh mount before the first `_sync_children()`
    pass renders the same server-only copy a local source shows, never a
    blank or crashing Findings sub-view."""
    app = AuditModeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        message = str(app.query_one("#mcp-audit-findings-empty-message").renderable)
        assert message == "Findings come from a tldw_server target."


@pytest.mark.asyncio
async def test_server_source_fetch_failure_shows_fail_soft_retry_hint():
    """`findings=None` under `source="server"` means the fetch itself
    failed (as opposed to `source != "server"`, meaning findings were never
    applicable) -- distinct copy from the local/builtin empty state, per
    the fail-soft "absent listing with a retry hint" contract."""
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_findings(None, source="server")
        await pilot.pause()

        table = app.query_one("#mcp-audit-findings-table", DataTable)
        assert table.display is False
        empty = app.query_one("#mcp-audit-findings-empty")
        assert empty.display is True
        message = str(app.query_one("#mcp-audit-findings-empty-message").renderable)
        assert message != "Findings come from a tldw_server target."
        assert message


@pytest.mark.asyncio
async def test_server_source_zero_findings_shows_distinct_empty_copy():
    """A successful fetch that legitimately returns zero findings is not
    the same case as a failed fetch -- distinct copy from both the
    local/builtin empty state and the fetch-failure hint."""
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_findings([], source="server")
        await pilot.pause()

        assert app.query_one("#mcp-audit-findings-table").display is False
        message = str(app.query_one("#mcp-audit-findings-empty-message").renderable)
        assert message not in ("Findings come from a tldw_server target.", "")

        await canvas.update_findings(None, source="server")
        await pilot.pause()
        failure_message = str(app.query_one("#mcp-audit-findings-empty-message").renderable)
        assert failure_message != message


@pytest.mark.asyncio
async def test_going_from_absent_to_populated_findings_hides_empty_state():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_findings(None, source="server")
        await pilot.pause()
        assert app.query_one("#mcp-audit-findings-empty").display is True

        await canvas.update_findings([_finding()], source="server")
        await pilot.pause()
        assert app.query_one("#mcp-audit-findings-empty").display is False
        table = app.query_one("#mcp-audit-findings-table", DataTable)
        assert table.display is True
        assert table.row_count == 1


# -- T8 (MCP Hub Phase 5): Findings selection ----------------------------


@pytest.mark.asyncio
async def test_findings_row_selection_posts_finding_selected_with_synthetic_index():
    app = AuditModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_findings(
            [_finding(message="a"), _finding(message="b")], source="server"
        )
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        table = app.query_one("#mcp-audit-findings-table", DataTable)
        table.focus()
        table.move_cursor(row=1)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert isinstance(event, MCPAuditMode.FindingSelected)
        assert event.index == 1


# -- T8 (MCP Hub Phase 5): real bundled CSS ------------------------------


@pytest.mark.asyncio
async def test_subview_strip_and_findings_table_have_nonzero_geometry_with_bundled_css():
    app = AuditModeAppWithBundledCSS()
    async with app.run_test(size=(120, 40)) as pilot:
        canvas = app.query_one(MCPAuditMode)
        await canvas.update_findings([_finding()], source="server")
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()

        exec_btn = app.query_one("#mcp-audit-subview-executions", Button)
        find_btn = app.query_one("#mcp-audit-subview-findings", Button)
        assert exec_btn.size.width > 0 and exec_btn.size.height > 0, (
            "Executions sub-view button collapsed to zero size under bundled CSS"
        )
        assert find_btn.size.width > 0 and find_btn.size.height > 0, (
            "Findings sub-view button collapsed to zero size under bundled CSS"
        )

        table = app.query_one("#mcp-audit-findings-table", DataTable)
        assert table.size.width > 0, "findings table collapsed to zero width under bundled CSS"
        assert table.size.height > 0, "findings table collapsed to zero height under bundled CSS"
