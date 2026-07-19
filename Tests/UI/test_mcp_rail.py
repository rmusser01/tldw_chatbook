# Tests/UI/test_mcp_rail.py
from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Select

import tldw_chatbook
from tldw_chatbook.MCP.readiness import STATE_CSS_CLASSES, ReadinessSnapshot, ReadinessState
from tldw_chatbook.UI.MCP_Modules.mcp_rail import (
    _MAX_ROW_LABEL,
    MCP_RAIL_ROW_PREFIX,
    MCPRail,
    _row_label,
)

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

    def on_mcp_rail_scope_changed(self, event: MCPRail.ScopeChanged) -> None:
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


# -- A4: rail rows are left-aligned with a fixed glyph gutter and a wider
# truncation budget ----------------------------------------------------------


def test_row_label_truncation_budget_fits_full_builtin_label():
    """The built-in server's full label ("tldw_chatbook (built-in)", 24
    chars) must render whole, not "tldw_chatbook (buil..." -- the old budget
    of 22 truncated it even though the rail has room at typical widths.
    """
    assert _MAX_ROW_LABEL >= 36
    snap = _snap("builtin:tldw_chatbook", "tldw_chatbook (built-in)")
    label = _row_label(snap)
    assert "tldw_chatbook (built-in)" in label
    assert "..." not in label


@pytest.mark.asyncio
async def test_all_servers_row_shares_left_gutter_with_glyph_prefixed_rows():
    """"All servers" has no readiness glyph, but its label must start at the
    same column as glyph-prefixed rows ("<glyph> label...") so the rail's
    label column has one hard left edge instead of "All servers" sitting
    flush against the rail edge while every other row is indented.
    """
    app = RailApp()
    async with app.run_test() as pilot:
        all_row = app.query_one(f"#{MCP_RAIL_ROW_PREFIX}0", Button)
        glyph_row = app.query_one(f"#{MCP_RAIL_ROW_PREFIX}2", Button)  # local:docs
        all_label = str(all_row.label)
        glyph_label = str(glyph_row.label)
        # Glyph rows are "<glyph><space>label..." -- a 2-char-wide gutter.
        assert glyph_label[1] == " "
        # "All servers" must carry the same 2-char gutter instead of sitting
        # flush left.
        assert all_label[:2] == "  "
        assert all_label.strip() == "All servers"


@pytest.mark.asyncio
async def test_rail_rows_are_left_aligned_with_bundled_css():
    """Button defaults to `text-align: center; content-align: center middle`
    (see Textual's own Button.DEFAULT_CSS) -- `.mcp-rail-row` must override
    both to left, mirroring `.library-rail-row` in _agentic_terminal.tcss.
    """
    app = RailAppWithBundledCSS()
    async with app.run_test(size=(80, 30)) as pilot:
        row = app.query_one(f"#{MCP_RAIL_ROW_PREFIX}0", Button)
        assert row.styles.text_align == "left"
        assert row.styles.content_align_horizontal == "left"


# -- Task 11: status colors + rail count alignment ---------------------------


@pytest.mark.asyncio
async def test_rail_row_carries_state_css_class_and_swaps_on_resync():
    app = RailApp()
    async with app.run_test() as pilot:
        rail = app.query_one(MCPRail)
        docs_row = app.query_one(f"#{MCP_RAIL_ROW_PREFIX}2", Button)  # local:docs
        assert STATE_CSS_CLASSES[ReadinessState.NEEDS_SETUP] in docs_row.classes

        rail.sync_state(
            source="local",
            snapshots=[
                _snap("builtin:tldw_chatbook", "tldw_chatbook (built-in)"),
                _snap("local:docs", "docs", ReadinessState.NEEDS_ATTENTION),
            ],
            selected_server_key=None,
            scope_options=[("Personal", "personal")],
            scope_value="personal",
            scope_ref_options=[],
            scope_ref_value=None,
        )
        await pilot.pause()
        docs_row = app.query_one(f"#{MCP_RAIL_ROW_PREFIX}2", Button)
        assert STATE_CSS_CLASSES[ReadinessState.NEEDS_ATTENTION] in docs_row.classes
        assert STATE_CSS_CLASSES[ReadinessState.NEEDS_SETUP] not in docs_row.classes


class AdaptiveCountRailApp(App):
    """Two rows with very different label lengths -- used to prove A6's
    per-compose() adaptive pad width, not the old fixed 36-char budget."""

    def compose(self) -> ComposeResult:
        yield MCPRail(
            source="local",
            snapshots=[
                ReadinessSnapshot(
                    server_key="local:a", label="a", source="local",
                    state=ReadinessState.READY, reasons=(), message="", tool_count=3,
                ),
                ReadinessSnapshot(
                    server_key="local:bb", label="a-longer-profile-name", source="local",
                    state=ReadinessState.READY, reasons=(), message="", tool_count=12,
                ),
            ],
            selected_server_key=None,
            scope_options=[("Personal", "personal")],
            scope_value="personal",
            scope_ref_options=[],
            scope_ref_value=None,
            id="mcp-rail",
        )


@pytest.mark.asyncio
async def test_row_label_right_aligns_tool_count_at_adaptive_column():
    """A6: `MCPRail.compose()` now computes the count column's pad width per
    call as the longest CURRENT (post-truncate, post-escape) label among its
    rows -- not the fixed 36-char truncation budget -- so a short label's
    count sits right after ITS row's own content instead of stranded far
    right of where a long label's count lands. Driven through the real
    MCPRail (not `_row_label()` directly): the adaptive width is a property
    of the whole row set at a given compose(), not any single snapshot.
    `_row_label()` still defaults its `pad_width` param to `_MAX_ROW_LABEL`
    for a standalone call (e.g. the blank-count check below).
    """
    app = AdaptiveCountRailApp()
    async with app.run_test() as pilot:
        short_row = app.query_one(f"#{MCP_RAIL_ROW_PREFIX}1", Button)  # "a"
        long_row = app.query_one(f"#{MCP_RAIL_ROW_PREFIX}2", Button)   # "a-longer-profile-name"
        short_label = str(short_row.label)
        long_label = str(long_row.label)
        assert len(short_label) == len(long_label)
        assert short_label[-3:] == "  3"
        assert long_label[-3:] == " 12"
        # Adaptive width == len("a-longer-profile-name") == 21 chars, far
        # short of the old fixed 36-char budget -- the rendered row is
        # correspondingly shorter than a direct `_row_label()` call at the
        # default (fixed) pad width.
        assert len(short_label) < len(_row_label(_snap("local:a", "a")))

    # No count discovered yet -- the count field renders blank, not "0" or
    # any other placeholder digit (unaffected by the adaptive-width change;
    # exercised directly at the default/standalone pad width).
    none_label = _row_label(_snap("local:c", "c", ReadinessState.READY))
    assert none_label[-3:] == "   "


class AdaptiveCountWithMarkupCharsRailApp(App):
    """One row's label contains a Rich-markup-special character (`[`) --
    `escape_markup()` lengthens it by one char (`[test-server]` -> 13 raw,
    14 escaped). Review-fix regression guard for A6: `pad_width` must be
    measured from the ESCAPED text (what actually renders), the same text
    `_row_label()` pads -- if a future change measured the raw label instead,
    the two rows' count fields would land one column apart.
    """

    def compose(self) -> ComposeResult:
        yield MCPRail(
            source="local",
            snapshots=[
                ReadinessSnapshot(
                    server_key="local:docs", label="docs", source="local",
                    state=ReadinessState.READY, reasons=(), message="", tool_count=3,
                ),
                ReadinessSnapshot(
                    server_key="local:brk", label="[test-server]", source="local",
                    state=ReadinessState.READY, reasons=(), message="", tool_count=12,
                ),
            ],
            selected_server_key=None,
            scope_options=[("Personal", "personal")],
            scope_value="personal",
            scope_ref_options=[],
            scope_ref_value=None,
            id="mcp-rail",
        )


@pytest.mark.asyncio
async def test_adaptive_pad_width_measures_escaped_text_not_raw_label():
    app = AdaptiveCountWithMarkupCharsRailApp()
    async with app.run_test() as pilot:
        docs_row = app.query_one(f"#{MCP_RAIL_ROW_PREFIX}1", Button)
        bracket_row = app.query_one(f"#{MCP_RAIL_ROW_PREFIX}2", Button)
        docs_label = str(docs_row.label)
        bracket_label = str(bracket_row.label)
        assert len(docs_label) == len(bracket_label), (
            "pad_width must be measured from the same escaped text that gets "
            "rendered -- a raw-length regression would misalign these two "
            "rows' count columns"
        )
        assert docs_label[-3:] == "  3"
        assert bracket_label[-3:] == " 12"


# -- Task 4: one-shot mount-echo guard (A -> B -> A must not be swallowed) --


@pytest.mark.asyncio
async def test_source_select_stale_mount_echo_is_dropped_after_source_moves_on():
    """T9 (P4) regression: the source select's mount echo (Textual 8.2.7
    posts a `Select.Changed` for a Select's own constructor value as part of
    mounting it) can be PROCESSED after `rail.source` has already moved on
    -- e.g. a restored view state switched the workbench to "server" while
    the "local"-constructed select's echo was still queued. The old
    `event.value != self.source` guard then saw a difference and forwarded
    the stale echo as a genuine change, silently reverting the restored
    source (caught end-to-end by
    `test_mcp_destination_restores_unified_mcp_view_state_after_mount` in
    test_destination_shells.py). The per-instance one-shot guard
    (`_mcp_mount_echo_value`, mcp_rail.py) must drop that first
    constructor-valued Changed regardless of what `self.source` says by
    then."""
    app = RailApp()
    async with app.run_test() as pilot:
        rail = app.query_one(MCPRail)
        select = app.query_one("#mcp-rail-source", Select)
        # Handler-contract reproduction (the full pump-interleaving race
        # needs the workbench's multi-await restore path around it -- that
        # end-to-end shape is what the destination-shells test above pins;
        # this one pins the guard itself): model a select whose
        # constructor echo has NOT yet been consumed (as compose() leaves
        # it) while the tracked source has already moved on, then deliver
        # the echo. Without the per-instance guard this is exactly the
        # forwarded stale event that reverted the restored source.
        select._mcp_mount_echo_value = "local"  # unconsumed, as compose() set it
        rail.source = "server"
        rail.on_select_changed(Select.Changed(select, "local"))
        await pilot.pause()
        changed = [e for e in app.events if isinstance(e, MCPRail.SourceChanged)]
        assert changed == []  # stale echo dropped, source not reverted


@pytest.mark.asyncio
async def test_source_a_b_a_round_trip_still_dispatches_after_echo_consumed():
    """T9 (P4): the per-instance echo guard is one-shot -- after each
    generation's echo is consumed, a genuine user round trip back to
    "local" must still dispatch, mirroring
    `test_scope_a_b_a_dispatches_three_changes_and_mount_echo_zero` below.
    The rail is resynced between changes exactly as the workbench does
    after a real source switch (`_switch_source()` -> `_sync_children()`
    -> `sync_state()`), so the second-line `event.value != self.source`
    dedup compares against the POST-switch source."""
    app = RailApp()
    async with app.run_test() as pilot:
        rail = app.query_one(MCPRail)
        select = app.query_one("#mcp-rail-source", Select)
        # The initial mount echo has already been consumed during startup
        # (RailApp composes the rail with source="local").
        select.value = "server"  # A -> B: genuine change, must dispatch
        await pilot.pause()
        # The workbench's response to a real switch: resync the rail with
        # the new source (recompose -> fresh "server"-valued select whose
        # own echo must also be dropped).
        rail.sync_state(
            source="server",
            snapshots=[_snap("server:main", "Main Server")],
            selected_server_key=None,
            scope_options=[("Personal", "personal")],
            scope_value="personal",
            scope_ref_options=[],
            scope_ref_value=None,
        )
        await pilot.pause()
        select = app.query_one("#mcp-rail-source", Select)
        select.value = "local"  # B -> A: must NOT be swallowed as an echo
        await pilot.pause()
        changed = [e.source for e in app.events if isinstance(e, MCPRail.SourceChanged)]
        assert changed == ["server", "local"]


@pytest.mark.asyncio
async def test_scope_a_b_a_dispatches_three_changes_and_mount_echo_zero():
    app = RailApp()
    async with app.run_test() as pilot:
        rail = app.query_one(MCPRail)
        rail.sync_state(
            source="server",
            snapshots=[_snap("server:main", "Main Server")],
            selected_server_key=None,
            scope_options=[("Personal", "personal"), ("Team", "team")],
            scope_value="personal",
            scope_ref_options=[],
            scope_ref_value=None,
        )
        await pilot.pause()
        changes = [e for e in app.events if isinstance(e, MCPRail.ScopeChanged)]
        assert changes == []  # mount echo suppressed
        select = app.query_one("#mcp-rail-scope-select", Select)
        select.value = "team"       # A -> B
        await pilot.pause()
        select.value = "personal"   # B -> A (must NOT be swallowed as echo)
        await pilot.pause()
        select.value = "team"       # A -> B again
        await pilot.pause()
        changes = [e.scope for e in app.events if isinstance(e, MCPRail.ScopeChanged)]
        assert changes == ["team", "personal", "team"]
