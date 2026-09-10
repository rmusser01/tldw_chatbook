"""Library rail fixes from critique #9 (tasks 32212, 32220, 32226, 32230, 32219).

Every geometry/paint assertion here runs against the PRODUCTION stylesheet
sequence (``LibraryProductionCSSHarness`` == ``TldwCli.CSS_PATH``), because the
rail's rules live in the ``screen_agentic_library.tcss`` split sheet -- a
bundle-only or widget-only harness sees none of them and would happily pass a
layout that is broken in the app.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, Static

from tldw_chatbook.Library.library_rail_state import LibraryRailPreferences
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SEARCH
from tldw_chatbook.Library.library_shell_state import LibraryShellState
from tldw_chatbook.UI.Library_Modules.library_rag_search_controller import (
    LibraryRagSearchController,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_rail import LibraryRail
from tldw_chatbook.Workspaces.display_state import (
    LibraryWorkspaceDepthState,
    LibraryWorkspaceSourceRow,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _library_host() -> LibraryProductionCSSHarness:
    """A Library screen under the exact production stylesheet sequence."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    return LibraryProductionCSSHarness(app)


def _painted(host, region) -> str:
    """The painted text inside ``region``, one screen row per list entry."""
    strips = host.screen._compositor.render_strips()
    lines = ["".join(segment.text for segment in strip) for strip in strips]
    out = []
    for y in range(region.y, region.bottom):
        if 0 <= y < len(lines):
            out.append(lines[y][region.x : region.right])
    return "\n".join(out)


# --- task-32212: the rail search row must fit its pane ---------------------


#: The rail pane's own right border glyph (``border: solid $ds-column-line``).
_RAIL_FRAME_GLYPH = "│"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (100, 30), (60, 24)])
async def test_the_rail_search_row_never_pushes_the_canvas_frame(size) -> None:
    """AC#1/#2: the Input + clear button stay inside the rail's own width and
    every painted search-row line keeps the rail's frame column.

    The boxes were always laid out inside the rail (the regions below passed
    on dev); what did NOT fit was the clear button's own CONTENT -- see the
    painted-frame assertion, which is the one that reproduced the defect.
    """
    host = _library_host()
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        rail = screen.query_one("#library-rail")
        row = screen.query_one("#library-rail-search-row")
        box = screen.query_one("#library-search-input")
        clear = screen.query_one("#library-search-clear")
        assert row.region.right <= rail.region.right, (row.region, rail.region)
        assert clear.region.right <= rail.region.right, (clear.region, rail.region)
        assert box.region.right <= clear.region.x, (box.region, clear.region)
        canvas = screen.query_one("#library-canvas")
        if canvas.display:
            # Below the compact breakpoint the rail is single-stage and the
            # canvas is deliberately not mounted beside it (task-32066).
            assert canvas.region.x >= rail.region.right, (canvas.region, rail.region)

        # AC#2, the real pin: the painted frame. Every line the search row
        # covers must still carry the rail's right border in the rail's own
        # last column -- the regression painted the middle line two cells
        # long, shifting that border (and the canvas's left border) right.
        strips = host.screen._compositor.render_strips()
        lines = ["".join(segment.text for segment in strip) for strip in strips]
        frame_column = rail.region.right - 1
        for y in range(row.region.y, row.region.bottom):
            painted = lines[y]
            assert painted[frame_column] == _RAIL_FRAME_GLYPH, (
                f"search-row line y={y} lost the rail frame at column "
                f"{frame_column} at size {size}: {painted[:frame_column + 4]!r}"
            )


# --- task-32220: the rail heading is never cut mid-word --------------------


@pytest.mark.asyncio
async def test_the_rail_heading_is_never_cut_mid_word() -> None:
    """AC#1: at the compact rail width the heading ellipsises, never clips."""
    host = _library_host()
    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        painted = _painted(host, screen.query_one("#library-rail-heading").region)
        assert "Navigati" not in painted or "Navigation" in painted, painted
        assert "Navigat…" in painted or "Navigation" in painted, painted


# --- task-32226: unsubmitted rail text stays on its own canvas -------------


@pytest.mark.asyncio
async def test_unsubmitted_rail_text_never_seeds_the_rag_query_box() -> None:
    """AC#1: text typed into the rail box on a browse canvas is never
    committed to the Search/RAG query state, so it cannot reappear in the
    Search/RAG query box on the next visit."""
    host = _library_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-filter")
        box = screen.query_one("#library-search-input", Input)
        box.focus()
        await pilot.press(*"draft")  # typed, never submitted
        await pilot.pause()
        assert box.value == "draft", "the keystrokes stay in the widget"
        screen.query_one("#library-row-browse-search").press()
        query_box = await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        assert query_box.value == "", query_box.value
        assert screen._rag_search_state.query == "", screen._rag_search_state.query


# --- task-32230 AC#1: one DB source per Details line -----------------------


@pytest.mark.asyncio
async def test_details_db_sizes_render_one_source_per_line() -> None:
    """AC#1: at the compact rail WIDTH each DB size occupies exactly one
    painted row, so no value is split across lines. The terminal is tall so
    the Details group sits above the rail's fold -- the wrap this pins is a
    width problem, and a clipped row paints nothing to read."""
    host = _library_host()
    sizes = {"prompts": "180.0 KB", "chachanotes": "1.1 MB", "media": "508.0 KB"}
    host.app_instance.db_sizes_status = dict(sizes)

    class _StubManager:
        """Keep the seeded reading; the real manager stats a fixture profile."""

        async def update_db_sizes(self) -> None:
            host.app_instance.db_sizes_status = dict(sizes)

    host.app_instance.db_status_manager = _StubManager()

    async with host.run_test(size=(100, 60)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen._set_library_rail_section("details", True)
        await _wait_for_selector(screen, pilot, "#library-details-db-sizes")
        await pilot.pause()

        rows = [
            widget
            for widget in screen.query(".library-details-row")
            if str(widget.id or "").startswith("library-details-db-sizes")
        ]
        assert [widget.id for widget in rows] == [
            "library-details-db-sizes-label",
            "library-details-db-sizes",
            "library-details-db-sizes-1",
            "library-details-db-sizes-2",
        ]
        painted = "\n".join(_painted(host, widget.region) for widget in rows)
        for needle in ("DB sizes", "180.0KB", "1.1MB", "508.0KB"):
            assert needle in painted, (needle, painted)
        for widget in rows:
            assert widget.region.height == 1, (widget.id, widget.region, painted)


# --- task-32230 AC#2: the Handoff row names the blocker and the remedy -----


def _blocked_source_row(**overrides) -> LibraryWorkspaceSourceRow:
    values = {
        "item_type": "conversation",
        "item_id": "chat-a",
        "title": "Alpha planning",
        "workspace_ids": (),
        "workspace_label": "Unscoped",
        "visible": True,
        "active_context_eligible": False,
        "authority_label": "",
        "context_label": "",
        "recovery_copy": (
            "Copy or link this conversation into workspace w-1 before using "
            "it in Console."
        ),
        "reason_code": "not_in_active_workspace",
    }
    values.update(overrides)
    return LibraryWorkspaceSourceRow(**values)


def _depth_state(handoff_label: str, rows) -> LibraryWorkspaceDepthState:
    return LibraryWorkspaceDepthState(
        heading="Workspaces",
        workspace_label="Workspace: Local Default",
        workspace_name="Local Default",
        visibility_label="",
        handoff_label=handoff_label,
        context_handoff_enabled=False,
        context_handoff_tooltip="",
        source_authority_label="",
        collections_membership_label="",
        import_export_label="",
        source_rows=tuple(rows),
    )


def test_the_handoff_row_names_the_blocker_and_its_remedy() -> None:
    """AC#2: the count alone was unactionable -- the row now carries the
    reason and the next step in the house `reason · next step` grammar."""
    state = _depth_state(
        "Console/RAG handoff: 0 eligible, 1 blocked", [_blocked_source_row()]
    )
    label = LibraryScreen._workspace_handoff_summary_label(None, state)
    assert (
        "1 blocked · not in this workspace · Link it from the conversation's header"
        in label
    ), label
    assert label == (
        "0 eligible · 1 blocked · not in this workspace · "
        "Link it from the conversation's header"
    ), label
    assert "●" not in label, label


def test_the_handoff_row_stays_a_bare_count_when_nothing_is_blocked() -> None:
    """AC#2: the unblocked case keeps its short form and grows no dot."""
    state = _depth_state("Console/RAG handoff: 0 eligible", [])
    assert LibraryScreen._workspace_handoff_summary_label(None, state) == "0 eligible"


def test_the_handoff_row_falls_back_to_the_rule_s_own_recovery_copy() -> None:
    """A block linking cannot resolve still names a reason and a next step."""
    state = _depth_state(
        "Console/RAG handoff: 0 eligible, 1 blocked",
        [
            _blocked_source_row(
                reason_code="no_active_workspace",
                recovery_copy=(
                    "Select an active workspace before using this item in Console."
                ),
            )
        ],
    )
    label = LibraryScreen._workspace_handoff_summary_label(None, state)
    assert label == (
        "0 eligible · 1 blocked · blocked for this workspace · "
        "Select an active workspace before using this item in Console"
    ), label


def test_the_handoff_row_generalises_across_a_mixed_blocked_set() -> None:
    """Two blocked items of different types share one reason: the remedy
    stays truthful without naming a type it cannot pick."""
    state = _depth_state(
        "Console/RAG handoff: 2 eligible, 2 blocked",
        [
            _blocked_source_row(
                item_type="note", item_id="note-cross", reason_code="cross_workspace"
            ),
            _blocked_source_row(
                item_type="conversation",
                item_id="chat-cross",
                reason_code="cross_workspace",
            ),
        ],
    )
    assert LibraryScreen._workspace_handoff_summary_label(None, state) == (
        "2 eligible · 2 blocked · in another workspace · "
        "Copy or link them into this workspace"
    )


# --- task-32219 AC#2: the rail says when Details runs past the fold --------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "expected"), [((235, 52), True), ((235, 90), False)]
)
async def test_the_rail_says_when_details_runs_past_the_fold(size, expected) -> None:
    """AC#2: at 52 rows the Details ▸ Actions group is below the fold with no
    cue; the rail now says so on its own last line, and stops saying it as
    soon as the whole rail fits."""
    host = _library_host()
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen._set_library_rail_section("details", True)
        await _wait_for_selector(screen, pilot, "#library-rail-fold-cue")
        for _ in range(20):
            await pilot.pause()
        cue = screen.query_one("#library-rail-fold-cue", Static)
        rail = screen.query_one("#library-rail")
        assert cue.display is expected, (size, rail.max_scroll_y, cue.display)
        if expected:
            assert "scroll for more" in str(cue.renderable)
            assert cue.region.height == 1, cue.region
            # The cue has to be ON SCREEN inside the rail to be a cue at
            # all: the first version sat at the end of the scrollable
            # content, i.e. below the very fold it was describing (caught
            # live at 235x52, not by this test's earlier `display` check).
            assert rail.region.contains_region(cue.region), (
                cue.region,
                rail.region,
            )
            assert "scroll for more" in _painted(host, cue.region)


# --- task-32219 AC#2: the two fold-cue defects only the live app showed ----
#
# Both of these were found by running the app, not by this suite, and the
# harness settles correctly either way -- so without these two pins, deleting
# the production code that fixes them leaves every other test in this file
# green. That is exactly the shape of fix that gets "simplified" away later.


def _bare_rail() -> LibraryRail:
    """A LibraryRail with no app behind it -- enough for the pure paths."""
    return LibraryRail(
        LibraryShellState(
            header_line="Library | Test",
            sections=(),
            details_lines=("Local", "Notes 0 · Media 0 · Conversations 0"),
            selected_row_id="",
            canvas_kind="empty",
            canvas_target="",
            canvas_empty_copy="",
        ),
        LibraryRailPreferences(details_open=True),
    )


def test_a_recomposed_fold_cue_keeps_the_visibility_the_rail_decided() -> None:
    """The rail recomposes on every count/evidence/route change. A cue that
    rebuilt itself hidden each time was reset faster than the post-layout
    measurement could turn it on, and live it never appeared at all -- so the
    decision has to be seeded from the RAIL, not defaulted on the widget."""
    rail = _bare_rail()
    assert rail._fold_cue_visible is False
    assert rail._build_fold_cue().display is False

    rail._fold_cue_visible = True
    assert rail._build_fold_cue().display is True, (
        "a recompose must carry the visibility the last measurement decided"
    )


def test_the_fold_measurement_is_deferred_past_the_layout_that_triggered_it() -> None:
    """``max_scroll_y`` reports the PREVIOUS layout inside ``on_mount`` and
    inside the ``virtual_size`` watcher, so measuring inline reads zero
    overflow for a rail that is about to overflow. The schedule must hand
    ``_sync_fold_cue`` to ``call_after_refresh``, never call it itself."""
    rail = _bare_rail()
    deferred: list[object] = []
    measured: list[int] = []
    rail.call_after_refresh = deferred.append  # type: ignore[method-assign]
    rail._sync_fold_cue = lambda: measured.append(1)  # type: ignore[method-assign]

    rail._running = True  # `is_running` gates the schedule
    rail._schedule_fold_cue_sync()

    assert measured == [], "the fold must not be measured inline"
    assert deferred == [rail._sync_fold_cue], deferred

    # A rail that is not running schedules nothing at all.
    deferred.clear()
    rail._running = False
    rail._schedule_fold_cue_sync()
    assert deferred == []


def test_only_a_conversation_block_is_sent_to_a_reader_header() -> None:
    """PR #2581 review (Qodo 3): "Link it from the <type>'s header" named a
    control that exists on exactly one reader. `library_conversation_reader`
    is the only widget in the repo that builds a "Link to workspace" button,
    so a blocked note or media item was told to press something that is not
    there. Those get the remedy their UI actually supports."""
    for item_type in ("note", "media"):
        state = _depth_state(
            "Console/RAG handoff: 0 eligible, 1 blocked",
            [_blocked_source_row(item_type=item_type, item_id=f"{item_type}-a")],
        )
        label = LibraryScreen._workspace_handoff_summary_label(None, state)
        assert "header" not in label, label
        assert label.endswith("Copy or link it into this workspace"), label

    conversation = _depth_state(
        "Console/RAG handoff: 0 eligible, 1 blocked",
        [_blocked_source_row(item_type="conversation")],
    )
    assert LibraryScreen._workspace_handoff_summary_label(
        None, conversation
    ).endswith("Link it from the conversation's header")


def test_a_mixed_but_wholly_linkable_block_keeps_a_linking_remedy() -> None:
    """PR #2581 review (Qodo 4): `not_in_active_workspace` and
    `cross_workspace` are BOTH link-resolvable, but a set holding one of
    each fell into the non-linkable fallback and printed the first row's
    singular recovery sentence as the remedy for all of them."""
    state = _depth_state(
        "Console/RAG handoff: 0 eligible, 2 blocked",
        [
            _blocked_source_row(
                item_id="chat-a", reason_code="not_in_active_workspace"
            ),
            _blocked_source_row(item_id="chat-b", reason_code="cross_workspace"),
        ],
    )
    label = LibraryScreen._workspace_handoff_summary_label(None, state)
    # The reason cannot claim either single code for the whole set, but the
    # remedy is still the real one: every row here can be linked.
    assert label == (
        "0 eligible · 2 blocked · blocked for this workspace · "
        "Link them from the conversation's header"
    ), label
    assert "Copy or link this conversation into workspace" not in label


# --- task-32226: the guard as a pure state transition -----------------------


class _RecordingSearchState:
    """The two fields `handle_library_search_changed` may write."""

    def __init__(self) -> None:
        self.query = "submitted"


def _search_controller(selected_row_id: str) -> LibraryRagSearchController:
    """A controller wired to nothing but the two seams the handler reads."""
    controller = object.__new__(LibraryRagSearchController)
    state = _RecordingSearchState()
    controller._rag_search_state_accessor = lambda: state
    controller._library_selected_row_id_accessor = lambda: selected_row_id
    mirrored: list[tuple[str, str]] = []
    # The controller exposes its injected seams as read-only properties over
    # `<name>_fn`, so the injection point is the backing attribute.
    controller._patch_sibling_library_search_input_fn = (
        lambda selector, value: mirrored.append((selector, value))
    )
    controller.mirrored = mirrored  # type: ignore[attr-defined]
    controller.state = state  # type: ignore[attr-defined]
    return controller


def test_the_rail_search_guard_is_a_pure_state_transition() -> None:
    """PR #2581 review (Qodo 1): the off-canvas guard had only production-
    harness coverage, so a defect in the state rule could not be told apart
    from navigation or composition. Driven here with no app at all."""
    stopped: list[bool] = []

    class _Event:
        value = "draft"

        def stop(self) -> None:
            stopped.append(True)

    off_row = _search_controller("browse-media")
    off_row.handle_library_search_changed(_Event())
    assert stopped == [True], "the event is always consumed"
    assert off_row.state.query == "submitted", "off-row keystrokes stay local"
    assert off_row.mirrored == [], "and never reach the Search/RAG box"

    on_row = _search_controller(LIBRARY_ROW_BROWSE_SEARCH)
    on_row.handle_library_search_changed(_Event())
    assert on_row.state.query == "draft", "on the Search/RAG row they commit"
    assert on_row.mirrored == [("#library-rag-query-input", "draft")]
