"""Library keyboard completeness -- critique #8 rows 2-4 (task-32051/2/3).

Three live-confirmed keyboard-only failures on the Library screen, each
pinned here:

- task-32051: Escape could not leave a focused text box. With focus in the
  rail "Search Library…" box or the Search/RAG query box, Escape was a
  no-op (every ``escape`` binding's ``check_action`` was False there) and
  the next printable key was INSERTED AS TEXT -- live, ``i`` landed in the
  search box instead of opening Import.
- task-32052: the New-note canvas opened with nothing focused, so Enter
  and Up/Down did nothing, and the first Tab walked into the top nav bar
  (where Enter switches to Home). Creating the first note was mouse-only.
- task-32053: the Search/RAG evidence cards ARE reachable by Tab (five
  Tabs from the query box -- the critique's "unreachable" reading came
  from the focus cue being a border COLOUR swap and nothing else), but
  that colour-only cue violates the shape-not-colour rule, and the footer
  advertised "enter select evidence" no matter which control had focus --
  including the query box, where Enter runs the search.
"""

from __future__ import annotations

import pytest

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Input

from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LibraryAdaptiveReaderPaneGrip,
)
from Tests.UI.consolidated_css import APP_STYLESHEETS

from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_CREATE_NOTE,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _StaticLibraryRagSearchService,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_library_rag_query_ready,
    _wait_for_library_shell,
    _wait_for_selector,
)

# `border-left: thick <color>` paints a full-block left edge -- the house
# focus convention (task-31983) and a monochrome-visible SHAPE change, not a
# colour swap.
_THICK_LEFT_GLYPH = "█"


def _painted_rows(host) -> list[str]:
    """The real painted frame, one plain-text string per screen row.

    Reads through the compositor (``test_library_row_focus_cue_t31983``'s
    pattern), NOT ``widget.render_lines``: the per-widget styles cache holds
    the strips from the last paint, so a widget rendered while blurred keeps
    returning its blurred border from a direct call even after it takes focus.
    """
    strips = host.screen._compositor.render_strips()
    return ["".join(segment.text for segment in strip) for strip in strips]


def _left_edge(rows: list[str], widget) -> str:
    """The painted cell on the widget's left border, below its top corner."""
    region = widget.region
    line = rows[region.y + 1]
    return line[region.x] if region.x < len(line) else ""


def _rag_results() -> dict:
    return {
        "results": [
            {
                "document_title": "Note Evidence",
                "snippet": "note snippet",
                "source_id": "note-1",
                "provenance": {"source_type": "note"},
            },
            {
                "document_title": "Media Evidence",
                "snippet": "media snippet",
                "source_id": "media-1",
                "provenance": {"source_type": "media"},
            },
        ],
    }


def _build_library_host() -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Research Note", "id": "note-1"}],
        media=_two_media_items(),
    )
    app.library_rag_search_service = _StaticLibraryRagSearchService(_rag_results())
    return LibraryHarness(app)


async def _run_library_rag_search(screen, pilot) -> None:
    """Drive a real Search/RAG run so the evidence cards are mounted."""
    screen.query_one("#library-row-browse-search").press()
    await _wait_for_selector(screen, pilot, "#library-rag-scope-toggle-media")
    screen.query_one("#library-rag-query-input", Input).value = "policy"
    await _wait_for_library_rag_query_ready(screen, pilot, "policy")
    screen.query_one("#library-rag-run-query", Button).press()
    await _wait_for_selector(screen, pilot, "#library-rag-result-card-1")
    await pilot.pause()


# --- task-32051: Escape leaves a focused text box ---------------------------


@pytest.mark.asyncio
async def test_escape_in_the_rail_search_box_hands_focus_to_the_canvas():
    """AC#1/#3: Escape leaves the rail box, and the next key is a canvas key."""
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        search_box = screen.query_one("#library-search-input", Input)
        search_box.focus()
        await pilot.pause()
        assert screen.focused is search_box

        await pilot.press("escape")
        await pilot.pause()
        assert screen.focused is not search_box, (
            "Escape left focus in the rail search box, so the next printable "
            "key is still typed into it."
        )
        assert not isinstance(screen.focused, Input)
        assert search_box.value == "", "Escape inserted text into the search box."

        # AC#3: the next printable key performs its canvas action.
        await pilot.press("i")
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        assert search_box.value == "", "`i` was typed into the rail search box."


@pytest.mark.asyncio
async def test_escape_in_the_search_rag_query_box_hands_focus_to_the_canvas():
    """AC#1: the Search/RAG query box releases Escape the same way."""
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        query_box = screen.query_one("#library-rag-query-input", Input)
        query_box.focus()
        await pilot.pause()
        assert screen.focused is query_box

        await pilot.press("escape")
        await pilot.pause()
        assert screen.focused is not query_box
        assert not isinstance(screen.focused, Input)
        assert query_box.value == ""


@pytest.mark.asyncio
async def test_escape_still_belongs_to_the_surface_that_already_owns_it():
    """The Ingest canvas keeps its own Escape while its path field has focus.

    The blur binding is declared AFTER every surface-back Escape and its
    gate mirrors that order, so it can never steal a surface's own exit --
    the failure mode that would also double the Escape row in F1's help.
    """
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-ingest-import-media").press()
        path = await _wait_for_selector(screen, pilot, "#library-ingest-path")

        path.focus()
        await pilot.pause()
        assert screen.check_action("library_ingest_back", ()) is True
        assert screen.check_action("library_blur_text_field", ()) is False


@pytest.mark.asyncio
async def test_escape_from_a_list_filter_box_still_goes_where_the_footer_says():
    """Fix round 1 (Important #1): the blur binding must not outrank a
    focus-rail hop that genuinely moves focus.

    On a list canvas `library_list_focus_rail` is check_action-True and is
    declared AFTER the blur binding, so an unconditional blur gate silently
    took Escape from the canvas filter box and left the footer's "esc focus
    rail" chip lying about where the caret would land.
    """
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-filter")
        # The real gesture: "/" focuses this canvas's own filter (task-32046)
        # and, being a key, also disarms the list's entry-focus arm -- a bare
        # .focus() races that arm and measures it instead of the key.
        await pilot.press("slash")
        await pilot.pause()
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
        assert screen.focused is screen.query_one("#library-media-filter", Input)

        # What the footer promises BEFORE the press.
        assert (
            "esc",
            "focus rail",
        ) in screen._library_footer_shortcuts_for_current_state()
        assert screen.check_action("library_blur_text_field", ()) is False

        await pilot.press("escape")
        await pilot.pause()
        assert screen.focused is screen.query_one("#library-search-input", Input), (
            "Escape left the filter box for the canvas while the footer said "
            "'focus rail'."
        )


@pytest.mark.asyncio
async def test_escape_still_claimed_when_the_rail_hop_would_be_a_no_op():
    """...but the rail search box itself keeps the blur, on the same canvas.

    This is the whole reason the binding exists: from inside
    `#library-search-input` the focus-rail hop re-focuses the widget that
    already has focus, so Escape was inert and the next key was typed.
    """
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-filter")
        search_box = screen.query_one("#library-search-input", Input)
        # A key first, so the list's entry-focus arm is disarmed (see the
        # sibling test) and the caret genuinely stays in the rail box.
        await pilot.press("slash")
        await pilot.pause()
        search_box.focus()
        await pilot.pause()
        assert screen.focused is search_box

        assert screen.check_action("library_blur_text_field", ()) is True
        await pilot.press("escape")
        await pilot.pause()
        assert screen.focused is not search_box
        assert not isinstance(screen.focused, Input)



# --- task-32052: the New-note canvas is keyboard-complete -------------------


@pytest.mark.asyncio
async def test_new_note_canvas_focuses_blank_note_on_entry_and_arrows_move():
    """AC#1/#2: entry parks focus on Blank note; Down/Up walk the templates."""
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await pilot.press("n")
        blank = await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        for _ in range(60):
            if screen.focused is blank:
                break
            await pilot.pause(0.02)
        assert screen.focused is blank, (
            "Entering the New-note canvas left focus elsewhere, so Enter "
            "cannot create a note."
        )
        assert screen._library_selected_row_id == LIBRARY_ROW_CREATE_NOTE

        await pilot.press("down")
        await pilot.pause()
        first_template = screen.query_one("#library-notes-template-0", Button)
        assert screen.focused is first_template

        await pilot.press("up")
        await pilot.pause()
        assert screen.focused is blank

        # AC#1 is "Enter creates a note immediately" -- assert the outcome,
        # not just the focus that makes it possible.
        await pilot.press("enter")
        editor = await _wait_for_selector(screen, pilot, "#library-note-body")
        assert editor is not None
        assert screen._library_notes_view == "editor"
        assert screen._library_note_session.snapshot is not None


@pytest.mark.asyncio
async def test_tab_from_a_library_canvas_never_reaches_the_nav_bar():
    """AC#3: Tab cycles inside the Library screen content, not the app chrome."""
    from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar

    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await pilot.press("n")
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        await pilot.pause()

        nav_bar = screen.query_one(MainNavigationBar)
        seen = []
        for _ in range(30):
            await pilot.press("tab")
            await pilot.pause()
            focused = screen.focused
            assert focused is not None
            seen.append(focused.id)
            assert nav_bar not in focused.ancestors, (
                f"Tab escaped the Library screen into the nav bar at {focused.id!r}; "
                f"walk so far: {seen}"
            )


@pytest.mark.asyncio
async def test_new_note_footer_enter_hint_follows_the_focused_control():
    """AC#4: "enter create note" only while a create row genuinely has focus."""
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await pilot.press("n")
        blank = await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        blank.focus()
        await pilot.pause()
        assert ("enter", "create note") in screen._library_notes_footer_shortcuts()

        screen.query_one("#library-notes-create-back", Button).focus()
        await pilot.pause()
        labels = dict(screen._library_notes_footer_shortcuts())
        assert labels.get("enter") != "create note", (
            "The footer promised Enter creates a note while the Back button "
            "had focus, where Enter goes back."
        )


@pytest.mark.asyncio
async def test_ctrl_n_into_new_note_also_focuses_blank_note():
    """AC#1 covers *entering the canvas*, not one route into it.

    Ctrl+N from the Notes list takes the retained-shell route
    (``_try_switch_retained_library_notes_route``), which is a different
    code path from the landing's ``n``; live, it left focus behind.
    """
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-filter")
        await pilot.pause()
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES

        await pilot.press("ctrl+n")
        blank = await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        for _ in range(80):
            if screen.focused is blank:
                break
            await pilot.pause(0.02)
        assert screen.focused is blank, (
            f"Ctrl+N left focus on {getattr(screen.focused, 'id', None)!r}, "
            "so Enter does not create a note."
        )


# --- task-32053: keyboard-visible Search/RAG evidence -----------------------


@pytest.mark.asyncio
async def test_evidence_card_focus_is_a_shape_change_not_a_colour_swap():
    """AC#1: the focused card carries the house left-edge block cue."""
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _run_library_rag_search(screen, pilot)

        card = screen.query_one("#library-rag-result-card-0")
        other = screen.query_one("#library-rag-result-card-1")

        assert _left_edge(_painted_rows(host), card) != _THICK_LEFT_GLYPH

        card.focus()
        await pilot.pause()
        rows = _painted_rows(host)
        assert _left_edge(rows, card) == _THICK_LEFT_GLYPH, (
            "The focused evidence card only changed border COLOUR -- there is "
            "no shape cue telling a keyboard user where they are."
        )
        assert _left_edge(rows, other) != _THICK_LEFT_GLYPH, (
            "The blurred sibling card paints the focus cue too."
        )


@pytest.mark.asyncio
async def test_evidence_cards_are_reachable_and_enter_selects_them():
    """AC#2: Tab reaches a card and Enter selects it, matching the footer."""
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _run_library_rag_search(screen, pilot)

        screen.query_one("#library-rag-query-input", Input).focus()
        await pilot.pause()
        for _ in range(6):
            await pilot.press("tab")
            await pilot.pause()
            focused_id = getattr(screen.focused, "id", "") or ""
            if focused_id.startswith("library-rag-result-card-"):
                break
        else:
            raise AssertionError("Six Tabs from the query box reached no card.")

        await pilot.press("enter")
        await pilot.pause(0.2)
        assert (
            str(screen.query_one("#library-rag-select-result-0", Button).label)
            == "Selected evidence"
        )


@pytest.mark.asyncio
async def test_search_rag_footer_names_the_focused_controls_enter_action():
    """AC#3: "enter run search" in the query box, "enter toggle Notes" on a
    source toggle, "enter select evidence" on a card."""
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _run_library_rag_search(screen, pilot)

        screen.query_one("#library-rag-query-input", Input).focus()
        await pilot.pause()
        assert ("enter", "run search") in screen._library_footer_shortcuts_for_current_state()

        screen.query_one("#library-rag-scope-toggle-notes", Button).focus()
        await pilot.pause()
        assert (
            "enter",
            "toggle Notes",
        ) in screen._library_footer_shortcuts_for_current_state()

        screen.query_one("#library-rag-result-card-0").focus()
        await pilot.pause()
        assert (
            "enter",
            "select evidence",
        ) in screen._library_footer_shortcuts_for_current_state()


@pytest.mark.asyncio
async def test_run_button_has_a_visible_focus_state():
    """AC#4: focusing Run changes the painted rows, not just a colour token."""
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-run-query")
        screen.query_one("#library-rag-query-input", Input).value = "policy"
        await _wait_for_library_rag_query_ready(screen, pilot, "policy")

        run = screen.query_one("#library-rag-run-query", Button)
        blurred = _painted_rows(host)[run.region.y][
            run.region.x : run.region.right
        ]

        run.focus()
        await pilot.pause()
        focused = _painted_rows(host)[run.region.y][run.region.x : run.region.right]
        assert focused != blurred, (
            "The Run button paints identically focused and blurred -- a "
            "keyboard user cannot see where they are."
        )


@pytest.mark.asyncio
async def test_tab_stays_inside_library_in_the_emergency_return_state():
    """Fix round 1 (Important #2): the narrow-terminal Tab path was unscoped.

    `on_key`'s `emergency_tab` branch handles Tab itself and stops the event,
    so `action_focus_next` never runs -- and it called the bare
    `Screen.focus_next()`, whose default `"*"` selector walks the nav bar.
    AC#3 says ANY Library canvas, which includes this one.
    """
    from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar

    host = _build_library_host()
    async with host.run_test(size=(60, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        screen.query_one("#library-rag-query-input", Input).focus()
        await pilot.pause()
        # The narrow-width gate arms this on its own; give it a beat and
        # fall back to the same two fields it sets if the resize watcher has
        # not run yet (the state, not the route to it, is what is under test).
        for _ in range(40):
            if screen._library_emergency_stage == "canvas-only":
                break
            await pilot.pause(0.02)
        if screen._library_emergency_stage != "canvas-only":
            screen._library_emergency_stage = "canvas-only"
            screen._library_emergency_restore_receipt = (
                screen._capture_library_emergency_restore_receipt(
                    screen.query_one("#library-rail"),
                    screen.query_one("#library-canvas"),
                )
            )
            screen._apply_library_notes_stage_visibility()
            await pilot.pause()
        assert screen._library_emergency_stage == "canvas-only"
        assert screen._library_emergency_restore_receipt is not None

        nav_bar = screen.query_one(MainNavigationBar)
        seen = []
        for _ in range(20):
            await pilot.press("tab")
            await pilot.pause()
            focused = screen.focused
            assert focused is not None
            seen.append(focused.id)
            assert nav_bar not in focused.ancestors, (
                f"emergency Tab escaped into the nav bar at {focused.id!r}; "
                f"walk so far: {seen}"
            )


class _GripFocusHost(App):
    """Two real pane grips inside a real shell container, production CSS.

    The `:focus` rule is scoped `.library-adaptive-reader-shell > .library-
    adaptive-reader-pane-grip:focus`, so the parent class is load-bearing.
    `AUTO_FOCUS = None` keeps both grips genuinely blurred until a test moves
    focus, so the observed treatment is the one a real Tab produces.
    """

    AUTO_FOCUS = None
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        with Horizontal(classes="library-adaptive-reader-shell"):
            yield LibraryAdaptiveReaderPaneGrip(
                "library", open=True, pane_label="Library", id="grip-a", width=1
            )
            yield LibraryAdaptiveReaderPaneGrip(
                "items", open=True, pane_label="Items", id="grip-b", width=1
            )


@pytest.mark.asyncio
async def test_pane_grip_focus_is_visible_against_its_blurred_sibling():
    """Controller ruling (B): the grip's focus treatment must actually show.

    `.library-adaptive-reader-pane-grip:focus` was background + colour + bold
    with no inversion, which is the "focus invisible on grips" critique #8
    measured. `reverse` makes the grip invert as a block; read through the
    real compositor, the focused grip's painted style must differ from its
    blurred sibling's.
    """
    app = _GripFocusHost()
    async with app.run_test(size=(80, 12)) as pilot:
        grip_a = app.query_one("#grip-a", LibraryAdaptiveReaderPaneGrip)
        grip_b = app.query_one("#grip-b", LibraryAdaptiveReaderPaneGrip)

        def _style_at(widget):
            region = widget.region
            y = region.y + region.height // 2
            column = 0
            for segment in app.screen._compositor.render_strips()[y]:
                for _char in segment.text:
                    if column == region.x:
                        return segment.style
                    column += 1
            return None

        blurred_both = (_style_at(grip_a), _style_at(grip_b))
        assert blurred_both[0] == blurred_both[1], (
            "both grips must start identically painted for this to mean anything"
        )

        grip_a.focus()
        await pilot.pause()
        assert grip_a.has_focus and not grip_b.has_focus
        assert _style_at(grip_a) != _style_at(grip_b), (
            "the focused grip paints identically to its blurred sibling -- "
            "keyboard focus on a pane grip is invisible."
        )
        assert "reverse" in str(grip_a.styles.text_style).lower(), (
            f"expected a reverse (block-inverting) focus style, got "
            f"{grip_a.styles.text_style!r}"
        )
