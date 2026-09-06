"""task-3302 (MI-03/04/05): Ingest mode keyboard & focus contracts.

Three live-confirmed keyboard-first failures on the Library screen's
Ingest mode, each pinned here:

- MI-03 entry focus: activating Ingest left focus wherever it was (the
  rail search box in the live walk), so typing a path ran a Library
  search. Entry must park the caret in ``#library-ingest-path``.
- MI-04 keyboard exit + advertised keys: Esc did nothing in Ingest
  (every ``escape`` binding was gated to other surfaces), ``i`` only
  worked from the landing, and the footer/F1 showed only the generic
  hints. Ingest now has its own per-mode shortcut set shared by the
  footer AND F1 (``LIBRARY_INGEST_SHORTCUTS`` -- the same shared-source
  rule task-2858 established), Esc returns to the hub landing, and ``i``
  enters Ingest from any Library canvas.
- MI-05 focus visibility: the top-level ingest fields
  (``.library-ingest-field``) and the compact ``.library-canvas-action``
  buttons showed a color-only focus change -- byte-identical plain-text
  panes across Tab in two independent live walks. Focus must now be
  glyph-level (structural), with no dimensional change on focus/blur.

The render assertions run under ``LibraryHarness``, which loads the REAL
app stylesheet bundle (the shared UI harnesses do not -- the established
CSS-true pattern, see ``_CssTrueConsoleHarness`` in
``test_console_composer_overflow.py``).
"""

import pytest
from textual.geometry import Region
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_INGEST_MEDIA,
)
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

# The heavy border/outline glyph family (Textual's "heavy" box set). Any of
# these appearing on focus is a structural, monochrome-visible cue; none of
# them appear in the unfocused tall-border rendering ("▊", "▔", "▎", "▁").
HEAVY_GLYPHS = ("┏", "┓", "┗", "┛", "━", "┃")


def _plain_rows(widget) -> list[str]:
    """The widget's fully painted rows (border/outline included) as text.

    ``render_lines`` goes through Textual's ``StylesCache``, which is where
    border and outline glyphs are applied -- ``render_line`` (singular)
    returns content-only strips and would miss exactly the treatment under
    test here.
    """
    width, height = widget.size.width, widget.size.height
    return [
        strip.text for strip in widget.render_lines(Region(0, 0, width, height))
    ]


async def _enter_ingest_mode(screen, pilot):
    """Drive the shared rail-row seam into Ingest and settle the recompose."""
    await screen._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)
    path_input = await _wait_for_selector(screen, pilot, "#library-ingest-path")
    await pilot.pause()
    return path_input


# --- AC#1: entry focus -------------------------------------------------------


@pytest.mark.asyncio
async def test_entering_ingest_mode_focuses_the_path_field_and_typing_edits_it():
    """MI-03: pressing ``i`` on the landing lands the caret in the path
    field, so the next keystrokes build a path -- they must NOT reach the
    rail search box (the live failure: typing a path ran a Library
    search)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await pilot.press("i")
        path_input = await _wait_for_selector(
            screen, pilot, "#library-ingest-path"
        )
        await _wait_for_condition(
            pilot,
            lambda: path_input.has_focus,
            message="entering Ingest never focused #library-ingest-path",
        )

        await pilot.press("h", "i")
        await pilot.pause()
        assert path_input.value == "hi"
        assert screen.query_one("#library-search-input", Input).value == ""


@pytest.mark.asyncio
async def test_rail_row_entry_into_ingest_focuses_the_path_field_too():
    """MI-03: the rail-row/seam route (every in-app route into Ingest
    funnels through ``_select_library_rail_row``) parks focus on the path
    field just like the accelerator."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        path_input = await _enter_ingest_mode(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: path_input.has_focus,
            message="rail-row entry into Ingest never focused the path field",
        )


@pytest.mark.asyncio
async def test_narrow_ingest_collapses_rail_and_keeps_source_contract_visible():
    """TASK-15702: 80-column entry spends width on the form, not the rail."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        path_input = await _enter_ingest_mode(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: path_input.has_focus,
            message="narrow Ingest never focused its path",
        )
        assert screen.query_one("#library-rail").display is False, (
            f"screen={screen.size.width}, shell="
            f"{screen.query_one('#library-shell-grid').region.width}, "
            f"collapsed={screen._library_rail_collapsed}, "
            f"auto={screen._ingest_state.auto_collapsed_rail}"
        )
        assert screen.query_one("#library-rail-handle").display is True
        canvas = screen.query_one("#library-ingest-canvas")
        for selector in ("#library-ingest-header", "#library-ingest-path-label"):
            widget = screen.query_one(selector, Static)
            assert canvas.region.y <= widget.region.y < canvas.region.bottom


# --- AC#2: Esc exits, `i` enters from any canvas -----------------------------


@pytest.mark.asyncio
async def test_escape_in_ingest_returns_to_the_library_hub_landing():
    """MI-04: Esc from Ingest (including from inside the path field, which
    entry focus just landed us in) returns to the hub landing -- previously
    every ``escape`` binding was gated to other surfaces and the key did
    nothing here."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _enter_ingest_mode(screen, pilot)
        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA

        await pilot.press("escape")
        await _wait_for_selector(screen, pilot, "#library-hub-action-import")
        assert screen._library_selected_row_id == ""

        # Keyboard continuity: the hub's first action holds focus, so the
        # landing accelerators (`i`/`n`) and Enter keep working -- Esc must
        # never strand focus nowhere (that is MI-03's failure mode again).
        hub_import = screen.query_one("#library-hub-action-import", Button)
        await _wait_for_condition(
            pilot,
            lambda: hub_import.has_focus,
            message="Esc-from-Ingest never focused the hub's first action",
        )


@pytest.mark.asyncio
async def test_i_enters_ingest_from_a_non_landing_library_canvas():
    """MI-04: ``i`` was landing-scoped, so there was no keyboard route into
    Ingest from any other canvas. It now enters Ingest from anywhere on the
    Library screen (still guarded: never while an Input/TextArea owns
    focus)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversations-filter")
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS

        await pilot.press("i")
        path_input = await _wait_for_selector(
            screen, pilot, "#library-ingest-path"
        )
        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA
        await _wait_for_condition(
            pilot,
            lambda: path_input.has_focus,
            message="`i` from a browse canvas never focused the path field",
        )


@pytest.mark.asyncio
async def test_i_still_types_literally_inside_text_fields_off_the_landing():
    """MI-04 guard: widening ``i`` beyond the landing must not let it steal
    focus out of a text field -- inside an Input it stays a literal
    keystroke (the F-012 `/` guard pattern)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversations-filter")
        filter_input = screen.query_one("#library-conversations-filter", Input)
        filter_input.focus()
        await pilot.pause()

        await pilot.press("i")
        await pilot.pause()
        assert filter_input.value == "i"
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS


# --- AC#3: footer and F1 share the ingest shortcut set -----------------------


@pytest.mark.asyncio
async def test_ingest_footer_advertises_enter_start_and_esc_back():
    """MI-04: the Ingest footer taught only the generic `/`+F6 hints; it now
    carries the mode's own keys (Enter starts the import, Esc goes back)
    from the shared per-mode set."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _enter_ingest_mode(screen, pilot)

        # Substring assertions (the `u`-hint test's style) rather than exact
        # equality: the footer appends its always-present global cluster and
        # filters reserved keys (F6) out of the context portion, and that
        # rendering evolves independently of this mode's registered set.
        footer = screen.query_one(AppFooterStatus)
        assert "enter start" in footer.shortcut_text
        assert "esc back" in footer.shortcut_text


def test_f1_help_in_ingest_mode_shows_the_same_shared_ingest_set(monkeypatch):
    """MI-04: F1 in Ingest listed the screen's raw Skills BINDINGS (the
    BINDINGS-only-sourcing trap task-2858 closed for other modes). F1 now
    reads the exact same per-mode set the footer registers --
    ``_library_footer_shortcuts_for_current_state`` -- so the two can never
    disagree. Mirrors ``test_action_show_workbench_help_includes_landing_
    footer_keys`` (test_screen_navigation.py)."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_INGEST_MEDIA

    # The footer and F1 read the same state-derived set. Retry is omitted
    # until a settled last submission exists.
    assert (
        screen._library_footer_shortcuts_for_current_state()
        == screen.LIBRARY_INGEST_SHORTCUTS
    )

    pushed = []

    class _FakeApp:
        def push_screen(self, panel):
            pushed.append(panel)

    monkeypatch.setattr(
        LibraryScreen, "app", property(lambda self: _FakeApp()), raising=False
    )

    screen.action_show_workbench_help()

    assert len(pushed) == 1
    panel = pushed[0]
    assert isinstance(panel, WorkbenchHelpPanel)
    shortcuts = list(panel.state.shortcuts)
    # The shared per-mode set leads the panel, exactly as registered.
    assert shortcuts[: len(screen.LIBRARY_INGEST_SHORTCUTS)] == list(
        screen.LIBRARY_INGEST_SHORTCUTS
    )
    keys = {key for key, _description in shortcuts}
    assert "enter" in keys
    assert "esc" in keys
    descriptions = {description for _key, description in shortcuts}
    # The Skills-editor contamination the original finding reproduced.
    assert "Save skill" not in descriptions
    assert "Back to skills list" not in descriptions


def test_check_action_gates_ingest_back_to_the_ingest_canvas():
    """The new ``escape`` binding must be inert everywhere but Ingest --
    the same disjoint-gate contract every other escape binding on this
    screen follows (and ``test_library_screen_bindings_are_all_gated_or_
    universal`` audits on the bare landing)."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)

    # Landing -- inactive.
    assert screen.check_action("library_ingest_back", ()) is False

    # A different canvas -- inactive.
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
    assert screen.check_action("library_ingest_back", ()) is False

    # Ingest -- active.
    screen._library_selected_row_id = LIBRARY_ROW_INGEST_MEDIA
    assert screen.check_action("library_ingest_back", ()) is True


# --- AC#4/#5: glyph-level focus, dimensionally stable -------------------------


@pytest.mark.asyncio
async def test_ingest_field_focus_is_glyph_level_under_the_real_stylesheet():
    """MI-05: Tab onto the path field produced a byte-identical plain-text
    pane (the tall border only swapped color). Focus must now change the
    painted GLYPHS -- the heavy edge family -- while the widget's region
    stays identical (no focus/blur layout shift)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        path_input = await _enter_ingest_mode(screen, pilot)

        # Park focus elsewhere first so the "unfocused" capture is honest.
        screen.query_one("#library-search-input", Input).focus()
        await pilot.pause()
        assert not path_input.has_focus
        region_before = path_input.region
        unfocused_rows = _plain_rows(path_input)
        assert not any(
            glyph in row for row in unfocused_rows for glyph in HEAVY_GLYPHS
        ), f"unfocused field already paints heavy glyphs: {unfocused_rows!r}"

        path_input.focus()
        await pilot.pause()
        assert path_input.has_focus
        focused_rows = _plain_rows(path_input)

        assert focused_rows != unfocused_rows, (
            "focus produced a byte-identical plain-text pane -- the "
            "color-only regression MI-05 pinned"
        )
        assert any(
            glyph in row for row in focused_rows for glyph in HEAVY_GLYPHS
        ), f"focused field shows no structural (heavy-edge) cue: {focused_rows!r}"
        # AC#5: dimensional stability -- same region, same painted row count.
        assert path_input.region == region_before
        assert len(focused_rows) == len(unfocused_rows)


@pytest.mark.asyncio
async def test_canvas_action_button_focus_is_glyph_level_and_keeps_its_label():
    """MI-05: the compact action buttons showed zero glyph-level change on
    focus. Focused, they must paint a structural cue (heavy side rails)
    WITHOUT eating the label (the task-2041 full-outline trap) and without
    any dimensional change."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _enter_ingest_mode(screen, pilot)

        browse = screen.query_one("#library-ingest-browse", Button)
        assert not browse.has_focus
        region_before = browse.region
        unfocused_rows = _plain_rows(browse)
        assert not any(
            glyph in row for row in unfocused_rows for glyph in HEAVY_GLYPHS
        ), f"unfocused button already paints heavy glyphs: {unfocused_rows!r}"

        browse.focus()
        await pilot.pause()
        assert browse.has_focus
        focused_rows = _plain_rows(browse)

        assert focused_rows != unfocused_rows, (
            "focus produced a byte-identical plain-text button -- the "
            "color-only regression MI-05 pinned"
        )
        assert any(
            glyph in row for row in focused_rows for glyph in HEAVY_GLYPHS
        ), f"focused button shows no structural cue: {focused_rows!r}"
        # The structural cue must not eat the label (task-2041's trap:
        # a full `outline: heavy` overwrites a 1-row button's only row).
        assert any("Browse" in row for row in focused_rows), (
            f"focus treatment ate the button label: {focused_rows!r}"
        )
        # AC#5: dimensional stability.
        assert browse.region == region_before
        assert len(focused_rows) == len(unfocused_rows)


# --- task-3312 (#1): F1 lists one escape row ---------------------------------


def test_f1_help_in_ingest_lists_exactly_one_escape_row(monkeypatch):
    """task-3312 (#1): the shared footer set spells the exit key "esc"
    while the BINDINGS entry spells it "escape", so the key-string dedupe
    in ``action_show_workbench_help`` missed it and F1 listed the exit
    twice ("esc: back to hub" + "escape: Back to Library hub"). Exactly
    one escape row survives -- the footer's, since the footer set leads
    the panel."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_INGEST_MEDIA

    pushed = []

    class _FakeApp:
        def push_screen(self, panel):
            pushed.append(panel)

    monkeypatch.setattr(
        LibraryScreen, "app", property(lambda self: _FakeApp()), raising=False
    )

    screen.action_show_workbench_help()

    assert len(pushed) == 1
    assert isinstance(pushed[0], WorkbenchHelpPanel)
    shortcuts = list(pushed[0].state.shortcuts)
    escape_rows = [
        (key, description)
        for key, description in shortcuts
        if key.strip().casefold() in ("esc", "escape")
    ]
    assert escape_rows == [("esc", "back")], shortcuts


# --- task-3312 (#4): collapsible panel-header focus is glyph-level -----------


@pytest.mark.asyncio
async def test_options_panel_header_focus_is_glyph_level_under_real_css():
    """task-3312 (#4): the collapsible options-panel header is a Tab stop
    whose focus was color-only -- the one focusable the task-3302
    structural-focus sweep missed. Focused, it must paint a structural
    (heavy-glyph) cue without eating the ▼/title text and without any
    dimensional change."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _enter_ingest_mode(screen, pilot)

        # The generic options panel is always composed, path or no path.
        title = screen.query_one("#type-group-generic CollapsibleTitle")
        assert not title.has_focus
        region_before = title.region
        unfocused_rows = _plain_rows(title)
        assert not any(
            glyph in row for row in unfocused_rows for glyph in HEAVY_GLYPHS
        ), f"unfocused header already paints heavy glyphs: {unfocused_rows!r}"

        title.focus()
        await pilot.pause()
        assert title.has_focus
        focused_rows = _plain_rows(title)

        assert focused_rows != unfocused_rows, (
            "focus produced a byte-identical plain-text header -- the "
            "color-only gap task-3312 (#4) pinned"
        )
        assert any(
            glyph in row for row in focused_rows for glyph in HEAVY_GLYPHS
        ), f"focused header shows no structural cue: {focused_rows!r}"
        # The cue must not eat the header text (task-2041's trap).
        assert any("Import behavior" in row for row in focused_rows), (
            f"focus treatment ate the header title: {focused_rows!r}"
        )
        # Dimensional stability: same region, same painted row count.
        assert title.region == region_before
        assert len(focused_rows) == len(unfocused_rows)
