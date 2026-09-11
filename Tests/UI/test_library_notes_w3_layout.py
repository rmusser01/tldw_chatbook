"""Library ▸ Notes wave-3 group `layout`: preview, primary actions, cues.

Covers tasks 32249 (the Markdown preview's height, focus, callouts and
status line), 32259 (a primary action stands beside the content it acts
on), 32270 (the `‹ Library / Notes` cue), 32261 (the compact select strip)
and 32390 (a dead component rule).

The two geometries the critique ran at are 235x52 and 100x30; the
resolver-level 60x24 case for task-32389 lives with the rest of the Notes
layout pins in ``test_library_notes_wave_list.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

#: The critique's wide geometry. The Notes compact breakpoint is 120, so
#: this is comfortably the wide layout.
WIDE = (235, 52)

#: The critique's compact geometry.
COMPACT = (100, 30)

#: A note carrying every construct the critique read in the preview. The
#: Obsidian callout is the one that leaked its marker.
CALLOUT_NOTE_BODY = """# Heading 1

Some prose.

> [!note] Obsidian callout
> This is a callout.

> [!warning]
> No title on this one.
"""


def _notes_host(body: str = CALLOUT_NOTE_BODY) -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Markdown showcase", "id": "note-1", "content": body}],
    )
    return LibraryHarness(app)


async def _open_the_note(screen, pilot):
    """Land on the Notes route with the seeded note open in Edit.

    ``LibraryNotesCanvas.on_button_pressed`` stops every ``Button.Pressed``
    while the canvas is not displayed, and a harness press issued the frame
    the row mounts can land in exactly that window -- measured 1 run in 6,
    where the press vanished with no selected note, no worker and no
    locator warning. Pressing until the view actually changes is what makes
    this deterministic; it is safe because a re-press only ever happens
    while the list is still the view.
    """
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes").press()
    await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")
    for _ in range(20):
        rows = screen.query(".library-notes-tree-note-row")
        if rows:
            rows.first(Button).press()
        for _ in range(10):
            await pilot.pause()
            if screen.query("#library-note-body"):
                await pilot.pause()
                await pilot.pause()
                return
    raise AssertionError("the seeded note never opened in the editor")


async def _show_preview(screen, pilot):
    """Press Preview and wait for the region to take the pane."""
    screen.query_one("#library-note-preview", Button).press()
    preview = await _wait_for_selector(screen, pilot, "#library-note-preview-region")
    await _wait_for_condition(
        pilot,
        lambda: screen.query_one("#library-note-preview-region").region.height > 0,
        message="Preview never took the work pane.",
    )
    return preview


# -- task-32249: the preview is the reading surface -----------------------


@pytest.mark.asyncio
async def test_the_wide_preview_fills_the_work_pane_at_the_critique_width():
    """235x52: no 20-row cap, the compact layout's `1fr` shape both ways.

    The critique measured the box closing at screen row 33 with 14 blank
    rows beneath it while `#library-shell-grid.library-notes-compact
    #library-note-preview-region` already had `height: 1fr`. task-32217
    deleted the cap; this pins it at the width the critique ran at (the
    existing crit-9 pin runs at 170x48).
    """
    host = _notes_host()
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _open_the_note(screen, pilot)
        preview = await _show_preview(screen, pilot)

        parent = preview.parent
        assert parent.content_region.height > 20, parent.content_region
        # One row of the region's own bottom margin is all that may be left.
        assert preview.region.bottom >= parent.content_region.bottom - 1, (
            f"Preview stops at {preview.region.bottom} in a pane whose content "
            f"ends at {parent.content_region.bottom}: "
            f"{parent.content_region.bottom - preview.region.bottom} blank rows "
            "under the rendered note."
        )


@pytest.mark.asyncio
async def test_activating_preview_focuses_its_scroll_owner():
    """The footer's `pgup/pgdn scroll` promise is true without a click.

    Live at 235x52 the keys did nothing from the landed state: the region
    is the sole scroll owner and focus was still on the Preview button.
    One click inside the box and the same key paged.
    """
    host = _notes_host()
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _open_the_note(screen, pilot)
        preview = await _show_preview(screen, pilot)

        assert screen.focused is preview, (
            "Preview landed with focus on "
            f"{getattr(screen.focused, 'id', None)!r}, so its paging keys are "
            "inert until the reader clicks inside the box."
        )


def test_an_obsidian_callout_keeps_its_words_and_drops_its_marker():
    """The pure rewrite behind the preview's callout rendering."""
    from tldw_chatbook.Utils.markdown_parsing import render_obsidian_callouts

    rendered = render_obsidian_callouts(CALLOUT_NOTE_BODY)

    assert "[!note]" not in rendered
    assert "[!warning]" not in rendered
    # The quote bar (the callout's own box) and both authored words stay.
    assert "> **Note: Obsidian callout**" in rendered
    assert "> **Warning**" in rendered
    assert "> This is a callout." in rendered
    # A note with no callout in it is not rewritten at all.
    assert render_obsidian_callouts("# Plain\n\n> quoted\n") == "# Plain\n\n> quoted\n"


def test_a_fenced_example_of_callout_syntax_is_left_alone():
    """Review F8: a note that DOCUMENTS callouts kept its example verbatim."""
    from tldw_chatbook.Utils.markdown_parsing import render_obsidian_callouts

    source = (
        "Write one like this:\n\n"
        "```markdown\n"
        "> [!note] Example\n"
        "```\n\n"
        "> [!note] A real one\n"
    )
    rendered = render_obsidian_callouts(source)

    assert "```markdown\n> [!note] Example\n```" in rendered
    assert "> **Note: A real one**" in rendered
    # A tilde fence, and a fence that is itself inside a blockquote.
    assert "[!tip]" in render_obsidian_callouts("~~~\n> [!tip] x\n~~~\n")
    assert "[!tip]" in render_obsidian_callouts("> ```\n> > [!tip] x\n> ```\n")


def test_a_nested_callout_and_an_acronym_type_survive_the_rewrite():
    """Review F8: `> > [!note]` matched one `>`, and `[!TODO]` became `Todo`."""
    from tldw_chatbook.Utils.markdown_parsing import render_obsidian_callouts

    assert render_obsidian_callouts("> > [!tip] Nested\n") == "> > **Tip: Nested**\n"
    assert render_obsidian_callouts("> [!TODO] Ship it\n") == "> **TODO: Ship it**\n"
    assert render_obsidian_callouts("> [!todo]\n") == "> **Todo**\n"


@pytest.mark.asyncio
async def test_the_preview_body_renders_the_callout_not_its_marker():
    """The rewrite reaches the mounted Markdown, on compose and on sync."""
    host = _notes_host()
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _open_the_note(screen, pilot)
        await _show_preview(screen, pilot)
        await pilot.pause()

        body = screen.query_one("#library-note-preview-body")
        assert "[!note]" not in body.source, body.source
        painted = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in screen._compositor.render_strips()
        )
        assert "[!note]" not in painted, "the preview still paints the raw marker"


@pytest.mark.asyncio
async def test_the_status_line_stops_promising_autosave_while_preview_shows():
    """Preview is read-only, so "changes save automatically" is not true."""
    host = _notes_host()
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _open_the_note(screen, pilot)

        authority = screen.query_one("#library-note-work-authority", Static)
        edit_copy = str(getattr(authority.renderable, "plain", authority.renderable))
        assert "Keep editing" in edit_copy, edit_copy

        await _show_preview(screen, pilot)
        await pilot.pause()

        preview_copy = str(
            getattr(authority.renderable, "plain", authority.renderable)
        )
        assert "Keep editing" not in preview_copy, preview_copy
        assert "changes save automatically" not in preview_copy, preview_copy
        assert "Next: Press Edit" in preview_copy, preview_copy


# -- task-32270: the cue the guide describes ------------------------------


@pytest.mark.asyncio
async def test_the_return_cue_stays_off_the_strip_in_wide_database_notes():
    """`library_browse_route_swap` can never display it in this state.

    `wide_focused_task` requires ``not adaptive_database_notes``, which is
    false for every wide Database-Notes canvas kind, so the cue's display
    flag is always ``False`` here. The guide used to promise it; it now
    describes the two source switches that really stand there, and this
    pins the behaviour the guide describes.
    """
    host = _notes_host()
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _open_the_note(screen, pilot)

        assert screen.query_one("#library-notes-task-return", Button).display is False
        for selector in (
            "#library-notes-source-database",
            "#library-notes-source-files",
        ):
            assert screen.query_one(selector, Button).display is True, selector


def test_the_notes_guide_does_not_promise_a_cue_the_wide_route_cannot_paint():
    """The guide's own sentences, checked against the same predicate."""
    guide = Path(__file__).resolve().parents[2] / "Docs/User_Guide/library/notes.md"
    text = guide.read_text(encoding="utf-8")

    assert "When Library navigation is closed, one stable cue names the return" not in text
    # The cue is still documented where it really renders.
    assert "‹ Library / Notes" in text


# -- task-32259: a primary action stands with its content -----------------


#: The two select-phase screens and the EXACT rows between the selection
#: summary and the primary action on each. Both are pinned to a measured
#: number rather than a bound (review finding F5): a bound whose ceiling is
#: the measured value has no headroom by accident, and the whole point of
#: this pin is that a control added to `_compose_selection` must be a
#: deliberate change, not a silent drift back toward the pane floor.
#:
#: `selected` composes, in order: "Add another file", "Change selection",
#: "Clear", the "Notes destination" label, its Input (3 rows of field
#: chrome) and its (empty, 0-row) error line -- 8 rows, then the action.
#: `empty` composes only "Choose a file or folder" -- 1 row.
_IMPORT_SELECT_SCREENS = (
    pytest.param(True, 8, id="two-files-selected"),
    pytest.param(False, 1, id="nothing-selected"),
)


@pytest.mark.parametrize("selected,expected_rows", _IMPORT_SELECT_SCREENS)
@pytest.mark.parametrize("size", [WIDE, COMPACT, (60, 24)])
@pytest.mark.asyncio
async def test_the_import_check_action_stands_with_the_selection_it_acts_on(
    selected: bool, expected_rows: int, size
):
    """task-32259 AC#1/AC#4: measured rows between summary and action.

    Live at 235x52 with two files chosen, the selection summary sat at rows
    7-11 and "Check selection" at row 49 -- the action was floated under a
    `1fr` scroll body that had nothing else in it. The empty-selection
    screen (the state `wave3-caps/layout/17-import-nosel.txt` captures) is
    pinned beside it so a regression cannot hide in whichever one the
    capture did not show.
    """
    from dataclasses import replace

    from Tests.UI.test_library_notes_wave_import_ux import (
        _import_snapshot,
        _ImportHost,
    )

    snapshot = _import_snapshot(
        selected_names=(
            ("vault/Archive/note-1.md", "vault/Archive/note-2.md")
            if selected
            else ()
        ),
        selection_kind="files" if selected else "",
        destination="Imported" if selected else "",
        can_check=selected,
        check_disabled_reason="" if selected else "Choose a source first.",
    )
    app = _ImportHost(replace(snapshot))

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        summary = app.query_one("#note-import-source-summary")
        check = app.query_one("#note-import-check", Button)
        body = app.query_one("#note-import-body")

        distance = check.region.y - summary.region.bottom
        assert distance == expected_rows, (
            f"'Check selection' is painted {distance} rows below the selection "
            f"summary it acts on, not {expected_rows} (summary "
            f"{summary.region!r}, action {check.region!r} in a "
            f"{app.size.height}-row terminal)."
        )
        # The action moved INSIDE the scroll body; the pane floor is what it
        # was floated above before.
        assert body in check.ancestors
        assert check.region.bottom <= body.region.bottom


def test_a_full_canvas_notes_task_owns_the_pane_width():
    """task-32259 AC#2: Import once / Add from files close the list beside them."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
        LibraryNotesController,
    )
    from tldw_chatbook.Utils.adaptive_reader_state import (
        AdaptiveReaderLayoutPreferences,
    )

    preferences = AdaptiveReaderLayoutPreferences()
    derive = LibraryNotesController._library_notes_work_first_preferences

    for view in ("import", "lasting_add", "lasting_roots"):
        fake = SimpleNamespace(
            _library_notes_view=view,
            _library_notes_work_session_phase=None,
        )
        assert derive(fake, preferences).items_open is False, view

    for view in ("list", "editor", "trash"):
        fake = SimpleNamespace(
            _library_notes_view=view,
            _library_notes_work_session_phase=None,
        )
        assert derive(fake, preferences).items_open is True, view


# -- task-32261: compact select strip, grip names -------------------------


@pytest.mark.asyncio
async def test_the_compact_select_strip_paints_every_action_inside_the_pane():
    """task-32261 AC#1/AC#2/AC#5: 100x30 gives the list pane 42 columns.

    The strip painted "0 selected  Done  All 10  Clear" and the guide's
    fifth action, "Export selected", started at the pane's right edge --
    one cell of it visible, unpressable. The same count was printed again
    on the line below.
    """
    from dataclasses import replace

    from Tests.UI.test_library_notes_wave_list import _CanvasApp, _list_state
    from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas

    state = replace(
        _list_state(),
        select_mode=True,
        selected_count=0,
        total_count=10,
        result_count=10,
    )
    # The Items width `resolve_adaptive_reader_layout` gives the list at 100
    # columns with no note open.
    app = _CanvasApp(pane_width=42, compact=True, list_state=state)

    async with app.run_test(size=COMPACT) as pilot:
        await pilot.pause()
        canvas = app.query_one("#library-notes-canvas", LibraryNotesCanvas)
        export = app.query_one("#library-notes-export-selected", Button)

        assert export.region.width > 0
        assert export.region.right <= canvas.region.right, (
            f"'Export selected' is painted to {export.region.right} on a "
            f"{canvas.region.width}-column pane."
        )

        painted = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in app.screen._compositor.render_strips()
        )
        assert painted.count("0 selected") == 1, painted

        # The surviving counter is on its own line, not jammed against the
        # focused Done button.
        status = app.query_one("#library-notes-selection-status", Static)
        toggle = app.query_one("#library-notes-select-toggle", Button)
        assert status.region.y != toggle.region.y


@pytest.mark.asyncio
async def test_the_notes_select_strip_uses_the_library_glyph_legend():
    """task-32261 AC#3 (revised): one meaning per glyph on this strip.

    The circle is `LIBRARY_DISABLED_ACTION_MARKER` -- the Library-wide
    disabled marker task-32235 settled and `Docs/User_Guide/library.md`
    documents -- and it appears only on a disabled action, never on a
    selection control, which uses `☐/☑`. Whether that marker should be
    dropped from disabled actions across the Library is the peer legend's
    call, not this canvas's.
    """
    from dataclasses import replace

    from Tests.UI.test_library_notes_wave_list import _CanvasApp, _list_state
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_DISABLED_ACTION_MARKER,
        LIBRARY_GLYPH_SELECTED,
        LIBRARY_GLYPH_UNSELECTED,
    )

    state = replace(
        _list_state(), select_mode=True, selected_count=0, total_count=10
    )
    app = _CanvasApp(pane_width=42, compact=True, list_state=state)

    async with app.run_test(size=COMPACT) as pilot:
        await pilot.pause()
        for button in app.query("#library-notes-selection-actions Button"):
            label = str(button.label)
            assert LIBRARY_GLYPH_SELECTED not in label, label
            assert LIBRARY_GLYPH_UNSELECTED not in label, label
            if LIBRARY_DISABLED_ACTION_MARKER in label:
                assert button.disabled, f"{button.id} marks a live action disabled"
                assert button.tooltip, f"{button.id} has no reason beside its marker"


@pytest.mark.asyncio
async def test_the_notes_pane_grips_carry_an_accessible_name():
    """task-32261 AC#4: on the Notes route, not only Conversations."""
    from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
        LibraryAdaptiveReaderPaneGrip,
    )

    host = _notes_host()
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")
        await pilot.pause()

        grips = {
            grip.pane: grip for grip in screen.query(LibraryAdaptiveReaderPaneGrip)
        }
        assert set(grips) == {"library", "items"}
        for pane, grip in grips.items():
            assert grip.name, f"the {pane} grip has no accessible name"
            assert grip.tooltip == grip.name
            assert grip.painted_name(), f"the {pane} grip paints no name"


# -- task-32390: the dead component rule ----------------------------------


def test_the_retired_template_section_rule_is_gone_from_source_and_bundle():
    """task-32356 removed the widget; the selector matched nothing since."""
    css = Path(__file__).resolve().parents[2] / "tldw_chatbook/css"
    for sheet in (
        css / "components/_agentic_terminal.tcss",
        css / "screen_agentic_library.tcss",
        css / "tldw_cli_modular.tcss",
    ):
        assert "#library-notes-template-section" not in sheet.read_text(
            encoding="utf-8"
        ), sheet
