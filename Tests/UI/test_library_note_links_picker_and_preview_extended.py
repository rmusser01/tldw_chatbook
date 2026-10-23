"""TASK-34000.11, the slower arms (NOT in ``scripts/ui_pr_gate_census.txt``).

Enter on the focused Preview follows the note's one link or lists several;
the picker at 80x24 stays inside the pane; a 30-title corpus caps the rows
at eight and Down wraps; a title carrying ``[b]`` paints literally in the
picker and lands in the inserted link.
"""

from __future__ import annotations

import pytest
from textual.widgets import OptionList, TextArea

from Tests.UI.library_note_links_support import (
    INDEX,
    LISBON,
    TWO,
    _app,
    _armed_editor,
    _open_note,
    _painted,
    _picker_open,
    _picker_or_none,
    _preview,
    _record_open_url,
    _title_value,
    _until,
    _until_painted,
)
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _library,
    _new_blank_note,
    _type,
)
from Tests.UI.test_library_notes_files_sync_journey import (
    _isolate_actor_pack_recovery,  # noqa: F401 - autouse fixture, re-exported
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

COMPACT = (80, 24)


def _footer_pairs(screen) -> tuple[tuple[str, str], ...]:
    return tuple(screen._notes_controller._library_notes_footer_shortcuts())


async def _body_focused(screen, pilot) -> TextArea:
    await _new_blank_note(screen, pilot)
    await _type(pilot, "T")
    body = screen.query_one("#library-note-body", TextArea)
    await pilot.click("#library-note-body")
    await _until(pilot, lambda: screen.focused is body, "the body to take focus")
    return body


async def test_enter_on_the_focused_preview_follows_the_only_link_and_lists_several(
    tmp_path, monkeypatch
) -> None:
    """Sharper AC#3 (owner question 1)."""
    app, profile = _app(tmp_path)
    opened = _record_open_url(monkeypatch, app)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        await _open_note(screen, pilot, INDEX, profile.index_id)
        await _preview(screen, pilot, LISBON, app)
        region = screen.query_one("#library-note-preview-region")
        await _until(pilot, lambda: screen.focused is region, "Preview to take focus")
        assert ("enter", "follow link") in _footer_pairs(screen), _footer_pairs(screen)

        await pilot.press("enter")
        await _armed_editor(screen, pilot, profile.lisbon_id)
        assert _title_value(screen) == LISBON
        assert opened == []

        # A note with no link: no chip, and Enter leaves the Preview alone.
        await _preview(screen, pilot, "itinerary", app)
        await _until(pilot, lambda: screen.focused is region_of(screen), "Preview focus")
        assert ("enter", "follow link") not in _footer_pairs(screen)
        await pilot.press("enter")
        await pilot.pause(0.4)
        assert screen._library_note_session.snapshot.note_id == profile.lisbon_id
        assert screen.query_one("#library-note-preview-region").display

        # Two links: Enter lists both labels; Down + Enter opens the second.
        await _open_note(screen, pilot, TWO, profile.two_id)
        await _preview(screen, pilot, INDEX, app)
        await _until(pilot, lambda: screen.focused is region_of(screen), "Preview focus")
        await pilot.press("enter")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", INDEX)
        painted = _painted(app, screen, "#library-note-link-picker")
        assert LISBON in painted and INDEX in painted, painted
        await pilot.press("down")
        await pilot.press("enter")
        await _armed_editor(screen, pilot, profile.index_id)
        assert _title_value(screen) == INDEX
        assert not _picker_open(screen)
        assert opened == []


def region_of(screen):
    found = screen.query("#library-note-preview-region")
    return found.first() if found else None


LONG_TITLE = (
    "A very long note title that keeps going so we can test how the tree, "
    "the editor header and the Info pane cope with it at every width"
)


async def test_the_picker_fits_inside_the_pane_at_80x24(tmp_path, monkeypatch) -> None:
    """The picker stays inside the editor pane, and a title wider than the
    picker takes ONE row, cut with an ellipsis (live at 160x45 and 80x24 the
    seed's long title wrapped to two rows and hid the eighth row)."""
    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    assert profile.db.add_note(LONG_TITLE, "wide")
    async with app.run_test(size=COMPACT) as pilot:
        screen = await _library(app, pilot)
        await _body_focused(screen, pilot)
        await _type(pilot, "[[")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        picker = _picker_or_none(screen)
        pane = screen.query_one("#library-note-editor-region").region
        assert picker.region.x >= pane.x and picker.region.right <= pane.right, (
            picker.region,
            pane,
        )
        assert picker.region.y >= 0 and picker.region.bottom <= app.size.height
        assert picker.region.width <= app.size.width
        options = screen.query_one("#library-note-link-picker-options", OptionList)
        await _until(pilot, lambda: options.option_count >= 2, "the rows")
        # Rows + the two border rows: no title may take a second row.
        assert picker.region.height == options.option_count + 2, (
            picker.region,
            options.option_count,
        )
        painted = _painted(app, screen, "#library-note-link-picker")
        assert "A very long note title" in painted and "…" in painted, painted
        assert "every width" not in painted, painted


async def test_thirty_titles_cap_the_rows_at_eight_and_down_wraps(
    tmp_path, monkeypatch
) -> None:
    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    for index in range(30):
        assert profile.db.add_note(f"Topic {index:02d}", "corpus")
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        await _body_focused(screen, pilot)
        # "to", not "t": a one-character last word is matched whole (Task 3's
        # rule, Review Focus 3), so "t" finds nothing by design -- the picker
        # then shows its one disabled "No note titled" row.
        await _type(pilot, "[[to")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", "Topic")
        options = screen.query_one("#library-note-link-picker-options", OptionList)
        await _until(pilot, lambda: options.option_count == 8, "eight rows")
        assert options.highlighted == 0
        for _ in range(9):
            await pilot.press("down")
        await _until(pilot, lambda: options.highlighted == 1, "Down x9 to wrap to row 1")


async def test_a_title_with_markup_paints_literally_and_lands_in_the_link(
    tmp_path, monkeypatch
) -> None:
    """Review Focus 5: ``[b]`` is painted as text, never parsed as markup; the
    inserted link is one parseable ``[[…]](note://id)``."""
    from tldw_chatbook.Notes.note_import_plan_models import stored_note_link

    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    bold_id = profile.db.add_note("Bold [b] title", "markup")
    assert bold_id
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _body_focused(screen, pilot)
        await _type(pilot, "[[Bold")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", "Bold [b] title")
        await pilot.press("enter")
        expected = stored_note_link("Bold [b] title", bold_id)
        await _until(pilot, lambda: body.text == expected, f"the body to read {expected!r}")
        assert "(b)" in expected and "[b]" not in expected
