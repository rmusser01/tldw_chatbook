"""Library ▸ Notes critique wave -- editor chrome strip (task-32143).

The critique found the word count buried under Info and the caret position
nowhere at all. The strip puts both under the body, where a terminal reader
looks. See ``backlog/tasks/task-32143*.md``.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static, TextArea

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
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    library_note_chrome_facts,
)

FACTS = "#library-note-chrome-facts"


def _build_notes_host(body: str = "alpha beta gamma delta epsilon") -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Research Note", "id": "note-1", "content": body}],
    )
    return LibraryHarness(app)


async def _open_first_note(screen, pilot) -> TextArea:
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, ".library-notes-row")
    await pilot.pause()
    rows = list(screen.query(".library-notes-row"))
    assert rows, "No note row Button found"
    rows[0].press()
    await _wait_for_selector(screen, pilot, "#library-note-body")
    await pilot.pause()
    return screen.query_one("#library-note-body", TextArea)


def _facts_text(screen) -> str:
    return str(screen.query_one(FACTS, Static).renderable)


# --- the pure copy rule ----------------------------------------------------


def test_chrome_facts_copy_counts_words_and_reports_a_one_based_caret():
    assert library_note_chrome_facts(0, 0, 0) == "0 words · 1:1"
    assert library_note_chrome_facts(1, 0, 0) == "1 word · 1:1"
    assert library_note_chrome_facts(5, 2, 13) == "5 words · 3:14"
    assert library_note_chrome_facts(12345, 0, 0) == "12,345 words · 1:1"


# --- one row, under the body, carrying no save state of its own ------------


@pytest.mark.asyncio
async def test_the_strip_is_one_row_under_the_body_and_never_a_focus_stop():
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        body = await _open_first_note(screen, pilot)

        facts = screen.query_one(FACTS, Static)
        assert facts.region.height == 1
        assert facts.region.y > body.region.bottom - 1

        # AC#2: it carries no save state of its own -- #library-note-status
        # is still the one line that reports saving, in its one place.
        status = screen.query_one("#library-note-status", Static)
        assert status.display is True
        assert "Saved" not in str(facts.renderable)
        assert len(screen.query("#library-note-status")) == 1

        # Chrome, never a focus stop.
        assert facts.can_focus is False


# --- the facts themselves --------------------------------------------------


@pytest.mark.asyncio
async def test_the_strip_reports_the_word_count_and_caret_and_follows_typing():
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        body = await _open_first_note(screen, pilot)

        assert _facts_text(screen) == "5 words · 1:1"

        body.focus()
        body.text = "one two three"
        body.post_message(TextArea.Changed(body))
        await _wait_for_condition(
            pilot,
            lambda: _facts_text(screen).startswith("3 words"),
            message=f"The strip kept reporting {_facts_text(screen)!r} after an edit.",
        )

        body.move_cursor((0, 7))
        await _wait_for_condition(
            pilot,
            lambda: _facts_text(screen) == "3 words · 1:8",
            message=f"The strip kept reporting {_facts_text(screen)!r} after a move.",
        )


@pytest.mark.asyncio
async def test_the_strip_reports_the_caret_line_after_a_multi_line_move():
    host = _build_notes_host(body="alpha\nbeta gamma\ndelta")
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        body = await _open_first_note(screen, pilot)

        assert _facts_text(screen) == "4 words · 1:1"

        body.focus()
        body.move_cursor((2, 3))
        await _wait_for_condition(
            pilot,
            lambda: _facts_text(screen) == "4 words · 3:4",
            message=f"The strip kept reporting {_facts_text(screen)!r} on line 3.",
        )


# --- the surfaces that have no caret, and the narrow terminal --------------


@pytest.mark.asyncio
async def test_the_strip_is_hidden_off_the_editor_and_below_eighty_columns():
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        assert screen.query_one(FACTS, Static).display is True

        screen.query_one("#library-note-context", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(FACTS, Static).display is False,
            message="Info kept a caret position on a surface with no caret.",
        )
        # The save state is still the one line Info shows (task-32177).
        assert screen.query_one("#library-note-status", Static).display is True

        screen.query_one("#library-note-edit", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(FACTS, Static).display is True,
            message="The strip did not come back with the editor.",
        )

        await pilot.resize_terminal(79, 24)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(FACTS, Static).display is False,
            message="The strip survived below 80 columns.",
        )
        assert screen.query_one("#library-note-status", Static).display is True

        await pilot.resize_terminal(*LIBRARY_TEST_SIZE)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(FACTS, Static).display is True,
            message="The strip did not return above 80 columns.",
        )
