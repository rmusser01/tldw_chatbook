"""Library ▸ Notes critique wave -- editor-keys group.

Tasks 32131, 32132, 32133, 32138, 32139, 32142. See
``backlog/tasks/task-32131*.md`` through ``task-32142*.md`` for the full
acceptance criteria; each test below is named after the task it pins.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.widgets import Button, Input, Static, TextArea

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
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Library.library_notes_state import LibraryNoteDeleteReceipt


def _build_notes_host() -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Research Note", "id": "note-1"}],
    )
    return LibraryHarness(app)


async def _open_notes_list(screen, pilot):
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, ".library-notes-row")
    await pilot.pause()


def _first_note_row(screen) -> Button:
    """Return the first note row Button, tree-projection or flat-list."""
    rows = list(screen.query(".library-notes-row"))
    assert rows, "No note row Button found"
    return rows[0]


async def _open_first_note_in_info(screen, pilot):
    """Open Notes, select the first note, and switch to the Info pane."""
    await _open_notes_list(screen, pilot)
    _first_note_row(screen).press()
    await _wait_for_selector(screen, pilot, "#library-note-body")
    await pilot.pause()
    screen.query_one("#library-note-context", Button).press()
    await pilot.pause()


# --- task-32131: "/" focuses the filter without typing into it -------------


@pytest.mark.asyncio
async def test_slash_focuses_the_notes_filter_without_inserting_itself():
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        row = _first_note_row(screen)
        row.focus()
        await pilot.pause()

        await pilot.press("/")
        await pilot.pause()

        filter_input = screen.query_one("#library-notes-filter", Input)
        assert screen.focused is filter_input
        assert filter_input.value == "", (
            f"'/' leaked into the filter it focused: {filter_input.value!r}"
        )

        await pilot.press("R", "e", "a", "d", "i", "n", "g")
        await pilot.pause()
        assert filter_input.value == "Reading"


@pytest.mark.asyncio
async def test_slash_on_the_already_focused_notes_filter_rearms_selection():
    """AC#1: a SECOND '/' while the filter already has focus must not type
    a literal slash -- reproduced live (task-32131 evidence): with the
    filter already focused from an earlier interaction, pressing '/' then
    typing 'Reading' produced '/Reading'. Screen.on_key never sees a
    printable key once an Input owns focus (the isinstance guard bails
    early), so this is the SAME class of bug task-1584/the rail search's
    ``LibraryRailSearchInput`` already fixed -- the notes filter needs the
    identical re-arm-on-second-slash behaviour.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        filter_input = screen.query_one("#library-notes-filter", Input)
        filter_input.focus()
        await pilot.pause()
        await pilot.press("a", "b", "c")
        await pilot.pause()
        assert filter_input.value == "abc"

        await pilot.press("/")
        await pilot.pause()
        assert filter_input.value == "abc", (
            f"'/' leaked into the already-focused filter: {filter_input.value!r}"
        )


