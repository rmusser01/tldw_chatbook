"""TASK-34000.13, the slower arms -- NOT in ``scripts/ui_pr_gate_census.txt``.

The lane file ``test_library_note_delete_prompt_visible.py`` pins the two
AC#4 sizes. This sibling covers the rest of AC#1 (the short 160x30 stage,
160x45, 235x52), the Review Focus resize while the prompt is open, the
"Linked from (N)" case through a real ``CharactersRAGDB`` (the static scope
service has no backlinks, and Info is taller with them -- the review's
repro note had one), and the data-integrity facts asserted against the
database: Cancel leaves the row alive, Delete soft-deletes it.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_library_note_delete_prompt_visible import (
    CANCEL,
    CONFIRM,
    DELETE,
    INFO,
    PROMPT,
    _assert_prompt_fully_inside_info,
    _notes_host,
    _open_delete_prompt,
    _open_first_note_in_info,
    _parts,
)
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_selector,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

pytestmark = pytest.mark.bootstrap_profile

#: AC#1's remaining sizes: the short-height stage, the size the review called
#: good (a regression pin), and the widest desk size.
EXTRA_SIZES = [(160, 30), (160, 45), (235, 52)]

_TARGET_TITLE = "Advisor meeting 2026-09-24"
_LINKING_TITLE = "Thesis outline"


class _LinkedNotesProfile:
    """A real ChaChaNotes database with one note that another links to."""

    def __init__(self, tmp_path) -> None:
        self.db = CharactersRAGDB(tmp_path / "notes.sqlite3", client_id="task-34000-13")
        self.target_id = self.db.add_note(
            _TARGET_TITLE, "- Cut chapter 3\n- Draft method chapter by Oct 15\n"
        )
        assert self.target_id
        self.linking_id = self.db.add_note(
            _LINKING_TITLE,
            f"# Outline\n\nSee [[{_TARGET_TITLE}]](note://{self.target_id}) for the cut list.\n",
        )
        assert self.linking_id
        self.interop = NotesInteropService(
            base_db_directory=tmp_path,
            api_client_id="task-34000-13",
            global_db_to_use=self.db,
        )
        self.scope_service = NotesScopeService(
            local_notes_service=self.interop,
            server_service=None,
            folder_repository=LocalNoteFolderRepository(self.db),
        )

    def deleted(self, note_id: str) -> bool:
        states = self.db.get_note_version_states([note_id])
        assert note_id in states, f"note {note_id} has no row at all"
        return bool(states[note_id]["deleted"])


def _linked_host(tmp_path) -> tuple[LibraryHarness, _LinkedNotesProfile]:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    profile = _LinkedNotesProfile(tmp_path)
    app.chachanotes_db = profile.db
    app.notes_scope_service = profile.scope_service
    app.notes_service = profile.interop
    return LibraryHarness(app), profile


async def _open_target_note_in_info(host, pilot, profile):
    """Open the linked-to note and its Info, with "Linked from (1)" loaded."""
    from Tests.UI.test_library_shell import _active_library_screen, _wait_for_library_shell

    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, ".library-notes-row")
    row = next(
        button
        for button in screen.query(".library-notes-row").results(Button)
        if _TARGET_TITLE in str(button.label)
    )
    row.press()
    await _wait_for_selector(screen, pilot, "#library-note-body")
    await _wait_for_condition(
        pilot,
        lambda: screen._notes_state.backlinks_status == "ready"
        and len(screen._notes_state.backlinks) == 1,
        message=lambda: (
            "Info's 'Linked from' never loaded the one backlink: status="
            f"{screen._notes_state.backlinks_status!r}, "
            f"backlinks={screen._notes_state.backlinks!r}"
        ),
    )
    screen.query_one("#library-note-context", Button).press()
    await pilot.pause()
    await pilot.pause()
    # Sanity: the backlink rows are mounted, so Info really is taller.
    assert len(screen.query("#library-note-context-backlinks .library-note-backlink")) == 1
    return screen


@pytest.mark.asyncio
@pytest.mark.parametrize("size", EXTRA_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
async def test_prompt_is_fully_inside_info_at_the_other_sizes(size) -> None:
    """AC#1: 160x30 (short), 160x45 (the review's good size, a regression
    pin) and 235x52 all show the whole prompt inside Info."""
    host = _notes_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_note_in_info(host, pilot)
        await _open_delete_prompt(screen, pilot)
        _assert_prompt_fully_inside_info(size, host, screen)
        assert screen.focused is screen.query_one(CANCEL, Button)


@pytest.mark.asyncio
async def test_resize_while_the_prompt_is_open_keeps_it_inside_info_and_cancel_restores() -> None:
    """Review Focus: a 160x45 -> 120x36 resize while the prompt is open.

    The prompt stays open, Tab is still trapped inside it, both buttons
    are still painted inside Info at the new size, and Cancel restores the
    ORIGINAL offset captured before the prompt opened (clamped to the new
    geometry -- here 0, which is also what the pre-prompt Info sat at).
    """
    host = _notes_host()
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_first_note_in_info(host, pilot)
        info = screen.query_one(INFO)
        origin = info.scroll_y
        await _open_delete_prompt(screen, pilot)
        _assert_prompt_fully_inside_info((160, 45), host, screen)

        await pilot.resize_terminal(120, 36)
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        assert screen._notes_state.compact is False
        assert screen._notes_state.confirming_delete is True
        info, prompt, _copy, _actions, cancel, confirm = _parts(screen)
        assert prompt.display is True
        assert screen.focused is cancel, f"focused={screen.focused!r}"
        _assert_prompt_fully_inside_info((120, 36), host, screen)

        await pilot.press("tab")
        await pilot.pause()
        assert screen.focused is confirm
        assert info.region.contains_region(confirm.region)
        await pilot.press("shift+tab")
        await pilot.pause()
        assert screen.focused is cancel

        cancel.press()
        await pilot.pause()
        await pilot.pause()
        assert prompt.display is False
        assert info.scroll_y == min(origin, info.max_scroll_y), (
            f"Cancel left Info at {info.scroll_y}; origin {origin}, "
            f"max {info.max_scroll_y}"
        )
        delete_button = screen.query_one(DELETE, Button)
        assert screen.focused is delete_button
        assert info.region.contains_region(delete_button.region)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 36), (80, 24)], ids=lambda s: f"{s[0]}x{s[1]}")
async def test_linked_from_note_shows_the_prompt_and_cancel_keeps_the_row(
    size, tmp_path
) -> None:
    """AC#1 on a note with a populated "Linked from (1)" (Info is taller),
    through a real database; Cancel leaves the note's row alive."""
    host, profile = _linked_host(tmp_path)
    async with host.run_test(size=size) as pilot:
        screen = await _open_target_note_in_info(host, pilot, profile)
        info = screen.query_one(INFO)
        delete_button = screen.query_one(DELETE, Button)
        delete_button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        origin = info.scroll_y

        await _open_delete_prompt(screen, pilot)
        _assert_prompt_fully_inside_info(size, host, screen)
        assert screen.focused is screen.query_one(CANCEL, Button)

        screen.query_one(CANCEL, Button).press()
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one(PROMPT).display is False
        assert info.scroll_y == origin
        assert screen.focused is delete_button

    row = profile.db.get_note_by_id(profile.target_id)
    assert row is not None and row["title"] == _TARGET_TITLE, (
        "Cancel must leave the note's row alive in the database"
    )
    assert profile.deleted(profile.target_id) is False


@pytest.mark.asyncio
async def test_confirming_delete_soft_deletes_the_row_in_the_database(tmp_path) -> None:
    """The delete semantics are unchanged: a visible Delete, pressed, soft-
    deletes the note (tombstone in the database, not a hard delete) and
    the linking note is untouched."""
    host, profile = _linked_host(tmp_path)
    async with host.run_test(size=(120, 36)) as pilot:
        screen = await _open_target_note_in_info(host, pilot, profile)
        await _open_delete_prompt(screen, pilot)
        _assert_prompt_fully_inside_info((120, 36), host, screen)

        await pilot.press("tab")
        await pilot.pause()
        confirm = screen.query_one(CONFIRM, Button)
        assert screen.focused is confirm
        assert screen.query_one(INFO).region.contains_region(confirm.region)
        confirm.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.view == "list"
            and profile.db.get_note_by_id(profile.target_id) is None,
            message="the confirmed delete never reached the database",
        )
        await pilot.pause()

    assert profile.db.get_note_by_id(profile.target_id) is None
    assert profile.deleted(profile.target_id) is True, "not a soft delete"
    linking = profile.db.get_note_by_id(profile.linking_id)
    assert linking is not None and linking["title"] == _LINKING_TITLE
