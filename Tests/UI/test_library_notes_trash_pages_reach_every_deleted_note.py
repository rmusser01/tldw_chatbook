"""Notes ▸ Recently deleted pages through every deleted note, at 160x45.

TASK-34000.29 (N-29), part 8a's first boot, split out of
``test_library_notes_trash_pages_and_purges.py`` in the polish-1b final fix
wave (2026-10-29): that file measured 30.26 s alone with its three boots under
load, over the census's 30 s budget, and this boot was its slowest (10.78 s of
the three). The test keeps its name; the helpers the two 100x30 boots share
stay in that file and are imported from it, the ``Restore`` gesture only this
boot uses moved here with it.

AC#10 and ruling T8-d, end to end: 25 soft-deleted notes page as "1–20 of 25
· Page 1 of 2" with Previous/Next; after Next on the last page focus is on
Previous, never on the disabled Next or nowhere; a Restore keeps focus on the
trash list (the row now at the restored index, clamped); a page past the
shrunken total clamps to the last page, never an empty one.

One boot of the real ``LibraryScreen`` with the production stylesheet over a
real ChaChaNotes database (``_NotesProfile`` plus 25 soft-deleted notes).
Every gesture is a real click or key; every claim is painted text, a focused
control, or the database read back through a second connection.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_library_notes_trash_pages_and_purges import (
    DELETED,
    WIDE,
    _focused_id,
    _mounted,
    _notes_tree_settled,
    _painted,
    _settled_on,
    _titles,
    _until,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_library_shell,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

TITLES = _titles(DELETED)


async def _restore_focused_row(host, pilot, screen, seed, title, *, then_range):
    """``r`` on the focused row restores exactly ``title``; the page re-reads.

    Settles first (``_settled_on``): a Restore pressed while the previous
    one still holds the Notes interlock is refused (``test_trash_restore_
    refuses_while_a_notes_mutation_is_in_flight``), which lost 3 of 16
    presses under 8-way load before this wait.
    """
    await _settled_on(
        pilot,
        host,
        screen,
        lambda: getattr(screen.focused, "note_title", "") == title,
        lambda: f"focus settled on {title} (focus: {_focused_id(screen)!r})",
    )
    await pilot.press("r")
    await _until(
        pilot,
        lambda: title not in seed.deleted() and then_range in _painted(host),
        lambda: f"{title} restored and {then_range!r} painted: {_painted(host)}",
    )


async def test_every_deleted_note_is_reachable_through_paging(tmp_path) -> None:
    """AC#10 and ruling T8-d's disabled-control and clamp arms, in one boot.

    Base (734e632f07): "1–20 of 25" never paints -- the view says "Showing
    the 20 most recently deleted of 25. Restore one to see the rest." and
    has no Next.
    """
    host, seed = _mounted(tmp_path)
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _notes_tree_settled(host, pilot, screen)

        await pilot.click("#library-notes-trash-open")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: "1–20 of 25 · Page 1 of 2" in _painted(host)
            and "Deleted 20" in _painted(host)
            and _focused_id(screen) == "library-notes-trash-restore-0",
            lambda: f"page 1 (focus: {_focused_id(screen)!r}): {_painted(host)}",
        )
        first_page = _painted(host)
        assert all(title in first_page for title in TITLES[:20])
        assert "Deleted 21" not in first_page

        # Next is the way on; the last page disables it, so focus is on
        # Previous -- not on the disabled Next, not on a pane grip.
        await pilot.click("#library-notes-trash-next")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: "21–25 of 25 · Page 2 of 2" in _painted(host)
            and _focused_id(screen) == "library-notes-trash-previous",
            lambda: f"page 2, focus on Previous (focus: {_focused_id(screen)!r}): "
            f"{_painted(host)}",
        )
        second_page = _painted(host)
        assert all(title in second_page for title in TITLES[20:])
        assert not any(title in second_page for title in TITLES[:20])
        assert screen.query_one("#library-notes-trash-next", Button).disabled

        # A real key from there: Enter on Previous reads page 1, where
        # Previous is disabled -- so focus is on Next; Enter again, page 2.
        await pilot.press("enter")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: "1–20 of 25 · Page 1 of 2" in _painted(host)
            and _focused_id(screen) == "library-notes-trash-next",
            lambda: f"page 1, focus on Next (focus: {_focused_id(screen)!r})",
        )
        await pilot.press("enter")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: "21–25 of 25 · Page 2 of 2" in _painted(host)
            and _focused_id(screen) == "library-notes-trash-previous",
            lambda: f"page 2 again (focus: {_focused_id(screen)!r})",
        )

        # Restore the last row: the page re-reads where it is, and focus
        # stays on the trash list -- on the row now at that index, clamped.
        await pilot.press("shift+tab")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: _focused_id(screen) == "library-notes-trash-restore-4",
            lambda: f"Shift+Tab to the last Restore (focus: {_focused_id(screen)!r})",
        )
        await _restore_focused_row(
            host, pilot, screen, seed, "Deleted 25", then_range="21–24 of 24"
        )
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: getattr(screen.focused, "note_title", "") == "Deleted 24",
            lambda: f"focus on the row now last (focus: {_focused_id(screen)!r})",
        )

        # Restore the other four: the page empties, so it clamps to page 1
        # -- never an empty page 2 -- and one page left means no controls.
        for title, after in (
            ("Deleted 24", "21–23 of 23"),
            ("Deleted 23", "21–22 of 22"),
            ("Deleted 22", "21 of 21"),
            ("Deleted 21", "1–20 of 20"),
        ):
            await _restore_focused_row(
                host, pilot, screen, seed, title, then_range=after
            )
            # Review R0 m-8: a one-row page names its row, not a range (a
            # substring wait cannot pin that -- "21–21 of 21" contains it).
            assert "21–21" not in _painted(host)
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: getattr(screen.focused, "note_title", "") == "Deleted 01",
            lambda: f"focus on the first row of page 1 (focus: {_focused_id(screen)!r})",
        )
        final = _painted(host)
        assert "Page 1 of 1" not in final and "Page 2" not in final
        assert not screen.query("#library-notes-trash-previous")
        assert not screen.query("#library-notes-trash-next")
        assert seed.deleted() == set(TITLES[:20])

        # A real key still works: Escape leaves for the list, whose opener
        # names the exact total that is left.
        await pilot.press("escape")
        await _until(
            pilot,
            lambda: "Recently deleted (20)" in _painted(host),
            lambda: f"the list's opener after Escape: {_painted(host)}",
        )
