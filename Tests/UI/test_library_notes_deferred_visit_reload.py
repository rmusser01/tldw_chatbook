"""Notes: a visit reload that a running mutation held back runs once it ends.

TASK-34000.29 (N-14), review R0 M-1, ruling T7-f ("deferred, never skipped").
Returning to Library (closing F1 help, a picker, any screen over it) reloads
the whole Notes tree, so changes made elsewhere while it was hidden show up.
A dialog's choice starts a Notes mutation as the dialog closes, and that
reload would clear the tree under it, so it stands down while the Notes
interlock is held. ``9a95797a2b`` stood down and forgot: the reload never
ran. Now it is owed, and the interlock's release runs it exactly once.

One boot of the real ``LibraryScreen`` over a real ChaChaNotes database
(``library_notes_bulk_support``). The mutation is a real ``add_placement``
held open at its database write; the placement is read back through a
second connection.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.screen import ModalScreen

from Tests.UI.library_notes_bulk_support import (
    mounted,
    note_rows,
    notes_ready,
    until,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

WIDE = (160, 45)


class _Overlay(ModalScreen[None]):
    """Any screen over Library; closing it resumes Library."""


async def test_a_visit_reload_held_back_by_a_running_mutation_runs_once_it_ends(
    tmp_path, monkeypatch
) -> None:
    host, seed = mounted(tmp_path)
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)

        reloads: list[str] = []
        real_begin = LibraryScreen._begin_library_notes_tree_visit

        def counting_begin(self) -> None:
            reloads.append("full reload")
            real_begin(self)

        monkeypatch.setattr(
            LibraryScreen, "_begin_library_notes_tree_visit", counting_begin
        )

        # Hold one real placement write open at the database call.
        service = seed.profile.scope_service
        real_attach = service.attach_note_to_folder
        writing, release = asyncio.Event(), asyncio.Event()

        async def held_attach(**kwargs):
            writing.set()
            await release.wait()
            return await real_attach(**kwargs)

        monkeypatch.setattr(service, "attach_note_to_folder", held_attach)
        screen._schedule_library_notes_tree_mutation(
            "add_placement",
            folder_id=seed.folder_ids["Recipes"],
            note_id=seed.note_ids["Ideas inbox"],
        )
        await until(pilot, writing.is_set, "the placement write to start")
        assert screen._library_notes_mutation_fenced()

        # Library resumes while the mutation still runs.
        host.push_screen(_Overlay())
        await until(pilot, lambda: isinstance(host.screen, _Overlay), "the overlay")
        host.pop_screen()
        await until(pilot, lambda: host.screen is screen, "Library to resume")
        await pilot.pause(0.3)
        assert reloads == [], "the tree was reloaded under the running mutation"
        assert screen._library_notes_mutation_fenced()

        # The mutation ends: the held-back reload runs, once.
        release.set()
        await until(
            pilot,
            lambda: not screen._library_notes_mutation_fenced(),
            "the placement write to finish",
        )
        await until(
            pilot,
            lambda: len(reloads) >= 1,
            lambda: f"the held-back visit reload after the mutation (ran {reloads})",
        )
        await pilot.pause(0.5)
        assert reloads == ["full reload"], reloads
        assert seed.note_ids["Ideas inbox"] in seed.placed_in("Recipes")

        # The reload's pages land, and the tree takes a real key again.
        await until(pilot, lambda: bool(note_rows(screen)), "the reloaded rows")
        first = note_rows(screen)[0]
        screen.set_focus(first)
        await until(pilot, lambda: screen.focused is first, "a note row to take focus")
        await pilot.press("down")
        await until(
            pilot,
            lambda: screen.focused is not first
            and screen.focused is not None
            and screen.focused.has_class("library-notes-row"),
            lambda: f"Down to move to the next row (focus {screen.focused!r})",
        )
