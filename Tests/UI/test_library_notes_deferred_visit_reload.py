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

Final review I-1 (M1): the last arm was flaky (1 in ~22). It waited for the
FIRST slice of the owed reload only, held a row widget across the remaining
slices' recompose, moved focus with ``set_focus`` and pressed Down; a later
slice landing in between put focus on the role fallback (the filter) or
reordered the rows (a folder row). The arm now settles on the whole reload
(no rows reloading, no branch loading, no pending identity), re-queries the
row, and drives focus with real keys only. The root ``folders`` slice of the
owed reload is slowed (``SLOW_SLICE``) so the note rows land FIRST and the
folders slice recomposes the tree after them -- the recorded failure shape,
made deterministic: the old arm is RED under it, this one GREEN. T7 N-1: the
focus the reload lands on is asserted by role before any key.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.screen import ModalScreen

from Tests.UI.library_notes_bulk_support import (
    folder_row,
    mounted,
    note_rows,
    notes_ready,
    row_for,
    until,
)
from tldw_chatbook.Library.library_notes_tree_paging import NotesBranchKey
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

WIDE = (160, 45)
#: How long the owed reload's root folders slice is held back, so the note
#: rows (the placements slice) are painted and focusable while the tree is
#: still mid-reload. Long enough to be certain the old arm acted inside it.
SLOW_SLICE = 0.4


class _Overlay(ModalScreen[None]):
    """Any screen over Library; closing it resumes Library."""


def _reload_settled(screen) -> bool:
    """The owed reload has landed whole: nothing reloading, loading or pending."""
    state = screen._notes_state
    return (
        not LibraryScreen._library_notes_rows_reloading(state)
        and not any(branch.loading for branch in state.tree_branches.values())
        and state.pending_focus_identity is None
        and bool(note_rows(screen))
    )


def _focus_role(screen) -> str:
    return screen._library_notes_semantic_role(screen.focused)


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

        # The single-note flow: the user is ON the note the mutation moves
        # (a Move starts from that row), with Recipes open so the note's new
        # placement is a rendered row the mutation's own landing can reach
        # (T7 N-1). Focus reaches the row by a real Down from its neighbour.
        moved_id = seed.note_ids["Ideas inbox"]
        await pilot.click(f"#{folder_row(screen, 'Recipes').id}")
        await until(
            pilot,
            lambda: any("Bread" in str(row.label) for row in note_rows(screen))
            and _reload_settled(screen),
            "Recipes to open and the tree to settle",
        )
        # (Measured under 41-way load: a late expansion recompose pruned the
        # row between the click and the first key.)
        await pilot.pause(0.1)
        assert _reload_settled(screen)
        ideas_row = row_for(screen, "Ideas inbox")
        screen.set_focus(ideas_row)
        await until(pilot, lambda: screen.focused is ideas_row, "the Ideas inbox row")
        await pilot.press("down")
        await until(
            pilot,
            lambda: screen.focused is not ideas_row
            and screen.focused is not None
            and screen.focused.has_class("library-notes-row"),
            lambda: f"Down to the next row (focus {screen.focused!r})",
        )
        await pilot.press("up")
        await until(
            pilot,
            lambda: getattr(screen.focused, "note_id", "") == moved_id,
            lambda: f"Up back onto Ideas inbox (focus {screen.focused!r})",
        )
        started_on = _focus_role(screen)
        assert started_on.startswith("note-placement:"), started_on

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

        # Hold the owed reload's root folders slice: the placements slice
        # (the note rows) lands first, and the folders slice is released the
        # moment a ``set_focus`` lands on a note row -- the window the old
        # arm acted inside (its ``set_focus(first)``, then Down) -- or after
        # SLOW_SLICE when nothing does, as this arm never does (it drives
        # focus by real keys only once the reload is whole). Measured: the
        # old arm was red 1 run in 3 under this (1 in ~22 bare); the released
        # slice still needs its read before it lands, so the key can win the
        # race. (The slice request's worker runs
        # ``_run_library_notes_tree_slice_request`` directly.)
        real_slice = LibraryScreen._run_library_notes_tree_slice_request
        real_set_focus = LibraryScreen.set_focus
        slowed: list[NotesBranchKey] = []
        row_focused = asyncio.Event()

        async def gated_folders_slice(self, key, **kwargs):
            if key == NotesBranchKey(None, "folders") and reloads:
                slowed.append(key)
                try:
                    await asyncio.wait_for(row_focused.wait(), timeout=SLOW_SLICE)
                except TimeoutError:
                    pass
            return await real_slice(self, key, **kwargs)

        def set_focus_opens_the_gate(self, widget, *args, **kwargs):
            if reloads and widget is not None and widget.has_class("library-notes-row"):
                row_focused.set()
            return real_set_focus(self, widget, *args, **kwargs)

        monkeypatch.setattr(
            LibraryScreen, "_run_library_notes_tree_slice_request", gated_folders_slice
        )
        monkeypatch.setattr(LibraryScreen, "set_focus", set_focus_opens_the_gate)

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

        # The reload lands WHOLE: no rows reloading, no branch loading, no
        # pending identity -- and holds across one more pause. The slowed
        # folders slice must have been the one to land last.
        for _ in range(2):
            await until(
                pilot,
                lambda: _reload_settled(screen),
                lambda: (
                    "the owed reload to settle whole: "
                    f"branches={ {k: b.loading for k, b in screen._notes_state.tree_branches.items()} } "
                    f"pending={screen._notes_state.pending_focus_identity!r}"
                ),
            )
            await pilot.pause(0.1)
        assert _reload_settled(screen)
        assert slowed == [NotesBranchKey(None, "folders")], slowed

        # T7 N-1: the mutation's own landing -- the reconcile SELECTS the
        # moved note's new placement under Recipes (``desired_target``,
        # ``focus=False``) -- survives the owed full reload, and the focus
        # landing names a role (never ``''``, a capture that read nothing).
        # Measured on every run at 29d2272b95 and here: the role is the
        # navigator's own fallback, the filter, for a mutation scheduled
        # without an opener control; so the keys below start there.
        recipes_id = seed.folder_ids["Recipes"]
        selected = str(screen._notes_state.tree_selected_placement_id or "")
        assert moved_id in selected and recipes_id in selected, (
            f"the owed reload lost the mutation's landing: selected {selected!r}, "
            f"started on {started_on!r}"
        )
        landed_on = _focus_role(screen)
        assert landed_on, f"the landing names no role (focus {screen.focused!r})"
        assert screen.focused is not None and screen.focused.id in {
            "library-notes-filter",
            *(row.id for row in note_rows(screen)),
        }, (landed_on, screen.focused)

        # The tree takes real keys again, with no ``set_focus``: Down from
        # the filter is the T5-a hand-off onto the tree's first row (a
        # folder row here), Down walks to the first note row (Recipes is
        # open: Bread), and the next Down moves to the next note row (Soup)
        # -- never a folder row. The rows are re-queried after the settle,
        # never held across it.
        for _ in range(8):
            if screen.focused is not None and screen.focused.has_class(
                "library-notes-row"
            ):
                break
            previous = screen.focused
            await pilot.press("down")
            await until(
                pilot,
                lambda: screen.focused is not None and screen.focused is not previous,
                lambda: f"Down to move focus (focus {screen.focused!r})",
            )
        assert screen.focused.has_class("library-notes-row"), screen.focused
        before = screen.focused
        assert before in note_rows(screen), screen.focused
        await pilot.press("down")
        await until(
            pilot,
            lambda: screen.focused is not before
            and screen.focused is not None
            and screen.focused.has_class("library-notes-row"),
            lambda: f"Down to move to the next row (focus {screen.focused!r})",
        )
        assert not screen.focused.has_class("library-notes-folder-row")
