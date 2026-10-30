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

Final review I-1 (M1): the last arm was flaky (1 in ~22). It acted once the
FIRST slice of the owed reload had landed: it took a row, moved focus to it
with ``set_focus`` and pressed Down. A later slice's canvas sync had already
captured Notes focus (the filter), and that slice's focus replay ran after
the ``set_focus``, so the Down started from the filter and handed off onto
the tree's first row, a folder row. The arm now settles on the whole reload
(no rows reloading, no branch loading, no pending identity), re-queries the
rows, and drives focus with real keys only.

What the pin proves, and how (residual R-1). The test forces that recorded
interleaving with events alone -- no timeout, no sleep, no timed slice:

- the owed reload's root ``folders`` slice is read only after the root
  placements slice (the note rows) has landed, so it recomposes the tree
  last;
- that slice's focus replay, captured before any arm acts, is held until a
  ``set_focus`` lands on a note row (the old arm's first act) and then runs
  on the screen's queue; this arm never calls ``set_focus`` and releases the
  replay explicitly;
- a Down the screen receives before that replay has run is held, without
  blocking the queue, and handled after it.

So the old arm's Down always starts from the replayed filter and lands on a
folder row: RED on every run (5 of 5, by an Edit mutation restoring the old
arm). This arm presses no key until the reload is whole, so no key of its is
ever held, and its Down walk ends on a note row: GREEN. T7 N-1: the focus the
reload lands on is asserted by role before any key.
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

        # The event-ordered injection (residual R-1: no timeout, no timer).
        # The recorded failure: a later slice's canvas sync captured Notes
        # focus (the filter) before the old arm's ``set_focus(first)``, and
        # that slice's post-recompose focus replay ran after it, putting
        # focus back on the filter before the arm's Down -- which handed off
        # onto the tree's first row, a folder row. Three events force that
        # order on every run:
        # * the owed reload's root ``folders`` slice is read only once the
        #   root placements slice (the note rows) has landed
        #   (``rows_landed``), so it recomposes the tree last;
        # * its focus replay (and any replay after it) is held until a
        #   ``set_focus`` lands on a note row -- the old arm's first act --
        #   and then runs on the screen's queue (``row_focused``). This arm
        #   never calls ``set_focus``; it releases the replay explicitly;
        # * a Down the screen receives before that replay has run
        #   (``slice_landed``) is held, never blocking the queue, and handled
        #   after it.
        # (The slice request's worker runs
        # ``_run_library_notes_tree_slice_request`` directly; the canvas's
        # post-recompose follow-up calls
        # ``_restore_library_notes_after_targeted_sync``.)
        root_folders = NotesBranchKey(None, "folders")
        root_placements = NotesBranchKey(None, "placements")
        real_slice = LibraryScreen._run_library_notes_tree_slice_request
        real_replay = LibraryScreen._restore_library_notes_after_targeted_sync
        real_set_focus = LibraryScreen.set_focus
        real_on_key = LibraryScreen.on_key
        gated: list[NotesBranchKey] = []
        held_replays: list[object] = []
        held_keys: list[str] = []
        rows_landed, folders_applied = asyncio.Event(), asyncio.Event()
        row_focused, slice_landed = asyncio.Event(), asyncio.Event()

        async def ordered_slice(self, key, **kwargs):
            if not reloads or key not in (root_folders, root_placements):
                return await real_slice(self, key, **kwargs)
            if key == root_folders:
                gated.append(key)
                await rows_landed.wait()
            try:
                return await real_slice(self, key, **kwargs)
            finally:
                (folders_applied if key == root_folders else rows_landed).set()

        def held_replay(self, identity) -> None:
            if folders_applied.is_set() and not row_focused.is_set():
                held_replays.append(identity)
                return
            real_replay(self, identity)

        def release_the_replay(owner) -> None:
            if row_focused.is_set():
                return
            row_focused.set()

            def replay() -> None:
                for identity in held_replays:
                    real_replay(owner, identity)
                slice_landed.set()

            owner.call_later(replay)

        def set_focus_releases_the_replay(self, widget, *args, **kwargs):
            result = real_set_focus(self, widget, *args, **kwargs)
            if (
                folders_applied.is_set()
                and widget is not None
                and widget.has_class("library-notes-row")
            ):
                release_the_replay(self)
            return result

        async def handle_after_the_landing(self, event) -> None:
            await slice_landed.wait()
            self.call_later(real_on_key, self, event)

        def on_key_holds_an_early_down(self, event) -> None:
            if event.key == "down" and reloads and not slice_landed.is_set():
                held_keys.append(event.key)
                event.stop()
                event.prevent_default()
                self.run_worker(handle_after_the_landing(self, event), group="held-down")
                return
            real_on_key(self, event)

        monkeypatch.setattr(
            LibraryScreen, "_run_library_notes_tree_slice_request", ordered_slice
        )
        monkeypatch.setattr(
            LibraryScreen, "_restore_library_notes_after_targeted_sync", held_replay
        )
        monkeypatch.setattr(LibraryScreen, "set_focus", set_focus_releases_the_replay)
        monkeypatch.setattr(LibraryScreen, "on_key", on_key_holds_an_early_down)

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

        # The window is open: the folders slice has been applied and its
        # recompose has run; only its focus replay waits.
        await until(pilot, lambda: bool(held_replays), "the folders slice's replay")

        # Nothing of this arm has released it. This arm never calls
        # ``set_focus``, so it releases the replay explicitly.
        assert not slice_landed.is_set()
        assert not row_focused.is_set(), "a note row took set_focus mid-reload"
        release_the_replay(screen)
        await until(pilot, slice_landed.is_set, "the folders slice to land")

        # The reload lands WHOLE: no rows reloading, no branch loading, no
        # pending identity -- and holds across one more pause. Exactly one
        # slice (the root folders slice) went through the gate.
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
        assert gated == [root_folders], gated

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
        assert held_keys == [], "a key of this arm was held: it pressed mid-reload"
