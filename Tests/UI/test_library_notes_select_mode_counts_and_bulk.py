"""Notes select mode: the count it names, Space, bulk Move, and a strip that fits.

TASK-34000.29 (N-14), review notes-library-ux-review-2026-10-02, and
TASK-34000.82, verified on ``9097b11dd7``:

- AC#3: the strip read "Select all 28 shown" over the 20 rows the tree had
  rendered -- every compose ran the in-place responsive path afterwards, and
  that path counted the FLAT list. Pressing it selected a third count, read
  from a freshly built projection.
- AC#4: Space did nothing on a note row in select mode (Media binds it; Notes
  did not), so each row cost Enter and Down.
- AC#5: Export was the only bulk action. "Actions…" now moves the checked
  notes to a folder, or adds/removes one keyword on all of them; a receipt
  names the count. Bulk delete stays with TASK-32635.
- .82: at 160x45 with a note open the strip painted
  " 0 selected   Done     Select all 27 shown     Clear     ○  " -- Export
  selected cut off at the pane edge; at 120x36 it was gone entirely. The
  compact rule followed the terminal width, not the pane the strip lives in.

Four boots (one per test) of the real ``LibraryScreen`` with the production
stylesheet over a real ChaChaNotes database (``library_notes_bulk_support``).
Every gesture is a real key or click, or a focus asserted before its key;
every claim is painted text or the database read back through a second
connection. The keyword arms live in the ``_extended`` sibling, to keep this
file inside the PR lane's 30-second budget.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from Tests.UI.library_notes_bulk_support import (
    RENDERED,
    TRIP_TITLES,
    enter_select_mode,
    folder_row,
    mounted,
    note_rows,
    notes_ready,
    painted,
    row_for,
    strip_text,
    until,
)
from tldw_chatbook.Widgets.Library.library_note_folder_dialog import (
    LibraryNoteFolderTargetDialog,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

WIDE = (160, 45)


def _selected(screen) -> int:
    return screen._notes_state.row_selection.count


async def test_select_all_names_the_rendered_count_and_selects_exactly_that_many(
    tmp_path,
) -> None:
    """AC#3: the label, the press and the compact label agree on one count."""
    host, _seed = mounted(tmp_path)
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        await enter_select_mode(host, pilot, screen)
        await until(
            pilot,
            lambda: f"Select all {RENDERED} shown" in strip_text(host, screen),
            lambda: f"'Select all {RENDERED} shown' over {RENDERED} rows: "
            f"{strip_text(host, screen)!r}",
        )

        await pilot.click("#library-notes-select-all")
        await until(
            pilot,
            lambda: f"{RENDERED} selected" in painted(host),
            lambda: f"'{RENDERED} selected': {painted(host)}",
        )
        assert _selected(screen) == RENDERED
        assert all(str(row.label).startswith("☑") for row in note_rows(screen))

        # Export and back: the strip still names the rows it would select.
        await pilot.click("#library-notes-export-selected")
        await until(
            pilot,
            lambda: bool(screen.query("#library-export-submit")),
            lambda: f"the export canvas: {painted(host)}",
        )
        await pilot.press("escape")
        await until(
            pilot,
            lambda: f"Select all {RENDERED} shown" in strip_text(host, screen),
            lambda: f"the strip after Export: {strip_text(host, screen)!r}",
        )

        # The compact spelling counts the same rows (the in-place path read
        # the flat list: "All 28").
        await pilot.resize_terminal(80, 24)
        await until(
            pilot,
            lambda: f"All {RENDERED}" in strip_text(host, screen),
            lambda: f"'All {RENDERED}' at 80x24: {strip_text(host, screen)!r}",
        )
        assert "All 28" not in strip_text(host, screen)
        await pilot.resize_terminal(*WIDE)

        # A filter: Done, filter Trip, Select -- three rows, three selected.
        await pilot.click("#library-notes-select-toggle")
        await until(pilot, lambda: not screen._notes_state.select_mode, "Done")
        await pilot.click("#library-notes-filter")
        await pilot.press(*"Trip", "enter")
        await until(
            pilot,
            lambda: len(note_rows(screen)) == len(TRIP_TITLES)
            and all(title in painted(host) for title in TRIP_TITLES),
            lambda: f"the three Trip rows: {[str(r.label) for r in note_rows(screen)]}",
        )
        await enter_select_mode(host, pilot, screen)
        await until(
            pilot,
            lambda: "Select all 3 shown" in strip_text(host, screen),
            lambda: f"'Select all 3 shown': {strip_text(host, screen)!r}",
        )
        await pilot.click("#library-notes-select-all")
        await until(pilot, lambda: "3 selected" in painted(host), "'3 selected'")
        assert _selected(screen) == 3

        # A real key still works: Escape leaves select mode.
        await pilot.press("escape")
        await until(pilot, lambda: not screen._notes_state.select_mode, "Escape")


async def test_space_toggles_the_focused_note_row_and_nothing_else(tmp_path) -> None:
    """AC#4, and the review focus: Space claims a note row, never a text field."""
    host, _seed = mounted(tmp_path)
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        await enter_select_mode(host, pilot, screen)
        for _ in range(30):
            if getattr(screen.focused, "note_id", ""):
                break
            await pilot.press("tab")
            await pilot.pause()
        first = screen.focused
        assert getattr(first, "note_id", ""), screen.focused

        await pilot.press("space")
        await until(
            pilot,
            lambda: _selected(screen) == 1 and str(first.label).startswith("☑"),
            lambda: f"Space to check {first.label!s}: {_selected(screen)} selected",
        )
        assert screen._notes_state.row_selection.is_selected(first.note_id)
        await until(pilot, lambda: "1 selected" in painted(host), "'1 selected'")
        assert screen.focused is first
        # The footer advertises the key exactly where it works.
        await until(
            pilot,
            lambda: "space toggle" in painted(host).splitlines()[-1],
            lambda: f"'space toggle' in the footer: {painted(host).splitlines()[-1]!r}",
        )

        await pilot.press("space")
        await until(
            pilot,
            lambda: _selected(screen) == 0 and str(first.label).startswith("☐"),
            "Space again to uncheck it",
        )

        # A real key moves on; Space checks the row it landed on, only that one.
        await pilot.press("down")
        await until(
            pilot,
            lambda: screen.focused is not first
            and bool(getattr(screen.focused, "note_id", "")),
            "Down to the next row",
        )
        second = screen.focused
        await pilot.press("space")
        await until(pilot, lambda: _selected(screen) == 1, "Space on the second row")
        assert screen._notes_state.row_selection.ids == frozenset({second.note_id})

        # A text field keeps its Space: typed into the filter, nothing toggled.
        await pilot.click("#library-notes-filter")
        await pilot.press("a", "space", "b")
        await until(
            pilot,
            lambda: screen.query_one("#library-notes-filter", Input).value == "a b",
            lambda: "the filter to read 'a b', got "
            f"{screen.query_one('#library-notes-filter', Input).value!r}",
        )
        assert _selected(screen) == 1
        assert screen._notes_state.select_mode

        # A Notes pane grip keeps its own Space (ruling T7-c): it toggles its
        # pane in select mode exactly as outside it, and checks nothing.
        await _space_toggles_the_grip(pilot, screen)
        assert screen._notes_state.select_mode and _selected(screen) == 1

        # Done: Space is not claimed there (a Textual Button presses on Enter
        # only), so nothing toggles; Enter, a real key, leaves select mode.
        done = screen.query_one("#library-notes-select-toggle", Button)
        done.focus()
        await until(pilot, lambda: screen.focused is done, "focus on Done")
        await pilot.press("space")
        await pilot.pause(0.2)
        assert screen._notes_state.select_mode and _selected(screen) == 1
        await pilot.press("enter")
        await until(pilot, lambda: not screen._notes_state.select_mode, "Enter on Done")

        # ...and outside select mode the grip's Space does the same thing.
        await _space_toggles_the_grip(pilot, screen)


async def _space_toggles_the_grip(pilot, screen) -> None:
    """Space on the first Notes pane grip flips its pane, then flips it back."""
    grip = next(
        grip
        for grip in screen.query(".library-adaptive-reader-pane-grip")
        if any(a.has_class("library-notes-route") for a in grip.ancestors)
    )
    for _ in range(2):
        grip.focus()
        await until(pilot, lambda: screen.focused is grip, "focus on the Notes grip")
        was_open = grip.pane_open
        canvas = screen.query_one("#library-notes-canvas")
        before = canvas.region
        await pilot.press("space")
        # Wait on the layout, not only the flag: the pane has really moved.
        await until(
            pilot,
            lambda: grip.pane_open is (not was_open) and canvas.region != before,
            lambda: f"Space on the {grip.pane} grip to toggle its pane "
            f"(open={grip.pane_open}, select mode={screen._notes_state.select_mode})",
        )
        await pilot.pause(0.2)


async def test_actions_move_moves_every_selected_note_and_names_the_count(
    tmp_path,
) -> None:
    """AC#5: one action moves every checked note and names the count.

    Three unfiled notes and one FILED note (Bread, in Recipes): the filed one
    is what proves Move is a move -- placed in the destination AND lifted out
    of its source folder (review R0 I-2). Escape at each step (the chooser,
    the picker) writes nothing: the ``_extended`` sibling's partial-move test,
    moved there to keep this file inside its 30-second budget.
    """
    host, seed = mounted(tmp_path)
    picked = ("Ideas inbox", "Reading list", "Trip Oslo", "Bread")
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        await pilot.click(f"#{folder_row(screen, 'Recipes').id}")
        await until(
            pilot,
            lambda: any("Bread" in str(row.label) for row in note_rows(screen)),
            lambda: f"Recipes to open: {painted(host)}",
        )
        await enter_select_mode(host, pilot, screen)
        for title in picked:
            await pilot.click(f"#{row_for(screen, title).id}")
        await until(pilot, lambda: _selected(screen) == 4, "four checked rows")
        bread_in_recipes = seed.placement_rows("Recipes", "Bread")
        assert [deleted for deleted, _ in bread_in_recipes] == [0], bread_in_recipes

        # Actions… opens the three choices; Move to folder… opens the
        # all-folders picker. (Escape at each step: the ``_extended`` sibling.)
        await pilot.click("#library-notes-select-actions")
        await until(
            pilot,
            lambda: bool(screen.query("#library-notes-select-action-move")),
            lambda: f"the Actions chooser: {painted(host)}",
        )
        for label in ("Move to folder…", "Add keyword…", "Remove keyword…"):
            assert label in painted(host), label
        await pilot.click("#library-notes-select-action-move")
        await until(
            pilot,
            lambda: isinstance(host.screen, LibraryNoteFolderTargetDialog)
            and bool(host.screen.query(".library-note-folder-target-row")),
            "the picker's first page",
        )
        await pilot.press(*"Thesis")
        await until(
            pilot,
            lambda: [
                str(row.label)
                for row in host.screen.query(".library-note-folder-target-row")
            ]
            == ["Projects / Thesis"],
            "the picker narrowed to Projects / Thesis",
        )
        await pilot.press("down", "enter")
        await until(
            pilot,
            lambda: "Moved 4 of 4 notes to Projects / Thesis" in painted(host),
            lambda: f"the receipt: {painted(host)}",
        )
        moved = {seed.note_ids[title] for title in picked}
        assert seed.placed_in("Projects / Thesis") == moved | {
            seed.note_ids["Thesis outline"]
        }
        assert not screen._notes_state.select_mode
        # Bread left Recipes: its one placement there is no longer live, and
        # the detach bumped that placement's version; Soup stayed.
        assert seed.placed_in("Recipes") == {seed.note_ids["Soup"]}
        (old_version,) = [version for _, version in bread_in_recipes]
        assert seed.placement_rows("Recipes", "Bread") == [(1, old_version + 1)]

        # Focus lands on Select (the strip it came from is gone) and stays
        # there through the tree refresh; a real key then works: Enter
        # enters select mode again.
        await until(
            pilot,
            lambda: getattr(screen.focused, "id", "") == "library-notes-select-toggle"
            and not screen._notes_state.mutation_in_flight
            and not any(
                branch.loading for branch in screen._notes_state.tree_branches.values()
            ),
            lambda: f"focus on Select after the move, got {screen.focused!r}",
        )
        await pilot.pause(0.2)
        assert getattr(screen.focused, "id", "") == "library-notes-select-toggle"
        await pilot.press("enter")
        await until(pilot, lambda: screen._notes_state.select_mode, "Enter on Select")


async def test_the_select_strip_paints_every_action_beside_an_open_note(
    tmp_path,
) -> None:
    """.82 AC#1-#3: every select-mode action is on the pane, at three sizes."""
    host, _seed = mounted(tmp_path)
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        await pilot.click(f"#{row_for(screen, 'Ideas inbox').id}")
        await until(
            pilot,
            lambda: screen._library_note_session.snapshot is not None
            and bool(screen.query("#library-note-body")),
            "the note to open beside the list",
        )
        await enter_select_mode(host, pilot, screen)
        await pilot.click(f"#{row_for(screen, 'Reading list').id}")
        await until(pilot, lambda: _selected(screen) == 1, "one checked row")

        def on_pane(size: tuple[int, int]) -> str:
            canvas = screen.query_one("#library-notes-canvas")
            problems = []
            for button_id in (
                "library-notes-select-toggle",
                "library-notes-select-all",
                "library-notes-select-clear",
                "library-notes-export-selected",
                "library-notes-select-actions",
            ):
                buttons = list(screen.query(f"#{button_id}").results(Button))
                if not buttons:
                    problems.append(f"{button_id} missing")
                    continue
                region = buttons[0].region
                if region.width == 0 or region.right > canvas.region.right:
                    problems.append(f"{button_id} at {region} on {canvas.region}")
            text = strip_text(host, screen)
            for word in ("Done", "Clear", "Export", "Actions…"):
                if word not in text:
                    problems.append(f"{word!r} not painted")
            if not problems:
                return ""
            return f"{size}: {'; '.join(problems)} -- painted {text!r}"

        await until(
            pilot,
            lambda: not on_pane(WIDE),
            lambda: on_pane(WIDE) or "the strip at 160x45",
        )
        assert "Select all" in strip_text(host, screen)

        await pilot.resize_terminal(120, 36)
        await until(pilot, lambda: not on_pane((120, 36)), lambda: on_pane((120, 36)))

        # Compact: the list is a stage of its own and an open note owns the
        # screen, so leave select mode (Escape), then the note (Escape), and
        # select again from the list.
        await pilot.press("escape")
        await until(pilot, lambda: not screen._notes_state.select_mode, "Escape")
        await pilot.resize_terminal(80, 24)
        await until(pilot, lambda: screen._notes_state.compact, "the compact shell")
        await pilot.press("escape")
        await until(
            pilot,
            lambda: bool(screen.query("#library-notes-select-toggle"))
            and screen.query_one("#library-notes-select-toggle").region.width > 0,
            lambda: f"the list stage at 80x24: {painted(host)}",
        )
        await enter_select_mode(host, pilot, screen)
        await pilot.click(f"#{row_for(screen, 'Reading list').id}")
        await until(pilot, lambda: _selected(screen) == 1, "one checked row at 80x24")
        await until(pilot, lambda: not on_pane((80, 24)), lambda: on_pane((80, 24)))
        assert f"All {RENDERED}" in strip_text(host, screen)

        # Clickable, not merely painted: Actions… opens its chooser.
        await pilot.click("#library-notes-select-actions")
        await until(
            pilot,
            lambda: bool(screen.query(".library-notes-select-action-choice")),
            "the chooser from a click at 80x24",
        )
