"""Notes bulk actions: the keyword arms, the open note, and a partial move.

TASK-34000.29 (N-14), the ``_extended`` sibling of the PR-gated
``test_library_notes_select_mode_counts_and_bulk.py`` (same seed and
harness, ``library_notes_bulk_support``). Three more boots of the real
``LibraryScreen`` with the production stylesheet over a real ChaChaNotes
database; every claim is painted text or the database read back through a
second connection.

- Add keyword… / Remove keyword… change one keyword on every checked note,
  and the receipt names the count -- including the notes that already had
  (or never had) it. Escape in the dialog writes nothing.
- The open note's editor writes its whole keyword list on every save, so a
  bulk keyword change that touched it must reach the editor, or the next
  keystroke in the body silently writes the old list back.
- A note whose move fails part-way is named in the receipt and stays
  checked, with select mode kept, so the same action retries only it. The
  failure is one injected refusal on the real scope service (the source
  detach of one note), the Task 6 precedent for an outcome no real database
  produces on demand.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, TextArea

from Tests.UI.library_notes_bulk_support import (
    enter_select_mode,
    mounted,
    notes_ready,
    painted,
    row_for,
    until,
)
from tldw_chatbook.Widgets.Library.library_note_folder_dialog import (
    LibraryNoteFolderTargetDialog,
    LibraryNoteKeywordDialog,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

WIDE = (160, 45)


def _selected(screen) -> int:
    return screen._notes_state.row_selection.count


async def _check(host, pilot, screen, titles) -> None:
    for title in titles:
        await pilot.click(f"#{row_for(screen, title).id}")
    await until(
        pilot,
        lambda: _selected(screen) == len(titles),
        lambda: f"{len(titles)} checked rows, have {_selected(screen)}",
    )


async def _choose_action(host, pilot, screen, option_id: str) -> None:
    await pilot.click("#library-notes-select-actions")
    await until(
        pilot,
        lambda: bool(screen.query(f"#{option_id}")),
        lambda: f"the Actions chooser: {painted(host)}",
    )
    await pilot.click(f"#{option_id}")


async def _keyword(host, pilot, screen, option_id: str, keyword: str, title: str):
    await _choose_action(host, pilot, screen, option_id)
    await until(
        pilot,
        lambda: isinstance(host.screen, LibraryNoteKeywordDialog)
        and title in painted(host),
        lambda: f"the keyword dialog titled {title!r}: {painted(host)}",
    )
    await pilot.press(*keyword, "enter")
    await until(pilot, lambda: host.screen is screen, "the dialog to apply")


async def test_actions_add_and_remove_a_keyword_on_every_selected_note(
    tmp_path,
) -> None:
    host, seed = mounted(tmp_path)
    picked = ("Ideas inbox", "Reading list", "Trip Oslo")
    ids = {seed.note_ids[title] for title in picked}
    # One of the three already carries the keyword, so the receipt says so.
    travel = seed.profile.db.add_keyword("travel")
    seed.profile.db.link_note_to_keyword(seed.note_ids["Trip Oslo"], travel)
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        await enter_select_mode(host, pilot, screen)
        await _check(host, pilot, screen, picked)

        # Escape in the dialog writes nothing and keeps the selection.
        await _choose_action(
            host, pilot, screen, "library-notes-select-action-add-keyword"
        )
        await until(
            pilot,
            lambda: isinstance(host.screen, LibraryNoteKeywordDialog),
            "the keyword dialog",
        )
        await pilot.press(*"food", "escape")
        await until(pilot, lambda: host.screen is screen, "the dialog to close")
        await pilot.pause(0.3)
        await until(
            pilot,
            lambda: getattr(screen.focused, "id", "") == "library-notes-select-actions",
            lambda: f"focus back on Actions…, got {screen.focused!r}",
        )
        assert seed.keyword_links("food") == set()
        assert seed.keyword_links("travel") == {seed.note_ids["Trip Oslo"]}
        assert screen._notes_state.select_mode and _selected(screen) == 3

        await _keyword(
            host,
            pilot,
            screen,
            "library-notes-select-action-add-keyword",
            "travel",
            "Add a keyword to 3 notes",
        )
        await until(
            pilot,
            lambda: "Added 'travel' to 2 notes · 1 already had it" in painted(host),
            lambda: f"the add receipt: {painted(host)}",
        )
        assert seed.keyword_links("travel") == ids
        assert not screen._notes_state.select_mode

        await enter_select_mode(host, pilot, screen)
        await _check(host, pilot, screen, picked)
        await _keyword(
            host,
            pilot,
            screen,
            "library-notes-select-action-remove-keyword",
            "Travel",
            "Remove a keyword from 3 notes",
        )
        await until(
            pilot,
            lambda: "Removed 'travel' from 3 notes" in painted(host),
            lambda: f"the remove receipt: {painted(host)}",
        )
        assert seed.keyword_links("travel") == set()
        assert not screen._notes_state.select_mode

        # A real key still works afterwards: Enter on Select re-enters it.
        screen.query_one("#library-notes-select-toggle").focus()
        await pilot.press("enter")
        await until(pilot, lambda: screen._notes_state.select_mode, "Enter on Select")


async def test_a_bulk_keyword_on_the_open_note_survives_its_next_save(
    tmp_path,
) -> None:
    host, seed = mounted(tmp_path)
    open_id = seed.note_ids["Ideas inbox"]
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        await pilot.click(f"#{row_for(screen, 'Ideas inbox').id}")
        await until(
            pilot,
            lambda: screen._library_note_session.snapshot is not None
            and screen._library_note_session.snapshot.note_id == open_id
            and screen._notes_state.editor_armed,
            "the note to open beside the list",
        )
        await enter_select_mode(host, pilot, screen)
        await _check(host, pilot, screen, ("Ideas inbox", "Reading list"))
        await _keyword(
            host,
            pilot,
            screen,
            "library-notes-select-action-add-keyword",
            "travel",
            "Add a keyword to 2 notes",
        )
        await until(
            pilot,
            lambda: "Added 'travel' to 2 notes" in painted(host),
            lambda: f"the receipt: {painted(host)}",
        )
        # The editor shows the keyword the database now holds...
        await until(
            pilot,
            lambda: bool(screen.query("#library-note-keywords"))
            and "travel"
            in screen.query_one("#library-note-keywords", Input).value,
            lambda: "the open note's keywords field to list travel, got "
            f"{[i.value for i in screen.query('#library-note-keywords').results(Input)]}",
        )
        # ...and a real edit to the body, saved, does not write the old list
        # back over it.
        body = screen.query_one("#library-note-body", TextArea)
        await pilot.click("#library-note-body")
        await until(pilot, lambda: screen.focused is body, "focus in the body")
        await pilot.press("ctrl+end", "x")
        await until(pilot, lambda: body.text.endswith("x"), "the typed key")
        await pilot.click("#library-note-save")
        await until(
            pilot,
            lambda: seed.profile.note(open_id)["content"].endswith("x"),
            "the edit to save",
        )
        assert open_id in seed.keyword_links("travel")
        assert seed.keyword_links("travel") == {
            open_id,
            seed.note_ids["Reading list"],
        }


async def test_a_partial_bulk_move_names_what_did_not_move_and_keeps_it_checked(
    tmp_path, monkeypatch
) -> None:
    host, seed = mounted(tmp_path)
    service = seed.profile.scope_service
    soup = seed.note_ids["Soup"]
    real_detach = service.detach_note_from_folder

    async def refuse_soup(**kwargs):
        if kwargs.get("note_id") == soup:
            return False
        return await real_detach(**kwargs)

    monkeypatch.setattr(service, "detach_note_from_folder", refuse_soup)
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        # Soup lives in Recipes; open the folder so its rows are rendered.
        await pilot.click(f"#{_folder_row(screen, 'Recipes').id}")
        await until(
            pilot,
            lambda: any("Soup" in str(row.label) for row in _rows(screen)),
            lambda: f"Recipes to open: {painted(host)}",
        )
        await enter_select_mode(host, pilot, screen)
        await _check(host, pilot, screen, ("Soup", "Ideas inbox"))
        await _choose_action(host, pilot, screen, "library-notes-select-action-move")
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
            lambda: _selected(screen) == 1
            and not screen._notes_state.mutation_in_flight,
            "the move to finish",
        )
        receipt = (
            "Moved 1 of 2 notes to Projects / Thesis · 1 did not move — "
            "still checked, so you can try again"
        )
        assert screen._notes_state.notice == receipt
        assert screen._notes_state.select_mode
        assert screen._notes_state.row_selection.ids == frozenset({soup})
        await until(
            pilot,
            lambda: any(
                str(row.label).startswith("☑") and "Soup" in str(row.label)
                for row in _rows(screen)
            ),
            lambda: f"Soup still checked after the reload: {painted(host)}",
        )
        # Soup was placed in Thesis and kept in Recipes: never in neither.
        assert soup in seed.placed_in("Projects / Thesis")
        assert soup in seed.placed_in("Recipes")
        assert seed.note_ids["Ideas inbox"] in seed.placed_in("Projects / Thesis")


def _rows(screen):
    return list(screen.query(".library-notes-tree-note-row"))


def _folder_row(screen, label: str):
    for row in screen.query(".library-notes-folder-row"):
        if label in str(row.label):
            return row
    raise AssertionError(f"no folder row {label!r}")
