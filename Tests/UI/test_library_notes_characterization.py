"""Characterization pins for genuinely-unpressed Library Notes handlers.

Wave-8 Task 1 (notes series 1/N, state PR; recipe: ``backlog/docs/
library-decomposition-recipe.md``; the collections/ingest/prompts/media
series' own ``test_library_*_characterization.py`` files are the precedent
this mirrors). These pins exist BEFORE the Notes extraction moves any state,
so a later move that silently breaks one of these dispatch paths goes red
rather than green-but-vacuous.

**Scope, and why it is only three tests.** Notes carries **93**
``@on``-decorated notes-named handlers on ``LibraryScreen``, of which **58**
carry a string selector and 35 are Message-typed, against 16 dedicated
``Tests/UI/test_library_*note*.py`` files plus ``test_library_shell.py``,
``Tests/Library/`` (6 files), ``Tests/Notes/`` and the canvas suites under
``Tests/Widgets/Library/``. The census ran every selector against ALL of
``Tests/`` with exact id/class boundaries (a substring match scores
``#library-notes-select`` as covered off ``#library-notes-select-clear``),
derived the ``#<id>-N`` row spelling for class-bound handlers (rows are
pressed by id, never by class), and looked for a press/click/``post_message``
within +/-4 lines of each hit; every non-covered verdict was then READ rather
than trusted.

Result, measured on the tree BEFORE this file existed: **48 of the 58
selector-bound handlers** carry press evidence by that automatic pass, and
**three** of the ten remaining were overturned by reading:

- ``handle_library_notes_sort_choice`` is bound on the CLASS
  ``.library-notes-sort-choice`` but pressed by the derived id
  ``#library-notes-sort-oldest`` (the same bound-by-class/pressed-by-id
  shape the prompts and media series each recorded). That press lived in
  ``test_library_shell.py`` when this file was written; task-32128 briefly
  took Sort off the folder tree and task-32172 put it back, so the shell
  press is live again -- and it is now pinned in two more places besides:
  ``test_library_notes_wave_list.py`` --
  ``test_sort_is_operable_by_keyboard_on_the_flat_list`` (Tab + Enter
  through the real Button) and
  ``test_pressing_a_sort_option_applies_that_sort`` (task-32175).
- ``handle_library_notes_select_clear`` is activated by KEYBOARD, through
  ``_task10_activate_with_keyboard(screen, pilot,
  "#library-notes-select-clear")``, with the selection count asserted either
  side -- a call shape a ``.press()``-anchored window cannot see.
- Six of the remaining seven (``handle_library_note_export_markdown``, the
  three ``handle_library_notes_placement_*`` and two of the
  ``handle_library_notes_folder_*``) touch NO field this PR moves, so a
  characterization pin here would prove nothing about the move.

That leaves the two genuine gaps pinned below, plus one Message-typed
handler whose message class appears nowhere in ``Tests/`` at all:

- ``handle_library_notes_folder_new`` (``@on(Button.Pressed,
  "#library-notes-folder-new")``) -- the button is queried twice in
  ``Tests/Widgets/Library/test_library_notes_canvas.py`` on a STANDALONE
  canvas host (which cannot dispatch a SCREEN ``@on`` handler at all) and
  pressed nowhere. Reads the tree projection keyed by
  ``_library_notes_tree_selected_placement_id`` and writes
  ``_library_notes_notice`` on its protected-folder branch.
- ``handle_library_note_work_pane_editor_ready``
  (``@on(LibraryNoteWorkPane.EditorReady)``) -- the message class
  ``LibraryNoteWorkPane.EditorReady`` occurs in ZERO test files; the handler
  reads ``_library_notes_view`` and ``_selected_note_id`` in its route guard
  and arms ``_library_note_editor_armed`` past it. Both routes are pinned,
  because a guard nobody exercises is exactly the kind of branch a move can
  invert without a red.

No live bugs were found writing these: both are coverage gaps, not behavior
bugs. Each test asserts through a signal provably tied to the handler's own
logic (the exact modal type the press pushes; the armed flag flipping only on
the editor route) rather than a bare DOM end-state, per recipe §3's warning
that an end-state assertion can be satisfied by an unrelated coincidence --
which is also why the editor-ready pins assert the SAME flag in both
directions from the same starting value.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _open_note_editor,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Widgets.Library.library_note_work_pane import LibraryNoteWorkPane


async def _open_notes_tree(host, pilot):
    """Mount the Library shell, select Browse Notes, await the folder tree."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, "#library-notes-folder-new")
    return screen


@pytest.mark.asyncio
async def test_notes_new_folder_opens_the_folder_name_dialog() -> None:
    """"New folder" with nothing selected pushes the name dialog.

    The button is only ever QUERIED, on a standalone canvas host that cannot
    dispatch this screen handler. This presses it for real on the Library
    screen and pins the exact modal the press produces -- the one signal tied
    to this handler's own body, since nothing else in the Notes route pushes
    ``LibraryNoteFolderNameDialog``.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_notes_tree(host, pilot)
        assert screen._notes_state.tree_selected_placement_id == ""

        screen.query_one("#library-notes-folder-new", Button).press()
        for _ in range(6):
            await pilot.pause()

        assert type(host.screen).__name__ == "LibraryNoteFolderNameDialog"
        # The protected-folder branch is the only one that writes a notice;
        # with no selection it must stay untouched.
        assert screen._notes_state.notice == ""


@pytest.mark.asyncio
async def test_work_pane_editor_ready_arms_dirty_tracking_on_the_editor_route() -> None:
    """``EditorReady`` arms the editor once the retained children mount.

    ``LibraryNoteWorkPane.EditorReady`` is named in no test in the repository.
    This drives the real note route, disarms the flag the mount already set,
    and posts the message again -- so the assertion can only pass if this
    handler ran and took its non-guarded path.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_notes_tree(host, pilot)
        await _open_note_editor(screen, pilot)
        assert screen._notes_state.view == "editor"
        assert screen._notes_state.selected_note_id == "n-1"

        screen._notes_state.editor_armed = False
        screen.post_message(LibraryNoteWorkPane.EditorReady())
        for _ in range(4):
            await pilot.pause()

        assert screen._notes_state.editor_armed is True


@pytest.mark.asyncio
async def test_work_pane_editor_ready_is_ignored_off_the_editor_route() -> None:
    """The route guard: the same message off the editor route arms nothing.

    Same screen, same message, same starting value as the test above -- only
    ``_notes_state.view`` differs, which is precisely the guard this pins.
    Without both directions an inverted guard would still look green.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_notes_tree(host, pilot)
        await _open_note_editor(screen, pilot)

        screen._notes_state.editor_armed = False
        screen._notes_state.view = "list"
        screen.post_message(LibraryNoteWorkPane.EditorReady())
        for _ in range(4):
            await pilot.pause()

        assert screen._notes_state.editor_armed is False
