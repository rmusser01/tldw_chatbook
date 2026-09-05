"""Characterization pins for genuinely-unpressed Library Prompts handlers.

Wave-6 Task 1 (prompts series 1/3, state PR; recipe:
``backlog/docs/library-decomposition-recipe.md``; the collections/ingest
series' own ``test_library_collections_characterization.py``/
``test_library_ingest_characterization.py`` are the precedent this file
mirrors). These pins exist BEFORE the Prompts extraction moves any state,
so a later move that silently breaks one of these ``@on`` dispatch paths
goes red rather than green-but-vacuous.

Scope: the four ``@on``-bound Prompts handlers that touch moved
``LibraryPromptsState`` fields and had NO real-dispatch coverage anywhere
in ``Tests/`` before this file existed. The census walked all of
``Tests/`` (including ``Tests/Prompt_Management/``, ``Tests/Prompt_Studio/``,
``Tests/Internal_Prompts/`` and ``Tests/Prompts_DB/``, which turned out to
carry zero ``LibraryScreen`` consumers at all) for each handler's own
``@on`` selector, then read every hit rather than trusting the grep:

- ``handle_library_prompts_import_path_changed``
  (``@on(Input.Changed, "#library-prompts-import-path")``) -- the Import
  row's path field was only ever asserted MOUNTED, never typed into on a
  real screen.
- ``handle_library_prompts_import_path_submitted``
  (``@on(Input.Submitted, "#library-prompts-import-path")``) -- same
  field, never submitted.
- ``handle_library_prompts_import_run``
  (``@on(Button.Pressed, "#library-prompts-import-run")``) -- the Import
  row's own "Import" action was only ever queried for presence/parent, on
  a standalone canvas host (which cannot dispatch a SCREEN ``@on``
  handler at all).
- ``handle_library_prompt_discard``
  (``@on(Button.Pressed, "#library-prompt-discard")``) -- the editor's
  Discard action had many label/disabled assertions and zero presses.

Three near-misses this census deliberately does NOT pin, because reading
the hits (not the same-line grep) proved them genuinely covered -- the
same "a same-line-only grep undercounts coverage" trap the collections
series' own report recorded:

- ``handle_library_prompts_empty_new`` / ``handle_library_prompts_empty_
  clear_filter`` are both activated by a real focused-Button ``enter``
  press inside ``test_library_shell.py::test_library_paged_empty_recovery_
  is_painted_and_keyboard_reachable[prompts]``.
- ``handle_library_prompts_sort_choice`` is bound by CLASS
  (``.library-prompts-sort-choice``) but pressed by ID
  (``#library-prompts-sort-name``) in ``test_library_prompts_reader.py::
  test_items_browse_settles_while_prompt_editor_remains_open`` -- a
  selector-string grep cannot see that.

No live bugs were found writing these: all four are coverage gaps, not
behavior bugs. Each test below asserts the SCREEN-owned state the handler
mutates, so it stays meaningful after the state PR reroutes those names
through ``self._prompts_state``.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.test_library_prompts_canvas import (
    _build_test_app,
    _open_prompt_editor,
    _open_prompts_list,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

_BLANK_PATH_STATUS = "Please enter a file or folder path."


def _seed_prompt(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _message = db.add_prompt(
        name="Release assistant",
        author="Ada",
        details="Prepares release notes",
        system_prompt="Be exact.",
        user_prompt="Summarize {changes}.",
        keywords=["release", "summary"],
    )
    return db, prompt_id, service


async def _open_import_row(screen, pilot) -> Input:
    """Open the Prompts Import row and return its real path ``Input``."""
    screen.query_one("#library-prompts-import", Button).press()
    path_input = await _wait_for_selector(
        screen, pilot, "#library-prompts-import-path"
    )
    await pilot.pause()
    assert isinstance(path_input, Input)
    return path_input


@pytest.mark.asyncio
async def test_import_row_typing_reaches_the_screen_owned_path(tmp_path) -> None:
    """``@on(Input.Changed, "#library-prompts-import-path")`` really fires.

    Typing into the mounted Import path field must reach
    ``handle_library_prompts_import_path_changed`` and land in the
    screen's own import-path state -- the value the Import action later
    reads. Nothing in ``Tests/`` typed into this field on a real screen
    before this pin.
    """
    db, _prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompts_list(screen, pilot)
            path_input = await _open_import_row(screen, pilot)

            assert screen._library_prompts_import_open is True
            assert screen._library_prompts_import_path == ""

            typed = str(tmp_path / "exported-prompts.json")
            path_input.value = typed
            await _wait_for_condition(
                pilot,
                lambda: screen._library_prompts_import_path == typed,
                message="Import path typing never reached the screen state",
            )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_import_run_button_reaches_the_blank_path_gate(tmp_path) -> None:
    """``@on(Button.Pressed, "#library-prompts-import-run")`` really fires.

    The Import row's own "Import" action was never pressed anywhere in
    ``Tests/``. Pressing it with a whitespace-only path must reach
    ``handle_library_prompts_import_run`` -> ``_start_library_prompts_
    import``'s blank-path gate, which paints its one-line outcome in
    place and starts NO worker.
    """
    db, _prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompts_list(screen, pilot)
            path_input = await _open_import_row(screen, pilot)

            path_input.value = "   "
            await _wait_for_condition(
                pilot,
                lambda: screen._library_prompts_import_path == "   ",
                message="Import path typing never reached the screen state",
            )

            screen.query_one("#library-prompts-import-run", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_prompts_import_status
                == _BLANK_PATH_STATUS,
                message="Import Run never reached the blank-path gate",
            )
            status = screen.query_one("#library-prompts-import-status", Static)
            assert _BLANK_PATH_STATUS in str(status.renderable)
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_import_path_enter_reaches_the_blank_path_gate(tmp_path) -> None:
    """``@on(Input.Submitted, "#library-prompts-import-path")`` really fires.

    Enter in the Import row's path field is a second, independent entry
    point into the same import start, and had no coverage at all. Pressing
    Enter on the focused field with a whitespace-only path must reach
    ``handle_library_prompts_import_path_submitted`` and paint the same
    blank-path outcome the Import action does.
    """
    db, _prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompts_list(screen, pilot)
            path_input = await _open_import_row(screen, pilot)

            path_input.value = "   "
            await _wait_for_condition(
                pilot,
                lambda: screen._library_prompts_import_path == "   ",
                message="Import path typing never reached the screen state",
            )
            path_input.focus()
            await pilot.pause()
            assert screen.focused is path_input

            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: screen._library_prompts_import_status
                == _BLANK_PATH_STATUS,
                message="Import path Enter never reached the blank-path gate",
            )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_editor_discard_drops_the_dirty_draft_and_returns_to_the_list(
    tmp_path,
) -> None:
    """``@on(Button.Pressed, "#library-prompt-discard")`` really fires.

    Discard had label/disabled assertions across several files and zero
    presses. Pressing it on a dirty editor must reach
    ``handle_library_prompt_discard``: the editor state resets to the
    list view, the dirty flag clears, and the unsaved edit never reaches
    the database.
    """
    db, prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            await _wait_for_condition(
                pilot,
                lambda: screen._library_prompt_editor_armed,
                message="Prompt editor did not arm",
            )
            assert screen._library_prompts_view == "editor"

            screen.query_one("#library-prompt-user", TextArea).load_text(
                "Discarded working copy."
            )
            await _wait_for_condition(
                pilot,
                lambda: screen._library_prompt_dirty,
                message="Prompt draft did not become dirty",
            )

            discard = screen.query_one("#library-prompt-discard", Button)
            assert discard.disabled is False
            discard.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_prompts_view == "list"
                    and screen._library_prompt_dirty is False
                ),
                message="Discard never left the dirty Prompt editor",
            )

            assert screen._library_prompt_detail is None
            assert screen._library_prompt_block_state is None
            stored = db.get_prompt_by_id(prompt_id)
            assert stored is not None
            assert stored["user_prompt"] == "Summarize {changes}."
    finally:
        db.close_connection()
