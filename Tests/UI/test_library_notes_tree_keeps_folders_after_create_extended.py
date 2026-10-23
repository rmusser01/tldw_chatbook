"""TASK-34000.14, the slower arms: the full-app palette walk, Move note, 80x24.

Not in ``scripts/ui_pr_gate_census.txt``; the PR-gated core is
``test_library_notes_tree_keeps_folders_after_create.py``.
"""

from __future__ import annotations

import dataclasses

import pytest
from textual.widgets import Button

from Tests.UI.library_quit_guard_support import (
    _armed_editor,
    _library,
    _new_blank_note,
    _until,
)
from Tests.UI.test_library_notes_files_sync_journey import (
    _isolate_actor_pack_recovery,  # noqa: F401 - autouse fixture, re-exported
)
from Tests.UI.test_library_notes_tree_keeps_folders_after_create import (
    FOLDERS,
    ROOT_FOLDERS,
    SIZE,
    TITLE,
    _assert_folders_above_unfiled,
    _folder_rows,
    _mounted,
    _open_folder_picker,
    _painted_list,
    _profiled_app,
    _shell,
    _title_then_escape,
    _tree_settled,
)
from Tests.UI.library_quit_guard_support import _until as _wait_until
from Tests.UI.test_library_shell import _wait_for_selector
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_NOTES_CREATE, TAB_LIBRARY
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: AC#5's second size.
COMPACT_SIZE = (80, 24)


async def test_the_palette_message_from_the_default_screen_lands_on_a_whole_tree(
    tmp_path,
) -> None:
    """The real palette route: ``NavigateToScreen`` posted on the full app.

    ``execute_quick_action("new_note")`` posts exactly this message; here it
    is posted from the app's default screen, so the Library mounts into
    Create through ``handle_screen_navigation`` itself.
    """
    app, _profile = _profiled_app(tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await _until(
            pilot,
            lambda: (
                app.screen is not None
                and app.screen.is_mounted
                and bool(getattr(app, "current_tab", ""))
            ),
            "the default screen to mount",
        )
        await pilot.pause(0.2)
        app.post_message(
            NavigateToScreen(TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_NOTES_CREATE: True})
        )
        try:
            screen = await _library(app, pilot)
        except AssertionError as exc:  # pragma: no cover - diagnostic only
            raise AssertionError(
                f"{exc}; screen stack: {[type(s).__name__ for s in app.screen_stack]}"
            ) from exc
        blank = await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        blank.press()
        await _armed_editor(screen, pilot)
        await _title_then_escape(screen, pilot, TITLE)
        await _tree_settled(screen, pilot)

        _assert_folders_above_unfiled(_painted_list(app, screen), note_title=TITLE)


async def test_move_note_lists_the_folders_straight_after_a_create(tmp_path) -> None:
    """AC#2's second half: Move note reads the same loaded branches."""
    host, _profile = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _shell(host, pilot)
        await _new_blank_note(screen, pilot)
        await _title_then_escape(screen, pilot, TITLE)
        await _tree_settled(screen, pilot)

        prompts = await _open_folder_picker(
            host, screen, pilot, "#library-notes-placement-move"
        )
        assert prompts == list(FOLDERS), prompts


async def test_rail_new_note_then_escape_shows_the_folders_at_80x24(tmp_path) -> None:
    """AC#1 at the compact size: the folder rows are there to go to."""
    host, _profile = _mounted(tmp_path)
    async with host.run_test(size=COMPACT_SIZE) as pilot:
        screen = await _shell(host, pilot)
        screen.query_one("#library-row-create-note", Button).press()
        blank = await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        blank.press()
        await _armed_editor(screen, pilot)
        await _title_then_escape(screen, pilot, TITLE)
        await _tree_settled(screen, pilot)
        await _until(
            pilot,
            lambda: len(_folder_rows(screen)) == len(FOLDERS),
            "the folder rows at 80x24",
        )
        await pilot.pause(0.1)

        labels = [str(row.label) for row in _folder_rows(screen)]
        assert [
            name for name in FOLDERS if any(name in label for label in labels)
        ] == list(FOLDERS), labels
        assert "▸ Journal" in _painted_list(host, screen), _painted_list(host, screen)


async def test_the_list_return_re_requests_a_root_slice_a_fence_left_behind(
    tmp_path,
) -> None:
    """The belt: an editor-to-list return owes a root slice a fence dropped.

    ``_fence_library_notes_tree_mutation`` leaves an in-flight slice as
    ``uninitialized`` and not loading, and the worker whose result it fenced
    is ignored when it lands -- the only shape in which a create arrival's
    own root request can fail to paint. The editor's return re-requests it.
    """
    host, _profile = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _shell(host, pilot)
        await _new_blank_note(screen, pilot)
        branches = screen._notes_state.tree_branches
        await _wait_until(
            pilot,
            lambda: (
                ROOT_FOLDERS in branches and branches[ROOT_FOLDERS].freshness == "fresh"
            ),
            "the create's own root folders load",
        )
        # The exact shape the fence leaves behind, with the loaded items
        # gone the way a superseded worker's result is dropped.
        branches[ROOT_FOLDERS] = dataclasses.replace(
            branches[ROOT_FOLDERS],
            items=(),
            item_ids=(),
            freshness="uninitialized",
            loading=False,
            generation=branches[ROOT_FOLDERS].generation + 1,
        )
        screen._notes_state.tree_request_generations[ROOT_FOLDERS] = branches[
            ROOT_FOLDERS
        ].generation

        await _title_then_escape(screen, pilot, TITLE)
        await _wait_until(
            pilot,
            lambda: len(_folder_rows(screen)) == len(FOLDERS),
            "the editor return to re-request the root folders slice",
            timeout=10.0,
        )
        await pilot.pause(0.1)
        _assert_folders_above_unfiled(_painted_list(host, screen), note_title=TITLE)
