"""A watcher-driven hold reaches the tree, list and editor with no user action.

TASK-34000.2 fix round 1 (review Important #2). Before the runtime status
listener, a hold produced by a watcher pass while the user sat idle -- a
disk edit colliding with an unsynced note edit -- reached the runtime's
status but no Notes surface until the next save, sync action or Library
visit; the list read "Ready", the tree "⇄ Sync managed", the open note
"Saved". This boots the real app over the production runtime, executor and
POSIX filesystem (real ChaChaNotes database, real ``.md`` in a temp vault),
opens the synced note, then changes BOTH sides underneath the idle UI and
lets the folder's own pass run -- exactly what the watcher does -- and
asserts the three surfaces turn within a bounded wait, with the stores and
the bytes on disk backing the painted text.

Not in the UI PR gate: the lane is near its cap and
``test_library_notes_sync_attention.py`` already holds its one gated boot.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.Notes.notes_sync_tail_edit_support import (
    VAULT_TEXT,
    Vault,
    build_owner,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _armed_editor,
    _library,
    _settings_without_splash,
    _until,
)
from Tests.UI.test_library_notes_sync_delete_restore import LIST, _live_button, _press
from Tests.UI.test_library_shell import _seed_conversations, _wait_for_selector
from Tests.app_module_patches import patch_app_global

NOTES_TREE_SYNC_ATTENTION_STATUS = "⚠ Needs attention"
SYNC_ATTENTION_LIST_STATUS = "⚠ A sync folder needs attention"
NOTE_LOCATION_ATTENTION = "⚠ Sync needs attention"

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


def _text(screen, selector: str) -> str:
    return " ".join(str(screen.query_one(selector, Static).render()).split())


def _is_vsync(row) -> bool:
    return "VSync" in str(row.label)


def _folder_row(screen) -> Button | None:
    """The live VSync row, or None while the tree rebuilds (TASK-38959).

    A bare ``next()`` raised ``StopIteration`` inside the ``_until``
    predicates whenever the row was briefly gone; this is the sibling
    file's lookup since 67406398ad.
    """
    return _live_button(screen, LIST, ".library-notes-folder-row", _is_vsync)


def _folder_row_reads(screen, status: str) -> bool:
    row = _folder_row(screen)
    return row is not None and status in str(row.label)


async def test_a_watcher_driven_hold_turns_the_three_surfaces_without_a_keypress(
    tmp_path,
):
    vault = Vault(tmp_path)
    owner = build_owner(vault)
    await owner.start()
    try:
        app = _build_test_app(configured_default="library")
        _seed_conversations(app, [])
        app.chachanotes_db = vault.database
        app.notes_scope_service = vault.scope_service
        app.notes_service = vault.interop
        app.notes_sync_runtime_owner = owner
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                screen.query_one("#library-row-browse-notes", Button).press()
                await _wait_for_selector(screen, pilot, ".library-notes-folder-row")
                await _until(
                    pilot,
                    lambda: _folder_row_reads(screen, "Sync managed"),
                    "the healthy folder's tree row",
                )
                assert "Ready" in _text(screen, "#library-notes-authority")

                await _press(
                    screen,
                    pilot,
                    LIST,
                    ".library-notes-folder-row",
                    "the healthy folder's tree row",
                    _is_vsync,
                )
                await _press(
                    screen,
                    pilot,
                    LIST,
                    ".library-notes-tree-note-row",
                    "the folder to list its note",
                    lambda row: "quotes" in str(row.label),
                )
                await _armed_editor(screen, pilot, "note-1")
                location = _text(screen, "#library-note-location")
                assert location.startswith("In a synced folder"), location
                assert NOTE_LOCATION_ATTENTION not in location

                # Both sides change underneath the idle UI -- a note edit that
                # never went through the editor (no save, no hint) and a
                # different disk edit -- then the folder's own pass runs, as
                # the watcher would run it. Not one key is pressed from here.
                vault.edit_note(VAULT_TEXT + "app side")
                disk_side = VAULT_TEXT + "disk side\n"
                vault.file.write_bytes(disk_side.encode("utf-8"))
                assert owner.schedule_hint("root-1") is not None

                await _until(
                    pilot,
                    lambda: _folder_row_reads(
                        screen, NOTES_TREE_SYNC_ATTENTION_STATUS
                    ),
                    "the tree row to say the folder needs attention",
                )
                await _until(
                    pilot,
                    lambda: (
                        SYNC_ATTENTION_LIST_STATUS
                        in _text(screen, "#library-notes-authority")
                    ),
                    "the list to say a sync folder needs attention",
                )
                await _until(
                    pilot,
                    lambda: (
                        NOTE_LOCATION_ATTENTION
                        in _text(screen, "#library-note-location")
                    ),
                    "the editor's location row to say the folder needs attention",
                )
                assert "Ready" not in _text(screen, "#library-notes-authority")

                # The hold is real and nothing was overwritten on either side.
                root = owner.snapshot().roots[0]
                assert (root.status, root.next_action) == (
                    "needs_attention",
                    "review_changes",
                )
                assert vault.incomplete() == []
                assert vault.note()["content"] == VAULT_TEXT + "app side"
                assert vault.file.read_bytes() == disk_side.encode("utf-8")
    finally:
        await owner.shutdown()
        vault.close()
