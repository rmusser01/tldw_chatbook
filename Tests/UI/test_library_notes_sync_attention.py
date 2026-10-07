"""A held sync folder says so on the tree, the list and the editor (TASK-34000.2).

Review finding N-02: after an end-of-note edit wedged a synced folder, the
tree row still read "⇄ Sync managed", the Notes list "Library notes · Ready",
and the editor "Saved 21:40 · Next: Keep editing" over "In a synced folder" --
while nothing reached the file. Recovery then failed with "RuntimeError" and
sent the user to a Check that refused the same open entry.

One real app boot over the production runtime, executor and POSIX filesystem,
a real ChaChaNotes database and a real ``.md`` file in a temp vault. The root
is wedged exactly the way the defect left it (the pre-fix postcondition
re-installed for the one save that wedges it), then healed through the real
Manage sync folders ▸ Recovery button. The durable rows and the bytes on disk
back every claim the painted text makes.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.Notes.notes_sync_tail_edit_support import (
    TAIL_EDIT as _TAIL_EDIT,
    Vault as _Vault,
    build_owner as _owner,
    wedge_root as _wedge,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _armed_editor,
    _library,
    _settings_without_splash,
    _until,
)
from Tests.UI.test_library_shell import _seed_conversations, _wait_for_selector
from Tests.app_module_patches import patch_app_global

#: The user-visible copy, spelled out so a RED run fails on behaviour.
NOTES_TREE_SYNC_ATTENTION_STATUS = "⚠ Needs attention"
SYNC_ATTENTION_LIST_STATUS = "⚠ A sync folder needs attention"
NOTE_LOCATION_ATTENTION = "⚠ Sync needs attention"

#: ``bootstrap_profile`` for the same reason as ``test_library_quit_guard.py``.
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


def _text(screen, selector: str) -> str:
    return " ".join(str(screen.query_one(selector, Static).render()).split())


def _folder_row(screen) -> Button:
    return next(
        button
        for button in screen.query(".library-notes-folder-row").results(Button)
        if "VSync" in str(button.label)
    )


async def test_a_held_sync_folder_reads_needs_attention_until_recovery_heals_it(
    tmp_path, monkeypatch
):
    vault = _Vault(tmp_path)
    owner = _owner(vault)
    await owner.start()
    try:
        await _wedge(vault, owner, monkeypatch)
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

                # Tree and list: the held folder, never "Sync managed"/"Ready".
                await _until(
                    pilot,
                    lambda: (
                        NOTES_TREE_SYNC_ATTENTION_STATUS
                        in str(_folder_row(screen).label)
                    ),
                    "the held folder's tree row to say it needs attention",
                )
                assert "Sync managed" not in str(_folder_row(screen).label)
                assert _folder_row(screen).has_class(
                    "library-notes-tree-needs-attention"
                )
                list_line = _text(screen, "#library-notes-authority")
                assert SYNC_ATTENTION_LIST_STATUS in list_line, list_line
                assert "Ready" not in list_line, list_line

                # Editor: the location row and the status both say it.
                _folder_row(screen).press()
                await _until(
                    pilot,
                    lambda: any(
                        "quotes" in str(button.label)
                        for button in screen.query(
                            ".library-notes-tree-note-row"
                        ).results(Button)
                    ),
                    "the held folder to list its note",
                )
                next(
                    button
                    for button in screen.query(".library-notes-tree-note-row").results(
                        Button
                    )
                    if "quotes" in str(button.label)
                ).press()
                await _armed_editor(screen, pilot, "note-1")
                await _until(
                    pilot,
                    lambda: (
                        NOTE_LOCATION_ATTENTION
                        in _text(screen, "#library-note-location")
                    ),
                    "the editor's location row to say the folder needs attention",
                )
                location = _text(screen, "#library-note-location")
                assert location.startswith("In a synced folder"), location
                work_line = _text(screen, "#library-note-work-authority")
                assert NOTE_LOCATION_ATTENTION in work_line, work_line
                assert "Keep editing; changes save automatically" not in work_line

                # Recovery heals it from the real button -- no RuntimeError.
                screen.query_one("#library-notes-manage-sync-folders", Button).press()
                recover = await _wait_for_selector(
                    screen, pilot, "#notes-sync-root-recover-0"
                )
                assert recover.has_class("console-action-primary")
                recover.press()
                controller = screen._library_notes_sync_controller
                await _until(
                    pilot,
                    lambda: controller.snapshot.status_line.startswith(
                        "Recovery finished"
                    ),
                    "Recovery to finish",
                )
                assert "RuntimeError" not in controller.snapshot.status_line
                assert vault.incomplete() == []
                assert vault.note()["content"] == _TAIL_EDIT
                assert vault.file.read_bytes() == (_TAIL_EDIT + "\n").encode("utf-8")

                back = await _wait_for_selector(screen, pilot, "#notes-sync-roots-back")
                assert "RuntimeError" not in _text(screen, "#notes-sync-roots-status")
                back.press()
                await _until(
                    pilot,
                    lambda: (
                        bool(screen.query(".library-notes-folder-row"))
                        and "Sync managed" in str(_folder_row(screen).label)
                    ),
                    "the healed folder's tree row to read Sync managed again",
                )
                assert NOTES_TREE_SYNC_ATTENTION_STATUS not in str(
                    _folder_row(screen).label
                )
    finally:
        await owner.shutdown()
        vault.close()
