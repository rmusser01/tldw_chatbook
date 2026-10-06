"""Deleting a synced note holds its folder; Undo brings it back to up to date.

TASK-32633 slice (TASK-34000 wave 1, Task 4; review finding N-03). Before
this, Delete and Undo in the Library never told lasting sync anything: the
root row went on reading "✓ Up to date" over a file that was still on disk
for a note that no longer existed. This boots the real app over the
production runtime, executor and POSIX filesystem (real ChaChaNotes database,
real ``.md`` in a temp vault), deletes the synced note through Info ▸ Delete ▸
confirm, reads Manage sync folders, then presses the receipt's Undo and reads
it again. The file's bytes and the stores back every painted word; the
healthy row names the minute it was confirmed.

One boot, both phases. Candidate for the UI PR gate at well under its 60 s
budget (measured locally; see the task's Implementation Notes).
"""

from __future__ import annotations

import asyncio
import re

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from Tests.app_module_patches import patch_app_global
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
from Tests.UI.test_library_shell import _seed_conversations, _wait_for_selector
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    DELETE_CONFIRM_COPY_SYNCED,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

ROOT_STATUS = ".library-notes-sync-root-status"
UP_TO_DATE_AS_OF = re.compile(r"^✓ Up to date as of \d\d:\d\d · Next: Check changes$")


def _text(screen, selector: str) -> str:
    return " ".join(str(screen.query_one(selector, Static).render()).split())


def _text_or_empty(screen, selector: str) -> str:
    """``_text`` for a predicate polled across recomposes: a Static that is
    unmounted for one frame reads as "" instead of raising ``NoMatches``."""

    try:
        return _text(screen, selector)
    except NoMatches:
        return ""


def _note_row(screen, title: str) -> Button:
    return next(
        button
        for button in screen.query(".library-notes-tree-note-row").results(Button)
        if title in str(button.label)
    )


async def _open_manage_sync_folders(screen, pilot) -> str:
    # Fix round 1 (review Important 1): Undo's own recompose can leave the
    # toolbar briefly unmounted; wait for the button and press the node the
    # settle pause re-queried, never a detached one.
    button = await _wait_for_selector(
        screen, pilot, "#library-notes-manage-sync-folders"
    )
    button.press()
    await _wait_for_selector(screen, pilot, ROOT_STATUS)
    await pilot.pause(0.1)
    return _text(screen, ROOT_STATUS)


async def _back_to_notes_list(screen, pilot) -> None:
    screen.query_one("#notes-sync-roots-back", Button).press()
    await _until(
        pilot,
        lambda: screen._notes_state.view == "list",
        "Manage sync folders to return to the list",
    )


async def _open_synced_folder(screen, pilot) -> None:
    await _until(
        pilot,
        lambda: any(
            getattr(row, "folder_id", "") == "folder-1"
            for row in screen.query(".library-notes-folder-row").results(Button)
        ),
        "the synced folder to load",
    )
    next(
        button
        for button in screen.query(".library-notes-folder-row").results(Button)
        if getattr(button, "folder_id", "") == "folder-1"
    ).press()


@pytest.mark.parametrize("delayed_root_folders", [False, True])
async def test_delete_holds_the_folder_and_undo_returns_it_to_up_to_date(
    tmp_path, monkeypatch, delayed_root_folders
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
                release_folders = asyncio.Event()
                readiness_reached = asyncio.Event()
                wait_for_selector = _wait_for_selector
                if delayed_root_folders:
                    # A real Unfiled row may paint before the independent
                    # folder slice. Gate that response, not the UI or data.
                    assert vault.database.add_note("Unfiled control", "not synced")
                    page_folders = vault.scope_service.page_note_folder_children

                    async def gated_folders(**kwargs):
                        if kwargs["parent_id"] is None:
                            await release_folders.wait()
                        return await page_folders(**kwargs)

                    monkeypatch.setattr(
                        vault.scope_service, "page_note_folder_children", gated_folders
                    )
                    until = _until

                    async def observed_until(*args, **kwargs):
                        readiness_reached.set()
                        return await until(*args, **kwargs)

                    async def observed_selector(*args, **kwargs):
                        row = await wait_for_selector(*args, **kwargs)
                        if args[2] == ".library-notes-folder-row":
                            readiness_reached.set()
                        return row

                    monkeypatch.setattr(__name__ + "._until", observed_until)
                    monkeypatch.setattr(
                        __name__ + "._wait_for_selector", observed_selector
                    )
                screen.query_one("#library-row-browse-notes", Button).press()
                if delayed_root_folders:
                    opening = asyncio.create_task(_open_synced_folder(screen, pilot))
                    try:
                        await wait_for_selector(
                            screen, pilot, ".library-notes-folder-row"
                        )
                        await until(
                            pilot,
                            lambda: readiness_reached.is_set() or opening.done(),
                            "the folder readiness handoff",
                        )
                        assert not any(
                            getattr(row, "folder_id", "") == "folder-1"
                            for row in screen.query(".library-notes-folder-row")
                        )
                        assert not opening.done(), "Unfiled is not VSync readiness"
                        release_folders.set()
                        await opening
                    finally:
                        release_folders.set()
                        await asyncio.gather(opening, return_exceptions=True)
                else:
                    await _open_synced_folder(screen, pilot)
                await _until(
                    pilot,
                    lambda: any(
                        "quotes" in str(b.label)
                        for b in screen.query(".library-notes-tree-note-row").results(
                            Button
                        )
                    ),
                    "the folder to list its note",
                )
                _note_row(screen, "quotes").press()
                await _armed_editor(screen, pilot, "note-1")

                # Phase 1: Info ▸ Delete ▸ confirm, through the real buttons.
                screen.query_one("#library-note-context", Button).press()
                await pilot.pause()
                screen.query_one("#library-note-context-delete", Button).press()
                await _wait_for_selector(screen, pilot, "#library-note-delete-confirm")
                # The prompt for a synced note says what happens to its file.
                assert (
                    _text(screen, "#library-note-delete-confirm-copy")
                    == DELETE_CONFIRM_COPY_SYNCED
                )
                screen.query_one("#library-note-delete-confirm", Button).press()
                await _wait_for_selector(screen, pilot, "#library-notes-delete-undo")
                await _until(
                    pilot,
                    lambda: owner.snapshot().roots[0].status == "needs_attention",
                    "the delete to reach lasting sync and hold the folder",
                )
                await owner.settle()
                tombstone = vault.database.get_note_version_states(["note-1"])["note-1"]
                assert tombstone["deleted"] is True
                # The file is untouched: a note-side deletion is a review, not
                # a file delete (no automatic winner).
                assert vault.file.read_bytes() == VAULT_TEXT.encode("utf-8")
                assert vault.incomplete() == []
                held = await _open_manage_sync_folders(screen, pilot)
                assert held == "⚠ Needs attention · Next: Review changes", held
                await _back_to_notes_list(screen, pilot)

                # Phase 2: Undo from the receipt.
                await _wait_for_selector(screen, pilot, "#library-notes-delete-undo")
                screen.query_one("#library-notes-delete-undo", Button).press()
                await _until(
                    pilot,
                    lambda: owner.snapshot().roots[0].status == "up_to_date",
                    "the restore to reach lasting sync and release the hold",
                )
                await owner.settle()
                assert vault.note()["content"] == VAULT_TEXT
                assert vault.file.read_bytes() == VAULT_TEXT.encode("utf-8")
                assert vault.incomplete() == []
                healthy = await _open_manage_sync_folders(screen, pilot)
                assert UP_TO_DATE_AS_OF.match(healthy), healthy
                assert "✓ Up to date ·" not in healthy

                # Phase 3 (review Minor 7): the rows follow the runtime while
                # the user sits on Manage sync folders. Both sides change
                # underneath the idle UI and the folder's own pass runs, as the
                # watcher would run it; the row turns without a keypress.
                vault.edit_note(VAULT_TEXT + "app side")
                vault.file.write_bytes((VAULT_TEXT + "disk side\n").encode("utf-8"))
                assert owner.schedule_hint("root-1") is not None
                await _until(
                    pilot,
                    lambda: (
                        _text_or_empty(screen, ROOT_STATUS)
                        == "⚠ Needs attention · Next: Review changes"
                    ),
                    "the Manage row to follow the hold without a keypress",
                )
                assert vault.note()["content"] == VAULT_TEXT + "app side"
                assert vault.file.read_bytes() == (VAULT_TEXT + "disk side\n").encode(
                    "utf-8"
                )
    finally:
        await owner.shutdown()
        vault.close()
