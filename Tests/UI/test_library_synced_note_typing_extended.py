"""Typing in, and quitting from, a note in a synced folder: the real app.

Final review I1 and I4, outside the PR gate (each test is a real app boot plus
real typing time). The gated pins are deterministic and need no app:
``Tests/Notes/test_notes_sync_resave_window.py``.

I1. TASK-34000.1's max wait makes an autosave fire while the user is still
typing, and the note session re-saves at once for keys that landed during it.
The second save used to commit inside the sync pass the first one hinted, which
fences the folder: steady typing held it for Recovery two to six times in about
33 s (final review probe P5). A save now waits, bounded, for that pass.

I4. Ctrl+Q flushed the note and exited without waiting for the pass its flush
hinted, so the file missed the last edit until the next launch.

Everything is real: ``TldwCli``, ``LibraryScreen``, the note session and its
port, the production sync runtime, executor and POSIX filesystem, a real
ChaChaNotes database and a real ``.md`` in a temp vault. Assertions read the
database row, the bytes on disk and the sync store.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button, TextArea

from Tests.app_module_patches import patch_app_global
from Tests.Notes.notes_sync_tail_edit_support import VAULT_TEXT, Vault, build_owner
from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _armed_editor,
    _ctrl_q,
    _library,
    _no_autosave,
    _scaled_autosave,
    _settings_without_splash,
    _type_at_end,
    _until,
)
from Tests.UI.test_library_shell import _seed_conversations, _wait_for_selector
from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutor

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: Statuses under which a folder is not syncing until the user acts.
_HELD = frozenset({"failed", "needs_attention", "partial"})
#: Steady typing: one key every ``_KEY_GAP`` s, well inside the scaled 1 s
#: debounce, for several scaled 2 s max waits.
_KEYS, _KEY_GAP = 70, 0.12


def _file_bytes(note_text: str) -> bytes:
    """The vault file for ``note_text`` under its final-newline profile."""

    return (note_text if note_text.endswith("\n") else note_text + "\n").encode("utf-8")


def _synced_app(vault: Vault, owner):
    """The production app over the vault's authorities and its sync runtime."""

    app = _build_test_app(configured_default="library")
    _seed_conversations(app, [])
    app.chachanotes_db = vault.database
    app.notes_scope_service = vault.scope_service
    app.notes_service = vault.interop
    app.notes_sync_runtime_owner = owner
    return app


async def _open_synced_note(screen, pilot) -> TextArea:
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, ".library-notes-folder-row")
    next(
        button
        for button in screen.query(".library-notes-folder-row").results(Button)
        if "VSync" in str(button.label)
    ).press()

    def _note_row() -> Button | None:
        return next(
            (
                button
                for button in screen.query(".library-notes-tree-note-row").results(
                    Button
                )
                if "quotes" in str(button.label)
            ),
            None,
        )

    await _until(pilot, lambda: _note_row() is not None, "the folder to list its note")
    _note_row().press()
    await _armed_editor(screen, pilot, "note-1")
    return screen.query_one("#library-note-body", TextArea)


async def test_steady_typing_in_a_synced_note_never_holds_its_folder(
    tmp_path, monkeypatch
):
    _scaled_autosave(monkeypatch)  # 1 s debounce / 2 s max wait
    vault = Vault(tmp_path)
    owner = build_owner(vault)
    await owner.start()
    held: list[tuple[int, str, str]] = []
    try:
        app = _synced_app(vault, owner)
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_synced_note(screen, pilot)
                body.focus()
                body.move_cursor(body.document.end)
                await pilot.pause()
                start_version = int(vault.note()["version"])
                for index in range(_KEYS):
                    await pilot.press("abcdefghij"[index % 10])
                    await pilot.pause(_KEY_GAP)
                    root = owner.snapshot().roots[0]
                    if root.status in _HELD:
                        held.append((index, root.status, root.next_action))
                typed = body.text
                await _until(
                    pilot,
                    lambda: (
                        vault.note()["content"] == typed
                        and not screen._library_note_session.snapshot.dirty
                        and not screen._library_note_session.snapshot.saving
                    ),
                    "the last keys to be autosaved",
                )
                await asyncio.wait_for(owner.settle(), 30)
                saves = int(vault.note()["version"]) - start_version
                final = owner.snapshot().roots[0]
                incomplete = vault.incomplete()
                on_disk = vault.file.read_bytes()
    finally:
        await owner.shutdown()
        vault.close()

    assert typed.startswith(VAULT_TEXT) and len(typed) == len(VAULT_TEXT) + _KEYS
    assert saves >= 2, f"the run must autosave mid-typing; it saved {saves} time(s)"
    assert held == [], f"typing held the synced folder: {held[:3]}"
    assert incomplete == [], f"typing left an operation for Recovery: {incomplete}"
    assert (final.status, final.next_action) == ("up_to_date", "sync_now")
    assert on_disk == _file_bytes(typed), "the file is not the note"


async def test_ctrl_q_in_a_synced_note_leaves_its_file_in_step(tmp_path, monkeypatch):
    """Save, quit, relaunch. The file is read at the moment the approved quit
    would exit; the folder is slow, as a large vault is."""

    _no_autosave(monkeypatch)  # only the quit may persist anything
    real_execute = NotesSyncExecutor.execute

    async def a_slow_folder(executor, request):
        await asyncio.sleep(0.5)
        return await real_execute(executor, request)

    monkeypatch.setattr(NotesSyncExecutor, "execute", a_slow_folder)
    vault = Vault(tmp_path)
    owner = build_owner(vault)
    await owner.start()
    events: list = []
    tail = " quit tail"
    try:
        app = _synced_app(vault, owner)

        async def _record_quit() -> None:
            events.append((vault.note()["content"], vault.file.read_bytes()))

        monkeypatch.setattr(app, "_run_approved_quit_cleanup", _record_quit)
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_synced_note(screen, pilot)
                await _type_at_end(pilot, body, tail)
                assert screen._library_note_session.snapshot.dirty

                prompt = await _ctrl_q(pilot, app, events)

                assert prompt is None, "a flushable edit must quit without asking"
    finally:
        await owner.shutdown()

    try:
        [(note_at_exit, file_at_exit)] = events
        assert note_at_exit == VAULT_TEXT + tail, "the quit flush lost the save"
        assert file_at_exit == _file_bytes(VAULT_TEXT + tail), (
            "the app would have exited before the sync pass its flush hinted ran"
        )
        relaunched = build_owner(vault)
        await relaunched.start()
        try:
            await relaunched.settle()
            root = relaunched.snapshot().roots[0]
            assert vault.incomplete() == []
            assert (root.status, root.next_action) == ("up_to_date", "sync_now")
            assert vault.file.read_bytes() == _file_bytes(vault.note()["content"])
        finally:
            await relaunched.shutdown()
    finally:
        vault.close()
