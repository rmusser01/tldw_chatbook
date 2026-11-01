"""Shared real-stack fixtures for the TASK-34000.2 (N-02) sync tests.

One real bound note/file pair in disposable authorities (a real
``CharactersRAGDB``, a real ``.md`` in a temp vault with a final-newline
profile), the production runtime owner over it, and the helper that wedges
the root exactly the way the defect left it. Imported by
``Tests/Notes/test_notes_sync_tail_edit.py`` and the Pilot tests under
``Tests/UI``; nothing here is a test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncRootRecord,
    NotesSyncStoreSetting,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutor
from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_runtime_owner

#: A vault file the way editors write it: every line ends with a newline.
VAULT_TEXT = (
    "> The best time to plant a tree was 20 years ago.\n\n> Writing is thinking.\n"
)
#: Ctrl+End, then a word without Enter: the note no longer ends in "\n".
TAIL_EDIT = VAULT_TEXT + "second edit line"


class Vault:
    """One real bound note/file pair in disposable authorities."""

    def __init__(
        self,
        tmp_path: Path,
        text: str = VAULT_TEXT,
        *,
        file_bytes: bytes | None = None,
    ) -> None:
        """Seed one bound pair.

        Args:
            tmp_path: Where the vault and both databases go.
            text: The note's content -- the LOGICAL text, LF newlines.
            file_bytes: The file's exact bytes; defaults to ``text`` in UTF-8.
                TASK-34000.48 seeds a CRLF file this way: the note keeps the
                logical text, the file keeps its own line endings, and the
                binding baseline is the one digest both sides share.
        """

        self.notes_path = tmp_path / "notes.sqlite3"
        self.state_path = tmp_path / "sync.sqlite3"
        self.root = tmp_path / "vault"
        self.root.mkdir()
        self.file = self.root / "quotes.md"
        self.file.write_bytes(text.encode("utf-8") if file_bytes is None else file_bytes)
        database = CharactersRAGDB(self.notes_path, client_id="task-34000-2-seed")
        folders = LocalNoteFolderRepository(database)
        assert database.add_note("quotes", text, "note-1") == "note-1"
        folders.create_folder(name="VSync", parent_id=None, folder_id="folder-1")
        folders.reconcile_managed(owner_id="root-1", desired=(("folder-1", "note-1"),))
        with PosixNotesSyncFilesystem(self.root) as filesystem:
            baseline = filesystem.observe("quotes.md")
        assert baseline.observation.serialization.final_newline is text.endswith("\n")
        note = database.get_note_by_id("note-1")
        assert note is not None
        store = NotesDeviceStateStore(self.state_path)
        store.initialize()
        store.create_root(
            NotesSyncRootRecord(
                root_id="root-1",
                note_scope_id="local_note",
                logical_folder_id="folder-1",
                canonical_path=str(self.root.resolve()),
                direction=NotesSyncDirection.BIDIRECTIONAL,
                state=NotesSyncRootState.ACTIVE,
            )
        )
        store.set_setting(
            NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1")
        )
        store.create_binding(
            NotesSyncBindingRecord(
                binding_id="binding-1",
                root_id="root-1",
                note_scope_id="local_note",
                note_id="note-1",
                normalized_relative_path="quotes.md",
                stable_identity_digest=NotesSyncExecutor.stable_identity_digest(
                    baseline
                ),
                state=NotesSyncBindingState.ACTIVE,
                serialization=baseline.observation.serialization,
                content_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                note_version=int(note["version"]),
            )
        )
        store.close()
        database.close_connection()
        self.database = CharactersRAGDB(self.notes_path, client_id="task-34000-2")
        self.interop = NotesInteropService(
            base_db_directory=self.notes_path.parent,
            api_client_id="task-34000-2",
            global_db_to_use=self.database,
        )
        self.scope_service = NotesScopeService(
            local_notes_service=self.interop,
            server_service=None,
            folder_repository=LocalNoteFolderRepository(self.database),
        )

    def note(self) -> dict:
        note = self.database.get_note_by_id("note-1")
        assert note is not None
        return note

    def edit_note(self, content: str) -> None:
        note = self.note()
        assert self.database.update_note(
            "note-1",
            {"title": note["title"], "content": content},
            int(note["version"]),
        )

    def incomplete(self) -> list[tuple[str, str, str | None]]:
        store = NotesDeviceStateStore(self.state_path)
        try:
            return [
                (item.kind, item.state.value, item.reason_code)
                for item in store.list_incomplete_operations()
            ]
        finally:
            store.close()

    def close(self) -> None:
        self.interop.close_all_user_connections()
        self.database.close_connection()


def build_owner(vault: Vault):
    return build_notes_sync_runtime_owner(
        notes_scope_service=vault.scope_service,
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=vault.state_path,
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )


def old_postcondition(file, note, reviewed, recorded=None) -> bool:
    """The comparison origin/dev 2d34cbf80d shipped (notes_sync_executor:5384).

    Re-installed only to put a root into the exact durable state the defect
    leaves -- ``update_file | needs_attention | postcondition_failed`` after
    the file write -- so the heal path is proved on the real rows.
    ``recorded`` is the binding profile TASK-34000.48's callers pass; the old
    comparison never looked at it.
    """

    return (
        file.observation.relative_path == reviewed.observation.relative_path
        and file.text == note.content
        and file.observation.content_digest == note.content_digest
        and file.observation.serialization == reviewed.observation.serialization
    )


async def wedge_root(vault: Vault, owner, monkeypatch: pytest.MonkeyPatch) -> str:
    """Tail-edit a synced note under the OLD postcondition; return the op id."""

    with monkeypatch.context() as patch:
        # On a build that still ships the old comparison (the RED run) the
        # defect wedges the root by itself; nothing needs re-installing.
        if hasattr(NotesSyncExecutor, "_file_holds_note"):
            patch.setattr(
                NotesSyncExecutor, "_file_holds_note", staticmethod(old_postcondition)
            )
        vault.edit_note(TAIL_EDIT)
        await owner.note_changed("note-1")
        await owner.settle()
    # The defect's footprint: the write landed, the entry is fenced, the root
    # holds every later change in both directions.
    assert vault.file.read_bytes() == (TAIL_EDIT + "\n").encode("utf-8")
    assert vault.incomplete() == [
        ("update_file", "needs_attention", "postcondition_failed")
    ]
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("needs_attention", "resolve_cleanup")
    assert root.action_id is not None
    with pytest.raises(RuntimeError, match="sync_recovery_unresolved"):
        await owner.check_root("root-1")
    return root.action_id


__all__ = [
    "TAIL_EDIT",
    "VAULT_TEXT",
    "Vault",
    "build_owner",
    "old_postcondition",
    "wedge_root",
]
