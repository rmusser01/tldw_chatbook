"""TASK-34000.2 (review finding N-02): an end-of-note edit must not wedge sync.

A vault whose files end with a newline is the ordinary case. Ctrl+End in the
note editor parks the caret AFTER that newline, so the first word typed there
leaves the note's content without a trailing newline. ``serialize`` re-adds
the newline the file's captured profile carries, and the write postcondition
then compared the file's text with the RAW note content -- a correctly written
file never matched its own note, the operation was fenced at
``postcondition_failed``, and the whole folder stopped syncing both ways.

Every test here runs the production executor, runtime and POSIX filesystem
over a real ``CharactersRAGDB`` and a real file in a temp vault. The bytes on
disk and the rows in the stores are the evidence; nothing is mocked.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
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
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notes.notes_sync_authority import NotesScopeSyncAuthority
from tldw_chatbook.Notes.notes_sync_executor import (
    NotesSyncExecutionRequest,
    NotesSyncExecutor,
)
from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncOperationState,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_runtime_owner

pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: A vault file the way editors write it: every line ends with a newline.
_VAULT_TEXT = (
    "> The best time to plant a tree was 20 years ago.\n\n> Writing is thinking.\n"
)
#: Ctrl+End, then a word without Enter: the note no longer ends in "\n".
_TAIL_EDIT = _VAULT_TEXT + "second edit line"


class _Vault:
    """One real bound note/file pair in disposable authorities."""

    def __init__(self, tmp_path: Path, text: str = _VAULT_TEXT) -> None:
        self.notes_path = tmp_path / "notes.sqlite3"
        self.state_path = tmp_path / "sync.sqlite3"
        self.root = tmp_path / "vault"
        self.root.mkdir()
        self.file = self.root / "quotes.md"
        self.file.write_bytes(text.encode("utf-8"))
        database = CharactersRAGDB(self.notes_path, client_id="task-34000-2-seed")
        folders = LocalNoteFolderRepository(database)
        assert database.add_note("quotes", text, "note-1") == "note-1"
        folders.create_folder(name="VSync", parent_id=None, folder_id="folder-1")
        folders.reconcile_managed(owner_id="root-1", desired=(("folder-1", "note-1"),))
        with PosixNotesSyncFilesystem(self.root) as filesystem:
            baseline = filesystem.observe("quotes.md")
        assert baseline.observation.serialization.final_newline is True
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


@pytest.fixture
def vault(tmp_path: Path):
    selected = _Vault(tmp_path)
    try:
        yield selected
    finally:
        selected.close()


def _owner(vault: _Vault):
    return build_notes_sync_runtime_owner(
        notes_scope_service=vault.scope_service,
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=vault.state_path,
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )


async def test_executor_completes_a_tail_edit_under_a_final_newline_profile(
    vault: _Vault,
) -> None:
    """AC#5: the executor-level pin. No runtime, no planner -- one request."""

    vault.edit_note(_TAIL_EDIT)
    store = NotesDeviceStateStore(vault.state_path)
    filesystem = PosixNotesSyncFilesystem(vault.root)
    filesystem.__enter__()
    try:
        authority = NotesScopeSyncAuthority(
            vault.scope_service,
            scope=ScopeType.LOCAL_NOTE,
            user_id="user-1",
            note_scope_id="local_note",
        )
        note = await authority.observe("note-1")
        file = filesystem.observe("quotes.md")
        assert not note.content.endswith("\n")
        executor = NotesSyncExecutor(
            store, authority, filesystem, recovery_capacity_bytes=1024 * 1024
        )
        result = await executor.execute(
            NotesSyncExecutionRequest(
                operation_id="operation-tail",
                root_id="root-1",
                logical_folder_id="folder-1",
                direction=NotesSyncDirection.BIDIRECTIONAL,
                binding_id="binding-1",
                observation_token="observation-tail",
                action_kind=NotesSyncActionKind.UPDATE_FILE,
                note=note,
                file=file,
                desired_title=note.title,
                recovery_id="recovery-operation-tail",
                recovery_expires_at=2**62,
            )
        )

        assert (result.state, result.reason_code) == (
            NotesSyncOperationState.COMPLETED,
            None,
        )
        # The file keeps its own convention: the note's words, then the
        # newline the vault's profile carries.
        assert vault.file.read_bytes() == (_TAIL_EDIT + "\n").encode("utf-8")
        assert store.list_incomplete_operations() == ()
        written = filesystem.observe("quotes.md")
        binding = store.get_binding("binding-1")
        assert binding.content_digest == written.observation.content_digest
        assert binding.serialization.final_newline is True
    finally:
        filesystem.__exit__(None, None, None)
        store.close()


async def test_a_tail_edit_reaches_the_file_and_disk_edits_keep_flowing(
    vault: _Vault,
) -> None:
    """AC#1, end to end through the production runtime."""

    owner = _owner(vault)
    await owner.start()
    try:
        vault.edit_note(_TAIL_EDIT)
        await owner.note_changed("note-1")
        await owner.settle()

        assert vault.file.read_bytes() == (_TAIL_EDIT + "\n").encode("utf-8")
        assert vault.incomplete() == []

        # The folder still syncs in both directions afterwards: a later disk
        # append arrives in the note ...
        appended = _TAIL_EDIT + "\nfrom the vault\n"
        vault.file.write_bytes(appended.encode("utf-8"))
        assert owner.schedule_hint("root-1") is not None
        await owner.settle()
        assert vault.note()["content"] == appended
        assert vault.incomplete() == []

        # ... and the next tail edit in Notes still reaches the file.
        vault.edit_note(appended + "and back")
        await owner.note_changed("note-1")
        await owner.settle()
        assert vault.file.read_bytes() == (appended + "and back\n").encode("utf-8")
        assert vault.incomplete() == []
    finally:
        await owner.shutdown()


def _old_postcondition(file, note, reviewed) -> bool:
    """The comparison origin/dev 2d34cbf80d shipped (notes_sync_executor:5384).

    Re-installed only to put a root into the exact durable state the defect
    leaves -- ``update_file | needs_attention | postcondition_failed`` after
    the file write -- so the heal path is proved on the real rows.
    """

    return (
        file.observation.relative_path == reviewed.observation.relative_path
        and file.text == note.content
        and file.observation.content_digest == note.content_digest
        and file.observation.serialization == reviewed.observation.serialization
    )


async def _wedge(vault: _Vault, owner, monkeypatch: pytest.MonkeyPatch) -> str:
    """Tail-edit a synced note under the OLD postcondition; return the op id."""

    with monkeypatch.context() as patch:
        # On a build that still ships the old comparison (the RED run) the
        # defect wedges the root by itself; nothing needs re-installing.
        if hasattr(NotesSyncExecutor, "_file_holds_note"):
            patch.setattr(
                NotesSyncExecutor, "_file_holds_note", staticmethod(_old_postcondition)
            )
        vault.edit_note(_TAIL_EDIT)
        await owner.note_changed("note-1")
        await owner.settle()
    # The defect's footprint: the write landed, the entry is fenced, the root
    # holds every later change in both directions.
    assert vault.file.read_bytes() == (_TAIL_EDIT + "\n").encode("utf-8")
    assert vault.incomplete() == [
        ("update_file", "needs_attention", "postcondition_failed")
    ]
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("needs_attention", "resolve_cleanup")
    assert root.action_id is not None
    with pytest.raises(RuntimeError, match="sync_recovery_unresolved"):
        await owner.check_root("root-1")
    return root.action_id


async def test_recovery_heals_a_root_wedged_by_the_old_comparison(
    vault: _Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#2: Recovery closes the entry; nothing is lost and sync resumes."""

    owner = _owner(vault)
    await owner.start()
    try:
        operation_id = await _wedge(vault, owner, monkeypatch)

        await owner.resolve_cleanup("root-1", operation_id)

        assert vault.incomplete() == []
        root = owner.snapshot().roots[0]
        assert (root.status, root.next_action) == ("up_to_date", "sync_now")
        # Neither side was rewritten by the heal.
        assert vault.note()["content"] == _TAIL_EDIT
        assert vault.file.read_bytes() == (_TAIL_EDIT + "\n").encode("utf-8")
        # Checks work again, and a later disk edit reaches the note.
        await owner.check_root("root-1")
        appended = _TAIL_EDIT + "\nafter the heal\n"
        vault.file.write_bytes(appended.encode("utf-8"))
        assert owner.schedule_hint("root-1") is not None
        await owner.settle()
        assert vault.note()["content"] == appended
        assert vault.incomplete() == []
    finally:
        await owner.shutdown()


async def test_recovery_heals_a_wedge_the_user_kept_typing_past(
    vault: _Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#2: the typing that continued after the wedge reaches the file."""

    owner = _owner(vault)
    await owner.start()
    try:
        operation_id = await _wedge(vault, owner, monkeypatch)
        later = _TAIL_EDIT + " that kept going"
        vault.edit_note(later)
        # The held root ignores the save's hint -- this is the old lie.
        assert await owner.note_changed("note-1") == ()

        await owner.resolve_cleanup("root-1", operation_id)

        assert vault.incomplete() == []
        assert vault.note()["content"] == later
        assert vault.file.read_bytes() == (later + "\n").encode("utf-8")
        assert owner.snapshot().roots[0].status == "up_to_date"
    finally:
        await owner.shutdown()


async def test_a_mid_write_disk_edit_ends_in_a_conflict_review_not_a_loop(
    vault: _Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#3: the app/disk race lands on an ordinary conflict, both sides kept."""

    owner = _owner(vault)
    await owner.start()
    disk_side = _VAULT_TEXT + "| riley-disk | 2 |\n"
    original_admit = NotesSyncExecutor._admit

    def racing_admit(self, request):
        admitted = original_admit(self, request)
        # The disk edit lands after the plan and before the write.
        vault.file.write_bytes(disk_side.encode("utf-8"))
        return admitted

    try:
        with monkeypatch.context() as patch:
            patch.setattr(NotesSyncExecutor, "_admit", racing_admit)
            vault.edit_note(_VAULT_TEXT + "| riley-app | 1 | app2")
            await owner.note_changed("note-1")
            await owner.settle()
        assert vault.incomplete() == [
            ("update_file", "needs_attention", "stale_observation")
        ]
        root = owner.snapshot().roots[0]
        assert (root.status, root.next_action) == ("needs_attention", "resolve_cleanup")

        await owner.resolve_cleanup("root-1", root.action_id)

        # The entry is closed; the two edits now wait as a conflict for an
        # explicit Keep file / Keep note / Keep both. Nothing was overwritten.
        assert vault.incomplete() == []
        root = owner.snapshot().roots[0]
        assert (root.status, root.next_action) == ("needs_attention", "review_changes")
        assert vault.note()["content"] == _VAULT_TEXT + "| riley-app | 1 | app2"
        assert vault.file.read_bytes() == disk_side.encode("utf-8")
        plan = await owner.check_root("root-1")
        assert [(item.kind.value, item.reason_code) for item in plan.attention] == [
            ("conflict", "both_sides_changed")
        ]
    finally:
        await owner.shutdown()


async def test_a_raw_note_baseline_still_reads_as_unchanged(tmp_path: Path) -> None:
    """Baselines kept before this fix (and migrated ones) hold the RAW digest.

    A migrated binding carries a placeholder profile (no final newline) and
    the raw note digest. Measuring its note only in represented form would
    report every note ending in a newline as changed and plan a write nobody
    asked for; either form matching the baseline must read as unchanged.
    """

    vault = _Vault(tmp_path)
    try:
        store = NotesDeviceStateStore(vault.state_path)
        binding = store.get_binding("binding-1")
        placeholder = replace(
            binding,
            serialization=replace(binding.serialization, final_newline=False),
        )
        with store.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE notes_sync_bindings SET final_newline = 0 WHERE binding_id = ?",
                ("binding-1",),
            )
        assert store.get_binding("binding-1") == placeholder
        store.close()
        owner = _owner(vault)
        await owner.start()
        try:
            root = NotesDeviceStateStore(vault.state_path).get_root("root-1")
            observations = await owner._adapter.observe_root(root)
            [observed] = observations.bindings
            assert observed.note_digest == observed.baseline_note_digest
            assert vault.incomplete() == []
        finally:
            await owner.shutdown()
    finally:
        vault.close()
