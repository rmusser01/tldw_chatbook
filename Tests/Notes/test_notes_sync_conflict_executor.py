"""Durable executor contracts for reviewed Keep file/Keep note choices."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import tldw_chatbook.Notes.notes_sync_executor as executor_module
from Tests.Notes.test_notes_sync_executor import (
    BlockingFilesystem,
    FakeFilesystem,
    FakeNoteAuthority,
    InjectedCrash,
    _execution_store,
    _file,
    _note,
    _request,
)
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncOperationRecord,
    NotesSyncRecoveryRecord,
    NotesSyncRootRecord,
    notes_sync_recovery_envelope_digest,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notes.notes_sync_authority import (
    ConflictNoteRequest,
    ManualFolderRequest,
    ManualPlacementRequest,
    NotesScopeSyncAuthority,
    NotesSyncAuthorityError,
    NotesSyncNoteSnapshot,
    VerifiedFolder,
    VerifiedPlacement,
)
from tldw_chatbook.Notes.notes_sync_executor import (
    NotesSyncDirectionOverride,
    NotesSyncExecutionRequest,
    NotesSyncExecutor,
    NotesSyncKeepBothAuthority,
)
from tldw_chatbook.Notes.notes_sync_conflicts import linked_undo_operation_id
from tldw_chatbook.Notes.notes_sync_filesystem import (
    NotesSyncFilesystemError,
    NotesSyncFileSnapshot,
    PosixNotesSyncFilesystem,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncOperationState,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner


pytestmark = pytest.mark.unit

_CONFLICT_SUBSTAGES = (
    "recovery_admitted",
    "folders_established",
    "copy_created",
    "placement_created",
    "copy_verified",
    "bound_note_updated",
    "file_reverified",
    "binding_updated",
    "verified",
)


def _conflict_cas_metadata(
    stage: str,
    payload: bytes,
    *,
    checkpointed: bool = False,
    extra: dict[str, object] | None = None,
) -> bytes:
    parent_id = "actual-parent" if checkpointed else ""
    parent_version = "7" if checkpointed else ""
    child_id = "actual-child" if checkpointed else ""
    child_version = "9" if checkpointed else ""
    stage_index = (
        _CONFLICT_SUBSTAGES.index(stage) if stage in _CONFLICT_SUBSTAGES else -1
    )
    copy_version = (
        "1" if stage_index >= _CONFLICT_SUBSTAGES.index("copy_created") else ""
    )
    placement_id = (
        "placement-1"
        if stage_index >= _CONFLICT_SUBSTAGES.index("placement_created")
        else ""
    )
    placement_version = "1" if placement_id else ""
    longest = max(map(len, _CONFLICT_SUBSTAGES))
    metadata: dict[str, object] = {
        "conflict_parent_actual_folder_id": parent_id,
        "conflict_parent_actual_folder_id_padding": " " * (256 - len(parent_id)),
        "conflict_parent_actual_folder_version": parent_version,
        "conflict_parent_actual_folder_version_padding": " "
        * (20 - len(parent_version)),
        "conflict_copy_note_id": "copy-note",
        "conflict_copy_note_version": copy_version,
        "conflict_copy_note_version_padding": " " * (20 - len(copy_version)),
        "conflict_placement_membership_id": placement_id,
        "conflict_placement_membership_id_padding": " " * (256 - len(placement_id)),
        "conflict_placement_version": placement_version,
        "conflict_placement_version_padding": " " * (20 - len(placement_version)),
        "conflict_root_actual_folder_id": child_id,
        "conflict_root_actual_folder_id_padding": " " * (256 - len(child_id)),
        "conflict_root_actual_folder_version": child_version,
        "conflict_root_actual_folder_version_padding": " " * (20 - len(child_version)),
        "conflict_substage": stage,
        "conflict_substage_padding": " " * max(0, longest - len(stage)),
        "recovery_payload_digest": hashlib.sha256(payload).hexdigest(),
    }
    if extra:
        metadata.update(extra)
    return json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()


class _AdmissionCheckingFilesystem(FakeFilesystem):
    def __init__(self, snapshot: object, check: object) -> None:
        super().__init__(snapshot)
        self._check = check

    def replace(self, *args: object, **kwargs: object) -> object:
        assert callable(self._check)
        self._check()
        return super().replace(*args, **kwargs)


class _PathPreservingFilesystem(FakeFilesystem):
    def replace(self, *args: object, **kwargs: object) -> object:
        relative_path = str(args[0])
        snapshot = super().replace(*args, **kwargs)
        self.snapshot = replace(
            snapshot,
            observation=replace(snapshot.observation, relative_path=relative_path),
            reviewed_state=replace(
                snapshot.reviewed_state,
                relative_path=Path(relative_path),
            ),
        )
        return self.snapshot


class _KeepBothAuthority(FakeNoteAuthority):
    def __init__(self, snapshot: NotesSyncNoteSnapshot) -> None:
        super().__init__(snapshot)
        self.folders: dict[str, VerifiedFolder] = {}
        self.copy_note: NotesSyncNoteSnapshot | None = None
        self.placement: VerifiedPlacement | None = None
        self.effects: list[str] = []
        self.deleted_copy = False
        self.cancel_after_delete = False

    async def observe(self, note_id: str) -> NotesSyncNoteSnapshot:
        if self.copy_note is not None and note_id == self.copy_note.note_id:
            if self.deleted_copy:
                raise NotesSyncAuthorityError("note_missing")
            return self.copy_note
        return await super().observe(note_id)

    async def delete(self, expected: NotesSyncNoteSnapshot) -> None:
        assert self.copy_note == expected
        self.deleted_copy = True
        if self.cancel_after_delete:
            raise asyncio.CancelledError

    async def create_or_verify_manual_folder(
        self, request: ManualFolderRequest
    ) -> VerifiedFolder:
        self.effects.append(
            "parent_folder" if request.parent_id is None else "child_folder"
        )
        verified = self.folders.get(request.folder_id)
        if verified is None:
            verified = VerifiedFolder(
                request.folder_id,
                request.parent_id,
                "/" + "/".join(request.path_segments).casefold(),
                1,
            )
            self.folders[request.folder_id] = verified
        return verified

    async def verify_manual_folder(
        self,
        request: ManualFolderRequest,
        expected: VerifiedFolder,
    ) -> VerifiedFolder:
        del request
        observed = self.folders[expected.folder_id]
        assert (
            observed.folder_id,
            observed.parent_id,
            observed.version,
        ) == (
            expected.folder_id,
            expected.parent_id,
            expected.version,
        )
        return observed

    async def create_or_verify_conflict_note(
        self, request: ConflictNoteRequest
    ) -> NotesSyncNoteSnapshot:
        self.effects.append("copy_note")
        if self.copy_note is None:
            self.copy_note = NotesSyncNoteSnapshot(
                "local_note",
                request.note_id,
                request.title,
                request.content,
                1,
                hashlib.sha256(request.content.encode()).hexdigest(),
            )
        assert self.copy_note.title == request.title
        assert self.copy_note.content == request.content
        return self.copy_note

    async def create_or_verify_manual_placement(
        self, request: ManualPlacementRequest
    ) -> VerifiedPlacement:
        self.effects.append("placement")
        if self.placement is None:
            self.placement = VerifiedPlacement(
                "placement-1", request.folder_id, request.note_id, 1
            )
        return self.placement

    async def verify_conflict_note(
        self, request: ConflictNoteRequest
    ) -> NotesSyncNoteSnapshot:
        self.effects.append("verify_copy")
        assert self.copy_note is not None
        assert self.copy_note.content == request.content
        return self.copy_note

    async def verify_manual_placement(
        self, request: ManualPlacementRequest
    ) -> VerifiedPlacement:
        self.effects.append("verify_placement")
        assert self.placement is not None
        return self.placement


class _ReusedFolderAuthority(_KeepBothAuthority):
    async def create_or_verify_manual_folder(
        self, request: ManualFolderRequest
    ) -> VerifiedFolder:
        parent = request.parent_id is None
        folder_id = "actual-parent" if parent else "actual-child"
        version = 7 if parent else 9
        verified = VerifiedFolder(
            folder_id,
            request.parent_id,
            "/" + "/".join(request.path_segments).casefold(),
            version,
        )
        self.folders[folder_id] = verified
        self.effects.append("parent_folder" if parent else "child_folder")
        return verified


class _DatabaseLocalNotes:
    """Adapt the real local Notes database to the scope-service call shape."""

    def __init__(self, database: CharactersRAGDB) -> None:
        self.database = database

    def get_note_by_id(self, user_id: str, note_id: str) -> object:
        del user_id
        record = self.database.get_note_by_id(note_id)
        if record is not None and not isinstance(record.get("last_modified"), str):
            record["last_modified"] = record["last_modified"].isoformat()
        return record

    def update_note(
        self,
        user_id: str,
        note_id: str,
        update_data: dict[str, object],
        expected_version: int,
    ) -> object:
        del user_id
        return self.database.update_note(note_id, update_data, expected_version)

    def add_note(
        self,
        user_id: str,
        title: str,
        content: str,
        *,
        note_id: str,
    ) -> object:
        del user_id
        return self.database.add_note(title, content, note_id)


def _real_keep_both_authority(database: CharactersRAGDB) -> NotesScopeSyncAuthority:
    return NotesScopeSyncAuthority(
        NotesScopeService(
            _DatabaseLocalNotes(database),
            None,
            folder_repository=LocalNoteFolderRepository(database),
        ),
        scope=ScopeType.LOCAL_NOTE,
        user_id="user-1",
    )


class _BlockingKeepBothAuthority:
    def __init__(
        self,
        authority: NotesScopeSyncAuthority,
        store: NotesDeviceStateStore,
        target: str,
    ) -> None:
        self.authority = authority
        self.store = store
        self.target = target
        self.started = threading.Event()
        self.release = threading.Event()
        self._blocked = False
        self.calls: list[str] = []

    def _substage(self) -> str:
        recovery = self.store.find_operation_recovery("operation-1")
        if recovery is None:
            return ""
        return json.loads(recovery.metadata)["conflict_substage"]

    def _pause(self, target: str) -> None:
        self.calls.append(f"pause:{target}:{self._substage()}")
        if self.target != target or self._blocked:
            return
        self._blocked = True
        self.started.set()
        assert self.release.wait(5)

    async def observe(self, note_id: str) -> NotesSyncNoteSnapshot:
        return await self.authority.observe(note_id)

    async def replace(
        self,
        expected: NotesSyncNoteSnapshot,
        *,
        title: str,
        content: str,
    ) -> NotesSyncNoteSnapshot:
        result = await self.authority.replace(expected, title=title, content=content)
        self._pause("bound_note")
        return result

    async def create_or_verify_manual_folder(
        self, request: ManualFolderRequest
    ) -> VerifiedFolder:
        result = await self.authority.create_or_verify_manual_folder(request)
        self._pause("parent_folder" if request.parent_id is None else "child_folder")
        return result

    async def create_or_verify_conflict_note(
        self, request: ConflictNoteRequest
    ) -> NotesSyncNoteSnapshot:
        result = await self.authority.create_or_verify_conflict_note(request)
        self._pause("copy_note")
        return result

    async def verify_manual_folder(
        self,
        request: ManualFolderRequest,
        expected: VerifiedFolder,
    ) -> VerifiedFolder:
        self.calls.append(
            "verify_parent_folder"
            if request.parent_id is None
            else "verify_child_folder"
        )
        return await self.authority.verify_manual_folder(request, expected)

    async def create_or_verify_manual_placement(
        self, request: ManualPlacementRequest
    ) -> VerifiedPlacement:
        try:
            result = await self.authority.create_or_verify_manual_placement(request)
        except Exception as error:
            self.calls.append(f"placement_error:{type(error).__name__}:{error}")
            raise
        self._pause("placement")
        return result

    async def verify_conflict_note(
        self, request: ConflictNoteRequest
    ) -> NotesSyncNoteSnapshot:
        try:
            result = await self.authority.verify_conflict_note(request)
        except Exception as error:
            self.calls.append(f"verify_error:{type(error).__name__}:{error}")
            raise
        try:
            stage = self._substage()
        except Exception as error:
            self.calls.append(f"stage_error:{type(error).__name__}:{error}")
            raise
        self.calls.append(f"verify_stage:{stage}")
        if stage == "placement_created":
            self._pause("copy_verification")
        return result

    async def verify_manual_placement(
        self, request: ManualPlacementRequest
    ) -> VerifiedPlacement:
        self.calls.append("verify_placement")
        return await self.authority.verify_manual_placement(request)


def _keep_both_request(
    note: NotesSyncNoteSnapshot,
    file: NotesSyncFileSnapshot,
    *,
    direction: NotesSyncDirection = NotesSyncDirection.BIDIRECTIONAL,
) -> NotesSyncExecutionRequest:
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE,
            note=note,
            file=file,
        ),
        direction=direction,
        journal_kind="resolve_keep_both",
        keep_both=NotesSyncKeepBothAuthority(
            parent_folder_id="conflict-parent",
            parent_folder_name="Conflict copies",
            root_folder_id="conflict-child",
            root_folder_name="My synced notes",
            copy_note_id="conflict-copy-note",
            copy_title=note.title,
        ),
    )
    if direction is NotesSyncDirection.NOTES_TO_FOLDER:
        request = replace(
            request,
            direction_override=NotesSyncDirectionOverride(
                request.operation_id,
                NotesSyncActionKind.UPDATE_NOTE,
                request.observation_token,
            ),
        )
    return request


async def _prepare_real_keep_both(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    NotesDeviceStateStore,
    CharactersRAGDB,
    NotesScopeSyncAuthority,
    NotesSyncExecutionRequest,
]:
    notes_path = tmp_path / "notes.sqlite3"
    state_path = tmp_path / "state.sqlite3"
    sync_root = tmp_path / "sync-root"
    sync_root.mkdir()
    (sync_root / "note.md").write_bytes(b"file side")
    database = CharactersRAGDB(notes_path, client_id="before")
    assert database.add_note("Title", "before", "note-1") == "note-1"
    for version in range(1, 4):
        assert database.update_note(
            "note-1", {"title": "Title", "content": "before"}, version
        )
    authority = _real_keep_both_authority(database)
    store = NotesDeviceStateStore(state_path)
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-1",
            note_scope_id="local_note",
            logical_folder_id="folder-1",
            canonical_path=str(sync_root.resolve()),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.ACTIVE,
        )
    )
    with PosixNotesSyncFilesystem(sync_root) as filesystem:
        note = await authority.observe("note-1")
        file = filesystem.observe("note.md")
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-1",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest=NotesSyncExecutor.stable_identity_digest(file),
            state=NotesSyncBindingState.ACTIVE,
            serialization=file.observation.serialization,
            content_digest=note.content_digest,
            note_version=note.version,
        )
    )
    return (
        notes_path,
        state_path,
        sync_root,
        store,
        database,
        authority,
        _keep_both_request(note, file),
    )


async def _crash_real_keep_both_at_substage(
    *,
    store: NotesDeviceStateStore,
    authority: NotesScopeSyncAuthority,
    sync_root: Path,
    request: NotesSyncExecutionRequest,
    crash_substage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_advance = store.advance_conflict_substage

    def crash_after_checkpoint(*args: object, **kwargs: object) -> object:
        result = original_advance(*args, **kwargs)
        if kwargs.get("next_substage") == crash_substage:
            raise InjectedCrash
        return result

    monkeypatch.setattr(store, "advance_conflict_substage", crash_after_checkpoint)

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if (
            crash_substage == "recovery_admitted"
            and stage is NotesSyncOperationState.RECOVERY_ADMITTED
        ):
            raise InjectedCrash

    with PosixNotesSyncFilesystem(sync_root) as filesystem:
        with pytest.raises(InjectedCrash):
            await NotesSyncExecutor(
                store,
                authority,
                filesystem,
                recovery_capacity_bytes=65_536,
                after_stage=crash_after_admission,
            ).execute(request)


def _keep_both_database_effects(database: CharactersRAGDB) -> tuple[object, ...]:
    connection = database.get_connection()
    return tuple(
        tuple(tuple(row) for row in connection.execute(query).fetchall())
        for query in (
            "SELECT id, title, content, version, deleted FROM notes ORDER BY id",
            "SELECT id, parent_id, normalized_path, version, deleted "
            "FROM note_folders ORDER BY id",
            "SELECT id, folder_id, note_id, version, deleted "
            "FROM note_folder_memberships ORDER BY id",
        )
    )


def test_conflict_recovery_retention_is_exactly_thirty_days() -> None:
    assert executor_module.CONFLICT_RECOVERY_RETENTION_NS == (
        30 * 24 * 60 * 60 * 1_000_000_000
    )


@pytest.mark.asyncio
async def test_keep_both_executes_exact_copy_before_bound_note_sequence(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="before", version=4)
    file = _file(content="file side")
    notes = _KeepBothAuthority(note)
    files = FakeFilesystem(file)
    request = _keep_both_request(note, file)

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=65_536,
    ).execute(request)

    assert result.state is NotesSyncOperationState.COMPLETED, result.reason_code
    assert notes.copy_note is not None
    assert (notes.copy_note.title, notes.copy_note.content) == ("Title", "before")
    assert notes.snapshot.content == "file side"
    assert [
        notes.effects.index(effect)
        for effect in (
            "parent_folder",
            "child_folder",
            "copy_note",
            "placement",
        )
    ] == sorted(
        notes.effects.index(effect)
        for effect in (
            "parent_folder",
            "child_folder",
            "copy_note",
            "placement",
        )
    )
    assert notes.effects.index("placement") < notes.effects.index("verify_placement")
    binding = store.get_binding("binding-1")
    assert (binding.content_digest, binding.note_version) == (
        notes.snapshot.content_digest,
        notes.snapshot.version,
    )
    metadata = json.loads(store.load_operation_recovery(request.operation_id).metadata)
    assert metadata["conflict_substage"] == "verified"
    assert (
        metadata["conflict_copy_note_version"],
        metadata["conflict_placement_membership_id"],
        metadata["conflict_placement_version"],
    ) == ("1", "placement-1", "1")
    assert {
        key: metadata[key]
        for key in (
            "conflict_parent_folder_id",
            "conflict_parent_folder_name",
            "conflict_root_folder_id",
            "conflict_root_folder_name",
            "conflict_copy_note_id",
            "conflict_copy_title",
        )
    } == {
        "conflict_parent_folder_id": "conflict-parent",
        "conflict_parent_folder_name": "Conflict copies",
        "conflict_root_folder_id": "conflict-child",
        "conflict_root_folder_name": "My synced notes",
        "conflict_copy_note_id": "conflict-copy-note",
        "conflict_copy_title": "Title",
    }


@pytest.mark.asyncio
async def test_keep_both_folders_checkpoint_actual_reused_ids_and_versions(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="before", version=4)
    file = _file(content="file side")
    notes = _ReusedFolderAuthority(note)
    request = _keep_both_request(note, file)

    admitted = await NotesSyncExecutor(
        store,
        notes,
        FakeFilesystem(file),
        recovery_capacity_bytes=65_536,
    ).execute(request)

    assert admitted.state is NotesSyncOperationState.COMPLETED
    metadata = json.loads(store.load_operation_recovery("operation-1").metadata)
    assert (
        metadata["conflict_parent_actual_folder_id"],
        metadata["conflict_parent_actual_folder_version"],
        metadata["conflict_root_actual_folder_id"],
        metadata["conflict_root_actual_folder_version"],
    ) == ("actual-parent", "7", "actual-child", "9")


@pytest.mark.asyncio
@pytest.mark.parametrize("substitution", ("identity", "version"))
async def test_keep_both_restart_rejects_checkpointed_folder_substitution_before_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    (
        notes_path,
        state_path,
        sync_root,
        store,
        database,
        authority,
        request,
    ) = await _prepare_real_keep_both(tmp_path)
    repository = LocalNoteFolderRepository(database)
    parent = repository.create_folder(
        folder_id="actual-parent",
        name="Conflict copies",
        parent_id=None,
    )
    child = repository.create_folder(
        folder_id="actual-child",
        name="My synced notes",
        parent_id=parent.folder_id,
    )
    original_advance = store.advance_conflict_substage

    def crash_after_folders(*args: object, **kwargs: object) -> object:
        result = original_advance(*args, **kwargs)
        if kwargs.get("next_substage") == "folders_established":
            raise InjectedCrash
        return result

    monkeypatch.setattr(store, "advance_conflict_substage", crash_after_folders)
    with PosixNotesSyncFilesystem(sync_root) as filesystem:
        with pytest.raises(InjectedCrash):
            await NotesSyncExecutor(
                store,
                authority,
                filesystem,
                recovery_capacity_bytes=65_536,
            ).execute(request)

    if substitution == "identity":
        repository.soft_delete_folder(
            child.folder_id,
            expected_version=child.version,
        )
        repository.create_folder(
            folder_id="replacement-child",
            name="My synced notes",
            parent_id=parent.folder_id,
        )
    else:
        with database.transaction() as cursor:
            cursor.execute(
                "UPDATE note_folders SET version = version + 1 WHERE id = ?",
                (parent.folder_id,),
            )
    database.close_connection()

    reopened_database = CharactersRAGDB(notes_path, client_id="substitution")
    reopened_store = NotesDeviceStateStore(state_path)
    with PosixNotesSyncFilesystem(sync_root) as fresh_filesystem:
        fresh_executor = NotesSyncExecutor(
            reopened_store,
            _real_keep_both_authority(reopened_database),
            fresh_filesystem,
            recovery_capacity_bytes=65_536,
        )
        reconstructed = await fresh_executor.reconstruct_request("operation-1")
        result = await fresh_executor.resume(reconstructed)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "folder_authority_changed"
    assert (
        reopened_database.get_connection()
        .execute("SELECT COUNT(*) FROM notes WHERE deleted = 0")
        .fetchone()[0]
        == 1
    )
    reopened_database.close_connection()


@pytest.mark.asyncio
async def test_keep_both_restart_rejects_same_length_durable_payload_mutation(
    tmp_path: Path,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    (
        notes_path,
        state_path,
        sync_root,
        store,
        database,
        authority,
        request,
    ) = await _prepare_real_keep_both(tmp_path)

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with PosixNotesSyncFilesystem(sync_root) as filesystem:
        with pytest.raises(InjectedCrash):
            await NotesSyncExecutor(
                store,
                authority,
                filesystem,
                recovery_capacity_bytes=65_536,
                after_stage=crash_after_admission,
            ).execute(request)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_recovery SET payload = ? WHERE recovery_id = ?",
            (b"alter!", request.recovery_id),
        )
    database.close_connection()

    reopened_database = CharactersRAGDB(notes_path, client_id="payload-mutation")
    reopened_store = NotesDeviceStateStore(state_path)
    with PosixNotesSyncFilesystem(sync_root) as fresh_filesystem:
        fresh_executor = NotesSyncExecutor(
            reopened_store,
            _real_keep_both_authority(reopened_database),
            fresh_filesystem,
            recovery_capacity_bytes=65_536,
        )
        rejected = False
        try:
            reconstructed = await fresh_executor.reconstruct_request("operation-1")
        except RuntimeError as error:
            rejected = str(error) == "recovery_authority_changed"
        else:
            result = await fresh_executor.resume(reconstructed)
            rejected = result.reason_code == "recovery_authority_changed"

    assert rejected
    assert (
        reopened_database.get_connection()
        .execute("SELECT COUNT(*) FROM note_folders WHERE deleted = 0")
        .fetchone()[0]
        == 0
    )
    reopened_database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("crash_substage", _CONFLICT_SUBSTAGES)
@pytest.mark.parametrize("authority_mutation", ("root_direction", "binding"))
async def test_keep_both_restart_refences_owner_before_any_later_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_substage: str,
    authority_mutation: str,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    (
        notes_path,
        state_path,
        sync_root,
        store,
        database,
        authority,
        request,
    ) = await _prepare_real_keep_both(tmp_path)
    await _crash_real_keep_both_at_substage(
        store=store,
        authority=authority,
        sync_root=sync_root,
        request=request,
        crash_substage=crash_substage,
        monkeypatch=monkeypatch,
    )
    with store.transaction(immediate=True) as connection:
        if authority_mutation == "root_direction":
            connection.execute(
                "UPDATE notes_sync_roots SET direction = ? WHERE root_id = ?",
                (NotesSyncDirection.FOLDER_TO_NOTES.value, request.root_id),
            )
        else:
            connection.execute(
                "UPDATE notes_sync_bindings SET content_digest = ? "
                "WHERE binding_id = ?",
                ("f" * 64, request.binding_id),
            )
    expected_binding = store.get_binding(request.binding_id)
    expected_effects = _keep_both_database_effects(database)
    database.close_connection()

    reopened_store = NotesDeviceStateStore(state_path)
    reopened_database = CharactersRAGDB(notes_path, client_id="authority-mutation")
    fresh_authority = _BlockingKeepBothAuthority(
        _real_keep_both_authority(reopened_database),
        reopened_store,
        "never",
    )
    with PosixNotesSyncFilesystem(sync_root) as fresh_filesystem:
        executor = NotesSyncExecutor(
            reopened_store,
            fresh_authority,
            fresh_filesystem,
            recovery_capacity_bytes=65_536,
        )
        try:
            reconstructed = await executor.reconstruct_request(request.operation_id)
        except RuntimeError as error:
            reason = str(error)
        else:
            result = await executor.resume(reconstructed)
            reason = result.reason_code
            assert result.state is NotesSyncOperationState.NEEDS_ATTENTION

    assert reason in {"binding_authority_changed", "recovery_authority_changed"}
    assert (
        reopened_store.get_operation(request.operation_id).state
        is not NotesSyncOperationState.COMPLETED
    )
    assert fresh_authority.calls == []
    assert reopened_store.get_binding(request.binding_id) == expected_binding
    assert _keep_both_database_effects(reopened_database) == expected_effects
    recovery = reopened_store.load_operation_recovery(request.operation_id)
    assert json.loads(recovery.metadata)["conflict_substage"] == crash_substage
    reopened_database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crash_substage",
    ("recovery_admitted", "folders_established", "copy_created"),
)
@pytest.mark.parametrize("freshness_drift", ("note", "file"))
async def test_keep_both_restart_refences_reviewed_pair_before_early_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_substage: str,
    freshness_drift: str,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    (
        notes_path,
        state_path,
        sync_root,
        store,
        database,
        authority,
        request,
    ) = await _prepare_real_keep_both(tmp_path)
    await _crash_real_keep_both_at_substage(
        store=store,
        authority=authority,
        sync_root=sync_root,
        request=request,
        crash_substage=crash_substage,
        monkeypatch=monkeypatch,
    )
    database.close_connection()

    reopened_store = NotesDeviceStateStore(state_path)
    reopened_database = CharactersRAGDB(notes_path, client_id="freshness-drift")
    fresh_authority = _BlockingKeepBothAuthority(
        _real_keep_both_authority(reopened_database),
        reopened_store,
        "never",
    )
    with PosixNotesSyncFilesystem(sync_root) as fresh_filesystem:
        executor = NotesSyncExecutor(
            reopened_store,
            fresh_authority,
            fresh_filesystem,
            recovery_capacity_bytes=65_536,
        )
        reconstructed = await executor.reconstruct_request(request.operation_id)
        if freshness_drift == "note":
            note = reopened_database.get_note_by_id("note-1")
            assert note is not None
            assert reopened_database.update_note(
                "note-1",
                {"title": "Title", "content": "changed after reconstruction"},
                int(note["version"]),
            )
        else:
            (sync_root / "note.md").write_bytes(b"changed after reconstruction")
        expected_effects = _keep_both_database_effects(reopened_database)
        expected_binding = reopened_store.get_binding(request.binding_id)
        result = await executor.resume(reconstructed)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "stale_observation"
    assert not any(call.startswith("pause:") for call in fresh_authority.calls)
    assert reopened_store.get_binding(request.binding_id) == expected_binding
    assert _keep_both_database_effects(reopened_database) == expected_effects
    recovery = reopened_store.load_operation_recovery(request.operation_id)
    assert json.loads(recovery.metadata)["conflict_substage"] == crash_substage
    assert (sync_root / "note.md").read_bytes() == (
        b"changed after reconstruction" if freshness_drift == "file" else b"file side"
    )
    reopened_database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("authority_mutation", ("copy_version", "placement_version"))
async def test_keep_both_restart_rejects_changed_copy_or_placement_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority_mutation: str,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    (
        notes_path,
        state_path,
        sync_root,
        store,
        database,
        authority,
        request,
    ) = await _prepare_real_keep_both(tmp_path)
    crash_substage = (
        "copy_created" if authority_mutation == "copy_version" else "placement_created"
    )
    await _crash_real_keep_both_at_substage(
        store=store,
        authority=authority,
        sync_root=sync_root,
        request=request,
        crash_substage=crash_substage,
        monkeypatch=monkeypatch,
    )
    if authority_mutation == "copy_version":
        copy = database.get_note_by_id("conflict-copy-note")
        assert copy is not None
        assert database.update_note(
            "conflict-copy-note",
            {"title": "Title", "content": "changed"},
            int(copy["version"]),
        )
        assert database.update_note(
            "conflict-copy-note",
            {"title": "Title", "content": "before"},
            int(copy["version"]) + 1,
        )
    else:
        repository = LocalNoteFolderRepository(database)
        placement = repository.get_exact_manual_membership(
            folder_id="conflict-child",
            note_id="conflict-copy-note",
        )
        assert placement is not None
        assert repository.detach_manual(
            folder_id=placement[0].folder_id,
            note_id=placement[0].note_id,
            expected_version=placement[0].version,
        )
        revived = repository.attach_manual(
            folder_id=placement[0].folder_id,
            note_id=placement[0].note_id,
            expected_note_version=1,
        )
        assert revived.membership_id == placement[0].membership_id
        assert revived.version == placement[0].version + 2
    expected_effects = _keep_both_database_effects(database)
    database.close_connection()

    reopened_store = NotesDeviceStateStore(state_path)
    reopened_database = CharactersRAGDB(notes_path, client_id="effect-mutation")
    fresh_authority = _real_keep_both_authority(reopened_database)
    with PosixNotesSyncFilesystem(sync_root) as fresh_filesystem:
        executor = NotesSyncExecutor(
            reopened_store,
            fresh_authority,
            fresh_filesystem,
            recovery_capacity_bytes=65_536,
        )
        reconstructed = await executor.reconstruct_request(request.operation_id)
        result = await executor.resume(reconstructed)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "recovery_authority_changed"
    assert (
        json.loads(
            reopened_store.load_operation_recovery(request.operation_id).metadata
        )["conflict_substage"]
        == crash_substage
    )
    assert reopened_database.get_note_by_id("note-1")["content"] == "before"
    assert _keep_both_database_effects(reopened_database) == expected_effects
    reopened_database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crash_substage",
    (
        "copy_verified",
        "bound_note_updated",
        "file_reverified",
        "binding_updated",
        "verified",
    ),
)
@pytest.mark.parametrize("authority_drift", ("copy_version", "placement_version"))
async def test_keep_both_restart_verifies_conflict_pair_before_every_later_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_substage: str,
    authority_drift: str,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    (
        notes_path,
        state_path,
        sync_root,
        store,
        database,
        authority,
        request,
    ) = await _prepare_real_keep_both(tmp_path)
    await _crash_real_keep_both_at_substage(
        store=store,
        authority=authority,
        sync_root=sync_root,
        request=request,
        crash_substage=crash_substage,
        monkeypatch=monkeypatch,
    )
    if authority_drift == "copy_version":
        copy = database.get_note_by_id("conflict-copy-note")
        assert copy is not None
        assert database.update_note(
            "conflict-copy-note",
            {"title": "Title", "content": "changed"},
            int(copy["version"]),
        )
        assert database.update_note(
            "conflict-copy-note",
            {"title": "Title", "content": "before"},
            int(copy["version"]) + 1,
        )
    else:
        repository = LocalNoteFolderRepository(database)
        placement = repository.get_exact_manual_membership(
            folder_id="conflict-child",
            note_id="conflict-copy-note",
        )
        assert placement is not None
        assert repository.detach_manual(
            folder_id=placement[0].folder_id,
            note_id=placement[0].note_id,
            expected_version=placement[0].version,
        )
        revived = repository.attach_manual(
            folder_id=placement[0].folder_id,
            note_id=placement[0].note_id,
            expected_note_version=1,
        )
        assert revived.membership_id == placement[0].membership_id
        assert revived.version == placement[0].version + 2
    expected_effects = _keep_both_database_effects(database)
    expected_binding = store.get_binding(request.binding_id)
    database.close_connection()

    reopened_store = NotesDeviceStateStore(state_path)
    reopened_database = CharactersRAGDB(notes_path, client_id="pair-drift")
    fresh_authority = _BlockingKeepBothAuthority(
        _real_keep_both_authority(reopened_database),
        reopened_store,
        "never",
    )
    with PosixNotesSyncFilesystem(sync_root) as fresh_filesystem:
        executor = NotesSyncExecutor(
            reopened_store,
            fresh_authority,
            fresh_filesystem,
            recovery_capacity_bytes=65_536,
        )
        reconstructed = await executor.reconstruct_request(request.operation_id)
        result = await executor.resume(reconstructed)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "recovery_authority_changed"
    assert not any(call.startswith("pause:") for call in fresh_authority.calls)
    assert reopened_store.get_binding(request.binding_id) == expected_binding
    assert _keep_both_database_effects(reopened_database) == expected_effects
    recovery = reopened_store.load_operation_recovery(request.operation_id)
    assert json.loads(recovery.metadata)["conflict_substage"] == crash_substage
    reopened_database.close_connection()


def test_keep_both_folders_checkpoint_records_authority_without_byte_growth(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    payload = b"note side"
    longest = max(map(len, _CONFLICT_SUBSTAGES))
    metadata = json.dumps(
        {
            "conflict_copy_note_id": "copy-note",
            "conflict_copy_note_version": "",
            "conflict_copy_note_version_padding": " " * 20,
            "conflict_parent_actual_folder_id": "",
            "conflict_parent_actual_folder_id_padding": " " * 256,
            "conflict_parent_actual_folder_version": "",
            "conflict_parent_actual_folder_version_padding": " " * 20,
            "conflict_placement_membership_id": "",
            "conflict_placement_membership_id_padding": " " * 256,
            "conflict_placement_version": "",
            "conflict_placement_version_padding": " " * 20,
            "conflict_root_actual_folder_id": "",
            "conflict_root_actual_folder_id_padding": " " * 256,
            "conflict_root_actual_folder_version": "",
            "conflict_root_actual_folder_version_padding": " " * 20,
            "conflict_substage": "recovery_admitted",
            "conflict_substage_padding": " " * (longest - len("recovery_admitted")),
            "recovery_payload_digest": hashlib.sha256(payload).hexdigest(),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    store.admit_operation_recovery(
        NotesSyncOperationRecord(
            "operation-1",
            "root-1",
            "binding-1",
            "resolve_keep_both",
            NotesSyncOperationState.PENDING,
            None,
            "observation-1",
            4,
            notes_sync_recovery_envelope_digest("resolve_keep_both", payload, metadata),
        ),
        NotesSyncRecoveryRecord(
            "recovery-operation-1",
            "operation-1",
            payload,
            metadata,
            100_000,
        ),
        capacity_bytes=65_536,
    )

    store.advance_conflict_substage(
        operation_id="operation-1",
        recovery_id="recovery-operation-1",
        expected_operation_state=NotesSyncOperationState.RECOVERY_ADMITTED,
        expected_substage="recovery_admitted",
        next_substage="folders_established",
        expected_payload_digest=hashlib.sha256(payload).hexdigest(),
        expected_metadata_length=len(metadata),
        folder_authority=("actual-parent", 7, "actual-child", 9),
    )

    recovery = store.load_operation_recovery("operation-1")
    decoded = json.loads(recovery.metadata)
    assert len(recovery.metadata) == len(metadata)
    assert (
        decoded["conflict_parent_actual_folder_id"],
        decoded["conflict_parent_actual_folder_version"],
        decoded["conflict_root_actual_folder_id"],
        decoded["conflict_root_actual_folder_version"],
    ) == ("actual-parent", "7", "actual-child", "9")


@pytest.mark.asyncio
async def test_keep_both_restart_reconstructs_all_private_authority_after_admission(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    note = _note(content="before", version=4)
    file = _file(content="file side")
    request = _keep_both_request(note, file)

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            _KeepBothAuthority(note),
            FakeFilesystem(file),
            recovery_capacity_bytes=65_536,
            after_stage=crash_after_admission,
        ).execute(request)

    reopened = NotesDeviceStateStore(database)
    fresh_notes = _KeepBothAuthority(note)
    fresh_files = FakeFilesystem(file)
    fresh_executor = NotesSyncExecutor(
        reopened,
        fresh_notes,
        fresh_files,
        recovery_capacity_bytes=65_536,
    )
    reconstructed = await fresh_executor.reconstruct_request(request.operation_id)

    assert reconstructed.keep_both == request.keep_both
    result = await fresh_executor.resume(reconstructed)
    assert result.state is NotesSyncOperationState.COMPLETED, result.reason_code


@pytest.mark.asyncio
@pytest.mark.parametrize("crash_substage", _CONFLICT_SUBSTAGES)
async def test_keep_both_restart_reopens_every_durable_substage_with_real_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_substage: str,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    notes_path = tmp_path / "notes.sqlite3"
    state_path = tmp_path / "state.sqlite3"
    sync_root = tmp_path / "sync-root"
    sync_root.mkdir()
    (sync_root / "note.md").write_bytes(b"file side")
    database = CharactersRAGDB(notes_path, client_id="restart-before")
    assert database.add_note("Title", "before", "note-1") == "note-1"
    for version in range(1, 4):
        assert database.update_note(
            "note-1", {"title": "Title", "content": "before"}, version
        )
    authority = _real_keep_both_authority(database)
    store = NotesDeviceStateStore(state_path)
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-1",
            note_scope_id="local_note",
            logical_folder_id="folder-1",
            canonical_path=str(sync_root.resolve()),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.ACTIVE,
        )
    )
    with PosixNotesSyncFilesystem(sync_root) as filesystem:
        note = await authority.observe("note-1")
        file = filesystem.observe("note.md")
        store.create_binding(
            NotesSyncBindingRecord(
                binding_id="binding-1",
                root_id="root-1",
                note_scope_id="local_note",
                note_id="note-1",
                normalized_relative_path="note.md",
                stable_identity_digest=NotesSyncExecutor.stable_identity_digest(file),
                state=NotesSyncBindingState.ACTIVE,
                serialization=file.observation.serialization,
                content_digest=note.content_digest,
                note_version=note.version,
            )
        )
        request = _keep_both_request(note, file)
        original_advance = store.advance_conflict_substage

        def crash_after_checkpoint(*args: object, **kwargs: object) -> object:
            result = original_advance(*args, **kwargs)
            if kwargs.get("next_substage") == crash_substage:
                raise InjectedCrash
            return result

        monkeypatch.setattr(store, "advance_conflict_substage", crash_after_checkpoint)

        def crash_after_admission(stage: NotesSyncOperationState) -> None:
            if (
                crash_substage == "recovery_admitted"
                and stage is NotesSyncOperationState.RECOVERY_ADMITTED
            ):
                raise InjectedCrash

        with pytest.raises(InjectedCrash):
            await NotesSyncExecutor(
                store,
                authority,
                filesystem,
                recovery_capacity_bytes=65_536,
                after_stage=crash_after_admission,
            ).execute(request)
    database.close_connection()

    reopened_store = NotesDeviceStateStore(state_path)
    reopened_database = CharactersRAGDB(notes_path, client_id="restart-after")
    fresh_authority = _real_keep_both_authority(reopened_database)
    with PosixNotesSyncFilesystem(sync_root) as fresh_filesystem:
        fresh_executor = NotesSyncExecutor(
            reopened_store,
            fresh_authority,
            fresh_filesystem,
            recovery_capacity_bytes=65_536,
        )
        reconstructed = await fresh_executor.reconstruct_request("operation-1")
        result = await fresh_executor.resume(reconstructed)

    assert result.state is NotesSyncOperationState.COMPLETED, result.reason_code
    assert reopened_database.get_note_by_id("note-1")["content"] == "file side"
    connection = reopened_database.get_connection()
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM note_folders WHERE deleted = 0"
        ).fetchone()[0]
        == 2
    )
    assert (
        connection.execute("SELECT COUNT(*) FROM notes WHERE deleted = 0").fetchone()[0]
        == 2
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM note_folder_memberships WHERE deleted = 0"
        ).fetchone()[0]
        == 1
    )
    reopened_database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancel_target", "following_substage"),
    (
        ("parent_folder", "folders_established"),
        ("child_folder", "folders_established"),
        ("copy_note", "copy_created"),
        ("placement", "placement_created"),
        ("copy_verification", "copy_verified"),
        ("bound_note", "bound_note_updated"),
        ("file_recheck", "file_reverified"),
        ("binding_update", "binding_updated"),
        ("final_verification", "verified"),
    ),
)
async def test_keep_both_cancellation_joins_effect_and_checkpoint_then_fresh_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_target: str,
    following_substage: str,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    (
        notes_path,
        state_path,
        sync_root,
        store,
        database,
        authority,
        request,
    ) = await _prepare_real_keep_both(tmp_path)
    blocking = _BlockingKeepBothAuthority(authority, store, cancel_target)
    if cancel_target in {"file_recheck", "binding_update", "final_verification"}:
        original_advance = store.advance_conflict_substage

        def pause_after_effect(*args: object, **kwargs: object) -> object:
            if kwargs.get("next_substage") == following_substage:
                blocking.calls.append(f"effect_returned:{cancel_target}")
                blocking._pause(cancel_target)
            return original_advance(*args, **kwargs)

        monkeypatch.setattr(store, "advance_conflict_substage", pause_after_effect)
    with PosixNotesSyncFilesystem(sync_root) as filesystem:
        task = asyncio.create_task(
            NotesSyncExecutor(
                store,
                blocking,
                filesystem,
                recovery_capacity_bytes=65_536,
            ).execute(request)
        )
        started = await asyncio.to_thread(blocking.started.wait, 5)
        if not started:
            unexpected = await task
            pytest.fail(
                "effect did not begin: "
                f"{unexpected.state}/{unexpected.reason_code}/{blocking.calls}/"
                f"{json.loads(store.load_operation_recovery('operation-1').metadata)}"
            )
        if cancel_target in {
            "file_recheck",
            "binding_update",
            "final_verification",
        }:
            assert f"effect_returned:{cancel_target}" in blocking.calls
        task.cancel()
        blocking.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    metadata = json.loads(store.load_operation_recovery("operation-1").metadata)
    assert metadata["conflict_substage"] == following_substage
    database.close_connection()

    reopened_database = CharactersRAGDB(notes_path, client_id="after")
    fresh_authority = _real_keep_both_authority(reopened_database)
    reopened_store = NotesDeviceStateStore(state_path)
    with PosixNotesSyncFilesystem(sync_root) as fresh_filesystem:
        fresh_executor = NotesSyncExecutor(
            reopened_store,
            fresh_authority,
            fresh_filesystem,
            recovery_capacity_bytes=65_536,
        )
        reconstructed = await fresh_executor.reconstruct_request("operation-1")
        result = await fresh_executor.resume(reconstructed)

    assert result.state is NotesSyncOperationState.COMPLETED, result.reason_code
    reopened_database.close_connection()


def test_keep_both_substage_cas_is_capacity_neutral_and_exactly_forward(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    payload = b"note side"
    longest = max(map(len, _CONFLICT_SUBSTAGES))
    metadata = _conflict_cas_metadata(
        "recovery_admitted", payload, extra={"private": "authority"}
    )
    admitted = store.admit_operation_recovery(
        NotesSyncOperationRecord(
            operation_id="operation-keep-both",
            root_id="root-1",
            binding_id="binding-1",
            kind="resolve_keep_both",
            state=NotesSyncOperationState.PENDING,
            reason_code=None,
            observation_token="observation-1",
            expected_note_version=4,
            expected_file_digest=notes_sync_recovery_envelope_digest(
                "resolve_keep_both", payload, metadata
            ),
        ),
        NotesSyncRecoveryRecord(
            recovery_id="recovery-keep-both",
            operation_id="operation-keep-both",
            payload=payload,
            metadata=metadata,
            expires_at=10_000,
        ),
        capacity_bytes=len(payload) + len(metadata),
    )
    assert admitted.admitted is True
    expected_length = len(metadata)
    expected_state = NotesSyncOperationState.RECOVERY_ADMITTED
    for current, following in zip(_CONFLICT_SUBSTAGES, _CONFLICT_SUBSTAGES[1:]):
        if following == "bound_note_updated":
            next_state = NotesSyncOperationState.FIRST_AUTHORITY_APPLIED
        elif following == "file_reverified":
            next_state = NotesSyncOperationState.SECOND_AUTHORITY_APPLIED
        elif following == "binding_updated":
            store.transition_operation(
                "operation-keep-both", NotesSyncOperationState.BINDING_UPDATED
            )
            expected_state = NotesSyncOperationState.BINDING_UPDATED
            next_state = NotesSyncOperationState.BINDING_UPDATED
        elif following == "verified":
            next_state = NotesSyncOperationState.VERIFIED
        else:
            next_state = expected_state
        store.advance_conflict_substage(
            operation_id="operation-keep-both",
            recovery_id="recovery-keep-both",
            expected_operation_state=expected_state,
            expected_substage=current,
            next_substage=following,
            expected_payload_digest=hashlib.sha256(payload).hexdigest(),
            expected_metadata_length=expected_length,
            folder_authority=("actual-parent", 7, "actual-child", 9)
            if current == "recovery_admitted"
            else None,
            copy_authority=("copy-note", 1)
            if current == "folders_established"
            else None,
            placement_authority=("placement-1", 1)
            if current == "copy_created"
            else None,
        )
        recovery = store.load_operation_recovery("operation-keep-both")
        decoded = json.loads(recovery.metadata)
        assert len(recovery.metadata) == expected_length
        assert decoded["conflict_substage"] == following
        assert len(decoded["conflict_substage_padding"]) == longest - len(following)
        if following == "copy_created":
            assert decoded["conflict_copy_note_version"] == "1"
            assert len(decoded["conflict_copy_note_version_padding"]) == 19
        if following == "placement_created":
            assert decoded["conflict_placement_membership_id"] == "placement-1"
            assert decoded["conflict_placement_version"] == "1"
            assert len(decoded["conflict_placement_membership_id_padding"]) == 245
            assert len(decoded["conflict_placement_version_padding"]) == 19
        assert store.get_operation("operation-keep-both").state is next_state
        expected_state = next_state


@pytest.mark.parametrize(
    ("current", "following"),
    (
        ("unknown", "folders_established"),
        ("recovery_admitted", "copy_created"),
        ("copy_created", "folders_established"),
    ),
)
def test_keep_both_substage_cas_rejects_unknown_skip_and_backward(
    tmp_path: Path,
    current: str,
    following: str,
) -> None:
    store, _database = _execution_store(tmp_path)
    longest = max(map(len, _CONFLICT_SUBSTAGES))
    metadata = json.dumps(
        {
            "conflict_substage": current,
            "conflict_substage_padding": " " * max(0, longest - len(current)),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    payload = b"note side"
    store.admit_operation_recovery(
        NotesSyncOperationRecord(
            "operation-keep-both",
            "root-1",
            "binding-1",
            "resolve_keep_both",
            NotesSyncOperationState.PENDING,
            None,
            "observation-1",
            4,
            notes_sync_recovery_envelope_digest("resolve_keep_both", payload, metadata),
        ),
        NotesSyncRecoveryRecord(
            "recovery-keep-both",
            "operation-keep-both",
            payload,
            metadata,
            10_000,
        ),
        capacity_bytes=4096,
    )

    with pytest.raises(Exception, match="substage|corrupt|allowed"):
        store.advance_conflict_substage(
            operation_id="operation-keep-both",
            recovery_id="recovery-keep-both",
            expected_operation_state=NotesSyncOperationState.RECOVERY_ADMITTED,
            expected_substage=current,
            next_substage=following,
            expected_payload_digest=hashlib.sha256(payload).hexdigest(),
            expected_metadata_length=len(metadata),
        )


@pytest.mark.parametrize(
    "corruption",
    ("padding", "metadata_length", "payload_digest", "operation_state"),
)
def test_keep_both_substage_cas_rejects_padding_length_digest_and_state_drift(
    tmp_path: Path,
    corruption: str,
) -> None:
    store, _database = _execution_store(tmp_path)
    payload = b"note side"
    metadata = _conflict_cas_metadata("recovery_admitted", payload)
    store.admit_operation_recovery(
        NotesSyncOperationRecord(
            "operation-keep-both",
            "root-1",
            "binding-1",
            "resolve_keep_both",
            NotesSyncOperationState.PENDING,
            None,
            "observation-1",
            4,
            notes_sync_recovery_envelope_digest("resolve_keep_both", payload, metadata),
        ),
        NotesSyncRecoveryRecord(
            "recovery-keep-both",
            "operation-keep-both",
            payload,
            metadata,
            10_000,
        ),
        capacity_bytes=4096,
    )
    if corruption == "padding":
        decoded = json.loads(metadata)
        decoded["conflict_substage_padding"] = (
            "x" + decoded["conflict_substage_padding"][1:]
        )
        with store.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE notes_sync_recovery SET metadata = ? WHERE recovery_id = ?",
                (
                    json.dumps(decoded, separators=(",", ":"), sort_keys=True).encode(),
                    "recovery-keep-both",
                ),
            )

    with pytest.raises(Exception, match="corrupt|substage|state"):
        store.advance_conflict_substage(
            operation_id="operation-keep-both",
            recovery_id="recovery-keep-both",
            expected_operation_state=(
                NotesSyncOperationState.PENDING
                if corruption == "operation_state"
                else NotesSyncOperationState.RECOVERY_ADMITTED
            ),
            expected_substage="recovery_admitted",
            next_substage="folders_established",
            expected_payload_digest=(
                "0" * 64
                if corruption == "payload_digest"
                else hashlib.sha256(payload).hexdigest()
            ),
            expected_metadata_length=(
                len(metadata) + 1 if corruption == "metadata_length" else len(metadata)
            ),
            folder_authority=("actual-parent", 7, "actual-child", 9),
        )


@pytest.mark.parametrize(
    ("journal_kind", "action"),
    (
        ("resolve_keep_file", NotesSyncActionKind.UPDATE_NOTE),
        ("resolve_keep_note", NotesSyncActionKind.UPDATE_FILE),
    ),
)
def test_resolution_journal_kind_requires_its_existing_underlying_action(
    journal_kind: str,
    action: NotesSyncActionKind,
) -> None:
    note = _note(content="note side", version=5)
    file = _file(content="file side")
    request = replace(
        _request(action=action, note=note, file=file),
        journal_kind=journal_kind,
    )

    assert request.action_kind is action
    assert request.journal_kind == journal_kind
    with pytest.raises(ValueError, match="journal_kind"):
        replace(
            request,
            action_kind=(
                NotesSyncActionKind.UPDATE_FILE
                if action is NotesSyncActionKind.UPDATE_NOTE
                else NotesSyncActionKind.UPDATE_NOTE
            ),
        )
    with pytest.raises(ValueError, match="journal_kind"):
        replace(request, journal_kind="resolve_keep_both")


@pytest.mark.parametrize(
    ("journal_kind", "action", "permitted", "disallowed"),
    (
        (
            "resolve_keep_file",
            NotesSyncActionKind.UPDATE_NOTE,
            NotesSyncDirection.FOLDER_TO_NOTES,
            NotesSyncDirection.NOTES_TO_FOLDER,
        ),
        (
            "resolve_keep_note",
            NotesSyncActionKind.UPDATE_FILE,
            NotesSyncDirection.NOTES_TO_FOLDER,
            NotesSyncDirection.FOLDER_TO_NOTES,
        ),
    ),
)
def test_resolution_override_is_exact_occurrence_authority(
    journal_kind: str,
    action: NotesSyncActionKind,
    permitted: NotesSyncDirection,
    disallowed: NotesSyncDirection,
) -> None:
    note = _note(content="note side", version=5)
    file = _file(content="file side")
    request = replace(
        _request(action=action, note=note, file=file),
        journal_kind=journal_kind,
    )

    assert request.direction is NotesSyncDirection.BIDIRECTIONAL
    assert request.direction_override is None
    assert replace(request, direction=permitted).direction_override is None
    with pytest.raises(ValueError, match="direction_override"):
        replace(request, direction=disallowed)

    exact_override = NotesSyncDirectionOverride(
        review_id=request.operation_id,
        action_kind=action,
        observation_token=request.observation_token,
    )
    overridden = replace(
        request,
        direction=disallowed,
        direction_override=exact_override,
    )
    assert overridden.direction_override == exact_override
    with pytest.raises(ValueError, match="direction_override"):
        replace(request, direction_override=exact_override)
    with pytest.raises(ValueError, match="direction_override"):
        replace(
            request,
            direction=disallowed,
            direction_override=replace(exact_override, review_id="operation-other"),
        )


@pytest.mark.asyncio
async def test_resolution_recovery_rejects_corrupt_override_review_id_before_write(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_roots SET direction = 'notes_to_folder' "
            "WHERE root_id = 'root-1'"
        )
    note = _note(content="note side", version=5)
    file = _file(content="file side")
    request = replace(
        _request(action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=file),
        direction=NotesSyncDirection.NOTES_TO_FOLDER,
        journal_kind="resolve_keep_file",
        direction_override=NotesSyncDirectionOverride(
            review_id="operation-1",
            action_kind=NotesSyncActionKind.UPDATE_NOTE,
            observation_token="observation-1",
        ),
    )
    notes = FakeNoteAuthority(note)
    files = FakeFilesystem(file)

    def stop_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=stop_after_admission,
        ).execute(request)

    with store.transaction(immediate=True) as connection:
        raw = connection.execute(
            "SELECT metadata FROM notes_sync_recovery WHERE operation_id = ?",
            (request.operation_id,),
        ).fetchone()[0]
        metadata = json.loads(raw)
        metadata["direction_override"]["review_id"] = "operation-other"
        connection.execute(
            "UPDATE notes_sync_recovery SET metadata = ? WHERE operation_id = ?",
            (
                json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode(),
                request.operation_id,
            ),
        )

    executor = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    )
    with pytest.raises(RuntimeError, match="recovery_authority_changed"):
        await executor.reconstruct_request(request.operation_id)
    result = await executor.resume(request)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.recovery_required is True
    assert notes.replace_calls == 0
    assert files.replace_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("journal_kind", "action"),
    (
        ("resolve_keep_file", NotesSyncActionKind.UPDATE_NOTE),
        ("resolve_keep_note", NotesSyncActionKind.UPDATE_FILE),
    ),
)
async def test_resolution_kind_and_underlying_action_survive_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal_kind: str,
    action: NotesSyncActionKind,
) -> None:
    import tldw_chatbook.Notes.notes_device_state_store as store_module

    store, database = _execution_store(tmp_path)
    admission_now = 50_000_000_000
    monkeypatch.setattr(store_module, "_now", lambda: admission_now)
    note = _note(content="note side", version=5)
    file = _file(content="file side")
    request = replace(
        _request(action=action, note=note, file=file),
        journal_kind=journal_kind,
        recovery_expires_at=900_000,
    )
    notes = FakeNoteAuthority(note)
    files = FakeFilesystem(file)

    def stop_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=stop_after_admission,
        ).execute(request)

    operation = store.get_operation(request.operation_id)
    recovery = store.load_operation_recovery(request.operation_id)
    metadata = json.loads(recovery.metadata)
    assert operation.kind == journal_kind
    assert metadata["underlying_action_kind"] == action.value
    assert metadata["action"] == action.value
    assert recovery.expires_at == (
        admission_now + executor_module.CONFLICT_RECOVERY_RETENTION_NS
    )

    reconstructed = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).reconstruct_request(request.operation_id)
    assert type(reconstructed) is NotesSyncExecutionRequest
    assert reconstructed.action_kind is action
    assert reconstructed.journal_kind == journal_kind


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("journal_kind", "action"),
    (
        ("resolve_keep_file", NotesSyncActionKind.UPDATE_NOTE),
        ("resolve_keep_note", NotesSyncActionKind.UPDATE_FILE),
    ),
)
async def test_conflict_recovery_is_admitted_before_first_write(
    tmp_path: Path,
    journal_kind: str,
    action: NotesSyncActionKind,
) -> None:
    store, _ = _execution_store(tmp_path)
    note = _note(content="note side", version=5)
    file = _file(content="file side")
    request = replace(
        _request(action=action, note=note, file=file),
        journal_kind=journal_kind,
    )

    def assert_admitted_before_write() -> None:
        assert (
            store.get_operation(request.operation_id).state
            is NotesSyncOperationState.RECOVERY_ADMITTED
        )
        assert store.load_operation_recovery(request.operation_id).operation_id == (
            request.operation_id
        )

    notes = FakeNoteAuthority(note, on_replace=assert_admitted_before_write)
    files = _AdmissionCheckingFilesystem(file, assert_admitted_before_write)

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).execute(request)

    assert result.state is NotesSyncOperationState.COMPLETED


@pytest.mark.asyncio
async def test_each_selected_conflict_gets_thirty_days_from_its_own_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.Notes.notes_device_state_store as store_module

    store, _ = _execution_store(tmp_path)
    first_file = _file(content="before")
    second_source = _file(content="before two", inode=22)
    second_file = replace(
        second_source,
        observation=replace(
            second_source.observation,
            relative_path="second.md",
        ),
        reviewed_state=replace(
            second_source.reviewed_state,
            relative_path=Path("second.md"),
        ),
    )
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-2",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-2",
            normalized_relative_path="second.md",
            stable_identity_digest=NotesSyncExecutor.stable_identity_digest(
                second_file
            ),
            state=NotesSyncBindingState.ACTIVE,
            serialization=second_file.observation.serialization,
            content_digest=second_file.observation.content_digest,
            note_version=4,
        )
    )
    first_now = 10_000_000_000
    second_now = first_now + 9_000_000_000
    clock = {"now": first_now}
    monkeypatch.setattr(store_module, "_now", lambda: clock["now"])
    constructed_expiry = first_now + executor_module.CONFLICT_RECOVERY_RETENTION_NS
    first_note = _note(content="after", version=4)
    second_note = replace(
        _note(content="after two", version=4),
        note_id="note-2",
    )
    first_request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_FILE,
            note=first_note,
            file=first_file,
            operation_id="operation-first",
        ),
        journal_kind="resolve_keep_note",
        recovery_expires_at=constructed_expiry,
    )
    second_request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_FILE,
            note=second_note,
            file=second_file,
            operation_id="operation-second",
        ),
        binding_id="binding-2",
        recovery_id="recovery-operation-second",
        journal_kind="resolve_keep_note",
        recovery_expires_at=constructed_expiry,
    )
    first_files = BlockingFilesystem(first_file)
    first = asyncio.create_task(
        NotesSyncExecutor(
            store,
            FakeNoteAuthority(first_note),
            first_files,
            recovery_capacity_bytes=65_536,
        ).execute(first_request)
    )
    try:
        assert await asyncio.to_thread(first_files.started.wait, 3.0)
        clock["now"] = second_now
        second = await NotesSyncExecutor(
            store,
            FakeNoteAuthority(second_note),
            _PathPreservingFilesystem(second_file),
            recovery_capacity_bytes=65_536,
        ).execute(second_request)
    finally:
        first_files.release.set()
    first_result = await first

    assert first_result.state is NotesSyncOperationState.COMPLETED
    assert second.state is NotesSyncOperationState.COMPLETED
    assert store.load_operation_recovery(first_request.operation_id).expires_at == (
        first_now + executor_module.CONFLICT_RECOVERY_RETENTION_NS
    )
    assert store.load_operation_recovery(second_request.operation_id).expires_at == (
        second_now + executor_module.CONFLICT_RECOVERY_RETENTION_NS
    )


@pytest.mark.parametrize(
    ("journal_kind", "action"),
    (
        ("resolve_keep_file", NotesSyncActionKind.UPDATE_NOTE),
        ("resolve_keep_note", NotesSyncActionKind.UPDATE_FILE),
    ),
)
@pytest.mark.asyncio
async def test_undo_resolution_restores_conflict_with_fresh_active_binding(
    tmp_path: Path,
    journal_kind: str,
    action: NotesSyncActionKind,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    file = _file(content="file side")
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(file)
    request = replace(
        _request(action=action, note=note, file=file),
        journal_kind=journal_kind,
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)

    resolved = await executor.execute(request)
    source_before = store.get_operation(request.operation_id)
    source_recovery = store.load_operation_recovery(request.operation_id)
    executor._require_source_resolution_recovery(
        source_before,
        source_recovery,
        json.loads(source_recovery.metadata),
        store.get_binding("binding-1"),
        notes.snapshot,
        files.snapshot,
    )
    undone = await executor.undo_resolution("root-1", request.operation_id)

    assert resolved.state is NotesSyncOperationState.COMPLETED
    linked = linked_undo_operation_id("root-1", request.operation_id)
    assert undone.state is NotesSyncOperationState.COMPLETED, (
        undone.reason_code,
        store.find_operation(linked),
        (
            json.loads(store.load_operation_recovery(linked).metadata)
            if store.find_operation_recovery(linked) is not None
            else None
        ),
    )
    assert notes.snapshot.content == "note side"
    assert files.snapshot.text == "file side"
    binding = store.get_binding("binding-1")
    assert binding.state is NotesSyncBindingState.ACTIVE
    assert binding.note_version == notes.snapshot.version
    assert binding.content_digest == hashlib.sha256(b"before").hexdigest()
    assert store.get_operation(request.operation_id) == source_before
    assert store.get_operation(linked).kind == "undo_resolution"

    duplicate = await executor.undo_resolution("root-1", request.operation_id)
    assert duplicate.state is NotesSyncOperationState.COMPLETED
    assert notes.replace_calls == (
        2 if action is NotesSyncActionKind.UPDATE_NOTE else 0
    )
    assert files.replace_calls == (
        2 if action is NotesSyncActionKind.UPDATE_FILE else 0
    )


@pytest.mark.asyncio
async def test_undo_uses_reviewed_note_version_not_pre_conflict_binding_version(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_bindings SET note_version = 3 "
            "WHERE binding_id = 'binding-1'"
        )
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE,
            note=note,
            file=files.snapshot,
        ),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED

    result = await executor.undo_resolution("root-1", request.operation_id)

    assert result.state is NotesSyncOperationState.COMPLETED, result.reason_code
    assert notes.snapshot.content == "note side"
    assert files.snapshot.text == "file side"


@pytest.mark.asyncio
async def test_second_resolution_after_undo_uses_fresh_note_version(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    file = _file(content="file side")
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(file)
    first = replace(
        _request(action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=file),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(first)).state is NotesSyncOperationState.COMPLETED
    assert (
        await executor.undo_resolution("root-1", first.operation_id)
    ).state is NotesSyncOperationState.COMPLETED
    second = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE,
            note=notes.snapshot,
            file=files.snapshot,
            operation_id="operation-2",
        ),
        observation_token="observation-2",
        journal_kind="resolve_keep_file",
    )

    result = await executor.execute(second)

    assert result.state is NotesSyncOperationState.COMPLETED, result.reason_code
    assert store.get_binding("binding-1").note_version == notes.snapshot.version


@pytest.mark.asyncio
async def test_undo_keep_both_restores_before_deleting_only_unchanged_copy(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    file = _file(content="file side")
    notes = _KeepBothAuthority(note)
    files = _PathPreservingFilesystem(file)
    request = _keep_both_request(note, file)
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED

    result = await executor.undo_resolution("root-1", request.operation_id)

    assert result.state is NotesSyncOperationState.COMPLETED, result.reason_code
    assert notes.snapshot.content == "note side"
    assert files.snapshot.text == "file side"
    assert notes.deleted_copy is True
    assert store.get_binding("binding-1").note_version == notes.snapshot.version


@pytest.mark.asyncio
async def test_undo_keep_both_refuses_edited_copy_without_mutating_authority(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    file = _file(content="file side")
    notes = _KeepBothAuthority(note)
    files = _PathPreservingFilesystem(file)
    request = _keep_both_request(note, file)
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    assert notes.copy_note is not None
    notes.copy_note = replace(
        notes.copy_note,
        content="edited copy",
        content_digest=hashlib.sha256(b"edited copy").hexdigest(),
        version=notes.copy_note.version + 1,
    )
    note_before = notes.snapshot

    result = await executor.undo_resolution("root-1", request.operation_id)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "changed_since_resolution"
    assert notes.snapshot == note_before
    assert notes.deleted_copy is False


@pytest.mark.asyncio
async def test_undo_expiry_boundary_and_wrong_root_refuse_before_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.Notes.notes_device_state_store as store_module

    now = 1_000_000_000
    monkeypatch.setattr(store_module, "_now", lambda: now)
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    file = _file(content="file side")
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(file)
    request = replace(
        _request(action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=file),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    expiry = store.load_operation_recovery(request.operation_id).expires_at

    with pytest.raises(ValueError, match="wrong_root"):
        await executor.undo_resolution(
            "root-other", request.operation_id, now=expiry - 1
        )
    expired = await executor.undo_resolution("root-1", request.operation_id, now=expiry)

    assert expired.reason_code == "undo_expired"
    assert (
        store.find_operation(linked_undo_operation_id("root-1", request.operation_id))
        is None
    )
    assert notes.snapshot.content == "file side"


@pytest.mark.asyncio
async def test_undo_projection_is_fresh_at_expiry_drift_partial_and_completion(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    file = _file(content="file side")
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(file)
    request = replace(
        _request(action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=file),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    expiry = store.load_operation_recovery(request.operation_id).expires_at

    available = await executor.inspect_resolution_undo(
        "root-1", request.operation_id, now=expiry - 1
    )
    assert available.undo_available is True
    assert available.undo_reason is None
    assert available.note_title == notes.snapshot.title
    assert available.relative_path == files.snapshot.observation.relative_path

    expired = await executor.inspect_resolution_undo(
        "root-1", request.operation_id, now=expiry
    )
    assert expired.undo_available is False
    assert expired.undo_reason == "Undo expired"

    notes.snapshot = replace(notes.snapshot, title="Edited after resolution")
    changed = await executor.inspect_resolution_undo(
        "root-1", request.operation_id, now=expiry - 1
    )
    assert changed.undo_available is False
    assert changed.undo_reason == "Changed since resolution"
    notes.snapshot = replace(notes.snapshot, title="Title")

    linked = linked_undo_operation_id("root-1", request.operation_id)

    def crash_after_admission(state: NotesSyncOperationState) -> None:
        if state is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=65_536,
            after_stage=crash_after_admission,
        ).undo_resolution("root-1", request.operation_id, now=expiry - 1)
    partial = await executor.inspect_resolution_undo(
        "root-1", request.operation_id, now=expiry - 1
    )
    assert partial.undo_available is False
    assert partial.undo_reason == "Undo in progress"

    resumed = await executor.resume(await executor.reconstruct_request(linked))
    assert resumed.state is NotesSyncOperationState.COMPLETED
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "DELETE FROM notes_sync_recovery WHERE operation_id = ?", (linked,)
        )
    undone = await executor.inspect_resolution_undo(
        "root-1", request.operation_id, now=expiry
    )
    assert undone.undo_available is False
    assert undone.undo_reason == "Undone"


@pytest.mark.asyncio
async def test_undo_projection_reports_read_failure_without_false_change_or_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    file = _file(content="file side")
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(file)
    request = replace(
        _request(action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=file),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    expiry = store.load_operation_recovery(request.operation_id).expires_at
    original_observe = notes.observe
    secret = "binding-1 /absolute/private.md content-hash backend-secret"

    async def fail_observe(_note_id: str) -> NotesSyncNoteSnapshot:
        raise RuntimeError(secret)

    monkeypatch.setattr(notes, "observe", fail_observe)
    unavailable = await executor.inspect_resolution_undo(
        "root-1", request.operation_id, now=expiry - 1
    )
    expired = await executor.inspect_resolution_undo(
        "root-1", request.operation_id, now=expiry
    )

    assert unavailable.state == "unavailable"
    assert unavailable.undo_available is False
    assert unavailable.undo_reason == "Unavailable"
    assert secret not in repr(unavailable)
    assert unavailable.note_title is None
    assert unavailable.relative_path is None
    assert expired.undo_reason == "Undo expired"

    monkeypatch.setattr(notes, "observe", original_observe)
    assert (
        await executor.undo_resolution("root-1", request.operation_id)
    ).state is NotesSyncOperationState.COMPLETED
    monkeypatch.setattr(notes, "observe", fail_observe)
    undone = await executor.inspect_resolution_undo(
        "root-1", request.operation_id, now=expiry
    )

    assert undone.state == "undone"
    assert undone.undo_reason == "Undone"
    assert secret not in repr(undone)


@pytest.mark.asyncio
async def test_linked_undo_restart_never_reads_deleted_source_recovery(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    file = _file(content="file side")
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(file)
    request = replace(
        _request(action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=file),
        journal_kind="resolve_keep_file",
    )
    assert (
        await NotesSyncExecutor(
            store, notes, files, recovery_capacity_bytes=65_536
        ).execute(request)
    ).state is NotesSyncOperationState.COMPLETED

    linked = linked_undo_operation_id("root-1", request.operation_id)

    def crash_after_undo_admission(state: NotesSyncOperationState) -> None:
        if (
            state is NotesSyncOperationState.RECOVERY_ADMITTED
            and store.find_operation(linked) is not None
        ):
            raise InjectedCrash

    executor = NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=65_536,
        after_stage=crash_after_undo_admission,
    )
    with pytest.raises(InjectedCrash):
        await executor.undo_resolution("root-1", request.operation_id)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "DELETE FROM notes_sync_recovery WHERE operation_id = ?",
            (request.operation_id,),
        )

    reopened = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=65_536,
    )
    reconstructed = await reopened.reconstruct_request(linked)
    result = await reopened.resume(reconstructed)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.snapshot.content == "note side"


@pytest.mark.parametrize(
    "crash_state",
    (
        NotesSyncOperationState.RECOVERY_ADMITTED,
        NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
        NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
        NotesSyncOperationState.BINDING_UPDATED,
        NotesSyncOperationState.VERIFIED,
    ),
)
@pytest.mark.asyncio
async def test_linked_undo_restarts_after_every_durable_boundary(
    tmp_path: Path,
    crash_state: NotesSyncOperationState,
) -> None:
    store, database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    file = _file(content="file side")
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(file)
    request = replace(
        _request(action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=file),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    linked = linked_undo_operation_id("root-1", request.operation_id)

    def crash(state: NotesSyncOperationState) -> None:
        if state is crash_state:
            raise InjectedCrash

    crashing = NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=65_536,
        after_stage=crash,
    )
    with pytest.raises(InjectedCrash):
        await crashing.undo_resolution("root-1", request.operation_id)
    assert store.list_resolution_history("root-1")[0].undone is False

    reopened_store = NotesDeviceStateStore(database)
    reopened = NotesSyncExecutor(
        reopened_store,
        notes,
        files,
        recovery_capacity_bytes=65_536,
    )
    result = await reopened.resume(await reopened.reconstruct_request(linked))

    assert result.state is NotesSyncOperationState.COMPLETED, result.reason_code
    assert reopened_store.list_resolution_history("root-1")[0].undone is True
    assert notes.snapshot.content == "note side"
    assert files.snapshot.text == "file side"


@pytest.mark.asyncio
async def test_linked_undo_substages_are_length_neutral_one_byte_below_capacity(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    store, _database = _execution_store(calibration)
    note = _note(content="note side", version=4)
    file = _file(content="file side")
    request = replace(
        _request(action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=file),
        journal_kind="resolve_keep_file",
    )
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(file)
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    assert (
        await executor.undo_resolution("root-1", request.operation_id)
    ).state is NotesSyncOperationState.COMPLETED
    with store.transaction() as connection:
        used = int(
            connection.execute(
                "SELECT SUM(length(payload) + length(metadata)) "
                "FROM notes_sync_recovery"
            ).fetchone()[0]
        )

    constrained = tmp_path / "constrained"
    constrained.mkdir()
    store, _database = _execution_store(constrained)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(file)
    lengths: list[int] = []
    linked = linked_undo_operation_id("root-1", request.operation_id)

    def record_length(_state: NotesSyncOperationState) -> None:
        recovery = store.find_operation_recovery(linked)
        if recovery is not None:
            lengths.append(len(recovery.metadata))

    executor = NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=used + 1,
        after_stage=record_length,
    )
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    result = await executor.undo_resolution("root-1", request.operation_id)

    assert result.state is NotesSyncOperationState.COMPLETED, result.reason_code
    assert len(lengths) == 5
    assert len(set(lengths)) == 1
    with store.transaction() as connection:
        final_used = int(
            connection.execute(
                "SELECT SUM(length(payload) + length(metadata)) "
                "FROM notes_sync_recovery"
            ).fetchone()[0]
        )
    assert final_used == used
    assert final_used == (used + 1) - 1


@pytest.mark.parametrize("effect", ("restoration", "binding", "copy_cleanup"))
@pytest.mark.asyncio
async def test_linked_undo_resumes_after_cancellation_following_named_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    effect: str,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    file = _file(content="file side")
    notes: FakeNoteAuthority
    if effect == "copy_cleanup":
        notes = _KeepBothAuthority(note)
        request = _keep_both_request(note, file)
    else:
        notes = FakeNoteAuthority(note)
        request = replace(
            _request(action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=file),
            journal_kind="resolve_keep_file",
        )
    files = _PathPreservingFilesystem(file)
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED

    effect_returned = threading.Event()
    release_effect = threading.Event()
    expected_state: NotesSyncOperationState
    expected_substage: str
    if effect == "restoration":
        original_replace = notes.replace

        async def gated_replace(*args: object, **kwargs: object) -> object:
            result = await original_replace(*args, **kwargs)
            effect_returned.set()
            assert release_effect.wait(5)
            return result

        monkeypatch.setattr(notes, "replace", gated_replace)
        expected_state = NotesSyncOperationState.FIRST_AUTHORITY_APPLIED
        expected_substage = "authority_restored"
    elif effect == "binding":
        original_commit = store.commit_binding_stage

        def gated_commit(*args: object, **kwargs: object) -> object:
            result = original_commit(*args, **kwargs)
            effect_returned.set()
            assert release_effect.wait(5)
            return result

        monkeypatch.setattr(store, "commit_binding_stage", gated_commit)
        expected_state = NotesSyncOperationState.BINDING_UPDATED
        expected_substage = "binding_updated"
    else:
        assert isinstance(notes, _KeepBothAuthority)
        original_delete = notes.delete

        async def gated_delete(*args: object, **kwargs: object) -> None:
            await original_delete(*args, **kwargs)
            effect_returned.set()
            assert release_effect.wait(5)

        monkeypatch.setattr(notes, "delete", gated_delete)
        expected_state = NotesSyncOperationState.BINDING_UPDATED
        expected_substage = "copy_cleanup_complete"

    task = asyncio.create_task(executor.undo_resolution("root-1", request.operation_id))
    try:
        assert await asyncio.to_thread(effect_returned.wait, 3)
        task.cancel()
        release_effect.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release_effect.set()

    linked = linked_undo_operation_id("root-1", request.operation_id)
    assert store.get_operation(linked).state is expected_state
    recovery = store.load_operation_recovery(linked)
    assert json.loads(recovery.metadata)["undo_substage"] == expected_substage
    reopened = NotesSyncExecutor(
        NotesDeviceStateStore(_database),
        notes,
        files,
        recovery_capacity_bytes=65_536,
    )
    result = await reopened.resume(await reopened.reconstruct_request(linked))

    assert result.state is NotesSyncOperationState.COMPLETED, result.reason_code
    assert notes.snapshot.content == "note side"
    assert files.snapshot.text == "file side"


@pytest.mark.asyncio
async def test_older_resolution_cannot_undo_after_newer_resolution_of_same_binding(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    files = _PathPreservingFilesystem(_file(content="file side"))
    notes = FakeNoteAuthority(note)
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    first = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    assert (await executor.execute(first)).state is NotesSyncOperationState.COMPLETED
    files.snapshot = replace(
        _file(content="newer file"),
        reviewed_state=replace(
            _file(content="newer file").reviewed_state,
            relative_path=Path("note.md"),
        ),
    )
    second = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE,
            note=notes.snapshot,
            file=files.snapshot,
            operation_id="operation-2",
        ),
        observation_token="observation-2",
        journal_kind="resolve_keep_file",
    )
    assert (await executor.execute(second)).state is NotesSyncOperationState.COMPLETED
    effects_before = notes.replace_calls

    result = await executor.undo_resolution("root-1", first.operation_id)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "changed_since_resolution"
    assert notes.replace_calls == effects_before
    assert (
        store.find_operation(linked_undo_operation_id("root-1", first.operation_id))
        is None
    )


@pytest.mark.parametrize(
    ("journal_kind", "action"),
    (
        ("resolve_keep_file", NotesSyncActionKind.UPDATE_NOTE),
        ("resolve_keep_note", NotesSyncActionKind.UPDATE_FILE),
    ),
)
@pytest.mark.asyncio
async def test_undo_rejects_same_length_source_recovery_corruption_before_effect(
    tmp_path: Path,
    journal_kind: str,
    action: NotesSyncActionKind,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(action=action, note=note, file=files.snapshot),
        journal_kind=journal_kind,
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    corrupted_payload = b"evil side"
    recovery = store.load_operation_recovery(request.operation_id)
    metadata = json.loads(recovery.metadata)
    metadata["recovery_payload_digest"] = hashlib.sha256(corrupted_payload).hexdigest()
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_recovery SET payload = ?, metadata = ? "
            "WHERE operation_id = ?",
            (
                corrupted_payload,
                json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode(),
                request.operation_id,
            ),
        )
    note_effects = notes.replace_calls
    file_effects = files.replace_calls

    result = await executor.undo_resolution("root-1", request.operation_id)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "changed_since_resolution"
    assert notes.replace_calls == note_effects
    assert files.replace_calls == file_effects


@pytest.mark.asyncio
async def test_linked_undo_rejects_colocated_embedded_source_metadata_corruption(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED

    def fail_after_admission(state: NotesSyncOperationState) -> None:
        if state is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise RuntimeError("transient_test_failure")

    assert (
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=65_536,
            after_stage=fail_after_admission,
        ).undo_resolution("root-1", request.operation_id)
    ).state is NotesSyncOperationState.NEEDS_ATTENTION
    linked = linked_undo_operation_id("root-1", request.operation_id)
    recovery = store.load_operation_recovery(linked)
    metadata = json.loads(recovery.metadata)
    source_metadata = json.loads(
        base64.b64decode(metadata["source_metadata"], validate=True)
    )
    source_metadata["recovery_title"] = "Evil!"
    metadata["source_metadata"] = base64.b64encode(
        json.dumps(source_metadata, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    replacement = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    assert len(replacement) == len(recovery.metadata)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "DELETE FROM notes_sync_recovery WHERE operation_id = ?",
            (request.operation_id,),
        )
        connection.execute(
            "UPDATE notes_sync_recovery SET metadata = ? WHERE operation_id = ?",
            (replacement, linked),
        )
    note_effects = notes.replace_calls
    file_effects = files.replace_calls

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=65_536,
    ).undo_resolution("root-1", request.operation_id)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "changed_since_resolution"
    assert notes.replace_calls == note_effects
    assert files.replace_calls == file_effects
    assert "Evil!" not in notes.snapshot.title


@pytest.mark.asyncio
async def test_incomplete_linked_undo_projection_rejects_embedded_metadata_corruption(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED

    def fail_after_admission(state: NotesSyncOperationState) -> None:
        if state is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise RuntimeError("transient_test_failure")

    assert (
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=65_536,
            after_stage=fail_after_admission,
        ).undo_resolution("root-1", request.operation_id)
    ).state is NotesSyncOperationState.NEEDS_ATTENTION
    linked = linked_undo_operation_id("root-1", request.operation_id)
    recovery = store.load_operation_recovery(linked)
    metadata = json.loads(recovery.metadata)
    source_metadata = json.loads(
        base64.b64decode(metadata["source_metadata"], validate=True)
    )
    source_metadata["recovery_title"] = "Evil!"
    metadata["source_metadata"] = base64.b64encode(
        json.dumps(source_metadata, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    replacement = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    assert len(replacement) == len(recovery.metadata)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_recovery SET metadata = ? WHERE operation_id = ?",
            (replacement, linked),
        )

    projection = await executor.inspect_resolution_undo("root-1", request.operation_id)

    assert projection.undo_available is False
    assert projection.undo_reason == "Unavailable"
    assert projection.state == "unavailable"
    assert projection.note_title is None
    assert projection.relative_path is None
    assert "Evil!" not in repr(projection)


@pytest.mark.parametrize(
    ("control", "expected_reason"),
    (
        ("expired", "Undo expired"),
        ("changed", "Changed since resolution"),
        ("undone", "Undone"),
    ),
)
@pytest.mark.asyncio
async def test_source_projection_labels_require_valid_recovery_envelope(
    tmp_path: Path,
    control: str,
    expected_reason: str,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    expiry = store.load_operation_recovery(request.operation_id).expires_at

    if control == "changed":

        def fail_after_admission(state: NotesSyncOperationState) -> None:
            if state is NotesSyncOperationState.RECOVERY_ADMITTED:
                raise RuntimeError("transient_test_failure")

        assert (
            await NotesSyncExecutor(
                store,
                notes,
                files,
                recovery_capacity_bytes=65_536,
                after_stage=fail_after_admission,
            ).undo_resolution("root-1", request.operation_id, now=expiry - 1)
        ).state is NotesSyncOperationState.NEEDS_ATTENTION
        notes.snapshot = replace(notes.snapshot, title="Edited after resolution")
    elif control == "undone":
        assert (
            await executor.undo_resolution(
                "root-1", request.operation_id, now=expiry - 1
            )
        ).state is NotesSyncOperationState.COMPLETED

    recovery = store.load_operation_recovery(request.operation_id)
    metadata = json.loads(recovery.metadata)
    metadata["recovery_title"] = "Evil!"
    replacement = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    assert len(replacement) == len(recovery.metadata)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_recovery SET metadata = ? WHERE operation_id = ?",
            (replacement, request.operation_id),
        )

    projection = await executor.inspect_resolution_undo(
        "root-1",
        request.operation_id,
        now=expiry if control == "expired" else expiry - 1,
    )

    assert projection.undo_available is False
    assert projection.undo_reason == expected_reason
    assert projection.note_title is None
    assert projection.relative_path is None
    assert (
        NotesSyncRuntimeOwner._resolution_item_label(request.operation_id, projection)
        == request.operation_id[:8]
    )
    assert "Evil!" not in repr(projection)


@pytest.mark.parametrize(
    "field",
    ("conflict_parent_actual_folder_id", "resolution_post_authority_digest"),
)
@pytest.mark.asyncio
async def test_undo_rejects_final_keep_both_envelope_corruption(
    tmp_path: Path,
    field: str,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = _KeepBothAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = _keep_both_request(note, files.snapshot)
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    recovery = store.load_operation_recovery(request.operation_id)
    metadata = json.loads(recovery.metadata)
    original = metadata[field]
    assert isinstance(original, str) and original
    metadata[field] = ("x" if original[0] != "x" else "y") + original[1:]
    replacement = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    assert len(replacement) == len(recovery.metadata)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_recovery SET metadata = ? WHERE operation_id = ?",
            (replacement, request.operation_id),
        )
    note_effects = notes.replace_calls
    file_effects = files.replace_calls

    result = await executor.undo_resolution("root-1", request.operation_id)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "changed_since_resolution"
    assert notes.replace_calls == note_effects
    assert files.replace_calls == file_effects


@pytest.mark.parametrize(
    ("authority", "reason_code"),
    (
        ("note", "note_missing"),
        ("note", "note_identity_changed"),
        ("note", "note_scope_changed"),
        ("file", "missing_target"),
    ),
)
@pytest.mark.asyncio
async def test_undo_maps_typed_missing_or_replaced_authority_to_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
    reason_code: str,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    if authority == "note":

        async def fail_note(_note_id: str) -> NotesSyncNoteSnapshot:
            raise NotesSyncAuthorityError(reason_code)

        monkeypatch.setattr(notes, "observe", fail_note)
    else:

        def fail_file(_relative_path: str) -> NotesSyncFileSnapshot:
            raise NotesSyncFilesystemError(reason_code)

        monkeypatch.setattr(files, "observe", fail_file)

    result = await executor.undo_resolution("root-1", request.operation_id)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "changed_since_resolution"
    assert (
        store.find_operation(linked_undo_operation_id("root-1", request.operation_id))
        is None
    )


@pytest.mark.parametrize(
    ("authority", "error"),
    (
        ("note", NotesSyncAuthorityError("note_observation_failed")),
        ("file", NotesSyncFilesystemError("root_unavailable")),
    ),
)
@pytest.mark.asyncio
async def test_undo_maps_true_observation_failure_to_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
    error: Exception,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    if authority == "note":

        async def fail_note(_note_id: str) -> NotesSyncNoteSnapshot:
            raise error

        monkeypatch.setattr(notes, "observe", fail_note)
    else:

        def fail_file(_relative_path: str) -> NotesSyncFileSnapshot:
            raise error

        monkeypatch.setattr(files, "observe", fail_file)

    result = await executor.undo_resolution("root-1", request.operation_id)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "unavailable"
    assert (
        store.find_operation(linked_undo_operation_id("root-1", request.operation_id))
        is None
    )


@pytest.mark.asyncio
async def test_undo_rejects_same_content_file_edit_revert_before_admission(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    files.snapshot = replace(
        files.snapshot,
        reviewed_state=replace(
            files.snapshot.reviewed_state,
            mtime_ns=files.snapshot.reviewed_state.mtime_ns + 1,
            ctime_ns=files.snapshot.reviewed_state.ctime_ns + 1,
        ),
    )

    result = await executor.undo_resolution("root-1", request.operation_id)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "changed_since_resolution"
    assert notes.replace_calls == 1


@pytest.mark.parametrize("drift", ("file_freshness", "binding"))
@pytest.mark.asyncio
async def test_undo_revalidates_exact_post_authority_before_first_resume_effect(
    tmp_path: Path,
    drift: str,
) -> None:
    store, database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    assert (
        await NotesSyncExecutor(
            store, notes, files, recovery_capacity_bytes=65_536
        ).execute(request)
    ).state is NotesSyncOperationState.COMPLETED

    def crash_after_admission(state: NotesSyncOperationState) -> None:
        if state is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=65_536,
            after_stage=crash_after_admission,
        ).undo_resolution("root-1", request.operation_id)
    if drift == "file_freshness":
        files.snapshot = replace(
            files.snapshot,
            reviewed_state=replace(
                files.snapshot.reviewed_state,
                mtime_ns=files.snapshot.reviewed_state.mtime_ns + 1,
            ),
        )
    else:
        with store.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE notes_sync_bindings SET note_version = note_version + 1 "
                "WHERE binding_id = 'binding-1'"
            )
    effects_before = notes.replace_calls
    reopened = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=65_536,
    )

    result = await reopened.resume(
        await reopened.reconstruct_request(
            linked_undo_operation_id("root-1", request.operation_id)
        )
    )

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "changed_since_resolution"
    assert notes.replace_calls == effects_before


@pytest.mark.asyncio
async def test_undo_rejects_same_content_opposite_authority_replacement_on_restart(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    assert (
        await NotesSyncExecutor(
            store, notes, files, recovery_capacity_bytes=65_536
        ).execute(request)
    ).state is NotesSyncOperationState.COMPLETED

    def crash_after_restore(state: NotesSyncOperationState) -> None:
        if state is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=65_536,
            after_stage=crash_after_restore,
        ).undo_resolution("root-1", request.operation_id)
    files.snapshot = replace(
        files.snapshot,
        observation=replace(
            files.snapshot.observation,
            identity=replace(files.snapshot.observation.identity, inode=999),
        ),
        reviewed_state=replace(
            files.snapshot.reviewed_state,
            identity=replace(files.snapshot.reviewed_state.identity, inode=999),
        ),
    )
    reopened = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=65_536,
    )

    result = await reopened.resume(
        await reopened.reconstruct_request(
            linked_undo_operation_id("root-1", request.operation_id)
        )
    )

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "changed_since_resolution"


@pytest.mark.parametrize(
    "failure_state",
    (
        NotesSyncOperationState.RECOVERY_ADMITTED,
        NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
        NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
        NotesSyncOperationState.BINDING_UPDATED,
        NotesSyncOperationState.VERIFIED,
    ),
)
@pytest.mark.asyncio
async def test_transient_undo_attention_resumes_from_exact_substage(
    tmp_path: Path,
    failure_state: NotesSyncOperationState,
) -> None:
    store, database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    assert (
        await NotesSyncExecutor(
            store, notes, files, recovery_capacity_bytes=65_536
        ).execute(request)
    ).state is NotesSyncOperationState.COMPLETED
    failed = False

    def fail_once(state: NotesSyncOperationState) -> None:
        nonlocal failed
        if state is failure_state and not failed:
            failed = True
            raise RuntimeError("transient_test_failure")

    first = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=65_536,
        after_stage=fail_once,
    ).undo_resolution("root-1", request.operation_id)
    assert first.state is NotesSyncOperationState.NEEDS_ATTENTION
    reopened = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=65_536,
    )
    linked = linked_undo_operation_id("root-1", request.operation_id)

    result = await reopened.resume(await reopened.reconstruct_request(linked))

    assert result.state is NotesSyncOperationState.COMPLETED, result.reason_code


@pytest.mark.asyncio
async def test_completed_duplicate_undo_does_not_require_linked_recovery(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    assert (
        await executor.undo_resolution("root-1", request.operation_id)
    ).state is NotesSyncOperationState.COMPLETED
    linked = linked_undo_operation_id("root-1", request.operation_id)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "DELETE FROM notes_sync_recovery WHERE operation_id = ?", (linked,)
        )

    duplicate = await executor.undo_resolution("root-1", request.operation_id)

    assert duplicate.state is NotesSyncOperationState.COMPLETED


@pytest.mark.asyncio
async def test_undo_admission_rechecks_source_expiry_after_async_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.Notes.notes_device_state_store as store_module

    admission_time = 1_000_000_000
    monkeypatch.setattr(store_module, "_now", lambda: admission_time)
    store, _database = _execution_store(tmp_path)
    note = _note(content="note side", version=4)
    notes = FakeNoteAuthority(note)
    files = _PathPreservingFilesystem(_file(content="file side"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE, note=note, file=files.snapshot
        ),
        journal_kind="resolve_keep_file",
    )
    executor = NotesSyncExecutor(store, notes, files, recovery_capacity_bytes=65_536)
    assert (await executor.execute(request)).state is NotesSyncOperationState.COMPLETED
    expiry = store.load_operation_recovery(request.operation_id).expires_at
    entered = threading.Event()
    release = threading.Event()
    original_observe = notes.observe

    async def blocking_observe(note_id: str) -> NotesSyncNoteSnapshot:
        entered.set()
        assert release.wait(5)
        return await original_observe(note_id)

    monkeypatch.setattr(notes, "observe", blocking_observe)
    clock = {"now": expiry - 1}
    monkeypatch.setattr(executor_module.time, "time_ns", lambda: clock["now"])
    task = asyncio.create_task(executor.undo_resolution("root-1", request.operation_id))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        clock["now"] = expiry
        release.set()
        result = await task
    finally:
        release.set()

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "undo_expired"
    assert (
        store.find_operation(linked_undo_operation_id("root-1", request.operation_id))
        is None
    )


def test_resolution_history_is_newest_first_bounded_and_derives_undone(
    tmp_path: Path,
) -> None:
    store, _database = _execution_store(tmp_path)
    with store.transaction(immediate=True) as connection:
        for index, kind in enumerate(
            (
                "update_note",
                "resolve_keep_file",
                "resolve_keep_note",
                "resolve_keep_both",
            )
        ):
            operation_id = f"history-{index}"
            connection.execute(
                "INSERT INTO notes_sync_operations "
                "(operation_id, root_id, binding_id, kind, state, reason_code, "
                "observation_token, expected_note_version, expected_file_digest, "
                "created_at, updated_at) VALUES (?, 'root-1', 'binding-1', ?, "
                "'completed', NULL, 'token-1', 4, ?, ?, ?)",
                (operation_id, kind, "a" * 64, index + 1, index + 1),
            )
        source_id = "history-1"
        undo_id = linked_undo_operation_id("root-1", source_id)
        connection.execute(
            "INSERT INTO notes_sync_operations "
            "(operation_id, root_id, binding_id, kind, state, reason_code, "
            "observation_token, expected_note_version, expected_file_digest, "
            "created_at, updated_at) VALUES (?, 'root-1', 'binding-1', "
            "'undo_resolution', 'completed', NULL, ?, 4, ?, 8, 8)",
            (undo_id, source_id, "a" * 64),
        )

    rows = store.list_resolution_history("root-1", limit=100, offset=0, now=9)

    assert [row.operation_id for row in rows] == [
        "history-3",
        "history-2",
        "history-1",
    ]
    assert rows[-1].undone is True
    assert all(row.kind.startswith("resolve_keep_") for row in rows)
    with pytest.raises(ValueError, match="limit"):
        store.list_resolution_history("root-1", limit=101)
