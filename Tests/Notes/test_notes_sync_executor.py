from __future__ import annotations

import asyncio
import importlib.util
import hashlib
import json
import os
import sqlite3
import threading
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateError,
    NotesDeviceStateStore,
    NotesSyncOperationRecord,
    NotesSyncBindingRecord,
    NotesSyncRecoveryRecord,
    NotesSyncRootRecord,
)
from tldw_chatbook.Notes.notes_sync_authority import (
    NotesSyncAuthorityError,
    NotesSyncNoteSnapshot,
)
from tldw_chatbook.Notes.notes_sync_filesystem import (
    PosixNotesSyncFilesystem,
    NotesSyncFilesystemError,
    NotesSyncFilesystemPartialError,
    NotesSyncFileSnapshot,
    NotesSyncPrivateCleanupHandle,
    WindowsNotesSyncObservation,
)
from tldw_chatbook.Notes.notes_sync_executor import (
    NotesSyncDirectionOverride,
    NotesSyncExecutionRequest,
    NotesSyncExecutionPartialError,
    NotesSyncExecutionResult,
    NotesSyncExecutor,
    NotesSyncRecoveryChoice,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncFileIdentity,
    NotesSyncFileObservation,
    NotesSyncOperationState,
    NotesSyncRootState,
    NotesSyncSerializationProfile,
)
from tldw_chatbook.Notes.sync_paths import SafeSyncBytes, SafeSyncFileIdentity
from tldw_chatbook import config as config_module


def test_notes_sync_executor_module_is_importable() -> None:
    assert (
        importlib.util.find_spec("tldw_chatbook.Notes.notes_sync_executor") is not None
    )


@pytest.mark.asyncio
async def test_create_note_executes_one_sided_plan_and_activates_binding(
    tmp_path: Path,
) -> None:
    store, _ = _store(tmp_path)
    notes = CreatingNoteAuthority()
    file = _file(content="from-file")
    files = FakeFilesystem(file)
    request = NotesSyncExecutionRequest(
        operation_id="operation-1",
        root_id="root-1",
        logical_folder_id="folder-1",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        binding_id="binding-1",
        observation_token="observation-1",
        action_kind=NotesSyncActionKind.CREATE_NOTE,
        note=None,
        file=file,
        desired_title="Title",
        recovery_id="recovery-operation-1",
        recovery_expires_at=100_000,
        candidate_note_scope_id="local_note",
        candidate_note_id="note-1",
    )

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).execute(request)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.snapshot.content == "from-file"
    assert store.get_binding("binding-1").state is NotesSyncBindingState.ACTIVE


@pytest.mark.asyncio
async def test_create_file_executes_one_sided_plan_and_activates_binding(
    tmp_path: Path,
) -> None:
    store, _ = _store(tmp_path)
    note = _note(content="from-note", version=4)
    notes = FakeNoteAuthority(note)
    files = CreatingFilesystem()
    profile = NotesSyncSerializationProfile(False, "lf", False, 0o600)
    request = NotesSyncExecutionRequest(
        operation_id="operation-1",
        root_id="root-1",
        logical_folder_id="folder-1",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        binding_id="binding-1",
        observation_token="observation-1",
        action_kind=NotesSyncActionKind.CREATE_FILE,
        note=note,
        file=None,
        desired_title="Title",
        recovery_id="recovery-operation-1",
        recovery_expires_at=100_000,
        candidate_relative_path="note.md",
        candidate_serialization=profile,
    )

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).execute(request)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert files.snapshot is not None and files.snapshot.text == "from-note"
    assert store.get_binding("binding-1").state is NotesSyncBindingState.ACTIVE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "observed_profile",
    (
        NotesSyncSerializationProfile(True, "lf", False, 0o600),
        NotesSyncSerializationProfile(False, "crlf", False, 0o600),
        NotesSyncSerializationProfile(False, "lf", True, 0o600),
        NotesSyncSerializationProfile(False, "lf", False, 0o640),
    ),
)
async def test_create_file_rejects_representation_drift_before_membership_binding(
    tmp_path: Path,
    observed_profile: NotesSyncSerializationProfile,
) -> None:
    store, _ = _store(tmp_path)
    note = _note(content="from-note", version=4)
    notes = FakeNoteAuthority(note)
    files = DriftingCreatingFilesystem(observed_profile)
    reviewed_profile = NotesSyncSerializationProfile(False, "lf", False, 0o600)
    request = NotesSyncExecutionRequest(
        operation_id="operation-1",
        root_id="root-1",
        logical_folder_id="folder-1",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        binding_id="binding-1",
        observation_token="observation-1",
        action_kind=NotesSyncActionKind.CREATE_FILE,
        note=note,
        file=None,
        desired_title="Title",
        recovery_id="recovery-operation-1",
        recovery_expires_at=100_000,
        candidate_relative_path="note.md",
        candidate_serialization=reviewed_profile,
    )

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).execute(request)

    assert result.reason_code == "postcondition_failed"
    assert notes.memberships == []
    with pytest.raises(NotesDeviceStateError):
        store.get_binding("binding-1")


@pytest.mark.asyncio
async def test_move_file_executes_guarded_move_and_updates_binding_path(
    tmp_path: Path,
) -> None:
    store, _ = _execution_store(tmp_path)
    note = _note(content="before", version=4)
    notes = FakeNoteAuthority(note)
    source = _file(content="before")
    files = MovingFilesystem(source)
    request = replace(
        _request(action=NotesSyncActionKind.UPDATE_FILE, note=note, file=source),
        action_kind=NotesSyncActionKind.MOVE_FILE,
        move_destination_relative_path="moved.md",
    )

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).execute(request)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert store.get_binding("binding-1").normalized_relative_path == "moved.md"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "observed_profile",
    (
        NotesSyncSerializationProfile(True, "lf", False, 0o600),
        NotesSyncSerializationProfile(False, "crlf", False, 0o600),
        NotesSyncSerializationProfile(False, "lf", True, 0o600),
        NotesSyncSerializationProfile(False, "lf", False, 0o640),
    ),
)
async def test_move_file_rejects_representation_drift_before_membership_binding(
    tmp_path: Path,
    observed_profile: NotesSyncSerializationProfile,
) -> None:
    store, _ = _execution_store(tmp_path)
    note = _note(content="before", version=4)
    notes = FakeNoteAuthority(note)
    source = _file(content="before")
    files = DriftingMovingFilesystem(source, observed_profile)
    request = replace(
        _request(action=NotesSyncActionKind.UPDATE_FILE, note=note, file=source),
        action_kind=NotesSyncActionKind.MOVE_FILE,
        move_destination_relative_path="moved.md",
    )

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).execute(request)

    assert result.reason_code == "postcondition_failed"
    assert notes.memberships == []
    assert store.get_binding("binding-1").normalized_relative_path == "note.md"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (
        NotesSyncActionKind.CREATE_NOTE,
        NotesSyncActionKind.CREATE_FILE,
        NotesSyncActionKind.MOVE_FILE,
    ),
)
async def test_create_and_move_reconstruct_after_first_stage_without_replay(
    tmp_path: Path,
    action: NotesSyncActionKind,
) -> None:
    if action is NotesSyncActionKind.CREATE_NOTE:
        store, database = _store(tmp_path)
        notes: FakeNoteAuthority = CreatingNoteAuthority()
        file = _file(content="authority")
        files: object = FakeFilesystem(file)
        request = NotesSyncExecutionRequest(
            operation_id="operation-1",
            root_id="root-1",
            logical_folder_id="folder-1",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            binding_id="binding-1",
            observation_token="observation-1",
            action_kind=action,
            note=None,
            file=file,
            desired_title="Title",
            recovery_id="recovery-operation-1",
            recovery_expires_at=100_000,
            candidate_note_scope_id="local_note",
            candidate_note_id="note-1",
        )
    elif action is NotesSyncActionKind.CREATE_FILE:
        store, database = _store(tmp_path)
        note = _note(content="authority", version=4)
        notes = FakeNoteAuthority(note)
        files = CreatingFilesystem()
        request = NotesSyncExecutionRequest(
            operation_id="operation-1",
            root_id="root-1",
            logical_folder_id="folder-1",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            binding_id="binding-1",
            observation_token="observation-1",
            action_kind=action,
            note=note,
            file=None,
            desired_title="Title",
            recovery_id="recovery-operation-1",
            recovery_expires_at=100_000,
            candidate_relative_path="note.md",
            candidate_serialization=NotesSyncSerializationProfile(
                False, "lf", False, 0o600
            ),
        )
    else:
        store, database = _execution_store(tmp_path)
        note = _note(content="before", version=4)
        notes = FakeNoteAuthority(note)
        file = _file(content="before")
        files = MovingFilesystem(file)
        request = replace(
            _request(action=NotesSyncActionKind.UPDATE_FILE, note=note, file=file),
            action_kind=action,
            move_destination_relative_path="moved.md",
        )

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)

    executor = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    )
    reconstructed = await executor.reconstruct_request("operation-1")
    result = await executor.resume(reconstructed)

    assert result.state is NotesSyncOperationState.COMPLETED


@pytest.mark.asyncio
async def test_create_note_from_windows_observation_reconstructs_exact_authority(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_roots SET direction = 'folder_to_notes' WHERE root_id = 'root-1'"
        )
    notes = CreatingNoteAuthority()
    file = _windows_file(content="windows authority")
    files = FakeWindowsObservationFilesystem(file)
    request = NotesSyncExecutionRequest(
        operation_id="operation-1",
        root_id="root-1",
        logical_folder_id="folder-1",
        direction=NotesSyncDirection.FOLDER_TO_NOTES,
        binding_id="binding-1",
        observation_token="observation-1",
        action_kind=NotesSyncActionKind.CREATE_NOTE,
        note=None,
        file=file,
        desired_title="Title",
        recovery_id="recovery-operation-1",
        recovery_expires_at=100_000,
        candidate_note_scope_id="local_note",
        candidate_note_id="note-1",
    )

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)

    executor = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    )
    reconstructed = await executor.reconstruct_request("operation-1")
    result = await executor.resume(reconstructed)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.snapshot.content == "windows authority"
    assert reconstructed.file == file


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "direction"),
    (
        (NotesSyncActionKind.CREATE_NOTE, NotesSyncDirection.NOTES_TO_FOLDER),
        (NotesSyncActionKind.CREATE_FILE, NotesSyncDirection.FOLDER_TO_NOTES),
        (NotesSyncActionKind.MOVE_FILE, NotesSyncDirection.NOTES_TO_FOLDER),
    ),
)
async def test_new_action_direction_fence_refuses_before_admission(
    tmp_path: Path,
    action: NotesSyncActionKind,
    direction: NotesSyncDirection,
) -> None:
    store, database, notes, files, request = _new_action_fixture(tmp_path, action)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_roots SET direction = ? WHERE root_id = 'root-1'",
            (direction.value,),
        )

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).execute(replace(request, direction=direction))

    assert result.reason_code == "direction_disallows_action"
    assert _new_action_mutation_count(action, notes, files) == 0
    assert _counts(database) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (
        NotesSyncActionKind.CREATE_NOTE,
        NotesSyncActionKind.CREATE_FILE,
        NotesSyncActionKind.MOVE_FILE,
    ),
)
async def test_new_action_capacity_refusal_precedes_every_mutation(
    tmp_path: Path,
    action: NotesSyncActionKind,
) -> None:
    store, database, notes, files, request = _new_action_fixture(tmp_path, action)

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=1,
    ).execute(request)

    assert result.reason_code == "recovery_capacity_exceeded"
    assert _new_action_mutation_count(action, notes, files) == 0
    assert _counts(database) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (
        NotesSyncActionKind.CREATE_NOTE,
        NotesSyncActionKind.CREATE_FILE,
        NotesSyncActionKind.MOVE_FILE,
    ),
)
async def test_new_action_cancellation_after_first_stage_resumes_without_replay(
    tmp_path: Path,
    action: NotesSyncActionKind,
) -> None:
    store, database, notes, files, request = _new_action_fixture(tmp_path, action)

    def cancel_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=cancel_after_first,
        ).execute(request)

    executor = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    )
    reconstructed = await executor.reconstruct_request(request.operation_id)
    result = await executor.resume(reconstructed)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert _new_action_mutation_count(action, notes, files) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (
        NotesSyncActionKind.CREATE_NOTE,
        NotesSyncActionKind.CREATE_FILE,
        NotesSyncActionKind.MOVE_FILE,
    ),
)
async def test_new_action_root_pause_after_first_stage_blocks_membership_and_binding(
    tmp_path: Path,
    action: NotesSyncActionKind,
) -> None:
    store, database, notes, files, request = _new_action_fixture(tmp_path, action)

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)
    store.transition_root("root-1", NotesSyncRootState.PAUSED)

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).resume(request)

    assert result.reason_code == "binding_authority_changed"
    assert notes.memberships == []
    if action is NotesSyncActionKind.MOVE_FILE:
        assert store.get_binding("binding-1").normalized_relative_path == "note.md"
    else:
        with pytest.raises(NotesDeviceStateError):
            store.get_binding("binding-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (NotesSyncActionKind.CREATE_NOTE, NotesSyncActionKind.CREATE_FILE),
)
async def test_create_candidate_claimed_after_first_stage_fails_before_membership(
    tmp_path: Path,
    action: NotesSyncActionKind,
) -> None:
    store, database, notes, files, request = _new_action_fixture(tmp_path, action)

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-other",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest="b" * 64,
            state=NotesSyncBindingState.ACTIVE,
            serialization=NotesSyncSerializationProfile(False, "lf", False, 0o600),
            content_digest=_digest("authority"),
            note_version=1,
        )
    )

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).resume(request)

    assert result.reason_code == "binding_authority_changed"
    assert notes.memberships == []
    with pytest.raises(NotesDeviceStateError):
        store.get_binding("binding-1")


@pytest.mark.asyncio
async def test_move_binding_drift_after_first_stage_is_never_overwritten(
    tmp_path: Path,
) -> None:
    store, database, notes, files, request = _new_action_fixture(
        tmp_path, NotesSyncActionKind.MOVE_FILE
    )

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_bindings SET content_digest = ? WHERE binding_id = 'binding-1'",
            (_digest("external"),),
        )

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).resume(request)

    assert result.reason_code == "binding_authority_changed"
    assert notes.memberships == []
    assert store.get_binding("binding-1").content_digest == _digest("external")


@pytest.mark.asyncio
async def test_move_reconstruction_rejects_same_version_changed_note_authority(
    tmp_path: Path,
) -> None:
    store, database, notes, files, request = _new_action_fixture(
        tmp_path, NotesSyncActionKind.MOVE_FILE
    )

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_admission,
        ).execute(request)
    notes.snapshot = _note(content="forged", version=4)

    with pytest.raises(RuntimeError, match="recovery_authority_changed"):
        await NotesSyncExecutor(
            NotesDeviceStateStore(database),
            notes,
            files,
            recovery_capacity_bytes=4096,
        ).reconstruct_request(request.operation_id)

    assert files.move_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "direction"),
    (
        (NotesSyncActionKind.CREATE_NOTE, NotesSyncDirection.NOTES_TO_FOLDER),
        (NotesSyncActionKind.CREATE_FILE, NotesSyncDirection.FOLDER_TO_NOTES),
        (NotesSyncActionKind.MOVE_FILE, NotesSyncDirection.NOTES_TO_FOLDER),
    ),
)
@pytest.mark.parametrize("resolution", ("resume", "restore", "disconnect"))
async def test_new_action_reconstructs_reviewed_direction_override_for_every_choice(
    tmp_path: Path,
    action: NotesSyncActionKind,
    direction: NotesSyncDirection,
    resolution: str,
) -> None:
    store, database, notes, files, request = _new_action_fixture(tmp_path, action)
    if action is NotesSyncActionKind.MOVE_FILE:
        assert request.file is not None
        files = RestorableMovingFilesystem(request.file)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_roots SET direction = ? WHERE root_id = 'root-1'",
            (direction.value,),
        )
    override = NotesSyncDirectionOverride(
        review_id="review-1",
        action_kind=action,
        observation_token=request.observation_token,
    )
    request = replace(request, direction=direction, direction_override=override)

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)
    if resolution != "resume":
        store.mark_operation_attention(request.operation_id, f"{resolution}_requested")
    executor = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    )
    reconstructed = await executor.reconstruct_request(request.operation_id)

    result = await getattr(executor, resolution)(reconstructed)

    assert reconstructed.direction_override == override
    assert result.state is NotesSyncOperationState.COMPLETED


@pytest.mark.asyncio
async def test_new_action_reconstruction_rejects_override_token_corruption(
    tmp_path: Path,
) -> None:
    store, database, notes, files, request = _new_action_fixture(
        tmp_path, NotesSyncActionKind.CREATE_NOTE
    )
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_roots SET direction = 'notes_to_folder' WHERE root_id = 'root-1'"
        )
    request = replace(
        request,
        direction=NotesSyncDirection.NOTES_TO_FOLDER,
        direction_override=NotesSyncDirectionOverride(
            review_id="review-1",
            action_kind=request.action_kind,
            observation_token=request.observation_token,
        ),
    )

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)
    with sqlite3.connect(database) as connection:
        raw = connection.execute(
            "SELECT metadata FROM notes_sync_recovery WHERE operation_id = ?",
            (request.operation_id,),
        ).fetchone()[0]
        metadata = json.loads(raw)
        metadata["direction_override"]["observation_token"] = "forged-token"
        connection.execute(
            "UPDATE notes_sync_recovery SET metadata = ? WHERE operation_id = ?",
            (
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(),
                request.operation_id,
            ),
        )

    with pytest.raises(RuntimeError, match="recovery_authority_changed"):
        await NotesSyncExecutor(
            NotesDeviceStateStore(database),
            notes,
            files,
            recovery_capacity_bytes=4096,
        ).reconstruct_request(request.operation_id)


def test_recovery_capacity_has_one_bounded_config_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = tomllib.loads(config_module.CONFIG_TOML_CONTENT)
    monkeypatch.delenv("TLDW_NOTES_SYNC_RECOVERY_CAPACITY_BYTES", raising=False)

    assert defaults["notes"]["recovery_capacity_bytes"] == 256 * 1024 * 1024
    assert (
        config_module.get_notes_sync_recovery_capacity_bytes(defaults)
        == 256 * 1024 * 1024
    )
    assert (
        config_module.get_notes_sync_recovery_capacity_bytes({"notes": {}})
        == 256 * 1024 * 1024
    )
    assert config_module.get_notes_sync_recovery_capacity_bytes({}) == 256 * 1024 * 1024
    assert (
        config_module.get_notes_sync_recovery_capacity_bytes({"notes": "invalid"})
        == 256 * 1024 * 1024
    )
    monkeypatch.setenv("TLDW_NOTES_SYNC_RECOVERY_CAPACITY_BYTES", "8192")
    assert (
        config_module.get_notes_sync_recovery_capacity_bytes(
            {"notes": {"recovery_capacity_bytes": 4096}}
        )
        == 8192
    )
    monkeypatch.delenv("TLDW_NOTES_SYNC_RECOVERY_CAPACITY_BYTES")
    with pytest.raises(ValueError, match="recovery_capacity_bytes"):
        config_module.get_notes_sync_recovery_capacity_bytes(
            {"notes": {"recovery_capacity_bytes": 0}}
        )


def _store(tmp_path: Path) -> tuple[NotesDeviceStateStore, Path]:
    database = tmp_path / "notes-sync.sqlite3"
    store = NotesDeviceStateStore(database)
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-1",
            note_scope_id="local_note",
            logical_folder_id="folder-1",
            canonical_path="/private/root",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.ACTIVE,
        )
    )
    return store, database


def _operation(operation_id: str = "operation-1") -> NotesSyncOperationRecord:
    return NotesSyncOperationRecord(
        operation_id=operation_id,
        root_id="root-1",
        binding_id=None,
        kind="update_note",
        state=NotesSyncOperationState.PENDING,
        reason_code=None,
        observation_token="observation-1",
        expected_note_version=4,
        expected_file_digest="a" * 64,
    )


def _recovery(
    operation_id: str = "operation-1",
    *,
    payload: bytes = b"before",
    metadata: bytes = b"metadata",
    expires_at: int = 10_000,
) -> NotesSyncRecoveryRecord:
    return NotesSyncRecoveryRecord(
        recovery_id=f"recovery-{operation_id}",
        operation_id=operation_id,
        payload=payload,
        metadata=metadata,
        expires_at=expires_at,
    )


def _counts(database: Path) -> tuple[int, int]:
    with sqlite3.connect(database) as connection:
        return (
            connection.execute("SELECT COUNT(*) FROM notes_sync_operations").fetchone()[
                0
            ],
            connection.execute("SELECT COUNT(*) FROM notes_sync_recovery").fetchone()[
                0
            ],
        )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _note(*, content: str, version: int) -> NotesSyncNoteSnapshot:
    return NotesSyncNoteSnapshot(
        note_scope_id="local_note",
        note_id="note-1",
        title="Title",
        content=content,
        version=version,
        content_digest=_digest(content),
    )


def _file(*, content: str, inode: int = 11) -> NotesSyncFileSnapshot:
    raw = content.encode("utf-8")
    profile = NotesSyncSerializationProfile(
        utf8_bom=False,
        newline="lf",
        final_newline=content.endswith("\n"),
        mode=0o600,
    )
    identity = NotesSyncFileIdentity(device=7, inode=inode, link_count=1)
    reviewed = SafeSyncBytes(
        relative_path=Path("note.md"),
        content=raw,
        identity=SafeSyncFileIdentity(device=7, inode=inode, link_count=1),
        mode=0o600,
        size=len(raw),
        mtime_ns=20,
        ctime_ns=20,
        owner_user=os.geteuid(),
        owner_group=os.getegid(),
        flags=0,
        extended_attributes=(),
        has_extended_acl=False,
    )
    return NotesSyncFileSnapshot(
        observation=NotesSyncFileObservation(
            relative_path="note.md",
            identity=identity,
            content_digest=_digest(content),
            size_bytes=len(raw),
            serialization=profile,
        ),
        text=content,
        raw_bytes=raw,
        reviewed_state=reviewed,
        representation_digest=hashlib.sha256(raw).hexdigest(),
    )


def _file_at(
    relative_path: str,
    *,
    content: str,
    inode: int = 11,
) -> NotesSyncFileSnapshot:
    snapshot = _file(content=content, inode=inode)
    return replace(
        snapshot,
        observation=replace(snapshot.observation, relative_path=relative_path),
        reviewed_state=replace(
            snapshot.reviewed_state,
            relative_path=Path(relative_path),
        ),
    )


def _execution_store(tmp_path: Path) -> tuple[NotesDeviceStateStore, Path]:
    store, database = _store(tmp_path)
    file_before = _file(content="before")
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-1",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest=NotesSyncExecutor.stable_identity_digest(
                file_before
            ),
            state=NotesSyncBindingState.ACTIVE,
            serialization=file_before.observation.serialization,
            content_digest=_digest("before"),
            note_version=4,
        )
    )
    return store, database


class FakeNoteAuthority:
    def __init__(
        self,
        snapshot: NotesSyncNoteSnapshot,
        *,
        on_replace: object | None = None,
        cancel_after_replace: bool = False,
        replace_error: Exception | None = None,
        cancel_after_membership: bool = False,
    ) -> None:
        self.snapshot = snapshot
        self.replace_calls = 0
        self.memberships: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        self._on_replace = on_replace
        self._cancel_after_replace = cancel_after_replace
        self._replace_error = replace_error
        self.cancel_after_membership = cancel_after_membership

    async def observe(self, note_id: str) -> NotesSyncNoteSnapshot:
        assert note_id == self.snapshot.note_id
        return self.snapshot

    async def replace(
        self,
        expected: NotesSyncNoteSnapshot,
        *,
        title: str,
        content: str,
    ) -> NotesSyncNoteSnapshot:
        assert expected == self.snapshot
        self.replace_calls += 1
        if callable(self._on_replace):
            self._on_replace()
        if self._replace_error is not None:
            raise self._replace_error
        self.snapshot = NotesSyncNoteSnapshot(
            note_scope_id=expected.note_scope_id,
            note_id=expected.note_id,
            title=title,
            content=content,
            version=expected.version + 1,
            content_digest=_digest(content),
        )
        if self._cancel_after_replace:
            raise __import__("asyncio").CancelledError
        return self.snapshot

    async def reconcile_managed_memberships(
        self,
        *,
        owner_id: str,
        desired: tuple[tuple[str, str], ...],
    ) -> None:
        placement = (owner_id, desired)
        if placement not in self.memberships:
            self.memberships.append(placement)
        if self.cancel_after_membership:
            raise __import__("asyncio").CancelledError


class FakeFilesystem:
    def __init__(self, snapshot: NotesSyncFileSnapshot) -> None:
        self.snapshot = snapshot
        self.replace_calls = 0

    def observe(self, relative_path: str) -> NotesSyncFileSnapshot:
        assert relative_path == self.snapshot.observation.relative_path
        return self.snapshot

    def replace(
        self,
        relative_path: str,
        text: str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot:
        assert expected == self.snapshot
        self.replace_calls += 1
        self.snapshot = _file(
            content=text, inode=expected.observation.identity.inode + 1
        )
        return self.snapshot


class PartialFilesystem(FakeFilesystem):
    def __init__(self, snapshot: NotesSyncFileSnapshot) -> None:
        super().__init__(snapshot)
        self.cleanup_calls: list[NotesSyncPrivateCleanupHandle] = []

    def replace(
        self,
        relative_path: str,
        text: str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot:
        self.snapshot = _file(
            content=text, inode=expected.observation.identity.inode + 1
        )
        self.replace_calls += 1
        raise NotesSyncFilesystemPartialError(
            "replacement_cleanup_pending",
            NotesSyncPrivateCleanupHandle(
                ".notes-sync-private-recovery",
                "replacement_cleanup_pending",
                SafeSyncFileIdentity(device=7, inode=11, link_count=1),
            ),
        )

    def resolve_cleanup(self, handle: NotesSyncPrivateCleanupHandle) -> None:
        self.cleanup_calls.append(handle)


class NullCleanupPartialFilesystem(PartialFilesystem):
    def replace(
        self,
        relative_path: str,
        text: str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot:
        self.snapshot = _file(
            content=text, inode=expected.observation.identity.inode + 1
        )
        self.replace_calls += 1
        raise NotesSyncFilesystemPartialError(
            "replacement_commit_unverified",
            NotesSyncPrivateCleanupHandle(None),
        )


class DuplicateBlockingFilesystem(FakeFilesystem):
    def __init__(self, snapshot: NotesSyncFileSnapshot) -> None:
        super().__init__(snapshot)
        self.started = threading.Event()
        self.release = threading.Event()

    def replace(
        self,
        relative_path: str,
        text: str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot:
        self.replace_calls += 1
        self.started.set()
        assert self.release.wait(3.0)
        self.snapshot = _file(
            content=text, inode=expected.observation.identity.inode + 1
        )
        return self.snapshot


class BlockingFilesystem(FakeFilesystem):
    def __init__(self, snapshot: NotesSyncFileSnapshot) -> None:
        super().__init__(snapshot)
        self.started = threading.Event()
        self.release = threading.Event()

    def replace(
        self,
        relative_path: str,
        text: str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot:
        self.started.set()
        assert self.release.wait(3.0)
        return super().replace(relative_path, text, expected=expected)


class FakeWindowsObservationFilesystem:
    def __init__(self, observation: WindowsNotesSyncObservation) -> None:
        self.observation = observation
        self.observe_calls = 0

    def observe(self) -> tuple[WindowsNotesSyncObservation, ...]:
        self.observe_calls += 1
        return (self.observation,)

    def replace(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("Windows observation authority must never write files")


class CreatingNoteAuthority(FakeNoteAuthority):
    def __init__(self) -> None:
        super().__init__(_note(content="placeholder", version=0))
        self.created = False
        self.create_calls = 0
        self.delete_calls = 0

    async def observe(self, note_id: str) -> NotesSyncNoteSnapshot:
        if not self.created:
            raise NotesSyncAuthorityError("note_missing")
        return await super().observe(note_id)

    async def create(
        self,
        *,
        note_id: str,
        title: str,
        content: str,
    ) -> NotesSyncNoteSnapshot:
        self.create_calls += 1
        self.snapshot = NotesSyncNoteSnapshot(
            note_scope_id="local_note",
            note_id=note_id,
            title=title,
            content=content,
            version=1,
            content_digest=_digest(content),
        )
        self.created = True
        return self.snapshot

    async def delete(self, expected: NotesSyncNoteSnapshot) -> None:
        assert expected == self.snapshot
        self.delete_calls += 1
        self.created = False


class CreatingFilesystem:
    def __init__(self) -> None:
        self.snapshot: NotesSyncFileSnapshot | None = None
        self.create_calls = 0
        self.delete_calls = 0

    def observe(self, relative_path: str) -> NotesSyncFileSnapshot:
        if self.snapshot is None:
            raise NotesSyncFilesystemError("missing_target")
        return self.snapshot

    def create(
        self,
        relative_path: str,
        text: str,
        *,
        profile: NotesSyncSerializationProfile,
    ) -> NotesSyncFileSnapshot:
        self.create_calls += 1
        self.snapshot = replace(
            _file_at(relative_path, content=text),
            observation=replace(
                _file_at(relative_path, content=text).observation,
                serialization=profile,
            ),
        )
        return self.snapshot

    def delete(self, *, expected: NotesSyncFileSnapshot) -> None:
        assert expected == self.snapshot
        self.delete_calls += 1
        self.snapshot = None


class DriftingCreatingFilesystem(CreatingFilesystem):
    def __init__(self, profile: NotesSyncSerializationProfile) -> None:
        super().__init__()
        self._profile = profile

    def create(
        self,
        relative_path: str,
        text: str,
        *,
        profile: NotesSyncSerializationProfile,
    ) -> NotesSyncFileSnapshot:
        del profile
        payload = PosixNotesSyncFilesystem.serialize(text, self._profile)
        original = _file_at(relative_path, content=text)
        self.create_calls += 1
        self.snapshot = replace(
            original,
            observation=replace(
                original.observation,
                serialization=self._profile,
                size_bytes=len(payload),
            ),
            raw_bytes=payload,
            reviewed_state=replace(
                original.reviewed_state,
                content=payload,
                mode=self._profile.mode,
                size=len(payload),
            ),
            representation_digest=hashlib.sha256(payload).hexdigest(),
        )
        return self.snapshot


class MovingFilesystem(FakeFilesystem):
    def __init__(self, snapshot: NotesSyncFileSnapshot) -> None:
        super().__init__(snapshot)
        self.destination: NotesSyncFileSnapshot | None = None
        self.move_calls = 0

    def observe(self, relative_path: str) -> NotesSyncFileSnapshot:
        if relative_path == self.snapshot.observation.relative_path:
            if self.destination is not None:
                raise NotesSyncFilesystemError("missing_target")
            return self.snapshot
        if self.destination is None:
            raise NotesSyncFilesystemError("missing_target")
        return self.destination

    def move(
        self,
        destination_path: str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot:
        self.move_calls += 1
        self.destination = replace(
            expected,
            observation=replace(
                expected.observation,
                relative_path=destination_path,
            ),
            reviewed_state=replace(
                expected.reviewed_state,
                relative_path=Path(destination_path),
            ),
        )
        return self.destination


class DriftingMovingFilesystem(MovingFilesystem):
    def __init__(
        self,
        snapshot: NotesSyncFileSnapshot,
        profile: NotesSyncSerializationProfile,
    ) -> None:
        super().__init__(snapshot)
        self._profile = profile

    def move(
        self,
        destination_path: str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot:
        moved = super().move(destination_path, expected=expected)
        self.destination = replace(
            moved,
            observation=replace(moved.observation, serialization=self._profile),
            reviewed_state=replace(moved.reviewed_state, mode=self._profile.mode),
        )
        return self.destination


class RestorableMovingFilesystem:
    def __init__(self, snapshot: NotesSyncFileSnapshot) -> None:
        self.current = snapshot
        self.move_calls = 0

    def observe(self, relative_path: str) -> NotesSyncFileSnapshot:
        if relative_path != self.current.observation.relative_path:
            raise NotesSyncFilesystemError("missing_target")
        return self.current

    def move(
        self,
        destination_path: str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot:
        if expected != self.current:
            raise NotesSyncFilesystemError("target_identity_changed")
        self.move_calls += 1
        self.current = replace(
            expected,
            observation=replace(
                expected.observation,
                relative_path=destination_path,
            ),
            reviewed_state=replace(
                expected.reviewed_state,
                relative_path=Path(destination_path),
            ),
        )
        return self.current


def _new_action_fixture(
    tmp_path: Path,
    action: NotesSyncActionKind,
) -> tuple[
    NotesDeviceStateStore,
    Path,
    FakeNoteAuthority,
    object,
    NotesSyncExecutionRequest,
]:
    if action is NotesSyncActionKind.CREATE_NOTE:
        store, database = _store(tmp_path)
        notes = CreatingNoteAuthority()
        file = _file(content="authority")
        files: object = FakeFilesystem(file)
        request = NotesSyncExecutionRequest(
            operation_id="operation-1",
            root_id="root-1",
            logical_folder_id="folder-1",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            binding_id="binding-1",
            observation_token="observation-1",
            action_kind=action,
            note=None,
            file=file,
            desired_title="Title",
            recovery_id="recovery-operation-1",
            recovery_expires_at=100_000,
            candidate_note_scope_id="local_note",
            candidate_note_id="note-1",
        )
        return store, database, notes, files, request
    if action is NotesSyncActionKind.CREATE_FILE:
        store, database = _store(tmp_path)
        note = _note(content="authority", version=4)
        notes = FakeNoteAuthority(note)
        files = CreatingFilesystem()
        request = NotesSyncExecutionRequest(
            operation_id="operation-1",
            root_id="root-1",
            logical_folder_id="folder-1",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            binding_id="binding-1",
            observation_token="observation-1",
            action_kind=action,
            note=note,
            file=None,
            desired_title="Title",
            recovery_id="recovery-operation-1",
            recovery_expires_at=100_000,
            candidate_relative_path="note.md",
            candidate_serialization=NotesSyncSerializationProfile(
                False, "lf", False, 0o600
            ),
        )
        return store, database, notes, files, request
    store, database = _execution_store(tmp_path)
    note = _note(content="before", version=4)
    notes = FakeNoteAuthority(note)
    file = _file(content="before")
    files = MovingFilesystem(file)
    request = replace(
        _request(action=NotesSyncActionKind.UPDATE_FILE, note=note, file=file),
        action_kind=action,
        move_destination_relative_path="moved.md",
    )
    return store, database, notes, files, request


def _new_action_mutation_count(
    action: NotesSyncActionKind,
    notes: FakeNoteAuthority,
    files: object,
) -> int:
    if action is NotesSyncActionKind.CREATE_NOTE:
        return notes.create_calls  # type: ignore[attr-defined]
    if action is NotesSyncActionKind.CREATE_FILE:
        return files.create_calls  # type: ignore[attr-defined]
    return files.move_calls  # type: ignore[attr-defined]


def _request(
    *,
    action: NotesSyncActionKind,
    note: NotesSyncNoteSnapshot,
    file: NotesSyncFileSnapshot,
    operation_id: str = "operation-1",
) -> NotesSyncExecutionRequest:
    return NotesSyncExecutionRequest(
        operation_id=operation_id,
        root_id="root-1",
        logical_folder_id="folder-1",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        binding_id="binding-1",
        observation_token="observation-1",
        action_kind=action,
        note=note,
        file=file,
        desired_title="Title",
        recovery_id=f"recovery-{operation_id}",
        recovery_expires_at=100_000,
    )


def _windows_file(*, content: str) -> WindowsNotesSyncObservation:
    raw = content.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return WindowsNotesSyncObservation(
        relative_path="note.md",
        text=content,
        content_digest=_digest(content),
        representation_digest=digest,
        stable_identity_digest="d" * 64,
        freshness_digest="e" * 64,
        size_bytes=len(raw),
        serialization=NotesSyncSerializationProfile(False, "lf", False, 0o600),
    )


def test_capacity_and_recovery_are_admitted_atomically_before_mutation(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    operation = _operation()
    recovery = _recovery()

    decision = store.admit_operation_recovery(
        operation,
        recovery,
        capacity_bytes=len(recovery.payload) + len(recovery.metadata),
    )

    assert decision.admitted is True
    assert decision.required_bytes == 14
    assert decision.available_bytes == 14
    assert store.get_operation("operation-1").state is (
        NotesSyncOperationState.RECOVERY_ADMITTED
    )
    assert store.load_recovery("recovery-operation-1") == recovery
    assert _counts(database) == (1, 1)


def test_capacity_failure_persists_neither_intent_nor_recovery(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)

    decision = store.admit_operation_recovery(
        _operation(),
        _recovery(),
        capacity_bytes=13,
    )

    assert decision.admitted is False
    assert decision.reason_code == "recovery_capacity_exceeded"
    assert decision.required_bytes == 14
    assert decision.available_bytes == 13
    assert _counts(database) == (0, 0)


def test_exact_recovery_admission_replay_is_idempotent_and_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    operation = _operation()
    recovery = _recovery()

    first = store.admit_operation_recovery(
        operation,
        recovery,
        capacity_bytes=28,
    )
    second = store.admit_operation_recovery(
        operation,
        recovery,
        capacity_bytes=28,
    )

    assert first.admitted is second.admitted is True
    assert _counts(database) == (1, 1)
    with pytest.raises(NotesDeviceStateError, match="conflicts"):
        store.admit_operation_recovery(
            operation,
            _recovery(payload=b"different"),
            capacity_bytes=28,
        )
    assert store.load_recovery("recovery-operation-1") == recovery


def test_pending_and_attention_recovery_are_never_evicted_even_when_expired(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    first = _operation("operation-1")
    first_recovery = _recovery("operation-1", expires_at=1)
    assert store.admit_operation_recovery(
        first,
        first_recovery,
        capacity_bytes=28,
    ).admitted
    store.transition_operation(
        "operation-1",
        NotesSyncOperationState.NEEDS_ATTENTION,
    )

    decision = store.admit_operation_recovery(
        _operation("operation-2"),
        _recovery("operation-2", expires_at=20_000),
        capacity_bytes=14,
        now=10_000,
    )

    assert decision.admitted is False
    assert store.load_recovery("recovery-operation-1") == first_recovery
    assert _counts(database) == (1, 1)


def test_only_completed_expired_recovery_is_reclaimed_for_admission(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    first_recovery = _recovery("operation-1", expires_at=1)
    assert store.admit_operation_recovery(
        _operation("operation-1"),
        first_recovery,
        capacity_bytes=14,
    ).admitted
    for state in (
        NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
        NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
        NotesSyncOperationState.BINDING_UPDATED,
        NotesSyncOperationState.VERIFIED,
        NotesSyncOperationState.COMPLETED,
    ):
        store.transition_operation("operation-1", state)

    decision = store.admit_operation_recovery(
        _operation("operation-2"),
        _recovery("operation-2", expires_at=20_000),
        capacity_bytes=14,
        now=10_000,
    )

    assert decision.admitted is True
    with pytest.raises(NotesDeviceStateError, match="does not exist"):
        store.load_recovery("recovery-operation-1")
    assert _counts(database) == (2, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "note_content", "note_version", "file_content"),
    (
        (NotesSyncActionKind.UPDATE_NOTE, "before", 4, "after"),
        (NotesSyncActionKind.UPDATE_FILE, "after", 5, "before"),
    ),
)
async def test_executor_advances_every_durable_stage_and_completes_last(
    tmp_path: Path,
    action: NotesSyncActionKind,
    note_content: str,
    note_version: int,
    file_content: str,
) -> None:
    store, _ = _execution_store(tmp_path)
    note_authority = FakeNoteAuthority(
        _note(content=note_content, version=note_version)
    )
    filesystem = FakeFilesystem(_file(content=file_content))
    stages: list[NotesSyncOperationState] = []
    executor = NotesSyncExecutor(
        store,
        note_authority,
        filesystem,
        recovery_capacity_bytes=2048,
        after_stage=stages.append,
    )

    result = await executor.execute(
        _request(
            action=action,
            note=note_authority.snapshot,
            file=filesystem.snapshot,
        )
    )

    assert result == NotesSyncExecutionResult(
        operation_id="operation-1",
        state=NotesSyncOperationState.COMPLETED,
        recovery_required=False,
    )
    assert stages == [
        NotesSyncOperationState.RECOVERY_ADMITTED,
        NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
        NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
        NotesSyncOperationState.BINDING_UPDATED,
        NotesSyncOperationState.VERIFIED,
        NotesSyncOperationState.COMPLETED,
    ]
    assert note_authority.snapshot.content == "after"
    assert filesystem.snapshot.text == "after"
    assert note_authority.replace_calls == (action is NotesSyncActionKind.UPDATE_NOTE)
    assert filesystem.replace_calls == (action is NotesSyncActionKind.UPDATE_FILE)
    assert note_authority.memberships == [("root-1", (("folder-1", "note-1"),))]
    binding = store.get_binding("binding-1")
    assert binding.content_digest == _digest("after")
    assert binding.note_version == note_authority.snapshot.version
    assert store.get_operation("operation-1").state is (
        NotesSyncOperationState.COMPLETED
    )


@pytest.mark.asyncio
async def test_windows_observation_can_update_note_but_never_write_file(
    tmp_path: Path,
) -> None:
    store, _ = _store(tmp_path)
    windows_file = _windows_file(content="after")
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-1",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest=windows_file.stable_identity_digest,
            state=NotesSyncBindingState.ACTIVE,
            serialization=windows_file.serialization,
            content_digest=_digest("before"),
            note_version=4,
        )
    )
    notes = FakeNoteAuthority(_note(content="before", version=4))
    filesystem = FakeWindowsObservationFilesystem(windows_file)
    request = NotesSyncExecutionRequest(
        operation_id="operation-1",
        root_id="root-1",
        logical_folder_id="folder-1",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        binding_id="binding-1",
        observation_token="observation-1",
        action_kind=NotesSyncActionKind.UPDATE_NOTE,
        note=notes.snapshot,
        file=windows_file,
        desired_title="Title",
        recovery_id="recovery-operation-1",
        recovery_expires_at=100_000,
    )

    result = await NotesSyncExecutor(
        store,
        notes,
        filesystem,
        recovery_capacity_bytes=2048,
    ).execute(request)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.snapshot.content == "after"
    assert filesystem.observe_calls >= 1
    assert store.get_binding("binding-1").stable_identity_digest == "d" * 64


def test_capacity_is_durable_before_the_first_authority_mutation(
    tmp_path: Path,
) -> None:
    store, _ = _execution_store(tmp_path)

    def assert_admitted() -> None:
        assert store.get_operation("operation-1").state is (
            NotesSyncOperationState.RECOVERY_ADMITTED
        )
        assert store.load_recovery("recovery-operation-1").payload == b"before"

    note_authority = FakeNoteAuthority(
        _note(content="before", version=4), on_replace=assert_admitted
    )
    executor = NotesSyncExecutor(
        store,
        note_authority,
        FakeFilesystem(_file(content="after")),
        recovery_capacity_bytes=2048,
    )

    __import__("asyncio").run(
        executor.execute(
            _request(
                action=NotesSyncActionKind.UPDATE_NOTE,
                note=note_authority.snapshot,
                file=_file(content="after"),
            )
        )
    )


@pytest.mark.asyncio
async def test_conflict_capacity_refusal_has_no_recovery_or_recovery_actions(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    note_authority = FakeNoteAuthority(_note(content="before", version=4))
    filesystem = FakeFilesystem(_file(content="after"))
    executor = NotesSyncExecutor(
        store,
        note_authority,
        filesystem,
        recovery_capacity_bytes=1,
    )

    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE,
            note=note_authority.snapshot,
            file=filesystem.snapshot,
        ),
        journal_kind="resolve_keep_file",
    )

    result = await executor.execute(request)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "recovery_capacity_exceeded"
    assert result.recovery_required is False
    assert result.choices == ()
    assert note_authority.replace_calls == 0
    assert filesystem.replace_calls == 0
    assert store.find_operation(request.operation_id) is None
    assert store.find_operation_recovery(request.operation_id) is None
    assert _counts(database) == (0, 0)


@pytest.mark.asyncio
async def test_journal_observation_failure_cannot_be_misread_as_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="before", version=4))
    files = FakeFilesystem(_file(content="after"))

    def fail_observation(operation_id: str) -> object:
        raise NotesDeviceStateError("private database path")

    monkeypatch.setattr(store, "find_operation", fail_observation, raising=False)
    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).execute(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE,
            note=notes.snapshot,
            file=files.snapshot,
        )
    )

    assert result.reason_code == "executor_failed"
    assert notes.replace_calls == files.replace_calls == 0
    assert _counts(database) == (0, 0)


class InjectedCrash(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crash_stage",
    (
        NotesSyncOperationState.RECOVERY_ADMITTED,
        NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
        NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
        NotesSyncOperationState.BINDING_UPDATED,
        NotesSyncOperationState.VERIFIED,
    ),
)
async def test_reopen_resumes_each_interrupted_stage_without_blind_replay(
    tmp_path: Path,
    crash_stage: NotesSyncOperationState,
) -> None:
    store, database = _execution_store(tmp_path)
    note_authority = FakeNoteAuthority(_note(content="before", version=4))
    filesystem = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=note_authority.snapshot,
        file=filesystem.snapshot,
    )

    def crash_after(stage: NotesSyncOperationState) -> None:
        if stage is crash_stage:
            raise InjectedCrash

    first_executor = NotesSyncExecutor(
        store,
        note_authority,
        filesystem,
        recovery_capacity_bytes=1024,
        after_stage=crash_after,
    )
    with pytest.raises(InjectedCrash):
        await first_executor.execute(request)
    assert store.get_operation("operation-1").state is crash_stage

    reopened = NotesDeviceStateStore(database)
    result = await NotesSyncExecutor(
        reopened,
        note_authority,
        filesystem,
        recovery_capacity_bytes=1024,
    ).resume(request)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert note_authority.replace_calls == 1


@pytest.mark.asyncio
async def test_resume_with_changed_observation_becomes_explicit_attention(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    note_authority = FakeNoteAuthority(_note(content="before", version=4))
    filesystem = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=note_authority.snapshot,
        file=filesystem.snapshot,
    )

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            note_authority,
            filesystem,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_admission,
        ).execute(request)
    filesystem.snapshot = _file(content="external-change")

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        note_authority,
        filesystem,
        recovery_capacity_bytes=1024,
    ).resume(request)

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "stale_observation"
    assert result.recovery_required is True
    assert result.choices == tuple(NotesSyncRecoveryChoice)
    assert note_authority.replace_calls == 0


@pytest.mark.asyncio
async def test_cancellation_after_uncertain_mutation_is_durable_attention(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    note_authority = FakeNoteAuthority(
        _note(content="before", version=4), cancel_after_replace=True
    )
    filesystem = FakeFilesystem(_file(content="after"))
    executor = NotesSyncExecutor(
        store,
        note_authority,
        filesystem,
        recovery_capacity_bytes=1024,
    )

    with pytest.raises(__import__("asyncio").CancelledError):
        await executor.execute(
            _request(
                action=NotesSyncActionKind.UPDATE_NOTE,
                note=_note(content="before", version=4),
                file=filesystem.snapshot,
            )
        )

    reopened = NotesDeviceStateStore(database)
    assert reopened.get_operation("operation-1").state is (
        NotesSyncOperationState.NEEDS_ATTENTION
    )
    assert reopened.load_recovery("recovery-operation-1").payload == b"before"
    assert note_authority.snapshot.content == "after"


@pytest.mark.asyncio
async def test_stale_resume_token_fences_the_durable_operation(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    note_authority = FakeNoteAuthority(_note(content="before", version=4))
    filesystem = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=note_authority.snapshot,
        file=filesystem.snapshot,
    )

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            note_authority,
            filesystem,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_admission,
        ).execute(request)

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        note_authority,
        filesystem,
        recovery_capacity_bytes=1024,
    ).resume(replace(request, observation_token="observation-2"))

    assert result.reason_code == "stale_operation_token"
    reopened = NotesDeviceStateStore(database)
    assert reopened.get_operation("operation-1").state is (
        NotesSyncOperationState.NEEDS_ATTENTION
    )
    assert note_authority.replace_calls == 0


@pytest.mark.asyncio
async def test_resume_rejects_changed_desired_title_bound_by_recovery_intent(
    tmp_path: Path,
) -> None:
    store, _ = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="before", version=4))
    files = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=notes.snapshot,
        file=files.snapshot,
    )

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_admission,
        ).execute(request)

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).resume(replace(request, desired_title="Changed title"))

    assert result.reason_code == "recovery_authority_changed"
    assert notes.replace_calls == 0
    assert store.get_operation("operation-1").state is (
        NotesSyncOperationState.NEEDS_ATTENTION
    )


@pytest.mark.asyncio
async def test_root_paused_after_admission_blocks_resume_before_mutation(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    note_authority = FakeNoteAuthority(_note(content="before", version=4))
    filesystem = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=note_authority.snapshot,
        file=filesystem.snapshot,
    )

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            note_authority,
            filesystem,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_admission,
        ).execute(request)
    store.transition_root("root-1", NotesSyncRootState.PAUSED)

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        note_authority,
        filesystem,
        recovery_capacity_bytes=1024,
    ).resume(request)

    assert result.reason_code == "binding_authority_changed"
    assert note_authority.replace_calls == 0
    assert store.get_operation("operation-1").state is (
        NotesSyncOperationState.NEEDS_ATTENTION
    )


@pytest.mark.asyncio
async def test_root_paused_after_first_mutation_blocks_all_later_stages(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="before", version=4))
    files = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=notes.snapshot,
        file=files.snapshot,
    )

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_first,
        ).execute(request)
    store.transition_root("root-1", NotesSyncRootState.PAUSED)

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).resume(request)

    assert result.reason_code == "binding_authority_changed"
    assert notes.memberships == []
    assert store.get_operation("operation-1").state is (
        NotesSyncOperationState.NEEDS_ATTENTION
    )


@pytest.mark.asyncio
async def test_binding_baseline_drift_after_first_stage_is_never_overwritten(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="before", version=4))
    files = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=notes.snapshot,
        file=files.snapshot,
    )

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_first,
        ).execute(request)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            """
            UPDATE notes_sync_bindings
            SET note_version = 99, content_digest = ?
            WHERE binding_id = 'binding-1'
            """,
            (_digest("external"),),
        )

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).resume(request)

    assert result.reason_code == "binding_authority_changed"
    assert notes.memberships == []
    assert store.get_binding("binding-1").note_version == 99
    assert store.get_binding("binding-1").content_digest == _digest("external")


@pytest.mark.asyncio
async def test_folder_owner_change_before_membership_stage_never_redirects_placement(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="before", version=4))
    files = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=notes.snapshot,
        file=files.snapshot,
    )

    def crash_after_second(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.SECOND_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_second,
        ).execute(request)
    store.assign_root_folder("root-1", "folder-2")

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).resume(request)

    assert result.reason_code == "binding_authority_changed"
    assert notes.memberships == []
    assert store.get_binding("binding-1").note_version == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direction", "action"),
    (
        (NotesSyncDirection.FOLDER_TO_NOTES, NotesSyncActionKind.UPDATE_FILE),
        (NotesSyncDirection.NOTES_TO_FOLDER, NotesSyncActionKind.UPDATE_NOTE),
    ),
)
async def test_configured_direction_refuses_out_of_direction_execution(
    tmp_path: Path,
    direction: NotesSyncDirection,
    action: NotesSyncActionKind,
) -> None:
    store, database = _execution_store(tmp_path)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE notes_sync_roots SET direction = ? WHERE root_id = 'root-1'",
            (direction.value,),
        )
    notes = FakeNoteAuthority(
        _note(
            content="before" if action is NotesSyncActionKind.UPDATE_NOTE else "after",
            version=4,
        )
    )
    files = FakeFilesystem(
        _file(
            content="after" if action is NotesSyncActionKind.UPDATE_NOTE else "before"
        )
    )
    request = replace(
        _request(action=action, note=notes.snapshot, file=files.snapshot),
        direction=direction,
    )

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).execute(request)

    assert result.reason_code == "direction_disallows_action"
    assert notes.replace_calls == files.replace_calls == 0
    assert _counts(database) == (0, 0)


@pytest.mark.asyncio
async def test_reviewed_conflict_override_is_typed_persisted_and_token_bound(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    with store.transaction(immediate=True) as connection:
        connection.execute(
            """
            UPDATE notes_sync_roots SET direction = 'notes_to_folder'
            WHERE root_id = 'root-1'
            """
        )
    notes = FakeNoteAuthority(_note(content="before", version=4))
    files = FakeFilesystem(_file(content="after"))
    request = replace(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE,
            note=notes.snapshot,
            file=files.snapshot,
        ),
        direction=NotesSyncDirection.NOTES_TO_FOLDER,
        direction_override=NotesSyncDirectionOverride(
            review_id="review-1",
            action_kind=NotesSyncActionKind.UPDATE_NOTE,
            observation_token="observation-1",
        ),
    )

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).execute(request)
    metadata = json.loads(
        NotesDeviceStateStore(database)
        .load_recovery("recovery-operation-1")
        .metadata.decode("utf-8")
    )

    assert result.state is NotesSyncOperationState.COMPLETED
    assert metadata["direction_override"] == {
        "action": "update_note",
        "observation_token": "observation-1",
        "review_id": "review-1",
    }
    with pytest.raises(ValueError, match="observation_token"):
        replace(request, observation_token="observation-2")


@pytest.mark.asyncio
async def test_partial_filesystem_cleanup_authority_survives_private_store_reopen(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="after", version=4))
    files = PartialFilesystem(_file(content="before"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_FILE,
        note=notes.snapshot,
        file=files.snapshot,
    )

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=2048,
    ).execute(request)

    reopened = NotesDeviceStateStore(database)
    recovery = reopened.load_recovery("recovery-operation-1")
    metadata = json.loads(recovery.metadata.decode("utf-8"))
    assert result.reason_code == "replacement_cleanup_pending"
    assert reopened.get_operation("operation-1").state is (
        NotesSyncOperationState.NEEDS_ATTENTION
    )
    assert metadata["cleanup_relative_path"] == ".notes-sync-private-recovery"
    assert ".notes-sync-private-recovery" not in repr(result)


@pytest.mark.asyncio
async def test_reopened_executor_can_resolve_persisted_private_cleanup_authority(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="after", version=4))
    files = PartialFilesystem(_file(content="before"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_FILE,
        note=notes.snapshot,
        file=files.snapshot,
    )
    await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=2048,
    ).execute(request)

    reopened = NotesDeviceStateStore(database)
    result = await NotesSyncExecutor(
        reopened,
        notes,
        files,
        recovery_capacity_bytes=2048,
    ).resolve_filesystem_cleanup("operation-1")
    metadata = json.loads(
        reopened.load_recovery("recovery-operation-1").metadata.decode("utf-8")
    )

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert len(files.cleanup_calls) == 1
    assert repr(files.cleanup_calls[0]) == "NotesSyncPrivateCleanupHandle(<private>)"
    assert "cleanup_relative_path" not in metadata
    assert metadata["cleanup_pending"] is False
    assert metadata["cleanup_padding"]

    restored_recovery = reopened.load_recovery("recovery-operation-1")
    second = _recovery("operation-2", payload=b"x", metadata=b"y")
    exact_capacity = (
        len(restored_recovery.payload) + len(restored_recovery.metadata) + 2
    )
    assert reopened.admit_operation_recovery(
        _operation("operation-2"),
        second,
        capacity_bytes=exact_capacity,
    ).admitted
    second_partial = metadata.copy()
    second_partial.pop("cleanup_padding")
    second_partial["cleanup_pending"] = True
    second_partial["cleanup_relative_path"] = (
        ".note.md.tmp-0123456789abcdef0123456789abcdef"
    )
    second_partial["cleanup_reason_code"] = "replacement_cleanup_pending"
    second_partial["cleanup_identity"] = [7, 11, 1]
    encoded_partial = json.dumps(
        second_partial,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert len(encoded_partial) <= len(restored_recovery.metadata)
    reopened.mark_operation_partial_attention(
        "operation-1",
        "recovery-operation-1",
        "replacement_cleanup_pending",
        encoded_partial,
        capacity_bytes=exact_capacity,
    )


@pytest.mark.asyncio
async def test_reopened_executor_resolves_real_posix_private_cleanup_authority(
    tmp_path: Path,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="after", version=4))
    files = PartialFilesystem(_file(content="before"))
    private_leaf = ".note.md.tmp-0123456789abcdef0123456789abcdef"
    sync_root = tmp_path / "sync-root"
    sync_root.mkdir()
    (sync_root / "note.md").write_bytes(b"after")
    (sync_root / private_leaf).write_bytes(b"before")
    private_stat = (sync_root / private_leaf).stat()
    files.replace = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        NotesSyncFilesystemPartialError(
            "replacement_cleanup_pending",
            NotesSyncPrivateCleanupHandle(
                private_leaf,
                "replacement_cleanup_pending",
                SafeSyncFileIdentity(
                    device=private_stat.st_dev,
                    inode=private_stat.st_ino,
                    link_count=private_stat.st_nlink,
                ),
            ),
        )
    )
    request = _request(
        action=NotesSyncActionKind.UPDATE_FILE,
        note=notes.snapshot,
        file=files.snapshot,
    )
    await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=2048,
    ).execute(request)
    reopened = NotesDeviceStateStore(database)
    with PosixNotesSyncFilesystem(sync_root) as real_files:
        result = await NotesSyncExecutor(
            reopened,
            notes,
            real_files,
            recovery_capacity_bytes=2048,
        ).resolve_filesystem_cleanup("operation-1")

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert not (sync_root / private_leaf).exists()
    assert (sync_root / "note.md").read_bytes() == b"after"


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("locator", "identity"))
async def test_cleanup_locator_corruption_cannot_delete_another_private_file(
    tmp_path: Path,
    corruption: str,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    sync_root = tmp_path / "sync-root"
    sync_root.mkdir()
    leaf_a = ".note.md.tmp-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    leaf_b = ".note.md.tmp-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    path_a = sync_root / leaf_a
    path_b = sync_root / leaf_b
    path_a.write_bytes(b"authority-a")
    path_b.write_bytes(b"authority-b")
    identity_a = path_a.stat()
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="after", version=4))
    files = PartialFilesystem(_file(content="before"))
    files.replace = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        NotesSyncFilesystemPartialError(
            "replacement_cleanup_pending",
            NotesSyncPrivateCleanupHandle(
                leaf_a,
                "replacement_cleanup_pending",
                SafeSyncFileIdentity(
                    identity_a.st_dev,
                    identity_a.st_ino,
                    identity_a.st_nlink,
                ),
            ),
        )
    )
    await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=4096,
    ).execute(
        _request(
            action=NotesSyncActionKind.UPDATE_FILE,
            note=notes.snapshot,
            file=files.snapshot,
        )
    )
    with sqlite3.connect(database) as connection:
        raw = connection.execute(
            "SELECT metadata FROM notes_sync_recovery WHERE operation_id = ?",
            ("operation-1",),
        ).fetchone()[0]
        metadata = json.loads(raw.decode("utf-8"))
        if corruption == "locator":
            metadata["cleanup_relative_path"] = leaf_b
        else:
            identity_b = path_b.stat()
            metadata["cleanup_identity"] = [
                identity_b.st_dev,
                identity_b.st_ino,
                identity_b.st_nlink,
            ]
        connection.execute(
            "UPDATE notes_sync_recovery SET metadata = ? WHERE operation_id = ?",
            (
                json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode(),
                "operation-1",
            ),
        )

    with PosixNotesSyncFilesystem(sync_root) as real_files:
        with pytest.raises(NotesSyncFilesystemError, match="target_identity_changed"):
            await NotesSyncExecutor(
                NotesDeviceStateStore(database),
                notes,
                real_files,
                recovery_capacity_bytes=4096,
            ).resolve_filesystem_cleanup("operation-1")

    assert path_a.read_bytes() == b"authority-a"
    assert path_b.read_bytes() == b"authority-b"


def test_posix_cleanup_resolver_never_deletes_postcondition_target(
    tmp_path: Path,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    target = tmp_path / "note.md"
    target.write_bytes(b"authority")

    with PosixNotesSyncFilesystem(tmp_path) as filesystem:
        with pytest.raises(NotesSyncFilesystemError, match="cleanup_requires_review"):
            filesystem.resolve_cleanup(
                NotesSyncPrivateCleanupHandle(
                    "note.md",
                    "replacement_postcondition_failed",
                )
            )
        with pytest.raises(NotesSyncFilesystemError, match="invalid_cleanup_authority"):
            filesystem.resolve_cleanup(
                NotesSyncPrivateCleanupHandle(
                    "note.md",
                    "replacement_cleanup_pending",
                    SafeSyncFileIdentity(device=7, inode=11, link_count=1),
                )
            )

    assert target.read_bytes() == b"authority"


def test_posix_cleanup_quarantines_and_rolls_back_swapped_racer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    leaf = ".note.md.tmp-0123456789abcdef0123456789abcdef"
    target = tmp_path / leaf
    preserved = tmp_path / "preserved-authority"
    target.write_bytes(b"authority")
    target_stat = target.stat()

    with PosixNotesSyncFilesystem(tmp_path) as filesystem:

        def swap_before_quarantine(_relative_path: Path) -> None:
            target.rename(preserved)
            target.write_bytes(b"racer")

        monkeypatch.setattr(
            filesystem._root,
            "_before_cleanup_rename",
            swap_before_quarantine,
        )
        with pytest.raises(NotesSyncFilesystemError, match="target_identity_changed"):
            filesystem.resolve_cleanup(
                NotesSyncPrivateCleanupHandle(
                    leaf,
                    "replacement_cleanup_pending",
                    SafeSyncFileIdentity(
                        target_stat.st_dev,
                        target_stat.st_ino,
                        target_stat.st_nlink,
                    ),
                )
            )

    assert preserved.read_bytes() == b"authority"
    assert target.read_bytes() == b"racer"


def test_posix_create_file_is_guarded_no_clobber_and_preserves_profile(
    tmp_path: Path,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    profile = NotesSyncSerializationProfile(True, "crlf", True, 0o640)

    with PosixNotesSyncFilesystem(tmp_path) as filesystem:
        created = filesystem.create("created.md", "body", profile=profile)
        with pytest.raises(NotesSyncFilesystemError, match="target_identity_changed"):
            filesystem.create("created.md", "racer", profile=profile)

    assert created.observation.serialization == profile
    assert (tmp_path / "created.md").read_bytes() == b"\xef\xbb\xbfbody\r\n"
    assert (tmp_path / "created.md").stat().st_mode & 0o777 == 0o640


def test_posix_delete_file_quarantines_exact_created_authority(
    tmp_path: Path,
) -> None:
    if not PosixNotesSyncFilesystem.supports_writes():
        pytest.skip("guarded POSIX replacement is unavailable")
    profile = NotesSyncSerializationProfile(False, "lf", False, 0o600)

    with PosixNotesSyncFilesystem(tmp_path) as filesystem:
        created = filesystem.create("created.md", "body", profile=profile)
        filesystem.delete(expected=created)
        with pytest.raises(NotesSyncFilesystemError, match="missing_target"):
            filesystem.observe("created.md")

    assert not (tmp_path / "created.md").exists()
    assert not tuple(tmp_path.glob(".created.md.tmp-*"))


@pytest.mark.asyncio
async def test_null_cleanup_authority_remains_explicitly_fenced_after_reopen(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="after", version=4))
    files = NullCleanupPartialFilesystem(_file(content="before"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_FILE,
        note=notes.snapshot,
        file=files.snapshot,
    )
    await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=2048,
    ).execute(request)

    reopened = NotesDeviceStateStore(database)
    executor = NotesSyncExecutor(
        reopened,
        notes,
        files,
        recovery_capacity_bytes=2048,
    )
    result = await executor.resume(request)
    metadata = json.loads(
        reopened.load_recovery("recovery-operation-1").metadata.decode("utf-8")
    )

    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "replacement_commit_unverified"
    assert metadata["cleanup_pending"] is True
    assert metadata["cleanup_relative_path"] is None
    assert files.replace_calls == 1
    with pytest.raises(RuntimeError, match="recovery_authority_changed"):
        await executor.resolve_filesystem_cleanup("operation-1")


@pytest.mark.asyncio
async def test_unicode_path_cleanup_authority_fits_exact_admitted_capacity(
    tmp_path: Path,
) -> None:
    relative_path = f"{'界' * 80}.md"

    def setup_store(directory: Path) -> NotesDeviceStateStore:
        directory.mkdir()
        store, database = _execution_store(directory)
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE notes_sync_bindings SET normalized_relative_path = ?",
                (relative_path,),
            )
        return store

    note = _note(content="after", version=4)
    snapshot = _file_at(relative_path, content="before")
    sizing_store = setup_store(tmp_path / "sizing")
    sizing_request = _request(
        action=NotesSyncActionKind.UPDATE_FILE,
        note=note,
        file=snapshot,
    )
    sizing_executor = NotesSyncExecutor(
        sizing_store,
        FakeNoteAuthority(note),
        FakeFilesystem(snapshot),
        recovery_capacity_bytes=16 * 1024,
    )
    assert sizing_executor._admit(sizing_request)
    admitted = sizing_store.load_recovery("recovery-operation-1")
    exact_capacity = len(admitted.payload) + len(admitted.metadata)

    store = setup_store(tmp_path / "exact")
    files = PartialFilesystem(snapshot)
    private_path = f".{'界' * 80}.md.tmp-{'a' * 32}"
    files.replace = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        NotesSyncFilesystemPartialError(
            "replacement_cleanup_pending",
            NotesSyncPrivateCleanupHandle(
                private_path,
                "replacement_cleanup_pending",
                SafeSyncFileIdentity(7, 11, 1),
            ),
        )
    )
    result = await NotesSyncExecutor(
        store,
        FakeNoteAuthority(note),
        files,
        recovery_capacity_bytes=exact_capacity,
    ).execute(
        _request(
            action=NotesSyncActionKind.UPDATE_FILE,
            note=note,
            file=snapshot,
        )
    )

    assert result.reason_code == "replacement_cleanup_pending"
    assert store.get_operation("operation-1").state is (
        NotesSyncOperationState.NEEDS_ATTENTION
    )


@pytest.mark.asyncio
async def test_resume_never_completes_while_private_cleanup_is_pending(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="after", version=4))
    files = PartialFilesystem(_file(content="before"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_FILE,
        note=notes.snapshot,
        file=files.snapshot,
    )
    await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=2048,
    ).execute(request)

    reopened = NotesDeviceStateStore(database)
    result = await NotesSyncExecutor(
        reopened,
        notes,
        files,
        recovery_capacity_bytes=2048,
    ).resume(request)

    metadata = json.loads(
        reopened.load_recovery("recovery-operation-1").metadata.decode("utf-8")
    )
    assert result.state is NotesSyncOperationState.NEEDS_ATTENTION
    assert result.reason_code == "replacement_cleanup_pending"
    assert metadata["cleanup_relative_path"] == ".notes-sync-private-recovery"


@pytest.mark.asyncio
async def test_partial_cleanup_store_failure_keeps_private_authority_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="after", version=4))
    files = PartialFilesystem(_file(content="before"))
    monkeypatch.setattr(
        store,
        "mark_operation_partial_attention",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            NotesDeviceStateError("PRIVATE/path/store")
        ),
    )

    with pytest.raises(NotesSyncExecutionPartialError) as raised:
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=2048,
        ).execute(
            _request(
                action=NotesSyncActionKind.UPDATE_FILE,
                note=notes.snapshot,
                file=files.snapshot,
            )
        )

    assert raised.value.reason_code == "replacement_cleanup_pending"
    assert raised.value.cleanup_handle.private_relative_path is not None
    assert ".notes-sync-private-recovery" not in repr(raised.value)


@pytest.mark.asyncio
async def test_blocking_file_mutation_is_joined_before_cancellation_is_redelivered(
    tmp_path: Path,
) -> None:
    store, _ = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="after", version=4))
    files = BlockingFilesystem(_file(content="before"))
    stages: list[NotesSyncOperationState] = []
    executor = NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=2048,
        after_stage=stages.append,
    )
    task = asyncio.create_task(
        executor.execute(
            _request(
                action=NotesSyncActionKind.UPDATE_FILE,
                note=notes.snapshot,
                file=files.snapshot,
            )
        )
    )

    assert await asyncio.to_thread(files.started.wait, 3.0)
    task.cancel()
    files.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert files.snapshot.text == "after"
    assert NotesSyncOperationState.FIRST_AUTHORITY_APPLIED in stages
    assert store.get_operation("operation-1").state is (
        NotesSyncOperationState.NEEDS_ATTENTION
    )


@pytest.mark.asyncio
async def test_attention_store_failure_never_replaces_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = _execution_store(tmp_path)
    notes = FakeNoteAuthority(
        _note(content="before", version=4), cancel_after_replace=True
    )
    files = FakeFilesystem(_file(content="after"))
    monkeypatch.setattr(
        store,
        "mark_operation_attention",
        lambda *_args: (_ for _ in ()).throw(
            NotesDeviceStateError("PRIVATE attention write failed")
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=1024,
        ).execute(
            _request(
                action=NotesSyncActionKind.UPDATE_NOTE,
                note=_note(content="before", version=4),
                file=files.snapshot,
            )
        )


@pytest.mark.asyncio
async def test_arbitrary_reason_code_attribute_is_never_trusted(
    tmp_path: Path,
) -> None:
    class CredentialError(Exception):
        reason_code = "credential_secret"

    store, _ = _execution_store(tmp_path)
    notes = FakeNoteAuthority(
        _note(content="before", version=4),
        replace_error=CredentialError("PRIVATE/token/path"),
    )
    files = FakeFilesystem(_file(content="after"))

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).execute(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE,
            note=notes.snapshot,
            file=files.snapshot,
        )
    )

    assert result.reason_code == "executor_failed"
    assert store.get_operation("operation-1").reason_code == "executor_failed"


@pytest.mark.asyncio
async def test_reopen_lists_and_reconstructs_incomplete_request_without_memory(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="before", version=4))
    files = FakeFilesystem(_file(content="after"))

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_first,
        ).execute(
            _request(
                action=NotesSyncActionKind.UPDATE_NOTE,
                note=notes.snapshot,
                file=files.snapshot,
            )
        )

    reopened = NotesDeviceStateStore(database)
    executor = NotesSyncExecutor(
        reopened,
        notes,
        files,
        recovery_capacity_bytes=1024,
    )
    incomplete = reopened.list_incomplete_operations()
    reconstructed = await executor.reconstruct_request("operation-1")
    result = await executor.resume(reconstructed)

    assert tuple(record.operation_id for record in incomplete) == ("operation-1",)
    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.replace_calls == 1


@pytest.mark.asyncio
async def test_reopen_reconstructs_original_file_recovery_without_memory(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="after", version=5))
    files = FakeFilesystem(_file(content="before"))

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=2048,
            after_stage=crash_after_first,
        ).execute(
            _request(
                action=NotesSyncActionKind.UPDATE_FILE,
                note=notes.snapshot,
                file=files.snapshot,
            )
        )

    reopened = NotesDeviceStateStore(database)
    executor = NotesSyncExecutor(
        reopened,
        notes,
        files,
        recovery_capacity_bytes=2048,
    )
    reconstructed = await executor.reconstruct_request("operation-1")
    result = await executor.resume(reconstructed)

    assert reconstructed.file.raw_bytes == b"before"
    assert result.state is NotesSyncOperationState.COMPLETED
    assert files.replace_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "note_content", "note_version", "file_content"),
    (
        (NotesSyncActionKind.UPDATE_NOTE, "before", 4, "after"),
        (NotesSyncActionKind.UPDATE_FILE, "after", 5, "before"),
    ),
)
async def test_reconstruction_rejects_corrupt_recovery_payload_after_first_write(
    tmp_path: Path,
    action: NotesSyncActionKind,
    note_content: str,
    note_version: int,
    file_content: str,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content=note_content, version=note_version))
    files = FakeFilesystem(_file(content=file_content))

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=2048,
            after_stage=crash_after_first,
        ).execute(_request(action=action, note=notes.snapshot, file=files.snapshot))
    note_writes = notes.replace_calls
    file_writes = files.replace_calls
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE notes_sync_recovery SET payload = ? WHERE operation_id = ?",
            (b"corrupt-authority", "operation-1"),
        )

    reopened = NotesDeviceStateStore(database)
    with pytest.raises(RuntimeError, match="recovery_authority_changed"):
        await NotesSyncExecutor(
            reopened,
            notes,
            files,
            recovery_capacity_bytes=2048,
        ).reconstruct_request("operation-1")

    assert notes.replace_calls == note_writes
    assert files.replace_calls == file_writes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (
        NotesSyncActionKind.CREATE_NOTE,
        NotesSyncActionKind.CREATE_FILE,
        NotesSyncActionKind.MOVE_FILE,
    ),
)
async def test_new_action_reconstruction_rejects_corrupt_recovery_after_first(
    tmp_path: Path,
    action: NotesSyncActionKind,
) -> None:
    store, database, notes, files, request = _new_action_fixture(tmp_path, action)

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)
    writes = _new_action_mutation_count(action, notes, files)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE notes_sync_recovery SET payload = ? WHERE operation_id = ?",
            (b"corrupt-authority", request.operation_id),
        )

    with pytest.raises(RuntimeError, match="recovery_authority_changed"):
        await NotesSyncExecutor(
            NotesDeviceStateStore(database),
            notes,
            files,
            recovery_capacity_bytes=4096,
        ).reconstruct_request(request.operation_id)

    assert _new_action_mutation_count(action, notes, files) == writes


@pytest.mark.asyncio
async def test_create_note_reconstruction_rejects_same_bytes_on_replaced_file(
    tmp_path: Path,
) -> None:
    store, database, notes, files, request = _new_action_fixture(
        tmp_path, NotesSyncActionKind.CREATE_NOTE
    )

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)
    files.snapshot = _file(content="authority", inode=99)

    with pytest.raises(RuntimeError, match="recovery_authority_changed"):
        await NotesSyncExecutor(
            NotesDeviceStateStore(database),
            notes,
            files,
            recovery_capacity_bytes=4096,
        ).reconstruct_request(request.operation_id)


@pytest.mark.asyncio
async def test_update_note_reconstruction_rejects_same_bytes_on_replaced_identity(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="before", version=4))
    files = FakeFilesystem(_file(content="after"))

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=2048,
            after_stage=crash_after_first,
        ).execute(
            _request(
                action=NotesSyncActionKind.UPDATE_NOTE,
                note=notes.snapshot,
                file=files.snapshot,
            )
        )
    writes = notes.replace_calls
    files.snapshot = _file(content="after", inode=99)

    with pytest.raises(RuntimeError, match="recovery_authority_changed"):
        await NotesSyncExecutor(
            NotesDeviceStateStore(database),
            notes,
            files,
            recovery_capacity_bytes=2048,
        ).reconstruct_request("operation-1")

    assert notes.replace_calls == writes


@pytest.mark.asyncio
async def test_concurrent_calls_for_one_operation_coalesce_one_mutation(
    tmp_path: Path,
) -> None:
    store, _ = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="after", version=4))
    files = DuplicateBlockingFilesystem(_file(content="before"))
    executor = NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=2048,
    )
    request = _request(
        action=NotesSyncActionKind.UPDATE_FILE,
        note=notes.snapshot,
        file=files.snapshot,
    )

    first = asyncio.create_task(executor.execute(request))
    assert await asyncio.to_thread(files.started.wait, 3.0)
    second = asyncio.create_task(executor.execute(request))
    await asyncio.sleep(0.05)
    files.release.set()
    results = await asyncio.gather(first, second)

    assert files.replace_calls == 1
    assert [result.state for result in results] == [
        NotesSyncOperationState.COMPLETED,
        NotesSyncOperationState.COMPLETED,
    ]


@pytest.mark.asyncio
async def test_two_executors_sharing_store_coalesce_one_operation_mutation(
    tmp_path: Path,
) -> None:
    store, _ = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="after", version=4))
    files = DuplicateBlockingFilesystem(_file(content="before"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_FILE,
        note=notes.snapshot,
        file=files.snapshot,
    )
    first_executor = NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=2048,
    )
    second_executor = NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=2048,
    )

    first = asyncio.create_task(first_executor.execute(request))
    assert await asyncio.to_thread(files.started.wait, 3.0)
    second = asyncio.create_task(second_executor.execute(request))
    await asyncio.sleep(0.05)
    files.release.set()
    results = await asyncio.gather(first, second)

    assert files.replace_calls == 1
    assert [result.state for result in results] == [
        NotesSyncOperationState.COMPLETED,
        NotesSyncOperationState.COMPLETED,
    ]


@pytest.mark.asyncio
async def test_owner_reconciliation_preserves_other_active_binding_memberships(
    tmp_path: Path,
) -> None:
    store, _ = _execution_store(tmp_path)
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-2",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-2",
            normalized_relative_path="other.md",
            stable_identity_digest="c" * 64,
            state=NotesSyncBindingState.ACTIVE,
            serialization=_file(content="other").observation.serialization,
            content_digest=_digest("other"),
            note_version=2,
        )
    )
    note_authority = FakeNoteAuthority(_note(content="before", version=4))
    filesystem = FakeFilesystem(_file(content="after"))

    result = await NotesSyncExecutor(
        store,
        note_authority,
        filesystem,
        recovery_capacity_bytes=1024,
    ).execute(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE,
            note=note_authority.snapshot,
            file=filesystem.snapshot,
        )
    )

    assert result.state is NotesSyncOperationState.COMPLETED
    assert note_authority.memberships == [
        (
            "root-1",
            (("folder-1", "note-1"), ("folder-1", "note-2")),
        )
    ]


def test_execution_public_models_reject_ambiguous_states() -> None:
    with pytest.raises(TypeError, match="action_kind"):
        replace(
            _request(
                action=NotesSyncActionKind.UPDATE_NOTE,
                note=_note(content="before", version=4),
                file=_file(content="after"),
            ),
            action_kind="update_note",
        )
    with pytest.raises(ValueError, match="reason_code"):
        NotesSyncExecutionResult(
            operation_id="operation-1",
            state=NotesSyncOperationState.COMPLETED,
            recovery_required=False,
            reason_code="stale_observation",
        )
    with pytest.raises(TypeError, match="recovery_required"):
        NotesSyncExecutionResult(
            operation_id="operation-1",
            state=NotesSyncOperationState.NEEDS_ATTENTION,
            recovery_required="yes",  # type: ignore[arg-type]
            reason_code="stale_observation",
        )
    with pytest.raises(ValueError, match="recovery_required"):
        NotesSyncExecutionResult(
            operation_id="operation-1",
            state=NotesSyncOperationState.COMPLETED,
            recovery_required=True,
        )
    with pytest.raises(ValueError, match="write authority"):
        NotesSyncExecutionRequest(
            operation_id="operation-1",
            root_id="root-1",
            logical_folder_id="folder-1",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            binding_id="binding-1",
            observation_token="observation-1",
            action_kind=NotesSyncActionKind.UPDATE_FILE,
            note=_note(content="after", version=5),
            file=_windows_file(content="before"),
            desired_title="Title",
            recovery_id="recovery-operation-1",
            recovery_expires_at=100_000,
        )


@pytest.mark.asyncio
async def test_executor_never_promotes_raw_exception_text_to_reason_code(
    tmp_path: Path,
) -> None:
    store, _ = _execution_store(tmp_path)
    notes = FakeNoteAuthority(
        _note(content="before", version=4),
        replace_error=RuntimeError("credential_secret"),
    )
    files = FakeFilesystem(_file(content="after"))

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).execute(
        _request(
            action=NotesSyncActionKind.UPDATE_NOTE,
            note=notes.snapshot,
            file=files.snapshot,
        )
    )

    assert result.reason_code == "executor_failed"
    assert store.get_operation("operation-1").reason_code == "executor_failed"
    assert "credential_secret" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (NotesSyncActionKind.UPDATE_NOTE, NotesSyncActionKind.UPDATE_FILE),
)
async def test_restore_uses_recovery_verifies_then_disconnects_item(
    tmp_path: Path,
    action: NotesSyncActionKind,
) -> None:
    store, database = _execution_store(tmp_path)
    note_before = _note(
        content="before" if action is NotesSyncActionKind.UPDATE_NOTE else "after",
        version=4 if action is NotesSyncActionKind.UPDATE_NOTE else 5,
    )
    file_before = _file(
        content="after" if action is NotesSyncActionKind.UPDATE_NOTE else "before"
    )
    notes = FakeNoteAuthority(
        note_before,
        cancel_after_replace=action is NotesSyncActionKind.UPDATE_NOTE,
    )
    files = FakeFilesystem(file_before)
    request = _request(action=action, note=note_before, file=file_before)
    executor = NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=2048,
    )

    if action is NotesSyncActionKind.UPDATE_NOTE:
        with pytest.raises(__import__("asyncio").CancelledError):
            await executor.execute(request)
        notes._cancel_after_replace = False
    else:

        def stop_after_first(stage: NotesSyncOperationState) -> None:
            if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
                raise InjectedCrash

        with pytest.raises(InjectedCrash):
            await NotesSyncExecutor(
                store,
                notes,
                files,
                recovery_capacity_bytes=2048,
                after_stage=stop_after_first,
            ).execute(request)
        store.mark_operation_attention("operation-1", "restore_requested")

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=2048,
    ).restore(request)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.snapshot.content == note_before.content
    assert files.snapshot.raw_bytes == file_before.raw_bytes
    assert store.get_binding("binding-1").state is (NotesSyncBindingState.DISCONNECTED)
    assert store.load_recovery("recovery-operation-1").payload == (
        note_before.content.encode("utf-8")
        if action is NotesSyncActionKind.UPDATE_NOTE
        else file_before.raw_bytes
    )
    assert notes.memberships[-1] == ("root-1", ())


@pytest.mark.asyncio
async def test_restore_cancellation_reopens_idempotently_without_replaying_write(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(
        _note(content="before", version=4), cancel_after_replace=True
    )
    files = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=_note(content="before", version=4),
        file=files.snapshot,
    )
    executor = NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=1024,
    )
    with pytest.raises(__import__("asyncio").CancelledError):
        await executor.execute(request)
    with pytest.raises(__import__("asyncio").CancelledError):
        await executor.restore(request)
    assert notes.snapshot.content == "before"
    writes_after_restore = notes.replace_calls

    notes._cancel_after_replace = False
    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).restore(request)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.replace_calls == writes_after_restore


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (NotesSyncActionKind.CREATE_NOTE, NotesSyncActionKind.CREATE_FILE),
)
async def test_create_attention_executes_non_destructive_disconnect(
    tmp_path: Path,
    action: NotesSyncActionKind,
) -> None:
    store, database, notes, files, request = _new_action_fixture(tmp_path, action)

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)
    store.mark_operation_attention(request.operation_id, "disconnect_requested")
    executor = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    )

    result = await executor.disconnect(request)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.memberships == [("root-1", ())]
    assert _new_action_mutation_count(action, notes, files) == 1
    with pytest.raises(NotesDeviceStateError):
        store.get_binding("binding-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (NotesSyncActionKind.CREATE_NOTE, NotesSyncActionKind.CREATE_FILE),
)
async def test_create_restore_returns_to_original_missing_side(
    tmp_path: Path,
    action: NotesSyncActionKind,
) -> None:
    store, database, notes, files, request = _new_action_fixture(tmp_path, action)

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)
    store.mark_operation_attention(request.operation_id, "restore_requested")

    reopened = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    )
    reconstructed = await reopened.reconstruct_request(request.operation_id)
    result = await reopened.restore(reconstructed)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.memberships == [("root-1", ())]
    if action is NotesSyncActionKind.CREATE_NOTE:
        with pytest.raises(NotesSyncAuthorityError, match="note_missing"):
            await notes.observe("note-1")
        assert files.snapshot == request.file
    else:
        assert files.snapshot is None
        assert notes.snapshot == request.note
    with pytest.raises(NotesDeviceStateError):
        store.get_binding("binding-1")


@pytest.mark.asyncio
async def test_move_restore_moves_back_then_disconnects_without_content_loss(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    note = _note(content="before", version=4)
    notes = FakeNoteAuthority(note)
    source = _file(content="before")
    files = RestorableMovingFilesystem(source)
    request = replace(
        _request(action=NotesSyncActionKind.UPDATE_FILE, note=note, file=source),
        action_kind=NotesSyncActionKind.MOVE_FILE,
        move_destination_relative_path="moved.md",
    )

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)
    store.mark_operation_attention(request.operation_id, "restore_requested")

    reopened = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    )
    reconstructed = await reopened.reconstruct_request(request.operation_id)
    result = await reopened.restore(reconstructed)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert files.current.observation.relative_path == "note.md"
    assert files.current.raw_bytes == b"before"
    assert files.move_calls == 2
    assert store.get_binding("binding-1").state is NotesSyncBindingState.DISCONNECTED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (
        NotesSyncActionKind.CREATE_NOTE,
        NotesSyncActionKind.CREATE_FILE,
        NotesSyncActionKind.MOVE_FILE,
    ),
)
async def test_new_action_restore_cancellation_reopens_without_delete_or_move_replay(
    tmp_path: Path,
    action: NotesSyncActionKind,
) -> None:
    if action is NotesSyncActionKind.MOVE_FILE:
        store, database = _execution_store(tmp_path)
        note = _note(content="before", version=4)
        notes: FakeNoteAuthority = FakeNoteAuthority(note)
        source = _file(content="before")
        files: object = RestorableMovingFilesystem(source)
        request = replace(
            _request(action=NotesSyncActionKind.UPDATE_FILE, note=note, file=source),
            action_kind=action,
            move_destination_relative_path="moved.md",
        )
    else:
        store, database, notes, files, request = _new_action_fixture(tmp_path, action)

    def crash_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=crash_after_first,
        ).execute(request)
    store.mark_operation_attention(request.operation_id, "restore_requested")

    def cancel_after_first(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await NotesSyncExecutor(
            NotesDeviceStateStore(database),
            notes,
            files,
            recovery_capacity_bytes=4096,
            after_stage=cancel_after_first,
        ).restore(request)
    reopened = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=4096,
    )
    reconstructed = await reopened.reconstruct_request(request.operation_id)
    result = await reopened.restore(reconstructed)

    assert result.state is NotesSyncOperationState.COMPLETED
    if action is NotesSyncActionKind.CREATE_NOTE:
        assert notes.delete_calls == 1  # type: ignore[attr-defined]
    elif action is NotesSyncActionKind.CREATE_FILE:
        assert files.delete_calls == 1  # type: ignore[attr-defined]
    else:
        assert files.move_calls == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crash_stage",
    (
        NotesSyncOperationState.RECOVERY_ADMITTED,
        NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
        NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
        NotesSyncOperationState.BINDING_UPDATED,
        NotesSyncOperationState.VERIFIED,
    ),
)
async def test_restore_reopens_after_every_durable_stage_without_replay(
    tmp_path: Path,
    crash_stage: NotesSyncOperationState,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(
        _note(content="before", version=4), cancel_after_replace=True
    )
    files = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=_note(content="before", version=4),
        file=files.snapshot,
    )
    with pytest.raises(__import__("asyncio").CancelledError):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=1024,
        ).execute(request)
    notes._cancel_after_replace = False

    def crash_after(stage: NotesSyncOperationState) -> None:
        if stage is crash_stage:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            NotesDeviceStateStore(database),
            notes,
            files,
            recovery_capacity_bytes=1024,
            after_stage=crash_after,
        ).restore(request)
    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).restore(request)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.snapshot.content == "before"
    assert notes.replace_calls == 2


@pytest.mark.asyncio
async def test_restore_refuses_stale_source_and_keeps_recovery_attention(
    tmp_path: Path,
) -> None:
    store, _ = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="before", version=4))
    files = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=notes.snapshot,
        file=files.snapshot,
    )

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_admission,
        ).execute(request)
    store.mark_operation_attention("operation-1", "restore_requested")
    files.snapshot = _file(content="external")

    result = await NotesSyncExecutor(
        store,
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).restore(request)

    assert result.reason_code == "stale_restore_observation"
    assert notes.replace_calls == 0
    assert store.get_operation("operation-1").state is (
        NotesSyncOperationState.NEEDS_ATTENTION
    )
    assert store.load_recovery("recovery-operation-1").payload == b"before"


@pytest.mark.asyncio
async def test_disconnect_is_non_destructive_and_preserves_sibling_placement(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-2",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-2",
            normalized_relative_path="other.md",
            stable_identity_digest="c" * 64,
            state=NotesSyncBindingState.ACTIVE,
            serialization=_file(content="other").observation.serialization,
            content_digest=_digest("other"),
            note_version=2,
        )
    )
    notes = FakeNoteAuthority(_note(content="before", version=4))
    files = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=notes.snapshot,
        file=files.snapshot,
    )

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_admission,
        ).execute(request)
    store.mark_operation_attention("operation-1", "disconnect_requested")
    note_before, file_before = notes.snapshot, files.snapshot

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).disconnect(request)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.snapshot == note_before
    assert files.snapshot == file_before
    assert notes.replace_calls == files.replace_calls == 0
    assert store.get_binding("binding-1").state is NotesSyncBindingState.DISCONNECTED
    assert store.get_binding("binding-2").state is NotesSyncBindingState.ACTIVE
    assert notes.memberships == [("root-1", (("folder-1", "note-2"),))]


@pytest.mark.asyncio
async def test_disconnect_cancellation_is_attention_and_reopen_is_idempotent(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(
        _note(content="before", version=4), cancel_after_membership=True
    )
    files = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=notes.snapshot,
        file=files.snapshot,
    )

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_admission,
        ).execute(request)
    store.mark_operation_attention("operation-1", "disconnect_requested")
    executor = NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=1024,
    )
    with pytest.raises(__import__("asyncio").CancelledError):
        await executor.disconnect(request)
    assert store.get_operation("operation-1").state is (
        NotesSyncOperationState.NEEDS_ATTENTION
    )

    notes.cancel_after_membership = False
    result = await executor.disconnect(request)
    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.memberships == [("root-1", ())]


@pytest.mark.asyncio
async def test_disconnect_remains_available_when_recovery_is_missing(
    tmp_path: Path,
) -> None:
    store, database = _execution_store(tmp_path)
    notes = FakeNoteAuthority(_note(content="before", version=4))
    files = FakeFilesystem(_file(content="after"))
    request = _request(
        action=NotesSyncActionKind.UPDATE_NOTE,
        note=notes.snapshot,
        file=files.snapshot,
    )

    def crash_after_admission(stage: NotesSyncOperationState) -> None:
        if stage is NotesSyncOperationState.RECOVERY_ADMITTED:
            raise InjectedCrash

    with pytest.raises(InjectedCrash):
        await NotesSyncExecutor(
            store,
            notes,
            files,
            recovery_capacity_bytes=1024,
            after_stage=crash_after_admission,
        ).execute(request)
    store.mark_operation_attention("operation-1", "disconnect_requested")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM notes_sync_recovery WHERE operation_id = ?",
            ("operation-1",),
        )
        connection.commit()

    result = await NotesSyncExecutor(
        NotesDeviceStateStore(database),
        notes,
        files,
        recovery_capacity_bytes=1024,
    ).disconnect(request)

    assert result.state is NotesSyncOperationState.COMPLETED
    assert notes.replace_calls == files.replace_calls == 0
    assert store.get_binding("binding-1").state is NotesSyncBindingState.DISCONNECTED


def test_public_results_and_executor_source_disclose_no_private_values() -> None:
    result = NotesSyncExecutionResult(
        operation_id="operation-private",
        state=NotesSyncOperationState.NEEDS_ATTENTION,
        recovery_required=True,
        reason_code="stale_observation",
        choices=tuple(NotesSyncRecoveryChoice),
    )
    rendered = repr(result)
    source = Path("tldw_chatbook/Notes/notes_sync_executor.py").read_text(
        encoding="utf-8"
    )

    assert "operation-private" not in rendered
    assert "/private/notes/note.md" not in rendered
    assert "ChaChaNotes" not in source
    assert "FileNotes" not in source
    assert "Notes_Library" not in source
