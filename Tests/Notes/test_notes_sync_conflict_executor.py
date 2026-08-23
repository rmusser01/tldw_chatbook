"""Durable executor contracts for reviewed Keep file/Keep note choices."""

from __future__ import annotations

import asyncio
import json
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
)
from tldw_chatbook.Notes.notes_sync_executor import (
    NotesSyncExecutionRequest,
    NotesSyncExecutor,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncBindingState,
    NotesSyncOperationState,
)


pytestmark = pytest.mark.unit


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


def test_conflict_recovery_retention_is_exactly_thirty_days() -> None:
    assert executor_module.CONFLICT_RECOVERY_RETENTION_NS == (
        30 * 24 * 60 * 60 * 1_000_000_000
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
