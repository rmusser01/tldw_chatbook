"""Durable executor contracts for reviewed Keep file/Keep note choices."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import tldw_chatbook.Notes.notes_sync_executor as executor_module
from Tests.Notes.test_notes_sync_executor import (
    FakeFilesystem,
    FakeNoteAuthority,
    InjectedCrash,
    _execution_store,
    _file,
    _note,
    _request,
)
from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore
from tldw_chatbook.Notes.notes_sync_executor import (
    NotesSyncExecutionRequest,
    NotesSyncExecutor,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
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
    journal_kind: str,
    action: NotesSyncActionKind,
) -> None:
    store, database = _execution_store(tmp_path)
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
    assert recovery.expires_at == 900_000

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
