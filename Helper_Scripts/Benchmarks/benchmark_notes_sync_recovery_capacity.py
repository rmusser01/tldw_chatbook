"""Measure durable recovery using the production executor and managed move path."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncOperationRecord,
    NotesSyncRecoveryRecord,
    NotesSyncRootRecord,
)
from tldw_chatbook.Notes.notes_sync_authority import NotesSyncNoteSnapshot
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


MAX_REPRESENTATIVE_FILE_BYTES = 10 * 1024 * 1024
MANAGED_MOVE_ITEM_COUNT = 16
RECOVERY_HEADROOM_FACTOR = 1.5
REPRESENTATIVE_RECOVERY_BYTES = MAX_REPRESENTATIVE_FILE_BYTES * MANAGED_MOVE_ITEM_COUNT
DEFAULT_RECOVERY_CAPACITY_BYTES = 256 * 1024 * 1024


class _Notes:
    def __init__(self, snapshots: dict[str, NotesSyncNoteSnapshot]) -> None:
        self._snapshots = snapshots

    async def observe(self, note_id: str) -> NotesSyncNoteSnapshot:
        return self._snapshots[note_id]

    async def reconcile_managed_memberships(
        self,
        *,
        owner_id: str,
        desired: tuple[tuple[str, str], ...],
    ) -> None:
        del owner_id, desired


def _store(directory: Path, sync_root: Path) -> tuple[NotesDeviceStateStore, Path]:
    database = directory / "notes-sync-state.sqlite3"
    store = NotesDeviceStateStore(database)
    store.create_root(
        NotesSyncRootRecord(
            root_id="benchmark-root",
            note_scope_id="local_note",
            logical_folder_id="benchmark-folder",
            canonical_path=str(sync_root),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.ACTIVE,
        )
    )
    return store, database


def _recovery_bytes(database: Path) -> int:
    with sqlite3.connect(database) as connection:
        return int(
            connection.execute(
                "SELECT COALESCE(SUM(length(payload) + length(metadata)), 0) FROM notes_sync_recovery"
            ).fetchone()[0]
        )


def measure_replacement(*, payload_bytes: int) -> dict[str, int | float | str]:
    """Persist one replacement-sized record for the single-item boundary."""

    payload = b"r" * payload_bytes
    metadata = b'{"kind":"representative"}'
    with tempfile.TemporaryDirectory(prefix="notes-sync-recovery-") as temporary:
        directory = Path(temporary)
        sync_root = directory / "sync-root"
        sync_root.mkdir()
        store, database = _store(directory, sync_root)
        tracemalloc.start()
        started = time.perf_counter()
        decision = store.admit_operation_recovery(
            NotesSyncOperationRecord(
                operation_id="operation-0000",
                root_id="benchmark-root",
                binding_id=None,
                kind="update_file",
                state=NotesSyncOperationState.PENDING,
                reason_code=None,
                observation_token="observation-0000",
                expected_note_version=1,
                expected_file_digest="a" * 64,
            ),
            NotesSyncRecoveryRecord(
                recovery_id="recovery-0000",
                operation_id="operation-0000",
                payload=payload,
                metadata=metadata,
                expires_at=2**62,
            ),
            capacity_bytes=DEFAULT_RECOVERY_CAPACITY_BYTES,
        )
        if not decision.admitted:
            raise RuntimeError("representative replacement was not admitted")
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        recovery_bytes = _recovery_bytes(database)
    return {
        "case": "replacement",
        "items": 1,
        "payload_bytes_each": payload_bytes,
        "recovery_bytes": recovery_bytes,
        "elapsed_seconds": round(elapsed, 6),
        "peak_python_bytes": peak,
    }


async def _execute_managed_moves(
    store: NotesDeviceStateStore,
    filesystem: PosixNotesSyncFilesystem,
    *,
    item_count: int,
    content: str,
) -> None:
    digest = hashlib.sha256(content.encode()).hexdigest()
    notes = _Notes(
        {
            f"note-{index:04d}": NotesSyncNoteSnapshot(
                note_scope_id="local_note",
                note_id=f"note-{index:04d}",
                title="Benchmark",
                content=content,
                version=1,
                content_digest=digest,
            )
            for index in range(item_count)
        }
    )
    executor = NotesSyncExecutor(
        store,
        notes,
        filesystem,
        recovery_capacity_bytes=DEFAULT_RECOVERY_CAPACITY_BYTES,
    )
    for index in range(item_count):
        suffix = f"{index:04d}"
        source = filesystem.observe(f"source-{suffix}.md")
        note = await notes.observe(f"note-{suffix}")
        request = NotesSyncExecutionRequest(
            operation_id=f"operation-{suffix}",
            root_id="benchmark-root",
            logical_folder_id="benchmark-folder",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            binding_id=f"binding-{suffix}",
            observation_token=f"observation-{suffix}",
            action_kind=NotesSyncActionKind.MOVE_FILE,
            note=note,
            file=source,
            desired_title=note.title,
            recovery_id=f"recovery-{suffix}",
            recovery_expires_at=2**62,
            move_destination_relative_path=f"moved-{suffix}.md",
        )
        result = await executor.execute(request)
        if result.state is not NotesSyncOperationState.COMPLETED:
            raise RuntimeError("representative managed move did not complete")


def measure_managed_move(
    *,
    item_count: int,
    payload_bytes: int,
) -> dict[str, int | float | str]:
    """Execute guarded same-root moves while retaining every recovery payload."""

    if not PosixNotesSyncFilesystem.supports_writes():
        raise RuntimeError("guarded POSIX move is unavailable")
    content = "r" * payload_bytes
    encoded = content.encode()
    with tempfile.TemporaryDirectory(prefix="notes-sync-managed-move-") as temporary:
        directory = Path(temporary)
        sync_root = directory / "sync-root"
        sync_root.mkdir()
        store, database = _store(directory, sync_root)
        with PosixNotesSyncFilesystem(sync_root) as filesystem:
            for index in range(item_count):
                suffix = f"{index:04d}"
                (sync_root / f"source-{suffix}.md").write_bytes(encoded)
                snapshot = filesystem.observe(f"source-{suffix}.md")
                store.create_binding(
                    NotesSyncBindingRecord(
                        binding_id=f"binding-{suffix}",
                        root_id="benchmark-root",
                        note_scope_id="local_note",
                        note_id=f"note-{suffix}",
                        normalized_relative_path=f"source-{suffix}.md",
                        stable_identity_digest=NotesSyncExecutor.stable_identity_digest(
                            snapshot
                        ),
                        state=NotesSyncBindingState.ACTIVE,
                        serialization=snapshot.observation.serialization,
                        content_digest=snapshot.observation.content_digest,
                        note_version=1,
                    )
                )
            tracemalloc.start()
            started = time.perf_counter()
            asyncio.run(
                _execute_managed_moves(
                    store,
                    filesystem,
                    item_count=item_count,
                    content=content,
                )
            )
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        recovery_bytes = _recovery_bytes(database)
    return {
        "case": "managed_move",
        "items": item_count,
        "payload_bytes_each": payload_bytes,
        "recovery_bytes": recovery_bytes,
        "elapsed_seconds": round(elapsed, 6),
        "peak_python_bytes": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("replacement", "managed_move", "all"),
        default="all",
    )
    selected = parser.parse_args().case
    headroom_bytes = int(REPRESENTATIVE_RECOVERY_BYTES * RECOVERY_HEADROOM_FACTOR)
    assert DEFAULT_RECOVERY_CAPACITY_BYTES >= headroom_bytes
    results: list[dict[str, int | float | str]] = []
    if selected in {"replacement", "all"}:
        results.append(measure_replacement(payload_bytes=MAX_REPRESENTATIVE_FILE_BYTES))
    if selected in {"managed_move", "all"}:
        results.append(
            measure_managed_move(
                item_count=MANAGED_MOVE_ITEM_COUNT,
                payload_bytes=MAX_REPRESENTATIVE_FILE_BYTES,
            )
        )
    largest_measured = max(int(result["recovery_bytes"]) for result in results)
    assert DEFAULT_RECOVERY_CAPACITY_BYTES >= int(
        largest_measured * RECOVERY_HEADROOM_FACTOR
    )
    print(
        json.dumps(
            {
                "results": results,
                "largest_representative_bytes": largest_measured,
                "headroom_factor": RECOVERY_HEADROOM_FACTOR,
                "headroom_bytes": int(largest_measured * RECOVERY_HEADROOM_FACTOR),
                "selected_default_bytes": DEFAULT_RECOVERY_CAPACITY_BYTES,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
