"""Contracts for redacted, paused Notes lasting-sync roots."""

from __future__ import annotations

import inspect
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from threading import Barrier

import pytest

from tldw_chatbook.Notes import notes_sync_state as sync_module
from tldw_chatbook.Notes import notes_sync_state_schema as schema_module
from tldw_chatbook.Notes.note_import_execution_models import (
    approve_note_import_plan,
)
from tldw_chatbook.Notes.note_import_plan_models import ImportBounds, NoteImportPlan
from tldw_chatbook.Notes.note_import_receipts import NoteImportReceiptRepository
from tldw_chatbook.Notes.notes_sync_state import (
    MAX_SYNC_ROOTS,
    NotesSyncStateError,
    NotesSyncStateRepository,
    SyncRootRecord,
    SyncRootState,
    SyncStateCapacityError,
    SyncStateConflictError,
    SyncStateCorruptionError,
)
from tldw_chatbook.Notes.notes_sync_state_schema import (
    notes_sync_state_transaction,
)


_DIRECTIONS = ("folder_to_notes", "notes_to_folder", "bidirectional")
_PRIVATE_PATH = "~/Private/../秘密\\mixed//é"
_PRIVATE_ID = "private-root-id"
_PRIVATE_DIGEST = "a" * 64
_PRIVATE_MIGRATION_ID = "00000000-0000-4000-8000-000000000097"
_RAW_SQLITE_SENTINEL = "sqlite leaked /private/alice/root"
_MIN_SQLITE_INTEGER = -(2**63)
_MAX_SQLITE_INTEGER = 2**63 - 1


def _repository(tmp_path: Path) -> NotesSyncStateRepository:
    return NotesSyncStateRepository(tmp_path / "notes-sync-state.sqlite3")


def _approved_empty_plan():
    return approve_note_import_plan(
        NoteImportPlan(
            bounds=ImportBounds(
                max_files=1,
                max_file_bytes=1,
                max_total_bytes=1,
                max_depth=0,
            ),
            items=(),
            proposed_folder_paths=(),
        ),
        approval_id="00000000-0000-4000-8000-000000000098",
    )


def _assert_exact_v2(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }
    assert {
        "import_sessions",
        "sync_roots",
        "sync_bindings",
        "sync_migration_runs",
        "sync_migration_items",
    } <= tables


def test_public_root_api_is_narrow_and_exports_required_types() -> None:
    assert {
        "MAX_SYNC_ROOTS",
        "NotesSyncStateError",
        "NotesSyncStateRepository",
        "SyncRootRecord",
        "SyncRootState",
        "SyncStateCapacityError",
        "SyncStateConflictError",
        "SyncStateCorruptionError",
    } <= set(sync_module.__all__)
    public_methods = {
        name
        for name, method in inspect.getmembers(
            NotesSyncStateRepository, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    assert public_methods == {
        "create_candidate_root",
        "disconnect_root",
        "get_root",
        "list_roots",
        "pause_root",
        "update_candidate_root",
    }
    assert "set_state" not in public_methods
    assert {state.value for state in SyncRootState} == {
        "candidate",
        "paused",
        "disconnected",
    }


def test_root_projection_maps_every_column_and_redacts_private_values(
    tmp_path: Path,
) -> None:
    database = tmp_path / "projection.sqlite3"
    with notes_sync_state_transaction(database, immediate=True) as connection:
        connection.execute(
            """INSERT INTO sync_migration_runs (
                   migration_id, source_kind, source_revision_before,
                   source_revision_after, state, created_at, updated_at
               ) VALUES (?, 'legacy_notes_sync_v1', ?, ?, 'matched_recheck', 10, 11)""",
            (_PRIVATE_MIGRATION_ID, "b" * 64, "b" * 64),
        )
        connection.execute(
            """INSERT INTO sync_roots (
                   root_id, lexical_root_path, display_name, direction, state,
                   row_version, needs_rescan, reason_code, source_kind,
                   source_locator_digest, source_migration_id, created_at, updated_at
               ) VALUES (?, ?, 'Review me', 'unspecified', 'paused', 7, 1,
                         'legacy_direction_invalid', 'legacy_notes_sync_v1', ?, ?, 12, 13)""",
            (
                _PRIVATE_ID,
                _PRIVATE_PATH,
                _PRIVATE_DIGEST,
                _PRIVATE_MIGRATION_ID,
            ),
        )

    record = NotesSyncStateRepository(database).get_root(_PRIVATE_ID)

    assert tuple(field.name for field in fields(SyncRootRecord)) == (
        "root_id",
        "lexical_root_path",
        "display_name",
        "direction",
        "state",
        "row_version",
        "needs_rescan",
        "reason_code",
        "source_kind",
        "source_locator_digest",
        "source_migration_id",
        "created_at",
        "updated_at",
    )
    assert record == SyncRootRecord(
        root_id=_PRIVATE_ID,
        lexical_root_path=_PRIVATE_PATH,
        display_name="Review me",
        direction="unspecified",
        state=SyncRootState.PAUSED,
        row_version=7,
        needs_rescan=True,
        reason_code="legacy_direction_invalid",
        source_kind="legacy_notes_sync_v1",
        source_locator_digest=_PRIVATE_DIGEST,
        source_migration_id=_PRIVATE_MIGRATION_ID,
        created_at=12,
        updated_at=13,
    )
    assert SyncRootRecord.__dataclass_params__.frozen is True
    assert hasattr(record, "__slots__")
    with pytest.raises(FrozenInstanceError):
        record.display_name = "changed"  # type: ignore[misc]
    rendered = repr(record)
    assert rendered == (
        "SyncRootRecord(state='paused', row_version=7, needs_rescan=True, "
        "reason_code='legacy_direction_invalid')"
    )
    for private in (
        _PRIVATE_ID,
        _PRIVATE_PATH,
        _PRIVATE_DIGEST,
        _PRIVATE_MIGRATION_ID,
        "legacy_notes_sync_v1",
    ):
        assert private not in rendered


def test_repository_repr_and_errors_never_disclose_private_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    assert repr(repository) == "NotesSyncStateRepository(<private>)"
    assert str(tmp_path) not in repr(repository)

    with pytest.raises(ValueError) as invalid_path:
        repository.create_candidate_root(
            _PRIVATE_PATH + "\x00secret", "Private", "bidirectional"
        )
    with pytest.raises(ValueError) as invalid_id:
        repository.get_root(_PRIVATE_ID + "\x00secret")

    @contextmanager
    def fail_transaction(*_args: object, **_kwargs: object):
        raise sqlite3.OperationalError(_RAW_SQLITE_SENTINEL)
        yield  # pragma: no cover

    monkeypatch.setattr(sync_module, "notes_sync_state_transaction", fail_transaction)
    with pytest.raises(NotesSyncStateError) as sqlite_failure:
        repository.list_roots()

    for caught in (invalid_path.value, invalid_id.value, sqlite_failure.value):
        message = str(caught)
        assert len(message) <= 160
        for private in (
            _PRIVATE_PATH,
            _PRIVATE_ID,
            _PRIVATE_DIGEST,
            _PRIVATE_MIGRATION_ID,
            _RAW_SQLITE_SENTINEL,
        ):
            assert private not in message
    assert sqlite_failure.value.__context__ is None


def test_create_get_and_list_manual_candidate_roots_preserve_lexical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)

    def filesystem_path_use(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("candidate path reached pathlib")

    monkeypatch.setattr(sync_module, "Path", filesystem_path_use)
    first = repository.create_candidate_root(
        _PRIVATE_PATH, "Private root", "bidirectional"
    )
    second = repository.create_candidate_root(
        "relative/../Second", "Second root", "folder_to_notes"
    )

    assert first.lexical_root_path == _PRIVATE_PATH
    assert first.display_name == "Private root"
    assert first.direction == "bidirectional"
    assert first.state is SyncRootState.CANDIDATE
    assert first.row_version == 1
    assert first.needs_rescan is True
    assert first.reason_code is None
    assert first.source_kind is None
    assert first.source_locator_digest is None
    assert first.source_migration_id is None
    assert first.created_at > 0
    assert first.updated_at == first.created_at
    assert repository.get_root(first.root_id) == first
    assert repository.list_roots() == (first, second)


@pytest.mark.parametrize(
    ("lexical_root_path", "display_name", "direction", "error_type"),
    (
        (None, "Root", "bidirectional", TypeError),
        ("root", None, "bidirectional", TypeError),
        ("root", "Root", None, TypeError),
        ("", "Root", "bidirectional", ValueError),
        ("root", "", "bidirectional", ValueError),
        ("root\x00private", "Root", "bidirectional", ValueError),
        ("root", "Root\x00private", "bidirectional", ValueError),
        ("x" * 32_769, "Root", "bidirectional", ValueError),
        ("root", "x" * 256, "bidirectional", ValueError),
        ("root", "Root", "unspecified", ValueError),
        ("root", "Root", "active", ValueError),
        ("root", "Root", "running", ValueError),
        ("root", "Root", "x" * 65, ValueError),
    ),
)
def test_create_root_rejects_invalid_manual_values_without_opening_database(
    tmp_path: Path,
    lexical_root_path: object,
    display_name: object,
    direction: object,
    error_type: type[Exception],
) -> None:
    database = tmp_path / "invalid.sqlite3"
    with pytest.raises(error_type):
        NotesSyncStateRepository(database).create_candidate_root(
            lexical_root_path,  # type: ignore[arg-type]
            display_name,  # type: ignore[arg-type]
            direction,  # type: ignore[arg-type]
        )
    assert not database.exists()


@pytest.mark.parametrize(
    ("operation", "value", "error_type"),
    (
        ("get", None, TypeError),
        ("get", "", ValueError),
        ("get", "x\x00y", ValueError),
        ("get", "x" * 257, ValueError),
        ("update_version", True, TypeError),
        ("update_version", 0, ValueError),
        ("update_version", 2**63, ValueError),
        ("reason", None, TypeError),
        ("reason", "", ValueError),
        ("reason", "Uppercase", ValueError),
        ("reason", "a" * 65, ValueError),
    ),
)
def test_root_operations_reject_invalid_ids_versions_and_reasons(
    tmp_path: Path,
    operation: str,
    value: object,
    error_type: type[Exception],
) -> None:
    repository = _repository(tmp_path)
    root = repository.create_candidate_root("root", "Root", "bidirectional")
    with pytest.raises(error_type):
        if operation == "get":
            repository.get_root(value)  # type: ignore[arg-type]
        elif operation == "update_version":
            repository.update_candidate_root(
                root.root_id,
                value,
                display_name="Changed",  # type: ignore[arg-type]
            )
        else:
            repository.pause_root(root.root_id, root.row_version, value)  # type: ignore[arg-type]


def test_update_candidate_root_uses_exact_version_and_advances_projection(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    original = repository.create_candidate_root("root", "Original", "folder_to_notes")
    updated = repository.update_candidate_root(
        original.root_id,
        original.row_version,
        display_name="Changed",
        direction="notes_to_folder",
    )

    assert updated.display_name == "Changed"
    assert updated.direction == "notes_to_folder"
    assert updated.row_version == original.row_version + 1
    assert updated.updated_at > original.updated_at
    assert updated.created_at == original.created_at
    with pytest.raises(SyncStateConflictError):
        repository.update_candidate_root(
            original.root_id,
            original.row_version,
            display_name="Stale write",
        )
    assert repository.get_root(original.root_id) == updated
    with pytest.raises(ValueError):
        repository.update_candidate_root(original.root_id, updated.row_version)


def test_pause_and_disconnect_are_named_terminal_operations(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    original = repository.create_candidate_root("root", "Root", "bidirectional")
    paused = repository.pause_root(
        original.root_id, original.row_version, "awaiting_review"
    )
    assert paused.state is SyncRootState.PAUSED
    assert paused.reason_code == "awaiting_review"
    assert paused.row_version == original.row_version + 1
    assert paused.updated_at > original.updated_at

    disconnected = repository.disconnect_root(paused.root_id, paused.row_version)
    assert disconnected.state is SyncRootState.DISCONNECTED
    assert disconnected.row_version == paused.row_version + 1
    assert disconnected.updated_at > paused.updated_at
    for operation in (
        lambda: repository.update_candidate_root(
            disconnected.root_id,
            disconnected.row_version,
            display_name="Reopened",
        ),
        lambda: repository.pause_root(
            disconnected.root_id,
            disconnected.row_version,
            "reopen_attempt",
        ),
        lambda: repository.disconnect_root(
            disconnected.root_id, disconnected.row_version
        ),
    ):
        with pytest.raises(SyncStateConflictError):
            operation()


def test_disconnect_root_rolls_back_child_changes_on_stale_version(
    tmp_path: Path,
) -> None:
    database = tmp_path / "atomic-disconnect.sqlite3"
    repository = NotesSyncStateRepository(database)
    root = repository.create_candidate_root("root", "Root", "bidirectional")
    with notes_sync_state_transaction(database, immediate=True) as connection:
        connection.execute(
            """INSERT INTO sync_bindings (
                   binding_id, root_id, note_id, lexical_relative_path, path_key,
                   state, row_version, needs_rescan, reason_code, source_kind,
                   source_locator_digest, source_migration_id, created_at, updated_at
               ) VALUES ('binding-1', ?, 'note-1', 'note.md', NULL,
                         'candidate', 1, 1, NULL, NULL, NULL, NULL, 1, 1)""",
            (root.root_id,),
        )
    repository.update_candidate_root(
        root.root_id, root.row_version, display_name="Version two"
    )

    with pytest.raises(SyncStateConflictError):
        repository.disconnect_root(root.root_id, root.row_version)

    with notes_sync_state_transaction(database) as connection:
        child = connection.execute(
            "SELECT state, row_version FROM sync_bindings WHERE binding_id = 'binding-1'"
        ).fetchone()
    assert child == ("candidate", 1)


def test_disconnect_root_version_bumps_all_live_children(tmp_path: Path) -> None:
    database = tmp_path / "child-disconnect.sqlite3"
    repository = NotesSyncStateRepository(database)
    root = repository.create_candidate_root("root", "Root", "bidirectional")
    with notes_sync_state_transaction(database, immediate=True) as connection:
        connection.executemany(
            """INSERT INTO sync_bindings (
                   binding_id, root_id, note_id, lexical_relative_path, path_key,
                   state, row_version, needs_rescan, reason_code, source_kind,
                   source_locator_digest, source_migration_id, created_at, updated_at
               ) VALUES (?, ?, ?, ?, NULL, ?, ?, 1, NULL, NULL, NULL, NULL, 1, ?)""",
            (
                ("binding-1", root.root_id, "note-1", "one.md", "candidate", 2, 2),
                (
                    "binding-2",
                    root.root_id,
                    "note-2",
                    "two.md",
                    "needs_attention",
                    4,
                    4,
                ),
                ("binding-3", root.root_id, "note-3", "three.md", "disconnected", 6, 6),
            ),
        )

    repository.disconnect_root(root.root_id, root.row_version)

    with notes_sync_state_transaction(database) as connection:
        children = connection.execute(
            """SELECT binding_id, state, row_version, updated_at
               FROM sync_bindings ORDER BY binding_id"""
        ).fetchall()
    assert children[0][0:3] == ("binding-1", "disconnected", 3)
    assert children[0][3] > 2
    assert children[1][0:3] == ("binding-2", "disconnected", 5)
    assert children[1][3] > 4
    assert children[2] == ("binding-3", "disconnected", 6, 6)


def test_disconnect_root_advances_newer_child_timestamps_during_clock_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "child-timestamp.sqlite3"
    repository = NotesSyncStateRepository(database)
    root = repository.create_candidate_root("root", "Root", "bidirectional")
    child_updated_at = 2**62
    with notes_sync_state_transaction(database, immediate=True) as connection:
        connection.execute(
            """INSERT INTO sync_bindings (
                   binding_id, root_id, note_id, lexical_relative_path, path_key,
                   state, row_version, needs_rescan, reason_code, source_kind,
                   source_locator_digest, source_migration_id, created_at, updated_at
               ) VALUES ('binding-1', ?, 'note-1', 'note.md', NULL,
                         'candidate', 3, 1, NULL, NULL, NULL, NULL, 1, ?)""",
            (root.root_id, child_updated_at),
        )
    monkeypatch.setattr(sync_module.time, "time_ns", lambda: 1)

    disconnected = repository.disconnect_root(root.root_id, root.row_version)

    with notes_sync_state_transaction(database) as connection:
        child = connection.execute(
            """SELECT state, row_version, updated_at
               FROM sync_bindings WHERE binding_id = 'binding-1'"""
        ).fetchone()
    assert child == ("disconnected", 4, child_updated_at + 1)
    assert disconnected.updated_at == child_updated_at + 1


@pytest.mark.parametrize(
    ("column", "malformed"),
    (
        ("root_id", ""),
        ("root_id", "private\x00root"),
        ("lexical_root_path", ""),
        ("lexical_root_path", "private\x00path"),
        ("display_name", ""),
        ("display_name", "private\x00name"),
        ("direction", "active"),
        ("state", "running"),
        ("row_version", 0),
        ("needs_rescan", 2),
        ("reason_code", "Private-Reason"),
        ("source_kind", "private_source"),
        ("source_locator_digest", "private-digest"),
        ("created_at", 0),
        ("updated_at", 0),
    ),
)
def test_root_projection_rejects_noncanonical_durable_rows_without_disclosure(
    tmp_path: Path,
    column: str,
    malformed: object,
) -> None:
    database = tmp_path / f"corrupt-{column}.sqlite3"
    repository = NotesSyncStateRepository(database)
    root = repository.create_candidate_root("root", "Root", "bidirectional")
    with notes_sync_state_transaction(database, immediate=True) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            f"UPDATE sync_roots SET {column} = ? WHERE root_id = ?",
            (malformed, root.root_id),
        )

    with pytest.raises(SyncStateCorruptionError) as caught:
        repository.list_roots()
    if isinstance(malformed, str) and malformed:
        assert malformed not in str(caught.value)
    assert len(str(caught.value)) <= 160


def test_root_projection_rejects_nul_source_migration_id_without_disclosure(
    tmp_path: Path,
) -> None:
    database = tmp_path / "corrupt-source-migration.sqlite3"
    repository = NotesSyncStateRepository(database)
    root = repository.create_candidate_root("root", "Root", "bidirectional")
    malformed_migration_id = "m" * 17 + "\x00" + "m" * 18
    with notes_sync_state_transaction(database, immediate=True) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """INSERT INTO sync_migration_runs (
                   migration_id, source_kind, source_revision_before,
                   state, created_at, updated_at
               ) VALUES (?, 'legacy_notes_sync_v1', ?, 'pending_recheck', 1, 1)""",
            (malformed_migration_id, "a" * 64),
        )
        connection.execute(
            """UPDATE sync_roots
               SET source_kind = 'legacy_notes_sync_v1',
                   source_locator_digest = ?, source_migration_id = ?
               WHERE root_id = ?""",
            ("b" * 64, malformed_migration_id, root.root_id),
        )

    with pytest.raises(SyncStateCorruptionError) as caught:
        repository.get_root(root.root_id)
    assert malformed_migration_id not in str(caught.value)
    assert len(str(caught.value)) <= 160


def test_root_corruption_error_chain_redacts_private_durable_value(
    tmp_path: Path,
) -> None:
    database = tmp_path / "corrupt-private-chain.sqlite3"
    repository = NotesSyncStateRepository(database)
    root = repository.create_candidate_root("root", "Root", "bidirectional")
    private_sentinel = "private_state_sentinel"
    with notes_sync_state_transaction(database, immediate=True) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE sync_roots SET state = ? WHERE root_id = ?",
            (private_sentinel, root.root_id),
        )

    with pytest.raises(SyncStateCorruptionError) as caught:
        repository.get_root(root.root_id)

    pending: list[BaseException] = [caught.value]
    chain: list[BaseException] = []
    while pending:
        error = pending.pop()
        if error in chain:
            continue
        chain.append(error)
        if error.__context__ is not None:
            pending.append(error.__context__)
        if error.__cause__ is not None:
            pending.append(error.__cause__)
    assert private_sentinel not in "\n".join(str(error) for error in chain)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("clock_value", "expected_timestamp"),
    (
        (_MIN_SQLITE_INTEGER, 1),
        (0, 1),
        (1, 1),
        (_MAX_SQLITE_INTEGER, _MAX_SQLITE_INTEGER),
    ),
)
def test_root_creation_bounds_in_range_clock_to_valid_sqlite_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock_value: int,
    expected_timestamp: int,
) -> None:
    monkeypatch.setattr(sync_module.time, "time_ns", lambda: clock_value)

    root = _repository(tmp_path).create_candidate_root("root", "Root", "bidirectional")

    assert root.created_at == expected_timestamp
    assert root.updated_at == expected_timestamp


@pytest.mark.parametrize(
    "clock_value",
    (_MIN_SQLITE_INTEGER - 1, _MAX_SQLITE_INTEGER + 1),
)
def test_root_creation_rejects_out_of_range_clock_without_opening_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock_value: int,
) -> None:
    database = tmp_path / "out-of-range-clock.sqlite3"
    monkeypatch.setattr(sync_module.time, "time_ns", lambda: clock_value)

    with pytest.raises(NotesSyncStateError) as caught:
        NotesSyncStateRepository(database).create_candidate_root(
            "root", "Root", "bidirectional"
        )

    assert str(clock_value) not in str(caught.value)
    assert len(str(caught.value)) <= 160
    assert caught.value.__context__ is None
    assert not database.exists()


def test_root_update_rejects_unadvanceable_maximum_timestamp_with_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    monkeypatch.setattr(sync_module.time, "time_ns", lambda: _MAX_SQLITE_INTEGER)
    root = repository.create_candidate_root("root", "Root", "bidirectional")
    monkeypatch.setattr(sync_module.time, "time_ns", lambda: 1)

    with pytest.raises(NotesSyncStateError) as caught:
        repository.update_candidate_root(
            root.root_id, root.row_version, display_name="Changed"
        )

    assert str(_MAX_SQLITE_INTEGER) not in str(caught.value)
    assert len(str(caught.value)) <= 160
    assert caught.value.__context__ is None
    assert repository.get_root(root.root_id) == root


def test_missing_root_errors_are_typed_bounded_and_redacted(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    missing = "private-missing-root"
    for operation in (
        lambda: repository.get_root(missing),
        lambda: repository.update_candidate_root(missing, 1, display_name="Changed"),
        lambda: repository.pause_root(missing, 1, "awaiting_review"),
        lambda: repository.disconnect_root(missing, 1),
    ):
        with pytest.raises(NotesSyncStateError) as caught:
            operation()
        assert missing not in str(caught.value)
        assert len(str(caught.value)) <= 160


def test_live_root_capacity_is_exact_atomic_and_released_by_disconnect(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    roots = tuple(
        repository.create_candidate_root(
            f"root-{index}", f"Root {index}", "bidirectional"
        )
        for index in range(MAX_SYNC_ROOTS)
    )
    assert len(repository.list_roots()) == 64

    with pytest.raises(SyncStateCapacityError) as caught:
        repository.create_candidate_root("root-65", "Root 65", "bidirectional")
    assert "64" in str(caught.value)
    assert len(repository.list_roots()) == 64

    repository.disconnect_root(roots[0].root_id, roots[0].row_version)
    replacement = repository.create_candidate_root(
        "replacement", "Replacement", "bidirectional"
    )
    all_roots = repository.list_roots()
    assert len(all_roots) == 65
    assert sum(root.state is not SyncRootState.DISCONNECTED for root in all_roots) == 64
    assert replacement.state is SyncRootState.CANDIDATE


def test_concurrent_creates_at_capacity_admit_exactly_one_root(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    for index in range(MAX_SYNC_ROOTS - 1):
        repository.create_candidate_root(
            f"root-{index}", f"Root {index}", "bidirectional"
        )
    barrier = Barrier(2)

    def create(index: int) -> SyncRootRecord:
        barrier.wait(timeout=5)
        return NotesSyncStateRepository(
            tmp_path / "notes-sync-state.sqlite3"
        ).create_candidate_root(
            f"racing-root-{index}", f"Racing root {index}", "bidirectional"
        )

    outcomes: list[SyncRootRecord | SyncStateCapacityError] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(create, 1), executor.submit(create, 2))
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except SyncStateCapacityError as error:
                outcomes.append(error)

    assert sum(isinstance(value, SyncRootRecord) for value in outcomes) == 1
    assert sum(isinstance(value, SyncStateCapacityError) for value in outcomes) == 1
    assert (
        sum(
            root.state is not SyncRootState.DISCONNECTED
            for root in repository.list_roots()
        )
        == MAX_SYNC_ROOTS
    )


def test_every_root_mutation_requests_an_immediate_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TransactionObserved(Exception):
        pass

    observed: list[bool] = []

    @contextmanager
    def observe_transaction(
        _database_path: Path,
        *,
        immediate: bool = False,
    ):
        observed.append(immediate)
        raise TransactionObserved
        yield  # pragma: no cover

    monkeypatch.setattr(sync_module, "_repository_transaction", observe_transaction)
    repository = _repository(tmp_path)
    operations = (
        lambda: repository.create_candidate_root("root", "Root", "bidirectional"),
        lambda: repository.update_candidate_root("root", 1, display_name="Changed"),
        lambda: repository.pause_root("root", 1, "awaiting_review"),
        lambda: repository.disconnect_root("root", 1),
    )
    for operation in operations:
        with pytest.raises(TransactionObserved):
            operation()
    assert observed == [True, True, True, True]


def test_public_methods_return_only_typed_records(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    created = repository.create_candidate_root("root", "Root", "bidirectional")
    values = (
        created,
        repository.get_root(created.root_id),
        repository.list_roots(),
        repository.update_candidate_root(
            created.root_id, created.row_version, display_name="Changed"
        ),
    )
    assert all(
        not isinstance(value, (sqlite3.Row, sqlite3.Connection)) for value in values
    )
    assert isinstance(values[2], tuple)
    assert all(isinstance(record, SyncRootRecord) for record in values[2])


@pytest.mark.parametrize("receipt_first", (True, False))
def test_repository_initialization_order_keeps_both_repositories_usable(
    tmp_path: Path,
    receipt_first: bool,
) -> None:
    database = tmp_path / f"order-{receipt_first}.sqlite3"
    receipts = NoteImportReceiptRepository(database)
    sync = NotesSyncStateRepository(database)
    approved = _approved_empty_plan()

    if receipt_first:
        receipt = receipts.begin(approved, batch_size=1)
        root = sync.create_candidate_root("root", "Root", "bidirectional")
    else:
        root = sync.create_candidate_root("root", "Root", "bidirectional")
        receipt = receipts.begin(approved, batch_size=1)

    assert receipts.get_session(approved.approval_id) == receipt
    assert sync.get_root(root.root_id) == root
    _assert_exact_v2(database)


def test_concurrent_repository_initialization_and_writes_converge(
    tmp_path: Path,
) -> None:
    database = tmp_path / "concurrent.sqlite3"
    schema_module.connect_private_sqlite("notes.sync_state", database).close()
    barrier = Barrier(2)
    approved = _approved_empty_plan()

    def write_receipt() -> object:
        repository = NoteImportReceiptRepository(database)
        barrier.wait(timeout=5)
        return repository.begin(approved, batch_size=1)

    def write_root() -> SyncRootRecord:
        repository = NotesSyncStateRepository(database)
        barrier.wait(timeout=5)
        return repository.create_candidate_root(
            "concurrent-root", "Concurrent root", "bidirectional"
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipt_future = executor.submit(write_receipt)
        root_future = executor.submit(write_root)
        receipt = receipt_future.result(timeout=10)
        root = root_future.result(timeout=10)

    receipts = NoteImportReceiptRepository(database)
    sync = NotesSyncStateRepository(database)
    assert receipts.get_session(approved.approval_id) == receipt
    assert sync.get_root(root.root_id) == root
    assert (
        sync.create_candidate_root(
            "after-contention", "After contention", "bidirectional"
        ).state
        is SyncRootState.CANDIDATE
    )
    _assert_exact_v2(database)


def test_sync_state_exception_hierarchy_is_public_and_specific() -> None:
    assert issubclass(SyncStateConflictError, NotesSyncStateError)
    assert issubclass(SyncStateCapacityError, NotesSyncStateError)
    assert issubclass(SyncStateCorruptionError, NotesSyncStateError)
