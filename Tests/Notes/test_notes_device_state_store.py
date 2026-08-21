from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

from tldw_chatbook.Notes import notes_device_state_schema
from tldw_chatbook.Notes.notes_device_state_schema import (
    HISTORICAL_V1_IMPORT_LEDGER_DDL,
    LATEST_NOTES_DEVICE_SCHEMA_VERSION,
)
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateError,
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncLegacyMigrationRecord,
    NotesSyncOperationRecord,
    NotesSyncRecoveryRecord,
    NotesSyncRootRecord,
    NotesSyncStoreSetting,
)
from tldw_chatbook.Notes.note_import_receipts import NoteImportReceiptRepository
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncOperationState,
    NotesSyncRootState,
    NotesSyncSerializationProfile,
)


EXPECTED_TABLES = {
    "import_sessions",
    "import_items",
    "import_payload_effects",
    "import_folder_effects",
    "import_membership_effects",
    "notes_sync_roots",
    "notes_sync_bindings",
    "notes_sync_operations",
    "notes_sync_recovery",
    "notes_sync_legacy_migrations",
    "notes_sync_store_settings",
}
PINNED_HISTORICAL_V1_DDL_SHA256 = (
    "0f0be956444fd98b1c14dd7626d3d2aca2ac2746092db0c0502cf4d762f980bb"
)


def _tables(database: Path) -> set[str]:
    with sqlite3.connect(database) as connection:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }


def test_historical_v1_fixture_is_pinned_independently_of_current_bootstrap() -> None:
    assert (
        hashlib.sha256(HISTORICAL_V1_IMPORT_LEDGER_DDL.encode("utf-8")).hexdigest()
        == PINNED_HISTORICAL_V1_DDL_SHA256
    )


def _root(
    *,
    root_id: str = "root-1",
    state: NotesSyncRootState = NotesSyncRootState.PENDING,
    logical_folder_id: str | None = None,
) -> NotesSyncRootRecord:
    return NotesSyncRootRecord(
        root_id=root_id,
        note_scope_id="scope-1",
        logical_folder_id=logical_folder_id,
        canonical_path="/private/notes",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        state=state,
    )


def _binding(
    *,
    binding_id: str = "binding-1",
    root_id: str = "root-1",
    note_id: str = "note-1",
    relative_path: str = "folder/note.md",
    identity_digest: str = "a" * 64,
    state: NotesSyncBindingState = NotesSyncBindingState.ACTIVE,
) -> NotesSyncBindingRecord:
    return NotesSyncBindingRecord(
        binding_id=binding_id,
        root_id=root_id,
        note_scope_id="scope-1",
        note_id=note_id,
        normalized_relative_path=relative_path,
        stable_identity_digest=identity_digest,
        state=state,
        serialization=NotesSyncSerializationProfile(
            utf8_bom=False,
            newline="lf",
            final_newline=True,
            mode=0o600,
        ),
        content_digest="b" * 64,
        note_version=3,
    )


def test_empty_v0_initializes_current_schema_in_an_isolated_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    store = NotesDeviceStateStore(database)

    store.initialize()

    assert _tables(database) == EXPECTED_TABLES
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (
            LATEST_NOTES_DEVICE_SCHEMA_VERSION,
        )


def test_pinned_historical_v1_receipt_rows_survive_value_for_value_and_indexes_repair(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(HISTORICAL_V1_IMPORT_LEDGER_DDL)
        connection.execute(
            """
            INSERT INTO import_sessions (
                session_id, approval_id, plan_digest, state, batch_size,
                total_count, reason_code, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session-1",
                "00000000-0000-0000-0000-000000000001",
                "a" * 64,
                "completed",
                25,
                1,
                None,
                1,
                2,
            ),
        )
        connection.execute(
            """
            INSERT INTO import_items (
                session_id, item_id, source_locator_digest, selected_action,
                outcome_count, outcome, target_note_id, expected_version,
                observed_version, reason_code, retryable, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session-1",
                "item-1",
                "b" * 64,
                "update_existing",
                1,
                "updated",
                "note-1",
                4,
                5,
                None,
                0,
                1,
                2,
            ),
        )
        connection.execute(
            """
            INSERT INTO import_payload_effects (
                effect_id, session_id, item_id, payload_index, payload_digest,
                effect_kind, state, target_note_id, expected_version,
                observed_version, reason_code, retryable, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "payload-1",
                "session-1",
                "item-1",
                0,
                "c" * 64,
                "replace_content",
                "applied",
                "note-1",
                4,
                5,
                None,
                0,
                1,
                2,
            ),
        )
        connection.execute(
            """
            INSERT INTO import_folder_effects (
                effect_id, session_id, folder_ordinal, path_digest,
                parent_effect_id, effect_kind, state, target_folder_id,
                reason_code, retryable, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "folder-1",
                "session-1",
                0,
                "d" * 64,
                None,
                "ensure_folder",
                "applied",
                "target-folder-1",
                None,
                0,
                1,
                2,
            ),
        )
        connection.execute(
            """
            INSERT INTO import_membership_effects (
                effect_id, session_id, item_id, payload_index,
                membership_ordinal, folder_path_digest, effect_kind, state,
                target_note_id, target_folder_id, reason_code, retryable,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "membership-1",
                "session-1",
                "item-1",
                0,
                0,
                "d" * 64,
                "attach_membership",
                "applied",
                "note-1",
                "target-folder-1",
                None,
                0,
                1,
                2,
            ),
        )
        connection.execute("DROP INDEX idx_import_items_target")
        before = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in (
                "import_sessions",
                "import_items",
                "import_payload_effects",
                "import_folder_effects",
                "import_membership_effects",
            )
        }
        connection.commit()

    NotesDeviceStateStore(database).initialize()

    with sqlite3.connect(database) as connection:
        after = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in before
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'index'"
            )
        }
    assert after == before
    assert "idx_import_items_target" in indexes
    assert EXPECTED_TABLES <= _tables(database)
    snapshot = NoteImportReceiptRepository(database).load_session_snapshot(
        "00000000-0000-0000-0000-000000000001"
    )
    receipt = NoteImportReceiptRepository(database).aggregate_receipt(
        "00000000-0000-0000-0000-000000000001"
    )
    assert (
        snapshot.state.value,
        len(snapshot.items),
        len(snapshot.payload_effects),
    ) == (
        "completed",
        1,
        1,
    )
    assert (receipt.updated, receipt.failed, receipt.completed) == (1, 0, 1)


def test_current_reopen_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    store = NotesDeviceStateStore(database)
    store.initialize()
    with sqlite3.connect(database) as connection:
        schema_before = tuple(
            connection.execute(
                "SELECT type, name, sql FROM sqlite_schema ORDER BY type, name"
            )
        )

    store.initialize()

    with sqlite3.connect(database) as connection:
        schema_after = tuple(
            connection.execute(
                "SELECT type, name, sql FROM sqlite_schema ORDER BY type, name"
            )
        )
    assert schema_after == schema_before


def test_migration_failure_rolls_back_all_v2_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(HISTORICAL_V1_IMPORT_LEDGER_DDL)
        connection.commit()
    real_migration = notes_device_state_schema.migrate_v1_to_current

    def fail_after_migration(connection: sqlite3.Connection) -> None:
        real_migration(connection)
        raise RuntimeError("private injected failure")

    monkeypatch.setattr(
        notes_device_state_schema, "migrate_v1_to_current", fail_after_migration
    )

    with pytest.raises(NotesDeviceStateError) as caught:
        NotesDeviceStateStore(database).initialize()

    assert "private injected failure" not in str(caught.value)
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
    assert not ({"notes_sync_roots", "notes_sync_bindings"} & _tables(database))


def test_newer_schema_and_malformed_v1_fail_closed(tmp_path: Path) -> None:
    newer = tmp_path / "newer.sqlite3"
    with sqlite3.connect(newer) as connection:
        connection.execute(
            f"PRAGMA user_version = {LATEST_NOTES_DEVICE_SCHEMA_VERSION + 1}"
        )
    with pytest.raises(NotesDeviceStateError, match="Unsupported"):
        NotesDeviceStateStore(newer).initialize()

    malformed = tmp_path / "malformed.sqlite3"
    with sqlite3.connect(malformed) as connection:
        connection.executescript(
            "CREATE TABLE import_sessions (wrong TEXT); PRAGMA user_version = 1;"
        )
    with pytest.raises(NotesDeviceStateError, match="incompatible"):
        NotesDeviceStateStore(malformed).initialize()

    colliding = tmp_path / "colliding.sqlite3"
    with sqlite3.connect(colliding) as connection:
        connection.execute("CREATE TABLE notes_sync_roots (private_text TEXT)")
    with pytest.raises(NotesDeviceStateError, match="incompatible"):
        NotesDeviceStateStore(colliding).initialize()

    partial_v1 = tmp_path / "partial-v1.sqlite3"
    with sqlite3.connect(partial_v1) as connection:
        connection.executescript(HISTORICAL_V1_IMPORT_LEDGER_DDL)
        connection.execute("CREATE TABLE notes_sync_roots (root_id TEXT PRIMARY KEY)")
    with pytest.raises(NotesDeviceStateError, match="incompatible"):
        NotesDeviceStateStore(partial_v1).initialize()

    colliding_index = tmp_path / "colliding-index.sqlite3"
    with sqlite3.connect(colliding_index) as connection:
        connection.executescript(HISTORICAL_V1_IMPORT_LEDGER_DDL)
        connection.execute("DROP INDEX idx_import_items_target")
        connection.execute(
            "CREATE INDEX idx_import_items_target ON import_items(outcome)"
        )
    with pytest.raises(NotesDeviceStateError, match="incompatible"):
        NotesDeviceStateStore(colliding_index).initialize()


def test_active_root_requires_logical_folder_owner_and_root_transitions_fail_closed(
    tmp_path: Path,
) -> None:
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")

    with pytest.raises(sqlite3.IntegrityError):
        store.create_root(_root(state=NotesSyncRootState.ACTIVE))
    created = store.create_root(_root())
    assert created.state is NotesSyncRootState.PENDING
    with pytest.raises(NotesDeviceStateError, match="transition"):
        store.transition_root("root-1", NotesSyncRootState.ACTIVE)
    store.assign_root_folder("root-1", "folder-1")
    active = store.transition_root("root-1", NotesSyncRootState.ACTIVE)
    assert active.logical_folder_id == "folder-1"
    with pytest.raises(NotesDeviceStateError, match="transition"):
        store.transition_root("root-1", NotesSyncRootState.PENDING)


def test_active_binding_ownership_is_transactionally_unique(tmp_path: Path) -> None:
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")
    store.create_root(
        _root(logical_folder_id="folder-1", state=NotesSyncRootState.ACTIVE)
    )
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-2",
            note_scope_id="scope-1",
            logical_folder_id="folder-2",
            canonical_path="/private/other-notes",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.ACTIVE,
        )
    )
    store.create_binding(_binding())

    conflicts = (
        _binding(
            binding_id="binding-2",
            root_id="root-1",
            note_id="note-1",
            relative_path="other.md",
            identity_digest="c" * 64,
        ),
        _binding(
            binding_id="binding-3",
            root_id="root-1",
            note_id="note-2",
            relative_path="folder/note.md",
            identity_digest="d" * 64,
        ),
        _binding(
            binding_id="binding-4",
            root_id="root-1",
            note_id="note-3",
            relative_path="third.md",
            identity_digest="a" * 64,
        ),
        _binding(
            binding_id="binding-5",
            root_id="root-2",
            note_id="note-1",
            relative_path="other.md",
            identity_digest="e" * 64,
        ),
        _binding(
            binding_id="binding-6",
            root_id="root-2",
            note_id="note-6",
            relative_path="folder/note.md",
            identity_digest="a" * 64,
        ),
    )
    for conflict in conflicts:
        with pytest.raises(sqlite3.IntegrityError):
            store.create_binding(conflict)

    assert store.list_bindings("root-1") == (_binding(),)
    root_scoped_path = store.create_binding(
        _binding(
            binding_id="binding-7",
            root_id="root-2",
            note_id="note-7",
            relative_path="folder/note.md",
            identity_digest="f" * 64,
        )
    )
    assert root_scoped_path.root_id == "root-2"


def test_active_binding_requires_an_active_owned_root(tmp_path: Path) -> None:
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")
    store.create_root(_root())

    with pytest.raises(NotesDeviceStateError, match="active root"):
        store.create_binding(_binding())

    candidate = store.create_binding(_binding(state=NotesSyncBindingState.CANDIDATE))
    assert candidate.state is NotesSyncBindingState.CANDIDATE
    with pytest.raises(NotesDeviceStateError, match="active root"):
        store.transition_binding("binding-1", NotesSyncBindingState.ACTIVE)


def test_binding_scope_must_match_its_parent_root_transactionally(
    tmp_path: Path,
) -> None:
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")
    store.create_root(
        _root(logical_folder_id="folder-1", state=NotesSyncRootState.ACTIVE)
    )
    mismatched = NotesSyncBindingRecord(
        binding_id="binding-1",
        root_id="root-1",
        note_scope_id="other-scope",
        note_id="note-1",
        normalized_relative_path="note.md",
        stable_identity_digest="a" * 64,
        state=NotesSyncBindingState.ACTIVE,
        serialization=NotesSyncSerializationProfile(
            utf8_bom=False,
            newline="lf",
            final_newline=True,
            mode=0o600,
        ),
        content_digest="b" * 64,
        note_version=1,
    )

    with pytest.raises(NotesDeviceStateError, match="note scope"):
        store.create_binding(mismatched)

    assert store.list_bindings("root-1") == ()


def test_operation_binding_must_belong_to_the_same_root_transactionally(
    tmp_path: Path,
) -> None:
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")
    store.create_root(
        _root(logical_folder_id="folder-1", state=NotesSyncRootState.ACTIVE)
    )
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-2",
            note_scope_id="scope-2",
            logical_folder_id="folder-2",
            canonical_path="/private/other-notes",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.ACTIVE,
        )
    )
    store.create_binding(_binding())

    with pytest.raises(NotesDeviceStateError, match="same root"):
        store.create_operation(
            NotesSyncOperationRecord(
                operation_id="operation-1",
                root_id="root-2",
                binding_id="binding-1",
                kind="update_note",
                state=NotesSyncOperationState.PENDING,
                reason_code=None,
                observation_token="observation-1",
                expected_note_version=3,
                expected_file_digest="b" * 64,
            )
        )


def test_records_round_trip_and_illegal_binding_operation_transitions_fail_closed(
    tmp_path: Path,
) -> None:
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")
    store.create_root(
        _root(logical_folder_id="folder-1", state=NotesSyncRootState.ACTIVE)
    )
    binding = store.create_binding(_binding())
    operation = store.create_operation(
        NotesSyncOperationRecord(
            operation_id="operation-1",
            root_id="root-1",
            binding_id="binding-1",
            kind="update_note",
            state=NotesSyncOperationState.PENDING,
            reason_code=None,
            observation_token="observation-1",
            expected_note_version=3,
            expected_file_digest="b" * 64,
        )
    )
    recovery = NotesSyncRecoveryRecord(
        recovery_id="recovery-1",
        operation_id="operation-1",
        payload=b"private recovery bytes",
        metadata=b"private metadata",
        expires_at=500,
    )

    store.put_recovery(recovery)
    store.record_legacy_migration(
        NotesSyncLegacyMigrationRecord(
            migration_id="migration-1",
            source_fingerprint="c" * 64,
            state="pending_review",
            reason_code="review_required",
        )
    )
    store.set_setting(NotesSyncStoreSetting(key="recovery_capacity", value="1024"))

    assert store.get_binding("binding-1") == binding
    assert store.get_operation("operation-1") == operation
    assert store.load_recovery("recovery-1") == recovery
    assert store.get_setting("recovery_capacity") == NotesSyncStoreSetting(
        key="recovery_capacity", value="1024"
    )
    assert "private recovery bytes" not in repr(recovery)
    with pytest.raises(NotesDeviceStateError, match="transition"):
        store.transition_binding("binding-1", NotesSyncBindingState.CANDIDATE)
    with pytest.raises(NotesDeviceStateError, match="transition"):
        store.transition_operation("operation-1", NotesSyncOperationState.COMPLETED)


def test_root_cursor_latest_status_settings_and_public_projections_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")
    root = NotesSyncRootRecord(
        root_id="root-1",
        note_scope_id="scope-1",
        logical_folder_id="folder-1",
        canonical_path="/private/notes",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        state=NotesSyncRootState.ACTIVE,
        cursor="private-server-cursor",
        last_status_code="up_to_date",
    )
    store.create_root(root)
    store.create_binding(_binding())

    def reject_private_binding_materialization(*_args, **_kwargs):
        raise AssertionError("public projection materialized a private binding")

    monkeypatch.setattr(store, "list_bindings", reject_private_binding_materialization)
    root_summaries = store.list_root_summaries()
    binding_summaries = store.list_binding_summaries("root-1")
    assert root_summaries[0].root_id == "root-1"
    assert root_summaries[0].last_status_code == "up_to_date"
    assert binding_summaries[0].binding_id == "binding-1"
    for private_value in (
        "/private/notes",
        "private-server-cursor",
        "folder/note.md",
        "a" * 64,
        "b" * 64,
    ):
        assert private_value not in asdict(root_summaries[0]).values()
        assert private_value not in asdict(binding_summaries[0]).values()

    with pytest.raises(ValueError, match="setting"):
        store.set_setting(NotesSyncStoreSetting(key="unknown_setting", value="1"))
    with pytest.raises(ValueError, match="setting"):
        store.set_setting(
            NotesSyncStoreSetting(key="recovery_capacity", value="x" * 257)
        )


def test_read_only_connect_requires_existing_database_and_cannot_write(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    store = NotesDeviceStateStore(database)
    with pytest.raises(Exception):
        store._connect(read_only=True, must_exist=True)
    assert not database.exists()
    store.initialize()

    connection = store._connect(read_only=True, must_exist=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM notes_sync_roots")
    finally:
        connection.close()
