from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import closing
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


def test_nonempty_unversioned_database_is_not_adopted_or_changed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated_private_owner (value TEXT)")
        connection.execute("INSERT INTO unrelated_private_owner VALUES ('sentinel')")
        connection.commit()
    database.chmod(0o600)
    before = database.read_bytes()

    with pytest.raises(NotesDeviceStateError, match="incompatible"):
        NotesDeviceStateStore(database).initialize()

    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert connection.execute(
            "SELECT value FROM unrelated_private_owner"
        ).fetchone() == ("sentinel",)
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_schema WHERE name LIKE 'notes_sync_%'"
        ).fetchone() == (0,)


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
        queries = {
            "import_sessions": "SELECT * FROM import_sessions",
            "import_items": "SELECT * FROM import_items",
            "import_payload_effects": "SELECT * FROM import_payload_effects",
            "import_folder_effects": "SELECT * FROM import_folder_effects",
            "import_membership_effects": "SELECT * FROM import_membership_effects",
        }
        before = {
            table: connection.execute(query).fetchall()
            for table, query in queries.items()
        }
        connection.commit()

    NotesDeviceStateStore(database).initialize()

    with sqlite3.connect(database) as connection:
        after = {
            table: connection.execute(query).fetchall()
            for table, query in queries.items()
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


def test_root_pause_and_disconnect_propagate_child_binding_lifecycle_atomically(
    tmp_path: Path,
) -> None:
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
    store.create_binding(
        _binding(
            binding_id="binding-attention",
            note_id="note-attention",
            relative_path="attention.md",
            identity_digest="c" * 64,
            state=NotesSyncBindingState.NEEDS_ATTENTION,
        )
    )
    store.create_binding(
        _binding(
            binding_id="binding-candidate",
            note_id="note-candidate",
            relative_path="candidate.md",
            identity_digest="d" * 64,
            state=NotesSyncBindingState.CANDIDATE,
        )
    )

    paused = store.transition_root("root-1", NotesSyncRootState.PAUSED)

    assert paused.state is NotesSyncRootState.PAUSED
    assert store.get_binding("binding-1").state is NotesSyncBindingState.PAUSED
    assert (
        store.get_binding("binding-attention").state
        is NotesSyncBindingState.NEEDS_ATTENTION
    )
    assert (
        store.get_binding("binding-candidate").state is NotesSyncBindingState.CANDIDATE
    )
    released = store.create_binding(
        _binding(
            binding_id="binding-replacement",
            root_id="root-2",
            note_id="note-1",
            relative_path="replacement.md",
            identity_digest="a" * 64,
        )
    )
    assert released.root_id == "root-2"

    disconnected = store.transition_root("root-1", NotesSyncRootState.DISCONNECTED)

    assert disconnected.state is NotesSyncRootState.DISCONNECTED
    assert {binding.state for binding in store.list_bindings("root-1")} == {
        NotesSyncBindingState.DISCONNECTED
    }


def test_root_and_child_lifecycle_propagation_rolls_back_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    store = NotesDeviceStateStore(database)
    store.create_root(
        _root(logical_folder_id="folder-1", state=NotesSyncRootState.ACTIVE)
    )
    store.create_binding(_binding())
    original_connect = store._connect

    def rejecting_connect(**kwargs):
        connection = original_connect(**kwargs)

        def authorize(
            action: int,
            table: str | None,
            _column: str | None,
            _database: str | None,
            _trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_UPDATE and table == "notes_sync_roots":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(store, "_connect", rejecting_connect)
    store.close()  # force the next operation to reconnect through the seam

    with pytest.raises(sqlite3.DatabaseError):
        store.transition_root("root-1", NotesSyncRootState.PAUSED)

    monkeypatch.setattr(store, "_connect", original_connect)
    store.close()
    assert store.get_root("root-1").state is NotesSyncRootState.ACTIVE
    assert store.get_binding("binding-1").state is NotesSyncBindingState.ACTIVE


@pytest.mark.parametrize("extra_kind", ["table", "trigger"])
def test_v1_migration_rejects_unexpected_user_objects_without_changes(
    tmp_path: Path,
    extra_kind: str,
) -> None:
    database = tmp_path / f"v1-extra-{extra_kind}.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(HISTORICAL_V1_IMPORT_LEDGER_DDL)
        if extra_kind == "table":
            connection.execute("CREATE TABLE unexpected_private (value TEXT)")
        else:
            connection.execute(
                """
                CREATE TRIGGER unexpected_private
                AFTER INSERT ON import_sessions
                BEGIN
                    SELECT 1;
                END
                """
            )
        connection.commit()
    database.chmod(0o600)
    before = database.read_bytes()

    with pytest.raises(NotesDeviceStateError, match="incompatible"):
        NotesDeviceStateStore(database).initialize()

    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert connection.execute(
            "SELECT type FROM sqlite_schema WHERE name = 'unexpected_private'"
        ).fetchone() == (extra_kind,)


@pytest.mark.parametrize("extra_kind", ["table", "trigger", "index"])
def test_current_schema_rejects_unexpected_user_objects_without_changes(
    tmp_path: Path,
    extra_kind: str,
) -> None:
    database = tmp_path / f"current-extra-{extra_kind}.sqlite3"
    store = NotesDeviceStateStore(database)
    store.initialize()
    with sqlite3.connect(database) as connection:
        if extra_kind == "table":
            connection.execute("CREATE TABLE unexpected_private (value TEXT)")
        elif extra_kind == "trigger":
            connection.execute(
                """
                CREATE TRIGGER unexpected_private
                AFTER INSERT ON import_sessions
                BEGIN
                    SELECT 1;
                END
                """
            )
        else:
            connection.execute(
                "CREATE INDEX unexpected_private ON import_sessions(state)"
            )
        connection.commit()
    database.chmod(0o600)
    before = database.read_bytes()

    with pytest.raises(NotesDeviceStateError, match="incompatible"):
        store.initialize()

    assert database.read_bytes() == before


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


@pytest.mark.parametrize(
    ("column", "corrupt_value"),
    [
        ("payload", 4),
        ("payload", "not-a-blob"),
        ("metadata", 4),
        ("metadata", "not-a-blob"),
    ],
)
def test_recovery_reads_reject_non_blob_storage_without_coercion(
    tmp_path: Path,
    column: str,
    corrupt_value: object,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    store = NotesDeviceStateStore(database)
    store.create_root(
        _root(logical_folder_id="folder-1", state=NotesSyncRootState.ACTIVE)
    )
    store.create_operation(
        NotesSyncOperationRecord(
            operation_id="operation-1",
            root_id="root-1",
            binding_id=None,
            kind="update_note",
            state=NotesSyncOperationState.PENDING,
            reason_code=None,
            observation_token="observation-1",
            expected_note_version=3,
            expected_file_digest="b" * 64,
        )
    )
    store.put_recovery(
        NotesSyncRecoveryRecord(
            recovery_id="recovery-1",
            operation_id="operation-1",
            payload=b"payload",
            metadata=b"metadata",
            expires_at=2_000_000_000,
        )
    )
    assert column in {"payload", "metadata"}
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            f"UPDATE notes_sync_recovery SET {column} = ? WHERE recovery_id = ?",
            (corrupt_value, "recovery-1"),
        )
        connection.commit()

    with pytest.raises(NotesDeviceStateError, match="corrupt"):
        store.load_recovery("recovery-1")


@pytest.mark.parametrize(
    "state",
    [
        NotesSyncOperationState.RECOVERY_ADMITTED,
        NotesSyncOperationState.NEEDS_ATTENTION,
        NotesSyncOperationState.COMPLETED,
    ],
)
def test_new_operations_must_enter_through_the_pending_stage(
    tmp_path: Path,
    state: NotesSyncOperationState,
) -> None:
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")
    store.create_root(
        _root(logical_folder_id="folder-1", state=NotesSyncRootState.ACTIVE)
    )
    store.create_binding(_binding())

    with pytest.raises(NotesDeviceStateError, match="pending"):
        store.create_operation(
            NotesSyncOperationRecord(
                operation_id="operation-1",
                root_id="root-1",
                binding_id="binding-1",
                kind="update_note",
                state=state,
                reason_code=None,
                observation_token="observation-1",
                expected_note_version=3,
                expected_file_digest="b" * 64,
            )
        )

    with store.transaction() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM notes_sync_operations"
        ).fetchone() == (0,)


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


def test_root_runtime_status_updates_atomically_and_is_validated(
    tmp_path: Path,
) -> None:
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")
    store.create_root(_root())

    updated = store.update_root_status("root-1", "offline")

    assert updated.last_status_code == "offline"
    assert store.list_root_summaries()[0].last_status_code == "offline"
    with pytest.raises(ValueError):
        store.update_root_status("root-1", "not a reason code")


def test_public_summaries_fail_closed_on_path_like_persisted_identifiers(
    tmp_path: Path,
) -> None:
    root_database = tmp_path / "root-summary.sqlite3"
    root_store = NotesDeviceStateStore(root_database)
    root_store.create_root(
        _root(logical_folder_id="folder-1", state=NotesSyncRootState.ACTIVE)
    )
    with sqlite3.connect(root_database) as connection:
        connection.execute("UPDATE notes_sync_roots SET root_id = '/private/root'")
        connection.commit()

    with pytest.raises(ValueError, match="opaque"):
        root_store.list_root_summaries()

    binding_database = tmp_path / "binding-summary.sqlite3"
    binding_store = NotesDeviceStateStore(binding_database)
    binding_store.create_root(
        _root(logical_folder_id="folder-1", state=NotesSyncRootState.ACTIVE)
    )
    binding_store.create_binding(_binding())
    with sqlite3.connect(binding_database) as connection:
        connection.execute("UPDATE notes_sync_bindings SET note_id = 'folder/note'")
        connection.commit()

    with pytest.raises(ValueError, match="opaque"):
        binding_store.list_binding_summaries("root-1")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("recovery_capacity", "0"),
        ("recovery_capacity", "01"),
        ("recovery_capacity", "-1"),
        ("recovery_capacity", "9223372036854775808"),
        ("cutover_marker", "contains space"),
        ("cutover_marker", "/private/path"),
        ("cutover_marker", "x" * 257),
    ],
)
def test_store_settings_validate_each_allowlisted_key(key: str, value: str) -> None:
    with pytest.raises(ValueError, match="setting"):
        NotesSyncStoreSetting(key=key, value=value)


def test_store_settings_accept_canonical_values_and_fail_closed_on_corrupt_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    store = NotesDeviceStateStore(database)
    capacity = NotesSyncStoreSetting(key="recovery_capacity", value="1048576")
    marker = NotesSyncStoreSetting(key="cutover_marker", value="cutover:v1")

    store.set_setting(capacity)
    store.set_setting(marker)

    assert store.get_setting("recovery_capacity") == capacity
    assert store.get_setting("cutover_marker") == marker
    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO notes_sync_store_settings (
                    setting_key, setting_value, updated_at
                ) VALUES ('recovery_capacity', '01', 1)
                ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value
                """
            )
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            UPDATE notes_sync_store_settings
            SET setting_value = '01' WHERE setting_key = 'recovery_capacity'
            """
        )
        connection.commit()

    with pytest.raises(ValueError, match="setting"):
        store.get_setting("recovery_capacity")


def test_held_connection_reads_back_wal_normal_and_true_autocommit(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    store = NotesDeviceStateStore(database)
    store.initialize()

    with store.transaction() as connection:
        held = connection
    with store.transaction() as connection:
        assert connection is held

    assert held.isolation_level is None
    assert held.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert held.execute("PRAGMA synchronous").fetchone() == (1,)
    assert held.execute("PRAGMA foreign_keys").fetchone() == (1,)
    with closing(sqlite3.connect(database)) as independent:
        assert independent.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_schema_census_runs_once_per_connection_lifetime_not_per_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    census_calls = 0
    real_initialize = notes_device_state_schema.initialize_notes_device_schema

    def counting_initialize(connection: sqlite3.Connection) -> None:
        nonlocal census_calls
        census_calls += 1
        real_initialize(connection)

    monkeypatch.setattr(
        notes_device_state_schema,
        "initialize_notes_device_schema",
        counting_initialize,
    )
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")
    store.initialize()
    assert census_calls == 1

    store.create_root(_root())
    store.get_root("root-1")
    store.list_root_summaries()
    with store.transaction() as connection:
        held = connection
    assert census_calls == 1

    statements: list[str] = []
    held.set_trace_callback(statements.append)
    try:
        store.get_root("root-1")
    finally:
        held.set_trace_callback(None)
    assert 1 <= len(statements) <= 5
    joined = "\n".join(statements).upper()
    assert "SQLITE_SCHEMA" not in joined
    assert "CREATE INDEX" not in joined


def test_each_thread_holds_its_own_connection_and_work_is_visible_across_threads(
    tmp_path: Path,
) -> None:
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")
    store.initialize()
    with store.transaction() as connection:
        main_thread_connection = connection

    worker_connections: list[sqlite3.Connection] = []

    def create_in_worker() -> None:
        store.create_root(_root())
        with store.transaction() as connection:
            worker_connections.append(connection)
        with store.transaction() as connection:
            worker_connections.append(connection)

    worker = threading.Thread(target=create_in_worker)
    worker.start()
    worker.join()

    assert len(worker_connections) == 2
    assert worker_connections[0] is worker_connections[1]
    assert worker_connections[0] is not main_thread_connection
    assert store.get_root("root-1").state is NotesSyncRootState.PENDING


def test_close_releases_held_connections_of_every_thread_and_the_store_reopens(
    tmp_path: Path,
) -> None:
    store = NotesDeviceStateStore(tmp_path / "notes-sync.sqlite3")
    store.create_root(_root())
    with store.transaction() as connection:
        main_thread_connection = connection
    worker_connections: list[sqlite3.Connection] = []

    def observe_in_worker() -> None:
        with store.transaction() as connection:
            worker_connections.append(connection)

    worker = threading.Thread(target=observe_in_worker)
    worker.start()
    worker.join()

    store.close()

    with pytest.raises(sqlite3.ProgrammingError):
        main_thread_connection.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        worker_connections[0].execute("SELECT 1")
    assert store.get_root("root-1").state is NotesSyncRootState.PENDING
    store.close()


def test_refused_foreign_database_is_never_switched_to_wal(tmp_path: Path) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE unrelated_private_owner (value TEXT)")
        connection.commit()
    database.chmod(0o600)

    with pytest.raises(NotesDeviceStateError, match="incompatible"):
        NotesDeviceStateStore(database).initialize()

    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)


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
