import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB


_WORKSPACE_V2_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY NOT NULL
);
INSERT INTO schema_version (version) VALUES (1);
INSERT INTO schema_version (version) VALUES (2);

CREATE TABLE workspace_records (
    workspace_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    authority TEXT NOT NULL,
    sync_status TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE workspace_memberships (
    membership_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    item_type TEXT NOT NULL,
    item_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'source',
    transfer_policy TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(workspace_id) REFERENCES workspace_records(workspace_id) ON DELETE CASCADE,
    UNIQUE(workspace_id, item_type, item_id, role)
);

CREATE TABLE workspace_runtime_bindings (
    binding_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    binding_kind TEXT NOT NULL,
    label TEXT NOT NULL,
    locator TEXT NOT NULL,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(workspace_id) REFERENCES workspace_records(workspace_id) ON DELETE CASCADE
);

CREATE TABLE workspace_handoff_audit (
    audit_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    manifest_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(workspace_id) REFERENCES workspace_records(workspace_id) ON DELETE CASCADE
);

CREATE TABLE workspace_rag_scopes (
    workspace_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(workspace_id) REFERENCES workspace_records(workspace_id) ON DELETE CASCADE
);

CREATE TABLE workspace_change_review (
    workspace_id TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(workspace_id) REFERENCES workspace_records(workspace_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX idx_workspace_records_name_ci
ON workspace_records (lower(name)) WHERE archived = 0;
"""

_OPERATION_COLUMNS = {
    "operation_id",
    "idempotency_key",
    "data_source",
    "server_profile_id",
    "principal_id",
    "workspace_id",
    "ingest_job_id",
    "canonical_item_type",
    "canonical_item_id",
    "workspace_source_id",
    "desired_selected",
    "catalog_status",
    "association_status",
    "readiness_status",
    "error_stage",
    "error_code",
    "error_message",
    "revision",
    "created_at",
    "updated_at",
}

_QUICK_NOTE_RECEIPT_COLUMNS = {
    "receipt_id",
    "data_source",
    "server_profile_id",
    "principal_id",
    "workspace_id",
    "local_user_id",
    "operation_token",
    "operation_kind",
    "canonical_note_id",
    "owner_proof",
    "lease_token",
    "lease_expires_at",
    "expected_version",
    "state",
    "revision",
    "failure_count",
    "next_retry_at",
    "blocked_reason_code",
    "created_at",
    "updated_at",
}

_EARLY_V4_RECEIPT_SCHEMA = """
CREATE TABLE research_quick_note_receipts (
    receipt_id TEXT PRIMARY KEY,
    data_source TEXT NOT NULL DEFAULT 'local',
    workspace_id TEXT NOT NULL,
    local_user_id TEXT NOT NULL,
    operation_token TEXT NOT NULL,
    operation_kind TEXT NOT NULL,
    canonical_note_id TEXT NOT NULL,
    expected_version INTEGER DEFAULT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    revision INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_research_quick_note_receipts_reconcile
ON research_quick_note_receipts (local_user_id, state, updated_at, receipt_id);
INSERT INTO schema_version (version) VALUES (4);
"""


def _create_genuine_v3_database(path: Path) -> None:
    _create_genuine_v2_database(path)
    connection = sqlite3.connect(path)
    connection.executescript(WorkspaceDB._MIGRATE_V2_TO_V3_SQL)
    connection.execute(
        """
        INSERT INTO workspace_memberships
          (membership_id, workspace_id, item_type, item_id, role,
           transfer_policy, title, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-pending",
            "local-kept",
            "note",
            "research-note-legacy",
            "note_pending",
            "reference",
            "",
            "2026-08-24T00:00:00Z",
        ),
    )
    connection.commit()
    connection.close()


def _create_early_branch_v4_database(path: Path) -> None:
    _create_genuine_v3_database(path)
    connection = sqlite3.connect(path)
    connection.executescript(_EARLY_V4_RECEIPT_SCHEMA)
    connection.execute(
        "UPDATE workspace_memberships SET role = 'note' WHERE role = 'note_pending'"
    )
    connection.execute(
        """
        INSERT INTO workspace_memberships
          (membership_id, workspace_id, item_type, item_id, role,
           transfer_policy, title, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legitimate-blank-note",
            "local-kept",
            "note",
            "ordinary-note",
            "note",
            "reference",
            "",
            "2026-08-24T00:00:00Z",
        ),
    )
    connection.execute(
        """
        INSERT INTO research_quick_note_receipts
          (receipt_id, data_source, workspace_id, local_user_id,
           operation_token, operation_kind, canonical_note_id,
           expected_version, state, revision, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "unsafe-early-receipt",
            "local",
            "local-kept",
            "notes-user",
            "research-note-123e4567e89b42d3a456426614174000",
            "create",
            "unsafe-note",
            None,
            "pending",
            1,
            "2026-08-24T00:00:00+00:00",
            "2026-08-24T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()


def _create_genuine_v2_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(_WORKSPACE_V2_SCHEMA)
    connection.execute(
        """
        INSERT INTO workspace_records
          (workspace_id, name, description, authority, sync_status, active,
           archived, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "local-kept",
            "Kept workspace",
            "unrelated row",
            "local",
            "local_only",
            1,
            0,
            "2026-08-20T00:00:00Z",
            "2026-08-20T00:00:00Z",
        ),
    )
    connection.execute(
        """
        INSERT INTO workspace_memberships
          (membership_id, workspace_id, item_type, item_id, role,
           transfer_policy, title, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "membership-kept",
            "local-kept",
            "media",
            "41",
            "source",
            "reference",
            "Kept source",
            "2026-08-20T00:00:00Z",
        ),
    )
    connection.commit()
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(workspace_records)")
    }
    assert "research_source_operations" not in {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert columns == {
        "workspace_id",
        "name",
        "description",
        "authority",
        "sync_status",
        "active",
        "archived",
        "created_at",
        "updated_at",
    }
    connection.close()


def test_fresh_workspace_db_has_v3_operation_schema_without_workspace_fk(
    tmp_path: Path,
) -> None:
    db = WorkspaceDB(tmp_path / "fresh.sqlite")

    assert db.get_schema_version() == WorkspaceDB._CURRENT_SCHEMA_VERSION
    with db.connection() as connection:
        assert _OPERATION_COLUMNS <= {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(research_source_operations)"
            )
        }
        assert (
            connection.execute(
                "PRAGMA foreign_key_list(research_source_operations)"
            ).fetchall()
            == []
        )
    db.close()


def test_genuine_v2_upgrade_preserves_unrelated_rows_and_accepts_server_target(
    tmp_path: Path,
) -> None:
    path = tmp_path / "historical-v2.sqlite"
    _create_genuine_v2_database(path)

    db = WorkspaceDB(path)

    assert db.get_schema_version() == WorkspaceDB._CURRENT_SCHEMA_VERSION
    with db.transaction() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_version ORDER BY version"
        ).fetchall()
        assert [row[0] for row in versions] == [1, 2, 3, 4, 5]
        kept = connection.execute(
            "SELECT name, description FROM workspace_records WHERE workspace_id = ?",
            ("local-kept",),
        ).fetchone()
        membership = connection.execute(
            "SELECT item_id, title FROM workspace_memberships WHERE membership_id = ?",
            ("membership-kept",),
        ).fetchone()
        assert tuple(kept) == ("Kept workspace", "unrelated row")
        assert tuple(membership) == ("41", "Kept source")

        # A Server workspace ID deliberately has no corresponding Local row.
        connection.execute(
            """
            INSERT INTO research_source_operations
              (operation_id, idempotency_key, data_source, server_profile_id,
               principal_id, workspace_id, canonical_item_type, desired_selected,
               catalog_status, association_status, readiness_status, revision,
               created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "operation-server",
                "server-profile:principal:workspace-900:source-1",
                "server",
                "server-profile",
                "principal",
                "workspace-900",
                "server_media",
                1,
                "pending",
                "pending",
                "pending",
                1,
                "2026-08-24T00:00:00Z",
                "2026-08-24T00:00:00Z",
            ),
        )
    db.close()

    reopened = WorkspaceDB(path)
    with reopened.connection() as connection:
        assert (
            connection.execute(
                "SELECT workspace_id FROM research_source_operations WHERE operation_id = ?",
                ("operation-server",),
            ).fetchone()[0]
            == "workspace-900"
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM workspace_records WHERE workspace_id = ?",
                ("workspace-900",),
            ).fetchone()[0]
            == 0
        )
    reopened.close()


def test_workspace_v3_inline_migration_matches_packaged_sql_byte_for_byte() -> None:
    migration_path = (
        Path(__file__).parents[2]
        / "tldw_chatbook/DB/migrations/workspaces_v2_to_v3_research_source_operations.sql"
    )

    assert (
        migration_path.read_text(encoding="utf-8") == WorkspaceDB._MIGRATE_V2_TO_V3_SQL
    )


def test_genuine_v3_upgrade_adds_payload_free_receipts_and_drops_unverifiable_legacy_role(
    tmp_path: Path,
) -> None:
    path = tmp_path / "historical-v3.sqlite"
    _create_genuine_v3_database(path)

    db = WorkspaceDB(path)

    assert db.get_schema_version() == 5
    with db.connection() as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(research_quick_note_receipts)"
            )
        }
        assert columns == _QUICK_NOTE_RECEIPT_COLUMNS
        assert not columns & {
            "title",
            "content",
            "body",
            "tags",
            "keywords",
            "provenance",
            "path",
            "url",
        }
        assert connection.execute(
            "SELECT COUNT(*) FROM workspace_memberships WHERE role = 'note_pending'"
        ).fetchone()[0] == 0
        assert connection.execute(
            """
            SELECT role FROM workspace_memberships
            WHERE membership_id = 'legacy-pending'
            """
        ).fetchone() is None
    db.close()


def test_workspace_v4_inline_migration_matches_packaged_sql_and_rolls_back(
    tmp_path: Path,
) -> None:
    migration_path = (
        Path(__file__).parents[2]
        / "tldw_chatbook/DB/migrations/workspaces_v3_to_v4_quick_note_receipts.sql"
    )
    assert migration_path.read_text(encoding="utf-8") == WorkspaceDB._MIGRATE_V3_TO_V4_SQL

    path = tmp_path / "rollback-v3.sqlite"
    _create_genuine_v3_database(path)

    class BrokenV4WorkspaceDB(WorkspaceDB):
        _MIGRATE_V3_TO_V4_SQL = WorkspaceDB._MIGRATE_V3_TO_V4_SQL.replace(
            "COMMIT;", "INSERT INTO table_that_does_not_exist VALUES (1);\n\nCOMMIT;"
        )

    with pytest.raises(sqlite3.Error):
        BrokenV4WorkspaceDB(path)

    connection = sqlite3.connect(path)
    assert connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 3
    assert connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='research_quick_note_receipts'"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT role FROM workspace_memberships WHERE membership_id='legacy-pending'"
    ).fetchone()[0] == "note_pending"
    connection.close()


def test_early_branch_v4_upgrade_quarantines_unsafe_receipts_and_phantom_links(
    tmp_path: Path,
) -> None:
    path = tmp_path / "early-v4.sqlite"
    _create_early_branch_v4_database(path)

    db = WorkspaceDB(path)

    assert db.get_schema_version() == 5
    migration_path = (
        Path(__file__).parents[2]
        / "tldw_chatbook/DB/migrations/workspaces_v4_to_v5_quick_note_receipts.sql"
    )
    assert migration_path.read_text(encoding="utf-8") == WorkspaceDB._MIGRATE_V4_TO_V5_SQL
    with db.connection() as connection:
        assert connection.execute(
            "SELECT 1 FROM research_quick_note_receipts WHERE receipt_id = ?",
            ("unsafe-early-receipt",),
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM workspace_memberships WHERE membership_id = ?",
            ("legacy-pending",),
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM workspace_memberships WHERE membership_id = ?",
            ("legitimate-blank-note",),
        ).fetchone() is not None
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(research_quick_note_receipts)"
            )
        }
        assert columns == _QUICK_NOTE_RECEIPT_COLUMNS
    db.close()


def test_workspace_v5_remediation_rolls_back_early_v4_atomically(tmp_path: Path) -> None:
    path = tmp_path / "early-v4-rollback.sqlite"
    _create_early_branch_v4_database(path)

    class BrokenV5WorkspaceDB(WorkspaceDB):
        _MIGRATE_V4_TO_V5_SQL = WorkspaceDB._MIGRATE_V4_TO_V5_SQL.replace(
            "COMMIT;", "INSERT INTO missing_v5_table VALUES (1);\n\nCOMMIT;"
        )

    with pytest.raises(sqlite3.Error):
        BrokenV5WorkspaceDB(path)

    connection = sqlite3.connect(path)
    assert connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 4
    assert connection.execute(
        "SELECT 1 FROM research_quick_note_receipts WHERE receipt_id = ?",
        ("unsafe-early-receipt",),
    ).fetchone() is not None
    assert connection.execute(
        "SELECT role FROM workspace_memberships WHERE membership_id = ?",
        ("legacy-pending",),
    ).fetchone()[0] == "note"
    connection.close()


@pytest.mark.parametrize(
    ("overrides"),
    [
        {"data_source": "server"},
        {"operation_kind": "archive"},
        {"state": "done"},
        {"revision": 0},
        {"state": "owner_committed", "revision": 1},
        {"state": "blocked", "revision": 2, "blocked_reason_code": ""},
        {"local_user_id": ""},
        {"created_at": ""},
        {"updated_at": ""},
        {"lease_expires_at": "not-a-timestamp"},
        {"updated_at": "2026-08-23T00:00:00Z"},
        {"operation_kind": "create", "expected_version": 1},
        {"operation_kind": "delete", "expected_version": None},
    ],
)
def test_quick_note_receipt_table_rejects_invalid_owner_state_and_version_guards(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    db = WorkspaceDB(tmp_path / "invalid-receipt.sqlite")
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

    LocalWorkspaceRegistryService(db).create_workspace(
        workspace_id="workspace-1", name="Workspace"
    )
    values: dict[str, object] = {
        "receipt_id": "receipt-1",
        "data_source": "local",
        "server_profile_id": "",
        "principal_id": "",
        "workspace_id": "workspace-1",
        "local_user_id": "notes-user",
        "operation_token": "research-note-123e4567e89b42d3a456426614174000",
        "operation_kind": "delete",
        "canonical_note_id": "note-1",
        "owner_proof": "owner-proof-1234567890abcdef1234567890abcdef",
        "lease_token": "lease-token-1234567890abcdef1234567890abcdef",
        "lease_expires_at": "2026-08-24T00:00:30+00:00",
        "expected_version": 1,
        "state": "pending",
        "revision": 1,
        "failure_count": 0,
        "next_retry_at": "2026-08-24T00:00:00+00:00",
        "blocked_reason_code": "",
        "created_at": "2026-08-24T00:00:00Z",
        "updated_at": "2026-08-24T00:00:00Z",
    }
    values.update(overrides)
    fields = tuple(values)
    with pytest.raises(sqlite3.IntegrityError):
        with db.transaction() as connection:
            connection.execute(
                f"""
                INSERT INTO research_quick_note_receipts ({', '.join(fields)})
                VALUES ({', '.join('?' for _ in fields)})
                """,
                tuple(values[field] for field in fields),
            )
    db.close()


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("data_source", "cloud"),
        ("catalog_status", "complete"),
        ("association_status", "complete"),
        ("readiness_status", "complete"),
        ("desired_selected", 2),
        ("revision", 0),
    ],
)
def test_operation_table_rejects_invalid_raw_enum_status_and_guard_values(
    tmp_path: Path,
    column: str,
    invalid_value: object,
) -> None:
    db = WorkspaceDB(tmp_path / f"invalid-{column}.sqlite")
    fields = (
        "operation_id",
        "idempotency_key",
        "data_source",
        "server_profile_id",
        "principal_id",
        "workspace_id",
        "canonical_item_type",
        "desired_selected",
        "catalog_status",
        "association_status",
        "readiness_status",
        "revision",
        "created_at",
        "updated_at",
    )
    values: dict[str, object] = {
        "operation_id": f"operation-{column}",
        "idempotency_key": f"key-{column}",
        "data_source": "local",
        "server_profile_id": "",
        "principal_id": "",
        "workspace_id": "workspace-1",
        "canonical_item_type": "local_library",
        "desired_selected": 1,
        "catalog_status": "pending",
        "association_status": "pending",
        "readiness_status": "pending",
        "revision": 1,
        "created_at": "2026-08-24T00:00:00Z",
        "updated_at": "2026-08-24T00:00:00Z",
    }
    values[column] = invalid_value

    with pytest.raises(sqlite3.IntegrityError):
        with db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO research_source_operations
                  (operation_id, idempotency_key, data_source, server_profile_id,
                   principal_id, workspace_id, canonical_item_type, desired_selected,
                   catalog_status, association_status, readiness_status, revision,
                   created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(values[field] for field in fields),
            )
    db.close()
