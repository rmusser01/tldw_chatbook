from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


SCHEMA_NAME = "rag_char_chat_schema"
PROVENANCE_TABLES = {
    "rag_identity_context",
    "rag_citation_traces",
    "rag_evidence_runs",
    "rag_evidence_snapshots",
    "rag_answer_attempt_payloads",
    "rag_trace_evidence_refs",
    "rag_message_trace_owners",
    "rag_source_observations",
    "rag_payload_tombstones",
    "rag_artifact_owner_leases",
    "rag_artifact_owner_operations",
    "rag_legacy_migration_journal",
}


def _fresh_db(path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(path, client_id="citation-migration-test")


def _minimal_v24(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            f"""
            PRAGMA foreign_keys = ON;
            CREATE TABLE db_schema_version(
                schema_name TEXT PRIMARY KEY NOT NULL,
                version INTEGER NOT NULL
            );
            INSERT INTO db_schema_version VALUES ('{SCHEMA_NAME}', 24);
            CREATE TABLE conversations(id TEXT PRIMARY KEY);
            CREATE TABLE messages(
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE
            );
            """
        )


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _version(connection: sqlite3.Connection) -> int:
    return connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()[0]


def test_fresh_database_reaches_v25_with_stable_identity_context(
    tmp_path: Path,
) -> None:
    db = _fresh_db(tmp_path / "fresh.sqlite")
    connection = db.get_connection()

    assert db._CURRENT_SCHEMA_VERSION == 25
    assert _version(connection) == 25
    assert PROVENANCE_TABLES <= _table_names(connection)
    row = connection.execute("SELECT * FROM rag_identity_context").fetchone()
    assert row["context_name"] == "default"
    assert re.fullmatch(r"profile_[0-9a-f]{32}", row["profile_id"])
    assert re.fullmatch(r"authority_[0-9a-f]{32}", row["local_authority_id"])
    assert re.fullmatch(r"fpkey_[0-9a-f]{32}", row["fingerprint_key_id"])
    assert row["created_at"]


def test_v24_upgrade_uses_the_exact_standalone_sql_schema(tmp_path: Path) -> None:
    path = tmp_path / "upgrade.sqlite"
    _minimal_v24(path)

    db = _fresh_db(path)
    connection = db.get_connection()

    assert _version(connection) == 25
    assert PROVENANCE_TABLES <= _table_names(connection)

    expected_columns = {
        "rag_identity_context": {
            "context_name",
            "profile_id",
            "local_authority_id",
            "fingerprint_key_id",
            "created_at",
        },
        "rag_citation_traces": {
            "profile_id",
            "trace_id",
            "schema_version",
            "request_id",
            "generation_id",
            "origin_scope_id",
            "origin",
            "lifecycle",
            "completeness_at_seal",
            "selected_attempt_id",
            "policy_version",
            "aggregate_json",
            "visibility_state",
            "created_at",
            "sealed_at",
            "connection_authority_id",
            "tenant_id",
            "server_trace_id",
            "wire_schema_version",
            "import_package_fingerprint",
            "external_trace_id",
            "legacy_conversation_id",
            "legacy_message_id",
        },
        "rag_evidence_runs": {
            "profile_id",
            "trace_id",
            "run_id",
            "run_ordinal",
            "stage",
            "redaction_state",
            "run_payload_json",
            "started_at",
            "ended_at",
            "purged_at",
        },
        "rag_evidence_snapshots": {
            "profile_id",
            "payload_id",
            "governance_scope_id",
            "authority_id",
            "confidentiality_policy_id",
            "revocation_scope_id",
            "origin_namespace",
            "origin_payload_id",
            "storage_mode",
            "redaction_state",
            "retention_class",
            "snapshot_text",
            "title",
            "source_identity_json",
            "locator_json",
            "lineage_json",
            "transformations_json",
            "content_hash",
            "comparison_fingerprint",
            "created_at",
            "retain_until",
            "purged_at",
        },
        "rag_answer_attempt_payloads": {
            "profile_id",
            "payload_id",
            "trace_id",
            "attempt_id",
            "redaction_state",
            "retention_class",
            "answer_body",
            "body_integrity_hmac",
            "created_at",
            "retain_until",
            "purged_at",
        },
        "rag_trace_evidence_refs": {
            "profile_id",
            "trace_id",
            "prompt_set_id",
            "evidence_ordinal",
            "run_id",
            "snapshot_payload_id",
            "marker_ordinal",
            "storage_mode",
        },
        "rag_message_trace_owners": {
            "profile_id",
            "message_id",
            "message_revision",
            "trace_id",
            "state",
            "body_fingerprint",
            "idempotency_key",
            "created_at",
            "updated_at",
        },
        "rag_source_observations": {
            "profile_id",
            "trace_id",
            "prompt_set_id",
            "evidence_ordinal",
            "snapshot_payload_id",
            "resolver_kind",
            "resolver_version",
            "availability",
            "permission_state",
            "content_state",
            "location_state",
            "capabilities_json",
            "request_nonce",
            "observed_at",
            "error_code",
        },
        "rag_payload_tombstones": {
            "profile_id",
            "origin_namespace",
            "origin_payload_id",
            "revocation_scope_id",
            "reason_code",
            "policy_version",
            "revoked_at",
            "retain_until",
        },
        "rag_artifact_owner_leases": {
            "profile_id",
            "artifact_store_id",
            "artifact_id",
            "artifact_revision",
            "trace_id",
            "lease_id",
            "state",
            "created_at",
            "updated_at",
            "retain_until",
        },
        "rag_artifact_owner_operations": {
            "profile_id",
            "operation_id",
            "artifact_store_id",
            "artifact_id",
            "artifact_revision",
            "trace_id",
            "operation_kind",
            "state",
            "created_at",
            "updated_at",
        },
        "rag_legacy_migration_journal": {
            "profile_id",
            "conversation_id",
            "source_fingerprint",
            "state",
            "attempt_count",
            "started_at",
            "updated_at",
            "next_message_cursor",
            "error_code",
            "completed_at",
        },
    }
    for table, expected in expected_columns.items():
        actual = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        assert actual == expected


def test_schema_enforces_origin_shapes_and_null_safe_uniqueness(
    tmp_path: Path,
) -> None:
    db = _fresh_db(tmp_path / "constraints.sqlite")
    connection = db.get_connection()
    context = connection.execute("SELECT * FROM rag_identity_context").fetchone()
    profile = context["profile_id"]
    common = (
        1,
        "request",
        "generation",
        "sealed",
        "complete",
        "attempt",
        "policy",
        "{}",
        "active",
        "2026-07-24T00:00:00Z",
        "2026-07-24T00:00:00Z",
    )
    insert_trace = """
        INSERT INTO rag_citation_traces(
            profile_id, trace_id, schema_version, request_id, generation_id,
            origin_scope_id, origin, lifecycle, completeness_at_seal,
            selected_attempt_id, policy_version, aggregate_json,
            visibility_state, created_at, sealed_at,
            connection_authority_id, tenant_id, server_trace_id,
            wire_schema_version, import_package_fingerprint, external_trace_id,
            legacy_conversation_id, legacy_message_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    connection.execute(
        insert_trace,
        (
            profile,
            "local",
            *common[:3],
            profile,
            "local",
            *common[3:],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            insert_trace,
            (
                profile,
                "bad-local",
                *common[:3],
                "wrong",
                "local",
                *common[3:],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            insert_trace,
            (
                profile,
                "bad-cross-origin",
                *common[:3],
                profile,
                "local",
                *common[3:],
                "server",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )

    server_values = (
        profile,
        "server-local-a",
        *common[:3],
        "authority-root",
        "server",
        *common[3:],
        "server-authority",
        None,
        "external-trace",
        "grounding_trace/v1",
        None,
        None,
        None,
        None,
    )
    connection.execute(insert_trace, server_values)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            insert_trace,
            (profile, "server-local-b", *server_values[2:]),
        )

    snapshot = (
        profile,
        "snapshot-a",
        profile,
        context["local_authority_id"],
        "policy",
        "revocation",
        "local_payload_v1",
        "origin-a",
        "embedded",
        "available",
        "default",
        "same",
        None,
        "{}",
        "{}",
        "{}",
        "[]",
        "content-hmac",
        "comparison-hmac",
        "2026-07-24T00:00:00Z",
        None,
        None,
    )
    connection.execute(
        "INSERT INTO rag_evidence_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        snapshot,
    )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO rag_evidence_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (profile, "snapshot-b", *snapshot[2:]),
        )


def test_schema_has_exact_foreign_key_delete_policies(tmp_path: Path) -> None:
    db = _fresh_db(tmp_path / "foreign-keys.sqlite")
    connection = db.get_connection()

    def policies(table: str) -> set[tuple[str, str, str]]:
        return {
            (row["table"], row["from"], row["on_delete"])
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        }

    assert ("rag_citation_traces", "trace_id", "CASCADE") in policies(
        "rag_evidence_runs"
    )
    assert ("rag_evidence_snapshots", "snapshot_payload_id", "RESTRICT") in policies(
        "rag_trace_evidence_refs"
    )
    assert ("messages", "message_id", "CASCADE") in policies("rag_message_trace_owners")
    assert ("rag_citation_traces", "trace_id", "RESTRICT") in policies(
        "rag_message_trace_owners"
    )
    assert ("conversations", "conversation_id", "CASCADE") in policies(
        "rag_legacy_migration_journal"
    )


@pytest.mark.parametrize(
    ("table", "insert_sql"),
    [
        (
            "rag_evidence_snapshots",
            """
            INSERT INTO rag_evidence_snapshots(
                profile_id, payload_id, governance_scope_id, authority_id,
                confidentiality_policy_id, revocation_scope_id,
                origin_namespace, origin_payload_id, storage_mode,
                redaction_state, retention_class, created_at
            ) VALUES (?, 'payload', 'scope', 'authority', 'policy', 'revoke',
                      'local_payload_v1', 'origin', 'redacted', 'redacted',
                      'default', '2026-07-24T00:00:00Z')
            """,
        ),
        (
            "rag_payload_tombstones",
            """
            INSERT INTO rag_payload_tombstones VALUES (
                ?, 'local_payload_v1', 'payload', 'revoke', 'reason', 'policy',
                '2026-07-24T00:00:00Z', '2027-07-24T00:00:00Z'
            )
            """,
        ),
        (
            "rag_legacy_migration_journal",
            """
            INSERT INTO rag_legacy_migration_journal VALUES (
                ?, 'conversation', 'fingerprint', 'pending', 0,
                '2026-07-24T00:00:00Z', '2026-07-24T00:00:00Z',
                NULL, NULL, NULL
            )
            """,
        ),
    ],
)
def test_standalone_profile_identifiers_are_utf8_bounded(
    tmp_path: Path,
    table: str,
    insert_sql: str,
) -> None:
    path = tmp_path / f"{table}.sqlite"
    _minimal_v24(path)
    db = _fresh_db(path)
    connection = db.get_connection()
    connection.execute("INSERT INTO conversations(id) VALUES ('conversation')")

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(insert_sql, ("é" * 129,))


@pytest.mark.parametrize("failure_kind", ["ddl", "version_update"])
def test_migration_failure_rolls_back_schema_context_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    path = tmp_path / f"rollback-{failure_kind}.sqlite"
    _minimal_v24(path)

    if failure_kind == "ddl":
        original = CharactersRAGDB._execute_citation_migration_statement

        def fail_on_runs(self, cursor, statement):
            if "CREATE TABLE rag_evidence_runs" in statement:
                raise sqlite3.OperationalError("forced ddl failure")
            return original(self, cursor, statement)

        monkeypatch.setattr(
            CharactersRAGDB,
            "_execute_citation_migration_statement",
            fail_on_runs,
        )
    else:
        monkeypatch.setattr(
            CharactersRAGDB,
            "_update_citation_schema_version",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                sqlite3.OperationalError("forced version failure")
            ),
        )

    with pytest.raises(Exception, match="forced"):
        _fresh_db(path)

    with sqlite3.connect(path) as connection:
        assert _version(connection) == 24
        assert not (PROVENANCE_TABLES & _table_names(connection))


def test_migration_sql_is_ddl_only_and_provenance_is_not_indexed_or_synced(
    tmp_path: Path,
) -> None:
    sql_path = (
        Path(__file__).parents[2]
        / "tldw_chatbook"
        / "DB"
        / "migrations"
        / "chachanotes_v24_to_v25_citation_provenance.sql"
    )
    sql = sql_path.read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    ).lower()
    assert "begin" not in executable
    assert "commit" not in executable
    assert "db_schema_version" not in executable
    assert "create trigger" not in executable

    db = _fresh_db(tmp_path / "inventories.sqlite")
    connection = db.get_connection()
    rag_objects = connection.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE name LIKE 'rag_%'
        """
    ).fetchall()
    assert all(row["type"] != "trigger" for row in rag_objects)
    assert all("fts5" not in (row["sql"] or "").lower() for row in rag_objects)
