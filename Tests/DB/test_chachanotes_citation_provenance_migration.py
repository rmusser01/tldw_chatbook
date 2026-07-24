from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError


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

TABLE_INFO_CONTRACT = {
    "rag_identity_context": """
        context_name:T:1:1 profile_id:T:1:0 local_authority_id:T:1:0
        fingerprint_key_id:T:1:0 created_at:T:1:0
    """,
    "rag_citation_traces": """
        profile_id:T:1:1 trace_id:T:1:2 schema_version:I:1:0 request_id:T:1:0
        generation_id:T:1:0 origin_scope_id:T:1:0 origin:T:1:0
        lifecycle:T:1:0 completeness_at_seal:T:1:0
        selected_attempt_id:T:1:0 policy_version:T:1:0 aggregate_json:T:1:0
        visibility_state:T:1:0 created_at:T:1:0 sealed_at:T:1:0
        connection_authority_id:T:0:0 tenant_id:T:0:0 server_trace_id:T:0:0
        wire_schema_version:T:0:0 import_package_fingerprint:T:0:0
        external_trace_id:T:0:0 legacy_conversation_id:T:0:0
        legacy_message_id:T:0:0
    """,
    "rag_evidence_runs": """
        profile_id:T:1:1 trace_id:T:1:2 run_id:T:1:3 run_ordinal:I:1:0
        stage:T:1:0 redaction_state:T:1:0 run_payload_json:T:0:0
        started_at:T:1:0 ended_at:T:0:0 purged_at:T:0:0
    """,
    "rag_evidence_snapshots": """
        profile_id:T:1:1 payload_id:T:1:2 governance_scope_id:T:1:0
        authority_id:T:1:0 confidentiality_policy_id:T:1:0
        revocation_scope_id:T:1:0 origin_namespace:T:1:0
        origin_payload_id:T:1:0 storage_mode:T:1:0 redaction_state:T:1:0
        retention_class:T:1:0 snapshot_text:T:0:0 title:T:0:0
        source_identity_json:T:0:0 locator_json:T:0:0 lineage_json:T:0:0
        transformations_json:T:0:0 content_hash:T:0:0
        comparison_fingerprint:T:0:0 created_at:T:1:0 retain_until:T:0:0
        purged_at:T:0:0
    """,
    "rag_answer_attempt_payloads": """
        profile_id:T:1:1 payload_id:T:1:2 trace_id:T:1:0 attempt_id:T:1:0
        redaction_state:T:1:0 retention_class:T:1:0 answer_body:T:0:0
        body_integrity_hmac:T:0:0 created_at:T:1:0 retain_until:T:0:0
        purged_at:T:0:0
    """,
    "rag_trace_evidence_refs": """
        profile_id:T:1:1 trace_id:T:1:2 prompt_set_id:T:1:3
        evidence_ordinal:I:1:4 run_id:T:1:0 snapshot_payload_id:T:1:0
        marker_ordinal:I:1:0 storage_mode:T:1:0
    """,
    "rag_message_trace_owners": """
        profile_id:T:1:1 message_id:T:1:2 message_revision:I:1:3
        trace_id:T:1:4 state:T:1:0 body_fingerprint:T:1:0
        idempotency_key:T:1:0 created_at:T:1:0 updated_at:T:1:0
    """,
    "rag_source_observations": """
        profile_id:T:1:1 trace_id:T:1:2 prompt_set_id:T:1:3
        evidence_ordinal:I:1:4 snapshot_payload_id:T:1:5 resolver_kind:T:1:6
        resolver_version:T:1:7 availability:T:1:0 permission_state:T:1:0
        content_state:T:1:0 location_state:T:1:0 capabilities_json:T:1:0
        request_nonce:T:1:0 observed_at:T:1:0 error_code:T:0:0
    """,
    "rag_payload_tombstones": """
        profile_id:T:1:1 origin_namespace:T:1:2 origin_payload_id:T:1:3
        revocation_scope_id:T:1:0 reason_code:T:1:0 policy_version:T:1:0
        revoked_at:T:1:0 retain_until:T:1:0
    """,
    "rag_artifact_owner_leases": """
        profile_id:T:1:1 artifact_store_id:T:1:2 artifact_id:T:1:3
        artifact_revision:I:1:4 trace_id:T:1:5 lease_id:T:1:0 state:T:1:0
        created_at:T:1:0 updated_at:T:1:0 retain_until:T:0:0
    """,
    "rag_artifact_owner_operations": """
        profile_id:T:1:1 operation_id:T:1:2 artifact_store_id:T:1:0
        artifact_id:T:1:0 artifact_revision:I:1:0 trace_id:T:1:0
        operation_kind:T:1:0 state:T:1:0 created_at:T:1:0 updated_at:T:1:0
    """,
    "rag_legacy_migration_journal": """
        profile_id:T:1:1 conversation_id:T:1:2 source_fingerprint:T:1:0
        state:T:1:0 attempt_count:I:1:0 started_at:T:1:0 updated_at:T:1:0
        next_message_cursor:T:0:0 error_code:T:0:0 completed_at:T:0:0
    """,
}

COMPOSITE_OR_PARTIAL_INDEX_CONTRACT = {
    "rag_citation_traces": {
        (1, 1, ("profile_id", "import_package_fingerprint", "external_trace_id")),
        (
            1,
            1,
            (
                "connection_authority_id",
                "origin_scope_id",
                "server_trace_id",
                "wire_schema_version",
            ),
        ),
        (1, 0, ("profile_id", "trace_id")),
    },
    "rag_evidence_runs": {
        (1, 0, ("profile_id", "trace_id", "run_ordinal")),
        (1, 0, ("profile_id", "trace_id", "run_id")),
    },
    "rag_evidence_snapshots": {
        (
            1,
            1,
            (
                "governance_scope_id",
                "authority_id",
                "confidentiality_policy_id",
                "revocation_scope_id",
                "content_hash",
            ),
        ),
        (1, 0, ("profile_id", "payload_id")),
    },
    "rag_answer_attempt_payloads": {
        (1, 0, ("profile_id", "trace_id", "attempt_id")),
        (1, 0, ("profile_id", "payload_id")),
    },
    "rag_trace_evidence_refs": {
        (1, 0, ("profile_id", "trace_id", "prompt_set_id", "marker_ordinal")),
        (1, 0, ("profile_id", "trace_id", "prompt_set_id", "evidence_ordinal")),
    },
    "rag_message_trace_owners": {
        (1, 1, ("profile_id", "message_id", "message_revision")),
        (1, 0, ("profile_id", "idempotency_key")),
        (1, 0, ("profile_id", "message_id", "message_revision", "trace_id")),
    },
    "rag_source_observations": {
        (
            1,
            0,
            (
                "profile_id",
                "trace_id",
                "prompt_set_id",
                "evidence_ordinal",
                "snapshot_payload_id",
                "resolver_kind",
                "resolver_version",
            ),
        ),
    },
    "rag_payload_tombstones": {
        (1, 0, ("profile_id", "origin_namespace", "origin_payload_id")),
    },
    "rag_artifact_owner_leases": {
        (
            1,
            0,
            (
                "profile_id",
                "artifact_store_id",
                "artifact_id",
                "artifact_revision",
                "trace_id",
            ),
        ),
    },
    "rag_artifact_owner_operations": {
        (1, 0, ("profile_id", "operation_id")),
        (
            1,
            0,
            (
                "profile_id",
                "artifact_store_id",
                "artifact_id",
                "artifact_revision",
                "trace_id",
                "operation_kind",
            ),
        ),
    },
    "rag_legacy_migration_journal": {
        (1, 0, ("profile_id", "conversation_id")),
    },
}

FOREIGN_KEY_CONTRACT = {
    "rag_evidence_runs": {
        (
            "rag_citation_traces",
            ("profile_id", "trace_id"),
            ("profile_id", "trace_id"),
            "CASCADE",
        ),
    },
    "rag_answer_attempt_payloads": {
        (
            "rag_citation_traces",
            ("profile_id", "trace_id"),
            ("profile_id", "trace_id"),
            "CASCADE",
        ),
    },
    "rag_trace_evidence_refs": {
        (
            "rag_evidence_snapshots",
            ("profile_id", "snapshot_payload_id"),
            ("profile_id", "payload_id"),
            "RESTRICT",
        ),
        (
            "rag_evidence_runs",
            ("profile_id", "trace_id", "run_id"),
            ("profile_id", "trace_id", "run_id"),
            "CASCADE",
        ),
        (
            "rag_citation_traces",
            ("profile_id", "trace_id"),
            ("profile_id", "trace_id"),
            "CASCADE",
        ),
    },
    "rag_message_trace_owners": {
        (
            "rag_citation_traces",
            ("profile_id", "trace_id"),
            ("profile_id", "trace_id"),
            "RESTRICT",
        ),
        ("messages", ("message_id",), ("id",), "CASCADE"),
    },
    "rag_source_observations": {
        (
            "rag_evidence_snapshots",
            ("profile_id", "snapshot_payload_id"),
            ("profile_id", "payload_id"),
            "CASCADE",
        ),
        (
            "rag_citation_traces",
            ("profile_id", "trace_id"),
            ("profile_id", "trace_id"),
            "CASCADE",
        ),
    },
    "rag_artifact_owner_leases": {
        (
            "rag_citation_traces",
            ("profile_id", "trace_id"),
            ("profile_id", "trace_id"),
            "RESTRICT",
        ),
    },
    "rag_artifact_owner_operations": {
        (
            "rag_artifact_owner_leases",
            (
                "profile_id",
                "artifact_store_id",
                "artifact_id",
                "artifact_revision",
                "trace_id",
            ),
            (
                "profile_id",
                "artifact_store_id",
                "artifact_id",
                "artifact_revision",
                "trace_id",
            ),
            "RESTRICT",
        ),
    },
    "rag_legacy_migration_journal": {
        ("conversations", ("conversation_id",), ("id",), "CASCADE"),
    },
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


def _insert_local_trace_and_snapshot(connection: sqlite3.Connection) -> str:
    context = connection.execute("SELECT * FROM rag_identity_context").fetchone()
    profile_id = context["profile_id"]
    connection.execute(
        """
        INSERT INTO rag_citation_traces(
            profile_id, trace_id, schema_version, request_id, generation_id,
            origin_scope_id, origin, lifecycle, completeness_at_seal,
            selected_attempt_id, policy_version, aggregate_json,
            visibility_state, created_at, sealed_at
        ) VALUES (
            ?, 'trace', 1, 'request', 'generation', ?, 'local', 'sealed',
            'complete', 'attempt', 'policy', '{}', 'active',
            '2026-07-24T00:00:00Z', '2026-07-24T00:00:00Z'
        )
        """,
        (profile_id, profile_id),
    )
    connection.execute(
        """
        INSERT INTO rag_evidence_snapshots(
            profile_id, payload_id, governance_scope_id, authority_id,
            confidentiality_policy_id, revocation_scope_id,
            origin_namespace, origin_payload_id, storage_mode,
            redaction_state, retention_class, created_at
        ) VALUES (
            ?, 'snapshot', ?, ?, 'policy', 'revocation',
            'local_payload_v1', 'snapshot', 'redacted',
            'redacted', 'default', '2026-07-24T00:00:00Z'
        )
        """,
        (profile_id, profile_id, context["local_authority_id"]),
    )
    return profile_id


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

    type_names = {"T": "TEXT", "I": "INTEGER"}
    for table, compact_contract in TABLE_INFO_CONTRACT.items():
        expected = [
            (name, type_names[type_code], int(not_null), int(primary_key_ordinal))
            for token in compact_contract.split()
            for name, type_code, not_null, primary_key_ordinal in (token.split(":"),)
        ]
        actual = [
            (row["name"], row["type"], row["notnull"], row["pk"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        ]
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


def test_schema_has_every_composite_and_partial_index(tmp_path: Path) -> None:
    db = _fresh_db(tmp_path / "indexes.sqlite")
    connection = db.get_connection()

    for table in PROVENANCE_TABLES:
        actual = set()
        for index in connection.execute(f"PRAGMA index_list({table})"):
            columns = tuple(
                row["name"]
                for row in connection.execute(f"PRAGMA index_info({index['name']})")
            )
            if len(columns) > 1 or index["partial"]:
                actual.add((index["unique"], index["partial"], columns))
        assert actual == COMPOSITE_OR_PARTIAL_INDEX_CONTRACT.get(table, set())

    expected_partial_predicates = {
        "rag_citation_traces_server_identity_uq": "where origin = 'server'",
        "rag_citation_traces_import_identity_uq": "where origin = 'imported'",
        "rag_evidence_snapshots_content_dedupe_uq": ("where content_hash is not null"),
        "rag_message_trace_owners_active_message_uq": "where state = 'active'",
    }
    table_placeholders = ",".join("?" for _ in PROVENANCE_TABLES)
    actual_partial_indexes = {
        row["name"]: " ".join(row["sql"].lower().split())
        for row in connection.execute(
            f"""
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'index'
              AND sql LIKE '%WHERE%'
              AND tbl_name IN ({table_placeholders})
            """,
            tuple(PROVENANCE_TABLES),
        )
    }
    assert actual_partial_indexes.keys() == expected_partial_predicates.keys()
    for index_name, predicate in expected_partial_predicates.items():
        assert actual_partial_indexes[index_name].endswith(predicate)


def test_schema_has_exact_grouped_foreign_keys_and_delete_policies(
    tmp_path: Path,
) -> None:
    db = _fresh_db(tmp_path / "foreign-keys.sqlite")
    connection = db.get_connection()

    for table in PROVENANCE_TABLES:
        grouped: dict[int, list[sqlite3.Row]] = {}
        for row in connection.execute(f"PRAGMA foreign_key_list({table})"):
            grouped.setdefault(row["id"], []).append(row)
        actual = {
            (
                rows[0]["table"],
                tuple(
                    row["from"] for row in sorted(rows, key=lambda item: item["seq"])
                ),
                tuple(row["to"] for row in sorted(rows, key=lambda item: item["seq"])),
                rows[0]["on_delete"],
            )
            for rows in grouped.values()
        }
        assert actual == FOREIGN_KEY_CONTRACT.get(table, set())


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


def test_source_observation_prompt_set_identifier_is_utf8_bounded(
    tmp_path: Path,
) -> None:
    db = _fresh_db(tmp_path / "source-observation-prompt-bound.sqlite")
    connection = db.get_connection()
    profile_id = _insert_local_trace_and_snapshot(connection)

    connection.execute(
        """
        INSERT INTO rag_source_observations VALUES (
            ?, 'trace', ?, 1, 'snapshot', 'resolver', 'v1',
            'available', 'allowed', 'same', 'same', '{}', 'nonce',
            '2026-07-24T00:00:00Z', NULL
        )
        """,
        (profile_id, "x" * 256),
    )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO rag_source_observations VALUES (
                ?, 'trace', ?, 1, 'snapshot', 'resolver', 'v1',
                'available', 'allowed', 'same', 'same', '{}', 'nonce',
                '2026-07-24T00:00:00Z', NULL
            )
            """,
            (profile_id, "é" * 129),
        )


def test_representative_run_state_check_rejects_unknown_state(
    tmp_path: Path,
) -> None:
    db = _fresh_db(tmp_path / "run-state.sqlite")
    connection = db.get_connection()
    profile_id = _insert_local_trace_and_snapshot(connection)

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO rag_evidence_runs VALUES (
                ?, 'trace', 'run', 1, 'retrieval', 'unknown',
                '{}', '2026-07-24T00:00:00Z', NULL, NULL
            )
            """,
            (profile_id,),
        )


def test_preexisting_partial_provenance_schema_is_rejected_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial-provenance.sqlite"
    _minimal_v24(path)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE rag_evidence_runs(marker TEXT)")

    with pytest.raises(
        CharactersRAGDBError,
        match="Migration from V24 to V25 failed",
    ):
        _fresh_db(path)

    with sqlite3.connect(path) as connection:
        assert _version(connection) == 24
        assert PROVENANCE_TABLES & _table_names(connection) == {"rag_evidence_runs"}
        assert connection.execute(
            "PRAGMA table_info(rag_evidence_runs)"
        ).fetchall() == [(0, "marker", "TEXT", 0, None, 0)]


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
