"""ChaChaNotes v53 -> v56 semantic-trace storage migration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.Chat.console_trace_models import new_opaque_id
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


SCHEMA_NAME = CharactersRAGDB._SCHEMA_NAME

TRACE_TABLES = {
    "console_trace_artifacts",
    "console_trace_calls",
    "console_trace_events",
    "console_trace_graph_epoch",
    "console_trace_header_components",
    "console_trace_maintenance_state",
    "console_trace_migration_state",
    "console_trace_owners",
    "console_trace_policies",
    "console_trace_redaction_spans",
    "console_trace_request_headers",
    "console_trace_response_links",
    "console_trace_revision_bindings",
    "console_trace_segments",
    "console_trace_semantic_revisions",
    "console_trace_surface_nodes",
    "console_trace_surface_replacements",
}

EXPECTED_INDEXES = {
    "idx_console_trace_artifacts_identity",
    "idx_console_trace_calls_owner_order",
    "idx_console_trace_calls_segment_order",
    "idx_console_trace_events_call_order",
    "idx_console_trace_events_segment_order",
    "idx_console_trace_header_components_artifact",
    "idx_console_trace_migration_status",
    "idx_console_trace_owners_root_segment",
    "idx_console_trace_redaction_artifact",
    "idx_console_trace_redaction_revision",
    "idx_console_trace_response_artifact",
    "idx_console_trace_response_revision",
    "idx_console_trace_revision_bindings_artifact",
    "idx_console_trace_segments_parent_boundary",
    "idx_console_trace_semantic_revisions_source",
    "idx_console_trace_surface_nodes_predecessor",
    "idx_console_trace_surface_nodes_segment_order",
    "idx_console_trace_surface_replacements_predecessor",
    "uq_console_trace_calls_idempotency",
    "uq_console_trace_calls_owner_sequence",
    "uq_console_trace_semantic_revisions_live_message",
}

IMMUTABLE_TABLES = {
    "console_trace_artifacts",
    "console_trace_events",
    "console_trace_header_components",
    "console_trace_policies",
    "console_trace_redaction_spans",
    "console_trace_request_headers",
    "console_trace_response_links",
    "console_trace_revision_bindings",
    "console_trace_segments",
    "console_trace_surface_nodes",
    "console_trace_surface_replacements",
}

EXPECTED_TRIGGERS = {
    *(f"{table}_no_update" for table in IMMUTABLE_TABLES),
    *(f"{table}_no_delete" for table in TRACE_TABLES),
    "console_trace_calls_insert_reserved",
    "console_trace_calls_binding_guard",
    "console_trace_calls_lifecycle_guard",
    "console_trace_calls_owner_lineage",
    "console_trace_calls_set_once_guard",
    "console_trace_calls_immutable_guard",
    "console_trace_calls_terminal_guard",
    "console_trace_events_append_order",
    "console_trace_events_lineage_guard",
    "console_trace_events_owner_guard",
    "console_trace_events_shape_guard",
    "console_trace_graph_epoch_monotonic",
    "console_trace_maintenance_state_immutable_key",
    "console_trace_migration_state_immutable_key",
    "console_trace_owners_detach_only",
    "console_trace_owners_active_prefix_guard",
    "console_trace_owners_empty_root_guard",
    "console_trace_response_links_owner_guard",
    "console_trace_semantic_revisions_lineage",
    "console_trace_semantic_revisions_locator_only",
    "console_trace_segments_inherited_surface",
    "console_trace_segments_parent_owner_guard",
    "console_trace_surface_nodes_contiguous",
    "console_trace_surface_nodes_owner_guard",
    "console_trace_surface_replacements_lineage",
    "console_trace_surface_replacements_owner_guard",
}


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _object_names(connection: sqlite3.Connection, object_type: str) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ?", (object_type,)
        )
    }


def _seed_genuine_v53(path: Path) -> tuple[str, str, tuple[object, ...]]:
    """Build a real v53 profile through the production historical chain."""
    with chachanotes_db_at_version(path, 53, client_id="v56-historical-seed") as db:
        conversation_id = db.add_conversation({"title": "v53 sentinel"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "ordinary message content must remain byte-identical",
            }
        )
        assert conversation_id is not None
        assert message_id is not None
        db.append_message_exchanges_local(
            message_id,
            [
                {
                    "run_tag": "v53-existing-capture",
                    "seq": 0,
                    "status": "complete",
                    "abandoned": False,
                    "capture_detail": "full",
                    "capture_blob": b"existing-message-exchange-bytes",
                    "created_at": "2026-08-28T00:00:00Z",
                }
            ],
        )
        connection = db.get_connection()
        before = tuple(
            connection.execute(
                """
                SELECT m.content, e.run_tag, e.seq, e.status, e.capture_detail,
                       e.capture_blob, e.created_at
                  FROM messages AS m
                  JOIN message_exchanges AS e ON e.message_id = m.id
                 WHERE m.id = ?
                """,
                (message_id,),
            ).fetchone()
        )
        assert _version(connection) == 53
        assert TRACE_TABLES.isdisjoint(_object_names(connection, "table"))
    return conversation_id, message_id, before


def _insert_minimal_graph(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
    message_id: str,
) -> dict[str, str]:
    ids = {
        "segment": "00000000-0000-4000-8000-000000000001",
        "owner": "00000000-0000-4000-8000-000000000002",
        "policy": "00000000-0000-4000-8000-000000000003",
        "revision": "00000000-0000-4000-8000-000000000004",
        "artifact": "00000000-0000-4000-8000-000000000005",
        "node": "00000000-0000-4000-8000-000000000006",
        "header": "00000000-0000-4000-8000-000000000007",
        "call": "00000000-0000-4000-8000-000000000008",
        "event": "00000000-0000-4000-8000-000000000009",
        "response": "00000000-0000-4000-8000-000000000010",
        "span": "00000000-0000-4000-8000-000000000011",
        "turn": "00000000-0000-4000-8000-000000000012",
        "run": "00000000-0000-4000-8000-000000000013",
    }
    connection.execute(
        "INSERT INTO console_trace_segments(segment_id) VALUES (?)",
        (ids["segment"],),
    )
    connection.execute(
        """
        INSERT INTO console_trace_owners(
            owner_id, conversation_id, root_segment_id
        ) VALUES (?, ?, ?)
        """,
        (ids["owner"], conversation_id, ids["segment"]),
    )
    connection.execute(
        """
        INSERT INTO console_trace_policies(
            policy_id, credential_filter_version, pii_redaction_enabled
        ) VALUES (?, 'credentials-v1', 0)
        """,
        (ids["policy"],),
    )
    initial_revision = connection.execute(
        """
        SELECT revision_id
          FROM console_trace_semantic_revisions
         WHERE live_message_id = ?
        """,
        (message_id,),
    ).fetchone()
    if initial_revision is None:
        connection.execute(
            """
            INSERT INTO console_trace_semantic_revisions(
                revision_id, source_conversation_id, source_message_id,
                revision_sequence, normalized_role, content_kind,
                creation_reason, live_message_id
            ) VALUES (?, ?, ?, 0, 'user', 'text', 'capture', ?)
            """,
            (ids["revision"], conversation_id, message_id, message_id),
        )
    else:
        ids["revision"] = str(initial_revision[0])
    connection.execute(
        """
        INSERT INTO console_trace_artifacts(
            artifact_id, identity_digest, media_type,
            normalization_version, sanitized_bytes, byte_length
        ) VALUES (?, ?, 'application/json', 'v1', ?, ?)
        """,
        (ids["artifact"], "a" * 64, b'{"sanitized":true}', 18),
    )
    connection.execute(
        """
        INSERT INTO console_trace_revision_bindings(
            revision_id, policy_id, binding_outcome, artifact_id
        ) VALUES (?, ?, 'artifact', ?)
        """,
        (ids["revision"], ids["policy"], ids["artifact"]),
    )
    connection.execute(
        """
        INSERT INTO console_trace_surface_nodes(
            node_id, segment_id, sequence, component_kind,
            reference_kind, semantic_revision_id
        ) VALUES (?, ?, 0, 'message', 'revision', ?)
        """,
        (ids["node"], ids["segment"], ids["revision"]),
    )
    connection.execute(
        """
        INSERT INTO console_trace_request_headers(
            header_id, provider_name, model_name, route_identity,
            endpoint_identity, generation_parameters_json,
            adapter_defaults_json, response_format_json,
            reasoning_controls_json
        ) VALUES (?, 'openai', 'gpt-test', 'primary', 'https://example.invalid/v1',
                  '{}', '{}', '{}', '{}')
        """,
        (ids["header"],),
    )
    connection.execute(
        """
        INSERT INTO console_trace_header_components(
            header_id, component_kind, ordinal, artifact_id
        ) VALUES (?, 'tool_schema', 0, ?)
        """,
        (ids["header"], ids["artifact"]),
    )
    connection.execute(
        """
        INSERT INTO console_trace_calls(
            call_id, owner_id, segment_id, turn_id, run_id,
            call_sequence, idempotency_key, policy_id
        ) VALUES (?, ?, ?, ?, ?, 0, 'idempotency-key-1', ?)
        """,
        (
            ids["call"],
            ids["owner"],
            ids["segment"],
            ids["turn"],
            ids["run"],
            ids["policy"],
        ),
    )
    connection.execute(
        """
        INSERT INTO console_trace_events(
            event_id, segment_id, sequence, event_type, surface_node_id
        ) VALUES (?, ?, 0, 'surface_append', ?)
        """,
        (ids["event"], ids["segment"], ids["node"]),
    )
    connection.execute(
        """
        INSERT INTO console_trace_response_links(
            response_link_id, call_id, link_kind, artifact_id,
            verification_outcome
        ) VALUES (?, ?, 'artifact', ?, 'sanitized_artifact')
        """,
        (ids["response"], ids["call"], ids["artifact"]),
    )
    connection.execute(
        """
        INSERT INTO console_trace_redaction_spans(
            span_id, policy_id, source_kind, semantic_revision_id,
            field_path, start_codepoint, end_codepoint, category,
            rule_id, detector_version, outcome
        ) VALUES (?, ?, 'revision', ?, '$.content', 1, 4,
                  'credential', 'builtin-api-key', 'credentials-v1', 'applied')
        """,
        (ids["span"], ids["policy"], ids["revision"]),
    )
    return ids


def _insert_legacy_message_without_revision(
    db: CharactersRAGDB,
    *,
    conversation_id: str,
    sender: str,
    content: str,
) -> str:
    """Seed the pre-coordinator row shape needed by ledger-constraint tests."""

    message_id = new_opaque_id()
    db.get_connection().execute(
        "INSERT INTO messages(id, conversation_id, sender, content, client_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (message_id, conversation_id, sender, content, db.client_id),
    )
    return message_id


def _insert_trace_event(
    connection: sqlite3.Connection,
    *,
    event_id: str,
    segment_id: str,
    sequence: int,
    event_type: str,
    turn_id: str | None = None,
    call_id: str | None = None,
    surface_node_id: str | None = None,
    surface_replacement_id: str | None = None,
    request_header_id: str | None = None,
    semantic_revision_id: str | None = None,
    artifact_id: str | None = None,
    omission_reason_code: str | None = None,
) -> None:
    """Insert one trace event through the public SQL boundary."""
    connection.execute(
        """
        INSERT INTO console_trace_events(
            event_id, segment_id, sequence, event_type, turn_id, call_id,
            surface_node_id, surface_replacement_id, request_header_id,
            semantic_revision_id, artifact_id, omission_reason_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            segment_id,
            sequence,
            event_type,
            turn_id,
            call_id,
            surface_node_id,
            surface_replacement_id,
            request_header_id,
            semantic_revision_id,
            artifact_id,
            omission_reason_code,
        ),
    )


def test_genuine_v53_upgrade_is_fast_ddl_only_and_reopen_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "genuine-v53.sqlite"
    conversation_id, message_id, before = _seed_genuine_v53(path)

    with chachanotes_db_at_version(path, 56, client_id="v56-upgrade") as migrated:
        connection = migrated.get_connection()
        assert _version(connection) == 56
        after = tuple(
            connection.execute(
                """
                SELECT m.content, e.run_tag, e.seq, e.status, e.capture_detail,
                       e.capture_blob, e.created_at
                  FROM messages AS m
                  JOIN message_exchanges AS e ON e.message_id = m.id
                 WHERE m.id = ? AND m.conversation_id = ?
                """,
                (message_id, conversation_id),
            ).fetchone()
        )
        assert after == before
        first_schema = {
            (str(row[0]), str(row[1]), str(row[2]))
            for row in connection.execute(
                """
                SELECT type, name, sql FROM sqlite_master
                 WHERE name LIKE 'console_trace_%'
                 ORDER BY type, name
                """
            )
        }
    with chachanotes_db_at_version(path, 56, client_id="v56-reopen") as reopened:
        connection = reopened.get_connection()
        assert _version(connection) == 56
        second_schema = {
            (str(row[0]), str(row[1]), str(row[2]))
            for row in connection.execute(
                """
                SELECT type, name, sql FROM sqlite_master
                 WHERE name LIKE 'console_trace_%'
                 ORDER BY type, name
                """
            )
        }
        assert second_schema == first_schema


def test_fresh_v56_schema_has_complete_objects_and_initial_state(
    tmp_path: Path,
) -> None:
    with chachanotes_db_at_version(
        tmp_path / "fresh-v56.sqlite", 56, client_id="v56-fresh"
    ) as db:
        connection = db.get_connection()
        assert _version(connection) == 56
        assert TRACE_TABLES <= _object_names(connection, "table")
        assert EXPECTED_INDEXES <= _object_names(connection, "index")
        assert EXPECTED_TRIGGERS <= _object_names(connection, "trigger")
        assert tuple(
            connection.execute(
                "SELECT singleton_id, epoch FROM console_trace_graph_epoch"
            ).fetchone()
        ) == (1, 0)
        assert tuple(
            connection.execute(
                """
                SELECT migration_name, status, last_exchange_id,
                       processed_rows, processed_bytes
                  FROM console_trace_migration_state
                """
            ).fetchone()
        ) == ("legacy_exchange_normalization", "pending", None, 0, 0)
        assert tuple(
            connection.execute(
                """
                SELECT singleton_id, state, lease_id, lease_owner,
                       lease_expires_at, marked_epoch
                  FROM console_trace_maintenance_state
                """
            ).fetchone()
        ) == (1, "idle", None, None, None, None)
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert list(connection.execute("PRAGMA foreign_key_check")) == []


def test_trace_schema_has_foreign_keys_and_no_forbidden_content_metadata(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "v56-shape.sqlite", client_id="v56-shape")
    try:
        connection = db.get_connection()
        foreign_key_tables = {
            table
            for table in TRACE_TABLES
            if connection.execute(f"PRAGMA foreign_key_list({table})").fetchone()
        }
        assert foreign_key_tables == TRACE_TABLES - {
            "console_trace_artifacts",
            "console_trace_graph_epoch",
            "console_trace_maintenance_state",
            "console_trace_migration_state",
            "console_trace_policies",
            "console_trace_request_headers",
        }

        columns = {
            table: {
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
            }
            for table in TRACE_TABLES
        }
        assert {
            "identity_digest",
            "sanitized_bytes",
            "media_type",
            "normalization_version",
        } <= columns["console_trace_artifacts"]
        assert "inherited_surface_head_id" in columns["console_trace_segments"]
        forbidden_names = {
            "body",
            "canonical_body",
            "canonical_content",
            "canonical_digest",
            "content_digest",
            "message_digest",
            "history",
            "history_json",
            "source_ids",
            "source_ids_json",
            "shadowed_sources",
            "shadowed_sources_json",
            "matched_value",
            "matched_hash",
            "matched_substring",
            "regex",
            "regex_text",
            "pattern",
            "pattern_text",
        }
        for table, table_columns in columns.items():
            assert forbidden_names.isdisjoint(table_columns), table
        assert not any(
            "digest" in column
            for table, table_columns in columns.items()
            if table != "console_trace_artifacts"
            for column in table_columns
        )
        assert not any(
            "history" in column or "shadow" in column
            for table_columns in columns.values()
            for column in table_columns
        )
    finally:
        db.close_connection()


def test_trace_uniqueness_checks_and_structural_foreign_keys(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "v56-constraints.sqlite", client_id="v56-fk")
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "constraints"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "canonical content stays in messages",
            }
        )
        assert conversation_id is not None
        assert message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )

        connection.execute(
            """
            INSERT INTO console_trace_artifacts(
                artifact_id, identity_digest, media_type,
                normalization_version, sanitized_bytes, byte_length
            ) VALUES ('00000000-0000-4000-8000-000000000099', ?,
                      'application/json', 'v1', ?, ?)
            """,
            ("a" * 64, b'{"collision":true}', 18),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO console_trace_artifacts(
                    artifact_id, identity_digest, media_type,
                    normalization_version, sanitized_bytes, byte_length
                ) VALUES ('00000000-0000-4000-8000-000000000088', ?,
                          'application/json', 'v1', ?, ?)
                """,
                ("A" * 64, b'{"sanitized":true}', 18),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO console_trace_calls(
                    call_id, owner_id, segment_id, turn_id, run_id,
                    call_sequence, idempotency_key, policy_id
                ) VALUES ('00000000-0000-4000-8000-000000000098', ?, ?, ?, ?,
                          1, 'idempotency-key-1', ?)
                """,
                (
                    ids["owner"],
                    ids["segment"],
                    ids["turn"],
                    ids["run"],
                    ids["policy"],
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO console_trace_events(
                    event_id, segment_id, sequence, event_type, call_id
                ) VALUES ('00000000-0000-4000-8000-000000000097', ?, 0,
                          'call_boundary', ?)
                """,
                (ids["segment"], ids["call"]),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO console_trace_semantic_revisions(
                    revision_id, source_conversation_id, source_message_id,
                    revision_sequence, normalized_role, content_kind,
                    creation_reason, live_message_id
                ) VALUES ('00000000-0000-4000-8000-000000000096', ?, ?, 1,
                          'user', 'text', 'edit', 'missing-message')
                """,
                (conversation_id, message_id),
            )

        child_segment = "00000000-0000-4000-8000-000000000095"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (child_segment, ids["segment"], ids["node"]),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO console_trace_segments(
                    segment_id, parent_segment_id, inherited_through_sequence,
                    inherited_surface_head_id
                ) VALUES ('00000000-0000-4000-8000-000000000094', ?, 9, ?)
                """,
                (ids["segment"], ids["node"]),
            )

        second_node = "00000000-0000-4000-8000-000000000093"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 1, ?, 'provider_context', 'artifact', ?)
            """,
            (second_node, ids["segment"], ids["node"], ids["artifact"]),
        )
        child_node = "00000000-0000-4000-8000-000000000090"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 1, ?, 'provider_context', 'artifact', ?)
            """,
            (child_node, child_segment, ids["node"], ids["artifact"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_surface_replacements(
                replacement_id, segment_id, predecessor_head_id,
                start_node_id, start_sequence, end_node_id, end_sequence,
                replacement_node_id
            ) VALUES ('00000000-0000-4000-8000-000000000089', ?, ?, ?, 0,
                      ?, 0, ?)
            """,
            (
                child_segment,
                ids["node"],
                ids["node"],
                ids["node"],
                child_node,
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="contiguous"):
            connection.execute(
                """
                INSERT INTO console_trace_surface_nodes(
                    node_id, segment_id, sequence, predecessor_node_id,
                    component_kind, reference_kind, artifact_id
                ) VALUES ('00000000-0000-4000-8000-000000000092', ?, 3, ?,
                          'provider_context', 'artifact', ?)
                """,
                (ids["segment"], second_node, ids["artifact"]),
            )
        predecessor = second_node
        for sequence in range(2, 302):
            node_id = f"long-range-node-{sequence}"
            connection.execute(
                """
                INSERT INTO console_trace_surface_nodes(
                    node_id, segment_id, sequence, predecessor_node_id,
                    component_kind, reference_kind, artifact_id
                ) VALUES (?, ?, ?, ?, 'provider_context', 'artifact', ?)
                """,
                (node_id, ids["segment"], sequence, predecessor, ids["artifact"]),
            )
            predecessor = node_id
        connection.execute(
            """
            INSERT INTO console_trace_surface_replacements(
                replacement_id, segment_id, predecessor_head_id,
                start_node_id, start_sequence, end_node_id, end_sequence,
                replacement_node_id
            ) VALUES ('00000000-0000-4000-8000-000000000091', ?, ?, ?, 0,
                      ?, 300, ?)
            """,
            (
                ids["segment"],
                "long-range-node-300",
                ids["node"],
                "long-range-node-300",
                "long-range-node-301",
            ),
        )
    finally:
        db.close_connection()


def test_surface_nodes_and_replacements_reject_unrelated_lineage(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        tmp_path / "v56-surface-lineage.sqlite", client_id="v56-surface"
    )
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "surface lineage"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "canonical",
            }
        )
        assert conversation_id is not None
        assert message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )

        parent_second = "surface-parent-second"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 1, ?, 'provider_context', 'artifact', ?)
            """,
            (parent_second, ids["segment"], ids["node"], ids["artifact"]),
        )
        child_segment = "surface-child-segment"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (child_segment, ids["segment"], ids["node"]),
        )
        child_node = "surface-child-node"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 1, ?, 'provider_context', 'artifact', ?)
            """,
            (child_node, child_segment, ids["node"], ids["artifact"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_surface_replacements(
                replacement_id, segment_id, predecessor_head_id,
                start_node_id, start_sequence, end_node_id, end_sequence,
                replacement_node_id
            ) VALUES ('surface-valid-inherited-replacement', ?, ?, ?, 0, ?, 0, ?)
            """,
            (child_segment, ids["node"], ids["node"], ids["node"], child_node),
        )

        unrelated_segment = "surface-unrelated-segment"
        unrelated_node = "surface-unrelated-node"
        connection.execute(
            "INSERT INTO console_trace_segments(segment_id) VALUES (?)",
            (unrelated_segment,),
        )
        unrelated_conversation = db.add_conversation(
            {"title": "surface unrelated owner"}
        )
        assert unrelated_conversation is not None
        connection.execute(
            """
            INSERT INTO console_trace_owners(owner_id, conversation_id, root_segment_id)
            VALUES ('surface-unrelated-owner', ?, ?)
            """,
            (unrelated_conversation, unrelated_segment),
        )
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, component_kind,
                reference_kind, artifact_id
            ) VALUES (?, ?, 0, 'provider_context', 'artifact', ?)
            """,
            (unrelated_node, unrelated_segment, ids["artifact"]),
        )

        unrelated_child = "surface-unrelated-child"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (unrelated_child, ids["segment"], ids["node"]),
        )
        with pytest.raises(sqlite3.IntegrityError, match="lineage"):
            connection.execute(
                """
                INSERT INTO console_trace_surface_nodes(
                    node_id, segment_id, sequence, predecessor_node_id,
                    component_kind, reference_kind, artifact_id
                ) VALUES ('surface-invalid-unrelated-predecessor', ?, 1, ?,
                          'provider_context', 'artifact', ?)
                """,
                (unrelated_child, unrelated_node, ids["artifact"]),
            )

        boundary_child = "surface-boundary-child"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (boundary_child, ids["segment"], ids["node"]),
        )
        with pytest.raises(sqlite3.IntegrityError, match="lineage"):
            connection.execute(
                """
                INSERT INTO console_trace_surface_nodes(
                    node_id, segment_id, sequence, predecessor_node_id,
                    component_kind, reference_kind, artifact_id
                ) VALUES ('surface-invalid-past-boundary', ?, 2, ?,
                          'provider_context', 'artifact', ?)
                """,
                (boundary_child, parent_second, ids["artifact"]),
            )

        connection.execute(
            """
            INSERT INTO console_trace_events(
                event_id, segment_id, sequence, event_type, surface_node_id
            ) VALUES ('surface-root-event-one', ?, 1, 'surface_append', ?)
            """,
            (ids["segment"], parent_second),
        )
        middle_segment = "surface-middle-segment"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 1, ?)
            """,
            (middle_segment, ids["segment"], parent_second),
        )
        middle_node = "surface-middle-node"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 2, ?, 'provider_context', 'artifact', ?)
            """,
            (middle_node, middle_segment, parent_second, ids["artifact"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_events(
                event_id, segment_id, sequence, event_type, surface_node_id
            ) VALUES ('surface-middle-event-zero', ?, 0, 'surface_append', ?)
            """,
            (middle_segment, middle_node),
        )
        grandchild_segment = "surface-grandchild-segment"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (grandchild_segment, middle_segment, middle_node),
        )
        grandchild_node = "surface-grandchild-node"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 3, ?, 'provider_context', 'artifact', ?)
            """,
            (grandchild_node, grandchild_segment, middle_node, ids["artifact"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_surface_replacements(
                replacement_id, segment_id, predecessor_head_id,
                start_node_id, start_sequence, end_node_id, end_sequence,
                replacement_node_id
            ) VALUES ('surface-valid-deep-inherited-replacement', ?, ?, ?, 0,
                      ?, 2, ?)
            """,
            (
                grandchild_segment,
                middle_node,
                ids["node"],
                middle_node,
                grandchild_node,
            ),
        )

        parent_later = "surface-parent-later"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 2, ?, 'provider_context', 'artifact', ?)
            """,
            (parent_later, ids["segment"], parent_second, ids["artifact"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_events(
                event_id, segment_id, sequence, event_type, surface_node_id
            ) VALUES ('surface-root-event-two', ?, 2, 'surface_append', ?)
            """,
            (ids["segment"], parent_later),
        )
        with pytest.raises(sqlite3.IntegrityError, match="inherited surface"):
            connection.execute(
                """
                INSERT INTO console_trace_segments(
                    segment_id, parent_segment_id, inherited_through_sequence,
                    inherited_surface_head_id
                ) VALUES ('surface-invalid-later-head', ?, 1, ?)
                """,
                (ids["segment"], parent_later),
            )
        with pytest.raises(sqlite3.IntegrityError, match="inherited surface"):
            connection.execute(
                """
                INSERT INTO console_trace_segments(
                    segment_id, parent_segment_id, inherited_through_sequence,
                    inherited_surface_head_id
                ) VALUES ('surface-invalid-unrelated-head', ?, 1, ?)
                """,
                (ids["segment"], unrelated_node),
            )
        with pytest.raises(sqlite3.IntegrityError, match="lineage"):
            connection.execute(
                """
                INSERT INTO console_trace_surface_nodes(
                    node_id, segment_id, sequence, predecessor_node_id,
                    component_kind, reference_kind, artifact_id
                ) VALUES ('surface-invalid-later-parent-node', ?, 3, ?,
                          'provider_context', 'artifact', ?)
                """,
                (middle_segment, parent_later, ids["artifact"]),
            )

        with pytest.raises(sqlite3.IntegrityError, match="lineage"):
            connection.execute(
                """
                INSERT INTO console_trace_surface_replacements(
                    replacement_id, segment_id, predecessor_head_id,
                    start_node_id, start_sequence, end_node_id, end_sequence,
                    replacement_node_id
                ) VALUES ('surface-invalid-replacement-head', ?, ?, ?, 0, ?, 0, ?)
                """,
                (
                    child_segment,
                    unrelated_node,
                    unrelated_node,
                    unrelated_node,
                    child_node,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="lineage"):
            connection.execute(
                """
                INSERT INTO console_trace_surface_replacements(
                    replacement_id, segment_id, predecessor_head_id,
                    start_node_id, start_sequence, end_node_id, end_sequence,
                    replacement_node_id
                ) VALUES ('surface-invalid-replacement-node', ?, ?, ?, 0, ?, 0, ?)
                """,
                (
                    child_segment,
                    ids["node"],
                    ids["node"],
                    ids["node"],
                    unrelated_node,
                ),
            )

        child_second = "surface-child-second"
        child_third = "surface-child-third"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 2, ?, 'provider_context', 'artifact', ?)
            """,
            (child_second, child_segment, child_node, ids["artifact"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 3, ?, 'provider_context', 'artifact', ?)
            """,
            (child_third, child_segment, child_second, ids["artifact"]),
        )
        with pytest.raises(sqlite3.IntegrityError, match="lineage"):
            connection.execute(
                """
                INSERT INTO console_trace_surface_replacements(
                    replacement_id, segment_id, predecessor_head_id,
                    start_node_id, start_sequence, end_node_id, end_sequence,
                    replacement_node_id
                ) VALUES ('surface-invalid-sibling-range', ?, ?, ?, 1, ?, 1, ?)
                """,
                (
                    child_segment,
                    child_second,
                    parent_second,
                    parent_second,
                    child_third,
                ),
            )
    finally:
        db.close_connection()


def test_event_sequences_reject_late_backfill_after_fork_boundary(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "v56-event-order.sqlite", client_id="v56-events")
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "event order"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "canonical",
            }
        )
        assert conversation_id is not None
        assert message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )

        connection.execute(
            """
            INSERT INTO console_trace_events(
                event_id, segment_id, sequence, event_type, turn_id
            ) VALUES ('event-sequence-two', ?, 2, 'turn_boundary', 'turn-two')
            """,
            (ids["segment"],),
        )
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES ('event-order-child', ?, 2, ?)
            """,
            (ids["segment"], ids["node"]),
        )

        with pytest.raises(sqlite3.IntegrityError, match="append order"):
            connection.execute(
                """
                INSERT INTO console_trace_events(
                    event_id, segment_id, sequence, event_type, turn_id
                ) VALUES ('event-sequence-one-late', ?, 1,
                          'turn_boundary', 'turn-one-late')
                """,
                (ids["segment"],),
            )
    finally:
        db.close_connection()


def test_event_type_shapes_and_ownership_lineage_are_enforced(tmp_path: Path) -> None:
    db = CharactersRAGDB(
        tmp_path / "v56-event-shapes.sqlite", client_id="v56-event-shapes"
    )
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "event shapes"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "canonical",
            }
        )
        assert conversation_id is not None
        assert message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET surface_node_id = ?, request_header_id = ?,
                   provider_name = 'openai', model_name = 'gpt-test',
                   route_identity = 'primary'
             WHERE call_id = ?
            """,
            (ids["node"], ids["header"], ids["call"]),
        )
        append_node = "event-shape-append-node"
        replacement_node = "event-shape-replacement-node"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 1, ?, 'provider_context', 'artifact', ?)
            """,
            (append_node, ids["segment"], ids["node"], ids["artifact"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 2, ?, 'provider_context', 'artifact', ?)
            """,
            (replacement_node, ids["segment"], append_node, ids["artifact"]),
        )
        replacement_id = "event-shape-replacement"
        connection.execute(
            """
            INSERT INTO console_trace_surface_replacements(
                replacement_id, segment_id, predecessor_head_id,
                start_node_id, start_sequence, end_node_id, end_sequence,
                replacement_node_id
            ) VALUES (?, ?, ?, ?, 1, ?, 1, ?)
            """,
            (
                replacement_id,
                ids["segment"],
                append_node,
                append_node,
                append_node,
                replacement_node,
            ),
        )

        valid_events = (
            ("turn_boundary", {"turn_id": ids["turn"]}),
            ("call_boundary", {"call_id": ids["call"]}),
            ("surface_append", {"surface_node_id": append_node}),
            ("surface_replace", {"surface_replacement_id": replacement_id}),
            ("tool_call", {"call_id": ids["call"], "artifact_id": ids["artifact"]}),
            (
                "tool_result",
                {"call_id": ids["call"], "semantic_revision_id": ids["revision"]},
            ),
            (
                "request_header_selection",
                {"call_id": ids["call"], "request_header_id": ids["header"]},
            ),
            (
                "provider_route_selection",
                {"call_id": ids["call"], "request_header_id": ids["header"]},
            ),
            (
                "response_selection",
                {"call_id": ids["call"], "artifact_id": ids["artifact"]},
            ),
            ("call_outcome", {"call_id": ids["call"]}),
            ("usage", {"call_id": ids["call"]}),
            ("gap", {"omission_reason_code": "capture_unavailable"}),
        )
        for sequence, (event_type, references) in enumerate(valid_events, start=1):
            _insert_trace_event(
                connection,
                event_id=f"valid-{event_type}",
                segment_id=ids["segment"],
                sequence=sequence,
                event_type=event_type,
                **references,
            )

        invalid_shapes: tuple[tuple[str, dict[str, str]], ...] = (
            ("turn_boundary", {}),
            ("turn_boundary", {"turn_id": ids["turn"], "artifact_id": ids["artifact"]}),
            ("call_boundary", {}),
            ("call_boundary", {"call_id": ids["call"], "turn_id": ids["turn"]}),
            ("surface_append", {}),
            (
                "surface_append",
                {"surface_node_id": append_node, "artifact_id": ids["artifact"]},
            ),
            ("surface_replace", {}),
            (
                "surface_replace",
                {
                    "surface_replacement_id": replacement_id,
                    "surface_node_id": replacement_node,
                },
            ),
            ("tool_call", {"call_id": ids["call"]}),
            (
                "tool_call",
                {
                    "call_id": ids["call"],
                    "artifact_id": ids["artifact"],
                    "semantic_revision_id": ids["revision"],
                },
            ),
            ("tool_result", {"call_id": ids["call"]}),
            ("request_header_selection", {"request_header_id": ids["header"]}),
            (
                "provider_route_selection",
                {
                    "call_id": ids["call"],
                    "request_header_id": ids["header"],
                    "artifact_id": ids["artifact"],
                },
            ),
            ("response_selection", {"call_id": ids["call"]}),
            (
                "response_selection",
                {
                    "call_id": ids["call"],
                    "artifact_id": ids["artifact"],
                    "semantic_revision_id": ids["revision"],
                },
            ),
            ("call_outcome", {"call_id": ids["call"], "artifact_id": ids["artifact"]}),
            ("usage", {"call_id": ids["call"], "artifact_id": ids["artifact"]}),
            ("gap", {}),
            ("gap", {"omission_reason_code": "gap", "artifact_id": ids["artifact"]}),
        )
        next_sequence = len(valid_events) + 1
        for ordinal, (event_type, references) in enumerate(invalid_shapes):
            with pytest.raises(sqlite3.IntegrityError, match="trace event"):
                _insert_trace_event(
                    connection,
                    event_id=f"invalid-shape-{ordinal}",
                    segment_id=ids["segment"],
                    sequence=next_sequence,
                    event_type=event_type,
                    **references,
                )

        unrelated_segment = "event-unrelated-segment"
        unrelated_node = "event-unrelated-node"
        unrelated_replacement_node = "event-unrelated-replacement-node"
        unrelated_replacement = "event-unrelated-replacement"
        connection.execute(
            "INSERT INTO console_trace_segments(segment_id) VALUES (?)",
            (unrelated_segment,),
        )
        unrelated_conversation = db.add_conversation({"title": "unrelated owner"})
        assert unrelated_conversation is not None
        connection.execute(
            """
            INSERT INTO console_trace_owners(owner_id, conversation_id, root_segment_id)
            VALUES ('event-unrelated-owner', ?, ?)
            """,
            (unrelated_conversation, unrelated_segment),
        )
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, component_kind,
                reference_kind, artifact_id
            ) VALUES (?, ?, 0, 'provider_context', 'artifact', ?)
            """,
            (unrelated_node, unrelated_segment, ids["artifact"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 1, ?, 'provider_context', 'artifact', ?)
            """,
            (
                unrelated_replacement_node,
                unrelated_segment,
                unrelated_node,
                ids["artifact"],
            ),
        )
        connection.execute(
            """
            INSERT INTO console_trace_surface_replacements(
                replacement_id, segment_id, predecessor_head_id,
                start_node_id, start_sequence, end_node_id, end_sequence,
                replacement_node_id
            ) VALUES (?, ?, ?, ?, 0, ?, 0, ?)
            """,
            (
                unrelated_replacement,
                unrelated_segment,
                unrelated_node,
                unrelated_node,
                unrelated_node,
                unrelated_replacement_node,
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="event ownership lineage"):
            _insert_trace_event(
                connection,
                event_id="invalid-unrelated-surface",
                segment_id=ids["segment"],
                sequence=next_sequence,
                event_type="surface_append",
                surface_node_id=unrelated_node,
            )
        with pytest.raises(sqlite3.IntegrityError, match="event ownership lineage"):
            _insert_trace_event(
                connection,
                event_id="invalid-unrelated-replacement",
                segment_id=ids["segment"],
                sequence=next_sequence,
                event_type="surface_replace",
                surface_replacement_id=unrelated_replacement,
            )

        other_header = "event-other-header"
        connection.execute(
            """
            INSERT INTO console_trace_request_headers(
                header_id, provider_name, model_name, route_identity,
                endpoint_identity, generation_parameters_json,
                adapter_defaults_json, response_format_json,
                reasoning_controls_json
            ) VALUES (?, 'openai', 'gpt-test', 'secondary',
                      'https://example.invalid/v2', '{}', '{}', '{}', '{}')
            """,
            (other_header,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="event ownership lineage"):
            _insert_trace_event(
                connection,
                event_id="invalid-call-header-pair",
                segment_id=ids["segment"],
                sequence=next_sequence,
                event_type="request_header_selection",
                call_id=ids["call"],
                request_header_id=other_header,
            )

        unrelated_call = "event-unrelated-call"
        connection.execute(
            """
            INSERT INTO console_trace_calls(
                call_id, owner_id, segment_id, turn_id, run_id,
                call_sequence, idempotency_key, policy_id
            ) VALUES (?, 'event-unrelated-owner', ?, 'unrelated-turn',
                      'unrelated-run', 0, 'event-unrelated-call-key', ?)
            """,
            (unrelated_call, unrelated_segment, ids["policy"]),
        )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET surface_node_id = ?, request_header_id = ?,
                   provider_name = 'openai', model_name = 'gpt-test',
                   route_identity = 'secondary'
             WHERE call_id = ?
            """,
            (unrelated_node, other_header, unrelated_call),
        )
        _insert_trace_event(
            connection,
            event_id="valid-globally-reused-artifact",
            segment_id=unrelated_segment,
            sequence=0,
            event_type="tool_call",
            call_id=unrelated_call,
            artifact_id=ids["artifact"],
        )

        child_segment = "event-valid-child-segment"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, ?, ?)
            """,
            (child_segment, ids["segment"], len(valid_events), replacement_node),
        )
        child_node = "event-valid-child-node"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 3, ?, 'provider_context', 'artifact', ?)
            """,
            (child_node, child_segment, replacement_node, ids["artifact"]),
        )
        child_call = "event-valid-child-call"
        connection.execute(
            """
            INSERT INTO console_trace_calls(
                call_id, owner_id, segment_id, turn_id, run_id,
                call_sequence, idempotency_key, policy_id
            ) VALUES (?, ?, ?, 'child-turn', 'child-run', 0,
                      'event-valid-child-key', ?)
            """,
            (child_call, ids["owner"], child_segment, ids["policy"]),
        )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET surface_node_id = ?, request_header_id = ?,
                   provider_name = 'openai', model_name = 'gpt-test',
                   route_identity = 'primary'
             WHERE call_id = ?
            """,
            (replacement_node, ids["header"], child_call),
        )
        _insert_trace_event(
            connection,
            event_id="valid-inherited-surface-event",
            segment_id=child_segment,
            sequence=0,
            event_type="surface_append",
            surface_node_id=child_node,
        )
        _insert_trace_event(
            connection,
            event_id="valid-inherited-header-event",
            segment_id=child_segment,
            sequence=1,
            event_type="request_header_selection",
            call_id=child_call,
            request_header_id=ids["header"],
        )
        with pytest.raises(sqlite3.IntegrityError, match="event ownership lineage"):
            _insert_trace_event(
                connection,
                event_id="invalid-cross-segment-call",
                segment_id=ids["segment"],
                sequence=next_sequence,
                event_type="call_boundary",
                call_id=child_call,
            )
    finally:
        db.close_connection()


def test_owner_roots_are_globally_reserved_across_attach_and_detach(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        tmp_path / "v56-owner-roots.sqlite", client_id="v56-owner-roots"
    )
    try:
        connection = db.get_connection()
        first_conversation = db.add_conversation({"title": "first owner"})
        second_conversation = db.add_conversation({"title": "second owner"})
        assert first_conversation is not None
        assert second_conversation is not None
        connection.execute(
            "INSERT INTO console_trace_segments(segment_id) VALUES ('owner-root-one')"
        )
        connection.execute(
            """
            INSERT INTO console_trace_owners(owner_id, conversation_id, root_segment_id)
            VALUES ('owner-one', ?, 'owner-root-one')
            """,
            (first_conversation,),
        )
        connection.execute(
            """
            INSERT INTO console_trace_events(
                event_id, segment_id, sequence, event_type, omission_reason_code
            ) VALUES ('owner-root-populated', 'owner-root-one', 0, 'gap', 'reserved')
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO console_trace_owners(
                    owner_id, conversation_id, root_segment_id
                ) VALUES ('owner-alias', ?, 'owner-root-one')
                """,
                (second_conversation,),
            )

        connection.execute(
            "INSERT INTO console_trace_segments(segment_id) VALUES ('owner-root-detached')"
        )
        connection.execute(
            """
            INSERT INTO console_trace_owners(
                owner_id, conversation_id, root_segment_id, attached, detached_at
            ) VALUES ('owner-detached-reservation', NULL, 'owner-root-detached', 0,
                      '2026-08-28T02:00:00Z')
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO console_trace_owners(
                    owner_id, conversation_id, root_segment_id
                ) VALUES ('owner-alias-detached', ?, 'owner-root-detached')
                """,
                (second_conversation,),
            )
    finally:
        db.close_connection()


def test_nearest_owner_root_controls_nested_segments_and_detached_boundaries(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        tmp_path / "v56-nearest-owner.sqlite", client_id="v56-nearest-owner"
    )
    try:
        connection = db.get_connection()
        conversation_a = db.add_conversation({"title": "owner A"})
        conversation_b = db.add_conversation({"title": "owner B"})
        conversation_c = db.add_conversation({"title": "owner C"})
        message_a = db.add_message(
            {
                "conversation_id": conversation_a,
                "sender": "user",
                "content": "canonical A",
            }
        )
        assert conversation_a is not None
        assert conversation_b is not None
        assert conversation_c is not None
        assert message_a is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_a,
            message_id=message_a,
        )

        def insert_call(
            call_id: str,
            owner_id: str,
            segment_id: str,
            call_sequence: int = 0,
        ) -> None:
            connection.execute(
                """
                INSERT INTO console_trace_calls(
                    call_id, owner_id, segment_id, turn_id, run_id,
                    call_sequence, idempotency_key, policy_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    owner_id,
                    segment_id,
                    f"{call_id}-turn",
                    f"{call_id}-run",
                    call_sequence,
                    f"{call_id}-key",
                    ids["policy"],
                ),
            )

        owner_a_descendant = "nearest-owner-a-descendant"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (owner_a_descendant, ids["segment"], ids["node"]),
        )
        insert_call("nearest-owner-a-valid", ids["owner"], owner_a_descendant)

        owner_b_root = "nearest-owner-b-root"
        owner_b = "nearest-owner-b"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (owner_b_root, ids["segment"], ids["node"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_owners(owner_id, conversation_id, root_segment_id)
            VALUES (?, ?, ?)
            """,
            (owner_b, conversation_b, owner_b_root),
        )
        insert_call("nearest-owner-b-root-valid", owner_b, owner_b_root)
        with pytest.raises(sqlite3.IntegrityError, match="effective owner"):
            insert_call("nearest-owner-a-on-b-root", ids["owner"], owner_b_root)

        owner_b_node = "nearest-owner-b-node"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 1, ?, 'provider_context', 'artifact', ?)
            """,
            (owner_b_node, owner_b_root, ids["node"], ids["artifact"]),
        )
        _insert_trace_event(
            connection,
            event_id="nearest-owner-b-surface-event",
            segment_id=owner_b_root,
            sequence=0,
            event_type="surface_append",
            surface_node_id=owner_b_node,
        )
        owner_b_descendant = "nearest-owner-b-descendant"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (owner_b_descendant, owner_b_root, owner_b_node),
        )
        insert_call("nearest-owner-b-descendant-valid", owner_b, owner_b_descendant)
        with pytest.raises(sqlite3.IntegrityError, match="effective owner"):
            insert_call(
                "nearest-owner-a-on-b-descendant", ids["owner"], owner_b_descendant
            )

        connection.execute(
            """
            UPDATE console_trace_owners
               SET attached = 0, conversation_id = NULL,
                   detached_at = '2026-08-28T04:00:00Z'
             WHERE owner_id = ?
            """,
            (owner_b,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="effective owner"):
            insert_call(
                "nearest-owner-b-detached-blocked",
                owner_b,
                owner_b_descendant,
                call_sequence=1,
            )
        with pytest.raises(sqlite3.IntegrityError, match="effective owner"):
            insert_call(
                "nearest-owner-a-behind-detached-b",
                ids["owner"],
                owner_b_descendant,
            )
        with pytest.raises(sqlite3.IntegrityError, match="active effective owner"):
            connection.execute(
                """
                INSERT INTO console_trace_surface_nodes(
                    node_id, segment_id, sequence, predecessor_node_id,
                    component_kind, reference_kind, artifact_id
                ) VALUES ('nearest-owner-detached-node', ?, 2, ?,
                          'provider_context', 'artifact', ?)
                """,
                (owner_b_descendant, owner_b_node, ids["artifact"]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="active effective owner"):
            _insert_trace_event(
                connection,
                event_id="nearest-owner-detached-event",
                segment_id=owner_b_descendant,
                sequence=0,
                event_type="gap",
                omission_reason_code="detached",
            )
        with pytest.raises(sqlite3.IntegrityError, match="active effective owner"):
            connection.execute(
                """
                INSERT INTO console_trace_response_links(
                    response_link_id, call_id, link_kind, artifact_id,
                    verification_outcome
                ) VALUES ('nearest-owner-detached-link',
                          'nearest-owner-b-descendant-valid', 'artifact', ?,
                          'sanitized_artifact')
                """,
                (ids["artifact"],),
            )

        populated_child = "nearest-owner-populated-child"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (populated_child, ids["segment"], ids["node"]),
        )
        populated_node = "nearest-owner-populated-node"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 1, ?, 'provider_context', 'artifact', ?)
            """,
            (populated_node, populated_child, ids["node"], ids["artifact"]),
        )
        with pytest.raises(sqlite3.IntegrityError, match="empty root"):
            connection.execute(
                """
                INSERT INTO console_trace_owners(
                    owner_id, conversation_id, root_segment_id
                ) VALUES ('nearest-owner-hijack', ?, ?)
                """,
                (conversation_c, populated_child),
            )
    finally:
        db.close_connection()


def test_canonical_revision_references_stay_in_the_effective_owner_domain(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        tmp_path / "v56-revision-owner-domain.sqlite",
        client_id="v56-revision-owner-domain",
    )
    try:
        connection = db.get_connection()
        conversation_a = db.add_conversation({"title": "revision owner A"})
        conversation_b = db.add_conversation({"title": "revision owner B"})
        message_a = db.add_message(
            {
                "conversation_id": conversation_a,
                "sender": "user",
                "content": "canonical A",
            }
        )
        message_b = db.add_message(
            {
                "conversation_id": conversation_b,
                "sender": "user",
                "content": "canonical B",
            }
        )
        assert conversation_a is not None
        assert conversation_b is not None
        assert message_a is not None
        assert message_b is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_a,
            message_id=message_a,
        )

        owner_b_root = "revision-domain-b-root"
        owner_b = "revision-domain-b-owner"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (owner_b_root, ids["segment"], ids["node"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_owners(owner_id, conversation_id, root_segment_id)
            VALUES (?, ?, ?)
            """,
            (owner_b, conversation_b, owner_b_root),
        )
        revision_b = str(
            connection.execute(
                """
                SELECT revision_id
                  FROM console_trace_semantic_revisions
                 WHERE live_message_id = ?
                """,
                (message_b,),
            ).fetchone()[0]
        )

        with pytest.raises(sqlite3.IntegrityError, match="matching revision domain"):
            connection.execute(
                """
                INSERT INTO console_trace_surface_nodes(
                    node_id, segment_id, sequence, predecessor_node_id,
                    component_kind, reference_kind, semantic_revision_id
                ) VALUES ('revision-domain-invalid-node', ?, 1, ?,
                          'message', 'revision', ?)
                """,
                (owner_b_root, ids["node"], ids["revision"]),
            )
        owner_b_node = "revision-domain-b-node"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, semantic_revision_id
            ) VALUES (?, ?, 1, ?, 'message', 'revision', ?)
            """,
            (owner_b_node, owner_b_root, ids["node"], revision_b),
        )
        artifact_node = "revision-domain-global-artifact-node"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES (?, ?, 2, ?, 'provider_context', 'artifact', ?)
            """,
            (artifact_node, owner_b_root, owner_b_node, ids["artifact"]),
        )
        _insert_trace_event(
            connection,
            event_id="revision-domain-surface-event",
            segment_id=owner_b_root,
            sequence=0,
            event_type="surface_append",
            surface_node_id=artifact_node,
        )

        call_b = "revision-domain-b-call"
        connection.execute(
            """
            INSERT INTO console_trace_calls(
                call_id, owner_id, segment_id, turn_id, run_id,
                call_sequence, idempotency_key, policy_id
            ) VALUES (?, ?, ?, 'revision-domain-turn', 'revision-domain-run',
                      0, 'revision-domain-b-call-key', ?)
            """,
            (call_b, owner_b, owner_b_root, ids["policy"]),
        )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET surface_node_id = ?, request_header_id = ?,
                   provider_name = 'openai', model_name = 'gpt-test',
                   route_identity = 'primary'
             WHERE call_id = ?
            """,
            (artifact_node, ids["header"], call_b),
        )
        with pytest.raises(sqlite3.IntegrityError, match="matching revision domain"):
            _insert_trace_event(
                connection,
                event_id="revision-domain-invalid-event",
                segment_id=owner_b_root,
                sequence=1,
                event_type="response_selection",
                call_id=call_b,
                semantic_revision_id=ids["revision"],
            )
        _insert_trace_event(
            connection,
            event_id="revision-domain-valid-event",
            segment_id=owner_b_root,
            sequence=1,
            event_type="response_selection",
            call_id=call_b,
            semantic_revision_id=revision_b,
        )
        _insert_trace_event(
            connection,
            event_id="revision-domain-global-artifact-event",
            segment_id=owner_b_root,
            sequence=2,
            event_type="tool_result",
            call_id=call_b,
            artifact_id=ids["artifact"],
        )

        with pytest.raises(sqlite3.IntegrityError, match="matching revision domain"):
            connection.execute(
                """
                INSERT INTO console_trace_response_links(
                    response_link_id, call_id, link_kind, semantic_revision_id,
                    verification_outcome
                ) VALUES ('revision-domain-invalid-link', ?, 'revision', ?,
                          'verified_equal')
                """,
                (call_b, ids["revision"]),
            )
        connection.execute(
            """
            INSERT INTO console_trace_response_links(
                response_link_id, call_id, link_kind, semantic_revision_id,
                verification_outcome
            ) VALUES ('revision-domain-valid-link', ?, 'revision', ?,
                      'verified_equal')
            """,
            (call_b, revision_b),
        )

        artifact_call = "revision-domain-artifact-call"
        connection.execute(
            """
            INSERT INTO console_trace_calls(
                call_id, owner_id, segment_id, turn_id, run_id,
                call_sequence, idempotency_key, policy_id
            ) VALUES (?, ?, ?, 'revision-domain-turn', 'revision-domain-run',
                      1, 'revision-domain-artifact-call-key', ?)
            """,
            (artifact_call, owner_b, owner_b_root, ids["policy"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_response_links(
                response_link_id, call_id, link_kind, artifact_id,
                verification_outcome
            ) VALUES ('revision-domain-global-artifact-link', ?, 'artifact', ?,
                      'sanitized_artifact')
            """,
            (artifact_call, ids["artifact"]),
        )
    finally:
        db.close_connection()


@pytest.mark.parametrize("append_kind", ("artifact_node", "gap_event"))
def test_ownerless_segments_reject_nodes_and_events(
    tmp_path: Path,
    append_kind: str,
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"v56-ownerless-{append_kind}.sqlite",
        client_id=f"v56-ownerless-{append_kind}",
    )
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "ownerless append"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "canonical",
            }
        )
        assert conversation_id is not None
        assert message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        ownerless_segment = f"ownerless-{append_kind}-segment"
        connection.execute(
            "INSERT INTO console_trace_segments(segment_id) VALUES (?)",
            (ownerless_segment,),
        )

        with pytest.raises(sqlite3.IntegrityError, match="active effective owner"):
            if append_kind == "artifact_node":
                connection.execute(
                    """
                    INSERT INTO console_trace_surface_nodes(
                        node_id, segment_id, sequence, component_kind,
                        reference_kind, artifact_id
                    ) VALUES ('ownerless-artifact-node', ?, 0,
                              'provider_context', 'artifact', ?)
                    """,
                    (ownerless_segment, ids["artifact"]),
                )
            else:
                _insert_trace_event(
                    connection,
                    event_id="ownerless-gap-event",
                    segment_id=ownerless_segment,
                    sequence=0,
                    event_type="gap",
                    omission_reason_code="ownerless",
                )
    finally:
        db.close_connection()


def test_ownerless_segments_reject_replacements(tmp_path: Path) -> None:
    db = CharactersRAGDB(
        tmp_path / "v56-ownerless-replacement.sqlite",
        client_id="v56-ownerless-replacement",
    )
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "ownerless replacement"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "canonical",
            }
        )
        assert conversation_id is not None
        assert message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        ownerless_segment = "ownerless-replacement-segment"
        connection.execute(
            "INSERT INTO console_trace_segments(segment_id) VALUES (?)",
            (ownerless_segment,),
        )
        connection.execute("DROP TRIGGER console_trace_surface_nodes_owner_guard")
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, component_kind,
                reference_kind, artifact_id
            ) VALUES ('ownerless-replacement-start', ?, 0,
                      'provider_context', 'artifact', ?)
            """,
            (ownerless_segment, ids["artifact"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES ('ownerless-replacement-end', ?, 1,
                      'ownerless-replacement-start',
                      'provider_context', 'artifact', ?)
            """,
            (ownerless_segment, ids["artifact"]),
        )

        with pytest.raises(sqlite3.IntegrityError, match="active effective owner"):
            connection.execute(
                """
                INSERT INTO console_trace_surface_replacements(
                    replacement_id, segment_id, predecessor_head_id,
                    start_node_id, start_sequence, end_node_id, end_sequence,
                    replacement_node_id
                ) VALUES ('ownerless-replacement', ?,
                          'ownerless-replacement-start',
                          'ownerless-replacement-start', 0,
                          'ownerless-replacement-start', 0,
                          'ownerless-replacement-end')
                """,
                (ownerless_segment,),
            )
    finally:
        db.close_connection()


def test_ownerless_segments_reject_calls_and_response_links(tmp_path: Path) -> None:
    db = CharactersRAGDB(
        tmp_path / "v56-ownerless-call-link.sqlite",
        client_id="v56-ownerless-call-link",
    )
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "ownerless call"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "canonical",
            }
        )
        assert conversation_id is not None
        assert message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        ownerless_segment = "ownerless-call-segment"
        connection.execute(
            "INSERT INTO console_trace_segments(segment_id) VALUES (?)",
            (ownerless_segment,),
        )
        call_sql = """
            INSERT INTO console_trace_calls(
                call_id, owner_id, segment_id, turn_id, run_id,
                call_sequence, idempotency_key, policy_id
            ) VALUES (?, ?, ?, 'ownerless-turn', 'ownerless-run', 0, ?, ?)
        """
        with pytest.raises(sqlite3.IntegrityError, match="active effective owner"):
            connection.execute(
                call_sql,
                (
                    "ownerless-rejected-call",
                    ids["owner"],
                    ownerless_segment,
                    "ownerless-rejected-key",
                    ids["policy"],
                ),
            )

        # Seed a malformed ownerless call to exercise the downstream link guard
        # independently; ordinary SQL cannot create this prerequisite.
        connection.execute("DROP TRIGGER console_trace_calls_owner_lineage")
        connection.execute(
            call_sql,
            (
                "ownerless-seeded-call",
                ids["owner"],
                ownerless_segment,
                "ownerless-seeded-key",
                ids["policy"],
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="active effective owner"):
            connection.execute(
                """
                INSERT INTO console_trace_response_links(
                    response_link_id, call_id, link_kind, artifact_id,
                    verification_outcome
                ) VALUES ('ownerless-response-link', 'ownerless-seeded-call',
                          'artifact', ?, 'sanitized_artifact')
                """,
                (ids["artifact"],),
            )
    finally:
        db.close_connection()


def test_child_owner_attached_before_source_detach_preserves_its_domain(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        tmp_path / "v56-owner-before-detach.sqlite",
        client_id="v56-owner-before-detach",
    )
    try:
        connection = db.get_connection()
        source_conversation = db.add_conversation({"title": "active source"})
        child_conversation = db.add_conversation({"title": "active child"})
        source_message = db.add_message(
            {
                "conversation_id": source_conversation,
                "sender": "user",
                "content": "canonical source",
            }
        )
        assert source_conversation is not None
        assert child_conversation is not None
        assert source_message is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=source_conversation,
            message_id=source_message,
        )
        child_segment = "owner-before-detach-child"
        child_owner = "owner-before-detach-child-owner"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (child_segment, ids["segment"], ids["node"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_owners(owner_id, conversation_id, root_segment_id)
            VALUES (?, ?, ?)
            """,
            (child_owner, child_conversation, child_segment),
        )
        connection.execute(
            """
            UPDATE console_trace_owners
               SET attached = 0, conversation_id = NULL,
                   detached_at = '2026-08-28T05:00:00Z'
             WHERE owner_id = ?
            """,
            (ids["owner"],),
        )
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, predecessor_node_id,
                component_kind, reference_kind, artifact_id
            ) VALUES ('owner-before-detach-artifact', ?, 1, ?,
                      'provider_context', 'artifact', ?)
            """,
            (child_segment, ids["node"], ids["artifact"]),
        )
        _insert_trace_event(
            connection,
            event_id="owner-before-detach-event",
            segment_id=child_segment,
            sequence=0,
            event_type="surface_append",
            surface_node_id="owner-before-detach-artifact",
        )
    finally:
        db.close_connection()


def test_detached_source_rejects_new_child_segment_staging(tmp_path: Path) -> None:
    db = CharactersRAGDB(
        tmp_path / "v56-owner-after-detach-segment.sqlite",
        client_id="v56-owner-after-detach-segment",
    )
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "detached source"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "canonical",
            }
        )
        assert conversation_id is not None
        assert message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        connection.execute(
            """
            UPDATE console_trace_owners
               SET attached = 0, conversation_id = NULL,
                   detached_at = '2026-08-28T05:10:00Z'
             WHERE owner_id = ?
            """,
            (ids["owner"],),
        )
        with pytest.raises(sqlite3.IntegrityError, match="detached owner prefix"):
            connection.execute(
                """
                INSERT INTO console_trace_segments(
                    segment_id, parent_segment_id, inherited_through_sequence,
                    inherited_surface_head_id
                ) VALUES ('owner-after-detach-child', ?, 0, ?)
                """,
                (ids["segment"], ids["node"]),
            )
    finally:
        db.close_connection()


def test_child_staged_before_detach_cannot_attach_owner_after_detach(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        tmp_path / "v56-owner-after-detach-attach.sqlite",
        client_id="v56-owner-after-detach-attach",
    )
    try:
        connection = db.get_connection()
        source_conversation = db.add_conversation({"title": "staged source"})
        child_conversation = db.add_conversation({"title": "staged child"})
        source_message = db.add_message(
            {
                "conversation_id": source_conversation,
                "sender": "user",
                "content": "canonical",
            }
        )
        assert source_conversation is not None
        assert child_conversation is not None
        assert source_message is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=source_conversation,
            message_id=source_message,
        )
        staged_child = "owner-before-detach-staged-child"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (staged_child, ids["segment"], ids["node"]),
        )
        connection.execute(
            """
            UPDATE console_trace_owners
               SET attached = 0, conversation_id = NULL,
                   detached_at = '2026-08-28T05:20:00Z'
             WHERE owner_id = ?
            """,
            (ids["owner"],),
        )
        with pytest.raises(sqlite3.IntegrityError, match="detached owner prefix"):
            connection.execute(
                """
                INSERT INTO console_trace_owners(
                    owner_id, conversation_id, root_segment_id
                ) VALUES ('owner-after-detach-child-owner', ?, ?)
                """,
                (child_conversation, staged_child),
            )
    finally:
        db.close_connection()


def test_call_owner_segment_lineage_and_state_timestamps_are_enforced(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "v56-call-lineage.sqlite", client_id="v56-call")
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "call lineage"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "canonical",
            }
        )
        assert conversation_id is not None
        assert message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )

        child_segment = "call-child-segment"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (child_segment, ids["segment"], ids["node"]),
        )
        connection.execute(
            """
            INSERT INTO console_trace_calls(
                call_id, owner_id, segment_id, turn_id, run_id,
                call_sequence, idempotency_key, policy_id
            ) VALUES ('call-valid-child', ?, ?, 'turn-child', 'run-child', 0,
                      'call-valid-child-key', ?)
            """,
            (ids["owner"], child_segment, ids["policy"]),
        )

        unrelated_segment = "call-unrelated-segment"
        connection.execute(
            "INSERT INTO console_trace_segments(segment_id) VALUES (?)",
            (unrelated_segment,),
        )
        unrelated_conversation = db.add_conversation({"title": "call unrelated owner"})
        assert unrelated_conversation is not None
        connection.execute(
            """
            INSERT INTO console_trace_owners(owner_id, conversation_id, root_segment_id)
            VALUES ('call-unrelated-owner', ?, ?)
            """,
            (unrelated_conversation, unrelated_segment),
        )
        unrelated_surface = "call-unrelated-surface"
        connection.execute(
            """
            INSERT INTO console_trace_surface_nodes(
                node_id, segment_id, sequence, component_kind,
                reference_kind, artifact_id
            ) VALUES (?, ?, 0, 'provider_context', 'artifact', ?)
            """,
            (unrelated_surface, unrelated_segment, ids["artifact"]),
        )
        with pytest.raises(sqlite3.IntegrityError, match="effective owner"):
            connection.execute(
                """
                INSERT INTO console_trace_calls(
                    call_id, owner_id, segment_id, turn_id, run_id,
                    call_sequence, idempotency_key, policy_id
                ) VALUES ('call-invalid-owner-lineage', ?, ?, 'turn-other',
                          'run-other', 0, 'call-invalid-owner-lineage-key', ?)
                """,
                (ids["owner"], unrelated_segment, ids["policy"]),
            )

        with pytest.raises(sqlite3.IntegrityError, match="call binding"):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET surface_node_id = ?, request_header_id = ?,
                       provider_name = 'openai', model_name = 'gpt-test',
                       route_identity = 'primary'
                 WHERE call_id = ?
                """,
                (unrelated_surface, ids["header"], ids["call"]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="call binding"):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET surface_node_id = ?, request_header_id = ?,
                       provider_name = 'anthropic', model_name = 'gpt-test',
                       route_identity = 'primary'
                 WHERE call_id = ?
                """,
                (ids["node"], ids["header"], ids["call"]),
            )
        connection.execute(
            """
            INSERT INTO console_trace_calls(
                call_id, owner_id, segment_id, turn_id, run_id,
                call_sequence, idempotency_key, policy_id
            ) VALUES ('call-dispatch-binding-probe', ?, ?, 'turn-probe',
                      'run-probe', 0, 'call-dispatch-binding-probe-key', ?)
            """,
            (ids["owner"], ids["segment"], ids["policy"]),
        )
        with pytest.raises(sqlite3.IntegrityError, match="call binding"):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET state = 'dispatch_started', surface_node_id = ?,
                       request_header_id = ?, provider_name = 'openai',
                       model_name = 'gpt-test', route_identity = 'primary',
                       dispatch_started_at = '2026-08-28T01:00:00Z'
                 WHERE call_id = 'call-dispatch-binding-probe'
                """,
                (unrelated_surface, ids["header"]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="call binding"):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET state = 'dispatch_started', surface_node_id = ?,
                       request_header_id = ?, provider_name = 'openai',
                       model_name = 'wrong-model', route_identity = 'primary',
                       dispatch_started_at = '2026-08-28T01:00:00Z'
                 WHERE call_id = 'call-dispatch-binding-probe'
                """,
                (ids["node"], ids["header"]),
            )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET dispatch_started_at = '2026-08-28T01:00:00Z'
                 WHERE call_id = ?
                """,
                (ids["call"],),
            )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET surface_node_id = ?, request_header_id = ?,
                   provider_name = 'openai', model_name = 'gpt-test',
                   route_identity = 'primary'
             WHERE call_id = ?
            """,
            (ids["node"], ids["header"], ids["call"]),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET state = 'not_dispatched',
                       dispatch_started_at = '2026-08-28T01:00:00Z',
                       settled_at = '2026-08-28T01:00:01Z'
                 WHERE call_id = ?
                """,
                (ids["call"],),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET state = 'dispatch_started',
                       dispatch_started_at = '2026-08-28T01:00:00Z',
                       response_started_at = '2026-08-28T01:00:01Z'
                 WHERE call_id = ?
                """,
                (ids["call"],),
            )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET state = 'dispatch_started',
                   dispatch_started_at = '2026-08-28T01:00:00Z'
             WHERE call_id = ?
            """,
            (ids["call"],),
        )
        with pytest.raises(sqlite3.IntegrityError, match="lifecycle"):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET state = 'error', outcome = 'error',
                       response_started_at = '2026-08-28T01:00:01Z',
                       settled_at = '2026-08-28T01:00:02Z'
                 WHERE call_id = ?
                """,
                (ids["call"],),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET response_started_at = '2026-08-28T01:00:01Z'
                 WHERE call_id = ?
                """,
                (ids["call"],),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET state = 'dispatch_unknown',
                       response_started_at = '2026-08-28T01:00:01Z',
                       settled_at = '2026-08-28T01:00:02Z'
                 WHERE call_id = ?
                """,
                (ids["call"],),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET state = 'abandoned', outcome = 'abandoned',
                       response_started_at = '2026-08-28T01:00:01Z',
                       settled_at = '2026-08-28T01:00:02Z',
                       provider_inactive_at = '2026-08-28T01:00:02Z'
                 WHERE call_id = ?
                """,
                (ids["call"],),
            )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET state = 'error', outcome = 'error',
                   settled_at = '2026-08-28T01:00:02Z'
             WHERE call_id = ?
            """,
            (ids["call"],),
        )

        connection.execute(
            """
            UPDATE console_trace_calls
               SET surface_node_id = ?, request_header_id = ?,
                   provider_name = 'openai', model_name = 'gpt-test',
                   route_identity = 'primary'
             WHERE call_id = 'call-valid-child'
            """,
            (ids["node"], ids["header"]),
        )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET state = 'dispatch_started',
                   dispatch_started_at = '2026-08-28T01:01:00Z'
             WHERE call_id = 'call-valid-child'
            """
        )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET state = 'response_started',
                   response_started_at = '2026-08-28T01:01:01Z'
             WHERE call_id = 'call-valid-child'
            """
        )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET state = 'error', outcome = 'error',
                   settled_at = '2026-08-28T01:01:02Z'
             WHERE call_id = 'call-valid-child'
            """
        )
    finally:
        db.close_connection()


def test_call_state_updates_do_not_rewalk_an_unchanged_long_surface(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "v56-call-opcodes.sqlite", client_id="v56-opcodes")
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "opcode witness"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "canonical",
            }
        )
        assert conversation_id is not None
        assert message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        child_segment = "opcode-child-segment"
        connection.execute(
            """
            INSERT INTO console_trace_segments(
                segment_id, parent_segment_id, inherited_through_sequence,
                inherited_surface_head_id
            ) VALUES (?, ?, 0, ?)
            """,
            (child_segment, ids["segment"], ids["node"]),
        )
        predecessor = ids["node"]
        short_surface = "opcode-surface-1"
        for sequence in range(1, 401):
            node_id = f"opcode-surface-{sequence}"
            connection.execute(
                """
                INSERT INTO console_trace_surface_nodes(
                    node_id, segment_id, sequence, predecessor_node_id,
                    component_kind, reference_kind, artifact_id
                ) VALUES (?, ?, ?, ?, 'provider_context', 'artifact', ?)
                """,
                (node_id, child_segment, sequence, predecessor, ids["artifact"]),
            )
            predecessor = node_id

        for call_id, call_sequence in (
            ("opcode-short-call", 0),
            ("opcode-long-call", 1),
        ):
            connection.execute(
                """
                INSERT INTO console_trace_calls(
                    call_id, owner_id, segment_id, turn_id, run_id,
                    call_sequence, idempotency_key, policy_id
                ) VALUES (?, ?, ?, 'opcode-turn', 'opcode-run', ?, ?, ?)
                """,
                (
                    call_id,
                    ids["owner"],
                    child_segment,
                    call_sequence,
                    f"{call_id}-key",
                    ids["policy"],
                ),
            )
        for call_id, surface_node_id in (
            ("opcode-short-call", short_surface),
            ("opcode-long-call", predecessor),
        ):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET surface_node_id = ?, request_header_id = ?,
                       provider_name = 'openai', model_name = 'gpt-test',
                       route_identity = 'primary'
                 WHERE call_id = ?
                """,
                (surface_node_id, ids["header"], call_id),
            )

        def dispatch_opcode_count(call_id: str, timestamp: str) -> int:
            opcodes = 0

            def count_opcode() -> int:
                nonlocal opcodes
                opcodes += 1
                return 0

            connection.set_progress_handler(count_opcode, 1)
            try:
                connection.execute(
                    """
                    UPDATE console_trace_calls
                       SET state = 'dispatch_started', dispatch_started_at = ?
                     WHERE call_id = ?
                    """,
                    (timestamp, call_id),
                )
            finally:
                connection.set_progress_handler(None, 0)
            return opcodes

        short_update_opcodes = dispatch_opcode_count(
            "opcode-short-call", "2026-08-28T03:00:00Z"
        )
        long_update_opcodes = dispatch_opcode_count(
            "opcode-long-call", "2026-08-28T03:00:01Z"
        )
        assert long_update_opcodes <= short_update_opcodes + 100
    finally:
        db.close_connection()


def test_semantic_revision_locator_and_predecessor_chain_are_consistent(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        tmp_path / "v56-revision-lineage.sqlite", client_id="v56-revision"
    )
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "revision lineage"})
        other_conversation_id = db.add_conversation({"title": "other"})
        message_id = _insert_legacy_message_without_revision(
            db,
            conversation_id=conversation_id,
            sender="user",
            content="canonical",
        )
        sibling_message_id = _insert_legacy_message_without_revision(
            db,
            conversation_id=conversation_id,
            sender="user",
            content="sibling",
        )
        other_message_id = _insert_legacy_message_without_revision(
            db,
            conversation_id=other_conversation_id,
            sender="user",
            content="other",
        )
        assert conversation_id is not None
        assert other_conversation_id is not None
        assert message_id is not None
        assert sibling_message_id is not None
        assert other_message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )

        with pytest.raises(sqlite3.IntegrityError, match="locator"):
            connection.execute(
                """
                INSERT INTO console_trace_semantic_revisions(
                    revision_id, source_conversation_id, source_message_id,
                    revision_sequence, normalized_role, content_kind,
                    creation_reason, live_message_id
                ) VALUES ('revision-invalid-message', ?, ?, 0, 'user', 'text',
                          'capture', ?)
                """,
                (conversation_id, message_id, sibling_message_id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="locator"):
            connection.execute(
                """
                INSERT INTO console_trace_semantic_revisions(
                    revision_id, source_conversation_id, source_message_id,
                    revision_sequence, normalized_role, content_kind,
                    creation_reason, live_message_id
                ) VALUES ('revision-invalid-conversation', ?, ?, 0, 'user',
                          'text', 'capture', ?)
                """,
                (conversation_id, other_message_id, other_message_id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="predecessor"):
            connection.execute(
                """
                INSERT INTO console_trace_semantic_revisions(
                    revision_id, source_conversation_id, source_message_id,
                    revision_sequence, normalized_role, content_kind,
                    creation_reason, predecessor_revision_id,
                    live_locator_retired_at
                ) VALUES ('revision-invalid-predecessor', ?, ?, 1, 'user',
                          'text', 'edit', ?, '2026-08-28T01:00:00Z')
                """,
                (conversation_id, sibling_message_id, ids["revision"]),
            )

        with db.transaction(immediate=True) as cursor:
            authorization = db._semantic_mutation_authorization_for_coordinator(
                connection
            )
            with authorization._authorize(
                message_id=message_id, operations={"locator_retire"}
            ):
                cursor.execute(
                    """
                    UPDATE console_trace_semantic_revisions
                       SET live_message_id = NULL,
                           live_locator_retired_at = '2026-08-28T01:00:00Z'
                     WHERE revision_id = ?
                    """,
                    (ids["revision"],),
                )
        connection.execute(
            """
            INSERT INTO console_trace_semantic_revisions(
                revision_id, source_conversation_id, source_message_id,
                revision_sequence, normalized_role, content_kind,
                creation_reason, predecessor_revision_id, live_message_id
            ) VALUES ('revision-valid-successor', ?, ?, 1, 'user', 'text',
                      'edit', ?, ?)
            """,
            (conversation_id, message_id, ids["revision"], message_id),
        )
    finally:
        db.close_connection()


def test_historical_rows_are_append_only_and_mutable_state_is_bounded(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "v56-immutability.sqlite", client_id="v56-guards")
    try:
        connection = db.get_connection()
        conversation_id = db.add_conversation({"title": "immutability"})
        message_id = _insert_legacy_message_without_revision(
            db,
            conversation_id=conversation_id,
            sender="user",
            content="canonical",
        )
        assert conversation_id is not None
        assert message_id is not None
        ids = _insert_minimal_graph(
            connection,
            conversation_id=conversation_id,
            message_id=message_id,
        )

        representative_updates = {
            "console_trace_artifacts": ("media_type", "text/plain"),
            "console_trace_events": ("event_type", "gap"),
            "console_trace_request_headers": ("model_name", "changed"),
            "console_trace_redaction_spans": ("category", "changed"),
            "console_trace_segments": ("inherited_through_sequence", 0),
            "console_trace_surface_nodes": ("component_kind", "changed"),
        }
        primary_keys = {
            "console_trace_artifacts": ("artifact_id", ids["artifact"]),
            "console_trace_events": ("event_id", ids["event"]),
            "console_trace_request_headers": ("header_id", ids["header"]),
            "console_trace_redaction_spans": ("span_id", ids["span"]),
            "console_trace_segments": ("segment_id", ids["segment"]),
            "console_trace_surface_nodes": ("node_id", ids["node"]),
        }
        for table, (column, value) in representative_updates.items():
            key_column, key_value = primary_keys[table]
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(
                    f"UPDATE {table} SET {column} = ? WHERE {key_column} = ?",
                    (value, key_value),
                )
            with pytest.raises(sqlite3.IntegrityError, match="deletion prohibited"):
                connection.execute(
                    f"DELETE FROM {table} WHERE {key_column} = ?", (key_value,)
                )

        with db.transaction(immediate=True) as cursor:
            authorization = db._semantic_mutation_authorization_for_coordinator(
                connection
            )
            with authorization._authorize(
                message_id=message_id, operations={"locator_retire"}
            ):
                cursor.execute(
                    """
                    UPDATE console_trace_semantic_revisions
                       SET live_message_id = NULL,
                           live_locator_retired_at = '2026-08-28T01:00:00Z'
                     WHERE revision_id = ?
                    """,
                    (ids["revision"],),
                )
        with pytest.raises(sqlite3.IntegrityError, match="locator retirement"):
            connection.execute(
                """
                UPDATE console_trace_semantic_revisions
                   SET normalized_role = 'assistant'
                 WHERE revision_id = ?
                """,
                (ids["revision"],),
            )

        connection.execute(
            """
            UPDATE console_trace_calls
               SET surface_node_id = ?, request_header_id = ?,
                   provider_name = 'openai', model_name = 'gpt-test',
                   route_identity = 'primary'
             WHERE call_id = ?
            """,
            (ids["node"], ids["header"], ids["call"]),
        )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET state = 'dispatch_started',
                   dispatch_started_at = '2026-08-28T01:01:00Z'
             WHERE call_id = ?
            """,
            (ids["call"],),
        )
        with pytest.raises(sqlite3.IntegrityError, match="lifecycle"):
            connection.execute(
                "UPDATE console_trace_calls SET state = 'reserved' WHERE call_id = ?",
                (ids["call"],),
            )
        with pytest.raises(sqlite3.IntegrityError, match="lifecycle"):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET state = 'complete', outcome = 'complete',
                       settled_at = '2026-08-28T01:02:00Z'
                 WHERE call_id = ?
                """,
                (ids["call"],),
            )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET state = 'response_started',
                   response_started_at = '2026-08-28T01:02:00Z'
             WHERE call_id = ?
            """,
            (ids["call"],),
        )
        connection.execute(
            """
            UPDATE console_trace_calls
               SET state = 'complete', outcome = 'complete',
                   usage_json = '{"input_tokens":1}',
                   integrity_state = 'complete',
                   settled_at = '2026-08-28T01:03:00Z'
             WHERE call_id = ?
            """,
            (ids["call"],),
        )
        with pytest.raises(sqlite3.IntegrityError, match="terminal"):
            connection.execute(
                """
                UPDATE console_trace_calls
                   SET usage_json = '{"input_tokens":2}'
                 WHERE call_id = ?
                """,
                (ids["call"],),
            )

        with pytest.raises(sqlite3.IntegrityError, match="reservation"):
            connection.execute(
                """
                INSERT INTO console_trace_calls(
                    call_id, owner_id, segment_id, turn_id, run_id,
                    call_sequence, idempotency_key, policy_id, state,
                    surface_node_id, request_header_id, provider_name,
                    model_name, route_identity, dispatch_started_at
                ) VALUES ('00000000-0000-4000-8000-000000000090', ?, ?,
                          '00000000-0000-4000-8000-000000000089',
                          '00000000-0000-4000-8000-000000000088', 0,
                          'invalid-direct-dispatch', ?, 'dispatch_started',
                          ?, ?, 'openai', 'gpt-test', 'primary',
                          '2026-08-28T01:00:00Z')
                """,
                (
                    ids["owner"],
                    ids["segment"],
                    ids["policy"],
                    ids["node"],
                    ids["header"],
                ),
            )

        connection.execute(
            """
            UPDATE console_trace_owners
               SET conversation_id = NULL, attached = 0,
                   detached_at = '2026-08-28T01:04:00Z'
             WHERE owner_id = ?
            """,
            (ids["owner"],),
        )
        with pytest.raises(sqlite3.IntegrityError, match="detach"):
            connection.execute(
                """
                UPDATE console_trace_owners
                   SET conversation_id = ?, attached = 1, detached_at = NULL
                 WHERE owner_id = ?
                """,
                (conversation_id, ids["owner"]),
            )

        connection.execute(
            """
            UPDATE console_trace_migration_state
               SET status = 'running', last_exchange_id = 1,
                   processed_rows = 1, processed_bytes = 32
             WHERE migration_name = 'legacy_exchange_normalization'
            """
        )
        connection.execute(
            """
            UPDATE console_trace_maintenance_state
               SET state = 'marking', lease_id = 'lease-1',
                   lease_owner = 'worker-1',
                   lease_expires_at = '2026-08-28T02:00:00Z', marked_epoch = 0
             WHERE singleton_id = 1
            """
        )
        connection.execute(
            "UPDATE console_trace_graph_epoch SET epoch = epoch + 1 WHERE singleton_id = 1"
        )
        with pytest.raises(sqlite3.IntegrityError, match="exactly one"):
            connection.execute(
                "UPDATE console_trace_graph_epoch SET epoch = epoch + 2 WHERE singleton_id = 1"
            )

    finally:
        db.close_connection()
