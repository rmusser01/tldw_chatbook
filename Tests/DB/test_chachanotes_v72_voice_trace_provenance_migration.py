"""ChaChaNotes v72 post-dispatch trace provenance and lookup indexes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.Chat import chat_persistence_service as persistence_module
from tldw_chatbook.Chat.console_trace_models import new_opaque_id
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def test_genuine_v70_upgrades_without_rewriting_semantic_rows(tmp_path: Path) -> None:
    path = tmp_path / "genuine-v70.sqlite"
    with chachanotes_db_at_version(path, 70, client_id="v70-fixture") as historical:
        from Tests.ChaChaNotesDB.test_canvas_migration import _canvas_schema

        connection = historical.get_connection()
        assert _version(connection) == 70
        buddy_schema_before = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name LIKE '%buddy%' OR name = 'visual_owner_bindings' "
                "ORDER BY type, name"
            )
        )
        assert {row[1] for row in buddy_schema_before} >= {
            "buddy_profiles",
            "buddy_visual_bindings",
            "visual_owner_bindings",
        }
        assert "reservation_provenance" not in {
            row[1]
            for row in connection.execute("PRAGMA table_info(console_trace_calls)")
        }
        source_guard_before = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'console_trace_call_boundary_source_guard'"
        ).fetchone()[0]
        canvas_before = _canvas_schema(historical.get_connection())
        privacy_schema_before = tuple(
            tuple(row)
            for row in historical.get_connection().execute(
                "SELECT type, name, sql FROM sqlite_master WHERE name LIKE '%privacy%' OR name = 'console_conversation_capture_policy' ORDER BY type, name"
            )
        )
        conversation_id = historical.add_conversation({"title": "v70"})
        assert conversation_id is not None
        with historical.transaction() as cursor:
            cursor.execute(
                """INSERT INTO console_conversation_capture_policy(
                conversation_id, capture_detail, capture_enabled, pii_redaction_enabled)
                VALUES (?, 'full', 0, 1)""",
                (conversation_id,),
            )
        privacy_before = tuple(
            historical.get_connection()
            .execute(
                "SELECT * FROM console_conversation_capture_policy WHERE conversation_id = ?",
                (conversation_id,),
            )
            .fetchone()
        )
        message_id = historical.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "semantic payload must remain untouched",
            }
        )
        assert message_id is not None
        before = tuple(
            historical.get_connection()
            .execute(
                """SELECT revision_id, source_conversation_id, source_message_id,
                          revision_sequence, normalized_role, content_kind,
                          creation_reason, predecessor_revision_id, live_message_id,
                          live_locator_retired_at, created_at
                     FROM console_trace_semantic_revisions
                    WHERE source_message_id = ?""",
                (message_id,),
            )
            .fetchone()
        )

    migrated = CharactersRAGDB(path, "v72-migrated")
    try:
        connection = migrated.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert (
            tuple(
                tuple(row)
                for row in connection.execute(
                    "SELECT type, name, sql FROM sqlite_master "
                    "WHERE name LIKE '%buddy%' OR name = 'visual_owner_bindings' "
                    "ORDER BY type, name"
                )
            )
            == buddy_schema_before
        )
        assert "reservation_provenance" in {
            row[1]
            for row in connection.execute("PRAGMA table_info(console_trace_calls)")
        }
        assert (
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'console_trace_call_boundary_source_guard'"
            ).fetchone()[0]
            == source_guard_before
        )
        after = tuple(
            connection.execute(
                """SELECT revision_id, source_conversation_id, source_message_id,
                          revision_sequence, normalized_role, content_kind,
                          creation_reason, predecessor_revision_id, live_message_id,
                          live_locator_retired_at, created_at
                     FROM console_trace_semantic_revisions
                    WHERE source_message_id = ?""",
                (message_id,),
            ).fetchone()
        )
        assert after == before
        assert _canvas_schema(connection) == canvas_before
        assert (
            tuple(
                tuple(row)
                for row in connection.execute(
                    "SELECT type, name, sql FROM sqlite_master WHERE name LIKE '%privacy%' OR name = 'console_conversation_capture_policy' ORDER BY type, name"
                )
            )
            == privacy_schema_before
        )
        assert (
            tuple(
                connection.execute(
                    "SELECT * FROM console_conversation_capture_policy WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
            )
            == privacy_before
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        migrated.close_connection()


def test_genuine_v71_preserves_archived_rows_columns_indexes_and_triggers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "genuine-v71.sqlite"
    with chachanotes_db_at_version(path, 71, client_id="archive-before-voice") as old:
        connection = old.get_connection()
        assert _version(connection) == 71
        conversation_id = old.add_conversation({"title": "Archived before voice"})
        row = old.get_conversation_by_id(conversation_id)
        old.set_conversations_archived(
            [conversation_id],
            archived=True,
            expected_versions={conversation_id: row["version"]},
        )
        archived_before = dict(old.get_conversation_by_id(conversation_id))
        columns_before = tuple(
            tuple(row) for row in connection.execute("PRAGMA table_info(conversations)")
        )
        schema_before = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE tbl_name = 'conversations' ORDER BY type, name"
            )
        )
        assert any(row[1] == "idx_conversations_archive" for row in schema_before)
        assert any(row[0] == "trigger" for row in schema_before)
    current = CharactersRAGDB(path, "archive-after-voice")
    try:
        connection = current.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert dict(current.get_conversation_by_id(conversation_id)) == archived_before
        assert (
            tuple(
                tuple(row)
                for row in connection.execute("PRAGMA table_info(conversations)")
            )
            == columns_before
        )
        assert (
            tuple(
                tuple(row)
                for row in connection.execute(
                    "SELECT type, name, sql FROM sqlite_master WHERE tbl_name = 'conversations' ORDER BY type, name"
                )
            )
            == schema_before
        )
    finally:
        current.close_connection()


def test_v72_provenance_defaults_and_cross_column_constraints() -> None:
    db = CharactersRAGDB(":memory:", "v72-provenance")
    try:
        connection = db.get_connection()
        columns = {
            str(row[1]): row
            for row in connection.execute("PRAGMA table_info(console_trace_calls)")
        }
        assert columns["reservation_provenance"][4] == "'crash_durable_reserved'"
        assert "import_reason_code" in columns

        conversation_id = db.add_conversation({"title": "ordinary"})
        assert conversation_id is not None
        with db.transaction(immediate=True) as cursor:
            segment_id = new_opaque_id()
            owner_id = new_opaque_id()
            policy_id = new_opaque_id()
            cursor.execute(
                "INSERT INTO console_trace_segments(segment_id) VALUES (?)",
                (segment_id,),
            )
            cursor.execute(
                """INSERT INTO console_trace_owners(
                       owner_id, conversation_id, root_segment_id, attached)
                     VALUES (?, ?, ?, 1)""",
                (owner_id, conversation_id, segment_id),
            )
            cursor.execute(
                """INSERT INTO console_trace_policies(
                       policy_id, credential_filter_version, pii_redaction_enabled)
                     VALUES (?, 'cred-v1', 0)""",
                (policy_id,),
            )
            call_id = new_opaque_id()
            cursor.execute(
                """INSERT INTO console_trace_calls(
                       call_id, owner_id, segment_id, turn_id, run_id,
                       call_sequence, idempotency_key, policy_id)
                     VALUES (?, ?, ?, 'turn', 'run', 0, ?, ?)""",
                (call_id, owner_id, segment_id, new_opaque_id(), policy_id),
            )
            row = cursor.execute(
                """SELECT reservation_provenance, import_reason_code
                       FROM console_trace_calls WHERE call_id = ?""",
                (call_id,),
            ).fetchone()
            assert tuple(row) == ("crash_durable_reserved", None)

            with pytest.raises(sqlite3.IntegrityError, match="provenance"):
                cursor.execute(
                    """INSERT INTO console_trace_calls(
                           call_id, owner_id, segment_id, turn_id, run_id,
                           call_sequence, idempotency_key, policy_id,
                           reservation_provenance, import_reason_code)
                         VALUES (?, ?, ?, 'turn-2', 'run-2', 1, ?, ?,
                                 'crash_durable_reserved', 'provisional_voice_promoted')""",
                    (
                        new_opaque_id(),
                        owner_id,
                        segment_id,
                        new_opaque_id(),
                        policy_id,
                    ),
                )
    finally:
        db.close_connection()


def test_every_completed_pair_reconciliation_locator_uses_indexed_search() -> None:
    db = CharactersRAGDB(":memory:", "v72-index-plans")
    try:
        connection = db.get_connection()
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'"
            ).fetchone()
            is None
        )
        expected_indexes = {
            (
                "console_dispatch_checkpoints",
                "user_message_id",
            ): "idx_console_dispatch_checkpoints_user_message",
            (
                "message_generation_metadata",
                "message_id",
            ): "idx_message_generation_metadata_message",
            (
                "message_trajectory_metadata",
                "message_id",
            ): "idx_message_trajectory_metadata_message",
            (
                "rag_citation_traces",
                "legacy_message_id",
            ): "idx_rag_citation_traces_legacy_message",
            (
                "rag_message_trace_owners",
                "message_id",
            ): "idx_rag_message_trace_owners_message",
            (
                "transcript_annotations",
                "message_id",
            ): "idx_transcript_annotations_message",
            (
                "console_trace_semantic_revisions",
                "source_message_id",
            ): "idx_console_trace_semantic_revisions_source_message_exact",
            (
                "console_trace_semantic_revisions",
                "predecessor_revision_id",
            ): "idx_console_trace_semantic_revisions_predecessor_exact",
            (
                "console_trace_events",
                "semantic_revision_id",
            ): "idx_console_trace_events_semantic_revision_exact",
            (
                "console_trace_redaction_spans",
                "semantic_revision_id",
            ): "idx_console_trace_redaction_spans_semantic_revision_exact",
            (
                "console_trace_response_links",
                "semantic_revision_id",
            ): "idx_console_trace_response_links_semantic_revision_exact",
            (
                "console_trace_surface_nodes",
                "semantic_revision_id",
            ): "idx_console_trace_surface_nodes_semantic_revision_exact",
        }
        locators = frozenset().union(
            persistence_module._VOICE_PROMOTION_MANDATORY_LOCATORS,
            persistence_module._VOICE_PROMOTION_FORBIDDEN_MESSAGE_LOCATORS,
            persistence_module._VOICE_PROMOTION_FORBIDDEN_REVISION_LOCATORS,
        )
        assert expected_indexes.keys() <= locators
        failures: list[str] = []
        for table, column in sorted(locators):
            plan = connection.execute(
                f'EXPLAIN QUERY PLAN SELECT 1 FROM "{table}" '
                f'WHERE "{column}" IN (?, ?) LIMIT 1',
                (new_opaque_id(), new_opaque_id()),
            ).fetchall()
            details = " | ".join(str(row[3]) for row in plan)
            if (table, column) in expected_indexes:
                assert expected_indexes[table, column] in details
            if "SEARCH" not in details or "SCAN" in details:
                failures.append(f"{table}.{column}: {details}")
        assert failures == []
    finally:
        db.close_connection()


def test_raw_terminal_promoted_insert_fails_closed_without_repository_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw-terminal.sqlite"
    db = CharactersRAGDB(path, "managed")
    conversation_id = db.add_conversation({"title": "raw"})
    assert conversation_id is not None
    db.close_connection()

    raw = sqlite3.connect(path)
    raw.execute("PRAGMA foreign_keys = ON")
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such function"):
            raw.execute(
                """INSERT INTO console_trace_calls(
                       call_id, owner_id, segment_id, turn_id, run_id,
                       call_sequence, idempotency_key, policy_id, state,
                       surface_node_id, request_header_id, provider_name,
                       model_name, route_identity, dispatch_started_at,
                       response_started_at, settled_at, outcome,
                       integrity_state, reservation_provenance, import_reason_code)
                     VALUES (?, ?, ?, 'turn', 'run', 0, ?, ?, 'complete',
                             ?, ?, 'provider', 'model', 'route',
                             '2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z',
                             '2026-01-01T00:00:02Z', 'complete', 'complete',
                             'post_dispatch_promoted', 'provisional_voice_promoted')""",
                tuple(new_opaque_id() for _ in range(7)),
            )
    finally:
        raw.close()
