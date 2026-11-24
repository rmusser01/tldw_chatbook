"""ChaChaNotes v72 post-dispatch trace provenance and lookup indexes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.Chat import chat_persistence_service as persistence_module
from tldw_chatbook.Chat.console_trace_models import new_opaque_id
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


_MIGRATIONS = Path(__file__).resolve().parents[2] / "tldw_chatbook/DB/migrations"


def _migration_statement(file_name: str, start: str) -> str:
    """One CREATE statement as its shipped migration file writes it."""
    sql = (_MIGRATIONS / file_name).read_text(encoding="utf-8")
    begin = sql.index(start)
    if start.startswith("CREATE TRIGGER"):
        return sql[begin : sql.index("END;", begin) + len("END")]
    return sql[begin : sql.index(";", begin)]


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
        # ADR-239's v81->v82 step (01b17108c7) rewrote the canvas sequence
        # index and added the single-root guard on purpose. Strip exactly
        # those two and the v72 guarantee still pins every other canvas
        # object (TASK-42517).
        assert _canvas_schema(connection) == canvas_before | {
            (
                "index",
                "idx_canvas_revisions_canvas_sequence",
            ): "CREATE INDEX idx_canvas_revisions_canvas_sequence\n"
            "  ON canvas_revisions(canvas_id, sequence, id)",
            (
                "index",
                "uq_canvas_revisions_single_root",
            ): "CREATE UNIQUE INDEX uq_canvas_revisions_single_root\n"
            "  ON canvas_revisions(canvas_id) WHERE parent_revision_id IS NULL",
        }
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
        # Archive with the v71-era statement `set_conversations_archived`
        # ran: today's method also writes `archived_at`, which v85 added
        # (dd8d60c9e8), so it cannot run on a v71 file (TASK-42517).
        with old.transaction(immediate=True) as cursor:
            assert (
                cursor.execute(
                    "UPDATE conversations SET archived = 1, version = version + 1, "
                    "last_modified = ? WHERE id = ? AND deleted = 0 "
                    "AND version = ? AND archived != 1",
                    (
                        old._get_current_utc_timestamp_iso(),
                        conversation_id,
                        row["version"],
                    ),
                ).rowcount
                == 1
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
        # v79 (ADR-216) is the first post-v71 step that adds conversations
        # indexes: migration may add exactly these two and nothing else.
        browse_order_indexes = (
            (
                "index",
                "idx_conversations_last_modified",
                "CREATE INDEX idx_conversations_last_modified\n"
                "  ON conversations(last_modified DESC, id DESC)",
            ),
            (
                "index",
                "idx_conversations_archived_browse_order",
                "CREATE INDEX idx_conversations_archived_browse_order\n"
                "  ON conversations(archived, last_modified DESC, id DESC)",
            ),
        )
    current = CharactersRAGDB(path, "archive-after-voice")
    try:
        connection = current.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        # ADR-166's v84->v85 step (dd8d60c9e8) added `archived_at`, NULL for
        # every archive made before it ("historical dates remain unknown").
        assert dict(current.get_conversation_by_id(conversation_id)) == (
            archived_before | {"archived_at": None}
        )
        assert tuple(
            tuple(row) for row in connection.execute("PRAGMA table_info(conversations)")
        ) == columns_before + ((len(columns_before), "archived_at", "TEXT", 0, None, 0),)
        schema_after = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE tbl_name = 'conversations' ORDER BY type, name"
            )
        )
        # Intentional conversations-schema changes after v72 (TASK-42517
        # added the last three): ADR-224's sargable keyset index (v78->v79,
        # conversations only), ADR-216's browse-order pair (v79->v80), the
        # response-rules cleanup trigger (v80->v81, d24c4c04b2), ADR-166's
        # archive chronology (v84->v85, dd8d60c9e8: the `archived_at` column,
        # one index and a rewritten `conversations_sync_update`), and the
        # prompt mix-ins' `conversations_ad` rewrite (v85->v86, 7970fbcada).
        # The later statements are read from their shipped migration files.
        # Strip exactly the sargable row and the v72 preservation guarantee
        # still pins everything else.
        rewritten = {
            ("trigger", "conversations_sync_update"): _migration_statement(
                "chachanotes_v84_to_v85_archive_chronology.sql",
                "CREATE TRIGGER conversations_sync_update",
            ),
            ("trigger", "conversations_ad"): _migration_statement(
                "chachanotes_prompt_mixins.sql", "CREATE TRIGGER conversations_ad"
            ),
        }
        later = (
            browse_order_indexes
            + (
                (
                    "trigger",
                    "console_response_rules_conversation_cleanup",
                    _migration_statement(
                        "chachanotes_v80_to_v81_response_rules.sql",
                        "CREATE TRIGGER console_response_rules_conversation_cleanup",
                    ),
                ),
                (
                    "index",
                    "idx_conversations_archive_chronology",
                    _migration_statement(
                        "chachanotes_v84_to_v85_archive_chronology.sql",
                        "CREATE INDEX idx_conversations_archive_chronology",
                    ),
                ),
            )
            + tuple((*key, sql) for key, sql in rewritten.items())
        )
        kept = tuple(
            (kind, name, sql[:-1] + ", archived_at TEXT)")
            if (kind, name) == ("table", "conversations")
            else (kind, name, sql)
            for kind, name, sql in schema_before
            if (kind, name) not in rewritten
        )
        assert tuple(
            row for row in schema_after if row[1] != "idx_conv_char_lm"
        ) == tuple(sorted(kept + later))
        assert any(row[1] == "idx_conv_char_lm" for row in schema_after)
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


@pytest.mark.parametrize("origin", ["fresh", "upgraded_from_v87"])
def test_every_completed_pair_reconciliation_locator_uses_indexed_search(
    tmp_path: Path, origin: str
) -> None:
    path = tmp_path / "locators.sqlite"
    if origin == "upgraded_from_v87":
        # The v87->v88 step (TASK-42517) must index an existing file too.
        with chachanotes_db_at_version(path, 87, client_id="v87-locators"):
            pass
    db = CharactersRAGDB(
        ":memory:" if origin == "fresh" else path, "v72-index-plans"
    )
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
            # Locators classified after v72 (38467b8fc6, d24c4c04b2) planned
            # as SCAN until v88 indexed them (TASK-42517).
            (
                "console_hook_continuation_receipts",
                "assistant_message_id",
            ): "idx_hook_continuation_receipts_assistant_message",
            (
                "console_hook_continuation_receipts",
                "parent_assistant_message_id",
            ): "idx_hook_continuation_receipts_parent_assistant_message",
            (
                "console_machine_followup_receipts",
                "assistant_message_id",
            ): "idx_machine_followup_receipts_assistant_message",
            (
                "console_machine_followup_receipts",
                "parent_assistant_message_id",
            ): "idx_machine_followup_receipts_parent_assistant_message",
            (
                "console_response_rule_assessments",
                "message_id",
            ): "idx_response_rule_assessments_message",
            (
                "console_response_rule_drafts",
                "message_id",
            ): "idx_response_rule_drafts_message",
            (
                "console_response_rule_fixtures",
                "message_id",
            ): "idx_response_rule_fixtures_message",
            (
                "console_response_rule_validations",
                "message_id",
            ): "idx_response_rule_validations_message",
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
