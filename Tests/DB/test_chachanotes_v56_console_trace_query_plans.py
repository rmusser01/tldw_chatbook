"""No-statistics query-plan pins for the Console semantic trace indexes."""

from __future__ import annotations

from collections.abc import Iterator
import sqlite3

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


RESPONSE_REVISION_LOCATOR = (
    "SELECT response_link_id FROM console_trace_response_links "
    "WHERE semantic_revision_id = ?"
)

TRACE_INDEX_QUERIES: dict[str, tuple[str, tuple[object, ...]]] = {
    "idx_console_trace_artifacts_identity": (
        "SELECT artifact_id FROM console_trace_artifacts "
        "WHERE identity_digest = ? AND media_type = ? AND normalization_version = ?",
        ("0" * 64, "application/json", "v1"),
    ),
    "idx_console_trace_calls_owner_order": (
        "SELECT call_id FROM console_trace_calls WHERE owner_id = ? "
        "ORDER BY turn_id, run_id, call_sequence",
        ("owner",),
    ),
    "idx_console_trace_calls_segment_order": (
        "SELECT call_id FROM console_trace_calls WHERE segment_id = ? "
        "ORDER BY turn_id, run_id, call_sequence",
        ("segment",),
    ),
    "idx_console_trace_calls_surface_policy": (
        "SELECT call_id FROM console_trace_calls "
        "WHERE surface_node_id = ? AND policy_id = ?",
        ("node", "policy"),
    ),
    "idx_console_trace_events_call_order": (
        "SELECT event_id FROM console_trace_events WHERE call_id = ? ORDER BY sequence",
        ("call",),
    ),
    "idx_console_trace_events_segment_order": (
        "SELECT event_id FROM console_trace_events WHERE segment_id = ? "
        "ORDER BY sequence",
        ("segment",),
    ),
    "idx_console_trace_header_components_artifact": (
        "SELECT header_id FROM console_trace_header_components "
        "WHERE artifact_id = ? ORDER BY header_id",
        ("artifact",),
    ),
    "idx_console_trace_migration_status": (
        "SELECT migration_name FROM console_trace_migration_state "
        "WHERE status = ? ORDER BY migration_name",
        ("pending",),
    ),
    "idx_console_trace_owners_root_segment": (
        "SELECT owner_id FROM console_trace_owners WHERE root_segment_id = ?",
        ("segment",),
    ),
    "idx_console_trace_redaction_artifact": (
        "SELECT span_id FROM console_trace_redaction_spans "
        "WHERE artifact_id = ? AND policy_id = ? AND field_path = ? "
        "ORDER BY start_codepoint",
        ("artifact", "policy", "$.value"),
    ),
    "idx_console_trace_redaction_revision": (
        "SELECT span_id FROM console_trace_redaction_spans "
        "WHERE semantic_revision_id = ? AND policy_id = ? AND field_path = ? "
        "ORDER BY start_codepoint",
        ("revision", "policy", "$.value"),
    ),
    "idx_console_trace_response_artifact": (
        "SELECT response_link_id FROM console_trace_response_links "
        "WHERE artifact_id = ?",
        ("artifact",),
    ),
    # Since schema 72 (b7dd8e53f2) the exact full index wins this locator
    # over v56's partial idx_console_trace_response_revision; the v56 index
    # is pinned on a v71 file below (TASK-42517).
    "idx_console_trace_response_links_semantic_revision_exact": (
        RESPONSE_REVISION_LOCATOR,
        ("revision",),
    ),
    "idx_console_trace_revision_bindings_artifact": (
        "SELECT revision_id FROM console_trace_revision_bindings WHERE artifact_id = ?",
        ("artifact",),
    ),
    "idx_console_trace_segments_parent_boundary": (
        "SELECT segment_id FROM console_trace_segments "
        "WHERE parent_segment_id = ? "
        "ORDER BY inherited_through_sequence, inherited_surface_head_id",
        ("parent",),
    ),
    "idx_console_trace_semantic_revisions_source": (
        "SELECT revision_id FROM console_trace_semantic_revisions "
        "WHERE source_conversation_id = ? AND source_message_id = ? "
        "ORDER BY revision_sequence",
        ("conversation", "message"),
    ),
    "idx_console_trace_surface_nodes_predecessor": (
        "SELECT node_id FROM console_trace_surface_nodes "
        "WHERE predecessor_node_id = ? AND segment_id = ?",
        ("predecessor", "segment"),
    ),
    "idx_console_trace_surface_nodes_revision": (
        "SELECT node_id FROM console_trace_surface_nodes "
        "WHERE semantic_revision_id = ? ORDER BY node_id",
        ("revision",),
    ),
    "idx_console_trace_surface_nodes_segment_order": (
        "SELECT node_id FROM console_trace_surface_nodes WHERE segment_id = ? "
        "ORDER BY sequence",
        ("segment",),
    ),
    "idx_console_trace_surface_replacements_predecessor": (
        "SELECT replacement_id FROM console_trace_surface_replacements "
        "WHERE segment_id = ? AND predecessor_head_id = ?",
        ("segment", "head"),
    ),
    "uq_console_trace_calls_idempotency": (
        "SELECT call_id FROM console_trace_calls WHERE idempotency_key = ?",
        ("idempotency",),
    ),
    "uq_console_trace_calls_owner_sequence": (
        "SELECT call_id FROM console_trace_calls "
        "WHERE owner_id = ? AND segment_id = ? AND turn_id = ? "
        "AND run_id = ? AND call_sequence = ?",
        ("owner", "segment", "turn", "run", 0),
    ),
    "uq_console_trace_semantic_revisions_live_message": (
        "SELECT revision_id FROM console_trace_semantic_revisions "
        "WHERE live_message_id = ?",
        ("message",),
    ),
}


@pytest.fixture(scope="module")
def trace_plan_connection() -> Iterator[sqlite3.Connection]:
    """Yield a fully migrated connection that has never run ANALYZE.

    Yields:
        The shared query-plan test connection.
    """

    db = CharactersRAGDB(":memory:", client_id="trace-query-plan-pins")
    connection = db.get_connection()
    yield connection
    db.close_connection()


@pytest.mark.parametrize("index_name", sorted(TRACE_INDEX_QUERIES))
def test_console_trace_query_uses_index_without_statistics(
    trace_plan_connection: sqlite3.Connection,
    index_name: str,
) -> None:
    """Each trace access path must choose its intended index without stats."""

    stats_table = trace_plan_connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_stat1'"
    ).fetchone()
    assert stats_table is None

    sql, params = TRACE_INDEX_QUERIES[index_name]
    rows = trace_plan_connection.execute(
        "EXPLAIN QUERY PLAN " + sql,
        params,
    ).fetchall()
    plan = "\n".join(str(row[-1]) for row in rows)
    assert index_name in plan, plan


def test_response_revision_index_served_its_locator_until_schema_72(tmp_path) -> None:
    """v56's partial index chose the revision locator on every v56-v71 file.

    Schema 72 added ``idx_console_trace_response_links_semantic_revision_exact``
    on the same column (ADR-338's completed-pair locators), and that full
    index wins the same plan on a current file (the parametrized pin above).
    No product query plans through the v56 index any more (TASK-42517), and
    schema 89 drops it (TASK-42544).
    """

    with chachanotes_db_at_version(tmp_path / "v71.sqlite", 71) as historical:
        connection = historical.get_connection()
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'sqlite_stat1'"
            ).fetchone()
            is None
        )
        plan = "\n".join(
            str(row[-1])
            for row in connection.execute(
                "EXPLAIN QUERY PLAN " + RESPONSE_REVISION_LOCATOR, ("revision",)
            )
        )
    assert "idx_console_trace_response_revision" in plan, plan


class _PlanCursor:
    """Answers each product statement with its query plan instead of its rows."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.plans: list[str] = []

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        self.plans.append(
            " | ".join(
                str(row[-1])
                for row in self.connection.execute("EXPLAIN QUERY PLAN " + sql, params)
            )
        )
        return self.connection.execute("SELECT 1 WHERE 0")


@pytest.mark.parametrize("origin", ["fresh", "upgraded_from_v88"])
def test_response_revision_queries_keep_the_exact_index_after_v89(
    tmp_path, origin
) -> None:
    """Schema 89 drops v56's partial index; no plan on the column changes.

    Four reads of ``console_trace_response_links.semantic_revision_id`` plan
    here with no ``sqlite_stat1``: this file's exact locator
    (``RESPONSE_REVISION_LOCATOR``), and three product reads run through
    their own methods -- the semantic-revision policy join, the
    voice-promotion locator check and the completed-replay terminal lookup.
    Each still plans through schema 72's exact index (TASK-42544).
    """
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_semantic_revision import (
        SemanticRevisionCoordinator,
    )
    from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

    path = tmp_path / "response-revision.sqlite"
    if origin == "upgraded_from_v88":
        with chachanotes_db_at_version(path, 88, client_id="v88-response-revision"):
            pass
    db = CharactersRAGDB(path, "response-revision-plans")
    try:
        connection = db.get_connection()
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'sqlite_stat1'"
            ).fetchone()
            is None
        )
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert "idx_console_trace_response_revision" not in indexes
        cursor = _PlanCursor(connection)
        cursor.execute(RESPONSE_REVISION_LOCATOR, ("revision",))
        SemanticRevisionCoordinator._reachable_policy_ids(cursor, "revision")
        ChatPersistenceService._voice_promotion_locator_exists(
            cursor,
            locators=frozenset({("console_trace_response_links", "semantic_revision_id")}),
            identifiers=("revision-a", "revision-b"),
        )
        # No row comes back, so the lookup stops at its first statement.
        with pytest.raises(ValueError, match="completed_replay_terminal"):
            ConsoleTraceService._completed_replay_values(None, cursor, "owner", "revision")
        assert len(cursor.plans) == 4
        for plan in cursor.plans:
            assert "idx_console_trace_response_links_semantic_revision_exact" in plan, plan
    finally:
        db.close_connection()
