"""No-statistics query-plan pins for the Console semantic trace indexes."""

from __future__ import annotations

from collections.abc import Iterator
import sqlite3

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


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
        "SELECT event_id FROM console_trace_events WHERE call_id = ? "
        "ORDER BY sequence",
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
    "idx_console_trace_response_revision": (
        "SELECT response_link_id FROM console_trace_response_links "
        "WHERE semantic_revision_id = ?",
        ("revision",),
    ),
    "idx_console_trace_revision_bindings_artifact": (
        "SELECT revision_id FROM console_trace_revision_bindings "
        "WHERE artifact_id = ?",
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
