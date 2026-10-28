"""Bounded repository tails and immutable source masks for bug reports."""

import sqlite3

import pytest

from tldw_chatbook.Chat.console_semantic_revision import (
    SemanticRevisionCoordinator,
    project_semantic_revision_trace_tool_fields,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    TraceOmission,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_redaction import (
    BUILTIN_PII_RULESET_REVISION_ID,
    CREDENTIAL_FILTER_VERSION,
    PIIRedactionSpan,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def database():
    db = CharactersRAGDB(":memory:", "bug-report-tests")
    yield db
    db.close_connection()


def test_report_event_tail_is_bounded_and_never_reads_payload(database):
    repo = ConsoleTraceRepository()
    conversation_id = database.add_conversation({"title": "PRIVATE_TITLE"})
    with database.transaction() as cursor:
        segment = repo.create_segment(cursor)
        repo.attach_owner(
            cursor, conversation_id=conversation_id, root_segment_id=segment.segment_id
        )
        for seq in range(700):
            repo.append_event(
                cursor,
                segment_id=segment.segment_id,
                sequence=seq,
                event_type="gap",
                omission_reason_code="test_gap",
            )
        tail = repo.read_report_event_tail(cursor, conversation_id, limit=7)
        assert [event.sequence for event in tail.events] == list(range(693, 700))
        assert tail.has_older and not tail.lineage_truncated


def test_fork_tail_respects_inherited_fence_and_segment_cap(database):
    repo = ConsoleTraceRepository()
    parent_conversation_id = database.add_conversation({"title": "parent"})
    conversation_id = database.add_conversation({"title": "fork"})
    with database.transaction() as cursor:
        parent = repo.create_segment(cursor)
        repo.attach_owner(
            cursor,
            conversation_id=parent_conversation_id,
            root_segment_id=parent.segment_id,
        )
        head = repo.append_surface_node(
            cursor,
            segment_id=parent.segment_id,
            sequence=0,
            predecessor_node_id=None,
            component_kind="message",
            reference=TraceOmission("message", "test_gap"),
        )
        repo.append_event(
            cursor,
            segment_id=parent.segment_id,
            sequence=0,
            event_type="surface_append",
            surface_node_id=head.node_id,
        )
        for seq in range(1, 12):
            repo.append_event(
                cursor,
                segment_id=parent.segment_id,
                sequence=seq,
                event_type="gap",
                omission_reason_code="test_gap",
            )
        child = repo.create_segment(
            cursor,
            parent_segment_id=parent.segment_id,
            inherited_through_sequence=5,
            inherited_surface_head_id=head.node_id,
        )
        repo.attach_owner(
            cursor, conversation_id=conversation_id, root_segment_id=child.segment_id
        )
        for seq in range(4):
            repo.append_event(
                cursor,
                segment_id=child.segment_id,
                sequence=seq,
                event_type="gap",
                omission_reason_code="test_gap",
            )
        tail = repo.read_report_event_tail(cursor, conversation_id, limit=20)
        assert [(event.segment_id, event.sequence) for event in tail.events] == (
            [(parent.segment_id, seq) for seq in range(6)]
            + [(child.segment_id, seq) for seq in range(4)]
        )
        short = repo.read_report_event_tail(cursor, conversation_id, max_segments=1)
        assert short.lineage_truncated and len(short.events) == 4


def _tool_source(database, pii):
    repo = ConsoleTraceRepository()
    cid = database.add_conversation({"title": "tool evidence"})
    mid = database.add_message(
        {
            "conversation_id": cid,
            "sender": "tool",
            "role": "tool",
            "content": "CUSTOM_PRIVATE_CONTENT",
        }
    )
    policy = FrozenTracePolicy(
        new_opaque_id(),
        CREDENTIAL_FILTER_VERSION,
        pii,
        BUILTIN_PII_RULESET_REVISION_ID if pii else None,
    )
    with database.transaction() as cursor:
        revision = SemanticRevisionCoordinator(database).ensure_current_revision(
            cursor, message_id=mid
        )
        repo.ensure_policy(cursor, policy)
    return cid, revision, policy


def test_frozen_custom_mask_is_not_recovered_from_canonical_tool_result(database):
    cid, revision, policy = _tool_source(database, True)
    repo = ConsoleTraceRepository()
    with database.transaction() as cursor:
        repo.ensure_redaction_spans(
            cursor,
            policy_id=policy.policy_id,
            semantic_revision_id=revision.revision_id,
            artifact_id=None,
            field_path="$/@0",
            spans=(PIIRedactionSpan(0, 22, "custom", "user-rule", "custom-v1"),),
        )
        result = project_semantic_revision_trace_tool_fields(
            cursor,
            revision_id=revision.revision_id,
            expected_conversation_id=cid,
            policy_id=policy.policy_id,
        )
        assert "CUSTOM_PRIVATE_CONTENT" not in str(result)
        assert "PII omitted" in str(result)
        assert (
            cursor.execute(
                "SELECT content FROM messages WHERE id = ?",
                (revision.source_message_id,),
            ).fetchone()[0]
            == "CUSTOM_PRIVATE_CONTENT"
        )


def test_missing_policy_has_no_raw_legacy_fallback(database):
    cid, revision, _policy = _tool_source(database, False)
    with database.transaction() as cursor:
        with pytest.raises(ValueError):
            project_semantic_revision_trace_tool_fields(
                cursor,
                revision_id=revision.revision_id,
                expected_conversation_id=cid,
                policy_id=new_opaque_id(),
            )


def test_tool_projection_never_reads_images_or_unselected_content(database):
    cid, revision, policy = _tool_source(database, False)
    connection = database.get_connection()

    def guard(action, first, second, *_):
        if action == sqlite3.SQLITE_READ and (
            second
            in {"image_data", "thinking_blocks_json", "provider_continuation_json"}
            or first == "message_attachments"
        ):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(guard)
    try:
        with database.transaction() as cursor:
            result = project_semantic_revision_trace_tool_fields(
                cursor,
                revision_id=revision.revision_id,
                expected_conversation_id=cid,
                policy_id=policy.policy_id,
            )
            assert result["result"] == "CUSTOM_PRIVATE_CONTENT"
    finally:
        connection.set_authorizer(None)


def test_large_omitted_tool_source_is_not_loaded(database):
    cid, revision, policy = _tool_source(database, False)
    with database.transaction() as cursor:
        result = project_semantic_revision_trace_tool_fields(
            cursor,
            revision_id=revision.revision_id,
            expected_conversation_id=cid,
            policy_id=policy.policy_id,
            max_source_bytes=8,
        )
    assert "CUSTOM_PRIVATE_CONTENT" not in str(result)
    assert "unavailable" in str(result)


def test_metadata_readers_exclude_message_bodies_and_agent_results(database):
    cid = database.add_conversation({"title": "metadata"})
    database.add_message(
        {"conversation_id": cid, "sender": "assistant", "content": "PRIVATE_MESSAGE"}
    )
    with database.transaction() as cursor:
        cursor.execute(
            "INSERT INTO message_trajectory_metadata(message_id, conversation_id, turn_id, seq, event_kind, payload_json) SELECT id, conversation_id, id, 0, 'tool_result', ? FROM messages WHERE conversation_id = ?",
            ('{"result":"PRIVATE_BODY"}', cid),
        )
    facts = database.get_report_trajectory_tail(cid)
    assert len(facts) == 1
    assert "PRIVATE_BODY" not in str(facts) and "PRIVATE_MESSAGE" not in str(facts)
    runs = AgentRunsDB(":memory:", "bug-report-agent-tests", reconcile_on_init=False)
    try:
        rid = runs.create_run(conversation_id=cid, agent_kind="primary")
        runs.append_steps(
            rid,
            [
                {
                    "kind": "tool_result",
                    "tool_name": "fs_read",
                    "result": "PRIVATE_BODY",
                    "status": "error",
                }
            ],
        )
        rows = runs.get_report_event_tail(cid, limit=1)
        assert len(rows) == 1 and "PRIVATE_BODY" not in str(rows)
        assert "fs_read" in str(rows)
    finally:
        runs.close()


@pytest.mark.parametrize("foreign", [False, True])
def test_tool_reader_validates_revision_owner_in_selected_lineage(database, foreign):
    source_cid, revision, policy = _tool_source(database, False)
    selected_cid = (
        database.add_conversation({"title": "other work"}) if foreign else source_cid
    )
    turn_id = database.add_message(
        {"conversation_id": source_cid, "sender": "user", "content": "question"}
    )
    repo = ConsoleTraceRepository()
    with database.transaction() as cursor:
        segment = repo.create_segment(cursor)
        owner = repo.attach_owner(
            cursor, conversation_id=source_cid, root_segment_id=segment.segment_id
        )
        node = repo.append_surface_node(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            predecessor_node_id=None,
            component_kind="tool_result",
            reference=SemanticRevisionRef(revision.revision_id),
        )
        call = repo.reserve_call(
            cursor,
            owner_id=owner.owner_id,
            segment_id=segment.segment_id,
            turn_id=turn_id,
            run_id="report-test-run",
            call_sequence=0,
            idempotency_key="report-test-call",
            policy_id=policy.policy_id,
        )
        header = repo.create_or_reuse_request_header(
            cursor,
            provider_name="test",
            model_name="test",
            route_identity="fresh",
            endpoint_identity="https://example.invalid",
            generation_parameters={},
            adapter_defaults={},
            response_format={},
            reasoning_controls={},
            components=(),
        )
        repo.bind_call(
            cursor,
            call_id=call.call_id,
            surface_node_id=node.node_id,
            request_header_id=header.header_id,
            provider_name="test",
            model_name="test",
            route_identity="fresh",
        )
        event = repo.append_event(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            event_type="tool_result",
            call_id=call.call_id,
            semantic_revision_id=revision.revision_id,
        )
        if foreign:
            other_segment = repo.create_segment(cursor)
            repo.attach_owner(
                cursor,
                conversation_id=selected_cid,
                root_segment_id=other_segment.segment_id,
            )
    fields = ConsoleTraceNativeReader(database).read_report_tool_fields(
        selected_cid, (event,)
    )
    if foreign:
        assert "CUSTOM_PRIVATE_CONTENT" not in str(fields)
        assert "unavailable" in str(fields)
    else:
        assert "CUSTOM_PRIVATE_CONTENT" in str(fields)
