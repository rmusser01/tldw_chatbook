"""Legacy exchange normalization into isolated reference-backed snapshots."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_exchange_capture import (
    CAPTURE_SAFE_HISTORY_TAIL_ROWS,
    CaptureDetail,
    ExchangeCapture,
    build_request_capture,
    capture_to_blob,
)
from tldw_chatbook.Chat.console_trace_legacy import LegacyTraceNormalizer
from tldw_chatbook.Chat.console_trace_models import TraceContentRef
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db() -> CharactersRAGDB:
    database = CharactersRAGDB(":memory:", "legacy-trace-normalizer-test")
    yield database
    database.close_connection()


def _untracked_message(
    db: CharactersRAGDB,
    *,
    conversation_id: str,
    role: str,
    content: str,
) -> str:
    message_id = db._generate_uuid()
    now = db._get_current_utc_timestamp_iso()
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            """INSERT INTO messages(
                   id, conversation_id, sender, content, timestamp,
                   last_modified, client_id, version, deleted, role)
                 VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?)""",
            (message_id, conversation_id, role, content, now, now, db.client_id, role),
        )
    return message_id


def _capture(*, messages: list[dict[str, object]]) -> ExchangeCapture:
    return ExchangeCapture(
        run_tag="legacy-run",
        seq=2,
        created_at="2026-08-28T12:00:00+00:00",
        provider="openai",
        model="gpt-test",
        endpoint="https://example.test/v1",
        request={
            "messages_payload": messages,
            "system_message": "legacy framing",
            "tools": [{"type": "function", "function": {"name": "clock"}}],
        },
        response={"content": "answer", "tool_calls": []},
        status="complete",
        usage_json=json.dumps({"uncached_input": 5, "output": 1}),
        omitted_keys=(),
        capture_detail=CaptureDetail.FULL,
    )


def _row(
    capture: ExchangeCapture,
    *,
    exchange_id: int,
    message_id: str,
) -> dict[str, object]:
    return {
        "id": exchange_id,
        "message_id": message_id,
        "run_tag": capture.run_tag,
        "seq": capture.seq,
        "status": capture.status,
        "abandoned": False,
        "capture_detail": capture.capture_detail.value,
        "capture_blob": capture_to_blob(capture),
        "created_at": capture.created_at,
    }


def test_normalize_full_capture_uses_unique_canonical_revision_and_snapshot(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy trace"})
    assert conversation_id is not None
    user_message_id = _untracked_message(
        db,
        conversation_id=conversation_id,
        role="user",
        content="saved ordinary message",
    )
    assistant_message_id = _untracked_message(
        db,
        conversation_id=conversation_id,
        role="assistant",
        content="answer",
    )
    capture = _capture(messages=[{"role": "user", "content": "saved ordinary message"}])
    row = _row(capture, exchange_id=17, message_id=assistant_message_id)
    repository = ConsoleTraceRepository()
    normalizer = LegacyTraceNormalizer(db, repository=repository)

    with db.transaction(immediate=True) as cursor:
        result = normalizer.normalize_exchange(cursor, row)

        revision = cursor.execute(
            """SELECT revision_id, live_message_id
                 FROM console_trace_semantic_revisions
                WHERE source_message_id = ?""",
            (user_message_id,),
        ).fetchone()
        assert revision is not None
        assert revision[1] == user_message_id

        message_nodes = cursor.execute(
            """SELECT component_kind, reference_kind, semantic_revision_id
                 FROM console_trace_surface_nodes
                WHERE component_kind = 'legacy_snapshot_message'"""
        ).fetchall()
        message_nodes = [tuple(item) for item in message_nodes]
        assert message_nodes == [("legacy_snapshot_message", "revision", revision[0])]
        terminal = repository.get_surface_node(cursor, result.surface_head_id)
        assert terminal is not None
        assert terminal.component_kind == "legacy_snapshot"
        assert terminal.predecessor_node_id is not None

        artifacts = cursor.execute(
            "SELECT sanitized_bytes FROM console_trace_artifacts"
        ).fetchall()
        assert all(
            b"saved ordinary message" not in bytes(item[0]) for item in artifacts
        )

    calls = normalizer.read_calls(assistant_message_id)
    assert len(calls) == 1
    assert calls[0].capture == capture
    assert calls[0].verified is True
    assert calls[0].provenance == "legacy_snapshot"
    assert calls[0].chronology == "recorded_call_only"


def test_repeated_legacy_history_reuses_persistent_prefix_nodes(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy prefix"})
    assert conversation_id is not None
    _untracked_message(
        db, conversation_id=conversation_id, role="user", content="first"
    )
    first_assistant = _untracked_message(
        db, conversation_id=conversation_id, role="assistant", content="one"
    )
    _untracked_message(
        db, conversation_id=conversation_id, role="user", content="second"
    )
    second_assistant = _untracked_message(
        db, conversation_id=conversation_id, role="assistant", content="two"
    )
    first = _capture(messages=[{"role": "user", "content": "first"}])
    second = replace(
        _capture(
            messages=[
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ]
        ),
        seq=3,
        created_at="2026-08-28T12:01:00+00:00",
        response={"content": "two", "tool_calls": []},
    )
    normalizer = LegacyTraceNormalizer(db)

    with db.transaction(immediate=True) as cursor:
        normalizer.normalize_exchange(
            cursor, _row(first, exchange_id=21, message_id=first_assistant)
        )
        normalizer.normalize_exchange(
            cursor, _row(second, exchange_id=22, message_id=second_assistant)
        )
        nodes = cursor.execute(
            """SELECT node_id, predecessor_node_id
                 FROM console_trace_surface_nodes
                WHERE component_kind = 'legacy_snapshot_message'
                ORDER BY sequence"""
        ).fetchall()
        terminals = cursor.execute(
            """SELECT predecessor_node_id
                 FROM console_trace_surface_nodes
                WHERE component_kind = 'legacy_snapshot'
                ORDER BY sequence"""
        ).fetchall()

    assert len(nodes) == 2
    assert nodes[1][1] == nodes[0][0]
    assert terminals[0][0] == nodes[0][0]
    assert terminals[1][0] == nodes[1][0]


def test_duplicate_canonical_matches_remain_ambiguous_legacy_artifact(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy ambiguity"})
    assert conversation_id is not None
    for _index in range(2):
        _untracked_message(
            db,
            conversation_id=conversation_id,
            role="user",
            content="duplicate",
        )
    assistant_message = _untracked_message(
        db, conversation_id=conversation_id, role="assistant", content="answer"
    )
    capture = _capture(messages=[{"role": "user", "content": "duplicate"}])
    normalizer = LegacyTraceNormalizer(db)

    with db.transaction(immediate=True) as cursor:
        result = normalizer.normalize_exchange(
            cursor, _row(capture, exchange_id=31, message_id=assistant_message)
        )
        node = cursor.execute(
            """SELECT reference_kind, semantic_revision_id, artifact_id
                 FROM console_trace_surface_nodes
                WHERE component_kind = 'legacy_snapshot_message'"""
        ).fetchone()

    assert node is not None
    assert tuple(node[:2]) == ("artifact", None)
    assert node[2] is not None
    assert "legacy_message_source_unknown" in result.uncertainty_codes
    restored_call = normalizer.read_calls(assistant_message)[0]
    assert restored_call.capture == capture
    assert "legacy_message_source_unknown" in restored_call.uncertainty_codes


def test_safe_aggregate_history_marker_becomes_explicit_legacy_omission(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy safe omission"})
    assert conversation_id is not None
    assistant_message = _untracked_message(
        db, conversation_id=conversation_id, role="assistant", content="answer"
    )
    history = [
        {"role": "user", "content": f"history-{index}"}
        for index in range(CAPTURE_SAFE_HISTORY_TAIL_ROWS + 4)
    ]
    request, omitted = build_request_capture(
        {"messages_payload": history},
        capture_detail=CaptureDetail.SAFE,
    )
    capture = replace(
        _capture(messages=[]),
        request=request,
        omitted_keys=omitted,
        capture_detail=CaptureDetail.SAFE,
    )
    normalizer = LegacyTraceNormalizer(db)

    with db.transaction(immediate=True) as cursor:
        result = normalizer.normalize_exchange(
            cursor, _row(capture, exchange_id=41, message_id=assistant_message)
        )
        omissions = cursor.execute(
            """SELECT component_kind, omission_reason_code
                 FROM console_trace_surface_nodes
                WHERE reference_kind = 'omission'
                  AND component_kind = 'legacy_snapshot_message'"""
        ).fetchall()

    assert [tuple(item) for item in omissions] == [
        ("legacy_snapshot_message", "legacy_history_omitted")
    ]
    assert "legacy_history_omitted" in result.uncertainty_codes
    restored_call = normalizer.read_calls(assistant_message)[0]
    assert "legacy_history_omitted" in restored_call.uncertainty_codes
    restored = restored_call.capture
    assert restored.capture_detail is CaptureDetail.SAFE
    assert restored.request["messages_payload"][0] == {
        "kind": "legacy_omission",
        "reason": "legacy_history_omitted",
    }


def test_current_credential_filter_preserves_redacted_canonical_match(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy credential filter"})
    assert conversation_id is not None
    credential = "Bearer eyJhbGciOiJIUzI1NiJ9.super-secret.signature"
    user_message_id = _untracked_message(
        db,
        conversation_id=conversation_id,
        role="user",
        content=f"use {credential}",
    )
    assistant_message_id = _untracked_message(
        db,
        conversation_id=conversation_id,
        role="assistant",
        content="answer",
    )
    capture = _capture(messages=[{"role": "user", "content": f"use {credential}"}])
    normalizer = LegacyTraceNormalizer(db)

    with db.transaction(immediate=True) as cursor:
        result = normalizer.normalize_exchange(
            cursor,
            _row(capture, exchange_id=51, message_id=assistant_message_id),
        )
        reference = cursor.execute(
            """SELECT semantic_revision_id FROM console_trace_surface_nodes
                WHERE component_kind = 'legacy_snapshot_message'"""
        ).fetchone()
        artifact_bytes = b"".join(
            bytes(row[0])
            for row in cursor.execute(
                "SELECT sanitized_bytes FROM console_trace_artifacts"
            )
        )

    assert result.verification_status == "verified"
    assert reference is not None and reference[0] is not None
    with db.transaction() as cursor:
        source = cursor.execute(
            """SELECT source_message_id FROM console_trace_semantic_revisions
                WHERE revision_id = ?""",
            (reference[0],),
        ).fetchone()
    assert source is not None and source[0] == user_message_id
    assert credential.encode() not in artifact_bytes
    restored = normalizer.read_calls(assistant_message_id)[0].capture
    restored_content = restored.request["messages_payload"][0]["content"]
    assert credential not in restored_content
    assert "[credential omitted]" in restored_content


def test_legacy_snapshot_uses_isolated_branch_when_native_owner_already_has_surface(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "mixed rollout"})
    assert conversation_id is not None
    assistant_message_id = _untracked_message(
        db,
        conversation_id=conversation_id,
        role="assistant",
        content="answer",
    )
    repository = ConsoleTraceRepository()
    normalizer = LegacyTraceNormalizer(db, repository=repository)

    with db.transaction(immediate=True) as cursor:
        native_segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=native_segment.segment_id,
        )
        artifact = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=b'"native"',
            media_type="application/json",
            normalization_version="test-v1",
        )
        native_node = repository.append_surface_node(
            cursor,
            segment_id=native_segment.segment_id,
            sequence=0,
            predecessor_node_id=None,
            component_kind="native_message",
            reference=TraceContentRef(artifact.artifact_id, "native_message"),
        )
        repository.append_event(
            cursor,
            segment_id=native_segment.segment_id,
            sequence=0,
            event_type="surface_append",
            surface_node_id=native_node.node_id,
        )

        result = normalizer.normalize_exchange(
            cursor,
            _row(
                _capture(messages=[{"role": "system", "content": "legacy"}]),
                exchange_id=61,
                message_id=assistant_message_id,
            ),
        )
        legacy_root = cursor.execute(
            """SELECT node_id, segment_id, predecessor_node_id
                 FROM console_trace_surface_nodes
                WHERE component_kind = 'legacy_snapshot_root'"""
        ).fetchone()
        call = repository.get_call(cursor, result.call_id)

    assert legacy_root is not None
    assert legacy_root[1] != owner.root_segment_id
    assert legacy_root[2] == native_node.node_id
    assert call is not None
    assert call.segment_id != owner.root_segment_id
