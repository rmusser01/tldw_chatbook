"""Trace v2 event-family coverage at the existing durable owner seams."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tldw_chatbook.Chat.trajectory import derive_trajectory


@dataclass(frozen=True)
class Sidecar:
    message_id: str
    conversation_id: str
    turn_id: str
    seq: int | None
    event_kind: str
    step_started_at: float | None = None
    first_token_at: float | None = None
    completed_at: float | None = None
    model: str | None = None
    provider: str | None = None
    payload_json: str | None = None


def _records(**sources):
    snapshot = derive_trajectory(
        messages=sources.pop("messages", ()),
        usage_by_id={},
        traj_rows=sources.pop("traj_rows", ()),
        variant_sets=(),
        compaction_records=sources.pop("compaction_records", ()),
        **sources,
    )
    return [record for turn in snapshot.turns for record in turn.records]


def test_approved_event_family_matrix_projects_every_owned_observation() -> None:
    messages = [
        {
            "id": "s1",
            "conversation_id": "conv-1",
            "sender": "system",
            "content": "internal system instructions: credential=sk-secret",
            "timestamp": 1.0,
            "parent_message_id": None,
            "deleted": False,
        },
        {
            "id": "u1",
            "conversation_id": "conv-1",
            "sender": "user",
            "content": "inspect",
            "timestamp": 2.0,
            "parent_message_id": "s1",
            "deleted": False,
        },
        {
            "id": "a1",
            "conversation_id": "conv-1",
            "sender": "assistant",
            "content": "done",
            "timestamp": 6.0,
            "parent_message_id": "u1",
            "deleted": False,
        },
    ]
    rows = [
        Sidecar("u1", "conv-1", "turn-1", 1, "user"),
        Sidecar(
            "a1",
            "conv-1",
            "turn-1",
            2,
            "assistant",
            step_started_at=3.0,
            first_token_at=4.0,
            completed_at=5.0,
            model="model",
            provider="provider",
        ),
        Sidecar("a1", "conv-1", "turn-1", 3, "user_feedback"),
        Sidecar("u1", "conv-1", "turn-1", 4, "message_edited"),
        Sidecar("a1", "conv-1", "turn-1", 5, "message_regenerated"),
        Sidecar("a1", "conv-1", "turn-1", 6, "branch_selected"),
        Sidecar("a1", "conv-1", "turn-1", 7, "context_attached"),
        Sidecar("a1", "conv-1", "turn-1", 8, "context_injected"),
    ]
    agent_steps = [
        {
            "run_id": "run-1",
            "conversation_id": "conv-1",
            "turn_id": "turn-1",
            "index": index,
            "kind": kind,
            "created_at": f"2026-08-22T12:00:{index:02d}Z",
            "status": status,
            "parent_event_id": parent,
            "field_states": {"args": "omitted", "result": "omitted"},
            "sensitivity": "diagnostic",
        }
        for index, (kind, status, parent) in enumerate(
            (
                ("model_retry", "retrying", "agent-run:run-1"),
                ("model_error", "failed", "agent-run:run-1"),
                ("model_cancelled", "cancelled", "agent-run:run-1"),
                ("tool_proposed", "proposed", "agent-run:run-1"),
                ("approval_requested", "pending", "agent-step:run-1:3"),
                ("approval_approved", "approved", "agent-step:run-1:4"),
                ("approval_denied", "denied", "agent-step:run-1:4"),
                ("approval_revoked", "revoked", "agent-step:run-1:4"),
                ("tool_execution_started", "running", "agent-step:run-1:3"),
                ("tool_succeeded", "succeeded", "agent-step:run-1:8"),
                ("tool_failed", "failed", "agent-step:run-1:8"),
                ("tool_timed_out", "timed_out", "agent-step:run-1:8"),
                ("tool_cancelled", "cancelled", "agent-step:run-1:8"),
            )
        )
    ]
    records = _records(
        messages=messages,
        traj_rows=rows,
        agent_runs=[
            {
                "id": "run-1",
                "conversation_id": "conv-1",
                "turn_id": "turn-1",
                "status": "running",
                "created_at": "2026-08-22T12:00:00Z",
            }
        ],
        agent_steps=agent_steps,
        retrieval_runs=[
            {
                "run_id": "rag-1",
                "conversation_id": "conv-1",
                "turn_id": "turn-1",
                "run_ordinal": 1,
                "stage": "hybrid_search",
                "started_at": "2026-08-22T11:59:58Z",
                "ended_at": "2026-08-22T11:59:59Z",
                "status": "complete",
                "trace_lifecycle": True,
            }
        ],
        compaction_records=[
            {
                "operation_id": "compact-1",
                "conversation_id": "conv-1",
                "purpose": "conversation_compaction",
                "status": "succeeded",
                "started_at": "2026-08-22T11:59:56Z",
                "finished_at": "2026-08-22T11:59:57Z",
                "trace_lifecycle": True,
            }
        ],
    )
    kinds = {record.kind for record in records}

    expected = {
        "system",
        "user",
        "assistant",
        "user_feedback",
        "message_edited",
        "message_regenerated",
        "branch_selected",
        "model_request_started",
        "model_first_token",
        "model_response_completed",
        "model_retry",
        "model_error",
        "model_cancelled",
        "tool_proposed",
        "approval_requested",
        "approval_approved",
        "approval_denied",
        "approval_revoked",
        "tool_execution_started",
        "tool_succeeded",
        "tool_failed",
        "tool_timed_out",
        "tool_cancelled",
        "retrieval_started",
        "retrieval_candidates_selected",
        "retrieval_completed",
        "context_attached",
        "context_injected",
        "compaction_started",
        "compaction_completed",
    }
    assert expected <= kinds

    system = next(record for record in records if record.kind == "system")
    assert system.content_preview == "System context attached"
    assert system.field_states["content_preview"] == "omitted"
    assert system.sensitivity == "system_context"
    assert "sk-secret" not in repr(system)

    by_kind = {record.kind: record for record in records}
    assert by_kind["model_request_started"].observed_at == 3.0
    assert (
        by_kind["model_first_token"].parent_event_id
        == by_kind["model_request_started"].event_id
    )
    assert (
        by_kind["model_response_completed"].parent_event_id
        == by_kind["model_first_token"].event_id
    )
    assert (
        by_kind["retrieval_candidates_selected"].field_states["observed_at"]
        == "not_available"
    )
    assert by_kind["retrieval_candidates_selected"].sensitivity == "retrieval_metadata"


@pytest.mark.parametrize(
    ("terminal_status", "expected_kind"),
    (("failed", "compaction_failed"), ("cancelled", "compaction_cancelled")),
)
def test_compaction_terminal_status_projects_distinct_outcome(
    terminal_status: str, expected_kind: str
) -> None:
    records = _records(
        messages=[
            {
                "id": "u1",
                "conversation_id": "conv-1",
                "sender": "user",
                "content": "go",
                "timestamp": 1,
                "parent_message_id": None,
                "deleted": False,
            }
        ],
        compaction_records=[
            {
                "operation_id": "compact-1",
                "conversation_id": "conv-1",
                "purpose": "conversation_compaction",
                "status": terminal_status,
                "started_at": 2,
                "finished_at": 3,
                "trace_lifecycle": True,
            }
        ],
    )
    assert [
        record.kind for record in records if record.kind.startswith("compaction_")
    ] == [
        "compaction_started",
        expected_kind,
    ]
