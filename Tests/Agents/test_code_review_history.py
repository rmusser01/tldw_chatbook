"""Accepted checkpoints, rather than run logs, permit whole-round reduction."""

import pytest


@pytest.mark.parametrize("native", [False, True])
def test_only_acknowledged_rounds_are_reduced(native):
    from tldw_chatbook.Agents.code_review_history import (
        ReviewHistoryCheckpoint,
        project_verifier_history,
    )

    checkpoint = ReviewHistoryCheckpoint(
        attempt_id="attempt",
        target_digest="a" * 64,
        submission_revision=1,
        acknowledged_cursors=("ack-1",),
        cursor="next",
        pending_unit_ids=("unit",),
    )
    prefix = [
        {"role": "system", "content": "trusted"},
        {"role": "user", "content": "original task"},
    ]
    if native:
        call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "code_review_submit", "arguments": "{}"},
                }
            ],
        }
        result = {
            "role": "tool",
            "tool_call_id": "call",
            "content": '{"accepted_ids":[],"submission_revision":1,"cursor":"ack-1"}',
        }
    else:
        call = {
            "role": "assistant",
            "content": '```tool_call\n{"name":"code_review_submit","arguments":{}}\n```',
        }
        result = {
            "role": "user",
            "content": 'Tool result for code_review_submit: {"accepted_ids":[],"submission_revision":1,"cursor":"ack-1"}',
        }
    current = {"role": "assistant", "content": "unfinished evidence"}
    messages = prefix + [call, result, current]
    reduced = project_verifier_history(
        messages, checkpoint, continuation_groups=(), native=native
    )
    assert reduced[:2] == prefix
    assert current in reduced
    assert call not in reduced and result not in reduced
    assert "next" in str(reduced)
    assert messages == prefix + [call, result, current]
    from dataclasses import replace

    # A pending check pins the actual active wait and its preceding history.
    pending = replace(checkpoint, pending_check_refs=("proposal",))
    assert (
        project_verifier_history(
            messages, pending, continuation_groups=(), native=native
        )
        == messages
    )
    unacknowledged = ReviewHistoryCheckpoint(
        attempt_id="attempt", target_digest="a" * 64, submission_revision=0
    )
    assert (
        project_verifier_history(
            messages, unacknowledged, continuation_groups=(), native=native
        )
        == messages
    )


from Tests.Agents import code_review_helpers, test_code_review_dispatch

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case
dispatch_case = test_code_review_dispatch.dispatch_case


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("native_tools", [False, True])
@pytest.mark.parametrize("run_log_enabled", [False, True])
def test_actual_verifier_reduces_history_without_run_log(
    dispatch_case, native_tools, run_log_enabled
):
    observed = dispatch_case.run_batched_verifier(
        units=24,
        context_window=8192,
        native_tools=native_tools,
        run_log_enabled=run_log_enabled,
    )
    assert observed.verifier_launch_count == 1
    assert observed.history_reductions > 0
    assert observed.all_provider_requests_fit
    assert len(observed.accepted_unit_dispositions) == 24
    assert observed.replayed_tool_executions == 0
    assert all(
        row["verdict"] == "confirmed" for row in observed.accepted_unit_dispositions
    )
    assert len(observed.expanded) == 24
    assert all(
        "search_run_log" not in str(request["messages_payload"])
        for request in observed.requests
    )


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("native_tools", [False, True])
def test_model_and_transport_capacity_are_independent(dispatch_case, native_tools):
    observed = dispatch_case.run_batched_verifier(
        units=6,
        context_window=32768,
        native_tools=native_tools,
        run_log_enabled=False,
        max_tool_result_chars=1024,
        oversized=True,
    )
    assert len(observed.accepted_unit_dispositions) == 6
    assert observed.all_provider_requests_fit
    assert (
        sum(
            row["verdict"] == "confirmed" for row in observed.accepted_unit_dispositions
        )
        == 5
    )
    assert (
        sum(
            row["reason"] == "review_record_too_large_for_tool_result"
            for row in observed.accepted_unit_dispositions
        )
        == 1
    )
    pages = [
        page
        for request in observed.requests
        for page in test_code_review_dispatch.model_pages(request)
    ]
    assert pages
    assert all(
        len(__import__("json").dumps(page, separators=(",", ":"), ensure_ascii=False))
        <= 1024
        for page in pages
    )


def test_private_owner_and_active_native_pair_are_indivisible():
    from Tests.Agents.test_provider_continuation_runtime import (
        _checkpoint,
        _pending_call,
    )
    from tldw_chatbook.Agents.code_review_history import (
        ReviewHistoryCheckpoint,
        project_verifier_history,
    )
    from tldw_chatbook.Chat.provider_continuation import ContinuationOwnerGroup

    private = _checkpoint(_pending_call("call"))
    owner = ContinuationOwnerGroup("owner", private, private.rounds)
    prefix = [
        {"role": "system", "content": "trusted"},
        {"role": "user", "content": "task"},
    ]
    call = {
        "role": "assistant",
        "_tldw_continuation_owner": "owner",
        "tool_calls": [
            {
                "id": "call",
                "function": {"name": "code_review_submit", "arguments": "{}"},
            }
        ],
    }
    result = {
        "role": "tool",
        "tool_call_id": "call",
        "content": '{"accepted_ids":[],"submission_revision":1,"cursor":"ack"}',
    }
    pending = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "pending",
                "function": {"name": "code_review_read", "arguments": "{}"},
            }
        ],
    }
    messages = prefix + [call, result, pending]
    checkpoint = ReviewHistoryCheckpoint(
        "attempt", "a" * 64, 1, acknowledged_cursors=("ack",)
    )
    assert (
        project_verifier_history(
            messages, checkpoint, continuation_groups=(owner,), native=True
        )
        == messages
    )
    without_private = project_verifier_history(
        messages, checkpoint, continuation_groups=(), native=True
    )
    assert pending in without_private
    assert call not in without_private
    assert owner.checkpoint.rounds[0].reasoning_blocks == ("private",)


def test_finished_reasoning_owner_is_not_removed_by_checkpoint():
    from Tests.Agents.test_provider_continuation_runtime import _checkpoint
    from tldw_chatbook.Agents.code_review_history import (
        ReviewHistoryCheckpoint,
        project_verifier_history,
    )
    from tldw_chatbook.Chat.provider_continuation import ContinuationOwnerGroup

    private = _checkpoint(state="complete")
    owner = ContinuationOwnerGroup("owner", private, private.rounds)
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "_tldw_continuation_owner": "owner",
            "content": "private owner",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "submit",
                    "function": {"name": "code_review_submit", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "submit",
            "content": '{"accepted_ids":[],"submission_revision":1,"cursor":"ack"}',
        },
    ]
    checkpoint = ReviewHistoryCheckpoint(
        "attempt", "a" * 64, 1, acknowledged_cursors=("ack",)
    )
    assert messages[1] in project_verifier_history(
        messages, checkpoint, continuation_groups=(owner,), native=True
    )
