"""Withdraw rejected verifier resubmissions without changing accepted decisions."""

import hashlib
import json
import os
from copy import deepcopy

import pytest

from Tests.Agents import code_review_helpers
from Tests.Agents.test_agent_service import fence
from Tests.Agents.test_code_review_completion_recovery import invoke
from Tests.Agents.test_code_review_verification import seed
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.code_review_history import project_verifier_history
from tldw_chatbook.Agents.run_context import use_run_actor

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case
pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def accepted_case(review_case):
    c = review_case
    prepared = seed(c, 3)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        provider.service.verifier_input_tokens = 10000
        provider.service.verifier_estimate = lambda text: len(text) // 4
        page = invoke(provider, "read", prepared.review_id, view="batch")
        assert page.ok
        records = [
            {
                "schema_version": 1,
                "verification_unit_id": row["verification_unit_id"],
                "verdict": "confirmed",
                "reason": "captured evidence",
                "evidence": row["claim"]["evidence"],
            }
            for row in json.loads(page.content)["records"]
        ]
        assert len(records) == 3
        yield c, prepared, provider, records


def submit(case, records, *, revision=0, sid="accepted", kind="dispositions"):
    _, prepared, provider, _ = case
    return invoke(
        provider,
        "submit",
        prepared.review_id,
        kind=kind,
        records=records,
        expected_revision=revision,
        submission_id=sid,
    )


def stream_for(case):
    c, prepared, _, _ = case
    with c.store.lease(prepared.review_id) as context:
        return context.streams[("V1", "attempt")]


def rejected_key(sid):
    return hashlib.sha256(sid.encode()).hexdigest()


def verifier_summary(case):
    c, prepared, _, _ = case
    with use_run_actor(c.actor):
        return next(
            slot
            for row in c.primary.read(prepared.review_id, "summary").records
            for slot in row["slots"]
            if slot["slot_id"] == "V1"
        )


@pytest.mark.parametrize("native", [True, False])
def test_conflict_feedback_recovers_real_loop_then_seals(accepted_case, native):
    case = accepted_case
    c, prepared, provider, records = case
    accepted = submit(case, records)
    assert accepted.ok
    stream = stream_for(case)
    original = deepcopy(stream.verification.dispositions)
    requests, results, reviewed = [], [], []
    recovery = None

    def model(messages, active):
        nonlocal recovery
        step = len(requests)
        messages = project_verifier_history(
            messages,
            provider.service.verifier_checkpoint(prepared.review_id),
            continuation_groups=(),
            native=native,
        )
        requests.append(deepcopy(messages))
        if step == 0:
            args = {
                "kind": "dispositions",
                "records": [
                    dict(r, verdict="refuted", reason="PRIVATE-REJECTED-REASON")
                    for r in records
                ],
                "expected_revision": 1,
                "submission_id": "conflict",
            }
        elif step == 1:
            assert results[-1].error == "disposition_conflict"
            # This is the actual next model input, not a test-created feedback row.
            assert any(
                results[-1].content in row.get("content", "") for row in messages
            )
            recovery = json.loads(results[-1].content)
            assert recovery["recovery"] == "withdraw_conflicting_submission"
            assert "PRIVATE-REJECTED-REASON" not in results[-1].content
            assert all(
                r["verification_unit_id"] not in results[-1].content for r in records
            )
            args = {
                "kind": "verifier_complete",
                "records": [],
                "expected_revision": 1,
                "submission_id": "complete",
            }
        elif step == 2:
            assert results[-1].error == "unresolved_submission_rejections"
            args = recovery["retry"]
            assert args == {
                "kind": "dispositions",
                "records": [],
                "expected_revision": 1,
                "submission_id": "conflict",
            }
        elif step == 3:
            assert results[-1].ok
            ack = json.loads(results[-1].content)
            assert ack["accepted_ids"] == [] and ack["submission_revision"] == 2
            # Withdrawing a conflict cannot evict the still-rejected completion ID.
            assert any(
                '"submission_id":"complete"' in row.get("content", "")
                for row in messages
            )
            assert stream.rejected == {
                rejected_key("complete"): "unresolved_submission_rejections"
            }
            assert stream.verification.dispositions == original
            args = {
                "kind": "verifier_complete",
                "records": [],
                "expected_revision": ack["submission_revision"],
                "submission_id": "complete",
            }
        else:
            assert step == 4 and results[-1].ok
            return ModelTurn(text="Verification complete")
        args = {"review_id": prepared.review_id, **args}
        call = ToolCall("code_review_submit", args, f"call-{step}")
        if not native:
            return ModelTurn(text=fence(call.name, args))
        return ModelTurn(
            text="",
            tool_calls=(call,),
            assistant_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(args)},
                    }
                ],
            },
        )

    def dispatch(call):
        result = provider.invoke("code_review:" + call.name, call.args)
        results.append(result)
        return result

    outcome = run_agent_loop(
        AgentConfig(
            model="offline",
            system_prompt="Verify",
            allowed_tools=("code_review_submit",),
            native_tools=native,
            budget=RunBudget(max_steps=24),
        ),
        [{"role": "user", "content": "Complete this verification"}],
        [provider.load_schema("code_review:code_review_submit")],
        LoopDeps(
            call_model=model,
            invoke_tool=dispatch,
            spawn=lambda *args: pytest.fail("No child launches"),
            find_tools=lambda query: [],
            load_schemas=lambda *args: [],
            should_cancel=lambda: False,
            clock=lambda: 0.0,
            review_tool_calls=lambda calls: reviewed.extend(calls) or {},
        ),
    )
    assert outcome.status == RUN_DONE
    assert len(reviewed) == len(results) == 4
    assert stream.revision == 3 and stream.completion_requested and not stream.rejected
    assert stream.verification.dispositions == original
    before = c.store.usage()
    # Replays return their original acknowledgements without advancing or charging.
    assert submit(case, records).content == accepted.content
    assert submit(case, [], revision=1, sid="conflict").content == results[2].content
    assert c.store.usage() == before and stream.revision == 3
    assert verifier_summary(case)["seal_receipt"] is None
    provider.service.capture_review_output(
        provider.service.actor.run_id, "offline verification", terminal=True
    )
    assert verifier_summary(case)["seal_receipt"] is None
    provider.service.settle_review_run(
        provider.service.actor.run_id,
        terminal_status="done",
        receipt_id="offline-terminal",
    )
    assert verifier_summary(case)["report_status"] == "sealed"
    with use_run_actor(c.actor):
        expanded = c.primary.expand_dispositions(prepared.review_id)
    assert len(expanded) == len({r["original_id"] for r in expanded}) == 3
    assert all(r["disposition"]["verdict"] == "confirmed" for r in expanded)


@pytest.mark.parametrize("fault", ["stale", "publish", "ack"])
def test_failed_withdrawal_retains_conflict_and_is_retryable(
    accepted_case, monkeypatch, fault
):
    case = accepted_case
    c, prepared, provider, records = case
    assert submit(case, records).ok
    assert (
        submit(case, records, revision=1, sid="conflict").error
        == "disposition_conflict"
    )
    stream = stream_for(case)
    original = deepcopy(stream.verification.dispositions)
    before = c.store.usage()
    with c.store.lease(prepared.review_id) as context:
        manifest = context.manifest_path.read_bytes()
    with monkeypatch.context() as patch:
        if fault == "publish":

            def fail(*args):
                raise OSError("offline disk failure")

            patch.setattr(os, "replace", fail)
        elif fault == "ack":
            from tldw_chatbook.Agents.code_review_models import ReviewRefused

            def refuse(*args):
                raise ReviewRefused("result_too_large")

            patch.setattr(provider.service, "encode", refuse)
        result = submit(case, [], revision=0 if fault == "stale" else 1, sid="conflict")
    assert not result.ok
    assert (
        result.error
        == {
            "stale": "submission_revision_conflict",
            "publish": "publication_failed",
            "ack": "result_too_large",
        }[fault]
    )
    assert stream.revision == 1 and stream.verification.dispositions == original
    assert stream.rejected == {rejected_key("conflict"): "disposition_conflict"}
    assert context.manifest_path.read_bytes() == manifest
    assert c.store.usage() == before
    repaired = submit(case, [], revision=1, sid="conflict")
    assert repaired.ok and not stream.rejected and not stream.completion_requested
    assert stream.verification.dispositions == original


@pytest.mark.parametrize("sid", ["fresh", "accepted", "other-rejection"])
def test_empty_dispositions_cannot_clear_other_ids(accepted_case, sid):
    case = accepted_case
    _, _, _, records = case
    assert submit(case, records).ok
    assert (
        submit(case, records, revision=1, sid="conflict").error
        == "disposition_conflict"
    )
    if sid == "other-rejection":
        assert not submit(case, [{}], revision=1, sid=sid).ok
    stream = stream_for(case)
    original = deepcopy(stream.verification.dispositions)
    result = submit(case, [], revision=1, sid=sid)
    assert not result.ok
    assert result.error == (
        "submission_conflict" if sid == "accepted" else "invalid_record_count"
    )
    assert rejected_key("conflict") in stream.rejected
    assert stream.revision == 1 and stream.verification.dispositions == original
    assert not submit(case, [], revision=1, sid="complete", kind="verifier_complete").ok


def test_withdrawal_of_mixed_conflict_does_not_dispose_missing_units(accepted_case):
    case = accepted_case
    _, _, _, records = case
    assert submit(case, records[:1]).ok
    assert (
        submit(case, records, revision=1, sid="conflict").error
        == "disposition_conflict"
    )
    assert submit(case, [], revision=1, sid="conflict").ok
    stream = stream_for(case)
    assert len(stream.verification.dispositions) == 1 and not stream.rejected
    result = submit(case, [], revision=2, sid="complete", kind="verifier_complete")
    assert result.error == "verification_incomplete"
    assert not stream.completion_requested
    assert submit(case, records[1:], revision=2, sid="remaining").ok
    assert submit(case, [], revision=3, sid="complete", kind="verifier_complete").ok


def test_closed_attempt_cannot_withdraw(accepted_case):
    case = accepted_case
    _, _, provider, records = case
    assert submit(case, records).ok
    assert (
        submit(case, records, revision=1, sid="conflict").error
        == "disposition_conflict"
    )
    stream = stream_for(case)
    provider.service.settle_review_run(
        provider.service.actor.run_id,
        terminal_status="done",
        receipt_id="offline-terminal",
    )
    before = deepcopy(stream.rejected)
    assert (
        submit(case, [], revision=1, sid="conflict").error
        == "verification_attempt_closed"
    )
    assert stream.rejected == before and stream.revision == 1
    assert verifier_summary(case)["seal_receipt"] is None


@pytest.mark.parametrize("role", ["primary", "reviewer"])
def test_other_roles_cannot_withdraw_verifier_rejection(accepted_case, role):
    from contextlib import nullcontext

    case = accepted_case
    c, prepared, _, records = case
    assert submit(case, records).ok
    assert (
        submit(case, records, revision=1, sid="conflict").error
        == "disposition_conflict"
    )
    stream = stream_for(case)
    original = deepcopy(stream.verification.dispositions)
    participant = (
        nullcontext(c.provider) if role == "primary" else c.participant(prepared)
    )
    with participant as provider, use_run_actor(provider.service.actor):
        result = invoke(
            provider,
            "submit",
            prepared.review_id,
            kind="dispositions",
            records=[],
            expected_revision=1,
            submission_id="conflict",
        )
    assert not result.ok
    assert stream.rejected == {rejected_key("conflict"): "disposition_conflict"}
    assert stream.verification.dispositions == original and stream.revision == 1
