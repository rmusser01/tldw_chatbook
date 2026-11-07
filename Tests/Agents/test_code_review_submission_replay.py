"""Accepted-ID conflicts recover only by replaying the unchanged submission."""

import json
import threading
from copy import deepcopy

import pytest

from Tests.Agents import code_review_helpers, test_code_review_conflict_recovery
from Tests.Agents.test_code_review_conflict_recovery import (
    rejected_key,
    stream_for,
    submit,
    verifier_summary,
)

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case
accepted_case = test_code_review_conflict_recovery.accepted_case
pytestmark = pytest.mark.bootstrap_profile


def test_accepted_id_conflict_replay_repairs_only_its_rejection_then_seals(
    accepted_case,
):
    case = accepted_case
    c, prepared, provider, records = case
    accepted = submit(case, records)
    assert accepted.ok
    stream = stream_for(case)
    original = deepcopy(stream.verification.dispositions)
    changed = [
        dict(row, verdict="refuted", reason="PRIVATE-CHANGED") for row in records
    ]
    refused = submit(case, changed, revision=1)
    assert refused.error == "submission_conflict"
    feedback = json.loads(refused.content)
    assert feedback == {
        "submission_id": "accepted",
        "reason": "submission_conflict",
        "recovery": "replay_original_accepted_submission",
        "submission_revision": 1,
        "retry": {
            "review_id": prepared.review_id,
            "kind": "dispositions",
            "records": records,
            "expected_revision": 1,
            "submission_id": "accepted",
        },
    }
    assert "PRIVATE-CHANGED" not in refused.content
    assert submit(case, [], revision=1).error == "submission_conflict"
    assert stream.verification.dispositions == original
    blocked = submit(case, [], revision=1, sid="complete", kind="verifier_complete")
    assert blocked.error == "unresolved_submission_rejections"
    # Historical expected_revision is valid only for the identical accepted payload.
    replay = provider.invoke("code_review:code_review_submit", feedback["retry"])
    assert replay == accepted
    assert stream.revision == 1 and stream.verification.dispositions == original
    assert stream.rejected == {
        rejected_key("complete"): "unresolved_submission_rejections"
    }
    assert not stream.completion_requested
    before = c.store.usage()
    assert submit(case, records) == accepted
    assert c.store.usage() == before
    assert submit(case, [], revision=1, sid="complete", kind="verifier_complete").ok
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
    assert stream.verification.dispositions == original


def test_conflict_feedback_obeys_tool_capacity_without_echoing_changed_input(
    accepted_case,
):
    case = accepted_case
    _, _, provider, records = case
    assert submit(case, records).ok
    provider.service.result_limit = 220
    refused = submit(
        case, [dict(r, reason="PRIVATE-CHANGED") for r in records], revision=1
    )
    assert refused.error == "submission_conflict"
    feedback = json.loads(refused.content)
    assert len(refused.content) <= 220
    assert feedback["recovery"] == "original_replay_exceeds_capacity"
    assert "retry" not in feedback and "PRIVATE-CHANGED" not in refused.content
    assert rejected_key("accepted") in stream_for(case).rejected


@pytest.mark.parametrize("revocation", ["excluded", "permission_denied"])
def test_retry_cannot_redisclose_revoked_source(accepted_case, revocation):
    from Tests.Agents.test_code_review_completion_recovery import invoke

    case = accepted_case
    c, prepared, provider, records = case
    records = [dict(row, reason="SOURCE-CONTENT-CANARY") for row in records]
    assert submit(case, records).ok
    ref = records[0]["evidence"][0]["source_ref"]
    with c.store.lease(prepared.review_id) as context:
        path = c.root / context.source_state.documents[ref].relative_path
    if revocation == "excluded":
        object.__setattr__(c.authority.admitted, "exclusions_provider", lambda: (path,))
    else:
        provider.service.source_permission = lambda *args: "source_permission_denied"
    assert not invoke(
        provider, "read", prepared.review_id, view="source", document_ref=ref
    ).ok
    refused = submit(case, [], revision=1)
    assert not refused.ok and "SOURCE-CONTENT-CANARY" not in refused.content
    assert rejected_key("accepted") in stream_for(case).rejected


@pytest.mark.parametrize("fault", ["ack", "receipt"])
def test_failed_accepted_replay_preserves_rejection_and_decisions(
    accepted_case, monkeypatch, fault
):
    case = accepted_case
    c, _, provider, records = case
    accepted = submit(case, records)
    assert accepted.ok
    stream = stream_for(case)
    original = deepcopy(stream.verification.dispositions)
    assert submit(case, [], revision=1).error == "submission_conflict"
    with monkeypatch.context() as patch:
        if fault == "ack":
            from tldw_chatbook.Agents.code_review_models import ReviewRefused

            def refuse(*args):
                raise ReviewRefused("result_too_large")

            patch.setattr(provider.service, "encode", refuse)
        else:
            patch.setattr(c.store, "_host_receipt", lambda *args: None)
        result = submit(case, records)
    assert not result.ok
    assert rejected_key("accepted") in stream.rejected
    assert stream.revision == 1 and stream.verification.dispositions == original
    assert submit(case, records) == accepted
    assert not stream.rejected


def test_accepted_replay_cannot_complete_missing_units_or_reopen_terminal(
    accepted_case,
):
    case = accepted_case
    _, _, provider, records = case
    accepted = submit(case, records[:1])
    assert accepted.ok
    assert submit(case, [], revision=1).error == "submission_conflict"
    assert submit(case, records[:1]) == accepted
    assert (
        submit(case, [], revision=1, sid="complete", kind="verifier_complete").error
        == "verification_incomplete"
    )
    stream = stream_for(case)
    assert not stream.completion_requested
    provider.service.settle_review_run(
        provider.service.actor.run_id,
        terminal_status="done",
        receipt_id="offline-terminal",
    )
    before = deepcopy(stream.rejected)
    assert submit(case, records[:1]).error == "verification_attempt_closed"
    assert stream.rejected == before
    assert verifier_summary(case)["seal_receipt"] is None


def test_replay_does_not_erase_other_rejection_recovery_from_history(accepted_case):
    from tldw_chatbook.Agents.code_review_history import project_verifier_history

    case = accepted_case
    _, prepared, provider, records = case
    assert submit(case, records).ok
    assert submit(case, [], revision=1).error == "submission_conflict"
    blocked = submit(case, [], revision=1, sid="complete", kind="verifier_complete")
    assert blocked.error == "unresolved_submission_rejections"
    replay = submit(case, records)
    assert replay.ok
    messages = [
        {"role": "assistant", "content": "Complete", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": blocked.content},
        {"role": "assistant", "content": "Replay", "tool_calls": [{"id": "c2"}]},
        {"role": "tool", "tool_call_id": "c2", "content": replay.content},
    ]
    checkpoint = provider.service.verifier_checkpoint(prepared.review_id)
    assert (
        project_verifier_history(
            messages, checkpoint, continuation_groups=(), native=True
        )
        == messages
    )


@pytest.mark.parametrize("change_during_callback", [False, True])
def test_original_retry_permission_is_unlocked_and_revalidated(
    accepted_case, change_during_callback
):
    case = accepted_case
    c, prepared, provider, records = case
    records = [dict(row, reason="SOURCE-CONTENT-CANARY") for row in records]
    assert submit(case, records).ok
    with c.store.lease(prepared.review_id) as context:
        ref = records[0]["evidence"][0]["source_ref"]
        path = c.root / context.source_state.documents[ref].relative_path
    observed = []

    def permission(*args):
        acquired = []

        def probe():
            ok = c.store._tab.lock.acquire(timeout=1)
            acquired.append(ok)
            if ok:
                c.store._tab.lock.release()

        worker = threading.Thread(target=probe)
        worker.start()
        worker.join(2)
        assert acquired == [True]
        observed.append(True)
        if change_during_callback:
            object.__setattr__(
                c.authority.admitted, "exclusions_provider", lambda: (path,)
            )

    provider.service.source_permission = permission
    result = submit(case, [], revision=1)
    assert observed
    if change_during_callback:
        assert result.error == "source_path_unavailable"
        assert "SOURCE-CONTENT-CANARY" not in result.content
    else:
        assert result.error == "submission_conflict"
        assert "SOURCE-CONTENT-CANARY" in result.content
