"""A result is settled only when every canonical allowance descendant settles."""

import pytest

from Tests.Chat.test_goal_conversation_provisioning import service
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture
from tldw_chatbook.Agents.agent_models import RunOutcome, RunTerminationReason
from tldw_chatbook.Agents.goal_models import (
    GoalIterationResult,
    GoalToolObservation,
    IterationReport,
)

stores = _stores_fixture

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def enable_goals(monkeypatch):
    monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", "true")


def accepted_result(stores):
    db, _, _, request = stores
    owner = service(stores)
    goal = owner.create(request, launch_id="parent")
    attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="owner")
    assert db.automatic_work.accept_goal_iteration(attempt.id, owner_id="owner")
    run_id = db.create_run(
        conversation_id=goal.conversation_id,
        agent_kind="primary",
        work_chain_id=goal.chain_id,
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE goal_iterations SET run_id=? WHERE attempt_id=?",
            (run_id, attempt.id),
        )
    report = IterationReport(
        summary="Observed", evidence_ids=("obs",), completion_recommended=True
    )
    observation = GoalToolObservation(
        "obs",
        goal.id,
        attempt.id,
        run_id,
        "fs_read",
        "a" * 64,
        "observed content",
        True,
    )
    result = GoalIterationResult(
        goal.id,
        attempt.ordinal,
        attempt.id,
        run_id,
        RunOutcome(status="done", steps=[], final_text=report.model_dump_json()),
        RunTerminationReason.DONE,
        observations=(observation,),
    )
    return owner, goal, result


def descendant(db, result, state):
    child = db.automatic_work.prepare_chat_start(
        attempt_id="child",
        source_run_id=result.native_run_id,
        target_conversation_id="child-chat",
        target_session_id="child-session",
        target_session_incarnation="incarnation",
        owner_id="owner",
        draft_revision=0,
        context_epoch=0,
        request_fingerprint="a" * 64,
    )
    if state != "prepared":
        assert db.automatic_work.accept_chat_start(child.id, owner_id="owner")
    if state in {"unknown", "settled"}:
        reservation = db.automatic_work.admit_call(
            child.chain_id,
            call_id="call",
            owner_id="owner",
            input_tokens=1,
            output_tokens=10,
        )
        db.automatic_work.settle(
            reservation,
            owner_id="owner",
            actual_amount=None if state == "unknown" else 2,
        )
        db.automatic_work.complete_chat_start(child.id, owner_id="owner")
    return child


@pytest.mark.parametrize("state", ["prepared", "accepted", "unknown"])
async def test_checkpoint_preserves_unsettled_descendant_recovery(stores, state):
    owner, goal, result = accepted_result(stores)
    descendant(stores[0], result, state)
    saved = owner.checkpoint(result)
    assert saved.status == "recovery_required"
    assert (
        stores[0]
        .automatic_work.read_goal_attempt(result.attempt_id, owner_id="owner")
        .state
        == "review_required"
    )
    assert owner.completion_check(goal.id).action == "recovery_required"
    with pytest.raises(ValueError, match="settled"):
        owner.remove_payloads(goal.id)


@pytest.mark.parametrize("state", ["prepared", "accepted", "unknown"])
async def test_late_descendant_work_cannot_be_quality_approved(stores, state):
    owner, goal, result = accepted_result(stores)
    saved = owner.checkpoint(result)
    assert saved.status == "awaiting_result_review"
    descendant(stores[0], result, state)
    assert owner.completion_check(goal.id).action == "recovery_required"
    checkpoint = saved.checkpoints[-1]
    with pytest.raises(ValueError, match="objective_proof_unavailable"):
        owner.review_result(
            goal.id,
            expected_revision=saved.revision,
            checkpoint_id=checkpoint.id,
            artifact_digest=checkpoint.artifact_digest,
            accepted=True,
        )
    assert owner.get(goal.id).status != "completed"


async def test_settled_descendant_does_not_block_quality_review(stores):
    owner, goal, result = accepted_result(stores)
    descendant(stores[0], result, "settled")
    saved = owner.checkpoint(result)
    assert saved.status == "awaiting_result_review"
    checkpoint = saved.checkpoints[-1]
    done = owner.review_result(
        goal.id,
        expected_revision=saved.revision,
        checkpoint_id=checkpoint.id,
        artifact_digest=checkpoint.artifact_digest,
        accepted=True,
    )
    assert done.status == "completed" and not done.accounting.uncertain
