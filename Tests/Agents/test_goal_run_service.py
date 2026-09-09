"""Service controls operate on fresh exact evidence, independently of execution budget."""

import pytest

from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture

stores = _stores_fixture
from Tests.Agents.test_goal_progress import run_verifier


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["none", "artifact", "identity", "failed"])
async def test_quality_review_requires_latest_objective_proof(
    stores, monkeypatch, tmp_path, change
):
    co, goal, result, fixture, *_ = await run_verifier(
        stores,
        monkeypatch,
        tmp_path,
        human=True,
        mode="fail" if change == "failed" else "pass",
        completion=True,
    )
    saved = co.service.checkpoint(result)
    cp = saved.checkpoints[-1]
    if change == "artifact":
        fixture.write_text("new artifact")
    if change == "none":
        # Reviewing a settled result must not call execution admission (nor need a free model slot).
        monkeypatch.setattr(
            co.ledger,
            "check_active",
            lambda *a, **kw: pytest.fail("review bought execution"),
        )
        done = co.service.review_result(
            goal.id,
            expected_revision=saved.revision,
            checkpoint_id=cp.id,
            artifact_digest=cp.artifact_digest,
            accepted=True,
        )
        assert done.status == "completed"
        assert done.accounting.used == saved.accounting.used
        assert done.accounting.deadline_at == saved.accounting.deadline_at
        with pytest.raises(ValueError):
            co.service.review_result(
                goal.id,
                expected_revision=done.revision,
                checkpoint_id=cp.id,
                artifact_digest=cp.artifact_digest,
                accepted=False,
            )
    else:
        with pytest.raises(
            ValueError,
            match={
                "artifact": "objective_proof_unavailable",
                "identity": "stale_result_review",
                "failed": "result_review_unavailable",
            }[change],
        ):
            co.service.review_result(
                goal.id,
                expected_revision=saved.revision,
                checkpoint_id=cp.id,
                artifact_digest="0" * 64
                if change == "identity"
                else cp.artifact_digest,
                accepted=True,
            )
        assert co.service.get(goal.id).status != "completed"


@pytest.mark.asyncio
async def test_last_allowance_review_survives_restart_without_another_increment(
    stores, monkeypatch, tmp_path
):
    runs, persistence, registry, req = stores
    req = req.model_copy(
        update={"policy": req.policy.model_copy(update={"iterations": 1})}
    )
    co, goal, result, *_ = await run_verifier(
        (runs, persistence, registry, req), monkeypatch, tmp_path, human=True
    )
    saved = co.service.checkpoint(result)
    assert saved.status == "awaiting_result_review"
    assert saved.accounting.available["generation"] == 0
    co.ledger.recover(current_owner_id="replacement")
    co.service.project_recovery(co.ledger.recovery_result)
    after = co.service.get(goal.id)
    assert after.status == "awaiting_result_review", (
        "restart must not demand another increment for quality review"
    )
    cp = after.checkpoints[-1]
    done = co.service.review_result(
        goal.id,
        expected_revision=after.revision,
        checkpoint_id=cp.id,
        artifact_digest=cp.artifact_digest,
        accepted=True,
    )
    assert done.status == "completed" and done.accounting.used == saved.accounting.used


@pytest.mark.asyncio
async def test_capacity_retry_cannot_outlive_original_deadline(stores, monkeypatch):
    from Tests.Chat.test_console_goal_dispatch import build_goal_rig
    from Tests.Chat.test_console_goal_scheduling import progress

    goal, _, _, _, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, lambda **kw: progress(1)
    )
    try:
        result = await co.dispatch_once(goal.id)
        saved = co.service.checkpoint(result)
        monkeypatch.setattr(
            co.ledger, "_wall_clock", lambda: saved.accounting.deadline_at + 1
        )
        deferred = co.service.defer(goal.id, reason="primary_capacity", delay=1)
        assert deferred.status == "paused" and deferred.pause_reason == "wall_budget"
        resumed = co.service.resume(goal.id, expected_revision=deferred.revision)
        assert (
            resumed.status == "paused"
            and resumed.accounting.deadline_at == saved.accounting.deadline_at
        )
        assert len(calls) == 1
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_stopped_settled_payload_removal_retains_readable_history(
    stores, monkeypatch
):
    from Tests.Chat.test_console_goal_dispatch import build_goal_rig
    from Tests.Chat.test_console_goal_scheduling import progress

    goal, _, _, _, co, gateway, _ = build_goal_rig(
        stores, monkeypatch, lambda **kw: progress(1)
    )
    try:
        saved = co.service.checkpoint(await co.dispatch_once(goal.id))
        stopped = co.service.stop(goal.id)
        assert stopped.status == "stopped"
        removed = co.service.remove_payloads(goal.id)
        assert removed.status == "removed" and removed.request is None
        assert removed.accounting.used == saved.accounting.used
        entries = co.service.list_goals(limit=1)
        assert len(entries) == 1 and entries[0].id == goal.id
        assert entries[0].status == "removed"
        assert not hasattr(entries[0], "request")
        with pytest.raises(ValueError):
            co.service.resume(goal.id, expected_revision=removed.revision)
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_public_recovery_observation_uses_same_audit_without_dispatch(
    stores, monkeypatch
):
    from Tests.Chat.test_console_goal_dispatch import build_goal_rig

    goal, _, _, controller, co, gateway, calls = build_goal_rig(stores, monkeypatch)
    try:
        controller.fleet_wake.start_recovery()
        assert hasattr(co, "recover"), "public recovery observation missing"
        first = await co.recover()
        assert await co.recover() == first
        assert co.active_goal_id is None and not calls
        assert co.service.list_goals(limit=1)[0].id == goal.id
    finally:
        await gateway.aclose()
