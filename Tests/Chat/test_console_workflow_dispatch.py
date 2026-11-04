"""Accepted planning and approved tasks use actual native Console owners."""

from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

pytestmark = pytest.mark.bootstrap_profile


async def journey(tmp_path, monkeypatch):
    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(hold=("task-1",))
    )
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        first = await rig.coordinator.approve(decision, command_id="approve-1")
        await rig.wait_for_provider_entry("task-1")
        again = await rig.coordinator.approve(decision, command_id="approve-1")
        assert (first.id, first.control_revision) == (again.id, again.control_revision)
        assert [key for key, _ in rig.provider.calls].count("task-1") == 1
        with rig.service.db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM workflow_task_attempts WHERE run_id IS NOT NULL"
                ).fetchone()[0]
                == 1
            )
            assert (
                conn.execute("SELECT count(*) FROM workflow_phase_attempts").fetchone()[
                    0
                ]
                == 2
            )
        owners = rig.runtime.execution_capacity.snapshot().executions
        assert sum(owner.child for owner in owners) == 1
        rig.provider_release()
        await rig.wait_for_status("awaiting_result_review")
    finally:
        await rig.close()


def test_plan_then_duplicate_approval_launches_one_native_task(tmp_path):
    _run(
        tmp_path,
        "dispatch",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def next_phase_journey(tmp_path, monkeypatch):
    import copy

    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    second = copy.deepcopy(rig.provider.plan["phases"][0])
    second["id"] = "phase-2"
    second["tasks"][0].update(id="task-2", instructions="Execute task-2")
    rig.provider.plan["phases"].append(second)
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        await rig.coordinator.approve(decision, command_id="phase-1-approve")
        reviewed = await rig.wait_for_status("awaiting_result_review", timeout=20)
        assert len(reviewed.results) == 1 and reviewed.results[0].status == "completed"
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        quality = await rig.service.call(
            rig.service.prepare_quality_review, rig.workflow.id, phase_id="phase-1"
        )
        next_review = await rig.coordinator.approve(
            quality, command_id="phase-1-quality"
        )
        assert next_review.status == "awaiting_phase_review"
        assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
        # A newly approved phase has its own exact native attempt and one task entry.
        next_decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-2"
        )
        await rig.coordinator.approve(next_decision, command_id="phase-2-approve")
        final = await rig.wait_for_status("awaiting_result_review", timeout=20)
        assert len(final.results) == 2 and all(
            result.status == "completed" for result in final.results
        )
        quality = await rig.service.call(
            rig.service.prepare_quality_review, rig.workflow.id, phase_id="phase-2"
        )
        final = await rig.coordinator.approve(quality, command_id="phase-2-quality")
        assert final.status == "completed" and final.accounting.used["generation"] == 3
        assert [key for key, _ in rig.provider.calls] == [
            "planning",
            "task-1",
            "task-2",
        ]
    finally:
        await rig.close()


def test_native_result_review_quality_acceptance_and_next_phase(tmp_path):
    _run(
        tmp_path,
        "next-phase",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def refusal_and_remaining_journey(tmp_path, monkeypatch):
    import asyncio
    import copy
    import sys

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage

    partial = sys.argv[1] == "partial-resume"
    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(hold=("task-1",) if partial else ())
    )
    if partial:
        second = copy.deepcopy(rig.provider.plan["phases"][0]["tasks"][0])
        second.update(
            id="task-2", instructions="Execute task-2", dependencies=["task-1"]
        )
        rig.provider.plan["phases"][0]["tasks"].append(second)
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await asyncio.shield(rig.coordinator._runs[rig.workflow.id].driver)
        before = await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        if partial:
            await rig.coordinator.approve(decision, command_id="first-partial")
            from tldw_chatbook.Agents.agent_service import WorkflowChildReceipt

            original = rig.coordinator._runs[rig.workflow.id]
            while True:
                prepared = original.preparations.get("task-1")
                launches = (
                    prepared.authorization._native_launch
                    if prepared is not None
                    else ()
                )
                if launches and type(launches[0]) is WorkflowChildReceipt:
                    receipt = launches[0]
                    authorization = prepared.authorization
                    native = authorization._native_preparation[0]
                    assert prepared.phase is original.phase is native.phase
                    assert original.phase._task_scopes["task-1"] is authorization
                    assert authorization._bridge_preparation[0] is prepared
                    assert receipt.attempt_id == authorization.claim.id
                    assert receipt.run_id == authorization.claim.launch_run_id
                    assert receipt.owner.is_admitted
                    break
                assert not original.driver.done(), (
                    "original task-1 launch did not return"
                )
                await asyncio.sleep(0.01)
            await rig.wait_for_provider_entry("task-1")
            running = await rig.coordinator.get(rig.workflow.id)
            assert rig.coordinator._runs[rig.workflow.id] is original
            paused = await rig.coordinator.pause(
                rig.workflow.id, expected_control_revision=running.control_revision
            )
            assert paused.status == "pausing" and not original.driver.done()
            assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
            rig.provider_release()
            await asyncio.shield(original.driver)
            paused = await rig.wait_for_status("paused")
            assert len(paused.results) == 1 and paused.results[0].status == "completed"
            retained = paused.results[0]
            assert (
                not paused.accounting.uncertain
                and rig.runtime.execution_capacity.snapshot().executions == ()
            )
            before = await rig.coordinator.resume(
                rig.workflow.id,
                expected_control_revision=paused.control_revision,
                command_id="partial-resume",
            )
            assert (
                before.status == "awaiting_phase_review"
                and before.plans == paused.plans
            )
            decision = await rig.service.call(
                rig.service.prepare_remaining_decision,
                rig.workflow.id,
                phase_id="phase-1",
            )
            assert (
                decision.reason == "remaining_ready"
                and retained.attempt_id in decision.result_refs
            )
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            counts = tuple(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("workflow_phase_attempts", "automatic_work_reservations")
            )
        stale = decision.model_copy(
            update={"control_revision": decision.control_revision - 1}
        )
        with pytest.raises(ValueError, match="stale_workflow_decision") as refusal:
            await rig.coordinator.approve(stale, command_id="stale-approval")
        command = rig.coordinator._commands[("approve", "stale-approval")][1]
        assert command.exception() is refusal.value
        from tldw_chatbook.Chat.console_workflow_runs import (
            _accept_remaining_workflow_phase,
        )

        traceback = refusal.value.__traceback__
        while (
            traceback.tb_frame.f_code is not _accept_remaining_workflow_phase.__code__
        ):
            traceback = traceback.tb_next
        cell = traceback.tb_frame.f_locals["cell"]
        assert cell.body_error is refusal.value and cell.preparation_entered is False
        assert cell.owner_return == [True] and not cell.owner.is_admitted
        assert (
            cell.read_return[1].exception() is None
            and cell.read_return[2].result() is None
        )
        assert cell.claim.id not in rig.runtime.worktree_activity._claims
        with pytest.raises(ValueError) as duplicate:
            await rig.coordinator.approve(stale, command_id="stale-approval")
        assert duplicate.value is refusal.value
        assert rig.workflow.id not in rig.coordinator._runs
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            assert (
                tuple(
                    conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in (
                        "workflow_phase_attempts",
                        "automatic_work_reservations",
                    )
                )
                == counts
            )
        current = await rig.coordinator.get(rig.workflow.id)
        assert (
            current.accounting == before.accounting
            and current.results == before.results
        )
        await rig.coordinator.approve(decision, command_id="separate-valid-approval")
        complete = await rig.wait_for_status("awaiting_result_review", timeout=20)
        assert all(result.status == "completed" for result in complete.results)
        assert [key for key, _ in rig.provider.calls] == (
            ["planning", "task-1", "task-2"] if partial else ["planning", "task-1"]
        )
        assert complete.accounting.used["generation"] == (3 if partial else 2)
        assert complete.accounting.used["child_launch"] == (2 if partial else 1)
        if partial:
            assert complete.results[0] == retained
            assert complete.plans == before.plans
        quality = await rig.service.call(
            rig.service.prepare_quality_review, rig.workflow.id, phase_id="phase-1"
        )
        assert (
            await rig.coordinator.approve(quality, command_id="quality-complete")
        ).status == "completed"
    finally:
        rig.provider_release()
        await rig.close()


@pytest.mark.parametrize("mode", ["stale-initial", "partial-resume"])
def test_refusal_returns_admission_and_explicit_remaining_preserves_sibling(
    tmp_path, mode
):
    _run(
        tmp_path,
        mode,
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def repair_journey(tmp_path, monkeypatch):
    import asyncio
    import hashlib
    import json

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.LLM_Calls.provider_outcomes import PreEffectPermanentError

    class LocalRefusalProvider(WorkflowProvider):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            if self.calls[-1][0] == "task-1":
                raise PreEffectPermanentError(
                    "trusted local fixture refused before dispatch"
                )
            return result

    rig = await build_workflow_rig(tmp_path, monkeypatch, LocalRefusalProvider())
    instruction = "Repair the failed task while preserving its original evidence."
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        planning = rig.coordinator._runs[rig.workflow.id]
        await asyncio.shield(planning.driver)
        initial = await rig.coordinator.get(rig.workflow.id)
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        await rig.coordinator.approve(decision, command_id="failed-phase")
        execution = rig.coordinator._runs[rig.workflow.id]
        await asyncio.shield(execution.driver)
        failed = await rig.coordinator.get(rig.workflow.id)
        assert failed.status == "paused" and len(failed.results) == 1
        assert failed.results[0].status == "failed" and not failed.accounting.uncertain
        assert (
            failed.accounting.used["generation"] == 2
            and failed.accounting.used["child_launch"] == 1
        )
        assert (
            failed.accounting.used["model_call"] == 2
            and failed.accounting.used["tokens"] == 10
        )
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        with pytest.raises(ValueError, match="remaining_requires_repair"):
            await rig.service.call(
                rig.service.prepare_remaining_decision,
                rig.workflow.id,
                phase_id="phase-1",
            )
        rig.provider.plan["phases"][0]["tasks"][0]["instructions"] = (
            "Repaired task-1 requires new approval"
        )
        await rig.coordinator.revise(
            rig.workflow.id,
            expected_control_revision=failed.control_revision,
            instruction=instruction,
            command_id="repair-explicit",
        )
        repair = rig.coordinator._runs[rig.workflow.id]
        assert repair.prepared._planning_repair_source[3] == instruction
        assert (
            repair.phase.attempt.repair_instruction_sha256
            == hashlib.sha256(instruction.encode()).hexdigest()
        )
        await asyncio.shield(repair.driver)
        reviewed = await rig.coordinator.get(rig.workflow.id)
        assert reviewed.status == "awaiting_phase_review" and len(reviewed.plans) == 2
        assert (
            reviewed.request == initial.request
            and reviewed.plans[0] == initial.plans[0]
        )
        assert reviewed.results == failed.results
        assert (
            reviewed.accounting.used["generation"] == 3
            and reviewed.accounting.used["child_launch"] == 1
        )
        assert (
            reviewed.accounting.used["model_call"] == 3
            and reviewed.accounting.used["tokens"] == 20
        )
        assert not reviewed.accounting.uncertain
        messages = rig.provider.calls[-1][1].get(
            "messages", rig.provider.calls[-1][1].get("messages_payload")
        )
        seed = next(
            row["content"].split("\n", 1)[1]
            for row in messages
            if row.get("content", "").startswith(
                "Produce exactly one strict JSON WorkflowPlan"
            )
        )
        body = json.loads(seed)["workflow_planning"]
        assert body["repair_instruction"] == instruction
        assert body["previous_plans"] == [initial.plans[0].model_dump(mode="json")]
        assert body["repair_evidence_untrusted"][0]["result"] == failed.results[
            0
        ].model_dump(mode="json")
        await rig.coordinator.revise(
            rig.workflow.id,
            expected_control_revision=failed.control_revision,
            instruction=instruction,
            command_id="repair-explicit",
        )
        assert [key for key, _kwargs in rig.provider.calls] == [
            "planning",
            "task-1",
            "planning",
        ]
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            attempts = conn.execute(
                "SELECT status,retirement FROM workflow_task_attempts"
            ).fetchall()
            tokens = conn.execute(
                "SELECT state,actual_amount FROM automatic_work_reservations WHERE kind='tokens' ORDER BY created_at"
            ).fetchall()
            phases = conn.execute(
                "SELECT purpose,retirement FROM workflow_phase_attempts ORDER BY ordinal"
            ).fetchall()
        assert [tuple(row) for row in attempts] == [("failed", "drained")]
        assert [tuple(row) for row in tokens] == [
            ("settled", 10),
            ("settled", 0),
            ("settled", 10),
        ]
        assert [tuple(row) for row in phases] == [
            ("planning", "drained"),
            ("execution", "drained"),
            ("planning", "drained"),
        ]
        assert (
            await rig.coordinator.get(rig.workflow.id)
        ).accounting == reviewed.accounting
        assert rig.runtime.execution_capacity.snapshot().executions == ()
    finally:
        await rig.close()


def test_native_failed_run_requires_explicit_repair_preserving_history_and_spend(
    tmp_path,
):
    _run(
        tmp_path,
        "repair",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )
