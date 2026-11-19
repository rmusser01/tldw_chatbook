"""Unknown planning charges do not impersonate physically live native work."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio, os, sys
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install(); real_profile_guard.install()
os.environ.update(TLDW_AGENTS_WORKFLOW_RUNS_ENABLED="true", TLDW_AGENTS_AUTOWAKE_ENABLED="false", OPENAI_API_KEY="private-workflow-test")
from Tests.Chat.test_workflow_recovery_cancellation import cancelled_planner_journey
with user_fixture_default_owner():
    asyncio.run(cancelled_planner_journey(Path(os.environ["XDG_DATA_HOME"]).parent, mode=sys.argv[2]))
"""


@pytest.mark.parametrize("mode", ["native", "repeated-cancel", "invalid-clock"])
def test_review_can_close_cancelled_planner_after_actual_physical_return(
    tmp_path, mode
):
    _run(tmp_path, "cancelled-planner", mode, script=_SCRIPT, timeout=55)


async def cancelled_planner_journey(tmp_path, *, mode="native"):
    import asyncio
    import os

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeFootprint,
        WorktreeScope,
    )

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider(hold=("planning",)))
    complete = False
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_provider_entry()
        run = rig.coordinator._runs[rig.workflow.id]
        prepared = run.prepared
        await rig.coordinator.stop(
            rig.workflow.id, expected_control_revision=rig.workflow.control_revision
        )
        if mode == "repeated-cancel":
            for _ in range(3):
                run.driver.cancel()
                await asyncio.sleep(0)
                assert not run.driver.done() and prepared.primary_owner.is_admitted
        elif mode == "invalid-clock":
            rig.service.db.automatic_work._monotonic_clock = lambda: -1.0
        rig.provider_release()
        snapshot = await rig.wait_for_status("recovery_required")
        await asyncio.gather(run.driver, return_exceptions=True)
        assert (
            snapshot.accounting.uncertain
            and snapshot.accounting.used["generation"] == 1
        )
        read, producer, retired = prepared._planning_read_return
        assert producer.done() and retired.done() and retired.result() is None
        assert read.callback_returned or producer.exception() is not None
        assert prepared.turn_lifeline._cleanup_returned
        assert prepared.turn_lifeline._driver_returned
        assert not prepared.turn_lifeline._thread.is_alive()
        assert not any(worker.is_alive() for worker in prepared._tool_workers)
        assert not prepared.primary_owner.is_admitted, (
            "unknown charges retained a physically returned planning owner"
        )
        scratch = tmp_path / "unrelated-scratch"
        scratch.mkdir()
        footprint = WorktreeFootprint(
            writes=(WorktreeScope(scratch, True, capture_directory_chain(scratch)),)
        )
        with rig.runtime.worktree_activity.prelaunch(footprint=footprint) as claim:
            assert claim is not None, "historical uncertainty blocked unrelated scratch"
        closed = await rig.coordinator.reconcile(
            snapshot.id,
            expected_control_revision=snapshot.control_revision,
            decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
            command_id="review-cancelled-planner",
        )
        assert closed.status == "stopped" and closed.accounting == snapshot.accounting
        assert not rig.runtime.worktree_activity._recovery
        assert not rig.runtime.execution_capacity.snapshot().executions
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            before = conn.execute(
                "SELECT result_bytes FROM workflow_phase_attempts WHERE workflow_id=?",
                (snapshot.id,),
            ).fetchone()[0]
        assert before > 0
        removed = await rig.coordinator.remove_results(
            closed.id,
            expected_control_revision=closed.control_revision,
            command_id="remove-cancelled-planner-results",
        )
        assert (
            removed.accounting == snapshot.accounting
            and removed.plans == snapshot.plans
        )
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT result_bytes FROM workflow_phase_attempts WHERE workflow_id=?",
                    (snapshot.id,),
                ).fetchone()[0]
                == 0
            )
        complete = True
    finally:
        rig.provider_release()
        if complete:
            await rig.close()
        else:
            # A failing custody control exits its private process without claiming drain.
            import traceback

            traceback.print_exc()
            print("CANCELLED_PLANNER_RECOVERY_UNFINISHED", flush=True)
            os._exit(17)


@pytest.mark.parametrize("failure", ["clock", "sql", "after-commit"])
def test_accounting_refusal_returns_planner_and_preserves_known_usage(
    tmp_path, failure
):
    script = _SCRIPT.replace(
        "from Tests.Chat.test_workflow_recovery_cancellation import cancelled_planner_journey",
        "from Tests.Chat.test_workflow_recovery_cancellation import failed_accounting_journey as cancelled_planner_journey",
    )
    _run(tmp_path, "failed-accounting", failure, script=script, timeout=55)


async def failed_accounting_journey(tmp_path, *, mode):
    import asyncio
    import os

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    complete = False
    try:
        ledger = rig.service.db.automatic_work
        original = ledger.settle_workflow_phase
        if mode == "sql":
            with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
                conn.execute(
                    "CREATE TRIGGER refuse_planning_completion BEFORE UPDATE OF state ON automatic_wake_attempts WHEN NEW.state='completed' AND OLD.attempt_kind='workflow_phase' BEGIN SELECT RAISE(ABORT,'planning completion refused'); END"
                )
        else:

            def settlement(*args, **kwargs):
                if mode == "clock":
                    before = ledger._wall_clock
                    ledger._wall_clock = lambda: before() - 3600
                result = original(*args, **kwargs)
                if mode == "after-commit":
                    raise RuntimeError("planning completion delivery refused")
                return result

            ledger.settle_workflow_phase = settlement
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        run = rig.coordinator._runs[rig.workflow.id]
        view = await rig.wait_for_status("recovery_required", timeout=30)
        await asyncio.gather(run.driver, return_exceptions=True)
        prepared = run.prepared
        read, producer, retired = prepared._planning_read_return
        assert read.source is prepared and read.task is run.driver
        assert producer.done() and producer.exception() is not None
        assert retired.done() and retired.result() is None
        assert (
            prepared.turn_lifeline._cleanup_returned
            and prepared.turn_lifeline._driver_returned
        )
        assert not prepared.turn_lifeline._thread.is_alive()
        assert not any(worker.is_alive() for worker in prepared._tool_workers)
        assert not prepared.primary_owner.is_admitted, (
            "accounting failure retained a physically returned owner"
        )
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            tokens = [
                tuple(row)
                for row in conn.execute(
                    "SELECT state,actual_amount FROM automatic_work_reservations WHERE kind='tokens'"
                )
            ]
            native = conn.execute(
                "SELECT a.state,p.status,p.retirement FROM automatic_wake_attempts a "
                "JOIN workflow_phase_attempts p ON p.native_attempt_id=a.id WHERE a.id=?",
                (prepared.phase.attempt.id,),
            ).fetchone()
        assert tokens == [("settled", 10)]
        assert tuple(native) == (
            ("completed", "completed", "drained")
            if mode == "after-commit"
            else ("review_required", "uncertain", "unknown")
        )
        assert bool(view.plans) is (mode == "after-commit")
        assert view.accounting.uncertain is (mode != "after-commit")
        assert (
            view.accounting.used["generation"]
            == view.accounting.used["model_call"]
            == 1
        )
        assert not rig.runtime.execution_capacity.snapshot().executions
        closed = await rig.coordinator.reconcile(
            view.id,
            expected_control_revision=view.control_revision,
            decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
            command_id="review-failed-accounting",
        )
        assert closed.status == "stopped" and closed.accounting == view.accounting
        assert not rig.runtime.worktree_activity._recovery
        complete = True
    finally:
        if complete:
            await rig.close()
        else:
            import traceback

            traceback.print_exc()
            print("FAILED_ACCOUNTING_OWNER_RETAINED", flush=True)
            os._exit(17)


@pytest.mark.parametrize(
    "mode",
    [
        "journal",
        "journal-shutdown",
        "recording-delivery",
        "unreviewed-shutdown",
        "planning-root",
        "planning-root-symlink",
    ],
)
def test_returned_planning_survives_journal_failure_shutdown_and_root_replacement(
    tmp_path, mode
):
    script = _SCRIPT.replace(
        "from Tests.Chat.test_workflow_recovery_cancellation import cancelled_planner_journey",
        "from Tests.Chat.test_workflow_recovery_cancellation import recovery_tail_journey as cancelled_planner_journey",
    )
    _run(tmp_path, "recovery-tail", mode, script=script, timeout=65)


async def recovery_tail_journey(tmp_path, *, mode):
    import asyncio
    import os
    import sqlite3

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeFootprint,
        WorktreeScope,
    )

    journal = mode.startswith("journal")
    rig = await build_workflow_rig(
        tmp_path, None, WorkflowProvider(hold=() if journal else ("planning",))
    )
    complete = closed = False
    try:
        service, db = rig.service, rig.service.db
        if mode == "recording-delivery":
            from types import MethodType

            original = service.record_returned_planning
            delivered = False

            def record_returned_planning(self, *args, **kwargs):
                nonlocal delivered
                result = original(*args, **kwargs)
                if not delivered:
                    delivered = True
                    raise RuntimeError("recovery recording delivery failed")
                return result

            service.record_returned_planning = MethodType(
                record_returned_planning, service
            )
        if journal:
            with owned_goal_storage(service), db.connection() as conn:
                conn.execute(
                    "CREATE TRIGGER refuse_planning_accounting BEFORE UPDATE OF state ON automatic_wake_attempts WHEN NEW.state IN ('completed','review_required') AND OLD.attempt_kind='workflow_phase' BEGIN SELECT RAISE(ABORT,'temporary accounting storage failure'); END"
                )
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        run = rig.coordinator._runs[rig.workflow.id]
        if not journal:
            await rig.wait_for_provider_entry()
            await rig.coordinator.stop(
                rig.workflow.id, expected_control_revision=rig.workflow.control_revision
            )
            rig.provider_release()
        view = await rig.wait_for_status("recovery_required", timeout=30)
        await asyncio.gather(run.driver, return_exceptions=True)
        prepared = run.prepared
        assert (
            prepared.turn_lifeline._cleanup_returned
            and prepared.turn_lifeline._driver_returned
        )
        assert not prepared.turn_lifeline._thread.is_alive()
        assert not any(worker.is_alive() for worker in prepared._tool_workers)
        assert not prepared.primary_owner.is_admitted, (
            "recovery journal retained a physically returned owner"
        )
        assert not rig.runtime.execution_capacity.snapshot().executions
        assert rig.store.received_turn_for_session(run.session_id) is None
        assert prepared._planning_intake is prepared._planning_token is None
        scratch = tmp_path / "unrelated-scratch"
        scratch.mkdir()
        scratch_fp = WorktreeFootprint(
            writes=(WorktreeScope(scratch, True, capture_directory_chain(scratch)),)
        )
        with rig.runtime.worktree_activity.prelaunch(footprint=scratch_fp) as claim:
            assert claim is not None, "nonlive accounting blocked unrelated scratch"
        if mode.startswith("planning-root"):
            assert not view.plans
            root = tmp_path / "project"
            root.rename(tmp_path / "old-project")
            if mode.endswith("symlink"):
                target = tmp_path / "replacement"
                target.mkdir()
                root.symlink_to(target, target_is_directory=True)
            else:
                root.mkdir()
            with owned_goal_storage(service):
                db.automatic_work.recover(current_owner_id="fresh-history-owner")
            assert (
                await service.call(
                    service.recovery_protections, db.automatic_work.recovery_result
                )
                == ()
            )
            chain = capture_directory_chain(root)
            replacement_fp = WorktreeFootprint(
                writes=(WorktreeScope(chain.canonical_root, True, chain),)
            )
            with rig.runtime.worktree_activity.prelaunch(
                footprint=replacement_fp
            ) as claim:
                assert claim is not None, (
                    "old planning history fenced a replacement root"
                )
        elif mode == "recording-delivery":
            reviewed = await rig.coordinator.reconcile(
                view.id,
                expected_control_revision=view.control_revision,
                decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
                command_id="retry-delivered-recording",
            )
            assert (
                reviewed.status == "stopped" and reviewed.accounting == view.accounting
            )
            assert reviewed.control_revision == view.control_revision + 1
            assert not rig.runtime.worktree_activity._recovery
        elif journal:
            root = tmp_path / "project"
            root_fp = WorktreeFootprint(
                writes=(WorktreeScope(root, True, capture_directory_chain(root)),)
            )
            with rig.runtime.worktree_activity.prelaunch(footprint=root_fp) as claim:
                assert claim is None, (
                    "pending recovery lost its captured historical protection"
                )
            if mode == "journal":
                with pytest.raises(
                    sqlite3.IntegrityError, match="temporary accounting storage failure"
                ):
                    await rig.coordinator.reconcile(
                        view.id,
                        expected_control_revision=view.control_revision,
                        decision=WorkflowExecutionRecovery(
                            reason="reviewed_prior_effects"
                        ),
                        command_id="storage-still-failed",
                    )
                with owned_goal_storage(service), db.connection() as conn:
                    conn.execute("DROP TRIGGER refuse_planning_accounting")
                changed = await service.call(
                    service.set_status,
                    view.id,
                    status="recovery_required",
                    expected_control_revision=view.control_revision,
                )
                with pytest.raises(ValueError, match="revision_conflict"):
                    await rig.coordinator.reconcile(
                        view.id,
                        expected_control_revision=view.control_revision,
                        decision=WorkflowExecutionRecovery(
                            reason="reviewed_prior_effects"
                        ),
                        command_id="stale-storage-review",
                    )
                reviewed = await rig.coordinator.reconcile(
                    view.id,
                    expected_control_revision=changed.control_revision,
                    decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
                    command_id="storage-repaired",
                )
                assert reviewed.status == "stopped" and reviewed.accounting.uncertain
                assert (
                    reviewed.accounting.used["generation"]
                    == reviewed.accounting.used["model_call"]
                    == 1
                )
                with owned_goal_storage(service), db.connection() as conn:
                    assert [
                        tuple(row)
                        for row in conn.execute(
                            "SELECT state,actual_amount FROM automatic_work_reservations WHERE kind='tokens'"
                        )
                    ] == [("settled", 10)]
                assert not rig.runtime.worktree_activity._recovery
        if mode.endswith("shutdown"):
            with owned_goal_storage(service), db.connection() as conn:
                database_path = conn.execute("PRAGMA database_list").fetchone()[2]
            await rig.close()
            closed = True
            assert rig.runtime._disposed and not rig.coordinator._runs
            with sqlite3.connect(database_path) as saved:
                status, marker = saved.execute(
                    "SELECT status,recovery_reviewed_revision FROM workflow_runs WHERE id=?",
                    (view.id,),
                ).fetchone()
                assert marker == 0 and status != "closed", (
                    "shutdown silently reviewed historical effects"
                )
                state = saved.execute(
                    "SELECT state FROM automatic_wake_attempts WHERE id=?",
                    (prepared.phase.attempt.id,),
                ).fetchone()[0]
                assert state == ("accepted" if journal else "review_required")
        complete = True
    finally:
        rig.provider_release()
        if complete:
            if not closed:
                await rig.close()
        else:
            import traceback

            traceback.print_exc()
            print("RECOVERY_TAIL_NOT_RETURNED", flush=True)
            os._exit(17)


@pytest.mark.parametrize(
    "mode", ["same-running", "changed-running", "same-stopping", "changed-stopping"]
)
def test_session_prompt_save_does_not_invalidate_admitted_planning_return(
    tmp_path, mode
):
    script = _SCRIPT.replace(
        "from Tests.Chat.test_workflow_recovery_cancellation import cancelled_planner_journey",
        "from Tests.Chat.test_workflow_recovery_cancellation import prompt_return_journey as cancelled_planner_journey",
    )
    _run(tmp_path, "prompt-return", mode, script=script, timeout=65)


async def prompt_return_journey(tmp_path, *, mode):
    import asyncio
    import json
    import os

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider(hold=("planning",)))
    complete = False
    admitted = "Use this admitted planning instruction."
    future = "Use this instruction for the next turn."
    try:
        session_id = rig.store.active_session_id
        assert session_id is not None
        _, persisted = rig.store.set_session_system_prompt(session_id, admitted)
        assert persisted
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        run = rig.coordinator._runs[rig.workflow.id]
        await rig.wait_for_provider_entry()
        if mode.endswith("stopping"):
            await rig.coordinator.stop(
                rig.workflow.id, expected_control_revision=rig.workflow.control_revision
            )
        selected = future if mode.startswith("changed") else admitted
        _, persisted = rig.store.set_session_system_prompt(run.session_id, selected)
        assert persisted
        rig.provider_release()
        await asyncio.wait_for(asyncio.shield(run.driver), 30)
        prepared = run.prepared
        assert (
            prepared.turn_lifeline._cleanup_returned
            and prepared.turn_lifeline._driver_returned
        )
        assert not prepared.turn_lifeline._thread.is_alive()
        assert not prepared.primary_owner.is_admitted
        assert not rig.runtime.execution_capacity.snapshot().executions
        assert rig.store.received_turn_for_session(run.session_id) is None
        assert rig.store.session_settings(run.session_id).system_prompt == selected
        assert len(rig.provider.calls) == 1
        payload = json.dumps(rig.provider.calls[0][1], default=str)
        assert admitted in payload and future not in payload
        view = await rig.coordinator.get(rig.workflow.id)
        assert view.status == (
            "recovery_required"
            if mode.endswith("stopping")
            else "awaiting_phase_review"
        )
        complete = True
    finally:
        rig.provider_release()
        if complete:
            await rig.close()
        else:
            import traceback

            traceback.print_exc()
            print("PROMPT_RETURN_NOT_PROVEN", flush=True)
            os._exit(17)


@pytest.mark.parametrize("mode", ["known", "unknown"])
def test_closing_review_delivery_retry_preserves_reviewed_terminal_history(
    tmp_path, mode
):
    script = _SCRIPT.replace(
        "from Tests.Chat.test_workflow_recovery_cancellation import cancelled_planner_journey",
        "from Tests.Chat.test_workflow_recovery_cancellation import closing_retry_journey as cancelled_planner_journey",
    )
    _run(tmp_path, "closing-retry", mode, script=script, timeout=65)


async def closing_retry_journey(tmp_path, *, mode):
    import asyncio
    import os
    from types import MethodType

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery

    known = mode == "known"
    rig = await build_workflow_rig(
        tmp_path, None, WorkflowProvider(hold=() if known else ("planning",))
    )
    complete = False
    try:
        service, ledger = rig.service, rig.service.db.automatic_work
        if known:
            original_settlement = ledger.settle_workflow_phase

            def lost_planning_delivery(*args, **kwargs):
                original_settlement(*args, **kwargs)
                raise RuntimeError("committed planning delivery lost")

            ledger.settle_workflow_phase = lost_planning_delivery
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        run = rig.coordinator._runs[rig.workflow.id]
        if not known:
            await rig.wait_for_provider_entry()
            await rig.coordinator.stop(
                rig.workflow.id, expected_control_revision=rig.workflow.control_revision
            )
            rig.provider_release()
        view = await rig.wait_for_status("recovery_required", timeout=30)
        await asyncio.gather(run.driver, return_exceptions=True)
        assert view.accounting.uncertain is not known
        original_review = service.resolve_recovery
        delivered = False

        def resolve_recovery(self, *args, **kwargs):
            nonlocal delivered
            result = original_review(*args, **kwargs)
            if not delivered:
                delivered = True
                raise RuntimeError("committed closing review delivery lost")
            return result

        service.resolve_recovery = MethodType(resolve_recovery, service)
        decision = WorkflowExecutionRecovery(reason="reviewed_prior_effects")
        with pytest.raises(
            RuntimeError, match="committed closing review delivery lost"
        ):
            await rig.coordinator.reconcile(
                view.id,
                expected_control_revision=view.control_revision,
                decision=decision,
                command_id="lost-review-response",
            )
        durable = await service.call(service.get, view.id)
        assert durable.status == "closed" and durable.accounting == view.accounting
        refreshed = await rig.coordinator.get(view.id)
        with pytest.raises(ValueError, match="revision_conflict"):
            await rig.coordinator.reconcile(
                view.id,
                expected_control_revision=view.control_revision,
                decision=decision,
                command_id="stale-closing-retry",
            )
        reviewed = await rig.coordinator.reconcile(
            view.id,
            expected_control_revision=refreshed.control_revision,
            decision=decision,
            command_id="fresh-closing-retry",
        )
        assert (
            reviewed.status == "stopped"
            and reviewed.control_revision == durable.control_revision
        )
        assert (
            reviewed.accounting == durable.accounting
            and reviewed.plans == durable.plans
        )
        assert (
            view.id not in rig.coordinator._runs
            and not rig.runtime.worktree_activity._recovery
        )
        removed = await rig.coordinator.remove_results(
            view.id,
            expected_control_revision=reviewed.control_revision,
            command_id="remove-after-closing-retry",
        )
        repeated = await service.call(
            service.resolve_recovery,
            view.id,
            expected_control_revision=removed.control_revision,
            decision=decision,
        )
        assert repeated == removed and repeated.status == "removed"
        with owned_goal_storage(service), service.db.connection() as conn:
            marker = conn.execute(
                "SELECT recovery_reviewed_revision FROM workflow_runs WHERE id=?",
                (view.id,),
            ).fetchone()[0]
        assert marker == durable.control_revision
        assert (
            len(rig.provider.calls) == 1
            and not rig.runtime.execution_capacity.snapshot().executions
        )
        complete = True
    finally:
        rig.provider_release()
        if complete:
            await rig.close()
        else:
            import traceback

            traceback.print_exc()
            print("CLOSING_RETRY_NOT_PROVEN", flush=True)
            os._exit(17)
