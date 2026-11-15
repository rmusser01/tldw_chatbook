"""A failed native phase returns its chat only when no work is left running (TASK-40718)."""

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
from Tests.Chat import test_workflow_failed_phase_release as journeys
with user_fixture_default_owner():
    asyncio.run(getattr(journeys, sys.argv[1])(Path(os.environ["XDG_DATA_HOME"]).parent))
print("retired and reopened")
"""


def test_blocked_phase_returns_its_chat_without_a_restart(tmp_path):
    """A phase whose remaining task waits on a task it will never run raised
    "workflow dependencies blocked". Nothing had launched, yet the run stayed
    `recovery_required`, kept the chat's intake claim ("A turn is already
    preparing."), reconcile refused it, and app shutdown could not retire it."""
    _run(tmp_path, "blocked_phase", "native", script=_SCRIPT, timeout=90)


def test_phase_failing_with_a_live_worker_keeps_custody_until_reviewed(tmp_path):
    """A failure while a worker still runs keeps the original custody, and
    reconcile refuses it. Once the worker has returned, reconcile closes the
    run as reviewed history and the chat gets its claim back; on dev 1455ceeb96
    it kept refusing with `workflow_resources_still_live` because execution
    runs have no planning preparation. A refused record leaves the phase able
    to retry, and a claim release that raises is finished by the next
    reconcile instead of wedging the chat."""
    _run(tmp_path, "live_worker_failure", "native", script=_SCRIPT, timeout=90)


def test_ownership_error_in_a_phase_keeps_a_claim_it_does_not_own(tmp_path):
    """The failed-phase path catches every ordinary exception, ownership errors
    included. A primary token that changed hands stays with its new owner:
    the run keeps custody until reconcile, once ownership is restored."""
    _run(tmp_path, "ownership_error", "native", script=_SCRIPT, timeout=90)


async def _started(rig):
    rig.provider.plan["phases"][0]["tasks"].append(
        dict(
            rig.provider.plan["phases"][0]["tasks"][0],
            id="task-2",
            instructions="Execute task-2",
            dependencies=["task-1"],
        )
    )
    rig.workflow = await rig.coordinator.start(
        rig.workflow.request, launch_id=rig.workflow.launch_id
    )
    await rig.wait_for_status("awaiting_phase_review", timeout=30)
    return await rig.service.call(
        rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
    )


async def blocked_phase(tmp_path):
    import asyncio
    import functools
    from types import MethodType

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    session_id = "workflow-test-chat"
    try:
        decision = await _started(rig)
        service = rig.service
        original = service.phase_task_ids
        reads = []

        @functools.wraps(original)
        def without_first(self, phase):
            # Only the driver's own read (the first) drops task-1; the read
            # inside reviewed_task_ids keeps it. task-1 is then neither
            # pending nor done, and task-2 depends on it: blocked.
            reads.append(phase)
            ids = original(phase)
            if len(reads) > 1:
                return ids
            return tuple(task for task in ids if task != "task-1")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(service, "phase_task_ids", MethodType(without_first, service))
            await rig.coordinator.approve(decision, command_id="approved-once")
            run = rig.coordinator._runs[rig.workflow.id]
            await asyncio.wait_for(asyncio.shield(run.driver), 15)
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        assert not rig.runtime.execution_capacity.snapshot().executions
        assert rig.workflow.id not in rig.coordinator._runs
        # The same resumable state as a phase whose task failed.
        assert (await rig.coordinator.get(rig.workflow.id)).status == "paused"
        sent = await rig.controller.submit_draft(
            "A new turn after the failed phase", session_id=session_id
        )
        assert sent.accepted, sent
        stream = rig.controller._active_stream_tasks.get(session_id)
        if stream is not None:
            await asyncio.wait_for(asyncio.shield(stream), 15)
    finally:
        await rig.close()


async def live_worker_failure(tmp_path):
    import asyncio
    import threading

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery
    from tldw_chatbook.Chat import console_workflow_wave

    from loguru import logger

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider(hold=("task-1",)))
    original = console_workflow_wave.run_workflow_wave
    waves = []
    logged = []
    sink = logger.add(logged.append, format="{message}", level="ERROR")

    async def failing_while_live(phase, selected, **kwargs):
        # The real wave keeps running; only the phase driver's wait fails.
        waves.append(asyncio.ensure_future(original(phase, selected, **kwargs)))
        await rig.wait_for_provider_entry("task-1")
        raise RuntimeError("wave observer failed")

    try:
        decision = await _started(rig)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                console_workflow_wave, "run_workflow_wave", failing_while_live
            )
            await rig.coordinator.approve(decision, command_id="approved-once")
            run = rig.coordinator._runs[rig.workflow.id]
            await asyncio.wait_for(asyncio.shield(run.driver), 15)
        assert rig.coordinator._runs.get(rig.workflow.id) is run
        # Never attempted as a returned failure: no cleanup over live work.
        assert not [line for line in logged if "phase failed" in line], logged
        snapshot = await rig.coordinator.get(rig.workflow.id)
        assert snapshot.status == "recovery_required"
        stored = await rig.service.call(rig.service.get, rig.workflow.id)
        try:
            await rig.coordinator.reconcile(
                rig.workflow.id,
                expected_control_revision=snapshot.control_revision,
                decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
                command_id="reconcile-live",
            )
        except ValueError as error:
            assert str(error) == "workflow_resources_still_live", error
        else:
            raise AssertionError("reconcile accepted a phase with a live worker")
        # The refusal recorded nothing.
        assert await rig.service.call(rig.service.get, rig.workflow.id) == stored
        rig.provider_release()
        await asyncio.wait_for(asyncio.gather(*waves, return_exceptions=True), 15)
        review = WorkflowExecutionRecovery(reason="reviewed_prior_effects")
        returned = await rig.service.call(rig.service.get, rig.workflow.id)
        # A record refused inside its transaction keeps the phase retryable.
        with pytest.raises(ValueError, match="revision_conflict"):
            await rig.coordinator.reconcile(
                rig.workflow.id,
                expected_control_revision=snapshot.control_revision + 1,
                decision=review,
                command_id="reconcile-stale",
            )
        assert not run.phase._retired
        assert await rig.service.call(rig.service.get, rig.workflow.id) == returned
        # A launch racing the record: it starts right after the live check
        # passes, through the launch gate's own protocol (take the phase's
        # control lock, refuse a retired phase; agent_native_execution.py).
        # It must be refused, or the record must refuse; never both proceed.
        live_check = rig.service._has_live_workflow_resources
        launches = []

        def racing_launch():
            with run.phase._control_lock:
                launches.append("refused" if run.phase._retired else "bound")

        def check_then_launch(workflow_id):
            live = live_check(workflow_id)
            if not launches:
                launcher = threading.Thread(target=racing_launch)
                launcher.start()
                launcher.join(10)
            return live

        # A release that raises is finished by the next reconcile.
        wake = rig.controller._fleet_wake
        release = wake.release_automatic_primary
        raised = []

        def release_once(session_id, token):
            if not raised:
                raised.append(session_id)
                raise RuntimeError("primary release failed")
            return release(session_id, token)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                rig.service, "_has_live_workflow_resources", check_then_launch
            )
            patch.setattr(wake, "release_automatic_primary", release_once)
            with pytest.raises(RuntimeError, match="primary release failed"):
                await rig.coordinator.reconcile(
                    rig.workflow.id,
                    expected_control_revision=snapshot.control_revision,
                    decision=review,
                    command_id="reconcile-returned",
                )
            assert launches == ["refused"], launches
            assert rig.coordinator._runs.get(rig.workflow.id) is run
            current = await rig.service.call(rig.service.get, rig.workflow.id)
            closed = await rig.coordinator.reconcile(
                rig.workflow.id,
                expected_control_revision=current.control_revision,
                decision=review,
                command_id="reconcile-retry",
            )
        assert raised == ["workflow-test-chat"]
        assert closed.status == "stopped"  # Closed history displays as stopped.
        assert rig.workflow.id not in rig.coordinator._runs
        service, db = rig.service, rig.service.db
        with owned_goal_storage(service), db.connection() as conn:
            row = conn.execute(
                "SELECT status,recovery_reviewed_revision FROM workflow_runs WHERE id=?",
                (rig.workflow.id,),
            ).fetchone()
        assert tuple(row) == ("closed", closed.control_revision)
        sent = await rig.controller.submit_draft(
            "A new turn after the reviewed phase", session_id="workflow-test-chat"
        )
        assert sent.accepted, sent
        stream = rig.controller._active_stream_tasks.get("workflow-test-chat")
        if stream is not None:
            await asyncio.wait_for(asyncio.shield(stream), 15)
    finally:
        logger.remove(sink)
        rig.provider_release()
        await asyncio.wait_for(asyncio.gather(*waves, return_exceptions=True), 15)
        await rig.close()


async def ownership_error(tmp_path):
    import asyncio
    import functools
    from types import MethodType

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    session_id = "workflow-test-chat"
    wake = rig.controller._fleet_wake
    foreign = object()
    try:
        decision = await _started(rig)
        service = rig.service
        original = service.phase_task_ids
        reads = []

        @functools.wraps(original)
        def token_changed_hands(self, phase):
            reads.append(phase)
            if len(reads) == 1:
                with wake._registry_lock:
                    wake._automatic_primary_claims[session_id] = foreign
                raise PermissionError("workflow original primary token changed")
            return original(phase)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                service, "phase_task_ids", MethodType(token_changed_hands, service)
            )
            await rig.coordinator.approve(decision, command_id="approved-once")
            run = rig.coordinator._runs[rig.workflow.id]
            await asyncio.wait_for(asyncio.shield(run.driver), 15)
        assert wake._automatic_primary_claims.get(session_id) is foreign
        assert rig.coordinator._runs.get(rig.workflow.id) is run
        assert (await rig.coordinator.get(rig.workflow.id)).status == (
            "recovery_required"
        )
        assert not (
            await rig.controller.submit_draft("Still held", session_id=session_id)
        ).accepted
        # Ownership restored: the review closes the run and returns the claim.
        with wake._registry_lock:
            wake._automatic_primary_claims[session_id] = run.token
        current = await service.call(service.get, rig.workflow.id)
        closed = await rig.coordinator.reconcile(
            rig.workflow.id,
            expected_control_revision=current.control_revision,
            decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
            command_id="reconcile-restored",
        )
        assert closed.status == "stopped"
        assert rig.workflow.id not in rig.coordinator._runs
        sent = await rig.controller.submit_draft(
            "A new turn after the review", session_id=session_id
        )
        assert sent.accepted, sent
        stream = rig.controller._active_stream_tasks.get(session_id)
        if stream is not None:
            await asyncio.wait_for(asyncio.shield(stream), 15)
    finally:
        with wake._registry_lock:
            if wake._automatic_primary_claims.get(session_id) is foreign:
                del wake._automatic_primary_claims[session_id]
        await rig.close()
