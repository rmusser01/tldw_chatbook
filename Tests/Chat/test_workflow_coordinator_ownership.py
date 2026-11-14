"""The native coordinator survives transparent operational wrappers."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_JOURNEY = r"""
import asyncio, functools, os
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install(); real_profile_guard.install()
os.environ.update(TLDW_AGENTS_WORKFLOW_RUNS_ENABLED="true", TLDW_AGENTS_AUTOWAKE_ENABLED="false", OPENAI_API_KEY="private-workflow-test")
from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
from tldw_chatbook.Chat.console_workflow_runs import ConsoleWorkflowCoordinator
import pytest

async def journey():
    rig = await build_workflow_rig(Path(os.environ["XDG_DATA_HOME"]).parent, None, WorkflowProvider())
    try:
        with pytest.MonkeyPatch.context() as patch:
            def asynchronous(original):
                @functools.wraps(original)
                async def wrapped(self, *args, **kwargs):
                    return await original(self, *args, **kwargs)
                return wrapped
            def synchronous(original):
                @functools.wraps(original)
                def wrapped(self, *args, **kwargs):
                    return original(self, *args, **kwargs)
                return wrapped
            for name in ("_plan", "_execute"):
                patch.setattr(ConsoleWorkflowCoordinator, name, asynchronous(getattr(ConsoleWorkflowCoordinator, name)))
            for name in ("_metadata", "_release", "_notify_execution_release"):
                patch.setattr(ConsoleWorkflowCoordinator, name, synchronous(getattr(ConsoleWorkflowCoordinator, name)))
            rig.workflow = await rig.coordinator.start(rig.workflow.request, launch_id=rig.workflow.launch_id)
            await rig.wait_for_status("awaiting_phase_review", timeout=30)
            decision = await rig.service.call(rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1")
            await rig.coordinator.approve(decision, command_id="approved-once")
            await rig.wait_for_status("awaiting_result_review", timeout=30)
            assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
            snapshot = await rig.coordinator.get(rig.workflow.id)
            assert len(snapshot.results) == 1 and snapshot.results[0].status == "completed"
            assert not rig.runtime.execution_capacity.snapshot().executions
    finally:
        await rig.close()

with user_fixture_default_owner():
    asyncio.run(journey())
print("retired and reopened")
"""


def test_wrapped_coordinator_keeps_native_approval_execution_and_retirement(tmp_path):
    _run(tmp_path, "ownership", "native", script=_JOURNEY, timeout=60)


_METADATA_SCRIPT = r"""
import asyncio, os
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install(); real_profile_guard.install()
os.environ.update(TLDW_AGENTS_WORKFLOW_RUNS_ENABLED="true", TLDW_AGENTS_AUTOWAKE_ENABLED="false", OPENAI_API_KEY="private-workflow-test")
from Tests.Chat.test_workflow_coordinator_ownership import recovery_during_metadata_read
with user_fixture_default_owner():
    asyncio.run(recovery_during_metadata_read(Path(os.environ["XDG_DATA_HOME"]).parent))
"""


def test_passive_metadata_read_does_not_block_review_of_nonlive_history(tmp_path):
    _run(tmp_path, "metadata-recovery", "native", script=_METADATA_SCRIPT, timeout=50)


async def recovery_during_metadata_read(tmp_path):
    import asyncio
    import json
    import threading

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import (
        WorkflowExecutionRecovery,
        WorkflowPlan,
    )

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    service, db = rig.service, rig.service.db
    entered, release = threading.Event(), threading.Event()
    reader = None
    try:
        with owned_goal_storage(service):
            saved = db.workflow_runs.create(
                rig.workflow.request, launch_id="interrupted"
            )
            saved = db.workflow_runs.save_plan(
                saved.id,
                WorkflowPlan.model_validate_json(json.dumps(rig.provider.plan)),
                expected_control_revision=saved.control_revision,
            )
            old = rig.controller._fleet_wake.runtime_owner_id
            phase = db.automatic_work.prepare_workflow_phase(
                saved.id,
                revision_id=1,
                phase_id="phase-1",
                owner_id=old,
                session_id="workflow-test-chat",
                command_id="interrupted-phase",
            )
            assert db.automatic_work.accept_workflow_phase(phase.id, owner_id=old)
            db.automatic_work.recover(current_owner_id="metadata-recovery")
        await rig.runtime._project_workflow_recovery(db.automatic_work.recovery_result)
        before = await service.call(service.get, saved.id)
        original = service.runtime_snapshot

        def held(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(service, "runtime_snapshot", held)
            reader = asyncio.create_task(rig.coordinator.get(saved.id))
            assert await asyncio.to_thread(entered.wait, 10)
            closed = await service.call(
                service.resolve_recovery,
                saved.id,
                expected_control_revision=before.control_revision,
                decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
            )
            assert closed.status == "closed" and closed.accounting == before.accounting
            assert not reader.done()
            assert not rig.runtime.worktree_activity._recovery
            release.set()
            await reader
        assert not rig.coordinator._metadata_cells
        repeated = await service.call(
            service.resolve_recovery,
            saved.id,
            expected_control_revision=closed.control_revision,
            decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
        )
        assert repeated == closed
    finally:
        release.set()
        if reader is not None:
            await asyncio.gather(reader, return_exceptions=True)
        await rig.close()
