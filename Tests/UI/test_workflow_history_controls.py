"""Recovery review and payload removal use real history and revision checks."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = """
import asyncio, os, sys
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install(); real_profile_guard.install()
os.environ.update(TLDW_AGENTS_WORKFLOW_RUNS_ENABLED="true", TLDW_AGENTS_AUTOWAKE_ENABLED="false", OPENAI_API_KEY="private-history-test")
from Tests.UI.test_workflow_history_controls import history_journey
with user_fixture_default_owner():
    asyncio.run(history_journey(Path(os.environ["XDG_DATA_HOME"]).parent, sys.argv[2]))
"""


@pytest.mark.parametrize("size", ["140x42", "80x24", "140x42-known"])
def test_review_and_remove_saved_history_without_running_agents(tmp_path, size):
    _run(tmp_path, "history-controls", size, script=_SCRIPT, timeout=80)


async def history_journey(tmp_path, size):
    import asyncio
    import json

    from textual.widgets import Button

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
    from tldw_chatbook.Widgets.Console.console_workflow_status import (
        ConsoleWorkflowStatus,
    )

    uncertain = not size.endswith("-known")
    size = size.removesuffix("-known")
    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    service, db = rig.service, rig.service.db
    capacity = rig.runtime.execution_capacity
    try:
        with owned_goal_storage(service):
            saved = db.workflow_runs.create(
                rig.workflow.request, launch_id="saved-history"
            )
            saved = db.workflow_runs.save_plan(
                saved.id,
                WorkflowPlan.model_validate_json(json.dumps(rig.provider.plan)),
                expected_control_revision=saved.control_revision,
            )
            if uncertain:
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
                db.automatic_work.recover(current_owner_id="reopened-history")
            else:
                service.set_status(
                    saved.id,
                    status="recovery_required",
                    expected_control_revision=saved.control_revision,
                )
        await rig.runtime._project_workflow_recovery(db.automatic_work.recovery_result)
        before = await service.call(service.get, saved.id)
        assert before.status == "recovery_required"
        assert before.accounting.uncertain is uncertain
        host = ConsoleHarness(rig.controller.app)

        async def until(pilot, predicate):
            async with asyncio.timeout(10):
                while not predicate():
                    await pilot.pause(0.03)

        async def click(pilot, selector):
            target = host.screen.query_one(selector, Button)
            await until(pilot, lambda: not target.disabled)
            target.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click(selector)

        async with host.run_test(size=tuple(map(int, size.split("x")))) as pilot:
            await pilot.pause()
            host.screen._workflows.open_detail(saved.request.conversation_id, saved.id)
            await until(pilot, lambda: isinstance(host.screen, ConsoleWorkflowStatus))
            status = host.screen
            await until(pilot, lambda: status.snapshot is not None)
            await click(pilot, "#workflow-recovery")
            await until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
            await click(pilot, "#cancel-button")
            await until(pilot, lambda: host.screen is status and not status._sending)
            assert (await service.call(service.get, saved.id)) == before

            await click(pilot, "#workflow-recovery")
            await until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
            changed = await service.call(
                service.set_status,
                saved.id,
                status="recovery_required",
                expected_control_revision=before.control_revision,
            )
            await click(pilot, "#confirm-button")
            await until(pilot, lambda: host.screen is status and not status._sending)
            await until(
                pilot,
                lambda: status.snapshot.control_revision == changed.control_revision,
            )
            assert status.snapshot.status == "recovery_required"

            await click(pilot, "#workflow-recovery")
            await until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
            await click(pilot, "#confirm-button")
            await until(
                pilot,
                lambda: (
                    host.screen is status
                    and status.snapshot.status == "stopped"
                    and not status._sending
                ),
            )
            reviewed = await service.call(service.get, saved.id)
            assert reviewed.status == "closed"
            assert reviewed.accounting.used == before.accounting.used
            assert reviewed.accounting.reserved == before.accounting.reserved
            assert reviewed.accounting.uncertain is uncertain
            assert not rig.runtime.worktree_activity._recovery
            host.save_screenshot(str(tmp_path / f"workflow-history-{size}.svg"))

            await click(pilot, "#workflow-remove")
            await until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
            await click(pilot, "#confirm-button")
            await until(
                pilot,
                lambda: (
                    host.screen is status
                    and status.snapshot.status == "removed"
                    and not status._sending
                ),
            )
            removed = await service.call(service.get, saved.id)
            assert removed.plans == before.plans
            assert removed.accounting.used == before.accounting.used
            assert removed.accounting.reserved == before.accounting.reserved
            assert removed.accounting.uncertain is uncertain
            assert not removed.results and not rig.provider.calls
            assert not rig.runtime.execution_capacity.snapshot().executions
            await pilot.press("escape")
    finally:
        rig.provider_release()
        await rig.runtime.dispose()
        assert not capacity.snapshot().executions
        await rig.gateway.aclose()
        rig.creators.retire()
        print("retired and reopened", flush=True)
