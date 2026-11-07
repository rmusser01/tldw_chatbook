"""Original public planning refuses a second saved workflow before any model call."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("mode", ["public", "mounted"])
def test_existing_workflow_is_the_actionable_refusal_target(tmp_path, mode):
    _run(
        tmp_path,
        "workflow-existing",
        mode,
        script="""
import asyncio, gc, os, sys, warnings
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
os.environ.update(TLDW_AGENTS_WORKFLOW_RUNS_ENABLED="true", TLDW_AGENTS_AUTOWAKE_ENABLED="true", OPENAI_API_KEY="private-native-workflow-test")
from Tests.UI.test_workflow_existing_admission import existing_journey
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", RuntimeWarning)
    with user_fixture_default_owner():
        asyncio.run(existing_journey(Path(os.environ["XDG_DATA_HOME"]).parent, sys.argv[2]))
    gc.collect()
runtime_warnings = [str(item.message) for item in caught if issubclass(item.category, RuntimeWarning)]
assert not runtime_warnings, runtime_warnings
""",
        timeout=60,
    )


async def existing_journey(tmp_path, mode):
    import asyncio
    import json

    from loguru import logger
    from textual.widgets import TextArea

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from Tests.DB.test_workflow_admission_boundaries import _rows
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan
    from tldw_chatbook.UI.Console_Modules.workflows import workflow_view_error
    from tldw_chatbook.Widgets.Console.console_workflow_setup import (
        ConsoleWorkflowSetup,
    )
    from tldw_chatbook.Widgets.Console.console_workflow_status import (
        ConsoleWorkflowStatus,
    )

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    capacity = rig.runtime.execution_capacity
    diagnostics = []
    sink = logger.add(diagnostics.append, level="WARNING", format="{message}")
    try:
        with owned_goal_storage(rig.service):
            saved = rig.service.db.workflow_runs.create(
                rig.workflow.request, launch_id="original-saved"
            )
            saved = rig.service.db.workflow_runs.save_plan(
                saved.id,
                WorkflowPlan.model_validate_json(json.dumps(rig.provider.plan)),
                expected_control_revision=saved.control_revision,
            )
        if mode == "public":
            for status in (
                "awaiting_approval",
                "paused",
                "awaiting_resume",
                "recovery_required",
            ):
                if saved.status != status:
                    saved = await rig.service.call(
                        rig.service.set_status,
                        saved.id,
                        status=status,
                        expected_control_revision=saved.control_revision,
                    )
                with owned_goal_storage(rig.service):
                    before = _rows(rig.service.db)
                with pytest.raises(
                    RuntimeError, match="^workflow_already_exists$"
                ) as refused:
                    await rig.coordinator.start(
                        rig.workflow.request, launch_id="second-" + status
                    )
                assert refused.value.workflow_id == saved.id
                message = workflow_view_error("Prepare plan", refused.value)
                assert "workflow_already_exists" in message
                with owned_goal_storage(rig.service):
                    assert _rows(rig.service.db) == before
                assert not rig.provider.calls
                assert (
                    not rig.coordinator._runs and not rig.coordinator._pending_planning
                )
                assert not rig.runtime.execution_capacity.snapshot().executions
        else:
            host = ConsoleHarness(rig.controller.app)

            async def until(pilot, predicate):
                async with asyncio.timeout(8):
                    while not predicate():
                        await pilot.pause(0.03)

            async with host.run_test(size=(140, 42)) as pilot:
                console = host.screen
                await pilot.pause()
                owner = next(s for s in rig.store.sessions() if s.workspace_id == "w")
                await until(
                    pilot,
                    lambda: bool(console.query("#console-session-tab-" + owner.id)),
                )
                if rig.store.active_session_id != owner.id:
                    assert await pilot.click("#console-session-tab-" + owner.id)
                    await until(pilot, lambda: rig.store.active_session_id == owner.id)
                console._workflows.command("new")
                await until(
                    pilot, lambda: isinstance(host.screen, ConsoleWorkflowSetup)
                )
                setup = host.screen
                setup.query_one("#workflow-objective", TextArea).load_text(
                    "SECOND_PRIVATE_OBJECTIVE"
                )
                await pilot.pause()
                assert await pilot.click("#workflow-plan")
                await until(
                    pilot, lambda: isinstance(host.screen, ConsoleWorkflowStatus)
                )
                assert host.screen.workflow_id == saved.id
                assert host.screen.conversation_id == saved.request.conversation_id
                await until(pilot, lambda: host.screen.snapshot is not None)
                assert host.screen.snapshot.id == saved.id
                assert not rig.provider.calls
                assert (
                    len(
                        await rig.service.call(
                            rig.service.list_history, saved.request.conversation_id
                        )
                    )
                    == 1
                )
                assert (
                    not rig.coordinator._runs and not rig.coordinator._pending_planning
                )
                assert not rig.runtime.execution_capacity.snapshot().executions
                assert setup not in host.screen_stack
                await pilot.press("escape")
                await pilot.pause()
        assert all(
            "SECOND_PRIVATE_OBJECTIVE" not in str(line)
            and saved.request.objective not in str(line)
            for line in diagnostics
        )
    finally:
        logger.remove(sink)
        rig.provider_release()
        await rig.runtime.dispose()
        assert capacity.snapshot().executions == ()
        await rig.gateway.aclose()
        rig.creators.retire()
        print("retired and reopened", flush=True)
