"""Mounted review controls operate on the displayed immutable workflow."""

import pytest

from Tests.UI.test_console_workflow_command import run_workflow_ui

pytestmark = pytest.mark.bootstrap_profile


def test_approve_then_accept_real_phase_from_its_detail(tmp_path):
    run_workflow_ui(tmp_path, "controls")


@pytest.mark.parametrize(
    "mode",
    [
        "pause",
        "stale",
        "revise",
        "preset",
        "selection",
        "malformed",
        "pending",
        "metadata",
        "capture",
        "two-phase",
        "clean-failure",
        "mixed-failure",
        "unknown-task",
        "detail-storage-error",
    ],
)
def test_native_workflow_control_boundaries(tmp_path, mode):
    run_workflow_ui(tmp_path, mode)


@pytest.mark.asyncio
@pytest.mark.parametrize("close_early", [False, True])
async def test_setup_completion_cannot_pop_a_cover_or_reopen_a_closed_setup(
    close_early,
):
    import asyncio

    from textual.app import App
    from textual.screen import ModalScreen

    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
    from tldw_chatbook.Agents.goal_models import GoalBindingRef
    from tldw_chatbook.Widgets.Console.console_workflow_setup import (
        ConsoleWorkflowSetup,
    )

    entered, released, returned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    accepted = object()
    results = []

    class HeldSubmit:
        async def start(self, setup, *, launch_id):
            entered.set()
            await released.wait()
            returned.set()
            return accepted

    host = App()
    binding = GoalBindingRef(
        workspace_id="workspace", binding_id="folder", locator="/tmp", access="rw"
    )
    modal = ConsoleWorkflowSetup(
        HeldSubmit(),
        "chat",
        ("openai", "test"),
        AutomaticWorkLimits.from_settings("workflow_phase"),
        (((binding, 0),), (), None),
    )
    cover = ModalScreen()
    async with host.run_test(size=(100, 30)) as pilot:
        await host.push_screen(modal, callback=results.append)
        modal.query_one("#workflow-objective").load_text("Held workflow")
        await pilot.pause()
        assert await pilot.click("#workflow-plan")
        await asyncio.wait_for(entered.wait(), 2)
        if close_early:
            await pilot.press("escape")
            await pilot.pause()
            assert results == [None] and modal not in host.screen_stack
        await host.push_screen(cover)
        released.set()
        await asyncio.wait_for(returned.wait(), 2)
        await pilot.pause()
        assert host.screen is cover
        assert results == ([None] if close_early else [])
        cover.dismiss()
        await pilot.pause()
        assert modal not in host.screen_stack
        assert results == ([None] if close_early else [accepted])
