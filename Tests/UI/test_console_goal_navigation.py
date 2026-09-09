"""Goal palette and review routing bind selected conversations rather than foreground."""

import pytest

from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider

stores = _stores_fixture


@pytest.mark.asyncio
async def test_palette_discovers_and_opens_goal_history(stores):
    app = _build_test_app()
    app.chachanotes_db = stores[1].db
    app.workspace_registry_service = stores[2]
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        console = host.screen
        provider = ConsoleCommandProvider(console)
        commands = [hit async for hit in provider.discover()]
        hits = [hit for hit in commands if "Goal runs" in hit.text]
        assert hits, "Goal history is missing from the actual palette"
        hits[0].command()
        await pilot.pause(0.2)
        assert host.screen.query("#goal-history")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (160, 44)])
async def test_production_styles_keep_goal_actions_visible_and_keyboard_accessible(
    stores, monkeypatch, tmp_path, size
):
    import os
    from pathlib import Path

    from textual.widgets import Button

    from Tests.Agents.test_goal_progress import run_verifier
    from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
    from tldw_chatbook.Widgets.Console.console_goal_setup_modal import (
        ConsoleGoalSetupModal,
    )
    from tldw_chatbook.Widgets.Console.console_goal_status import ConsoleGoalStatus

    co, goal, result, *_ = await run_verifier(stores, monkeypatch, tmp_path, human=True)
    co.service.checkpoint(result)
    app = _build_test_app()
    app.chachanotes_db = stores[1].db
    app.workspace_registry_service = stores[2]

    class StyledConsole(ConsoleHarness):
        CSS_PATH = str(BUNDLED_STYLESHEET)

    host = StyledConsole(app)
    async with host.run_test(size=size) as pilot:
        await pilot.pause()
        console = host.screen
        # Real rail opener; a collapsed fresh-profile rail is not geometry evidence.
        console._ensure_console_chat_controller().store.ensure_session(
            title="Goal review"
        )
        console._reveal_console_inspector_rail()
        await pilot.pause()
        assert console._current_console_rail_state().right_open
        rail = console.query_one("#console-right-rail")
        assert rail.display and rail.region.width > 0
        await pilot.pause()
        host.push_screen(ConsoleGoalStatus(co, goal.id))
        await pilot.pause()
        review = host.screen.query_one("#goal-review", Button)
        close = host.screen.query_one("#goal-close", Button)
        for button in (review, close):
            assert host.screen.region.contains_region(button.region)
            assert str(button.label) in host.export_screenshot()
        review.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert host.screen.query_one("#goal-accept", Button).display
        target = os.environ.get("TLDW_GOAL_SCREENSHOT_DIR")
        if target:
            Path(target).mkdir(parents=True, exist_ok=True)
            Path(target, f"goal-review-{size[0]}x{size[1]}.svg").write_text(
                host.export_screenshot()
            )
        await pilot.press("escape")

        async def no_start(request, launch_id):
            pytest.fail("Geometry inspection must not dispatch a provider")

        host.push_screen(
            ConsoleGoalSetupModal(
                goal.request,
                start=no_start,
                bindings=(goal.request.binding,),
                tool_ids=goal.request.tool_scope.catalog_tools,
                configure=lambda *args: None,
            )
        )
        await pilot.pause()
        start = host.screen.query_one("#goal-start", Button)
        assert host.screen.region.contains_region(start.region)
        start.focus()
        await pilot.pause()
        assert host.focused is start
        import html

        assert "Review launch" in html.unescape(host.export_screenshot()).replace(
            "\xa0", " "
        )
        if target:
            Path(target, f"goal-setup-{size[0]}x{size[1]}.svg").write_text(
                host.export_screenshot()
            )
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_changes_control_opens_exact_saved_iteration_and_conversation(
    stores, monkeypatch, tmp_path
):
    import asyncio

    from Tests.Agents.test_goal_progress import run_verifier
    from Tests.UI.test_console_goal_controls import GoalHarness
    from tldw_chatbook.UI.Console_Modules.goals import ConsoleGoalsController

    co, goal, result, *_ = await run_verifier(stores, monkeypatch, tmp_path, human=True)
    saved = co.service.checkpoint(result)
    opened = []
    ui = ConsoleGoalsController(
        app_instance=None,
        get_controller=lambda: co.controller,
        get_coordinator=lambda: co,
        push_screen=lambda *args: None,
        run_worker=asyncio.create_task,
        open_changes=lambda run_id, **kw: opened.append((run_id, kw)),
    )
    app = GoalHarness(co, goal.id)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.review_changes = ui.review_changes
        app.screen.load_goal()
        await pilot.pause()
        await pilot.click("#goal-changes")
        await pilot.pause()
        assert opened == [
            (result.native_run_id, {"conversation_id": goal.conversation_id})
        ]
        assert hasattr(co.service, "checkpoint_run_id"), (
            "exact iteration lookup belongs to the service"
        )
        assert (
            co.service.checkpoint_run_id(goal.id, saved.checkpoints[-1].id)
            == result.native_run_id
        )
        with pytest.raises(ValueError):
            co.service.checkpoint_run_id(goal.id, "foreign")


@pytest.mark.asyncio
async def test_older_history_reaches_retained_goal_behind_newer_tombstones(
    stores, monkeypatch
):
    import asyncio
    from types import SimpleNamespace

    from textual.app import App
    from textual.widgets import Select

    from Tests.Chat.test_console_goal_dispatch import build_goal_rig
    from tldw_chatbook.UI.Console_Modules.goals import ConsoleGoalsController

    old, _, _, controller, co, gateway, calls = build_goal_rig(stores, monkeypatch)
    old = co.service.pause(old.id)
    for number in range(50):
        newer = co.service.create(old.request, launch_id=f"history-{number}")
        co.service.pause(newer.id)
        co.service.remove_payloads(newer.id)
    assert old.id not in {g.id for g in co.service.list_goals()}
    app = App()
    ui = ConsoleGoalsController(
        app_instance=SimpleNamespace(
            notify=lambda *args, **kwargs: pytest.fail(str(args))
        ),
        get_controller=lambda: controller,
        get_coordinator=lambda: co,
        push_screen=app.push_screen,
        run_worker=asyncio.create_task,
        open_changes=lambda *args, **kw: None,
    )
    try:
        async with app.run_test() as pilot:
            ui.open_history()
            await pilot.pause()
            assert len(app.screen.rows) == 50
            assert all(g.status == "removed" for g in app.screen.rows)
            assert await pilot.click("#goal-history-older")
            await pilot.pause()
            assert len(app.screen.rows) == 1
            assert app.screen.rows[0].id == old.id
            assert await pilot.click("#goal-history-newer")
            await pilot.pause()
            assert len(app.screen.rows) == 50
            assert await pilot.click("#goal-history-older")
            await pilot.pause()
            app.screen.query_one("#goal-history", Select).value = old.id
            assert await pilot.click("#goal-history-open")
            await pilot.pause()
            assert app.screen.goal_id == old.id
            assert await pilot.click("#goal-remove")
            await pilot.pause()
            removed = co.service.get(old.id)
            assert removed.status == "removed" and removed.request is None
            assert removed.accounting.used == old.accounting.used
            assert removed.accounting.reserved == old.accounting.reserved
            assert not calls
    finally:
        await co.shutdown()
        await gateway.aclose()
