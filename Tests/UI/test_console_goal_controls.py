"""Mounted controls operate the actual runtime/SQLite service, without UI ownership."""

import asyncio
import importlib
import threading

import pytest
from textual.app import App

from Tests.Agents.test_goal_iteration_report import report
from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture

stores = _stores_fixture


def goal_view():
    spec = importlib.util.find_spec("tldw_chatbook.Widgets.Console.console_goal_status")
    assert spec is not None, "selected goal controls are missing"
    return importlib.import_module(spec.name).ConsoleGoalStatus


class GoalHarness(App):
    def __init__(self, coordinator, goal_id):
        super().__init__()
        self.coordinator, self.goal_id = coordinator, goal_id

    def on_mount(self):
        self.push_screen(goal_view()(self.coordinator, self.goal_id))


@pytest.mark.asyncio
async def test_mounted_pause_resume_stop_preserve_service_spend_and_foreground_draft(
    stores, monkeypatch, tmp_path
):
    entered, release = threading.Event(), threading.Event()

    def provider(**kwargs):
        entered.set()
        release.wait(3)
        return {
            "choices": [
                {"message": {"content": report(summary="Saved investigation")}}
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    goal, store, _session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
    from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker

    controller._agent_bridge._change_tracker = ChangeTurnTracker(
        service=ShadowRepoService(data_dir=tmp_path / "review-data")
    )
    ordinary = store.create_session()
    ordinary.draft = "ordinary composer draft"
    # The selected surface must not steal the active conversation.
    app = GoalHarness(coordinator, goal.id)
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert await pilot.click("#goal-resume")
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            from tldw_chatbook.UI.Console_Modules.goals import change_review_provider

            review = change_review_provider(
                controller, controller._agent_bridge, goal.conversation_id
            )
            assert review is not None and review.run_active()
            assert await pilot.click("#goal-pause")
            await pilot.pause()
            assert coordinator.service.get(goal.id).status == "pause_requested"
            release.set()
            await coordinator.start(goal.id)
            await pilot.pause()
            assert coordinator.service.get(goal.id).status == "paused"
            assert len(calls) == 1
            assert not review.run_active()
            entered.clear()
            release.clear()
            assert await pilot.click("#goal-resume")
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            assert await pilot.click("#goal-stop")
            await pilot.pause()
            release.set()
            await coordinator.start(goal.id)
            await pilot.pause()
            # Interrupting a provider can leave unknown spend; Stop must preserve it.
            assert coordinator.service.get(goal.id).status == "recovery_required"
            assert coordinator.service.get(goal.id).accounting.uncertain
            assert len(calls) == 2
            assert store.active_session_id == ordinary.id
            assert ordinary.draft == "ordinary composer draft"
            assert coordinator.service.get(goal.id).accounting.used["generation"] == 2
    finally:
        release.set()
        await coordinator.shutdown()
        await gateway.aclose()


@pytest.mark.asyncio
async def test_view_unmount_does_not_cancel_runtime_and_removed_history_has_no_resume(
    stores, monkeypatch
):
    entered, release = threading.Event(), threading.Event()

    def provider(**kwargs):
        entered.set()
        release.wait(3)
        return {
            "choices": [{"message": {"content": report()}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    goal, _, _, _, coordinator, gateway, _calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    try:
        app = GoalHarness(coordinator, goal.id)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click("#goal-resume")
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            await pilot.press("escape")
            assert coordinator.active_goal_id == goal.id
            coordinator.service.pause(goal.id)
            release.set()
            await coordinator.start(goal.id)
            app.push_screen(goal_view()(coordinator, goal.id))
            await pilot.pause()
            await pilot.click("#goal-remove")
            await pilot.pause()
            assert not app.screen.query_one("#goal-resume").display
            assert (
                "removed" in str(app.screen.query_one("#goal-state").render()).lower()
            )
    finally:
        release.set()
        await coordinator.shutdown()
        await gateway.aclose()


@pytest.mark.asyncio
async def test_runtime_notifications_are_body_free_detachable_and_do_not_reaudit(
    stores, monkeypatch
):
    goal, _, _, _, coordinator, gateway, _calls = build_goal_rig(stores, monkeypatch)
    observed = []
    try:
        assert hasattr(coordinator, "subscribe"), "runtime notifications missing"
        detach = coordinator.subscribe(observed.append)
        await coordinator.start(goal.id)
        assert observed and all(item == goal.id for item in observed)
        detach()
        count = len(observed)
        coordinator.notify_goal_changed(goal.id)
        await asyncio.sleep(0)
        assert len(observed) == count
    finally:
        await coordinator.shutdown()
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [False, True])
async def test_actual_review_controls_bind_artifact_and_recheck_before_accept(
    stores, monkeypatch, tmp_path, stale
):
    from Tests.Agents.test_goal_progress import run_verifier

    co, goal, result, fixture, *_ = await run_verifier(
        stores, monkeypatch, tmp_path, human=True
    )
    saved = co.service.checkpoint(result)
    assert saved.status == "awaiting_result_review"
    app = GoalHarness(co, goal.id)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.click("#goal-review")
        await pilot.pause()
        if stale:
            fixture.write_text("edited after review")
        await pilot.click("#goal-accept")
        await pilot.pause()
        after = co.service.get(goal.id)
        assert after.status == ("awaiting_result_review" if stale else "completed")
        assert after.accounting.used == saved.accounting.used
        if stale:
            assert "objective_proof_unavailable" in str(
                app.screen.query_one("#goal-error").render()
            )


@pytest.mark.asyncio
async def test_selecting_older_checkpoint_clears_quality_acceptance(
    stores, monkeypatch, tmp_path
):
    from textual.widgets import Select

    from Tests.Agents.test_goal_progress import run_verifier

    co, goal, result, *_ = await run_verifier(stores, monkeypatch, tmp_path, human=True)
    first = co.service.checkpoint(result)
    checkpoint = first.checkpoints[-1]
    paused = co.service.review_result(
        goal.id,
        expected_revision=first.revision,
        checkpoint_id=checkpoint.id,
        artifact_digest=checkpoint.artifact_digest,
        accepted=False,
    )
    co.service.resume(goal.id, expected_revision=paused.revision)
    second = co.service.checkpoint(await co.dispatch_once(goal.id))
    assert second.status == "awaiting_result_review"
    app = GoalHarness(co, goal.id)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#goal-review")
        await pilot.pause()
        assert app.screen.query_one("#goal-accept").display
        app.screen.query_one("#goal-iteration", Select).value = checkpoint.id
        await pilot.pause()
        assert not app.screen.query_one("#goal-accept").display
        await pilot.click("#goal-review")
        await pilot.pause()
        assert "stale_result_review" in str(
            app.screen.query_one("#goal-error").render()
        )
        assert co.service.get(goal.id).status == "awaiting_result_review"
