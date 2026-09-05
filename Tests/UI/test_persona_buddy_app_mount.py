"""Dynamic app-owned mount coverage for the floating Persona Buddy."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import threading

import pytest
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Persona_Buddy import (
    PersonaBuddyController,
    PersonaBuddyPreferences,
    PersonaBuddySelection,
)
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy_widget import (
    PersonaBuddyWidget,
)
from tldw_chatbook.app import TldwCli


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if predicate():
            return
        if loop.time() >= deadline:
            raise AssertionError("predicate did not become true before deadline")
        await asyncio.sleep(0.01)


class _BuddyScreen(BaseAppScreen):
    def __init__(self, app_instance, name: str = "buddy-test") -> None:
        super().__init__(app_instance, name)


class _BuddyModal(ModalScreen[None]):
    def compose(self) -> ComposeResult:
        yield Static("Buddy modal", id="buddy-modal")


class _BuddyApp(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET
    reconcile_persona_buddy_view = TldwCli.reconcile_persona_buddy_view
    _start_persona_buddy_overlay = TldwCli._start_persona_buddy_overlay
    _schedule_persona_buddy_overlay = TldwCli._schedule_persona_buddy_overlay
    _notify_persona_buddy_changed = TldwCli._notify_persona_buddy_changed
    on_persona_buddy_changed = TldwCli.on_persona_buddy_changed
    on_base_app_screen_contents_rebuilt = TldwCli.on_base_app_screen_contents_rebuilt
    _persona_buddy_authority = staticmethod(TldwCli._persona_buddy_authority)
    is_persona_buddy_confirmed_unavailable = (
        TldwCli.is_persona_buddy_confirmed_unavailable
    )
    confirm_persona_buddy_unavailable = TldwCli.confirm_persona_buddy_unavailable

    def __init__(
        self,
        preferences: PersonaBuddyPreferences,
        *,
        production_resolution: bool = False,
    ) -> None:
        super().__init__()
        self.persona_buddy_controller = PersonaBuddyController(
            preferences=preferences,
            preference_writer=lambda _preferences: True,
            on_change=self._notify_persona_buddy_changed,
        )
        self._persona_buddy_unavailable_authority = None
        if not production_resolution:

            async def unresolved_until_test_requests_it(*, cols: int, lines: int):
                return None

            self.persona_buddy_controller.resolve_current_visual = (
                unresolved_until_test_requests_it
            )
        self.initial_screen = _BuddyScreen(self)

    async def on_mount(self) -> None:
        self._start_persona_buddy_overlay()
        await self.push_screen(self.initial_screen)


def _enabled_preferences() -> PersonaBuddyPreferences:
    return PersonaBuddyPreferences(
        enabled=True,
        selection=PersonaBuddySelection("local", "persona-1"),
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_overlay_refresh_without_screens_starts_no_presentation_work(enabled):
    app = _BuddyApp(PersonaBuddyPreferences(enabled=enabled))
    assert app.screen_stack == []

    app._schedule_persona_buddy_overlay()

    assert getattr(app, "_persona_buddy_overlay", None) is None


@pytest.mark.asyncio
async def test_threaded_controller_changes_reconcile_without_manual_ui_calls():
    app = _BuddyApp(PersonaBuddyPreferences())
    controller = app.persona_buddy_controller
    controller._on_change = app._notify_persona_buddy_changed
    async with app.run_test(size=(100, 30)):
        await asyncio.to_thread(
            controller.apply_preferences_patch,
            enabled=True,
            selection=PersonaBuddySelection("local", "persona-1"),
        )
        await _wait_until(lambda: len(list(app.screen.query(PersonaBuddyWidget))) == 1)
        await asyncio.to_thread(controller.apply_preferences_patch, enabled=False)
        await _wait_until(lambda: not list(app.screen.query(PersonaBuddyWidget)))


@pytest.mark.asyncio
async def test_disabled_navigation_and_recompose_start_no_buddy_workers(monkeypatch):
    app = _BuddyApp(PersonaBuddyPreferences())
    calls = []
    from textual.dom import DOMNode

    original = DOMNode.run_worker

    def counted(node, *args, **kwargs):
        if "persona-buddy" in kwargs.get("group", ""):
            calls.append(kwargs["group"])
        return original(node, *args, **kwargs)

    monkeypatch.setattr(DOMNode, "run_worker", counted)
    async with app.run_test(size=(100, 30)) as pilot:
        await app.screen.recompose()
        await app.switch_screen(_BuddyScreen(app, "replacement"))
        await pilot.pause()
        assert calls == []


@pytest.mark.asyncio
async def test_rebuild_replaces_view_without_screen_owned_buddy_state():
    app = _BuddyApp(_enabled_preferences())
    async with app.run_test(size=(100, 30)):
        await _wait_until(lambda: len(list(app.screen.query(PersonaBuddyWidget))) == 1)
        before = app.screen.query_one(PersonaBuddyWidget)
        await app.screen.recompose()
        await _wait_until(
            lambda: (
                len(list(app.screen.query(PersonaBuddyWidget))) == 1
                and app.screen.query_one(PersonaBuddyWidget) is not before
            )
        )
        assert "_persona_buddy_view" not in vars(app.screen)
        assert "_persona_buddy_reconcile_lock" not in vars(app.screen)


@pytest.mark.asyncio
async def test_enable_mounts_on_current_screen_without_navigation():
    app = _BuddyApp(PersonaBuddyPreferences())
    async with app.run_test(size=(100, 30)):
        assert not list(app.screen.query(PersonaBuddyWidget))
        await app.persona_buddy_controller.update_preferences(_enabled_preferences())
        await app.reconcile_persona_buddy_view()
        await _wait_until(lambda: len(list(app.screen.query(PersonaBuddyWidget))) == 1)


@pytest.mark.asyncio
async def test_disable_unmounts_without_navigation():
    app = _BuddyApp(_enabled_preferences())
    async with app.run_test(size=(100, 30)):
        await _wait_until(lambda: len(list(app.screen.query(PersonaBuddyWidget))) == 1)
        preferences = app.persona_buddy_controller.current_preferences()
        await app.persona_buddy_controller.update_preferences(
            replace(preferences, enabled=False)
        )
        await app.reconcile_persona_buddy_view()
        await _wait_until(lambda: not list(app.screen.query(PersonaBuddyWidget)))


@pytest.mark.asyncio
async def test_close_removes_only_current_generation():
    app = _BuddyApp(_enabled_preferences())
    async with app.run_test(size=(100, 30)):
        await _wait_until(lambda: len(list(app.screen.query(PersonaBuddyWidget))) == 1)
        stale = app.screen.query_one(PersonaBuddyWidget)
        await app.reconcile_persona_buddy_view()
        current = app.screen.query_one(PersonaBuddyWidget)
        assert current is stale

        await stale.close_and_persist()
        await _wait_until(lambda: not list(app.screen.query(PersonaBuddyWidget)))
        assert stale.view_generation < app._persona_buddy_overlay.generation


@pytest.mark.asyncio
async def test_reopen_mounts_without_navigation():
    app = _BuddyApp(replace(_enabled_preferences(), open=False))
    async with app.run_test(size=(100, 30)):
        assert not list(app.screen.query(PersonaBuddyWidget))
        preferences = app.persona_buddy_controller.current_preferences()
        await app.persona_buddy_controller.update_preferences(
            replace(preferences, open=True)
        )
        await app.reconcile_persona_buddy_view()
        await _wait_until(lambda: len(list(app.screen.query(PersonaBuddyWidget))) == 1)


@pytest.mark.asyncio
async def test_recompose_unsubscribes_and_remounts_both_directions():
    app = _BuddyApp(_enabled_preferences())
    async with app.run_test(size=(100, 30)):
        await _wait_until(lambda: len(list(app.screen.query(PersonaBuddyWidget))) == 1)
        first = app.screen.query_one(PersonaBuddyWidget)
        app.screen.refresh(recompose=True)
        await _wait_until(
            lambda: (
                len(list(app.screen.query(PersonaBuddyWidget))) == 1
                and app.screen.query_one(PersonaBuddyWidget) is not first
            )
        )
        assert not first.is_attached

        preferences = app.persona_buddy_controller.current_preferences()
        await app.persona_buddy_controller.update_preferences(
            replace(preferences, open=False)
        )
        app.screen.refresh(recompose=True)
        await _wait_until(lambda: not list(app.screen.query(PersonaBuddyWidget)))


@pytest.mark.asyncio
async def test_navigation_replaces_screen_local_view_generation():
    app = _BuddyApp(_enabled_preferences())
    async with app.run_test(size=(100, 30)):
        await _wait_until(lambda: len(list(app.screen.query(PersonaBuddyWidget))) == 1)
        first = app.screen.query_one(PersonaBuddyWidget)
        replacement = _BuddyScreen(app, "replacement")
        await app.switch_screen(replacement)
        await _wait_until(lambda: len(list(replacement.query(PersonaBuddyWidget))) == 1)
        second = replacement.query_one(PersonaBuddyWidget)
        assert second is not first
        assert not first.is_attached

        before = app.persona_buddy_controller.current_preferences()
        stale_before = first._working_preferences
        await first.close_and_persist()
        first.action_toggle_collapse()
        first.action_move_left()
        await asyncio.sleep(0.05)
        assert first._working_preferences == stale_before
        assert app.persona_buddy_controller.current_preferences() == before
        assert replacement.query_one(PersonaBuddyWidget) is second


@pytest.mark.asyncio
async def test_cancelled_delayed_mount_cleans_only_the_created_view(monkeypatch):
    app = _BuddyApp(PersonaBuddyPreferences())
    # This case controls/cancels the explicit reconcile caller, not the worker.
    app.persona_buddy_controller._on_change = None
    async with app.run_test(size=(100, 30)):
        screen = app.screen
        preferences = _enabled_preferences()
        await app.persona_buddy_controller.update_preferences(preferences)
        started = asyncio.Event()
        release = asyncio.Event()
        original_mount = screen.mount

        async def delayed_mount(widget, *args, **kwargs):
            started.set()
            await release.wait()
            return await original_mount(widget, *args, **kwargs)

        monkeypatch.setattr(screen, "mount", delayed_mount)
        reconcile = asyncio.create_task(app.reconcile_persona_buddy_view())
        await asyncio.wait_for(started.wait(), timeout=1)
        reconcile.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await reconcile
        assert not list(screen.query(PersonaBuddyWidget))
        assert app._persona_buddy_overlay.view is None


@pytest.mark.asyncio
async def test_delayed_mount_with_superseded_generation_removes_stale_created_view(
    monkeypatch,
):
    app = _BuddyApp(PersonaBuddyPreferences())
    app.persona_buddy_controller._on_change = None
    async with app.run_test(size=(100, 30)):
        screen = app.screen
        await app.persona_buddy_controller.update_preferences(_enabled_preferences())
        started = asyncio.Event()
        release = asyncio.Event()
        original_mount = screen.mount

        async def delayed_mount(widget, *args, **kwargs):
            started.set()
            await release.wait()
            return await original_mount(widget, *args, **kwargs)

        monkeypatch.setattr(screen, "mount", delayed_mount)
        reconcile = asyncio.create_task(app.reconcile_persona_buddy_view())
        await asyncio.wait_for(started.wait(), timeout=1)
        app._persona_buddy_overlay.generation += 1
        release.set()
        await reconcile
        assert not list(screen.query(PersonaBuddyWidget))
        assert app._persona_buddy_overlay.view is None


@pytest.mark.asyncio
async def test_confirmed_unavailable_unmounts_then_profile_invalidation_remounts():
    app = _BuddyApp(_enabled_preferences(), production_resolution=True)
    controller = app.persona_buddy_controller
    original_resolve = controller.resolve_current_visual
    gates = (asyncio.Event(), asyncio.Event())
    calls = 0

    async def blocked_resolve(*, cols: int, lines: int):
        nonlocal calls
        index = min(calls, 1)
        calls += 1
        await gates[index].wait()
        return await original_resolve(cols=cols, lines=lines)

    controller.resolve_current_visual = blocked_resolve
    async with app.run_test(size=(100, 30)):
        await _wait_until(lambda: len(list(app.screen.query(PersonaBuddyWidget))) == 1)
        await _wait_until(lambda: calls == 1)
        gates[0].set()
        await _wait_until(lambda: not list(app.screen.query(PersonaBuddyWidget)))
        current = controller.current_preferences()
        assert current.enabled and current.selection == _enabled_preferences().selection

        controller.invalidate_profile()
        await app.reconcile_persona_buddy_view()
        await _wait_until(lambda: len(list(app.screen.query(PersonaBuddyWidget))) == 1)
        await _wait_until(lambda: calls == 2)
        gates[1].set()


@pytest.mark.asyncio
async def test_confirmed_unavailable_authority_survives_fresh_screens():
    app = _BuddyApp(_enabled_preferences(), production_resolution=True)
    controller = app.persona_buddy_controller
    original_resolve = controller.resolve_current_visual
    gates = (asyncio.Event(), asyncio.Event())
    calls = 0

    async def counted_resolve(*, cols: int, lines: int):
        nonlocal calls
        index = min(calls, 1)
        calls += 1
        await gates[index].wait()
        return await original_resolve(cols=cols, lines=lines)

    controller.resolve_current_visual = counted_resolve
    async with app.run_test(size=(100, 30)):
        first = app.screen
        await _wait_until(lambda: calls == 1)
        first_view = first.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: controller.snapshot().viewport_generation >= 1)
        gates[0].set()
        await _wait_until(lambda: not list(first.query(PersonaBuddyWidget)))
        first_snapshot = controller.snapshot()

        second = _BuddyScreen(app, "second")
        await app.switch_screen(second)
        await asyncio.sleep(0.2)
        assert not list(second.query(PersonaBuddyWidget))
        assert calls == 1

        controller.invalidate_profile()
        await app.reconcile_persona_buddy_view()
        await _wait_until(lambda: calls == 2)
        await _wait_until(
            lambda: (
                controller.snapshot().viewport_generation
                > first_snapshot.viewport_generation
            )
        )
        gates[1].set()
        await _wait_until(lambda: not list(second.query(PersonaBuddyWidget)))

        current_marker = app._persona_buddy_unavailable_authority
        assert current_marker == app._persona_buddy_authority(
            controller, controller.snapshot()
        )
        assert not app.confirm_persona_buddy_unavailable(
            screen=first,
            view=first_view,
            view_generation=first_view.view_generation,
            controller=controller,
            snapshot=first_snapshot,
            visual=first_snapshot.visual,
        )
        assert app._persona_buddy_unavailable_authority == current_marker
        third = _BuddyScreen(app, "third")
        await app.switch_screen(third)
        await asyncio.sleep(0.2)
        assert not list(third.query(PersonaBuddyWidget))
        assert calls == 2


@pytest.mark.asyncio
async def test_authority_change_during_unavailable_reconcile_restarts_resolution():
    app = _BuddyApp(_enabled_preferences(), production_resolution=True)
    controller = app.persona_buddy_controller
    original_resolve = controller.resolve_current_visual
    first_resolution_release = asyncio.Event()
    second_resolution_started = asyncio.Event()
    second_resolution_release = asyncio.Event()
    reconcile_started = asyncio.Event()
    reconcile_release = asyncio.Event()
    calls = 0

    async def controlled_resolve(*, cols: int, lines: int):
        nonlocal calls
        calls += 1
        if calls == 1:
            await first_resolution_release.wait()
        else:
            second_resolution_started.set()
            await second_resolution_release.wait()
        return await original_resolve(cols=cols, lines=lines)

    original_reconcile = app.reconcile_persona_buddy_view

    async def blocked_reconcile():
        reconcile_started.set()
        await reconcile_release.wait()
        return await original_reconcile()

    controller.resolve_current_visual = controlled_resolve
    app.reconcile_persona_buddy_view = blocked_reconcile
    async with app.run_test(size=(100, 30)):
        await _wait_until(lambda: calls == 1)
        view = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: controller.snapshot().viewport_generation >= 1)
        first_resolution_release.set()
        await asyncio.wait_for(reconcile_started.wait(), timeout=1)

        controller.invalidate_profile()
        reconcile_release.set()
        await asyncio.wait_for(second_resolution_started.wait(), timeout=1)
        assert app.screen.query_one(PersonaBuddyWidget) is view
        assert view.is_attached

        second_resolution_release.set()
        await _wait_until(lambda: not list(app.screen.query(PersonaBuddyWidget)))


@pytest.mark.asyncio
async def test_app_owned_preference_write_drains_once_before_new_owner_starts():
    app = _BuddyApp(_enabled_preferences())
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked_writer(_preferences):
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=2)
        return True

    app.persona_buddy_controller._preference_writer = blocked_writer
    async with app.run_test(size=(100, 30)):
        await _wait_until(lambda: len(list(app.screen.query(PersonaBuddyWidget))) == 1)
        first = app.screen.query_one(PersonaBuddyWidget)
        first.action_close()
        await _wait_until(started.is_set)
        preference_workers = [
            worker
            for worker in app.workers
            if worker.group == "persona-buddy-preferences"
        ]
        assert len(preference_workers) == 1
        assert preference_workers[0].node is app

        replacement = _BuddyScreen(app, "replacement")
        await app.switch_screen(replacement)
        app.persona_buddy_controller.apply_preferences_patch(open=True)
        await app.reconcile_persona_buddy_view()
        await _wait_until(lambda: len(list(replacement.query(PersonaBuddyWidget))) == 1)
        replacement.query_one(PersonaBuddyWidget).action_move_left()
        await asyncio.sleep(0.1)
        assert calls == 1
        assert not first.is_attached

        release.set()
        await _wait_until(lambda: calls == 2)


@pytest.mark.asyncio
async def test_modal_dismiss_reconciles_same_screen_and_restarts_resolution():
    app = _BuddyApp(_enabled_preferences())
    controller = app.persona_buddy_controller
    calls = 0

    async def counted_resolution(*, cols: int, lines: int):
        nonlocal calls
        calls += 1
        return None

    controller.resolve_current_visual = counted_resolution
    async with app.run_test(size=(100, 30)):
        await _wait_until(lambda: calls == 1)
        screen = app.screen
        view = screen.query_one(PersonaBuddyWidget)

        await app.push_screen(_BuddyModal())
        await _wait_until(lambda: view._resolution_worker.is_finished)
        controller.invalidate_profile()
        app.pop_screen()

        await _wait_until(lambda: calls == 2)
        assert app.screen is screen
        assert screen.query_one(PersonaBuddyWidget) is view


@pytest.mark.asyncio
async def test_modal_dismiss_replays_close_and_removes_same_screen_view():
    app = _BuddyApp(_enabled_preferences())
    async with app.run_test(size=(100, 30)):
        screen = app.screen
        await _wait_until(lambda: len(list(screen.query(PersonaBuddyWidget))) == 1)
        await app.push_screen(_BuddyModal())
        current = app.persona_buddy_controller.current_preferences()
        await app.persona_buddy_controller.update_preferences(
            replace(current, open=False)
        )

        app.pop_screen()

        await _wait_until(lambda: not list(screen.query(PersonaBuddyWidget)))
        assert app.screen is screen


@pytest.mark.asyncio
async def test_modal_dismiss_replays_collapsed_then_disabled_preferences():
    app = _BuddyApp(_enabled_preferences())
    async with app.run_test(size=(100, 30)):
        screen = app.screen
        view = screen.query_one(PersonaBuddyWidget)
        await app.push_screen(_BuddyModal())
        current = app.persona_buddy_controller.current_preferences()
        await app.persona_buddy_controller.update_preferences(
            replace(current, collapsed=True)
        )

        app.pop_screen()
        await _wait_until(lambda: view.has_class("persona-buddy-collapsed"))
        assert screen.query_one(PersonaBuddyWidget) is view

        await app.push_screen(_BuddyModal())
        current = app.persona_buddy_controller.current_preferences()
        await app.persona_buddy_controller.update_preferences(
            replace(current, enabled=False)
        )

        app.pop_screen()
        await _wait_until(lambda: not list(screen.query(PersonaBuddyWidget)))
        assert app.screen is screen


@pytest.mark.asyncio
async def test_blocked_close_and_stale_geometry_merge_immediately_and_durably():
    app = _BuddyApp(_enabled_preferences())
    entered = threading.Event()
    release = threading.Event()
    persisted: list[PersonaBuddyPreferences] = []

    def blocked_writer(preferences: PersonaBuddyPreferences) -> bool:
        if not persisted:
            entered.set()
            release.wait(timeout=2)
        persisted.append(preferences)
        return True

    app.persona_buddy_controller._preference_writer = blocked_writer
    async with app.run_test(size=(100, 30)):
        view = app.screen.query_one(PersonaBuddyWidget)
        before = view._clamped_geometry(
            app.persona_buddy_controller.current_preferences().geometry
        )

        # Keep the view current while two admitted edits race with persistence.
        # Once the owner retires a closed view, input must correctly be ignored.
        async with app._persona_buddy_overlay._lock:
            view.action_close()
            assert app.persona_buddy_controller.current_preferences().open is False
            await _wait_until(entered.is_set)
            view.action_move_left()

        current = app.persona_buddy_controller.current_preferences()
        assert current.open is False
        assert current.geometry.x == max(0, before.x - 1)
        release.set()
        await _wait_until(lambda: len(persisted) == 2)

        final = app.persona_buddy_controller.current_preferences()
        assert persisted[-1] == final
        assert final.open is False
        assert final.geometry == current.geometry


@pytest.mark.asyncio
async def test_change_burst_keeps_one_inflight_mount_worker(monkeypatch):
    app = _BuddyApp(PersonaBuddyPreferences())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entered = asyncio.Event()
        release = asyncio.Event()
        original_mount = app.screen.mount
        mounts = []

        async def blocked_mount(view):
            mounts.append(view)
            entered.set()
            await release.wait()
            return await original_mount(view)

        monkeypatch.setattr(app.screen, "mount", blocked_mount)
        app.persona_buddy_controller.apply_preferences_patch(
            enabled=True, selection=PersonaBuddySelection("local", "persona-1")
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        owner = app._persona_buddy_overlay
        worker = owner._worker
        for _ in range(20):
            app.persona_buddy_controller.invalidate_profile()
        await pilot.pause()
        assert owner._worker is worker
        assert not worker.is_cancelled
        release.set()
        await _wait_until(lambda: worker.is_finished)
        assert len(mounts) == 1
        assert owner.is_current(owner.view)


@pytest.mark.asyncio
async def test_disable_behind_modal_removes_retained_view():
    app = _BuddyApp(_enabled_preferences())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        owner = app._persona_buddy_overlay
        previous = owner.view
        await app.push_screen(_BuddyModal())
        app.persona_buddy_controller.apply_preferences_patch(enabled=False)
        await _wait_until(lambda: owner.view is None)
        assert not previous.is_attached
        assert not list(app.screen.query(PersonaBuddyWidget))
        await app.pop_screen()
        await pilot.pause()
        assert not list(app.screen.query(PersonaBuddyWidget))


@pytest.mark.asyncio
async def test_cancelled_retirement_replaces_invalidated_view_on_reenable(monkeypatch):
    app = _BuddyApp(_enabled_preferences())
    app.persona_buddy_controller._on_change = None
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        owner = app._persona_buddy_overlay
        previous = owner.view
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked_flush():
            entered.set()
            await release.wait()

        monkeypatch.setattr(previous, "flush_pending_geometry_persist", blocked_flush)
        app.persona_buddy_controller.apply_preferences_patch(enabled=False)
        retire = asyncio.create_task(app.reconcile_persona_buddy_view())
        await asyncio.wait_for(entered.wait(), timeout=2)
        retire.cancel()
        with pytest.raises(asyncio.CancelledError):
            await retire
        release.set()
        app.persona_buddy_controller.apply_preferences_patch(enabled=True)
        await app.reconcile_persona_buddy_view()
        assert owner.is_current(owner.view)
        assert owner.view is not previous
        assert not previous.is_attached


@pytest.mark.asyncio
async def test_shutdown_during_retirement_admits_no_replacement_mount(monkeypatch):
    app = _BuddyApp(_enabled_preferences())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        owner = app._persona_buddy_overlay
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked_flush():
            entered.set()
            await release.wait()

        monkeypatch.setattr(owner.view, "flush_pending_geometry_persist", blocked_flush)
        replacement = _BuddyScreen(app, "shutdown-replacement")
        mounts = []
        original_mount = replacement.mount

        def record_mount(*widgets, **kwargs):
            mounts.extend(
                widget for widget in widgets if isinstance(widget, PersonaBuddyWidget)
            )
            return original_mount(*widgets, **kwargs)

        monkeypatch.setattr(replacement, "mount", record_mount)
        await app.push_screen(replacement)
        await asyncio.wait_for(entered.wait(), timeout=2)
        shutdown = asyncio.create_task(owner.shutdown())
        try:
            await _wait_until(lambda: owner.closed)
        finally:
            release.set()
        await asyncio.wait_for(shutdown, timeout=2)
        assert mounts == []
        assert owner.view is None


class _ShutdownOrderBuddyApp(_BuddyApp):
    """Reproduce `TldwCli`'s exit order: Buddy drain, then screen teardown.

    `TldwCli._shutdown` runs `_shutdown_app_owned_lifecycles()` -- which ends
    the Buddy controller -- BEFORE `super()._shutdown()` closes screens, so
    the widget's own `on_unmount` runs against a controller that has already
    closed admission. A fake controller has no shutdown gate and structurally
    cannot show this (TASK-21122 review, MAJOR-1).
    """

    _shutdown_persona_buddy = TldwCli._shutdown_persona_buddy
    _flush_persona_buddy_geometry = getattr(
        TldwCli, "_flush_persona_buddy_geometry", None
    )

    def __init__(self, preferences: PersonaBuddyPreferences) -> None:
        super().__init__(preferences)
        # `_shutdown_persona_buddy` peeks the lazy slot, not the property.
        self._persona_buddy_controller = self.persona_buddy_controller
        self._persona_buddy_shutdown_task = None

    async def _shutdown(self) -> None:
        await self._shutdown_persona_buddy()
        await super()._shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("quit_after", [0.05, 0.15, 0.40])
async def test_debounced_geometry_survives_the_real_shutdown_order(quit_after: float):
    """TASK-21122: quitting inside the debounce window must not lose the nudge.

    0.05 s and 0.15 s land inside the 250 ms debounce (the window that used to
    drop the write and raise `WorkerFailed`); 0.40 s is past it, where the
    timer has already fired. All three must be durable.
    """

    app = _ShutdownOrderBuddyApp(_enabled_preferences())
    persisted: list[PersonaBuddyPreferences] = []

    def recording_writer(preferences: PersonaBuddyPreferences) -> bool:
        persisted.append(preferences)
        return True

    controller = app.persona_buddy_controller
    controller._preference_writer = recording_writer
    async with app.run_test(size=(100, 30)):
        view = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: view.region.width > 0)
        persisted.clear()
        view.action_move_left()
        expected = controller.current_preferences().geometry
        assert persisted == []
        await asyncio.sleep(quit_after)

    assert app._exception is None, app._exception
    assert persisted, "debounced geometry write was lost at shutdown"
    assert persisted[-1].geometry == expected


@pytest.mark.asyncio
async def test_unmount_after_controller_shutdown_raises_no_worker_error():
    """A closed controller must not turn the unmount flush into a traceback."""

    app = _ShutdownOrderBuddyApp(_enabled_preferences())
    controller = app.persona_buddy_controller
    controller._preference_writer = lambda _preferences: True
    async with app.run_test(size=(100, 30)) as pilot:
        view = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: view.region.width > 0)
        # Close admission first, THEN arm a debounce and unmount: any persist
        # attempt from here raises `persona_buddy_shutdown`.
        await controller.shutdown()
        view.action_move_left()
        await view.remove()
        await pilot.pause()

    assert app._exception is None, app._exception
