"""Dynamic app-owned mount coverage for the floating Persona Buddy."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

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


class _BuddyApp(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET
    reconcile_persona_buddy_view = TldwCli.reconcile_persona_buddy_view

    def __init__(self, preferences: PersonaBuddyPreferences) -> None:
        super().__init__()
        self.persona_buddy_controller = PersonaBuddyController(
            preferences=preferences,
            preference_writer=lambda _preferences: True,
        )
        self.initial_screen = _BuddyScreen(self)

    async def on_mount(self) -> None:
        await self.push_screen(self.initial_screen)


def _enabled_preferences() -> PersonaBuddyPreferences:
    return PersonaBuddyPreferences(
        enabled=True,
        selection=PersonaBuddySelection("local", "persona-1"),
    )


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
        await app.screen.reconcile_persona_buddy_view()
        current = app.screen.query_one(PersonaBuddyWidget)
        assert current is stale

        await stale.close_and_persist()
        await _wait_until(lambda: not list(app.screen.query(PersonaBuddyWidget)))
        assert stale.view_generation < app.screen.persona_buddy_view_generation


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
