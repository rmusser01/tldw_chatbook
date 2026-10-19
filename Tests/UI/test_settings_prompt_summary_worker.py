"""Prompt summaries must not block input or publish obsolete checked reads."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.screen import Screen
from textual.widgets import Static

from tldw_chatbook.UI.Screens import settings_screen as module

pytestmark = pytest.mark.bootstrap_profile


class SummaryScreen(Screen):
    """Mount the real summary lifecycle without unrelated Settings services."""

    _get_internal_prompts_customized_count = (
        module.SettingsScreen._get_internal_prompts_customized_count
    )
    _internal_prompts_status = module.SettingsScreen._internal_prompts_status
    _request_internal_prompts_count = (
        module.SettingsScreen._request_internal_prompts_count
    )
    _prepare_internal_prompts_count = (
        module.SettingsScreen._prepare_internal_prompts_count
    )
    _apply_internal_prompts_count = module.SettingsScreen._apply_internal_prompts_count
    _on_internal_prompts_modified = module.SettingsScreen._on_internal_prompts_modified

    def __init__(self):
        Screen.__init__(self)
        self._internal_prompts_customized_count = None
        self._internal_prompts_count_identity = None
        self._internal_prompts_count_revision = 0
        self._internal_prompts_count_pending = False
        self._internal_prompts_count_failed = False

    def compose(self):
        yield Static(self._internal_prompts_status(), id="summary")

    def on_mount(self):
        self._request_internal_prompts_count()

    def on_screen_resume(self):
        pass

    def on_unmount(self):
        pass

    def _refresh_internal_prompts_customized_widgets(self):
        self.query_one("#summary", Static).update(self._internal_prompts_status())


class SummaryApp(App):
    def on_mount(self):
        self.push_screen(SummaryScreen())


async def wait_for(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_original_checked_count_publishes_from_the_worker():
    expected = await asyncio.to_thread(
        module.internal_prompts_authoring.customized_count
    )
    async with SummaryApp().run_test() as pilot:
        screen = pilot.app.screen
        await wait_for(lambda: screen._internal_prompts_customized_count is not None)
        assert screen._internal_prompts_customized_count == expected
        assert not screen._internal_prompts_count_failed


@pytest.mark.asyncio
async def test_blocked_count_keeps_loop_responsive_and_publishes_after_release(
    monkeypatch,
):
    started, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    observed = []

    def count():
        observed.append(threading.get_ident())
        started.set()
        assert release.wait(3)
        return 7

    monkeypatch.setattr(module.internal_prompts_authoring, "customized_count", count)
    try:
        async with SummaryApp().run_test() as pilot:
            await wait_for(started.is_set)
            screen = pilot.app.screen
            assert observed == [observed[0]] and observed[0] != loop_thread
            assert screen._get_internal_prompts_customized_count() is None
            assert screen._internal_prompts_status() != "Defaults"
            release.set()
            await wait_for(lambda: screen._internal_prompts_customized_count == 7)
            assert (
                str(screen.query_one("#summary", Static).renderable) == "7 customized"
            )
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["edit", "config", "unmount", "cancel", "failure"])
async def test_late_count_cannot_replace_newer_or_unavailable_summary(
    monkeypatch, change
):
    started, release = threading.Event(), threading.Event()
    identity = [(1, "private-config")]
    calls = []

    def count():
        calls.append(1)
        if len(calls) == 1:
            started.set()
            assert release.wait(3)
            if change == "failure":
                raise ValueError("synthetic read failure")
            return 1
        return 9

    monkeypatch.setattr(
        module, "current_config_identity", lambda: identity[0], raising=False
    )
    monkeypatch.setattr(module.internal_prompts_authoring, "customized_count", count)
    try:
        async with SummaryApp().run_test() as pilot:
            await wait_for(started.is_set)
            screen = pilot.app.screen
            if change == "edit":
                screen._on_internal_prompts_modified(
                    SimpleNamespace(customized_count=8)
                )
            elif change == "config":
                identity[0] = (2, "replacement-config")
            elif change == "unmount":
                await pilot.app.pop_screen()
            elif change == "cancel":
                screen.workers.cancel_group(screen, "settings-prompt-summary")
            release.set()
            if change == "config":
                await wait_for(lambda: screen._internal_prompts_customized_count == 9)
            elif change == "failure":
                await wait_for(lambda: screen._internal_prompts_count_failed)
                assert screen._internal_prompts_status() == "Unavailable"
            else:
                await pilot.pause(0.1)
                assert screen._internal_prompts_customized_count == (
                    8 if change == "edit" else None
                )
    finally:
        release.set()
