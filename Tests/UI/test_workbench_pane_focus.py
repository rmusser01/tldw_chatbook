"""Shell-wide workbench pane focus convention tests."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Input, Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.workbench_focus import (
    WorkbenchPaneTarget,
    focus_relative_workbench_pane,
)


class _ConsoleHarness(App[None]):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


class _PersonasHarness(App[None]):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    def compose(self):
        yield AppFooterStatus(id="app-footer-status")

    async def on_mount(self) -> None:
        await self.push_screen(PersonasScreen(self.app_instance))


class _FocusFallbackScreen(Screen[None]):
    def compose(self):
        with Vertical(id="fallback-pane"):
            yield Static("Passive target", id="passive-target")
            yield Input(id="focusable-target")


class _FocusFallbackHarness(App[None]):
    async def on_mount(self) -> None:
        await self.push_screen(_FocusFallbackScreen())


def _mark_console_onboarding_complete(app) -> None:
    app.app_config = getattr(app, "app_config", {}) or {}
    console_config = app.app_config.setdefault("console", {})
    onboarding = console_config.setdefault("onboarding", {})
    onboarding["first_send_completed"] = True


async def _wait_for_focused_id(app: App[None], pilot, widget_id: str) -> None:
    for _ in range(40):
        if getattr(app.focused, "id", None) == widget_id:
            return
        await pilot.pause(0.05)
    raise AssertionError(
        f"Expected focus on {widget_id!r}, found {getattr(app.focused, 'id', None)!r}"
    )


@pytest.mark.asyncio
async def test_console_f6_cycles_between_workbench_panes_and_wraps_backward():
    app_instance = _build_test_app()
    _mark_console_onboarding_complete(app_instance)
    host = _ConsoleHarness(app_instance)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console._set_console_rail_preference(
            left_open=True,
            right_open=True,
            notify_on_failure=False,
        )
        await pilot.pause()
        console.query_one("#console-native-composer").focus()

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "console-context-rail-collapse")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "console-native-transcript")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "console-inspector-rail-collapse")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "console-native-composer")

        await pilot.press("shift+f6")
        await _wait_for_focused_id(host, pilot, "console-inspector-rail-collapse")


@pytest.mark.asyncio
async def test_personas_f6_cycles_between_workbench_panes_from_text_input():
    app_instance = _build_test_app()
    host = _PersonasHarness(app_instance)

    async with host.run_test(size=(140, 42)) as pilot:
        personas = host.screen_stack[-1]
        await _wait_for_selector(personas, pilot, "#personas-library-search")
        personas.query_one("#personas-library-search").focus()

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "personas-preview-toggle")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "personas-conversations-list")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "personas-library-search")

        await pilot.press("shift+f6")
        await _wait_for_focused_id(host, pilot, "personas-conversations-list")


def test_workbench_screens_expose_f6_bindings_without_ctrl_arrow_conflicts():
    screen_classes = (ChatScreen, PersonasScreen)
    for screen_class in screen_classes:
        bindings = getattr(screen_class, "BINDINGS", ())
        keys = {binding[0] if isinstance(binding, tuple) else binding.key for binding in bindings}
        assert "f6" in keys
        assert "shift+f6" in keys
        assert "ctrl+left" not in keys
        assert "ctrl+right" not in keys


@pytest.mark.asyncio
async def test_workbench_focus_skips_missing_and_non_focusable_preferred_targets():
    host = _FocusFallbackHarness()

    async with host.run_test(size=(80, 20)) as pilot:
        screen = host.screen_stack[-1]

        focused = focus_relative_workbench_pane(
            screen,
            (
                WorkbenchPaneTarget(
                    "fallback-pane",
                    ("missing-target", "passive-target", "focusable-target"),
                ),
            ),
            direction=1,
        )

        assert getattr(focused, "id", None) == "focusable-target"
        await _wait_for_focused_id(host, pilot, "focusable-target")
