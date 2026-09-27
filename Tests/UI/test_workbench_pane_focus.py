"""Shell-wide workbench pane focus convention tests."""

from __future__ import annotations

import pytest
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
import tldw_chatbook.app as app_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.workbench_focus import (
    WorkbenchPaneTarget,
    focus_relative_workbench_pane,
)


class _FocusFallbackScreen(Screen[None]):
    def compose(self):
        with Vertical(id="fallback-pane"):
            yield Static("Passive target", id="passive-target")
            yield Input(id="focusable-target")


@pytest.fixture(autouse=True)
def _disable_full_app_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


def _mark_console_onboarding_complete(app) -> None:
    app.app_config = getattr(app, "app_config", {}) or {}
    console_config = app.app_config.setdefault("console", {})
    onboarding = console_config.setdefault("onboarding", {})
    onboarding["first_send_completed"] = True


async def _wait_for_focused_id(app, pilot, widget_id: str) -> None:
    for _ in range(40):
        if getattr(app.focused, "id", None) == widget_id:
            return
        await pilot.pause(0.05)
    raise AssertionError(
        f"Expected focus on {widget_id!r}, found {getattr(app.focused, 'id', None)!r}"
    )


@pytest.mark.asyncio
async def test_console_f6_cycles_wraps_backward_and_targets_expand_when_collapsed():
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "chat"
    _mark_console_onboarding_complete(app)

    async with app.run_test(size=(160, 48)) as pilot:
        console = app.screen
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console._set_console_rail_preference(
            left_open=True,
            right_open=True,
            notify_on_failure=False,
        )
        await pilot.pause()
        console.query_one("#console-native-composer").focus()

        # TASK-32321: F6 now lands each rail on its first CONTENT control,
        # not the collapse button (a reflexive Enter used to hide the pane
        # the user just entered).
        await pilot.press("f6")
        await _wait_for_focused_id(app, pilot, "console-terminal-open")

        await pilot.press("f6")
        await _wait_for_focused_id(app, pilot, "console-native-transcript")

        await pilot.press("f6")
        await _wait_for_focused_id(app, pilot, "console-send-authority-summary")

        await pilot.press("f6")
        await _wait_for_focused_id(app, pilot, "console-native-composer")

        await pilot.press("shift+f6")
        await _wait_for_focused_id(app, pilot, "console-send-authority-summary")

        console._set_console_composer_collapsed(True)
        await pilot.pause()
        console.query_one("#console-inspector-rail-collapse").focus()
        await pilot.press("f6")
        await _wait_for_focused_id(app, pilot, "console-composer-expand")
        assert console.query_one("#console-native-composer").can_focus is False
        console._ensure_console_workbench_targets_focusable()
        assert console.query_one("#console-native-composer").can_focus is False


@pytest.mark.asyncio
async def test_personas_f6_cycles_between_workbench_panes_from_text_input():
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "personas"

    async with app.run_test(size=(140, 42)) as pilot:
        personas = app.screen
        await _wait_for_selector(personas, pilot, "#personas-library-search")
        personas.query_one("#personas-library-search").focus()

        await pilot.press("f6")
        await _wait_for_focused_id(app, pilot, "personas-preview-toggle")

        await pilot.press("f6")
        await _wait_for_focused_id(app, pilot, "personas-conversations-list")

        await pilot.press("f6")
        await _wait_for_focused_id(app, pilot, "personas-library-search")

        await pilot.press("shift+f6")
        await _wait_for_focused_id(app, pilot, "personas-conversations-list")


_SETTINGS_PANES = (
    "settings-category-pane",
    "settings-detail-pane",
    "settings-impact-pane",
)


def _settings_pane_holding_focus(app) -> str | None:
    focused = app.focused
    if focused is None:
        return None
    for ancestor in focused.ancestors_with_self:
        if getattr(ancestor, "id", None) in _SETTINGS_PANES:
            return ancestor.id
    return None


async def _wait_for_pane(app, pilot, pane_id: str, what: str) -> None:
    for _ in range(40):
        if _settings_pane_holding_focus(app) == pane_id:
            return
        await pilot.pause(0.05)
    raise AssertionError(
        f"{what}: expected focus in {pane_id!r}, found "
        f"{getattr(app.focused, 'id', None)!r} in "
        f"{_settings_pane_holding_focus(app)!r}"
    )


async def _press_and_wait_for_pane(app, pilot, key: str, pane_id: str) -> None:
    await pilot.press(key)
    await _wait_for_pane(app, pilot, pane_id, key)


@pytest.mark.asyncio
@private_profile_test
async def test_settings_f6_and_shift_f6_cycle_rail_detail_and_inspector(request):
    """TASK-33001.4: F6/Shift+F6 cycle the three Settings panes.

    Real key presses through the production app, so F6 travels the shipped
    route (the app-global binding delegating to the screen) and the app's
    "No workbench pane focus target" fallback would be seen if it fired.
    """
    app = _build_test_app(configured_default="settings")
    notices: list[str] = []
    real_notify = app.notify

    def record_notify(message, *args, **kwargs):
        notices.append(str(message))
        return real_notify(message, *args, **kwargs)

    app.notify = record_notify

    async with app.run_test(size=(211, 44)) as pilot:
        for _ in range(200):
            if isinstance(app.screen, SettingsScreen):
                break
            await pilot.pause(0.01)
        settings = app.screen
        assert isinstance(settings, SettingsScreen)
        await _wait_for_selector(settings, pilot, "#settings-impact-pane-body")
        first_category = settings.active_category

        # F6 from outside every pane enters the rail on the ACTIVE row.
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-category-pane")
        assert app.focused.id == f"settings-category-{first_category}"
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-detail-pane")
        # A control, not the pane's scroll body.
        assert app.focused.id != "settings-detail-pane-body"
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-impact-pane")
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-category-pane")
        assert app.focused.id == f"settings-category-{first_category}"

        # Shift+F6 walks the same ring backwards, wrapping from the rail.
        await _press_and_wait_for_pane(app, pilot, "shift+f6", "settings-impact-pane")
        await _press_and_wait_for_pane(app, pilot, "shift+f6", "settings-detail-pane")
        await _press_and_wait_for_pane(app, pilot, "shift+f6", "settings-category-pane")

        # Switch category with the rail keys, then leave the rail: F6 back
        # into it must land on the NEW active row, not the old one.
        for _ in range(40):
            if app.focused.id == "settings-category-appearance":
                break
            await pilot.press("down")
            await pilot.pause()
        assert app.focused.id == "settings-category-appearance"
        await pilot.press("enter")
        await _wait_for_selector(settings, pilot, "#settings-appearance-font-size")
        for _ in range(40):
            if (
                settings.active_category == "appearance"
                and app.focused.id == "settings-category-appearance"
                and not settings._category_pane_swap_pending
            ):
                break
            await pilot.pause(0.05)
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-detail-pane")

        # F6 from a focused text field moves on and leaves its value alone.
        field = settings.query_one("#settings-appearance-font-size", Input)
        field.focus()
        await pilot.pause()
        before = field.value
        assert before
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-impact-pane")
        assert field.value == before
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-category-pane")
        assert app.focused.id == "settings-category-appearance"

        # A filter that hides the active row: the rail stop is the filter.
        await pilot.press("slash", *"privacy")
        for _ in range(40):
            if not settings.query_one("#settings-category-appearance").display:
                break
            await pilot.pause(0.05)
        assert not settings.query_one("#settings-category-appearance").display
        assert settings.active_category == "appearance"
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-detail-pane")
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-impact-pane")
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-category-pane")
        assert app.focused.id == "settings-category-search"

    assert "No workbench pane focus target is available." not in notices


@pytest.mark.asyncio
@private_profile_test
async def test_settings_f6_pressed_mid_category_swap_lands_in_the_new_panes(request):
    """F6/Shift+F6 pressed while a category switch is still swapping panes.

    Until the swap runs, the detail and inspector panes hold the OUTGOING
    category's widgets (the TASK-2831 window), and the swap ends by focusing
    the new rail row. The swap lock is held so each key lands in that window
    on every run, not by timing luck.
    """
    app = _build_test_app(configured_default="settings")
    notices: list[str] = []
    real_notify = app.notify

    def record_notify(message, *args, **kwargs):
        notices.append(str(message))
        return real_notify(message, *args, **kwargs)

    app.notify = record_notify

    async with app.run_test(size=(211, 44)) as pilot:
        for _ in range(200):
            if isinstance(app.screen, SettingsScreen):
                break
            await pilot.pause(0.01)
        settings = app.screen
        assert isinstance(settings, SettingsScreen)
        await _wait_for_selector(settings, pilot, "#settings-impact-pane-body")
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-category-pane")

        async def switch_with_key_mid_swap(key: str) -> list:
            """Enter on the focused rail row, then ``key`` before the swap."""
            await settings._category_swap_lock.acquire()
            try:
                await pilot.press("enter")
                assert settings._category_pane_swap_pending
                outgoing = list(
                    settings.query("#settings-detail-pane *, #settings-impact-pane *")
                )
                await pilot.press(key)
            finally:
                settings._category_swap_lock.release()
            for _ in range(40):
                if not settings._category_pane_swap_pending:
                    break
                await pilot.pause(0.05)
            assert not settings._category_pane_swap_pending
            return outgoing

        for _ in range(40):
            if app.focused.id == "settings-category-appearance":
                break
            await pilot.press("down")
        outgoing = await switch_with_key_mid_swap("f6")
        await _wait_for_pane(app, pilot, "settings-detail-pane", "f6 mid-swap")
        assert settings.active_category == "appearance"
        assert app.focused.is_attached and app.focused not in outgoing
        assert app.focused.id != "settings-detail-pane-body"

        await _press_and_wait_for_pane(app, pilot, "shift+f6", "settings-category-pane")
        await pilot.press("down")
        assert app.focused.id == "settings-category-theme"
        outgoing = await switch_with_key_mid_swap("shift+f6")
        await _wait_for_pane(app, pilot, "settings-impact-pane", "shift+f6 mid-swap")
        assert settings.active_category == "theme"
        assert app.focused.is_attached and app.focused not in outgoing
        # Theme's inspector holds no control, so the stop is the inspector's
        # own scroll body (the keys then scroll it); dropping scroll bodies
        # from the ring instead of sorting them last would skip the pane.
        inspector = settings.query_one("#settings-impact-pane")
        assert [
            widget.id
            for widget in settings.focus_chain
            if inspector in widget.ancestors
        ] == ["settings-impact-pane-body"]
        assert app.focused.id == "settings-impact-pane-body"
        await _press_and_wait_for_pane(app, pilot, "f6", "settings-category-pane")
        await _press_and_wait_for_pane(app, pilot, "shift+f6", "settings-impact-pane")
        assert app.focused.id == "settings-impact-pane-body"

    assert "No workbench pane focus target is available." not in notices


def test_settings_binds_only_shift_f6_and_leaves_f6_to_the_app():
    """TASK-33001.4 + task-32943: F6 is app-global (ADR-031 rule 1) and
    reaches Settings through ``action_focus_next_workbench_pane``; a screen
    ``f6`` binding would shadow it. Exactly one ``shift+f6`` -- the merge of
    both F6 implementations once auto-merged two identical entries."""
    keys = [
        binding[0] if isinstance(binding, tuple) else binding.key
        for binding in SettingsScreen.BINDINGS
    ]
    assert keys.count("shift+f6") == 1
    assert "f6" not in keys
    assert "ctrl+left" not in keys
    assert "ctrl+right" not in keys


def test_workbench_screens_expose_f6_bindings_without_ctrl_arrow_conflicts():
    screen_classes = (ChatScreen, PersonasScreen)
    for screen_class in screen_classes:
        bindings = getattr(screen_class, "BINDINGS", ())
        keys = {
            binding[0] if isinstance(binding, tuple) else binding.key
            for binding in bindings
        }
        assert "f6" in keys
        assert "shift+f6" in keys
        assert "ctrl+left" not in keys
        assert "ctrl+right" not in keys


@pytest.mark.asyncio
async def test_workbench_focus_skips_missing_and_non_focusable_preferred_targets():
    app = _build_test_app()
    app.app_config["_first_run"] = False

    async with app.run_test(size=(80, 20)) as pilot:
        app.push_screen(_FocusFallbackScreen())
        await pilot.pause()
        screen = app.screen

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
        await _wait_for_focused_id(app, pilot, "focusable-target")
