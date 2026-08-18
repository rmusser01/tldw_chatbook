"""Settings contracts for the collapsed Console rail label preference."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Checkbox, Input, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _wait_for_selector,
    _visible_text,
)
from Tests.UI.test_settings_configuration_hub import (
    _open_settings_category,
    _settle_settings_mount_storm,
    _wait_for_settings_search_focus,
    _wait_for_settings_text,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.Widgets.Console.console_rail_handle import ConsoleRailHandle


RAIL_LABEL_TOGGLE = "#settings-console-stack-collapsed-rail-labels"


@pytest.mark.asyncio
async def test_console_rail_label_setting_carries_state_and_stages_from_keyboard():
    """Space changes the draft text without mutating the active runtime config."""
    app = _build_test_app()
    app.app_config["console"] = {"stack_collapsed_rail_labels": False}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        toggle = screen.query_one(RAIL_LABEL_TOGGLE, Checkbox)

        assert str(toggle.label) == "Stack collapsed rail labels"
        assert toggle.value is False
        assert "Saved style: Horizontal" in _visible_text(screen)

        paste_toggle = screen.query_one(
            "#settings-console-collapse-large-pastes-toggle", Checkbox
        )
        # task-17652: the status-row placement toggle sits between the rail
        # checkbox and the paste checkbox in focus order.
        position_toggle = screen.query_one(
            "#settings-console-status-row-position-toggle", Button
        )
        paste_toggle.focus()
        await pilot.press("shift+tab")
        assert host.focused is position_toggle
        await pilot.press("shift+tab")
        assert host.focused is toggle
        await pilot.press("tab")
        await pilot.press("tab")
        assert host.focused is paste_toggle

        toggle.focus()
        await pilot.press("space")
        await pilot.pause()

        assert host.focused is toggle
        assert toggle.value is True
        assert "Selected style: Stacked — unsaved" in _visible_text(screen)
        assert app.app_config["console"]["stack_collapsed_rail_labels"] is False


@pytest.mark.asyncio
async def test_console_rail_label_setting_is_searchable_and_has_focused_guidance():
    """The vertical alias lands on the checkbox and exposes its config contract."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        await pilot.press("/")
        await _wait_for_settings_search_focus(screen, pilot)
        search = screen.query_one("#settings-category-search", Input)
        assert search.has_focus
        await pilot.press(*"vertical")
        await _wait_for_settings_text(
            screen,
            pilot,
            "Console Behavior › Stacked vertical Context Inspector",
        )
        await pilot.press("enter")
        for _ in range(8):
            await pilot.pause()

        assert screen.active_category == SettingsCategoryId.CONSOLE_BEHAVIOR.value
        assert host.focused is not None and host.focused.id == (
            "settings-console-stack-collapsed-rail-labels"
        )
        visible = _visible_text(screen)
        assert "Purpose: Choose the collapsed Console rail label style." in visible
        assert "Consequences: Stacked uses narrower 3-column handles" in visible
        assert "Horizontal uses the established 13- and 11-column handles." in visible
        assert "Saved as: console." in visible
        assert "stack_collapsed_rail_labels" in visible
        assert "Applies: After saving, when Console is next opened." in visible
        assert "Save: staged - press s to save, r to revert" in visible


@pytest.mark.asyncio
async def test_console_rail_label_setting_saves_exact_payload_and_runtime_value(
    monkeypatch,
):
    """Successful category Save persists once, then activates the saved style."""
    app = _build_test_app()
    app.app_config["console"] = {"stack_collapsed_rail_labels": False}
    saved: list[dict[str, dict[str, bool]]] = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        await pilot.click(RAIL_LABEL_TOGGLE)

        assert app.app_config["console"]["stack_collapsed_rail_labels"] is False
        await pilot.click("#settings-save-category")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert saved == [{"console": {"stack_collapsed_rail_labels": True}}]
        assert app.app_config["console"]["stack_collapsed_rail_labels"] is True
        assert "Rail labels: Stacked" in _visible_text(screen)
        assert "Saved style: Stacked" in _visible_text(screen)

    console_host = ConsoleHarness(app)
    async with console_host.run_test(size=(160, 45)) as pilot:
        console = console_host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        left = console.query_one("#console-context-rail-handle", ConsoleRailHandle)
        right = console.query_one("#console-inspector-rail-handle", ConsoleRailHandle)
        assert left.styles.width.value == ConsoleRailHandle.VERTICAL_WIDTH
        assert right.styles.width.value == ConsoleRailHandle.VERTICAL_WIDTH
        assert left._display_label() == "C\no\nn\nt\ne\nx\nt"
        assert right._display_label() == "I\nn\ns\np\ne\nc\nt\no\nr"
        assert (
            console.query_one("#console-context-rail-open", Button).tooltip
            == "Open Context rail"
        )
        assert (
            console.query_one("#console-inspector-rail-open", Button).tooltip
            == "Open Inspector rail"
        )


@pytest.mark.asyncio
async def test_console_rail_label_failed_save_keeps_draft_and_active_style(
    monkeypatch,
):
    """Persistence failure keeps the selected value without activating it."""
    app = _build_test_app()
    app.app_config["console"] = {"stack_collapsed_rail_labels": False}

    class FailingAdapter:
        def save_sections(self, section_values):
            return False

    monkeypatch.setattr(
        settings_screen_module,
        "SettingsConfigAdapter",
        FailingAdapter,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        await pilot.click(RAIL_LABEL_TOGGLE)
        await pilot.click("#settings-save-category")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert screen.query_one(RAIL_LABEL_TOGGLE, Checkbox).value is True
        assert app.app_config["console"]["stack_collapsed_rail_labels"] is False
        visible = _visible_text(screen)
        assert "Your draft is still here" in visible
        assert "active rail-label style is still Horizontal" in visible

    console_host = ConsoleHarness(app)
    async with console_host.run_test(size=(160, 45)) as pilot:
        console = console_host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        left = console.query_one("#console-context-rail-handle", ConsoleRailHandle)
        right = console.query_one("#console-inspector-rail-handle", ConsoleRailHandle)
        assert left.styles.width.value == 13
        assert right.styles.width.value == 11
        assert left._display_label() == "Context->"
        assert right._display_label() == "<-Inspect"


@pytest.mark.asyncio
async def test_console_rail_label_revert_discards_every_console_behavior_draft():
    """Category Revert restores the rail and paste controls together."""
    app = _build_test_app()
    app.app_config["console"] = {
        "stack_collapsed_rail_labels": False,
        "collapse_large_pastes": True,
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        await pilot.click(RAIL_LABEL_TOGGLE)
        await pilot.click("#settings-console-collapse-large-pastes-toggle")

        draft = screen._settings_drafts[SettingsCategoryId.CONSOLE_BEHAVIOR]
        assert draft.dirty_keys == {
            "collapse_large_pastes",
            "stack_collapsed_rail_labels",
        }

        screen._revert_category(SettingsCategoryId.CONSOLE_BEHAVIOR)
        await pilot.pause()

        assert SettingsCategoryId.CONSOLE_BEHAVIOR not in screen._settings_drafts
        assert screen.query_one(RAIL_LABEL_TOGGLE, Checkbox).value is False
        assert (
            screen.query_one(
                "#settings-console-collapse-large-pastes-toggle", Checkbox
            ).value
            is True
        )
        result = screen.query_one("#settings-console-behavior-result", Static)
        assert "Rail labels: Horizontal" in str(result.renderable), (
            screen._console_behavior_result,
            str(result.renderable),
        )
