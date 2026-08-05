import pytest
from unittest.mock import patch
from textual.app import App
from textual.widgets import Input, OptionList, Select, Switch

from tldw_chatbook.Widgets.settings_splash_screen_viewer import (
    DEFAULT_SPLASH_CONFIG,
    SettingsSplashScreenViewer,
)


class _SplashTestApp(App[None]):
    """Minimal app for testing the splash settings viewer in isolation."""

    CSS = """
    Screen { align: center middle; }
    """

    def compose(self):
        yield SettingsSplashScreenViewer()


@pytest.fixture
def splash_app():
    return _SplashTestApp()


@pytest.mark.asyncio
async def test_settings_splash_viewer_can_compose(splash_app):
    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)
        assert viewer.is_mounted


@pytest.mark.asyncio
async def test_settings_splash_viewer_loads_defaults(splash_app):
    """Test that viewer loads defaults when config has no configured values."""

    def fake_get_cli_setting(section, key=None, default=None):
        # Return the default parameter to simulate no configured values
        # This tests the fallback behavior
        return default

    with patch(
        "tldw_chatbook.Widgets.settings_splash_screen_viewer.get_cli_setting",
        side_effect=fake_get_cli_setting,
    ):
        # Recreate the app with patched get_cli_setting
        splash_app = _SplashTestApp()
        async with splash_app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            viewer = splash_app.query_one(SettingsSplashScreenViewer)

            enabled = viewer.query_one("#settings-splash-enabled", Switch)
            assert enabled.value == DEFAULT_SPLASH_CONFIG["enabled"]

            duration = viewer.query_one("#settings-splash-duration", Input)
            assert float(duration.value) == DEFAULT_SPLASH_CONFIG["duration"]


@pytest.mark.asyncio
async def test_settings_splash_viewer_card_list_populated(splash_app):
    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)

        card_list = viewer.query_one("#settings-splash-card-list", OptionList)
        assert len(card_list.options) > 0


@pytest.mark.asyncio
async def test_settings_splash_viewer_default_select_contains_random(splash_app):
    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)

        select = viewer.query_one("#settings-splash-default-select", Select)
        values = [str(option[1]) for option in select._options]
        assert "random" in values


@pytest.mark.asyncio
async def test_settings_splash_viewer_selection_triggers_preview(splash_app):
    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)

        card_list = viewer.query_one("#settings-splash-card-list", OptionList)
        # Highlight the first real card
        card_list.highlighted = 0
        await pilot.pause()

        container = viewer.query_one("#settings-splash-preview-scroll")
        assert len(container.children) > 0


# ---- task-1561: text-labeled toggle states ----


@pytest.mark.asyncio
async def test_splash_switches_carry_text_state_labels(splash_app):
    """Each Switch row shows an On/Off Static that flips with the toggle.

    The Switch slider carries state by position/color only, which is
    unreadable in reduced-color terminals (task-1561).
    """
    from textual.widgets import Static

    from tldw_chatbook.Widgets.settings_splash_screen_viewer import (
        switch_state_label,
    )

    assert switch_state_label(True) == "On"
    assert switch_state_label(False) == "Off"

    with patch(
        "tldw_chatbook.Widgets.settings_splash_screen_viewer.save_setting_to_cli_config",
        return_value=True,
    ):
        async with splash_app.run_test(size=(120, 50)) as pilot:
            switch = pilot.app.query_one("#settings-splash-enabled", Switch)
            state = pilot.app.query_one("#settings-splash-enabled-state", Static)
            initial = bool(switch.value)
            assert str(state.renderable) == switch_state_label(initial)

            switch.toggle()
            await pilot.pause()

            assert str(state.renderable) == switch_state_label(not initial)
