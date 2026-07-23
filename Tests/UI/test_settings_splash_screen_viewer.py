import pytest
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
