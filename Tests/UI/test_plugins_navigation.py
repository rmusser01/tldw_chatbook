"""Plugins uses the production shell metadata, palette and screen factory."""

import pytest
from textual import on
from textual.widgets import Button

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.app import TabNavigationProvider, TldwCli
from tldw_chatbook.UI.Navigation.main_navigation import (
    MainNavigationBar,
    NavigateToScreen,
)
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route


class NavigationApp(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET

    def __init__(self):
        super().__init__()
        self.routes = []

    def compose(self):
        yield MainNavigationBar(active="home")

    @on(NavigateToScreen)
    def navigate(self, event):
        self.routes.append(event.screen_name)


@pytest.mark.asyncio
async def test_mounted_palette_and_navigation_expose_plugins_with_settings_control():
    app = NavigationApp()
    async with app.run_test(size=(220, 30)) as pilot:
        provider = TabNavigationProvider(app.screen)
        settings = [hit async for hit in provider.search("Settings")]
        assert settings
        next(hit for hit in settings if hit.command.args == ("settings",)).command()
        await pilot.pause()
        assert app.routes[-1] == "settings"
        plugins = [
            hit
            async for hit in provider.search("Plugins")
            if hit.command.args == ("plugins",)
        ]
        assert (
            plugins
        ), "Plugins must be discoverable through the mounted palette provider"
        plugins[0].command()
        await pilot.pause()
        assert app.routes[-1] == "plugins"
        button = app.query_one("#nav-plugins", Button)
        button.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.routes[-1] == "plugins"
        assert str(button.label) == "F8 Plugins"


def test_actual_factory_constructs_fresh_plugin_screen_without_starting_services():
    settings = resolve_screen_route("settings")
    assert settings is not None and settings.screen_name == "settings"
    route = resolve_screen_route("plugins")
    assert route is not None, "Plugins must be a lazy production route"
    screen_type = route.load_screen_class()
    assert screen_type is not None
    app = NavigationApp()
    first = TldwCli._create_navigation_screen(app, "plugins", screen_type)
    second = TldwCli._create_navigation_screen(app, "plugins", screen_type)
    assert first is not second
    assert first.app_instance is app
    assert not hasattr(app, "_plugin_service")
    assert not hasattr(app, "_plugin_source_service")


@pytest.mark.asyncio
async def test_production_app_f8_and_settings_link_mount_same_lazy_destination(
    monkeypatch,
):
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.config import save_setting_to_cli_config
    from tldw_chatbook.UI.Screens.plugins_screen import PluginsScreen
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app(configured_default="home")
    async with app.run_test(size=(120, 35)) as pilot:
        for _ in range(200):
            await pilot.pause(0.02)
            if (
                getattr(app, "_ui_ready", False)
                and getattr(app.screen, "screen_name", None) == "home"
            ):
                break
        assert getattr(app, "_ui_ready", False)
        assert getattr(app, "_plugin_source_service", None) is None
        await pilot.press("f8")
        for _ in range(150):
            await pilot.pause(0.02)
            if isinstance(app.screen, PluginsScreen):
                break
        assert isinstance(app.screen, PluginsScreen)
        source_owner = app.plugin_source_service
        assert source_owner is app.plugin_source_service
        await pilot.press("f4")
        for _ in range(150):
            await pilot.pause(0.02)
            if isinstance(app.screen, SettingsScreen):
                break
        assert isinstance(app.screen, SettingsScreen)
        link = app.screen.query_one("#settings-open-plugins", Button)
        link.focus()
        await pilot.press("enter")
        for _ in range(150):
            await pilot.pause(0.02)
            if isinstance(app.screen, PluginsScreen):
                break
        assert isinstance(app.screen, PluginsScreen)
        assert app.plugin_source_service is source_owner


def test_plugins_feature_sheet_uses_current_route_owner():
    from pathlib import Path

    from tldw_chatbook.UI.Screens.plugins_screen import PluginsScreen

    assert TldwCli._SCREEN_OWNED_ROUTE_CSS["plugins"] == ("screen_feature_plugins.tcss",)
    paths = PluginsScreen.CSS_PATH
    entries = [paths] if isinstance(paths, (str, Path)) else list(paths or ())
    assert all(Path(entry).name != "screen_feature_plugins.tcss" for entry in entries)
