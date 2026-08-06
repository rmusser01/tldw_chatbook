import pytest
from unittest.mock import patch
from textual.app import App
from textual.css.query import NoMatches
from textual.widgets import Button, Checkbox, Input, OptionList, Select

from tldw_chatbook.Widgets import settings_splash_screen_viewer as splash_module
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

            enabled = viewer.query_one("#settings-splash-enabled", Checkbox)
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
async def test_splash_checkboxes_carry_text_state_labels(splash_app):
    """Each toggle row shows an On/Off Static that flips with the toggle.

    The checkbox alone carries state visually, which is
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
            checkbox = pilot.app.query_one("#settings-splash-enabled", Checkbox)
            state = pilot.app.query_one("#settings-splash-enabled-state", Static)
            initial = bool(checkbox.value)
            assert str(state.renderable) == switch_state_label(initial)

            checkbox.toggle()
            await pilot.pause()

            assert str(state.renderable) == switch_state_label(not initial)


@pytest.mark.asyncio
async def test_settings_splash_viewer_has_single_default_card_control(splash_app):
    """task-1376: the 'Default card' Select is the one control that sets the
    default splash card; the duplicate 'Set as default' button is gone."""
    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)

        select = viewer.query_one("#settings-splash-default-select", Select)
        assert select is not None

        with pytest.raises(NoMatches):
            viewer.query_one("#settings-splash-set-default", Button)

        button_labels = [str(button.label) for button in viewer.query(Button)]
        assert button_labels == ["Play selected"]


@pytest.mark.asyncio
async def test_settings_splash_viewer_default_select_persists_on_change(
    splash_app, monkeypatch
):
    """task-1376: the remaining control stays instant-apply: changing the
    Select saves card_selection immediately (no separate commit step)."""
    saved: list[tuple[str, str, object]] = []

    def fake_save(section: str, key: str, value: object) -> None:
        saved.append((section, key, value))

    monkeypatch.setattr(splash_module, "save_setting_to_cli_config", fake_save)

    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)

        select = viewer.query_one("#settings-splash-default-select", Select)
        card_values = [
            option[1] for option in select._options if option[1] != "random"
        ]
        assert card_values, "expected at least one splash card option"

        target = card_values[0]
        select.value = target
        await pilot.pause()

        assert ("splash_screen", "card_selection", target) in saved
        assert viewer._config["card_selection"] == target
