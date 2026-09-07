"""Regression tests for task-740: [splash_screen] config must be honored."""

import tomllib
from unittest.mock import patch

import tldw_chatbook.config as config


def test_default_splash_duration_is_seven_seconds():
    """The out-of-the-box intro length is 7s everywhere it can be defaulted."""
    from tldw_chatbook.Widgets.settings_splash_screen_viewer import (
        DEFAULT_SPLASH_CONFIG as viewer_defaults,
    )
    from tldw_chatbook.Widgets.splash_screen import SplashScreen

    with patch(
        "tldw_chatbook.Widgets.splash_screen.get_cli_setting",
        side_effect=lambda section, key=None, default=None: default,
    ):
        splash = SplashScreen()
    assert splash.duration == 7.0
    assert viewer_defaults["duration"] == 7.0

    parsed = tomllib.loads(config.CONFIG_TOML_CONTENT)
    assert parsed["splash_screen"]["duration"] == 7.0
    # Skippable by default.
    assert parsed["splash_screen"]["skip_on_keypress"] is True


def test_splash_screen_reads_configured_duration():
    """A configured value must win over the hardcoded default."""

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "duration":
            return 9.5
        return default

    with patch(
        "tldw_chatbook.Widgets.splash_screen.get_cli_setting",
        side_effect=fake_get_cli_setting,
    ):
        from tldw_chatbook.Widgets.splash_screen import SplashScreen

        splash = SplashScreen()
        assert splash.config["duration"] == 9.5


def test_splash_screen_default_applies_only_when_key_absent():
    def fake_get_cli_setting(section, key=None, default=None):
        return default

    with patch(
        "tldw_chatbook.Widgets.splash_screen.get_cli_setting",
        side_effect=fake_get_cli_setting,
    ):
        from tldw_chatbook.Widgets.splash_screen import SplashScreen

        splash = SplashScreen()
        assert splash.config["duration"] == 7.0


def test_settings_splash_viewer_reads_configured_card_selection():
    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "card_selection":
            return "matrix"
        return default

    with patch(
        "tldw_chatbook.Widgets.settings_splash_screen_viewer.get_cli_setting",
        side_effect=fake_get_cli_setting,
    ):
        from tldw_chatbook.Widgets.settings_splash_screen_viewer import (
            SettingsSplashScreenViewer,
        )

        viewer = SettingsSplashScreenViewer()
        # Call _load_config directly (compose() requires a Textual app context)
        viewer._config = viewer._load_config()
        assert viewer._config["card_selection"] == "matrix"
