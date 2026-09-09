"""Goal settings use canonical drafts and one atomic [agents] save."""

import pytest

from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


def test_goal_settings_join_existing_atomic_save_without_console_or_fleet_namespace(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        SettingsConfigAdapter,
        "save_sections",
        lambda self, sections: calls.append(sections) or True,
    )
    assert SettingsScreen._save_console_behavior_values(
        {"max_parallel_runs": 3},
        {"temperature": 0.5},
        {"goal_runs_enabled": True, "max_goal_generations": 0},
    )
    assert calls == [
        {
            "console": {"max_parallel_runs": 3},
            "chat_defaults": {"temperature": 0.5},
            "agents": {"goal_runs_enabled": True, "max_goal_generations": 0},
        }
    ]


@pytest.mark.asyncio
async def test_mounted_goal_policy_save_revert_and_fleet_independence():
    import toml
    from textual.widgets import Checkbox, Input

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.test_settings_configuration_hub import (
        _open_settings_category,
        _wait_for_settings_text,
    )
    from tldw_chatbook import config

    app = _build_test_app()
    app.app_config["agents"] = {
        "goal_runs_enabled": False,
        "autowake_enabled": True,
        "max_goal_generations": 3,
    }
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = host.screen
        toggle = screen.query_one("#settings-goal-runs-enabled", Checkbox)
        limit = screen.query_one("#settings-max-goal-generations", Input)
        toggle.value = True
        limit.value = "2"
        await pilot.pause()
        assert app.app_config["agents"]["goal_runs_enabled"] is False
        await pilot.click("#settings-revert-category")
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()
        toggle = screen.query_one("#settings-goal-runs-enabled", Checkbox)
        limit = screen.query_one("#settings-max-goal-generations", Input)
        assert toggle.value is False and limit.value == "3"
        toggle.value = True
        limit.value = "0"
        await pilot.pause()
        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")
        assert app.app_config["agents"] == {
            "goal_runs_enabled": True,
            "autowake_enabled": True,
            "max_goal_generations": 0,
        }
        actual = toml.load(config.get_cli_config_path())["agents"]
        assert (
            actual["goal_runs_enabled"] is True and actual["max_goal_generations"] == 0
        )
