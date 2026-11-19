"""Workflow settings use canonical drafts and one atomic [agents] save."""

import pytest

from Tests.private_profile import private_profile_test

pytestmark = pytest.mark.bootstrap_profile

from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


def test_workflow_settings_join_existing_atomic_save_without_console_or_fleet_namespace(
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
        {"workflow_runs_enabled": True, "max_workflow_generations": 0},
    )
    assert calls == [
        {
            "console": {"max_parallel_runs": 3},
            "chat_defaults": {"temperature": 0.5},
            "agents": {"workflow_runs_enabled": True, "max_workflow_generations": 0},
        }
    ]


@pytest.mark.asyncio
@private_profile_test
async def test_mounted_workflow_policy_save_revert_and_fleet_independence(request):
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
        "workflow_runs_enabled": False,
        "autowake_enabled": True,
        "max_workflow_generations": 8,
    }
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = host.screen
        toggle = screen.query_one("#settings-workflow-runs-enabled", Checkbox)
        limit = screen.query_one("#settings-max-workflow-generations", Input)
        toggle.value = True
        limit.value = "2"
        await pilot.pause()
        assert app.app_config["agents"]["workflow_runs_enabled"] is False
        await pilot.click("#settings-revert-category")
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()
        toggle = screen.query_one("#settings-workflow-runs-enabled", Checkbox)
        limit = screen.query_one("#settings-max-workflow-generations", Input)
        assert toggle.value is False and limit.value == "8"
        toggle.value = True
        limit.value = "0"
        await pilot.pause()
        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")
        assert app.app_config["agents"] == {
            "workflow_runs_enabled": True,
            "autowake_enabled": True,
            "max_workflow_generations": 0,
        }
        actual = toml.load(config.get_cli_config_path())["agents"]
        assert (
            actual["workflow_runs_enabled"] is True
            and actual["max_workflow_generations"] == 0
        )


@pytest.mark.parametrize("value", [True, False, -1, "1.5", "nan", float("inf"), 2**63])
def test_workflow_limit_validation_refuses_nonfinite_or_boolean_values(value):
    from tldw_chatbook.UI.Screens.settings_workflow_policy import (
        normalize_workflow_setting,
    )

    with pytest.raises(ValueError):
        normalize_workflow_setting("max_workflow_wall_seconds", value)


def test_workflow_active_time_label_and_defaults():
    from tldw_chatbook.UI.Screens.settings_workflow_policy import (
        WORKFLOW_SETTING_FIELDS,
    )

    fields = {key: (label, default) for key, label, default in WORKFLOW_SETTING_FIELDS}
    assert fields["max_workflow_wall_seconds"] == ("Cumulative active seconds", 900)
    assert fields["max_workflow_child_launches"][1] == 12
