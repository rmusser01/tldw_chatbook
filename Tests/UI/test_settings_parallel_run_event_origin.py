"""Parallel-run refreshes preserve user-owned Settings drafts (TASK-35680)."""

import tomllib

import pytest
from textual.widgets import Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
)
from Tests.UI.test_settings_configuration_hub import (
    _open_settings_category,
    _visible_text,
    _wait_for_settings_text,
)
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


@private_profile_test
async def test_parallel_refresh_preserves_unrelated_draft_and_delivers_next_edit(
    request,
):
    app = _build_test_app()
    app.app_config["console"] = {
        "max_parallel_runs": 1,
        "collapse_large_pastes": True,
        "paste_collapse_threshold": 50,
    }
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = screen._settings_drafts.setdefault(category, SettingsDraft(category))
        draft.set_value("paste_collapse_threshold", 50, 120)
        result_before = screen._console_behavior_result
        app.app_config["console"]["max_parallel_runs"] = 2
        screen._sync_console_behavior_widgets()
        await pilot.pause()
        assert draft.values == {"paste_collapse_threshold": 120}
        assert draft.dirty_keys == {"paste_collapse_threshold"}
        assert screen._console_behavior_result == result_before

        control = screen.query_one("#settings-console-max-parallel-runs", Input)
        assert control.value == "2"
        # The first genuine edit can equal the value displayed before refresh.
        await _type_value(pilot, control, "1")
        assert draft.values == {"paste_collapse_threshold": 120, "max_parallel_runs": 1}
        assert draft.dirty_keys == {"paste_collapse_threshold", "max_parallel_runs"}
        await pilot.press("home", "shift+right")
        selection = control.selection
        screen._sync_console_behavior_widgets()
        await pilot.pause()
        assert screen.query_one("#settings-console-max-parallel-runs") is control
        assert host.focused is control
        assert control.selection == selection


@pytest.mark.parametrize("size", [(90, 30), (180, 50)])
@private_profile_test
async def test_parallel_keyboard_save_revert_and_category_return_use_real_writer(
    request, size
):
    adapter = SettingsConfigAdapter()
    assert adapter.save_sections(
        {"console": {"max_parallel_runs": 1, "paste_collapse_threshold": 50}}
    )
    app = _build_test_app()
    app.app_config["console"] = {
        "max_parallel_runs": 1,
        "paste_collapse_threshold": 50,
    }
    host = DestinationHarness(app, "settings")
    category = SettingsCategoryId.CONSOLE_BEHAVIOR
    async with host.run_test(size=size) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        field_id = "#settings-console-max-parallel-runs"
        control = screen.query_one(field_id, Input)
        assert control.value == "1"
        assert not screen._category_has_unsaved_changes(category)

        screen._submit_category_search("parallel")
        await pilot.pause()
        await _type_value(pilot, control, "0")
        await _category_key(pilot, "s")
        await _wait_for_settings_text(
            screen, pilot, "Max parallel agent runs must be an integer of at least 1."
        )
        assert _saved_parallel_runs(adapter) == 1

        await _type_value(pilot, control, "12")
        await _category_key(pilot, "s")
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")
        assert _saved_parallel_runs(adapter) == 12
        assert control.value == "12"
        assert not screen._category_has_unsaved_changes(category)
        assert "staged" not in screen._console_behavior_result

        paste = screen.query_one("#settings-console-paste-collapse-threshold", Input)
        await _type_value(pilot, paste, "120")
        await _open_settings_category(pilot, "#settings-category-overview")
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        control = screen.query_one(field_id, Input)
        assert control.value == "12"
        assert (
            screen._settings_drafts[category].values["paste_collapse_threshold"] == 120
        )
        assert "max_parallel_runs" not in screen._settings_drafts[category].values
        screen._submit_category_search("parallel")
        await pilot.pause()
        await _type_value(pilot, control, "1")
        assert screen._settings_drafts[category].dirty_keys == {
            "paste_collapse_threshold",
            "max_parallel_runs",
        }

        await _category_key(pilot, "r")
        assert isinstance(host.screen, ConfirmationDialog)
        assert control.value == "1"
        assert screen._category_has_unsaved_changes(category)
        await pilot.click("#confirm-button")
        await pilot.pause()
        assert control.value == "12"
        assert (
            screen.query_one("#settings-console-paste-collapse-threshold", Input).value
            == "50"
        )
        assert not screen._category_has_unsaved_changes(category)
        assert "staged" not in screen._console_behavior_result
        assert "reverted" in _visible_text(screen)
        assert _saved_parallel_runs(adapter) == 12


async def _type_value(pilot, control, value):
    control.scroll_visible(animate=False)
    await pilot.pause()
    control.focus()
    await pilot.pause()
    assert pilot.app.focused is control
    await pilot.press("home", "shift+end", *value)
    await pilot.pause()
    assert control.value == value


async def _category_key(pilot, key):
    await pilot.press("escape")
    await pilot.pause()
    await pilot.press(key)
    await pilot.pause()


def _saved_parallel_runs(adapter):
    return tomllib.loads(adapter.read_serialized())["console"]["max_parallel_runs"]
