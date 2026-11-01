"""Mounted regression for programmatic composer-paste form updates."""

import pytest
from textual.widgets import Checkbox, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
)
from Tests.UI.test_settings_configuration_hub import _open_settings_category
from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
)


@pytest.mark.parametrize("field", ["collapse_large_pastes", "paste_collapse_threshold"])
@private_profile_test
async def test_value_changing_paste_refresh_is_not_a_user_edit(request, field):
    app = _build_test_app()
    app.app_config["console"] = {
        "collapse_large_pastes": True,
        "paste_collapse_threshold": 50,
    }
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        # Change the backing value, then use the real programmatic refresh.
        app.app_config["console"][field] = (
            False if field == "collapse_large_pastes" else 120
        )
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = screen._settings_drafts.setdefault(category, SettingsDraft(category))
        # An unrelated edit must survive the refresh without acquiring extra
        # values from controls the user has not touched.
        draft.set_value("max_parallel_runs", 1, 2)
        before = dict(draft.values)
        screen._sync_console_behavior_widgets()
        await pilot.pause()
        draft = screen._settings_drafts.get(category)
        assert (dict(draft.values) if draft else {}) == before
        assert draft.dirty_keys == {"max_parallel_runs"}

        # A later genuine change to the old value must still be delivered.
        if field == "collapse_large_pastes":
            control = screen.query_one(
                "#settings-console-collapse-large-pastes-toggle", Checkbox
            )
            assert control.value is False
            control.value = True
        else:
            control = screen.query_one(
                "#settings-console-paste-collapse-threshold", Input
            )
            assert control.value == "120"
            control.value = "50"
        await pilot.pause()
        assert field in screen._settings_drafts[category].dirty_keys


@pytest.mark.parametrize("size", [(90, 30), (180, 50)])
@private_profile_test
async def test_paste_search_typing_and_category_return(request, size):
    app = _build_test_app()
    app.app_config["console"] = {
        "collapse_large_pastes": True,
        "paste_collapse_threshold": 50,
    }
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=size) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        screen._submit_category_search("threshold")
        await pilot.pause()
        field_id = "#settings-console-paste-collapse-threshold"
        control = screen.query_one(field_id, Input)
        assert host.focused is control
        await pilot.press("home", "shift+end", "1", "2", "0")
        await pilot.pause()
        assert screen.query_one(field_id) is control
        assert control.value == "120"
        assert host.focused is control
        await pilot.press("home", "shift+right")
        selection = control.selection
        screen._update_console_paste_summary()
        await pilot.pause()
        assert control.selection == selection
        assert host.focused is control

        await _open_settings_category(pilot, "#settings-category-overview")
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        control = screen.query_one(field_id, Input)
        assert control.value == "120"
        screen._submit_category_search("threshold")
        await pilot.pause()
        await pilot.press("home", "shift+end", "5", "0")
        await pilot.pause()
        assert control.value == "50"
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.CONSOLE_BEHAVIOR
        )
