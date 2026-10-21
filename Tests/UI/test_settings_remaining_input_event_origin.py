"""Remaining Console refreshes preserve user-owned input state (TASK-35867)."""

import asyncio
import json
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
    _wait_for_settings_text,
)
from Tests.UI.test_settings_parallel_run_event_origin import _category_key, _type_value
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
)
from tldw_chatbook.UI.Screens.settings_screen import AGENT_BUDGET_FIELDS
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

CATEGORY = SettingsCategoryId.CONSOLE_BEHAVIOR
FIELDS = {field.key: field.widget_id for field in AGENT_BUDGET_FIELDS}
FIELDS.update(
    tool_result_display_chars="settings-console-tool-result-display-chars",
    sidechat_model="settings-console-sidechat-model",
    sidechat_prompt_template="settings-console-sidechat-prompt-template",
)


def _values(version=0):
    values = {
        field.key: 200.0 + version if field.is_float else 1000 + version
        for field in AGENT_BUDGET_FIELDS
    }
    values.update(
        tool_result_display_chars=160 + version,
        sidechat_model=f"model-{version}",
        sidechat_prompt_template=f"Explain {version}: {{selection}}",
    )
    return values


def _controls(screen):
    return {
        key: screen.query_one(f"#{widget_id}", Input)
        for key, widget_id in FIELDS.items()
    }


async def _settle_input_edits(screen, pilot):
    async def delivered():
        while screen._pending_console_input_edits:
            await pilot.pause()

    await asyncio.wait_for(delivered(), timeout=5)


async def _settle_console_save(screen, pilot):
    for worker in list(screen.workers):
        if worker.node is screen and worker.group == "settings-save-console-behavior":
            await worker.wait()
    await pilot.pause()


@pytest.mark.parametrize("size", [(90, 30), (180, 50)], ids=["compact", "wide"])
@private_profile_test
async def test_remaining_refresh_tracks_backing_without_polluting_draft(request, size):
    app = _build_test_app()
    app.app_config["console"] = {**_values(), "paste_collapse_threshold": 50}
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=size) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        assert not screen._category_has_unsaved_changes(CATEGORY)
        assert "staged" not in screen._console_behavior_result
        draft = screen._settings_drafts.setdefault(CATEGORY, SettingsDraft(CATEGORY))
        draft.set_value("paste_collapse_threshold", 50, 120)
        controls = _controls(screen)
        result = screen._console_behavior_result
        focused = controls["sidechat_model"]
        focused.scroll_visible(animate=False)
        await pilot.pause()
        focused.focus()
        await pilot.pause()
        await pilot.press("home", "right", "shift+right")
        await pilot.pause()
        selection, cursor = focused.selection, focused.cursor_position
        assert host.focused is focused
        for version in (1, 2):
            app.app_config["console"].update(_values(version))
            screen._sync_console_behavior_widgets()
            await pilot.pause()
            assert draft.values == {"paste_collapse_threshold": 120}
            assert draft.originals == {"paste_collapse_threshold": 50}
            assert draft.dirty_keys == {"paste_collapse_threshold"}
            assert screen._console_behavior_result == result
            assert _controls(screen) == controls
            assert host.focused is focused
            assert (focused.selection, focused.cursor_position) == (selection, cursor)
            assert {key: control.value for key, control in controls.items()} == {
                key: str(value) for key, value in _values(version).items()
            }
        for key, control in controls.items():
            # A genuine first edit may equal an earlier programmatic display.
            await _type_value(pilot, control, str(_values(1)[key]))
            assert key in draft.dirty_keys
            await pilot.press("home", "shift+right")
            await pilot.pause()
            selection, cursor = control.selection, control.cursor_position
            assert host.focused is control
            screen._sync_console_behavior_widgets()
            await pilot.pause()
            assert _controls(screen)[key] is control
            assert host.focused is control
            assert (control.selection, control.cursor_position) == (selection, cursor)


@private_profile_test
async def test_remaining_stale_events_keep_latest_edit_and_invalid_draft(
    request, monkeypatch
):
    app = _build_test_app()
    app.app_config["console"] = _values()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        controls = _controls(screen)
        model = controls["sidechat_model"]
        guard = screen._console_input_edit_is_current
        refreshed = False

        def refresh_after_initialization(event, value):
            nonlocal refreshed
            current = guard(event, value)
            if event.input is model and not refreshed:
                refreshed = True
                app.app_config["console"]["sidechat_model"] = _values(2)[
                    "sidechat_model"
                ]
                screen._sync_console_behavior_widgets()
            return current

        monkeypatch.setattr(
            screen, "_console_input_edit_is_current", refresh_after_initialization
        )
        model.post_message(Input.Changed(model, model.value))
        model.value = _values(1)["sidechat_model"]
        model.value = _values()["sidechat_model"]
        await _settle_input_edits(screen, pilot)
        assert refreshed
        assert model.value == _values()["sidechat_model"]
        assert screen._settings_drafts[CATEGORY].values == {
            "sidechat_model": model.value
        }
        assert not screen._pending_console_input_edits
        monkeypatch.setattr(screen, "_console_input_edit_is_current", guard)
        app.app_config["console"] = _values()
        screen._revert_category(CATEGORY)
        await pilot.pause()

        for key, control in controls.items():
            # Queue initialization followed by a legitimate edit before delivery.
            control.post_message(Input.Changed(control, control.value))
            control.value = str(_values(1)[key])
            control.value = str(_values()[key])
        app.app_config["console"].update(_values(2))
        screen._sync_console_behavior_widgets()
        await _settle_input_edits(screen, pilot)
        draft = screen._settings_drafts[CATEGORY]
        assert draft.values == _values()
        assert draft.dirty_keys == set(FIELDS)
        assert not screen._pending_console_input_edits
        # A queued edit that returns to the previous projection is still an edit
        # when the backing value changes before the messages are delivered.
        for key, control in controls.items():
            control.value = str(_values(2)[key])
            control.value = str(_values(1)[key])
        app.app_config["console"].update(_values(3))
        screen._sync_console_behavior_widgets()
        await _settle_input_edits(screen, pilot)
        assert draft is screen._settings_drafts[CATEGORY]
        assert draft.values == _values(1), {
            "same_draft": draft is screen._settings_drafts.get(CATEGORY),
            "current_draft": screen._settings_drafts.get(CATEGORY),
            "controls": {key: control.value for key, control in controls.items()},
            "same_controls": _controls(screen) == controls,
            "pending": dict(screen._pending_console_input_edits),
            "backing": app.app_config["console"],
        }
        assert {key: control.value for key, control in controls.items()} == {
            key: str(value) for key, value in _values(1).items()
        }
        wall = controls["agent_max_wall_seconds"]
        await _type_value(pilot, wall, ".")
        await pilot.press("home", "shift+right")
        await pilot.pause()
        selection, cursor = wall.selection, wall.cursor_position
        result = screen._console_behavior_result
        app.app_config["console"].update(_values(2))
        screen._sync_console_behavior_widgets()
        await pilot.pause()
        assert draft.values["agent_max_wall_seconds"] == "."
        assert wall.value == "."
        assert host.focused is wall
        assert (wall.selection, wall.cursor_position) == (selection, cursor)
        assert screen._console_behavior_result == result


@pytest.mark.parametrize("size", [(90, 30), (180, 50)], ids=["compact", "wide"])
@private_profile_test
async def test_remaining_keyboard_save_revert_and_return_use_real_writer(request, size):
    adapter = SettingsConfigAdapter()
    assert adapter.save_sections({"console": _values()})
    app = _build_test_app()
    app.app_config["console"] = _values()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=size) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        controls = _controls(screen)
        before = adapter.read_serialized()
        await _type_value(pilot, controls["agent_max_steps"], "200000")
        await _category_key(pilot, "s")
        assert adapter.read_serialized() == before
        assert "199999" in screen._console_behavior_result.replace(",", "")
        for key, control in controls.items():
            await _type_value(pilot, control, str(_values(1)[key]))
        await _category_key(pilot, "s")
        await _settle_console_save(screen, pilot)
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")
        saved = tomllib.loads(adapter.read_serialized())["console"]
        assert {key: saved[key] for key in FIELDS} == _values(1)
        assert not screen._category_has_unsaved_changes(CATEGORY)
        assert "staged" not in screen._console_behavior_result
        for key, control in controls.items():
            await _type_value(pilot, control, str(_values(2)[key]))
        await _open_settings_category(pilot, "#settings-category-overview")
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        old_controls = controls
        controls = _controls(screen)
        assert all(controls[key] is not old_controls[key] for key in FIELDS)
        assert not screen._pending_console_input_edits
        returned = screen._settings_drafts[CATEGORY]
        extras = {
            key: {
                "value": value,
                "original": returned.originals[key],
                "dirty": key in returned.dirty_keys,
            }
            for key, value in returned.values.items()
            if key not in FIELDS
        }
        request.node.user_properties.append(
            (
                "console_return_origins",
                json.dumps(
                    {"dirty_keys": sorted(returned.dirty_keys), "extra_fields": extras}
                ),
            )
        )
        assert {key: returned.values[key] for key in FIELDS} == _values(2)
        assert {key: returned.originals[key] for key in FIELDS} == _values(1)
        assert returned.dirty_keys == set(FIELDS)
        assert all(
            item["value"] == item["original"] and not item["dirty"]
            for item in extras.values()
        )
        assert {key: control.value for key, control in controls.items()} == {
            key: str(value) for key, value in _values(2).items()
        }
        # The rebuilt controls observe their first pending edit.
        model = controls["sidechat_model"]
        model.value = "after-return"
        screen._sync_console_behavior_widgets()
        await pilot.pause()
        assert (
            screen._settings_drafts[CATEGORY].values["sidechat_model"] == "after-return"
        )
        assert not screen._pending_console_input_edits
        await _category_key(pilot, "r")
        assert isinstance(host.screen, ConfirmationDialog)
        assert screen._category_has_unsaved_changes(CATEGORY)
        await pilot.click("#confirm-button")
        await pilot.pause()
        assert not screen._category_has_unsaved_changes(CATEGORY)
        assert "staged" not in screen._console_behavior_result
        assert {key: control.value for key, control in controls.items()} == {
            key: str(value) for key, value in _values(1).items()
        }
        assert {
            key: tomllib.loads(adapter.read_serialized())["console"][key]
            for key in FIELDS
        } == _values(1)
