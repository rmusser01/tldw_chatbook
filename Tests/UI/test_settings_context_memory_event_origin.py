"""Mounted context-memory refresh and queued-edit contracts (TASK-40969)."""

import tomllib

from textual.widgets import Input, Select

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
from Tests.UI.test_settings_remaining_input_event_origin import _settle_console_save
from tldw_chatbook.state.settings_config_models import SettingsCategoryId, SettingsDraft
from tldw_chatbook.UI.Settings_Modules.settings_config_adapter import (
    SettingsConfigAdapter,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

CATEGORY = SettingsCategoryId.CONSOLE_BEHAVIOR
INPUTS = {
    "conversation_budget_tokens": "budget-tokens",
    "compaction_trigger_ratio": "trigger-percent",
    "compaction_target_ratio": "target-percent",
    "compaction_summary_max_tokens": "summary-max-tokens",
}
SELECTS = {
    "conversation_budget_mode": "budget-mode",
    "compaction_mode": "compaction-mode",
    "compaction_representation": "compaction-representation",
    "compaction_failure_behavior": "failure-behavior",
    "compaction_carry_forward_mode": "carry-forward-mode",
}


def _values(version=0):
    return {
        "conversation_budget_tokens": 32000 + version,
        "compaction_trigger_ratio": (80 + version) / 100,
        "compaction_target_ratio": (50 + version) / 100,
        "compaction_summary_max_tokens": 1024 + version,
        "conversation_budget_mode": "custom" if version % 2 else "automatic",
        "compaction_mode": ("ask", "automatic", "off")[version % 3],
        "compaction_representation": ("text_summary", "hybrid", "visual_transcript")[
            version % 3
        ],
        "compaction_failure_behavior": (
            "omit_older_context" if version % 2 else "stop_and_ask"
        ),
        "compaction_carry_forward_mode": (
            "memory_with_latest_exchange" if version % 2 else "memory_with_recent_turns"
        ),
    }


def _display(key, value):
    return f"{value * 100:g}" if key.endswith("_ratio") else str(value)


def _controls(screen):
    return {
        key: screen.query_one(f"#settings-console-context-{suffix}", control_type)
        for fields, control_type in ((INPUTS, Input), (SELECTS, Select))
        for key, suffix in fields.items()
    }


@private_profile_test
async def test_context_refresh_keeps_nine_clean_fields_out_of_unrelated_draft(request):
    app = _build_test_app()
    app.app_config["console"] = {**_values(), "paste_collapse_threshold": 50}
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        assert not screen._category_has_unsaved_changes(CATEGORY)
        draft = screen._settings_drafts.setdefault(CATEGORY, SettingsDraft(CATEGORY))
        draft.set_value("paste_collapse_threshold", 50, 120)
        controls = _controls(screen)
        result = screen._console_behavior_result
        for version in (1, 2):
            app.app_config["console"].update(_values(version))
            screen._sync_console_behavior_widgets()
            await pilot.pause()
            assert draft.values == {"paste_collapse_threshold": 120}
            assert draft.originals == {"paste_collapse_threshold": 50}
            assert draft.dirty_keys == {"paste_collapse_threshold"}
            assert screen._console_behavior_result == result
            assert _controls(screen) == controls
            assert {key: control.value for key, control in controls.items()} == {
                key: _display(key, value) for key, value in _values(version).items()
            }


@private_profile_test
async def test_context_validation_save_and_confirmed_revert_use_real_writer(request):
    adapter = SettingsConfigAdapter()
    assert adapter.save_sections({"console": _values()})
    app = _build_test_app()
    app.app_config["console"] = _values()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        controls = _controls(screen)
        before = adapter.read_serialized()
        trigger = controls["compaction_trigger_ratio"]
        await _type_value(pilot, trigger, ".")
        await _category_key(pilot, "s")
        assert adapter.read_serialized() == before
        assert trigger.value == "."
        await _type_value(pilot, trigger, "85.")
        await _type_value(pilot, controls["compaction_target_ratio"], "75")
        await _category_key(pilot, "s")
        assert adapter.read_serialized() == before
        assert "15 percentage points" in screen._console_behavior_result
        for key, control in controls.items():
            control.value = _display(key, _values(1)[key])
        await pilot.pause()
        await _category_key(pilot, "s")
        await _settle_console_save(screen, pilot)
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")
        saved = tomllib.loads(adapter.read_serialized())["console"]
        assert {key: saved[key] for key in INPUTS | SELECTS} == _values(1)
        assert not screen._category_has_unsaved_changes(CATEGORY)
        for key, control in controls.items():
            control.value = _display(key, _values(2)[key])
        await pilot.pause()
        await _category_key(pilot, "r")
        assert isinstance(host.screen, ConfirmationDialog)
        assert screen._category_has_unsaved_changes(CATEGORY)
        await pilot.click("#confirm-button")
        await pilot.pause()
        assert not screen._category_has_unsaved_changes(CATEGORY)
        assert {key: control.value for key, control in controls.items()} == {
            key: _display(key, value) for key, value in _values(1).items()
        }
        assert {
            key: tomllib.loads(adapter.read_serialized())["console"][key]
            for key in INPUTS | SELECTS
        } == _values(1)


@private_profile_test
async def test_context_queued_real_edits_survive_refresh_for_all_nine_controls(request):
    app = _build_test_app()
    app.app_config["console"] = _values()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        controls = _controls(screen)
        for key, control in controls.items():
            message = Input.Changed if key in INPUTS else Select.Changed
            control.post_message(message(control, control.value))
            # The newest real edit can return to the previous projection.
            control.value = _display(key, _values(1)[key])
            control.value = _display(key, _values()[key])
        app.app_config["console"].update(_values(1))
        screen._sync_console_behavior_widgets()
        await pilot.pause()
        assert {key: control.value for key, control in controls.items()} == {
            key: _display(key, value) for key, value in _values().items()
        }
        draft = screen._settings_drafts[CATEGORY]
        assert draft.values == _values()
        assert draft.dirty_keys == set(INPUTS) | set(SELECTS)
        assert not screen._pending_console_input_edits


@private_profile_test
async def test_context_refresh_preserves_raw_percent_and_editing_context(request):
    app = _build_test_app()
    app.app_config["console"] = _values()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(90, 30)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        control = _controls(screen)["compaction_trigger_ratio"]
        for raw, expected in (("80.", 0.80), ("85.", 0.85), (".", ".")):
            await _type_value(pilot, control, raw)
            assert (
                screen._console_behavior_value("compaction_trigger_ratio") == expected
            )
            await pilot.press("home", "shift+right")
            await pilot.pause()
            selection, cursor = control.selection, control.cursor_position
            result = screen._console_behavior_result
            # A formatting-only edit also remains raw when no semantic draft exists.
            screen._sync_console_behavior_widgets()
            await pilot.pause()
            assert control.value == raw
            assert (
                screen._console_behavior_value("compaction_trigger_ratio") == expected
            )
            assert host.focused is control
            assert (control.selection, control.cursor_position) == (selection, cursor)
            assert screen._console_behavior_result == result
