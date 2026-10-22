"""Submitted-only acknowledgement for the three ordinary staged Settings forms.

The screen still owns workers, drafts and checked writes. Capture mounted values
on the UI thread before publishing saved config; never resync over live editors.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING

from textual.css.query import QueryError
from textual.widgets import Checkbox, Input, Select

from ..Screens.settings_config_models import SettingsCategoryId, SettingsDraft

if TYPE_CHECKING:
    from ..Screens.settings_screen import SettingsScreen


def _loaded_values(screen: SettingsScreen, category: SettingsCategoryId) -> dict:
    if category is SettingsCategoryId.APPEARANCE:
        return screen._appearance_loaded_values()
    if category is SettingsCategoryId.STORAGE:
        return screen._storage_loaded_values()
    from ..Screens.settings_screen import AGENT_BUDGET_FIELDS

    values = screen._console_behavior_loaded_values()
    values.update(
        {
            field.key: screen._loaded_agent_budget_value(field)
            for field in AGENT_BUDGET_FIELDS
        }
    )
    values.update(
        {
            f"background_effects.{key}": value
            for key, value in screen._loaded_console_background_effects().items()
        }
    )
    return values


def _console_selectors(values: dict) -> dict[str, str]:
    from ..Screens.settings_goal_policy import GOAL_SETTING_KEYS
    from ..Screens.settings_screen import CONSOLE_BEHAVIOR_CHAT_DEFAULT_KEYS

    selectors = {
        key: f"#settings-console-{key.replace('_', '-')}"
        for key in values
        if key not in {"show_model_thinking", "thinking_history_policy_default"}
    }
    selectors.update(
        {
            key: f"#settings-console-default-{key.replace('_', '-')}"
            for key in CONSOLE_BEHAVIOR_CHAT_DEFAULT_KEYS
        }
    )
    selectors.update(
        {key: f"#settings-{key.replace('_', '-')}" for key in GOAL_SETTING_KEYS}
    )
    selectors.update(
        {
            "collapse_large_pastes": "#settings-console-collapse-large-pastes-toggle",
            "conversation_budget_mode": "#settings-console-context-budget-mode",
            "conversation_budget_tokens": "#settings-console-context-budget-tokens",
            "compaction_mode": "#settings-console-context-compaction-mode",
            "compaction_representation": "#settings-console-context-compaction-representation",
            "compaction_failure_behavior": "#settings-console-context-failure-behavior",
            "compaction_carry_forward_mode": "#settings-console-context-carry-forward-mode",
            "compaction_trigger_ratio": "#settings-console-context-trigger-percent",
            "compaction_target_ratio": "#settings-console-context-target-percent",
            "compaction_summary_max_tokens": "#settings-console-context-summary-max-tokens",
        }
    )
    selectors.update(
        {
            f"background_effects.{key}": f"#settings-console-background-effect-{suffix}"
            for key, suffix in (
                ("enabled", "enabled"),
                ("effect", "type"),
                ("scope", "scope"),
                ("intensity", "intensity"),
                ("fps", "fps"),
            )
        }
    )
    return selectors


def capture_staged_values(
    screen: SettingsScreen, category: SettingsCategoryId
) -> dict[str, object]:
    """Capture draft values and accepted control edits before queued delivery.

    Args:
        screen: Existing category/draft owner, called on the UI thread.
        category: One of Appearance, Storage or Console Behavior.

    Returns:
        Detached current values, including incomplete raw input.
    """
    values = _loaded_values(screen, category)
    draft = screen._settings_drafts.get(category)
    if draft is not None:
        values.update(draft.values)
    if category is SettingsCategoryId.APPEARANCE:
        selectors = {key: screen._appearance_field_selector(key) for key in values}
    elif category is SettingsCategoryId.STORAGE:
        selectors = {key: screen._storage_field_selector(key) for key in values}
    else:
        selectors = _console_selectors(values)
    for key, selector in selectors.items():
        if selector is None or key not in values:
            continue
        try:
            control = screen.query_one(selector)
        except QueryError:
            continue
        current = values[key]
        if isinstance(control, Input):
            if key in {"compaction_trigger_ratio", "compaction_target_ratio"}:
                from ..Screens.settings_context_memory import (
                    format_ratio_percent,
                    ratio_from_percent,
                )

                if control.value != format_ratio_percent(current):
                    values[key] = ratio_from_percent(control.value)
            elif control.value != screen._console_input_value(current):
                values[key] = control.value
        elif isinstance(control, Checkbox):
            values[key] = control.value
        elif isinstance(control, Select):
            if key == "streaming":
                values[key] = control.value == "true"
            elif control.value is Select.NULL:
                # Unknown saved enum choices deliberately render as blank.
                if current in (value for _, value in control._options):
                    values[key] = ""
            else:
                values[key] = control.value
    if category is SettingsCategoryId.CONSOLE_BEHAVIOR:
        _capture_reasoning_overrides(screen, values)
    return copy.deepcopy(values)


def _capture_reasoning_overrides(screen: SettingsScreen, values: dict) -> None:
    from ..Screens.settings_screen import reasoning_override_key

    target = getattr(screen, "_reasoning_override_target", None)
    if target is None:
        return
    override_key = reasoning_override_key(*target)
    for key, selector, expected in (
        (
            "reasoning_history_overrides",
            "#settings-console-reasoning-override",
            screen._reasoning_override_value(),
        ),
        (
            "reasoning_native_tool_overrides",
            "#settings-console-reasoning-native-tools",
            screen._reasoning_native_override_value(),
        ),
    ):
        try:
            control = screen.query_one(selector)
        except QueryError:
            continue
        if control.value == expected:
            continue
        overrides = dict(values.get(key) or {})
        if control.value in {"inherit", False}:
            overrides.pop(override_key, None)
        else:
            overrides[override_key] = control.value
        values[key] = overrides


def _acknowledge(
    screen: SettingsScreen,
    category: SettingsCategoryId,
    current: dict,
    submitted: Mapping[str, object] | None,
) -> None:
    baseline = _loaded_values(screen, category)
    draft = SettingsDraft(category=category)
    for key, value in current.items():
        if submitted is not None and value == submitted.get(key):
            # The writer can normalize unchanged raw input (e.g. trim a model).
            continue
        if value != baseline.get(key):
            draft.set_value(key, baseline.get(key), value)
    if draft.is_dirty:
        screen._settings_drafts[category] = draft
    else:
        screen._settings_drafts.pop(category, None)
    screen._update_draft_status_widgets(category)


def apply_appearance_save_result(
    screen: SettingsScreen,
    saved: bool,
    section_values: Mapping[str, object],
    submitted: Mapping[str, object] | None = None,
) -> None:
    if saved:
        current_values = capture_staged_values(screen, SettingsCategoryId.APPEARANCE)
        target = screen._app_config_update_target()
        for section, values in copy.deepcopy(dict(section_values)).items():
            current = target.get(section)
            target[section] = (
                {**current, **values} if isinstance(current, Mapping) else values
            )
        screen._signal_console_appearance_refresh()
        screen._signal_library_reader_layout_refresh()
        _acknowledge(screen, SettingsCategoryId.APPEARANCE, current_values, submitted)
        screen._appearance_result = "Appearance defaults saved."
        screen._set_static_text(
            "#settings-appearance-save-result", screen._appearance_result
        )
        screen._update_appearance_validation_classes()
        screen.app.notify("Appearance defaults saved.", severity="information")
        return
    screen._appearance_result = "Failed to save Appearance defaults."
    screen._set_static_text(
        "#settings-appearance-save-result", screen._appearance_result
    )
    screen.app.notify(screen._appearance_result, severity="error")


def apply_storage_save_result(
    screen: SettingsScreen,
    saved: bool,
    section_values: Mapping[str, object],
    submitted: Mapping[str, object] | None = None,
) -> None:
    if saved:
        current = capture_staged_values(screen, SettingsCategoryId.STORAGE)
        screen._app_config_update_target().update(copy.deepcopy(dict(section_values)))
        _acknowledge(screen, SettingsCategoryId.STORAGE, current, submitted)
        screen._storage_result = (
            "Storage defaults saved. Restart Chatbook to use saved paths."
        )
        screen._set_static_text("#settings-storage-save-result", screen._storage_result)
        screen._update_storage_validation_classes()
        screen.app.notify("Storage defaults saved.", severity="information")
        return
    screen._storage_result = "Failed to save Storage defaults."
    screen._set_static_text("#settings-storage-save-result", screen._storage_result)
    screen.app.notify(screen._storage_result, severity="error")


def apply_console_behavior_save_result(
    screen: SettingsScreen,
    saved: bool,
    console_values: Mapping[str, object],
    chat_default_values: Mapping[str, object],
    workbench_scope_fallback: bool = False,
    agents_values: Mapping[str, object] | None = None,
    submitted: Mapping[str, object] | None = None,
) -> None:
    from ..Screens.settings_screen import CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY

    if saved:
        current = capture_staged_values(screen, SettingsCategoryId.CONSOLE_BEHAVIOR)
        if agents_values:
            screen._app_config_update_target().setdefault("agents", {}).update(
                agents_values
            )
        normalized_console_values = dict(console_values)
        if "background_effects" in normalized_console_values:
            screen._console_settings()["background_effects"] = dict(
                normalized_console_values.pop("background_effects")
            )
        screen._console_settings().update(normalized_console_values)
        screen._chat_defaults().update(chat_default_values)
        if "background_effects" in console_values:
            screen._signal_console_appearance_refresh()
        if "user_display_name" in chat_default_values:
            screen._signal_console_identity_refresh()
        _acknowledge(screen, SettingsCategoryId.CONSOLE_BEHAVIOR, current, submitted)
        rail_result = f"Rail labels: {screen._console_rail_label_style_name(loaded=True)}. Return to Console to see the change."
        screen._console_behavior_result = (
            f"Console behavior settings saved. {rail_result}"
        )
        if workbench_scope_fallback:
            screen._console_behavior_result += (
                f" {CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY}"
            )
        screen._console_behavior_saved_this_session = True
        screen._set_static_text(
            "#settings-console-behavior-result", screen._console_behavior_result_text()
        )
        screen._set_static_text(
            "#settings-console-rail-label-style-status",
            screen._console_rail_label_style_status(),
        )
        screen._update_console_paste_summary()
        screen.app.notify("Console behavior settings saved.", severity="information")
        return
    screen._console_behavior_result = (
        "Failed to save Console behavior settings. Your draft is still here; "
        "the active rail-label style is still "
        f"{screen._console_rail_label_style_name(loaded=True)}. Try again."
    )
    screen._set_static_text(
        "#settings-console-behavior-result", screen._console_behavior_result
    )
    screen.app.notify("Failed to save Console behavior settings.", severity="error")
