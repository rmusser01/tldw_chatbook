"""One field table labels every model-configuration field (TASK-33002.1).

The popover, the Conversation settings modal, Settings Providers & Models
model defaults and Settings Console Behavior fallbacks each rendered their
own spelling of the same field ("Think budget" / "Budget" / "Thinking
budget"). These tests pin the table itself and collect every surface's
rendered labels against it, so a new private spelling fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from textual.widget import Widget
from textual.widgets import Checkbox, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_model_popover_geometry import PopoverGeometryHarness
from Tests.UI.test_console_model_switcher import Recorder, build_switcher
from Tests.UI.test_console_session_settings import StyledModalHarness, _basic_modal
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _wait_for_selector,
)
from Tests.UI.test_settings_configuration_hub import (
    _capture_provider_settings_mutations,
)
from tldw_chatbook.Chat.console_provider_support import (
    CARRY_FORWARD_OPTIONS,
    GENERATION_FIELD_REQUEST_KEYS,
    MODEL_CONFIG_FIELDS,
    MODEL_FIELD_LABELS,
    build_local_thinking_payload_fields,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    validate_console_session_settings,
)
from tldw_chatbook.Chat.console_settings_apply import FULL_MODEL_DEFAULT_FIELDS
from tldw_chatbook.Chat.provider_catalog import provider_display_name
from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    AUTO_REFRESH_PROVIDER_LIST_KEYS,
)
from tldw_chatbook.UI.Screens import settings_search_index
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import (
    MODEL_PROFILE_INPUT_PLACEHOLDERS,
    SettingsScreen,
)
from tldw_chatbook.Widgets.Console import console_settings_modal
from tldw_chatbook.Widgets.Console.console_settings_field_row import FIELD_ROW_FIELDS

#: The label columns in css/features/_console_panels.tcss: the Model view's
#: field rows ($ds-size-18) and the shared modal label ($ds-size-23); the
#: Context view's ($ds-size-24) is wider than the shared one.
_FIELD_ROW_LABEL_CELLS = 18
_MODAL_LABEL_CELLS = 23

_GENERATION_CONTROL_IDS = {
    "temperature": "temperature",
    "top-p": "top_p",
    "min-p": "min_p",
    "top-k": "top_k",
    "max-tokens": "max_tokens",
    "seed": "seed",
    "presence-penalty": "presence_penalty",
    "frequency-penalty": "frequency_penalty",
    "reasoning-effort": "reasoning_effort",
    "reasoning-summary": "reasoning_summary",
    "verbosity": "verbosity",
    "thinking-effort": "thinking_effort",
    "thinking-budget-tokens": "thinking_budget_tokens",
}


def _label_before(control: Widget, label_class: str) -> str:
    """Return the text of the nearest preceding sibling label of ``control``."""
    siblings = list(control.parent.children)
    for widget in reversed(siblings[: siblings.index(control)]):
        if isinstance(widget, Static) and widget.has_class(label_class):
            renderable = widget.renderable
            return str(getattr(renderable, "plain", renderable))
    raise AssertionError(f"no {label_class} label before #{control.id}")


def _select_options(screen, select_id: str) -> tuple:
    """Return a Select's (label, value) options without the blank entry."""
    select = screen.query_one(f"#{select_id}", Select)
    return tuple(
        (str(label), value) for label, value in select._options if value != Select.NULL
    )


def _label_drift(screen, controls: dict[str, str], label_class: str) -> list:
    """Collect (control id, rendered label, table label) for every mismatch."""
    drift = []
    for control_id, field_name in controls.items():
        control = screen.query_one(f"#{control_id}")
        rendered = _label_before(control, label_class)
        if rendered != MODEL_FIELD_LABELS[field_name]:
            drift.append((control_id, rendered, MODEL_FIELD_LABELS[field_name]))
    return drift


def test_field_table_covers_every_model_configuration_field():
    """AC#1: every model-default field plus Endpoint and the budget pair."""
    assert set(MODEL_CONFIG_FIELDS) == set(FULL_MODEL_DEFAULT_FIELDS) | {
        "endpoint",
        "conversation_budget_mode",
        "compaction_mode",
        # Captures (TASK-33002 follow-up): the two Context-view rows the modal
        # and Console Behavior spelled differently.
        "compaction_target_ratio",
        "compaction_carry_forward_mode",
    }
    for name, field in MODEL_CONFIG_FIELDS.items():
        assert field.name == name
        assert field.label and field.help and field.valid_range, name
        assert "\n" not in field.help, name


def test_generation_fields_carry_phase_one_request_keys_not_a_copy():
    """AC#1: request keys are read from the one definition, never re-typed."""
    for name, field in MODEL_CONFIG_FIELDS.items():
        if name in GENERATION_FIELD_REQUEST_KEYS:
            assert field.request_keys is GENERATION_FIELD_REQUEST_KEYS[name]
        else:
            assert field.request_keys == ()


def test_drift_pairs_resolve_to_one_label_each():
    """AC#4: the table's answer for every pair the editors disagreed on."""
    assert MODEL_FIELD_LABELS["thinking_budget_tokens"] == "Thinking budget"
    assert MODEL_FIELD_LABELS["endpoint"] == "Endpoint"
    assert MODEL_FIELD_LABELS["max_tokens"] == "Max tokens"
    assert MODEL_FIELD_LABELS["presence_penalty"] == "Presence penalty"
    assert MODEL_FIELD_LABELS["frequency_penalty"] == "Frequency penalty"
    assert MODEL_FIELD_LABELS["conversation_budget_mode"] == "Budget strategy"
    assert MODEL_FIELD_LABELS["compaction_mode"] == "When limit nears"


def test_labels_fit_the_existing_label_columns():
    """Parent AC#8, strict (TASK-33006.6 review item 5): every label is
    narrower than the column it paints in, so it never touches its control."""
    from rich.cells import cell_len

    for name, label in MODEL_FIELD_LABELS.items():
        column = _FIELD_ROW_LABEL_CELLS if name in FIELD_ROW_FIELDS else _MODAL_LABEL_CELLS
        assert cell_len(label) < column, (name, label)


def test_modal_settings_paths_name_real_settings_categories():
    """Captures flag 6: the Context view sent users to "F4 Settings > Console
    behavior", a category the Settings rail spells "Console Behavior".
    Checked against the rail's own titles, in any case (gap review Minor 5)."""
    screen = SettingsScreen.__new__(SettingsScreen)
    screen._internal_prompts_customized_count = 0
    categories = {summary.title for summary in screen._category_summaries()}
    source = Path(console_settings_modal.__file__).read_text(encoding="utf-8")
    paths = re.findall(r"F4 Settings > (\w[\w &]*\w)", source)
    assert paths
    assert set(paths) <= categories, set(paths) - categories


def test_model_list_refresh_search_rows_use_display_names():
    """Gap review Minor 4: "/" search spelled "MistralAI auto-refresh model
    list" while the checkbox said "Mistral AI: refresh"."""
    settings_search_index.build_field_search_index()
    entries = settings_search_index.FIELD_SEARCH_INDEX[
        SettingsCategoryId.PROVIDERS_MODELS
    ]

    assert (
        "settings-mc-auto-mistralai",
        "Mistral AI auto-refresh model list",
    ) in entries
    assert ("settings-mc-write-zai", "Z.ai save fetched models to config") in entries


def test_help_lines_use_plain_words_not_config_keys():
    """AC#6: no config key, request key or dotted config path in help."""
    for name, field in MODEL_CONFIG_FIELDS.items():
        assert "_" not in field.help, name
        assert "chat_defaults" not in field.help, name
        for key in field.request_keys:
            assert key not in field.help.lower().split(), name


def test_console_validation_names_fields_by_their_table_labels():
    """AC#4: the modal's errors name Max tokens and Endpoint as the rows do."""
    errors = validate_console_session_settings(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            max_tokens=0,
            base_url="not a url",
            thinking_effort="turbo",
            thinking_budget_tokens=10,
        ),
        app_config={
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}
        },
    )
    assert f"{MODEL_FIELD_LABELS['max_tokens']} must be 1 or greater." in errors
    assert f"{MODEL_FIELD_LABELS['endpoint']} must be a valid http(s) URL." in errors
    # Final review (Task 1 minor): the two thinking errors named the fields
    # "Thinking effort" and "Thinking budget tokens".
    assert (
        f"{MODEL_FIELD_LABELS['thinking_effort']} must be one of off, low, "
        "medium, high, xhigh, or max."
    ) in errors
    assert (
        f"{MODEL_FIELD_LABELS['thinking_budget_tokens']} must be at least 1024."
    ) in errors


@pytest.mark.parametrize(
    ("provider", "app_config"),
    [
        ("llama_cpp", {}),
        ("local_mlx_lm", {}),
        (
            "custom-ep:gpu-box",
            {
                "custom_endpoints": {
                    "gpu-box": {
                        "display_name": "GPU box",
                        "base_url": "http://127.0.0.1:8080",
                        "family": "llama_cpp",
                    }
                }
            },
        ),
    ],
)
def test_local_reasoning_select_offers_only_levels_the_request_sends(
    provider, app_config
):
    """Phase-1 rider: a strict local template drops "minimal" with only a
    debug log, so Settings' Reasoning select must not offer it. Checked
    against the real request builder, not a copied list."""
    options = SettingsScreen._model_profile_reasoning_effort_options(
        provider, "qwen", app_config
    )

    assert "minimal" not in options
    assert options
    for value in options:
        assert build_local_thinking_payload_fields("llama_cpp", value, None), value


def test_hosted_reasoning_select_keeps_minimal():
    """Hosted providers forward "minimal"; their list is unchanged."""
    options = SettingsScreen._model_profile_reasoning_effort_options

    assert "minimal" in options("openai", "gpt-5")
    assert "minimal" in options("custom", "any")


@pytest.mark.asyncio
async def test_popover_labels_come_from_the_field_table():
    """AC#2/#3: the Alt+M popover's value labels.

    Rewritten for TASK-33004.5: the value row labels Temperature, Max tokens
    and Streaming before their controls (an Input, an Input, an On/Off
    Select), all from the one field table. Captures flag 5 ("Response max"
    where the modal and Settings say Max tokens) stays fixed by this."""
    app = PopoverGeometryHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        await app.push_screen(build_switcher(Recorder()))
        await pilot.pause()
        screen = app.screen
        assert (
            _label_drift(
                screen,
                {
                    "console-popover-temperature": "temperature",
                    "console-popover-max-tokens": "max_tokens",
                    "console-popover-streaming": "streaming",
                },
                "console-popover-field-label",
            )
            == []
        )


@pytest.mark.asyncio
async def test_conversation_settings_modal_labels_come_from_the_field_table():
    """AC#2/#3: every Model-view and Context-view field label in the modal."""
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="model-a", base_url="http://127.0.0.1:9099"
    )
    # TASK-33006.1: Model view field rows label with their own 18-cell
    # class; Connection and the Context view keep the shared 23-cell one.
    field_rows = {
        f"console-settings-{suffix}": name
        for suffix, name in _GENERATION_CONTROL_IDS.items()
    }
    field_rows["console-settings-streaming"] = "streaming"
    controls = {
        "console-settings-base-url": "endpoint",
        "console-context-budget-mode": "conversation_budget_mode",
        "console-context-compaction-mode": "compaction_mode",
        "console-context-target-percent": "compaction_target_ratio",
        "console-context-carry-forward": "compaction_carry_forward_mode",
    }
    async with app.run_test(size=(211, 44)) as pilot:
        await app.push_screen(_basic_modal(settings, app))
        await pilot.pause()
        assert _label_drift(app.screen, field_rows, "console-settings-field-label") == []
        assert _label_drift(app.screen, controls, "console-settings-modal-label") == []
        assert (
            _select_options(app.screen, "console-context-carry-forward")
            == CARRY_FORWARD_OPTIONS
        )


@pytest.mark.asyncio
@private_profile_test
async def test_settings_model_default_labels_and_inspector_come_from_the_table(
    request,
):
    """AC#2/#3/#5: P&M form rows, dirty-field names and the inspector."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    host = DestinationHarness(app, "settings")
    controls = {
        f"settings-model-profile-{suffix}": name
        for suffix, name in _GENERATION_CONTROL_IDS.items()
    }
    controls |= {
        "settings-model-profile-streaming": "streaming",
        "settings-provider-endpoint-value": "endpoint",
    }
    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await _wait_for_selector(screen, pilot, "#settings-model-profile-temperature")
        await pilot.pause()
        assert _label_drift(screen, controls, "settings-input-label") == []

        # Inspector: the focused model-default field shows the table's help
        # and range (and the dirty-field names reuse the same labels).
        field = MODEL_CONFIG_FIELDS["presence_penalty"]
        screen._active_settings_field_id = "settings-model-profile-presence-penalty"
        rows = dict(screen._provider_field_guidance_rows())
        assert rows["Focused setting"] == field.label
        assert rows["Purpose"] == field.help
        assert field.valid_range in rows["Validation"]
        # Captures flag 10: "Saved as" printed "<model>" for the model in
        # the form.
        assert rows["Saved as"] == (
            "api_settings.openai.model_defaults.gpt-4.1.presence_penalty"
        )
        # No focused field (the live crash found at 211x44): generic rows.
        screen._active_settings_field_id = None
        assert screen._provider_field_guidance_rows()

        screen._stage_provider_value("model_profile_presence_penalty", "0.5")
        screen._stage_provider_value("model_profile_max_tokens", "512")
        assert set(screen._provider_return_dirty_field_names()) == {
            MODEL_FIELD_LABELS["presence_penalty"],
            MODEL_FIELD_LABELS["max_tokens"],
        }

        # Captures flag 7 (parent AC#5): the Automatic refresh list named
        # providers by their list keys ("MistralAI", "Moonshot", "ZAI").
        # TASK-33007.6, rewritten on purpose: each label now ends in its
        # state as a word (Catalog refresh, AC#5).
        boxes = {
            provider: screen.query_one(
                f"#settings-mc-auto-{provider.lower()}", Checkbox
            )
            for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS
        }
        assert {provider: str(box.label) for provider, box in boxes.items()} == {
            provider: (
                f"{provider_display_name(provider)}: refresh "
                f"{'On' if box.value else 'Off'}"
            )
            for provider, box in boxes.items()
        }


@pytest.mark.asyncio
@private_profile_test
async def test_settings_console_behavior_fallback_labels_come_from_the_table(
    request,
):
    """AC#2/#3: Console Behavior's global fallbacks and context pair."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    controls = {
        f"settings-console-default-{suffix}": name
        for suffix, name in _GENERATION_CONTROL_IDS.items()
    }
    controls |= {
        "settings-console-default-streaming": "streaming",
        "settings-console-context-budget-mode": "conversation_budget_mode",
        "settings-console-context-compaction-mode": "compaction_mode",
        "settings-console-context-target-percent": "compaction_target_ratio",
        "settings-console-context-carry-forward-mode": "compaction_carry_forward_mode",
    }
    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR.value)
        await _wait_for_selector(screen, pilot, "#settings-console-default-temperature")
        await pilot.pause()
        assert _label_drift(screen, controls, "settings-input-label") == []
        assert (
            _select_options(screen, "settings-console-context-carry-forward-mode")
            == CARRY_FORWARD_OPTIONS
        )
        # Captures flag 5: a fallback's placeholder matches the same field's
        # placeholder in Providers & Models ("optional deterministic seed").
        placeholders = {
            suffix: screen.query_one(f"#settings-console-default-{suffix}").placeholder
            for suffix, name in _GENERATION_CONTROL_IDS.items()
            if f"model_profile_{name}" in MODEL_PROFILE_INPUT_PLACEHOLDERS
        }
        assert placeholders == {
            suffix: MODEL_PROFILE_INPUT_PLACEHOLDERS[f"model_profile_{name}"]
            for suffix, name in _GENERATION_CONTROL_IDS.items()
            if f"model_profile_{name}" in MODEL_PROFILE_INPUT_PLACEHOLDERS
        }
        # Captures flag 10: the inspector's Scope row lost its bracketed
        # section names to markup ("Scope:  response fallbacks and  paste").
        rendered = [
            str(static.visual) for static in screen.query(".settings-detail-row")
        ]
        scope = [text for text in rendered if text.startswith("Scope: ")]
        assert scope and all("  " not in text for text in scope), scope

        # Final review I2: focusing a fallback shows the table's help and
        # range, as the Providers & Models inspector does.
        screen.query_one("#settings-console-default-temperature").focus()
        await pilot.pause()
        field = MODEL_CONFIG_FIELDS["temperature"]
        guide = [
            str(screen.query_one(f"#settings-console-behavior-field-guide-{i}").content)
            for i in range(4)
        ]
        assert guide == [
            f"Focused setting: {field.label}",
            f"Purpose: {field.help}",
            "Saved as: chat_defaults.temperature",
            f"Validation: {field.valid_range}",
        ]
        for name in GENERATION_FIELD_REQUEST_KEYS:
            screen._active_settings_field_id = (
                f"settings-console-default-{name.replace('_', '-')}"
            )
            rows = dict(screen._console_behavior_field_guidance_rows())
            assert rows["Purpose"] == MODEL_CONFIG_FIELDS[name].help, name
        # The Control guide rows read the same help lines.
        guide_text = " ".join(str(static.content) for static in screen.query(Static))
        for name in ("streaming", "temperature", "top_p", "max_tokens"):
            assert MODEL_CONFIG_FIELDS[name].help in guide_text, name


def _llamacpp_app_with_saved_minimal():
    """A llama.cpp profile saved with "minimal" before the list was narrowed."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "qwen"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099/v1/chat/completions",
            "model_defaults": {
                "qwen": {"reasoning_effort": "minimal", "temperature": 0.5}
            },
        }
    }
    return app


async def _open_llamacpp_model_defaults(host, pilot):
    screen = _active_destination_screen(host)
    screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
    await _wait_for_selector(screen, pilot, "#settings-model-profile-temperature")
    await pilot.pause()
    return screen


@pytest.mark.asyncio
@private_profile_test
async def test_saved_minimal_on_llamacpp_survives_revert(request):
    """Final review C1: Revert assigned "minimal" to a Select that no longer
    offered it, and Textual's InvalidSelectValueError took the app down. The
    saved value stays as a labelled option in all three Select writers."""
    host = DestinationHarness(_llamacpp_app_with_saved_minimal(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = await _open_llamacpp_model_defaults(host, pilot)
        select = screen.query_one("#settings-model-profile-reasoning-effort", Select)
        # Another model's profile rebuilds the options without "minimal".
        screen.query_one("#settings-model-value", Input).value = "other-model"
        await pilot.pause()
        await pilot.pause()
        assert "minimal" not in [value for _label, value in select._options]
        screen.action_settings_revert_category()
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()
        await pilot.pause()
        assert select.value == "minimal"
        assert ("minimal (not supported here)", "minimal") in select._options
        assert screen.query_one("#settings-model-value", Input).value == "qwen"


@pytest.mark.asyncio
@private_profile_test
async def test_saved_minimal_on_llamacpp_survives_an_unrelated_save(
    request, monkeypatch
):
    """Final review C1: a Save that touched only Temperature rebuilt the
    profile from a Select showing "Inherit default" and deleted "minimal"."""
    mutations = _capture_provider_settings_mutations(monkeypatch)
    host = DestinationHarness(_llamacpp_app_with_saved_minimal(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = await _open_llamacpp_model_defaults(host, pilot)
        screen.query_one("#settings-model-profile-temperature", Input).value = "0.9"
        await pilot.pause()
        await pilot.click("#settings-save-category")
        await pilot.pause()
        await pilot.pause()
    writes = [
        values["api_settings.llama_cpp"]["model_defaults"]["qwen"]
        for values, _deletes in mutations
        if "api_settings.llama_cpp" in values
    ]
    assert writes == [{"reasoning_effort": "minimal", "temperature": 0.9}]
