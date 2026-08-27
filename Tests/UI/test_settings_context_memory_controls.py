from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import asyncio
import threading
from unittest.mock import patch

import pytest
from textual.widgets import Button, Checkbox, Input, Select, Static

import tldw_chatbook
import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
)
from tldw_chatbook.Chat import provider_setup_persistence as provider_persistence_module
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_context_memory import (
    load_show_model_thinking,
    load_thinking_history_policy_default,
    load_context_memory_values,
    model_context_window_save_entry,
    model_context_window_state,
    normalize_context_memory_values,
    resolve_model_context_window,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Utils.token_counter import get_table_model_token_limit


class StyledSettingsDestinationHarness(DestinationHarness):
    """Settings harness using the same bundled stylesheet as the app."""

    CSS_PATH = str(
        Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
    )


def _context_values(**updates: object) -> dict[str, object]:
    values = load_context_memory_values({}).to_mapping()
    values.update(updates)
    return values


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _capture_provider_atomic_writes(monkeypatch):
    calls: list[tuple[dict[str, object], dict[str, tuple[str, ...]]]] = []

    def writer(section_values, *, delete_keys=None):
        calls.append((deepcopy(section_values), deepcopy(delete_keys or {})))
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        provider_persistence_module,
        "apply_settings_mutation_to_cli_config",
        writer,
    )
    return calls


@pytest.mark.parametrize(
    ("console_config", "expected"),
    [
        ({}, True),
        ({"show_model_thinking": False}, False),
        ({"show_model_thinking": "invalid"}, True),
    ],
)
def test_model_thinking_visibility_defaults_on_and_fails_safe(
    console_config: dict[str, object], expected: bool
) -> None:
    assert load_show_model_thinking(console_config) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "auto"),
        ("required", "auto"),
        ("auto", "auto"),
        ("include", "include"),
        ("exclude", "exclude"),
    ],
)
def test_thinking_history_default_normalizes_only_optional_values(
    raw: object, expected: str
) -> None:
    assert (
        load_thinking_history_policy_default(
            {} if raw is None else {"thinking_history_policy_default": raw}
        )
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        ConfigMutationResult(False, False, "before_replace"),
        ConfigMutationResult(
            False,
            False,
            None,
            conflict=True,
            conflict_reason="identity_changed",
        ),
    ],
    ids=["before-replace", "conflict"],
)
async def test_show_model_thinking_is_canonical_immediate_and_rolls_back(
    mutation: ConfigMutationResult,
) -> None:
    app = _build_test_app()
    app.app_config = {"console": {"show_model_thinking": False}}
    captured: list[tuple[bool, int]] = []
    refreshes: list[int] = []

    def _capture_persist(
        _screen: SettingsScreen,
        next_value: bool,
        revision: int,
    ) -> None:
        captured.append((next_value, revision))

    host = DestinationHarness(app, "settings")
    with (
        patch.object(
            SettingsScreen,
            "_settings_persist_thinking_visibility",
            new=_capture_persist,
        ),
        patch.object(
            SettingsScreen,
            "_signal_console_appearance_refresh",
            new=lambda _screen: refreshes.append(1),
        ),
    ):
        async with host.run_test(size=(110, 40)) as pilot:
            await pilot.app.workers.wait_for_complete()
            screen = _active_destination_screen(host)
            screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR.value)
            await pilot.pause()

            toggle = screen.query_one("#settings-console-show-model-thinking", Checkbox)
            assert toggle.value is False
            assert str(toggle.label) == "Show model thinking (Off)"

            toggle.value = True
            await pilot.pause()

            assert app.app_config["console"]["show_model_thinking"] is True
            assert str(toggle.label) == "Show model thinking (On)"
            assert captured == [(True, 1)]
            assert len(refreshes) == 1

            screen._apply_thinking_visibility_persist_result(
                mutation,
                True,
                1,
            )
            await pilot.pause()

            assert app.app_config["console"]["show_model_thinking"] is False
            assert toggle.value is False
            assert str(toggle.label) == "Show model thinking (Off)"
            assert len(refreshes) == 2
            result = screen.query_one("#settings-console-behavior-result", Static)
            assert "prior setting was restored" in _static_text(result)


@pytest.mark.asyncio
async def test_thinking_visibility_successful_noop_confirms_optimistic_value() -> None:
    """Catches treating a structured no-op as a failed write."""

    app = _build_test_app()
    app.app_config = {"console": {"show_model_thinking": False}}

    host = DestinationHarness(app, "settings")
    with patch.object(
        SettingsScreen,
        "_settings_persist_thinking_visibility",
        new=lambda _screen, _value, _revision: None,
    ):
        async with host.run_test(size=(110, 40)) as pilot:
            await pilot.app.workers.wait_for_complete()
            screen = _active_destination_screen(host)
            screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR.value)
            await pilot.pause()
            toggle = screen.query_one("#settings-console-show-model-thinking", Checkbox)

            toggle.value = True
            await pilot.pause()
            screen._apply_thinking_visibility_persist_result(
                ConfigMutationResult(False, False, None),
                True,
                1,
            )

            assert screen._thinking_visibility_confirmed_value is True
            assert app.app_config["console"]["show_model_thinking"] is True
            assert toggle.value is True
            result = screen.query_one("#settings-console-behavior-result", Static)
            assert "visibility saved" in _static_text(result).lower()


@pytest.mark.asyncio
async def test_thinking_visibility_write_drain_coalesces_and_rolls_back_latest_failure() -> (
    None
):
    """Catches overlapping workers persisting stale visibility after rapid toggles."""

    app = _build_test_app()
    app.app_config = {"console": {"show_model_thinking": False}}
    writes: list[tuple[bool, int]] = []
    refreshes: list[int] = []

    def _capture_persist(
        _screen: SettingsScreen,
        value: bool,
        revision: int,
    ) -> None:
        writes.append((value, revision))

    host = DestinationHarness(app, "settings")
    with (
        patch.object(
            SettingsScreen,
            "_settings_persist_thinking_visibility",
            new=_capture_persist,
        ),
        patch.object(
            SettingsScreen,
            "_signal_console_appearance_refresh",
            new=lambda _screen: refreshes.append(1),
        ),
    ):
        async with host.run_test(size=(110, 40)) as pilot:
            await pilot.app.workers.wait_for_complete()
            screen = _active_destination_screen(host)
            screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR.value)
            await pilot.pause()
            toggle = screen.query_one("#settings-console-show-model-thinking", Checkbox)

            toggle.value = True
            await pilot.pause()
            toggle.value = False
            await pilot.pause()

            assert app.app_config["console"]["show_model_thinking"] is False
            assert writes == [(True, 1)]

            screen._apply_thinking_visibility_persist_result(
                ConfigMutationResult(True, False, "cache_reload"),
                True,
                1,
            )
            assert writes == [(True, 1), (False, 2)]
            assert screen._thinking_visibility_confirmed_value is True
            result = screen.query_one("#settings-console-behavior-result", Static)
            assert "saved" in _static_text(result).lower()
            assert "refresh" in _static_text(result).lower()

            screen._apply_thinking_visibility_persist_result(
                ConfigMutationResult(False, False, "before_replace"),
                False,
                2,
            )
            await pilot.pause()

            assert app.app_config["console"]["show_model_thinking"] is True
            assert toggle.value is True
            assert str(toggle.label) == "Show model thinking (On)"
            assert len(refreshes) == 3
            result = screen.query_one("#settings-console-behavior-result", Static)
            assert "prior setting was restored" in _static_text(result)

            # A stale/out-of-order callback cannot change the confirmed disk view.
            screen._apply_thinking_visibility_persist_result(
                ConfigMutationResult(True, False, "cache_reload"),
                True,
                1,
            )
            assert app.app_config["console"]["show_model_thinking"] is True
            assert writes == [(True, 1), (False, 2)]


@pytest.mark.asyncio
async def test_thinking_visibility_overlapping_failure_matches_restart_value() -> None:
    """Proves a failed latest write leaves disk and optimistic state aligned."""

    app = _build_test_app()
    app.app_config = {"console": {"show_model_thinking": False}}
    first_started = threading.Event()
    release_first = threading.Event()
    writes: list[bool] = []
    persisted = {"show_model_thinking": False}

    def fake_mutation(sections) -> ConfigMutationResult:
        value = bool(sections["console"]["show_model_thinking"])
        writes.append(value)
        if len(writes) == 1:
            first_started.set()
            if not release_first.wait(2):
                return ConfigMutationResult(False, False, "before_replace")
            persisted["show_model_thinking"] = value
            return ConfigMutationResult(True, True, None)
        return ConfigMutationResult(False, False, "before_replace")

    host = DestinationHarness(app, "settings")
    with patch.object(
        settings_screen_module,
        "apply_settings_mutation_to_cli_config",
        new=fake_mutation,
    ):
        async with host.run_test(size=(110, 40)) as pilot:
            await pilot.app.workers.wait_for_complete()
            screen = _active_destination_screen(host)
            screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR.value)
            await pilot.pause()
            toggle = screen.query_one("#settings-console-show-model-thinking", Checkbox)

            toggle.value = True
            assert await asyncio.to_thread(first_started.wait, 2)
            toggle.value = False
            await pilot.pause()

            assert writes == [True]
            release_first.set()
            await pilot.app.workers.wait_for_complete()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert writes == [True, False]
            assert persisted["show_model_thinking"] is True
            assert app.app_config["console"]["show_model_thinking"] is True
            assert toggle.value is True
            assert load_show_model_thinking(persisted) is True


def test_context_memory_defaults_and_saved_overrides_share_policy_contract() -> None:
    defaults = load_context_memory_values({})
    assert defaults.conversation_budget_mode == "automatic"
    assert defaults.compaction_mode == "ask"
    assert defaults.compaction_trigger_ratio == 0.80
    assert defaults.compaction_target_ratio == 0.55

    saved = load_context_memory_values(
        {
            "conversation_budget_mode": "custom",
            "conversation_budget_tokens": 64000,
            "compaction_mode": "automatic",
            "compaction_trigger_ratio": 0.85,
            "compaction_target_ratio": 0.60,
            "compaction_summary_max_tokens": 2048,
            "compaction_failure_behavior": "omit_older_context",
            "compaction_carry_forward_mode": "memory_with_latest_exchange",
        }
    )
    assert saved.conversation_budget_tokens == 64000
    assert saved.compaction_mode == "automatic"
    assert saved.compaction_carry_forward_mode == "memory_with_latest_exchange"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"conversation_budget_mode": "custom", "conversation_budget_tokens": ""},
            "requires a positive token value",
        ),
        (
            {"compaction_trigger_ratio": 0.80, "compaction_target_ratio": 0.70},
            "at least 15 percentage points",
        ),
        (
            {"compaction_trigger_ratio": 0.96},
            "no greater than 95%",
        ),
    ],
)
def test_context_memory_validation_rejects_unsafe_combinations(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_context_memory_values(_context_values(**updates))


def test_model_context_window_uses_known_tier_without_presenting_unknown_fallback() -> (
    None
):
    assert get_table_model_token_limit("gpt-4o", "openai") == 128000
    assert resolve_model_context_window({}, "openai", "gpt-4o") == 128000
    assert resolve_model_context_window({}, "openai", "unlisted-model") is None


def test_model_context_window_save_entry_preserves_effective_capabilities() -> None:
    entry = model_context_window_save_entry({}, "openai", "gpt-4o", 256000)
    assert entry["context_window"] == 256000
    assert entry["vision"] is True


def test_model_context_window_state_distinguishes_detection_from_override() -> None:
    detected = model_context_window_state({}, "openai", "gpt-4o")
    assert detected.effective_tokens == 128000
    assert detected.detected_tokens == 128000
    assert detected.configured_override_tokens is None

    configured = model_context_window_state(
        {
            "model_capabilities": {
                "models": {
                    "gpt-4o": {
                        "vision": True,
                        "context_window": 256000,
                    }
                }
            }
        },
        "openai",
        "gpt-4o",
    )
    assert configured.effective_tokens == 256000
    assert configured.detected_tokens == 128000
    assert configured.configured_override_tokens == 256000


@pytest.mark.asyncio
async def test_console_memory_controls_mount_stage_and_fit_narrow_settings() -> None:
    app = _build_test_app()
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(80, 34)) as pilot:
        await pilot.app.workers.wait_for_complete()
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR.value)
        await pilot.pause()

        jump = screen.query_one("#settings-console-context-memory-jump", Button)
        assert "Conversation context and memory" in str(jump.label)
        jump.press()
        await pilot.pause()
        budget_mode = screen.query_one("#settings-console-context-budget-mode", Select)
        assert screen.focused is budget_mode

        conversation_max = screen.query_one(
            "#settings-console-context-budget-tokens",
            Input,
        )
        conversation_label = conversation_max.parent.query_one(
            ".settings-input-label",
            Static,
        )
        response_max = screen.query_one("#settings-console-default-max-tokens", Input)
        response_label = response_max.parent.query_one(
            ".settings-input-label",
            Static,
        )
        assert _static_text(conversation_label) == "Conversation max tokens"
        assert _static_text(response_label) == "Response max tokens"

        trigger = screen.query_one("#settings-console-context-trigger-percent", Input)
        mode = screen.query_one("#settings-console-context-compaction-mode", Select)
        representation = screen.query_one(
            "#settings-console-context-compaction-representation", Select
        )
        assert _static_text(
            trigger.parent.query_one(".settings-input-label", Static)
        ) == "Compact at (%)"
        assert _static_text(
            mode.parent.query_one(".settings-input-label", Static)
        ) == "When limit nears"
        assert _static_text(
            representation.parent.query_one(".settings-input-label", Static)
        ) == "Representation"
        advanced_labels = {
            "#settings-console-context-target-percent": "Reduce conversation to (%)",
            "#settings-console-context-summary-max-tokens": "Summary response max",
            "#settings-console-context-failure-behavior": "If compaction fails",
            "#settings-console-context-carry-forward-mode": "Keep after compaction",
        }
        for selector, expected in advanced_labels.items():
            control = screen.query_one(selector)
            assert _static_text(
                control.parent.query_one(".settings-input-label", Static)
            ) == expected
        trigger.value = "85"
        mode.value = "automatic"
        representation.value = "hybrid"
        await pilot.pause()

        draft = screen._settings_drafts[SettingsCategoryId.CONSOLE_BEHAVIOR]
        assert draft.values["compaction_trigger_ratio"] == 0.85
        assert draft.values["compaction_mode"] == "automatic"
        assert draft.values["compaction_representation"] == "hybrid"
        safety = screen.query_one("#settings-console-context-safety-copy", Static)
        assert "extra model call" in _static_text(safety)
        assert "remain stored" in _static_text(safety)
        assert safety.region.x + safety.region.width <= screen.region.width


@pytest.mark.asyncio
async def test_console_memory_save_blocks_invalid_trigger_target_pair() -> None:
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(110, 40)) as pilot:
        await pilot.app.workers.wait_for_complete()
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR.value)
        await pilot.pause()

        screen._stage_console_default_value("compaction_trigger_ratio", 0.80)
        screen._stage_console_default_value("compaction_target_ratio", 0.70)
        screen.action_settings_save_category(allow_text_entry_focus=True)
        await pilot.pause()

        result = screen.query_one("#settings-console-behavior-result", Static)
        assert "at least 15 percentage points" in _static_text(result)


@pytest.mark.asyncio
async def test_console_memory_save_routes_normalized_values_to_console_section() -> (
    None
):
    app = _build_test_app()
    captured: list[tuple[dict[str, object], dict[str, object], bool]] = []

    def _capture_save(
        _screen,
        console_values,
        chat_default_values,
        workbench_scope_fallback=False,
    ) -> None:
        captured.append(
            (
                dict(console_values),
                dict(chat_default_values),
                bool(workbench_scope_fallback),
            )
        )

    host = DestinationHarness(app, "settings")
    with patch.object(
        SettingsScreen,
        "_settings_save_console_behavior_worker",
        new=_capture_save,
    ):
        async with host.run_test(size=(110, 40)) as pilot:
            await pilot.app.workers.wait_for_complete()
            screen = _active_destination_screen(host)
            screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR.value)
            await pilot.pause()

            screen.query_one(
                "#settings-console-context-budget-mode", Select
            ).value = "custom"
            screen.query_one(
                "#settings-console-context-budget-tokens", Input
            ).value = "64000"
            screen.query_one(
                "#settings-console-context-compaction-mode", Select
            ).value = "automatic"
            screen.query_one(
                "#settings-console-context-compaction-representation", Select
            ).value = "visual_transcript"
            await pilot.pause()
            screen.action_settings_save_category(allow_text_entry_focus=True)

    assert len(captured) == 1
    console_values, chat_defaults, _fallback = captured[0]
    assert console_values["conversation_budget_mode"] == "custom"
    assert console_values["conversation_budget_tokens"] == 64000
    assert console_values["compaction_mode"] == "automatic"
    assert console_values["compaction_representation"] == "visual_transcript"
    assert chat_defaults == {}


@pytest.mark.asyncio
async def test_summary_prompt_route_focuses_existing_internal_prompt_editor() -> None:
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(110, 40)) as pilot:
        await pilot.app.workers.wait_for_complete()
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR.value)
        await pilot.pause()

        screen.query_one(
            "#settings-console-context-edit-summary-prompt", Button
        ).press()
        await pilot.pause()
        await pilot.pause()

        assert screen.active_category == SettingsCategoryId.INTERNAL_PROMPTS.value
        search = screen.query_one("#internal-prompts-search", Input)
        assert search.value == "console.rewind_summarize"
        focused = screen.focused
        assert focused is not None
        assert focused.id == "prompt-row-console__rewind_summarize"


@pytest.mark.asyncio
async def test_provider_context_window_repair_saves_to_model_capability_authority(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {
            "openai": {
                "api_base_url": "https://proxy.example.test/v1",
                "api_key_env_var": "OPENAI_TEST_KEY",
                "model": "gpt-4o",
            }
        },
        "provider_setup": {"confirmed": {"openai": True, "custom": True}},
        "model_capabilities": {},
    }
    connection_before = deepcopy(
        {
            key: app.app_config[key]
            for key in ("chat_defaults", "api_settings", "provider_setup")
        }
    )
    atomic_calls = _capture_provider_atomic_writes(monkeypatch)

    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(110, 40)) as pilot:
        await pilot.app.workers.wait_for_complete()
        screen: SettingsScreen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await pilot.pause()

        context_input = screen.query_one("#settings-model-context-window", Input)
        assert context_input.value == "128000"
        context_input.value = "256000"
        await pilot.pause()
        screen.action_settings_save_category(allow_text_entry_focus=True)
        await pilot.pause()

    assert len(atomic_calls) == 1
    sections, deletes = atomic_calls[0]
    assert set(sections) == {"model_capabilities.models"}
    assert deletes == {}
    saved_entry = sections["model_capabilities.models"]["gpt-4o"]
    assert saved_entry["context_window"] == 256000
    assert saved_entry["vision"] is True
    assert app.app_config["model_capabilities"]["models"]["gpt-4o"] == saved_entry
    assert {
        key: app.app_config[key]
        for key in ("chat_defaults", "api_settings", "provider_setup")
    } == connection_before


@pytest.mark.asyncio
async def test_provider_context_window_reset_preserves_other_capabilities(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {
            "openai": {
                "api_base_url": "https://proxy.example.test/v1",
                "api_key_env_var": "OPENAI_TEST_KEY",
                "model": "gpt-4o",
            }
        },
        "provider_setup": {"confirmed": {"openai": True, "custom": True}},
        "model_capabilities": {
            "models": {
                "gpt-4o": {
                    "vision": True,
                    "max_images": 10,
                    "context_window": 256000,
                }
            }
        },
    }
    connection_before = deepcopy(
        {
            key: app.app_config[key]
            for key in ("chat_defaults", "api_settings", "provider_setup")
        }
    )
    atomic_calls = _capture_provider_atomic_writes(monkeypatch)

    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(110, 40)) as pilot:
        await pilot.app.workers.wait_for_complete()
        screen: SettingsScreen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await pilot.pause()

        context_input = screen.query_one("#settings-model-context-window", Input)
        status = screen.query_one("#settings-model-context-window-status", Static)
        reset = screen.query_one("#settings-model-context-window-reset", Button)
        assert context_input.value == "256000"
        assert "Configured override: 256,000" in _static_text(status)
        assert reset.disabled is False

        reset.press()
        await pilot.pause()
        assert context_input.value == "128000"
        screen.action_settings_save_category(allow_text_entry_focus=True)
        await pilot.pause()

    assert atomic_calls == [
        (
            {
                "model_capabilities.models": {
                    "gpt-4o": {"vision": True, "max_images": 10}
                }
            },
            {},
        )
    ]
    assert app.app_config["model_capabilities"]["models"]["gpt-4o"] == {
        "vision": True,
        "max_images": 10,
    }
    assert {
        key: app.app_config[key]
        for key in ("chat_defaults", "api_settings", "provider_setup")
    } == connection_before
