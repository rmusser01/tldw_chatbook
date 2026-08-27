"""Mounted contracts for current-conversation context controls."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Button, OptionList, Select, Static

from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCompactionMode,
    ContextCompactionRepresentation,
)
from tldw_chatbook.Chat.console_context_repository import ConsoleMemoryRecord
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.Widgets.Console.console_context_controls import (
    build_console_context_control_state,
)
from tldw_chatbook.Widgets.Console.console_model_popover import (
    ConsoleModelPopover,
    ConsoleModelPopoverResult,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    ConsoleSettingsModal,
    ConsoleSettingsResult,
)


def _settings() -> ConsoleSessionSettings:
    return ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        max_tokens=4_000,
    )


def _memory() -> ConsoleMemoryRecord:
    return ConsoleMemoryRecord(
        memory_id="memory-1",
        conversation_id="conversation-1",
        boundary_message_id="message-4",
        captured_leaf_message_id="message-8",
        lineage_json='["message-1", "message-4", "message-8"]',
        summary_text="The user chose the local-first deployment plan.",
        provider="llama_cpp",
        model="model-a",
        prompt_id="console.rewind_summarize",
        prompt_revision=2,
        prompt_digest="prompt-digest",
        selected_units_json='["message-1", "message-4"]',
        summarized_prefix_digest="prefix-digest",
        input_tokens=12_000,
        output_tokens=700,
        before_tokens=52_000,
        after_tokens=24_000,
        created_at="2026-08-10T20:00:00+00:00",
    )


def _state(
    *,
    memory: ConsoleMemoryRecord | None = None,
    thinking_policy: str = "auto",
    required_reason: str | None = None,
):
    return build_console_context_control_state(
        settings=_settings(),
        estimate=ConsoleSettingsContextEstimate(
            used_tokens=42_000,
            token_limit=100_000,
            label="42,000 / 100,000 tokens",
        ),
        overrides=ConsoleContextPolicyOverrides(),
        conversation_tokens=32_000,
        request_overhead_tokens=10_000,
        safety_margin_tokens=2_000,
        active_memory=memory,
        thinking_history_policy=thinking_policy,
        thinking_history_required_reason=required_reason,
    )


class _ContextHarness(App[None]):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def __init__(self) -> None:
        super().__init__()
        self.result = None
        self.reset_calls = 0
        self.undo_calls: list[tuple[str, int]] = []
        self.reset_all_calls = 0

    def capture(self, result) -> None:
        self.result = result

    def reset_current(self) -> tuple[str, int]:
        self.reset_calls += 1
        return "memory-1", 2

    def undo_current(self, memory_id: str, revision: int) -> bool:
        self.undo_calls.append((memory_id, revision))
        return True

    def reset_all(self) -> int:
        self.reset_all_calls += 1
        return 3


@pytest.mark.asyncio
async def test_quick_popover_separates_request_conversation_and_policy() -> None:
    app = _ContextHarness()
    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(
            ConsoleModelPopover(
                settings=_settings(),
                providers_models={"llama_cpp": ["model-a"]},
                context_state=_state(),
            ),
            callback=app.capture,
        )
        assert "~42,000 / 94,000 safe input" in str(
            app.screen.query_one("#console-popover-request-usage", Static).renderable
        )
        assert "~32,000 / 84,000 max tokens" in str(
            app.screen.query_one(
                "#console-popover-conversation-usage", Static
            ).renderable
        )
        assert "4,000 tokens for the next reply" in str(
            app.screen.query_one("#console-popover-response-max", Static).renderable
        )
        assert "Automatic may add one extra model call" in str(
            app.screen.query_one(
                "#console-popover-compaction-help",
                Static,
            ).renderable
        )
        assert not app.screen.query("#console-popover-custom-budget")
        app.screen.query_one(
            "#console-popover-compaction-mode", Select
        ).value = ContextCompactionMode.AUTOMATIC.value
        await pilot.click("#console-popover-apply")
        await pilot.pause()

    assert isinstance(app.result, ConsoleModelPopoverResult)
    assert app.result.compaction_mode is ContextCompactionMode.AUTOMATIC


@pytest.mark.asyncio
async def test_full_modal_has_stable_views_and_saves_conversation_policy() -> None:
    app = _ContextHarness()
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=_state(memory=_memory()),
                can_save=True,
                focus_context=True,
            ),
            callback=app.capture,
        )
        assert not app.screen.query_one(
            "#console-settings-provider-model-section"
        ).display
        assert app.screen.query_one("#console-settings-context-view").display
        assert "local-first deployment" in str(
            app.screen.query_one("#console-settings-memory-review", Static).renderable
        )
        save_defaults = app.screen.query_one(
            "#console-settings-save-default",
            Button,
        )
        assert str(save_defaults.label) == "Save model defaults"
        assert save_defaults.display is False
        scope = str(
            app.screen.query_one("#console-settings-scope", Static).renderable
        )
        assert "this conversation" in scope
        assert "F9 Settings > Console behavior" in scope
        context_labels = {
            "#console-context-custom-budget": "Conversation max tokens",
            "#console-context-trigger-percent": "Compact at (%)",
            "#console-context-target-percent": "Reduce conversation to (%)",
            "#console-context-summary-max": "Summary response max",
            "#console-context-failure-behavior": "If compaction fails",
            "#console-context-carry-forward": "Keep after compaction",
            "#console-context-compaction-representation": "Representation",
        }
        for selector, expected in context_labels.items():
            control = app.screen.query_one(selector)
            label = control.parent.query_one(".console-settings-modal-label", Static)
            assert str(label.renderable) == expected
        representation = app.screen.query_one(
            "#console-context-compaction-representation", Select
        )
        representation_options = representation.query_one(OptionList)
        assert representation_options.get_option_at_index(1).disabled
        assert representation_options.get_option_at_index(2).disabled
        assert "vision-capable" in str(
            app.screen.query_one(
                "#console-context-representation-status", Static
            ).renderable
        )

        app.screen.query_one(
            "#console-context-budget-mode", Select
        ).value = ContextBudgetMode.CUSTOM.value
        app.screen.query_one("#console-context-custom-budget").value = "70000"
        app.screen.query_one(
            "#console-context-compaction-mode", Select
        ).value = ContextCompactionMode.AUTOMATIC.value
        await pilot.click("#console-settings-save")
        await pilot.pause()

    assert isinstance(app.result, ConsoleSettingsResult)
    assert (
        app.result.context_policy_overrides.compaction_mode
        is ContextCompactionMode.AUTOMATIC
    )
    assert app.result.context_policy_overrides.custom_budget_tokens == 70_000
    assert app.result.thinking_history_policy == "auto"


@pytest.mark.asyncio
async def test_thinking_history_required_preserves_saved_value_and_disables_edit() -> (
    None
):
    app = _ContextHarness()
    state = _state(
        thinking_policy="exclude",
        required_reason="Active provider continuation must be replayed.",
    )
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=state,
                can_save=True,
                focus_context=True,
            ),
            callback=app.capture,
        )
        select = app.screen.query_one(
            "#console-context-thinking-history-policy", Select
        )
        assert select.value == "exclude"
        assert select.disabled
        effective = str(
            app.screen.query_one(
                "#console-context-thinking-history-effective", Static
            ).renderable
        )
        assert "Effective: Required" in effective
        assert "provider continuation" in effective
        await pilot.click("#console-settings-cancel")


@pytest.mark.asyncio
async def test_thinking_history_default_write_is_bounded_and_live_for_new_chats(
    monkeypatch,
) -> None:
    from tldw_chatbook.Widgets.Console import console_settings_modal as modal_module

    writes: list[dict[str, dict[str, object]]] = []
    monkeypatch.setattr(
        modal_module,
        "save_settings_to_cli_config",
        lambda sections: writes.append(sections) or True,
    )
    app_config: dict[str, object] = {"api_settings": {"llama_cpp": {}}}
    app = _ContextHarness()
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config=app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=_state(thinking_policy="auto"),
                can_save=True,
                focus_context=True,
            ),
            callback=app.capture,
        )
        app.screen.query_one(
            "#console-context-thinking-history-policy", Select
        ).value = "exclude"
        app.screen.query_one(
            "#console-context-thinking-history-save-default", Button
        ).press()
        await pilot.pause()

        assert isinstance(app.screen, ConsoleSettingsModal)
        status = app.screen.query_one("#console-context-action-status", Static)
        assert "new conversations only" in str(status.renderable)
        await pilot.click("#console-settings-cancel")

    assert writes == [{"console": {"thinking_history_policy_default": "exclude"}}]
    assert app_config["console"] == {"thinking_history_policy_default": "exclude"}


@pytest.mark.asyncio
async def test_visual_representation_choices_enable_for_vision_model() -> None:
    app = _ContextHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="gpt-4o",
        max_tokens=4_000,
    )
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["gpt-4o"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=build_console_context_control_state(
                    settings=settings,
                    estimate=ConsoleSettingsContextEstimate(
                        42_000, 100_000, "42,000 / 100,000 tokens"
                    ),
                ),
                can_save=True,
                focus_context=True,
            ),
            callback=app.capture,
        )
        representation = app.screen.query_one(
            "#console-context-compaction-representation", Select
        )
        options = representation.query_one(OptionList)
        assert not options.get_option_at_index(1).disabled
        assert not options.get_option_at_index(2).disabled
        representation.value = ContextCompactionRepresentation.HYBRID.value
        await pilot.click("#console-settings-save")
        await pilot.pause()

    assert isinstance(app.result, ConsoleSettingsResult)
    assert (
        app.result.context_policy_overrides.compaction_representation
        is ContextCompactionRepresentation.HYBRID
    )


@pytest.mark.asyncio
async def test_provider_defaults_write_excludes_memory_and_prompt_ownership(
    monkeypatch,
) -> None:
    from tldw_chatbook.Widgets.Console import console_settings_modal as modal_module

    writes: list[dict[str, dict[str, object]]] = []
    monkeypatch.setattr(
        modal_module,
        "save_settings_to_cli_config",
        lambda sections: writes.append(sections) or True,
    )
    app = _ContextHarness()
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=_state(),
                can_save=True,
                focus_context=True,
            ),
            callback=app.capture,
        )
        app.screen.query_one(
            "#console-context-compaction-mode", Select
        ).value = ContextCompactionMode.AUTOMATIC.value
        await pilot.click("#console-settings-view-model")
        await pilot.pause()
        await pilot.click("#console-settings-save-default")
        await pilot.pause()

    assert len(writes) == 1
    assert set(writes[0]) <= {
        "api_settings.llama_cpp",
        "console.provider_defaults.llama_cpp",
        "chat_defaults",
    }
    serialized_keys = " ".join(
        f"{section} {' '.join(values)}" for section, values in writes[0].items()
    ).lower()
    assert "memory" not in serialized_keys
    assert "prompt" not in serialized_keys
    assert app.result.context_policy_overrides.compaction_mode is (
        ContextCompactionMode.AUTOMATIC
    )


@pytest.mark.asyncio
async def test_branch_reset_is_undoable_and_reset_all_is_separately_confirmed() -> None:
    app = _ContextHarness()
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=_state(memory=_memory()),
                can_save=True,
                focus_context=True,
                reset_current_memory=app.reset_current,
                undo_current_memory_reset=app.undo_current,
                reset_all_memories=app.reset_all,
            )
        )
        reset_current = app.screen.query_one("#console-context-reset-current", Button)
        reset_current.press()
        await pilot.pause()
        assert app.reset_calls == 1
        assert app.screen.query_one("#console-context-undo-reset", Button).display
        app.screen.query_one("#console-context-undo-reset", Button).press()
        await pilot.pause()
        assert app.undo_calls == [("memory-1", 2)]

        reset_all = app.screen.query_one("#console-context-reset-all", Button)
        reset_all.press()
        await pilot.pause()
        assert app.reset_all_calls == 0
        status = str(
            app.screen.query_one("#console-context-action-status", Static).renderable
        )
        assert "every branch" in status
        assert "Transcript messages will not change" in status
        app.screen.query_one("#console-context-confirm-reset-all", Button).press()
        await pilot.pause()
        assert app.reset_all_calls == 1
        assert not app.screen.query_one("#console-context-undo-reset", Button).display


def test_context_controls_add_no_forbidden_keybindings() -> None:
    forbidden = {
        "ctrl+c",
        "ctrl+v",
        "ctrl+x",
        "ctrl+s",
        "ctrl+d",
        "ctrl+z",
        "ctrl+a",
        "ctrl+r",
        "ctrl+w",
        "ctrl+p",
        "ctrl+q",
        "f1",
        "f6",
    }
    keys = {
        key
        for binding in (
            *ConsoleModelPopover.BINDINGS,
            *ConsoleSettingsModal.BINDINGS,
        )
        for key in str(
            binding.key if hasattr(binding, "key") else binding[0]
        ).split(",")
    }
    assert keys.isdisjoint(forbidden)


@pytest.mark.asyncio
async def test_context_view_fits_narrow_terminal_and_keeps_focusable_controls() -> None:
    app = _ContextHarness()
    async with app.run_test(size=(72, 24)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=_state(),
                can_save=True,
                focus_context=True,
            )
        )
        await pilot.pause()
        modal = app.screen.query_one("#console-settings-modal")
        assert modal.region.x >= 0
        assert modal.region.right <= 72
        assert modal.region.y >= 0
        assert modal.region.bottom <= 24
        budget = app.screen.query_one("#console-context-budget-mode", Select)
        await pilot.pause()
        assert app.focused is budget
        body = app.screen.query_one("#console-settings-body")
        hint = app.screen.query_one("#console-settings-fold-hint", Static)
        assert hint.display, (
            body.virtual_size,
            body.container_size,
            body.max_scroll_y,
        )
        actions = app.screen.query_one("#console-settings-actions")
        assert actions.region.bottom <= 24


@pytest.mark.asyncio
async def test_quick_popover_keeps_actions_visible_and_marks_the_narrow_fold() -> None:
    """Keep the context route discoverable before a new user starts scrolling."""
    app = _ContextHarness()
    async with app.run_test(size=(72, 24)) as pilot:
        await app.push_screen(
            ConsoleModelPopover(
                settings=_settings(),
                providers_models={"llama_cpp": ["model-a"]},
                context_state=_state(),
            )
        )
        await pilot.pause()
        await pilot.pause()

        hint = app.screen.query_one("#console-popover-fold-hint", Static)
        actions = app.screen.query_one("#console-popover-actions")
        context_button = app.screen.query_one(
            "#console-popover-full-settings",
            Button,
        )
        assert hint.display
        assert actions.region.bottom <= 24
        assert context_button.region.bottom <= 24
        focus_order: list[str] = []
        for _ in range(14):
            focused = app.focused
            focus_order.append(getattr(focused, "id", "") or "")
            if focus_order[-1] == "console-popover-full-settings":
                break
            await pilot.press("tab")
            await pilot.pause()
        assert focus_order.index("console-popover-temperature") < focus_order.index(
            "console-popover-streaming"
        )
        assert focus_order.index("console-popover-streaming") < focus_order.index(
            "console-popover-compaction-mode"
        )
        assert focus_order[-1] == "console-popover-full-settings"


@pytest.mark.asyncio
async def test_unverified_model_capacity_is_labeled_as_estimated() -> None:
    """Never present the 8,001-token fallback as model-verified capacity."""
    estimate = ConsoleSettingsContextEstimate(
        10,
        8001,
        "10 / 8,001 tokens (estimated; model unverified)",
        token_limit_verified=False,
        token_limit_source="provider fallback",
    )
    state = build_console_context_control_state(
        settings=_settings(),
        estimate=estimate,
    )
    app = _ContextHarness()
    async with app.run_test(size=(100, 34)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=estimate,
                context_state=state,
                can_save=True,
                focus_context=True,
            )
        )
        await pilot.pause()

        window = str(
            app.screen.query_one("#console-context-model-window", Static).renderable
        )
        status = str(
            app.screen.query_one("#console-context-capacity-status", Static).renderable
        )
        assert "Model window (est.)" in window
        assert "model capacity is unverified" in status
        assert "Providers & Models" in status


@pytest.mark.asyncio
async def test_quick_popover_mounts_with_no_model_selected() -> None:
    """A session with no model opens the popover on the blank model row.

    TASK-16502: on Textual 8.x ``Select.BLANK`` silently resolves to
    ``Widget.BLANK`` (``False``), which is not a legal Select value, so the
    popover crashed at mount with InvalidSelectValueError for any session
    whose settings carry no model.
    """
    app = _ContextHarness()
    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(
            ConsoleModelPopover(
                settings=ConsoleSessionSettings(
                    provider="llama_cpp",
                    model=None,
                    max_tokens=4_000,
                ),
                providers_models={"llama_cpp": ["model-a"]},
                context_state=_state(),
            ),
            callback=app.capture,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-popover-model", Select)
        assert model_select.value is Select.NULL
