"""Behavior tests for the Conversation settings provider picker."""

from __future__ import annotations

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, OptionList, Static

from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsOption
from tldw_chatbook.Widgets.Console.console_provider_picker import (
    ConsoleProviderPicker,
)


class ProviderPickerApp(App[None]):
    """Minimal mounted host for keyboard and rendering behavior."""

    def __init__(
        self,
        options: tuple[ConsoleSettingsOption, ...],
        *,
        current_provider: str | None = "openai",
    ) -> None:
        super().__init__()
        self._options = options
        self._current_provider = current_provider
        self.selected: list[str] = []
        self.escape_bubbled = 0

    def compose(self) -> ComposeResult:
        yield ConsoleProviderPicker(
            self._options,
            current_provider=self._current_provider,
            id="console-settings-provider-picker",
        )
        yield Button("After", id="after")

    @on(ConsoleProviderPicker.ProviderSelected)
    def _record_provider(self, event: ConsoleProviderPicker.ProviderSelected) -> None:
        self.selected.append(event.provider)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.escape_bubbled += 1


OPTIONS = (
    ConsoleSettingsOption("openai", "openai"),
    ConsoleSettingsOption("llama_cpp", "llama_cpp"),
    ConsoleSettingsOption("local_llamacpp", "local_llamacpp"),
    ConsoleSettingsOption("custom_2", "custom_2"),
    ConsoleSettingsOption("mystery[bold]", "mystery[bold]"),
)


def _prompts(results: OptionList) -> list[str]:
    return [str(option.prompt) for option in results.options]


@pytest.mark.asyncio
async def test_picker_groups_providers_in_connection_order_with_disabled_headings() -> (
    None
):
    """A missing group/order implementation makes cloud and local choices hard to scan."""
    app = ProviderPickerApp(OPTIONS)

    async with app.run_test() as pilot:
        picker = app.query_one(ConsoleProviderPicker)
        picker.focus_input()
        await pilot.pause()

        results = app.query_one("#console-settings-provider-picker-results", OptionList)
        assert _prompts(results) == [
            "Cloud",
            "OpenAI",
            "Local",
            "llama.cpp",
            "Custom",
            "Custom OpenAI-compatible #2",
            "llama.cpp (legacy alias)",
            "Other",
            "mystery\\[bold]",
        ]
        assert [option.disabled for option in results.options] == [
            True,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
        ]


@pytest.mark.asyncio
async def test_picker_filters_case_insensitively_by_display_name_and_config_key() -> (
    None
):
    """Filtering only raw keys or only labels breaks one of the two user vocabularies."""
    app = ProviderPickerApp(OPTIONS)

    async with app.run_test() as pilot:
        search = app.query_one("#console-settings-provider-picker-input", Input)
        search.focus()
        await pilot.pause()
        search.value = "LlAmA"
        await pilot.pause()
        assert app.query_one(ConsoleProviderPicker).visible_provider_ids() == (
            "llama_cpp",
            "local_llamacpp",
        )

        search.value = "custom_2"
        await pilot.pause()
        assert app.query_one(ConsoleProviderPicker).visible_provider_ids() == (
            "custom_2",
        )


@pytest.mark.asyncio
async def test_down_then_enter_selects_first_match_without_accepting_typed_ids() -> (
    None
):
    """Selection must come from a known option, never arbitrary input text."""
    app = ProviderPickerApp(OPTIONS)

    async with app.run_test() as pilot:
        search = app.query_one("#console-settings-provider-picker-input", Input)
        search.focus()
        await pilot.pause()
        search.value = "not-a-provider"
        await pilot.pause()
        await pilot.press("enter")
        assert app.selected == []

        search.value = "llama"
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.selected == ["llama_cpp"]
        assert app.query_one(ConsoleProviderPicker).value == "llama_cpp"
        assert search.value == "llama.cpp"
        assert app.focused is search
        assert not app.query_one(
            "#console-settings-provider-picker-results", OptionList
        ).display


@pytest.mark.asyncio
async def test_up_then_enter_selects_last_match_and_retains_committed_display() -> None:
    """Up from search reaches the final provider without leaving hidden focus."""
    app = ProviderPickerApp(OPTIONS)

    async with app.run_test() as pilot:
        picker = app.query_one(ConsoleProviderPicker)
        search = app.query_one("#console-settings-provider-picker-input", Input)
        results = app.query_one("#console-settings-provider-picker-results", OptionList)
        picker.focus_input()
        await pilot.pause()
        search.value = "llama"
        await pilot.pause()

        await pilot.press("up", "enter")
        await pilot.pause()

        assert app.selected == ["local_llamacpp"]
        assert picker.value == "local_llamacpp"
        assert search.value == "llama.cpp (legacy alias)"
        assert not results.display
        assert app.focused is search


@pytest.mark.asyncio
async def test_escape_from_results_restores_input_without_bubbling() -> None:
    """Escape from result focus must cancel search rather than close its modal."""
    app = ProviderPickerApp(OPTIONS, current_provider="openai")

    async with app.run_test() as pilot:
        picker = app.query_one(ConsoleProviderPicker)
        search = app.query_one("#console-settings-provider-picker-input", Input)
        results = app.query_one("#console-settings-provider-picker-results", OptionList)
        picker.focus_input()
        await pilot.pause()
        search.value = "llama"
        await pilot.pause()
        await pilot.press("down")
        assert app.focused is results

        await pilot.press("escape")
        await pilot.pause()

        assert search.value == "OpenAI"
        assert app.focused is search
        assert not results.display
        assert picker.value == "openai"
        assert app.escape_bubbled == 0


@pytest.mark.asyncio
async def test_escape_restores_current_value_and_no_match_copy_is_explicit() -> None:
    """A cancelled search must not erase the committed provider."""
    app = ProviderPickerApp(OPTIONS, current_provider="openai")

    async with app.run_test() as pilot:
        search = app.query_one("#console-settings-provider-picker-input", Input)
        search.focus()
        await pilot.pause()
        search.value = "no such provider"
        await pilot.pause()
        status = app.query_one("#console-settings-provider-picker-status", Static)
        assert str(status.renderable) == "No matching providers. Clear the filter."
        assert app.query_one(ConsoleProviderPicker).visible_provider_ids() == ()

        await pilot.press("escape")
        await pilot.pause()
        assert search.value == "OpenAI"
        assert app.query_one(ConsoleProviderPicker).value == "openai"
        assert app.selected == []


@pytest.mark.asyncio
async def test_leaving_picker_restores_committed_provider_copy() -> None:
    """Tabbing away must not leave an uncommitted filter looking selected."""
    app = ProviderPickerApp(OPTIONS, current_provider="openai")

    async with app.run_test() as pilot:
        search = app.query_one("#console-settings-provider-picker-input", Input)
        search.focus()
        await pilot.pause()
        search.value = "llama"
        await pilot.pause()
        app.query_one("#after", Button).focus()
        await pilot.pause(0.1)

        assert search.value == "OpenAI"
        assert app.query_one(ConsoleProviderPicker).value == "openai"
        assert not app.query_one(
            "#console-settings-provider-picker-results", OptionList
        ).display


@pytest.mark.asyncio
async def test_picker_caps_visible_providers_at_thirty() -> None:
    """A large configured provider map must not turn the modal into an unbounded list."""
    options = tuple(
        ConsoleSettingsOption(f"unknown-{index}", f"unknown-{index}")
        for index in range(35)
    )
    app = ProviderPickerApp(options, current_provider=None)

    async with app.run_test() as pilot:
        app.query_one(ConsoleProviderPicker).focus_input()
        await pilot.pause()
        assert len(app.query_one(ConsoleProviderPicker).visible_provider_ids()) == 30
