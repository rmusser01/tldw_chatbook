"""Console quick model popover (Ctrl+M)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_model_options,
    build_console_provider_options,
)

CONSOLE_POPOVER_OPEN_FULL_SETTINGS = "open-full-settings"


class ConsoleModelPopover(ModalScreen["ConsoleSessionSettings | str | None"]):
    """Quick provider/model/temperature/streaming switcher for the session."""

    DEFAULT_CSS = """
    ConsoleModelPopover {
        align: center middle;
    }

    #console-model-popover {
        width: 60;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-popover-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }
    """

    BINDINGS = [("escape", "dismiss_popover", "Cancel")]

    def __init__(
        self,
        *,
        settings: ConsoleSessionSettings,
        providers_models: Mapping[str, Sequence[str]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings
        self._providers_models = providers_models
        self._streaming = bool(settings.streaming)

    def compose(self) -> ComposeResult:
        provider_options = [
            (option.label, option.value)
            for option in build_console_provider_options(self._providers_models)
        ]
        model_options = [
            (option.label, option.value)
            for option in build_console_model_options(
                self._settings.provider, self._providers_models, self._settings.model
            )
        ]
        with Vertical(id="console-model-popover"):
            yield Static("Model", classes="console-modal-header")
            yield Select(
                provider_options,
                value=self._settings.provider,
                id="console-popover-provider",
            )
            yield Select(
                model_options,
                value=self._settings.model if self._settings.model else Select.BLANK,
                id="console-popover-model",
                allow_blank=True,
            )
            yield Input(
                value="" if self._settings.temperature is None else str(self._settings.temperature),
                placeholder="Temperature",
                id="console-popover-temperature",
            )
            yield Button(
                f"Streaming: {'on' if self._streaming else 'off'}",
                id="console-popover-streaming",
                compact=True,
            )
            with Horizontal(id="console-popover-actions"):
                yield Button("Full settings…", id="console-popover-full-settings", compact=True)
                yield Button("Apply", id="console-popover-apply", variant="primary", compact=True)

    @on(Select.Changed, "#console-popover-provider")
    def _provider_changed(self, event: Select.Changed) -> None:
        event.stop()
        provider = str(event.value)
        options = [
            (option.label, option.value)
            for option in build_console_model_options(
                provider, self._providers_models, None
            )
        ]
        model_select = self.query_one("#console-popover-model", Select)
        model_select.set_options(options)

    @on(Button.Pressed, "#console-popover-streaming")
    def _toggle_streaming(self, event: Button.Pressed) -> None:
        event.stop()
        self._streaming = not self._streaming
        event.button.label = f"Streaming: {'on' if self._streaming else 'off'}"

    @on(Button.Pressed, "#console-popover-full-settings")
    def _full_settings(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(CONSOLE_POPOVER_OPEN_FULL_SETTINGS)

    @on(Button.Pressed, "#console-popover-apply")
    def _apply(self, event: Button.Pressed) -> None:
        event.stop()
        provider_value = self.query_one("#console-popover-provider", Select).value
        model_value = self.query_one("#console-popover-model", Select).value
        temperature_text = self.query_one("#console-popover-temperature", Input).value.strip()
        temperature = self._settings.temperature
        if temperature_text:
            try:
                temperature = float(temperature_text)
            except ValueError:
                pass
        self.dismiss(
            replace(
                self._settings,
                provider=str(provider_value),
                model=None if model_value in (None, Select.BLANK) else str(model_value),
                temperature=temperature,
                streaming=self._streaming,
            )
        )

    def action_dismiss_popover(self) -> None:
        self.dismiss(None)
