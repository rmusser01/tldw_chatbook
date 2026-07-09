"""Console quick model popover (Alt+M)."""

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
from tldw_chatbook.Utils.input_validation import validate_text_input

CONSOLE_POPOVER_OPEN_FULL_SETTINGS = "open-full-settings"

# Mirrors ConsoleSettingsModal's temperature bounds (see
# Chat/console_session_settings.validate_console_session_settings, which
# rejects a temperature outside [0.0, 2.0] via a plain range comparison --
# NaN and +/-Inf always fail that comparison too). The popover has no error
# banner, so instead of blocking Apply like the full settings modal does, an
# invalid temperature here just keeps the prior value.
_CONSOLE_POPOVER_TEMPERATURE_MIN = 0.0
_CONSOLE_POPOVER_TEMPERATURE_MAX = 2.0


def _temperature_in_range(value: float) -> bool:
    """Return whether a parsed temperature is finite and within modal bounds.

    Args:
        value: Parsed temperature candidate.

    Returns:
        True if ``value`` is within ``[0.0, 2.0]``. NaN and infinite values
        always return False, since any comparison against them is False.
    """
    return _CONSOLE_POPOVER_TEMPERATURE_MIN <= value <= _CONSOLE_POPOVER_TEMPERATURE_MAX


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
        """Initialize the popover with the session's current settings.

        Args:
            settings: The Console session's current settings, used to seed
                the provider/model/temperature/streaming controls.
            providers_models: Mapping of provider key to its available model
                names, used to build the provider and model selects.
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        self._settings = settings
        self._providers_models = providers_models
        self._streaming = bool(settings.streaming)

    def compose(self) -> ComposeResult:
        """Build the provider, model, temperature, and streaming controls."""
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
        """Refresh the model options when the provider select changes.

        Args:
            event: The provider select's change event.
        """
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
        """Flip the local streaming toggle and relabel the button.

        Args:
            event: The streaming toggle button's press event.
        """
        event.stop()
        self._streaming = not self._streaming
        event.button.label = f"Streaming: {'on' if self._streaming else 'off'}"

    @on(Button.Pressed, "#console-popover-full-settings")
    def _full_settings(self, event: Button.Pressed) -> None:
        """Dismiss with the sentinel that tells the caller to open full settings.

        Args:
            event: The "Full settings…" button's press event.
        """
        event.stop()
        self.dismiss(CONSOLE_POPOVER_OPEN_FULL_SETTINGS)

    @on(Button.Pressed, "#console-popover-apply")
    def _apply(self, event: Button.Pressed) -> None:
        """Apply the popover's provider/model/temperature/streaming edits.

        An empty temperature input clears the value (``None``). A non-empty
        value that fails to parse as a float, or that parses to NaN/Inf/a
        value outside ``[0.0, 2.0]``, keeps the prior temperature instead of
        applying it -- mirroring ``ConsoleSettingsModal``'s rejection of
        out-of-range temperatures, minus its error banner (this popover has
        no error surface, so it silently falls back rather than blocking).

        Args:
            event: The "Apply" button's press event.
        """
        event.stop()
        provider_value = self.query_one("#console-popover-provider", Select).value
        model_value = self.query_one("#console-popover-model", Select).value
        temperature_text = self.query_one("#console-popover-temperature", Input).value.strip()
        if not temperature_text:
            temperature = None
        else:
            temperature = self._settings.temperature
            if validate_text_input(temperature_text, max_length=32):
                try:
                    candidate = float(temperature_text)
                except ValueError:
                    pass
                else:
                    if _temperature_in_range(candidate):
                        temperature = candidate
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
        """Dismiss the popover with no result (Escape)."""
        self.dismiss(None)
