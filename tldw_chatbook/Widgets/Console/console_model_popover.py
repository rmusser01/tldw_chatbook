"""Console quick model popover (Alt+M)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    build_console_model_options,
    build_console_provider_options,
)
from tldw_chatbook.Chat.console_context_policy import (
    ContextCompactionMode,
    ContextCompactionRepresentation,
)
from tldw_chatbook.Chat.provider_catalog import provider_display_name
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from .console_context_controls import (
    ConsoleContextControlState,
    build_console_context_control_state,
    format_context_tokens,
)
from tldw_chatbook.Widgets.model_search_picker import ModelSearchPicker

CONSOLE_POPOVER_OPEN_FULL_SETTINGS = "open-full-settings"


@dataclass(frozen=True, slots=True)
class ConsoleModelPopoverResult:
    """Quick settings and policy edits owned by the current conversation."""

    settings: ConsoleSessionSettings
    compaction_mode: ContextCompactionMode


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


class ConsoleModelPopover(
    SafeModalDismissMixin,
    ModalScreen["ConsoleModelPopoverResult | ConsoleSessionSettings | str | None"],
):
    """Quick provider/model/temperature/streaming switcher for the session."""

    DEFAULT_CSS = """
    ConsoleModelPopover {
        align: center middle;
    }

    #console-model-popover {
        width: 60;
        height: 90%;
        min-height: 18;
        max-height: 32;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-model-popover-body {
        height: 1fr;
        min-height: 0;
        overflow-y: auto;
        overflow-x: hidden;
    }

    .console-popover-field-label {
        color: $text-muted;
        margin: 1 0 0 0;
    }

    .console-popover-context-row {
        height: 1;
        color: $text-muted;
    }

    #console-popover-compaction-help {
        height: auto;
        color: $text-muted;
    }

    #console-popover-footer {
        height: auto;
        background: black;
    }

    #console-popover-fold-hint {
        height: 1;
        color: $text-muted;
    }

    #console-popover-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }
    """

    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#console-model-popover"

    def __init__(
        self,
        *,
        settings: ConsoleSessionSettings,
        providers_models: Mapping[str, Sequence[str]],
        context_state: ConsoleContextControlState | None = None,
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
        self._context_state = context_state or build_console_context_control_state(
            settings=settings,
            estimate=ConsoleSettingsContextEstimate(
                used_tokens=None,
                token_limit=None,
                label="Context: unavailable",
            ),
        )
        self._streaming = bool(settings.streaming)
        # TASK-364: the provider Select fires a mount-time Select.Changed for its
        # initial value; without this tracker `_provider_changed` would rebuild
        # the model options with current_model=None and wipe the prefilled model.
        # Only a REAL provider change (value differs from what the model options
        # currently reflect) should reset the model.
        self._model_options_provider = settings.provider

    def _provider_select_options(self) -> list[tuple[str, str]]:
        """Provider options labeled with the shared catalog display names.

        TASK-364: mirror ``ConsoleSettingsModal._provider_select_options`` so the
        quick popover shows the same names as the full modal (``llama.cpp``, not
        the raw ``llama_cpp`` key).
        """
        return [
            (provider_display_name(option.value), option.value)
            for option in build_console_provider_options(self._providers_models)
        ]

    def compose(self) -> ComposeResult:
        """Build the provider, model, temperature, and streaming controls."""
        provider_options = self._provider_select_options()
        model_options = [
            (option.label, option.value)
            for option in build_console_model_options(
                self._settings.provider, self._providers_models, self._settings.model
            )
        ]
        with Vertical(id="console-model-popover"):
            with VerticalScroll(id="console-model-popover-body"):
                yield Static("Model", classes="console-modal-header")
                yield Select(
                    provider_options,
                    value=self._settings.provider,
                    id="console-popover-provider",
                )
                model_select = Select(
                    model_options,
                    # Select.NULL, not Select.BLANK: on this Textual version
                    # BLANK doesn't exist on Select and silently resolves to
                    # Widget.BLANK (False), an illegal value that crashes the
                    # Select at mount (TASK-16502).
                    value=(
                        self._settings.model
                        if self._settings.model
                        else Select.NULL
                    ),
                    id="console-popover-model",
                    allow_blank=True,
                )
                model_select.display = False
                yield model_select
                yield ModelSearchPicker(
                    id="console-popover-model-search",
                    provider_select_id="#console-popover-provider",
                    current_model=self._settings.model,
                    providers_models=self._providers_models,
                )
                yield Static("Temperature", classes="console-popover-field-label")
                yield Input(
                    value=(
                        ""
                        if self._settings.temperature is None
                        else str(self._settings.temperature)
                    ),
                    placeholder="Temperature",
                    id="console-popover-temperature",
                )
                yield Button(
                    f"Streaming: {'on' if self._streaming else 'off'}",
                    id="console-popover-streaming",
                    compact=True,
                )
                yield Static(
                    "Response max  "
                    f"{format_context_tokens(self._settings.max_tokens)} tokens for the next reply",
                    id="console-popover-response-max",
                    classes="console-popover-context-row",
                    markup=False,
                )
                yield Static(
                    f"Request       {self._context_state.request_row}",
                    id="console-popover-request-usage",
                    classes="console-popover-context-row",
                    markup=False,
                )
                yield Static(
                    f"Conversation  {self._context_state.conversation_row}",
                    id="console-popover-conversation-usage",
                    classes="console-popover-context-row",
                    markup=False,
                )
            with Vertical(id="console-popover-footer"):
                yield Static(
                    "Compaction    at "
                    f"{format_context_tokens(self._context_state.compaction_trigger_tokens)} tokens",
                    id="console-popover-compaction-threshold",
                    classes="console-popover-context-row",
                    markup=False,
                )
                yield Static(
                    self._compaction_help_text(),
                    id="console-popover-compaction-help",
                    markup=False,
                )
                yield Select(
                    [
                        ("Ask", ContextCompactionMode.ASK.value),
                        ("Automatic", ContextCompactionMode.AUTOMATIC.value),
                        ("Off", ContextCompactionMode.OFF.value),
                    ],
                    value=self._context_state.resolved_policy.policy.compaction_mode.value,
                    id="console-popover-compaction-mode",
                    disabled=self._context_state.busy,
                )
                fold_hint = Static(
                    "▼ more — scroll for conversation settings",
                    id="console-popover-fold-hint",
                    markup=False,
                )
                fold_hint.display = False
                yield fold_hint
                with Horizontal(id="console-popover-actions"):
                    yield Button(
                        "Context & memory…",
                        id="console-popover-full-settings",
                        compact=True,
                    )
                    yield Button(
                        "Apply",
                        id="console-popover-apply",
                        variant="primary",
                        compact=True,
                    )

    def _compaction_help_text(self) -> str:
        representation = (
            self._context_state.resolved_policy.policy.compaction_representation
        )
        if representation is ContextCompactionRepresentation.VISUAL_TRANSCRIPT:
            return "Renders older turns on-device; no summary model call."
        if representation is ContextCompactionRepresentation.HYBRID:
            return "Adds local pages to a text summary; Automatic adds one model call."
        return "Summarizes older turns. Automatic may add one extra model call."

    def on_mount(self) -> None:
        """Settle the narrow-height fold affordance after first layout."""
        self.call_after_refresh(self._sync_fold_hint)

    def on_resize(self, _event: events.Resize) -> None:
        """Recompute the fold affordance when the terminal size changes."""
        self.call_after_refresh(self._sync_fold_hint)

    def _sync_fold_hint(self) -> None:
        """Expose hidden quick settings while keeping the actions pinned."""
        body = self.query_one("#console-model-popover-body", VerticalScroll)
        hint = self.query_one("#console-popover-fold-hint", Static)
        hint.display = body.virtual_size.height > body.container_size.height

    @on(Select.Changed, "#console-popover-provider")
    def _provider_changed(self, event: Select.Changed) -> None:
        """Refresh the model options when the provider select changes.

        Args:
            event: The provider select's change event.
        """
        event.stop()
        provider = str(event.value)
        # TASK-364: ignore the mount-time echo and redundant same-provider events
        # so the prefilled model survives; only a genuine provider change resets
        # the model options (a stale model from another provider must not linger).
        if provider == self._model_options_provider:
            return
        self._model_options_provider = provider
        options = [
            (option.label, option.value)
            for option in build_console_model_options(
                provider, self._providers_models, None
            )
        ]
        model_select = self.query_one("#console-popover-model", Select)
        model_select.set_options(options)
        self.query_one(
            "#console-popover-model-search", ModelSearchPicker
        ).refresh_provider(provider, current_model=None)

    @on(ModelSearchPicker.ModelSelected)
    def _model_search_selected(self, event: ModelSearchPicker.ModelSelected) -> None:
        """Insert a picked model as a transient option and select it (ADR-020)."""
        event.stop()
        model_id = event.model_id.strip()
        if not model_id:
            return
        provider = str(self.query_one("#console-popover-provider", Select).value)
        model_select = self.query_one("#console-popover-model", Select)
        options = [
            (option.label, option.value)
            for option in build_console_model_options(
                provider, self._providers_models, model_id
            )
        ]
        model_select.set_options(options)
        model_select.value = model_id

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
        model_value = self.query_one(
            "#console-popover-model-search", ModelSearchPicker
        ).value
        temperature_text = self.query_one(
            "#console-popover-temperature", Input
        ).value.strip()
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
        settings = replace(
            self._settings,
            provider=str(provider_value),
            model=None if model_value in (None, Select.NULL) else str(model_value),
            temperature=temperature,
            streaming=self._streaming,
        )
        mode = ContextCompactionMode(
            str(self.query_one("#console-popover-compaction-mode", Select).value)
        )
        if mode is self._context_state.resolved_policy.policy.compaction_mode:
            self.dismiss(settings)
            return
        self.dismiss(ConsoleModelPopoverResult(settings=settings, compaction_mode=mode))

    async def action_dismiss_popover(self) -> None:
        """Dismiss the popover with no result (Escape)."""
        await self.request_safe_cancel(source="visible")
