"""Searchable, grouped provider picker for Conversation settings."""

from __future__ import annotations

from collections.abc import Sequence

from rich.markup import escape as escape_markup
from textual import events, on
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Chat.console_provider_endpoints import URL_BASED_PROVIDER_KEYS
from tldw_chatbook.Chat.console_provider_support import (
    resolve_console_provider_identity,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsOption
from tldw_chatbook.Chat.provider_catalog import (
    PROVIDER_CUSTOM_GROUP_KEYS,
    provider_display_name,
)
from tldw_chatbook.Chat.provider_readiness import (
    PROVIDERS_REQUIRING_API_KEY_KEYS,
    provider_config_key,
)

_BLUR_RESTORE_DELAY_SECONDS = 0.05


class ConsoleProviderPickerInput(Input):
    """Input that lets the compound picker restore on Escape."""

    class EscapePressed(Message):
        """Posted before ``Input`` consumes Escape as an edit rollback."""

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.post_message(self.EscapePressed())
            return
        await super()._on_key(event)


class ConsoleProviderPicker(Widget):
    """Controlled provider picker with grouped, in-memory search results."""

    MAX_RESULTS = 30
    _GROUP_ORDER = ("Cloud", "Local", "Custom", "Other")

    BUNDLED_CSS = """
    ConsoleProviderPicker {
        height: auto;
        width: 1fr;
    }

    ConsoleProviderPicker #console-settings-provider-picker-input {
        width: 100%;
    }

    ConsoleProviderPicker #console-settings-provider-picker-status {
        height: auto;
        color: $text-muted;
    }

    ConsoleProviderPicker #console-settings-provider-picker-results {
        max-height: 12;
    }
    """

    class ProviderSelected(Message):
        """Posted when the user commits a provider from the option list."""

        def __init__(self, provider: str) -> None:
            super().__init__()
            self.provider = provider

    def __init__(
        self,
        provider_options: Sequence[ConsoleSettingsOption],
        current_provider: str | None,
        **kwargs: object,
    ) -> None:
        """Initialize the controlled picker from the modal's known options."""
        super().__init__(**kwargs)
        options_by_value: dict[str, ConsoleSettingsOption] = {}
        for option in provider_options:
            value = str(option.value or "").strip()
            if value and value not in options_by_value:
                options_by_value[value] = option
        self._options = tuple(options_by_value.values())
        self._known_provider_ids = frozenset(options_by_value)
        self._display_names = {
            option.value: self._display_name_for_option(option)
            for option in self._options
        }
        self._provider_groups = {
            option.value: self._classify_provider(option.value)
            for option in self._options
        }
        current = str(current_provider or "").strip()
        self._value = current if current in options_by_value else None
        self._visible_provider_ids: tuple[str, ...] = ()
        self._row_provider_ids: list[str | None] = []
        self._suppress_input_events = False
        self._preserve_committed_on_next_input_focus = False

    @property
    def value(self) -> str | None:
        """Return the currently committed provider key."""
        return self._value

    def compose(self) -> ComposeResult:
        """Compose the searchable input, status copy, and grouped results."""
        yield ConsoleProviderPickerInput(
            value=self._display_name(self._value),
            placeholder="Choose or search providers",
            id="console-settings-provider-picker-input",
            name="provider-search",
            tooltip="Choose or search the provider for this conversation.",
        )
        yield Static(
            self._selected_status(),
            id="console-settings-provider-picker-status",
            markup=False,
        )
        results = OptionList(
            id="console-settings-provider-picker-results",
            name="provider-options",
        )
        results.tooltip = "Matching providers; use arrow keys and Enter to select."
        yield results

    def on_mount(self) -> None:
        """Start with the current selection visible and results collapsed."""
        self._hide_results()

    def focus_input(self) -> None:
        """Focus the searchable provider input."""
        self.query_one("#console-settings-provider-picker-input", Input).focus()

    def visible_provider_ids(self) -> tuple[str, ...]:
        """Return provider keys represented by the current result rows."""
        return self._visible_provider_ids

    def set_provider(self, provider: str | None) -> None:
        """Synchronize a provider selected through a compatibility adapter."""
        normalized = str(provider or "").strip()
        self._value = normalized if normalized in self._known_provider_ids else None
        self._set_input_value(self._display_name(self._value))
        self._hide_results()
        self._set_status(self._selected_status())

    @staticmethod
    def _classify_provider(provider: str) -> str:
        key = provider_config_key(provider)
        identity = resolve_console_provider_identity(provider)
        if key in PROVIDER_CUSTOM_GROUP_KEYS or identity.execution_key.startswith(
            "custom-openai-api"
        ):
            return "Custom"
        if key in PROVIDERS_REQUIRING_API_KEY_KEYS:
            return "Cloud"
        execution_key = identity.execution_key
        if (
            key in URL_BASED_PROVIDER_KEYS
            or identity.uses_direct_llama_path
            or execution_key.startswith("local")
            or execution_key in {"mlx_lm"}
        ):
            return "Local"
        return "Other"

    @staticmethod
    def _display_name_for_option(option: ConsoleSettingsOption) -> str:
        label = provider_display_name(option.value)
        if option.label.endswith(" (WIP)"):
            return f"{label} (WIP)"
        return label

    def _display_name(self, provider: str | None) -> str:
        if not provider:
            return ""
        return self._display_names.get(provider, provider_display_name(provider))

    def _provider_group(self, provider: str) -> str:
        return self._provider_groups.get(provider, "Other")

    def _selected_status(self) -> str:
        if self._value is None:
            return "Choose a provider."
        return f"Selected: {self._display_name(self._value)}"

    def _matching_options(self, query: str) -> list[ConsoleSettingsOption]:
        normalized_query = query.strip().casefold()
        matches = [
            option
            for option in self._options
            if not normalized_query
            or normalized_query in option.value.casefold()
            or normalized_query in self._display_name(option.value).casefold()
        ]
        return sorted(
            matches,
            key=lambda option: (
                self._GROUP_ORDER.index(self._provider_group(option.value)),
                self._display_name(option.value).casefold(),
                option.value,
            ),
        )[: self.MAX_RESULTS]

    def _render_matches(self, query: str) -> None:
        matches = self._matching_options(query)
        self._visible_provider_ids = tuple(option.value for option in matches)
        results = self.query_one(
            "#console-settings-provider-picker-results", OptionList
        )
        results.clear_options()
        self._row_provider_ids = []
        grouped = {
            group: [
                option
                for option in matches
                if self._provider_group(option.value) == group
            ]
            for group in self._GROUP_ORDER
        }
        for group in self._GROUP_ORDER:
            group_options = grouped[group]
            if not group_options:
                continue
            results.add_option(Option(group, disabled=True))
            self._row_provider_ids.append(None)
            for option in group_options:
                results.add_option(
                    Option(escape_markup(self._display_name(option.value)))
                )
                self._row_provider_ids.append(option.value)
        results.display = bool(matches)
        selection = self._selected_status().rstrip(".")
        if matches:
            self._set_status(
                f"{selection} · {len(matches)} "
                f"provider{'s' if len(matches) != 1 else ''} available."
            )
        else:
            self._set_status(
                f"{selection} · No matching providers. Clear the filter."
            )

    def _hide_results(self) -> None:
        if not self.is_mounted:
            return
        results = self.query_one(
            "#console-settings-provider-picker-results", OptionList
        )
        results.clear_options()
        results.display = False
        self._visible_provider_ids = ()
        self._row_provider_ids = []

    def _set_input_value(self, value: str) -> None:
        if not self.is_mounted:
            return
        input_widget = self.query_one("#console-settings-provider-picker-input", Input)
        self._suppress_input_events = True
        try:
            with input_widget.prevent(Input.Changed):
                input_widget.value = value
        finally:
            self._suppress_input_events = False

    def _set_status(self, copy: str) -> None:
        if self.is_mounted:
            self.query_one("#console-settings-provider-picker-status", Static).update(
                copy
            )

    def _commit_provider(self, provider: str) -> None:
        if provider not in self._known_provider_ids:
            return
        results_had_focus = self.query_one(
            "#console-settings-provider-picker-results", OptionList
        ).has_focus
        if results_had_focus:
            self._preserve_committed_on_next_input_focus = True
        self._value = provider
        self._set_input_value(self._display_name(provider))
        self._hide_results()
        self._set_status(self._selected_status())
        if results_had_focus:
            self.focus_input()
        self.post_message(self.ProviderSelected(provider))

    def _restore_committed_value(self, event: Message) -> None:
        input_widget = self.query_one("#console-settings-provider-picker-input", Input)
        if (
            input_widget.value != self._display_name(self._value)
            or self._row_provider_ids
        ):
            self._set_input_value(self._display_name(self._value))
            self._hide_results()
            self._set_status(self._selected_status())
            event.stop()

    @on(Input.Changed, "#console-settings-provider-picker-input")
    def _input_changed(self, event: Input.Changed) -> None:
        if not self._suppress_input_events:
            self._render_matches(event.value)

    @on(Input.Submitted, "#console-settings-provider-picker-input")
    def _input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip().casefold()
        exact = next(
            (
                provider
                for provider in self._visible_provider_ids
                if provider.casefold() == query
                or self._display_name(provider).casefold() == query
            ),
            None,
        )
        if exact is not None:
            self._commit_provider(exact)
        elif len(self._visible_provider_ids) == 1:
            self._commit_provider(self._visible_provider_ids[0])

    @on(OptionList.OptionSelected, "#console-settings-provider-picker-results")
    def _option_selected(self, event: OptionList.OptionSelected) -> None:
        index = event.option_index
        if index is None or not (0 <= index < len(self._row_provider_ids)):
            return
        provider = self._row_provider_ids[index]
        if provider is not None:
            self._commit_provider(provider)

    @on(ConsoleProviderPickerInput.EscapePressed)
    def _escape_pressed(self, event: ConsoleProviderPickerInput.EscapePressed) -> None:
        self._restore_committed_value(event)

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if (
            getattr(event.control, "id", None)
            != "console-settings-provider-picker-input"
        ):
            return
        if self._preserve_committed_on_next_input_focus:
            self._preserve_committed_on_next_input_focus = False
            return
        self._set_input_value("")
        self._render_matches("")

    def on_descendant_blur(self, _event: events.DescendantBlur) -> None:
        """Restore the committed label after focus leaves the compound picker."""
        self.set_timer(
            _BLUR_RESTORE_DELAY_SECONDS,
            self._restore_committed_display_after_blur,
        )

    def _restore_committed_display_after_blur(self) -> None:
        """Keep the visible field from implying an uncommitted filter was selected."""
        if not self.is_mounted:
            return
        focused = self.app.focused
        if focused is not None and self in focused.ancestors_with_self:
            return
        self._set_input_value(self._display_name(self._value))
        self._hide_results()
        self._set_status(self._selected_status())

    @on(events.Key)
    def _handle_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self._preserve_committed_on_next_input_focus = True
            self._set_input_value(self._display_name(self._value))
            self._hide_results()
            self._set_status(self._selected_status())
            self.focus_input()
            return
        if event.key not in {"down", "up"} or not self._row_provider_ids:
            return
        results = self.query_one(
            "#console-settings-provider-picker-results", OptionList
        )
        selectable = [
            index
            for index, provider in enumerate(self._row_provider_ids)
            if provider is not None
        ]
        if not selectable:
            return
        results.focus()
        results.highlighted = selectable[0] if event.key == "down" else selectable[-1]
        event.stop()
