"""Console session settings modal."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping

from textual import events
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.Chat.console_provider_endpoints import (
    first_configured_endpoint,
    normalize_generic_endpoint_for_compare,
)
from tldw_chatbook.Chat.local_server_discovery import (
    normalize_probe_base_url,
    LocalModelProbeResult,
    endpoint_display,
    probe_models_endpoint,
)
from tldw_chatbook.Chat.provider_catalog import provider_display_name
from tldw_chatbook.config import save_settings_to_cli_config
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    DEFAULT_LLAMACPP_BASE_URL,
    URL_BASED_PROVIDER_KEYS,
    build_console_model_options,
    build_console_provider_options,
    build_console_settings_readiness,
    normalize_console_model_value,
    normalize_llamacpp_base_url,
    validate_console_session_settings,
)
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Utils.input_validation import validate_url
from rich.markup import escape as escape_markup


MODEL_INPUT_PLACEHOLDER = "Enter model id"
MODAL_BODY_MIN_HEIGHT = 0
MODAL_CONTROL_HEIGHT = 3
MODAL_LABEL_WIDTH = 16
MODEL_CUSTOM_BUTTON_WIDTH = 18
MODEL_DISCOVER_BUTTON_ID = "console-settings-model-discover"
MODEL_DISCOVER_STATUS_ID = "console-settings-model-discover-status"
MODEL_DISCOVER_BUTTON_LABEL = "Discover models"
MODEL_DISCOVER_BUTTON_WIDTH = 19
MODEL_DISCOVER_MISSING_URL_COPY = "Enter a base URL to discover models."
MODEL_DISCOVER_INVALID_URL_COPY = "Enter a valid http(s) endpoint URL to discover models."
ModelProber = Callable[[str, str], Awaitable[LocalModelProbeResult]]
STREAMING_TOGGLE_WIDTH = 12
PROVIDER_CHOICE_INPUT_MAX_LENGTH = 64
# (label, input id, accepted-values placeholder) - placeholders mirror the
# Settings screen's enumerated hints for these provider-specific fields.
PROVIDER_CHOICE_INPUTS = (
    (
        "Reasoning effort",
        "console-settings-reasoning-effort",
        "none, minimal, low, medium, high, xhigh",
    ),
    (
        "Reasoning summary",
        "console-settings-reasoning-summary",
        "auto, concise, detailed, none",
    ),
    ("Verbosity", "console-settings-verbosity", "low, medium, high"),
    (
        "Thinking effort",
        "console-settings-thinking-effort",
        "off, low, medium, high, xhigh, max",
    ),
)
STREAMING_ON_LABEL = "On"
STREAMING_OFF_LABEL = "Off"
CONSOLE_SETTINGS_SCOPE_COPY = (
    "Save applies to this session only. "
    "Save as default also writes provider + streaming defaults to config."
)
CONSOLE_SETTINGS_SAVE_DEFAULT_FAILED_COPY = (
    "Could not write defaults to the config file; session values still apply."
)
# Draft fields persisted under [api_settings.<provider>] by Save as default.
PROVIDER_DEFAULT_PERSIST_FIELDS = (
    "temperature",
    "top_p",
    "min_p",
    "top_k",
    "max_tokens",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "reasoning_effort",
    "reasoning_summary",
    "verbosity",
    "thinking_effort",
    "thinking_budget_tokens",
)
_ENDPOINT_PERSIST_KEYS = ("api_base_url", "api_base", "base_url", "api_url")


def _settings_screen_region(widget: Any) -> Any:
    """Return a mounted settings widget region in screen coordinates.

    Args:
        widget: Textual widget or test double with a mounted region.

    Returns:
        The widget's absolute screen region when the installed Textual version
        exposes one; otherwise the mounted widget region used by this project.
    """
    return getattr(widget, "screen_region", None) or widget.region


async def _default_model_prober(base_url: str, provider_key: str) -> LocalModelProbeResult:
    """Probe a models endpoint with the shared discovery helper.

    Args:
        base_url: Endpoint root taken from the current base-URL draft.
        provider_key: Normalized provider config key (enables the Ollama
            ``/api/tags`` fallback).

    Returns:
        The probe result, including honest failure copy on error.
    """
    return await probe_models_endpoint(base_url, provider_key=provider_key)


class ConsoleSettingsInput(Input):
    """Input field with browser-safe focus handoff behavior."""

    BINDINGS = [
        (
            Binding("home", "home", binding.description, show=binding.show)
            if binding.key == "home,ctrl+a"
            else binding
        )
        for binding in Input.BINDINGS
    ] + [
        Binding("ctrl+a,super+a", "select_all", "Select all", show=False),
    ]

    def on_click(self, event: events.Click | None = None) -> None:
        """Avoid trapping later Select clicks after browser text editing.

        Args:
            event: Optional click event to forward when Textual Web redirects a
                select click through the focused input.
        """
        self.select_all()
        self.release_mouse()
        if event is None:
            return
        handler = getattr(self.screen, "_open_select_from_redirected_settings_click", None)
        if callable(handler):
            handler(event)

    def on_blur(self) -> None:
        """Avoid trapping later Select clicks after browser text editing."""
        self.release_mouse()


class ConsoleSettingsModal(ModalScreen[ConsoleSessionSettings | None]):
    """Edit a draft of the current Console session settings."""

    DEFAULT_CSS = f"""
    ConsoleSettingsModal #console-settings-body {{
        height: 1fr;
        min-height: {MODAL_BODY_MIN_HEIGHT};
        overflow-y: auto;
        overflow-x: hidden;
    }}

    ConsoleSettingsModal .console-settings-modal-section {{
        height: auto;
    }}

    ConsoleSettingsModal .console-settings-modal-row {{
        height: auto;
        min-height: {MODAL_CONTROL_HEIGHT};
    }}

    ConsoleSettingsModal .console-settings-modal-label {{
        height: {MODAL_CONTROL_HEIGHT};
        min-height: {MODAL_CONTROL_HEIGHT};
    }}

    ConsoleSettingsModal Input,
    ConsoleSettingsModal Select,
    ConsoleSettingsModal Button {{
        height: {MODAL_CONTROL_HEIGHT};
        min-height: {MODAL_CONTROL_HEIGHT};
    }}
    """

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(
        self,
        *,
        settings: ConsoleSessionSettings,
        app_config: Mapping[str, object],
        providers_models: Mapping[str, list[str]],
        context_estimate: ConsoleSettingsContextEstimate,
        can_save: bool,
        focus_model: bool = False,
        model_prober: ModelProber | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._app_config = app_config
        self._providers_models = providers_models
        self._context_estimate = context_estimate
        self._can_save = can_save
        self._focus_model = focus_model
        self._model_prober: ModelProber = model_prober or _default_model_prober
        self._discovered_model_ids: dict[str, tuple[str, ...]] = {}
        self._streaming_draft = bool(settings.streaming)
        self._active_provider = settings.provider
        self._provider_model_drafts: dict[str, str | None] = {}
        self._set_provider_model_draft(settings.provider, settings.model)
        self._provider_base_url_drafts: dict[str, str] = {}
        initial_base_url = self._initial_base_url_for_provider(
            settings.provider,
            settings.base_url,
        )
        if initial_base_url:
            self._provider_base_url_drafts[settings.provider] = initial_base_url

    def compose(self) -> ComposeResult:
        provider_options = self._provider_select_options()
        selected_model = self._model_for_provider(self._settings.provider)
        base_url = self._base_url_for_provider(self._settings.provider)
        uses_base_url = self._provider_uses_base_url(self._settings.provider)
        model_options = self._model_select_options(self._settings.provider, selected_model)
        has_model_options = bool(model_options)
        use_model_select = self._should_use_model_select(
            self._settings.provider,
            selected_model,
            model_options,
        )
        readiness = build_console_settings_readiness(self._settings, app_config=self._app_config)

        with Vertical(id="console-settings-modal"):
            yield Static("Console Settings", classes="console-modal-header")
            yield Static(
                self._readiness_detail(readiness.detail),
                id="console-settings-readiness",
                classes="console-settings-modal-row",
                markup=False,
            )
            yield Static(
                CONSOLE_SETTINGS_SCOPE_COPY,
                id="console-settings-scope",
                classes="console-settings-modal-row",
                markup=False,
            )
            yield Static(
                "",
                id="console-settings-error",
                classes="console-settings-error console-settings-error-summary",
                markup=False,
            )

            body = ScrollableContainer(
                id="console-settings-body",
                classes="console-settings-body",
            )
            # The global `*:focus` outline peeks through the 1-row section
            # margins as stray "|" fragments when this scroll container takes
            # focus (it was the first focusable widget on open). Keyboard
            # scrolling still works through its bindings while a child input
            # is focused, so the container stays out of the focus chain.
            body.can_focus = False
            with body:
                with Vertical(
                    id="console-settings-provider-model-section",
                    classes=self._provider_model_section_classes(),
                ):
                    yield Static("Provider and model", classes="destination-section")
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Provider")
                        yield Select(
                            provider_options,
                            value=self._settings.provider,
                            allow_blank=False,
                            id="console-settings-provider",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Model")
                        model_select = Select(
                            model_options or [("No configured models", "")],
                            value=selected_model or "",
                            allow_blank=False,
                            id="console-settings-model-select",
                            disabled=not use_model_select,
                            classes="console-settings-control",
                        )
                        model_select.styles.width = "1fr"
                        model_select.styles.min_width = 0
                        model_select.display = use_model_select
                        yield model_select
                        model_input = ConsoleSettingsInput(
                            value=selected_model or "",
                            placeholder=MODEL_INPUT_PLACEHOLDER,
                            id="console-settings-model-input",
                            disabled=has_model_options,
                            classes="console-settings-control",
                        )
                        model_input.styles.width = "1fr"
                        model_input.styles.min_width = 0
                        model_input.display = not use_model_select
                        yield model_input
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("")
                        model_custom = Button(
                            "Custom model",
                            id="console-settings-model-custom",
                            disabled=not has_model_options,
                        )
                        model_custom.styles.width = MODEL_CUSTOM_BUTTON_WIDTH
                        model_custom.styles.min_width = MODEL_CUSTOM_BUTTON_WIDTH
                        model_custom.styles.max_width = MODEL_CUSTOM_BUTTON_WIDTH
                        model_custom.display = has_model_options
                        yield model_custom
                        supports_discovery = self._provider_supports_model_discovery(
                            self._settings.provider
                        )
                        model_discover = Button(
                            MODEL_DISCOVER_BUTTON_LABEL,
                            id=MODEL_DISCOVER_BUTTON_ID,
                            disabled=not supports_discovery,
                        )
                        model_discover.tooltip = (
                            "List models served at the Base URL (/v1/models)"
                        )
                        model_discover.styles.width = MODEL_DISCOVER_BUTTON_WIDTH
                        model_discover.styles.min_width = MODEL_DISCOVER_BUTTON_WIDTH
                        model_discover.styles.max_width = MODEL_DISCOVER_BUTTON_WIDTH
                        model_discover.display = supports_discovery
                        yield model_discover
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Base URL")
                        base_url_input = ConsoleSettingsInput(
                            value=base_url or "",
                            id="console-settings-base-url",
                            disabled=not uses_base_url,
                            classes="console-settings-control",
                        )
                        base_url_input.display = uses_base_url
                        yield base_url_input
                    discover_status = Static(
                        "",
                        id=MODEL_DISCOVER_STATUS_ID,
                        classes="console-settings-modal-row",
                        markup=False,
                    )
                    discover_status.display = False
                    yield discover_status

                with Vertical(classes="console-settings-modal-section"):
                    yield Static("Sampling", classes="destination-section")
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Temperature")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.temperature),
                            id="console-settings-temperature",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Top P")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.top_p),
                            id="console-settings-top-p",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Min P")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.min_p),
                            id="console-settings-min-p",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Top K")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.top_k),
                            id="console-settings-top-k",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Max tokens")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.max_tokens),
                            id="console-settings-max-tokens",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Seed")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.seed),
                            id="console-settings-seed",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Presence")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.presence_penalty),
                            id="console-settings-presence-penalty",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Frequency")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.frequency_penalty),
                            id="console-settings-frequency-penalty",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Streaming")
                        streaming_toggle = Button(
                            self._streaming_toggle_label(),
                            id="console-settings-streaming",
                        )
                        streaming_toggle.tooltip = "Toggle streaming on or off for this session"
                        streaming_toggle.styles.width = STREAMING_TOGGLE_WIDTH
                        streaming_toggle.styles.min_width = STREAMING_TOGGLE_WIDTH
                        streaming_toggle.styles.max_width = STREAMING_TOGGLE_WIDTH
                        yield streaming_toggle

                with Vertical(classes="console-settings-modal-section"):
                    yield Static("Provider-specific", classes="destination-section")
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Reasoning")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.reasoning_effort),
                            placeholder=self._choice_placeholder("console-settings-reasoning-effort"),
                            id="console-settings-reasoning-effort",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Summary")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.reasoning_summary),
                            placeholder=self._choice_placeholder("console-settings-reasoning-summary"),
                            id="console-settings-reasoning-summary",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Verbosity")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.verbosity),
                            placeholder=self._choice_placeholder("console-settings-verbosity"),
                            id="console-settings-verbosity",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Thinking")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.thinking_effort),
                            placeholder=self._choice_placeholder("console-settings-thinking-effort"),
                            id="console-settings-thinking-effort",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Budget")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.thinking_budget_tokens),
                            id="console-settings-thinking-budget-tokens",
                            classes="console-settings-control",
                        )

                with Vertical(classes="console-settings-modal-section"):
                    yield Static("Context", classes="destination-section")
                    yield Static(
                        f"Current         {self._context_label()}",
                        id="console-settings-context-current",
                        classes="console-settings-modal-row",
                        markup=False,
                    )
                    yield Static(
                        f"Sources         {self._sources_label()}",
                        id="console-settings-context-sources",
                        classes="console-settings-modal-row",
                        markup=False,
                    )
                    yield Static(
                        "Estimate only; no truncation changes in this version.",
                        id="console-settings-context-note",
                        classes="console-settings-modal-row",
                        markup=False,
                    )

                with Vertical(classes="console-settings-modal-section"):
                    yield Static("Identity", classes="destination-section")
                    yield Static(
                        f"Current         {self._identity_current_label()}",
                        id="console-settings-identity-current",
                        classes="console-settings-modal-row",
                        markup=False,
                    )
                    yield Static(
                        f"Persona         {self._persona_label()} [read-only]",
                        id="console-settings-persona-readonly",
                        classes="console-settings-modal-row",
                        markup=False,
                    )
                    yield Static(
                        f"Character       {self._character_label()} [read-only]",
                        id="console-settings-character-readonly",
                        classes="console-settings-modal-row",
                        markup=False,
                    )

            with Horizontal(
                id="console-settings-actions",
                classes="console-settings-modal-row console-settings-modal-actions",
            ):
                yield Button("Cancel", id="console-settings-cancel")
                save_default = Button(
                    "Save as default",
                    id="console-settings-save-default",
                    disabled=not self._can_save,
                )
                save_default.tooltip = (
                    "Apply to this session and write these values to your config "
                    "defaults (provider settings + streaming)."
                )
                # Match the 1-row Cancel/Save action styling (their sizes come
                # from id-scoped app CSS this button's id does not inherit).
                save_default.styles.height = 1
                save_default.styles.min_height = 1
                save_default.styles.width = 17
                save_default.styles.min_width = 17
                yield save_default
                yield Button("Save", id="console-settings-save", variant="primary", disabled=not self._can_save)

    def on_mount(self) -> None:
        if self._focus_model:
            self._focus_model_control()

    def on_click(self, event: events.Click) -> None:
        """Recover select clicks redirected through focused Textual Web inputs.

        Args:
            event: Click event that may have been redirected from a focused
                settings input.
        """
        self._open_select_from_redirected_settings_click(event)

    def _open_select_from_redirected_settings_click(self, event: events.Click) -> None:
        """Recover settings controls when an input-held click lands on them.

        Args:
            event: Click event to recover when Textual Web keeps routing clicks
                through a focused settings input.
        """
        captured_widget = self.app.mouse_captured
        click_origin = getattr(event, "widget", None)
        focused_widget = self.app.focused
        screen_routed_click = click_origin is self and isinstance(focused_widget, ConsoleSettingsInput)
        if (
            not isinstance(captured_widget, ConsoleSettingsInput)
            and not isinstance(click_origin, ConsoleSettingsInput)
            and not screen_routed_click
        ):
            return
        if isinstance(captured_widget, ConsoleSettingsInput):
            captured_widget.release_mouse()

        if event.button != 1 or event.screen_x is None or event.screen_y is None:
            return

        for select in self.query(Select):
            if select.disabled or not select.display:
                continue
            select_region = _settings_screen_region(select)
            if select_region.contains(event.screen_x, event.screen_y):
                select.focus()
                select.action_show_overlay()
                event.stop()
                return
        for button in self.query(Button):
            if button.disabled or not button.display:
                continue
            button_region = _settings_screen_region(button)
            if button_region.contains(event.screen_x, event.screen_y):
                button.focus()
                button.press()
                event.stop()
                return

    def _has_selected_model(self) -> bool:
        try:
            return bool(self._current_model_value())
        except (NoMatches, QueryError):
            return bool(self._model_for_provider(self._active_provider))

    def _is_model_setup_mode(self) -> bool:
        return self._focus_model and not self._has_selected_model()

    def _readiness_detail(self, default_detail: str) -> str:
        if self._is_model_setup_mode():
            guidance = "Choose a model to enable sending."
            detail = default_detail.strip()
            if detail and not self._is_ready_readiness_detail(detail):
                return f"{guidance}\n{detail}"
            return guidance
        return default_detail

    @staticmethod
    def _is_ready_readiness_detail(detail: str) -> bool:
        normalized = detail.strip().lower()
        return normalized in {"", "ready."} or " is ready" in normalized

    def _provider_model_section_classes(self) -> str:
        classes = "console-settings-modal-section"
        if self._is_model_setup_mode():
            classes += " console-settings-primary-section"
        return classes

    def _modal_label(self, text: str) -> Static:
        label = Static(text, classes="console-settings-modal-label")
        label.styles.width = MODAL_LABEL_WIDTH
        label.styles.min_width = MODAL_LABEL_WIDTH
        label.styles.max_width = MODAL_LABEL_WIDTH
        return label

    def action_dismiss(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#console-settings-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#console-settings-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        draft = self._validated_draft_or_show_errors()
        if draft is None:
            return
        self.dismiss(draft)

    @on(Button.Pressed, "#console-settings-save-default")
    def _save_as_default(self, event: Button.Pressed) -> None:
        """Apply the draft to the session and write it through to config defaults."""
        event.stop()
        draft = self._validated_draft_or_show_errors()
        if draft is None:
            return
        try:
            saved = save_settings_to_cli_config(self._default_persist_sections(draft))
        except Exception:
            saved = False
        if not saved:
            self.query_one("#console-settings-error", Static).update(
                CONSOLE_SETTINGS_SAVE_DEFAULT_FAILED_COPY
            )
            return
        self.dismiss(draft)

    def _validated_draft_or_show_errors(self) -> ConsoleSessionSettings | None:
        """Build the draft, surfacing validation errors in the modal when invalid."""
        draft = self._build_draft()
        errors = [
            *self._required_sampling_errors(),
            *self._provider_choice_input_errors(),
            *validate_console_session_settings(draft, app_config=self._app_config),
        ]
        if errors:
            self.query_one("#console-settings-error", Static).update("\n".join(errors))
            return None
        return draft

    def _default_persist_sections(
        self,
        draft: ConsoleSessionSettings,
    ) -> dict[str, dict[str, object]]:
        """Build config sections written through by Save as default.

        Provider-scoped values land in ``[api_settings.<provider>]`` (the source
        ``build_default_console_session_settings`` reads on the next boot);
        streaming lands on the canonical ``chat_defaults.streaming`` key (the
        legacy ``enable_streaming`` bridge only applies when the canonical key
        is absent). ``chat_defaults.provider`` is written too — the default
        provider itself resolves ONLY from that key, so omitting it would make
        "Save as default" keep booting into the previous provider. ``None``
        values are skipped rather than deleting existing defaults.
        """
        sections: dict[str, dict[str, object]] = {}
        provider_key = provider_config_key(draft.provider)
        provider_values: dict[str, object] = {}
        model = normalize_console_model_value(draft.model)
        if model:
            provider_values["model"] = model
        base_url = (draft.base_url or "").strip()
        if base_url and self._provider_uses_base_url(draft.provider):
            provider_values[self._endpoint_persist_key(provider_key)] = base_url
        for field_name in PROVIDER_DEFAULT_PERSIST_FIELDS:
            value = getattr(draft, field_name)
            if value is not None:
                provider_values[field_name] = value
        if provider_key and provider_values:
            sections[f"api_settings.{provider_key}"] = provider_values
        chat_defaults: dict[str, object] = {"streaming": bool(draft.streaming)}
        if provider_key:
            chat_defaults["provider"] = provider_key
        sections["chat_defaults"] = chat_defaults
        return sections

    def _endpoint_persist_key(self, provider_key: str) -> str:
        """Return the endpoint config key to write, preferring the configured one."""
        provider_settings = self._provider_settings(provider_key)
        for key in _ENDPOINT_PERSIST_KEYS:
            value = provider_settings.get(key)
            if isinstance(value, str) and value.strip():
                return key
        return "api_url"

    @on(Button.Pressed, "#console-settings-streaming")
    def _toggle_streaming(self, event: Button.Pressed) -> None:
        """Cycle the streaming draft between on and off."""
        event.stop()
        self._streaming_draft = not self._streaming_draft
        event.button.label = self._streaming_toggle_label()

    def _streaming_toggle_label(self) -> str:
        return STREAMING_ON_LABEL if self._streaming_draft else STREAMING_OFF_LABEL

    def _choice_placeholder(self, input_id: str) -> str:
        """Return the accepted-values placeholder for an enumerated choice input."""
        for _label, choice_input_id, placeholder in PROVIDER_CHOICE_INPUTS:
            if choice_input_id == input_id:
                return placeholder
        return ""

    @on(Select.Changed, "#console-settings-provider")
    def _provider_changed(self, event: Select.Changed) -> None:
        self._store_current_model_for_provider(self._active_provider)
        self._store_current_base_url_for_provider(self._active_provider)
        provider = str(event.value or "")
        model = self._model_for_provider(provider)
        base_url = self._base_url_for_provider(provider)
        self._active_provider = provider
        self._sync_model_controls(provider, model)
        self._sync_base_url_control(provider, base_url)
        self._sync_model_discover_controls(provider)
        self._sync_readiness_display()

    @on(Select.Changed, "#console-settings-model-select")
    def _model_select_changed(self, event: Select.Changed) -> None:
        self._sync_readiness_display()

    @on(Input.Changed, "#console-settings-model-input")
    def _model_input_changed(self, event: Input.Changed) -> None:
        self._sync_readiness_display()

    @on(Button.Pressed, "#console-settings-model-custom")
    def _model_custom_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._toggle_manual_model_input()

    @staticmethod
    def _provider_supports_model_discovery(provider: str) -> bool:
        """Return whether the provider serves an OpenAI-compatible model list."""
        return provider_config_key(provider) in URL_BASED_PROVIDER_KEYS

    @on(Button.Pressed, f"#{MODEL_DISCOVER_BUTTON_ID}")
    def _model_discover_pressed(self, event: Button.Pressed) -> None:
        """Probe the current base-URL draft for its served model list."""
        event.stop()
        provider = str(self.query_one("#console-settings-provider", Select).value or "")
        if not self._provider_supports_model_discovery(provider):
            return
        base_url = self._current_base_url_value(provider) or ""
        if not base_url:
            self._set_model_discover_status(MODEL_DISCOVER_MISSING_URL_COPY)
            return
        normalized_probe_url = normalize_probe_base_url(base_url)
        # PR #608 review: user-entered endpoint must pass the shared
        # input_validation boundary before any network use.
        if normalized_probe_url is None or not validate_url(normalized_probe_url):
            self._set_model_discover_status(MODEL_DISCOVER_INVALID_URL_COPY)
            return
        event.button.disabled = True
        self._set_model_discover_status(f"Contacting {endpoint_display(base_url)}…")
        self.run_worker(
            self._run_model_discovery(provider, base_url),
            exclusive=True,
            group="console-settings-model-discovery",
        )

    async def _run_model_discovery(self, provider: str, base_url: str) -> None:
        """Run the model probe off the draft URL and apply the outcome.

        Args:
            provider: Provider draft value at press time.
            base_url: Base-URL draft at press time.
        """
        try:
            result = await self._model_prober(base_url, provider_config_key(provider))
        except Exception:
            result = LocalModelProbeResult(
                ok=False,
                base_url=base_url,
                detail=f"No models endpoint at {endpoint_display(base_url)}.",
            )
        self._apply_model_discovery_result(provider, result)

    def _apply_model_discovery_result(
        self,
        provider: str,
        result: LocalModelProbeResult,
    ) -> None:
        """Surface a probe outcome: model Select on success, honest copy otherwise.

        Args:
            provider: Provider the probe was started for; results for a
                provider the user has since switched away from are dropped.
            result: Probe outcome from the discovery module.
        """
        try:
            discover = self.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        except (NoMatches, QueryError):
            return
        discover.disabled = not self._provider_supports_model_discovery(self._active_provider)
        if provider != self._active_provider:
            self._set_model_discover_status("")
            return
        display = endpoint_display(result.base_url)
        if not result.ok:
            self._set_model_discover_status(result.detail or f"No models endpoint at {display}.")
            return
        if not result.model_ids:
            self._set_model_discover_status(f"No models reported at {display}.")
            return
        self._discovered_model_ids[provider] = tuple(result.model_ids)
        self._sync_model_controls(provider, self._current_model_value())
        count = len(result.model_ids)
        noun = "model" if count == 1 else "models"
        self._set_model_discover_status(f"Found {count} {noun} at {display}.")
        self._sync_readiness_display()

    def _set_model_discover_status(self, text: str) -> None:
        """Update the inline discovery status line, hiding it when blank."""
        try:
            status = self.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static)
        except (NoMatches, QueryError):
            return
        status.update(text)
        status.display = bool(text.strip())

    def _sync_model_discover_controls(self, provider: str) -> None:
        """Show the discovery affordance only for URL-based providers."""
        supports_discovery = self._provider_supports_model_discovery(provider)
        try:
            discover = self.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        except (NoMatches, QueryError):
            return
        discover.display = supports_discovery
        discover.disabled = not supports_discovery
        self._set_model_discover_status("")

    def _sync_readiness_display(self) -> None:
        draft = self._build_draft()
        readiness = build_console_settings_readiness(draft, app_config=self._app_config)
        self.query_one("#console-settings-readiness", Static).update(
            self._readiness_detail(readiness.detail)
        )
        self._sync_provider_model_section_emphasis()

    def _sync_provider_model_section_emphasis(self) -> None:
        section = self.query_one("#console-settings-provider-model-section", Vertical)
        if self._is_model_setup_mode():
            section.add_class("console-settings-primary-section")
        else:
            section.remove_class("console-settings-primary-section")

    def _build_draft(self) -> ConsoleSessionSettings:
        provider = str(self.query_one("#console-settings-provider", Select).value or "")
        return ConsoleSessionSettings(
            provider=provider,
            model=self._current_model_value(),
            base_url=self._current_base_url_value(provider),
            temperature=self._parse_float_input("console-settings-temperature", self._settings.temperature),
            top_p=self._parse_float_input("console-settings-top-p", self._settings.top_p),
            min_p=self._parse_optional_float_input("console-settings-min-p"),
            top_k=self._parse_optional_int_input("console-settings-top-k"),
            max_tokens=self._parse_optional_int_input("console-settings-max-tokens"),
            seed=self._parse_optional_int_input("console-settings-seed"),
            presence_penalty=self._parse_optional_float_input("console-settings-presence-penalty"),
            frequency_penalty=self._parse_optional_float_input("console-settings-frequency-penalty"),
            reasoning_effort=self._parse_optional_choice_input("console-settings-reasoning-effort"),
            reasoning_summary=self._parse_optional_choice_input("console-settings-reasoning-summary"),
            verbosity=self._parse_optional_choice_input("console-settings-verbosity"),
            thinking_effort=self._parse_optional_choice_input("console-settings-thinking-effort"),
            thinking_budget_tokens=self._parse_optional_int_input("console-settings-thinking-budget-tokens"),
            streaming=self._streaming_draft,
            persona_label=self._settings.persona_label,
            character_label=self._settings.character_label,
        )

    def _sync_model_controls(self, provider: str, current_model: str | None) -> None:
        model_select = self.query_one("#console-settings-model-select", Select)
        model_input = self.query_one("#console-settings-model-input", Input)
        model_custom = self.query_one("#console-settings-model-custom", Button)
        current_model = normalize_console_model_value(current_model)
        model_options = self._model_select_options(provider, current_model)
        if model_options:
            model_select.set_options(model_options)
            option_values = {str(value) for _, value in model_options}
            selected = current_model if current_model in option_values else str(model_options[0][1])
            model_select.value = selected
            use_model_select = self._should_use_model_select(provider, selected, model_options)
            model_select.disabled = not use_model_select
            model_select.display = use_model_select
            model_input.disabled = True
            model_input.display = not use_model_select
            model_input.value = selected
            model_custom.label = "Custom model"
            model_custom.disabled = False
            model_custom.display = True
            return

        fallback = current_model or ""
        model_select.set_options([("No configured models", "")])
        model_select.value = ""
        model_select.disabled = True
        model_select.display = False
        model_input.value = fallback
        model_input.disabled = False
        model_input.display = True
        model_custom.label = "Custom model"
        model_custom.disabled = True
        model_custom.display = False

    def _toggle_manual_model_input(self) -> None:
        model_select = self.query_one("#console-settings-model-select", Select)
        model_input = self.query_one("#console-settings-model-input", Input)
        model_custom = self.query_one("#console-settings-model-custom", Button)

        if model_input.display:
            if model_input.disabled:
                model_input.disabled = False
                model_custom.label = "Model list"
                model_input.focus()
                self._sync_readiness_display()
                return
            provider = str(self.query_one("#console-settings-provider", Select).value or "")
            current_model = normalize_console_model_value(model_input.value)
            self._sync_model_controls(provider, current_model)
            self._sync_readiness_display()
            if model_select.display and not model_select.disabled:
                model_select.focus()
            else:
                model_custom.focus()
            return

        model_input.value = normalize_console_model_value(model_select.value) or ""
        model_select.display = False
        model_select.disabled = True
        model_input.display = True
        model_input.disabled = False
        model_custom.label = "Model list"
        model_input.focus()
        self._sync_readiness_display()

    def _focus_model_control(self) -> None:
        model_select = self.query_one("#console-settings-model-select", Select)
        if model_select.display and not model_select.disabled:
            model_select.focus()
            return
        model_input = self.query_one("#console-settings-model-input", Input)
        model_input.focus()

    def _provider_select_options(self) -> list[tuple[str, str]]:
        """Return provider options labeled with shared catalog display names.

        Option values stay raw provider config keys (task-191); only the
        rendered labels change, and the ``(WIP)`` marker from the underlying
        option builder is preserved.
        """
        options: list[tuple[str, str]] = []
        for option in build_console_provider_options(self._providers_models):
            label = provider_display_name(option.value)
            if option.label.endswith(" (WIP)"):
                label = f"{label} (WIP)"
            options.append((label, option.value))
        if self._settings.provider and self._settings.provider not in {value for _, value in options}:
            options.append((provider_display_name(self._settings.provider), self._settings.provider))
        return options or [(provider_display_name(self._settings.provider), self._settings.provider)]

    def _model_select_options(self, provider: str, current_model: str | None) -> list[tuple[str, str]]:
        options = [
            (option.label, option.value)
            for option in build_console_model_options(provider, self._providers_models, current_model)
        ]
        option_values = {value for _, value in options}
        for model_id in self._discovered_model_ids.get(provider, ()):
            normalized = normalize_console_model_value(model_id)
            if normalized and normalized not in option_values:
                option_values.add(normalized)
                # Server-supplied text: escape so Rich-markup-like ids cannot
                # style or spoof the option label (PR #608 review).
                options.append((escape_markup(normalized), normalized))
        return options

    def _configured_model_select_options(self, provider: str) -> list[tuple[str, str]]:
        return [
            (option.label, option.value)
            for option in build_console_model_options(provider, self._providers_models, None)
        ]

    def _should_use_model_select(
        self,
        provider: str,
        selected_model: str | None,
        model_options: list[tuple[str, str]],
    ) -> bool:
        """Return whether the model list should be an interactive Select.

        The approved single-model fix only applies to the steady-state local
        runtime case where the user already has that exact model selected. Other
        single-option states still need an interactive list because they are
        recovery/setup flows, custom/freeform providers, or provider-switch
        transitions where saving must capture the resolved model.
        """
        if not model_options:
            return False
        if len(model_options) > 1:
            return True

        provider_key = provider_config_key(provider)
        if provider_key == "custom":
            return True
        configured_values = {str(value) for _, value in self._configured_model_select_options(provider)}
        selected_model = normalize_console_model_value(selected_model)

        if self._focus_model:
            return True
        if provider != self._settings.provider:
            if provider_key in {"llama_cpp", "local_llamacpp"}:
                return True
            return bool(selected_model and selected_model not in configured_values)

        settings_model = normalize_console_model_value(self._settings.model)
        if not settings_model:
            return True
        if provider_key not in {"llama_cpp", "local_llamacpp", "openai"}:
            return True
        if selected_model and selected_model not in configured_values:
            return True
        return not selected_model or selected_model != settings_model

    def _set_provider_model_draft(self, provider: str, value: object) -> None:
        """Store a per-provider draft model, normalized once at this boundary."""
        self._provider_model_drafts[provider] = normalize_console_model_value(value)

    def _provider_model_draft(self, provider: str) -> str | None:
        """Return the already-normalized draft model stored for a provider."""
        return self._provider_model_drafts.get(provider)

    def _store_current_model_for_provider(self, provider: str) -> None:
        if provider:
            self._set_provider_model_draft(provider, self._current_model_value())

    def _store_current_base_url_for_provider(self, provider: str) -> None:
        if provider and self._provider_uses_base_url(provider):
            self._provider_base_url_drafts[provider] = (
                self.query_one("#console-settings-base-url", Input).value.strip()
            )

    def _model_for_provider(self, provider: str) -> str | None:
        if provider in self._provider_model_drafts:
            stored_model = normalize_console_model_value(self._provider_model_drafts[provider])
            if stored_model:
                return stored_model
        configured_model = self._default_model_for_provider(provider)
        if configured_model:
            return configured_model
        if provider == self._settings.provider:
            settings_model = normalize_console_model_value(self._settings.model)
            if settings_model:
                return settings_model
        configured_model_options = self._configured_model_select_options(provider)
        if configured_model_options:
            return configured_model_options[0][1]
        return None

    def _default_model_for_provider(self, provider: str) -> str | None:
        provider_key = provider_config_key(provider)
        provider_settings = self._provider_settings(provider_key)
        for key in ("model", "api_model", "default_model"):
            configured_model = normalize_console_model_value(provider_settings.get(key))
            if configured_model:
                return configured_model
        return None

    def _sync_base_url_control(self, provider: str, base_url: str | None) -> None:
        base_url_input = self.query_one("#console-settings-base-url", Input)
        uses_base_url = self._provider_uses_base_url(provider)
        base_url_input.value = base_url or ""
        base_url_input.disabled = not uses_base_url
        base_url_input.display = uses_base_url

    def _current_base_url_value(self, provider: str) -> str | None:
        if not self._provider_uses_base_url(provider):
            return None
        return self.query_one("#console-settings-base-url", Input).value.strip() or None

    def _base_url_for_provider(self, provider: str) -> str | None:
        if not self._provider_uses_base_url(provider):
            return None
        if provider in self._provider_base_url_drafts:
            return self._provider_base_url_drafts[provider] or None
        if provider == self._settings.provider and self._settings.base_url:
            return self._initial_base_url_for_provider(provider, self._settings.base_url)
        return self._default_base_url_for_provider(provider)

    def _provider_uses_base_url(self, provider: str) -> bool:
        provider_key = provider_config_key(provider)
        provider_settings = self._provider_settings(provider_key)
        return provider_key in URL_BASED_PROVIDER_KEYS or any(
            key in provider_settings for key in ("api_base_url", "api_url", "base_url", "api_base")
        )

    def _provider_settings(self, provider_key: str) -> Mapping[str, object]:
        api_settings = self._app_config.get("api_settings", {})
        if not isinstance(api_settings, Mapping):
            return {}
        for configured_provider, configured_settings in api_settings.items():
            if (
                provider_config_key(str(configured_provider)) == provider_key
                and isinstance(configured_settings, Mapping)
            ):
                return configured_settings
        return {}

    def _default_base_url_for_provider(self, provider: str) -> str | None:
        provider_key = provider_config_key(provider)
        provider_settings = self._provider_settings(provider_key)
        base_url = first_configured_endpoint(provider_settings)
        if provider_key in {"llama_cpp", "local_llamacpp"}:
            return normalize_llamacpp_base_url(base_url or DEFAULT_LLAMACPP_BASE_URL)
        return base_url

    def _initial_base_url_for_provider(self, provider: str, session_base_url: str | None) -> str | None:
        provider_key = provider_config_key(provider)
        provider_settings = self._provider_settings(provider_key)
        configured_base_url = self._default_base_url_for_provider(provider)
        session_base_url = self._normalized_base_url_for_provider(provider_key, session_base_url)
        if not session_base_url:
            return configured_base_url
        if (
            configured_base_url
            and self._matches_lower_priority_configured_endpoint(
                provider_key,
                session_base_url,
                provider_settings,
            )
        ):
            return configured_base_url
        return session_base_url

    def _matches_lower_priority_configured_endpoint(
        self,
        provider_key: str,
        session_base_url: str,
        provider_settings: Mapping[str, object],
    ) -> bool:
        configured_endpoint = self._normalized_base_url_for_provider(
            provider_key,
            first_configured_endpoint(provider_settings),
        )
        if not configured_endpoint:
            return False
        session_identity = self._endpoint_identity_for_provider(provider_key, session_base_url)
        configured_identity = self._endpoint_identity_for_provider(provider_key, configured_endpoint)
        if session_identity == configured_identity:
            return False
        for key in ("api_url", "base_url", "api_base", "api_endpoint", "endpoint"):
            lower_priority_endpoint = self._normalized_base_url_for_provider(
                provider_key,
                provider_settings.get(key),
            )
            if (
                lower_priority_endpoint
                and session_identity
                == self._endpoint_identity_for_provider(provider_key, lower_priority_endpoint)
            ):
                return True
        return False

    @staticmethod
    def _normalized_base_url_for_provider(provider_key: str, base_url: object | None) -> str | None:
        value = str(base_url or "").strip()
        if not value:
            return None
        if provider_key in {"llama_cpp", "local_llamacpp"}:
            return normalize_llamacpp_base_url(value)
        return value

    @staticmethod
    def _endpoint_identity_for_provider(provider_key: str, base_url: str | None) -> str:
        if provider_key in {"llama_cpp", "local_llamacpp"}:
            base_url = normalize_llamacpp_base_url(base_url)
        return normalize_generic_endpoint_for_compare(base_url)

    @staticmethod
    def _first_string(*values: object) -> str | None:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _current_model_value(self) -> str | None:
        model_select = self.query_one("#console-settings-model-select", Select)
        model_input = self.query_one("#console-settings-model-input", Input)
        if model_select.display and not model_select.disabled:
            return normalize_console_model_value(model_select.value)
        return normalize_console_model_value(model_input.value)

    def _context_label(self) -> str:
        label = self._context_estimate.label.strip() or "unknown"
        return label if "token" in label.lower() else f"{label} tokens"

    def _sources_label(self) -> str:
        if self._context_estimate.staged_context_summary.strip():
            return self._context_estimate.staged_context_summary.strip()
        return "None"

    def _identity_current_label(self) -> str:
        persona = self._persona_label()
        character = self._settings.character_label.strip()
        return f"{persona} / {character}" if character else persona

    def _persona_label(self) -> str:
        return self._settings.persona_label.strip() or "General"

    def _character_label(self) -> str:
        return self._settings.character_label.strip() or "None"

    @staticmethod
    def _format_value(value: object) -> str:
        return "" if value is None else str(value)

    def _parse_float_input(self, input_id: str, fallback: float) -> object:
        raw_value = self.query_one(f"#{input_id}", Input).value.strip()
        if not raw_value:
            return fallback
        try:
            return float(raw_value)
        except ValueError:
            return raw_value

    def _required_sampling_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.query_one("#console-settings-temperature", Input).value.strip():
            errors.append("Temperature is required.")
        if not self.query_one("#console-settings-top-p", Input).value.strip():
            errors.append("Top P is required.")
        return errors

    def _provider_choice_input_errors(self) -> list[str]:
        errors: list[str] = []
        for label, input_id, _placeholder in PROVIDER_CHOICE_INPUTS:
            raw_value = self.query_one(f"#{input_id}", Input).value.strip()
            if raw_value and not validate_text_input(
                raw_value,
                max_length=PROVIDER_CHOICE_INPUT_MAX_LENGTH,
                allow_html=False,
            ):
                errors.append(f"{label} contains unsupported text.")
        return errors

    def _parse_optional_float_input(self, input_id: str) -> object:
        raw_value = self.query_one(f"#{input_id}", Input).value.strip()
        if not raw_value:
            return None
        try:
            return float(raw_value)
        except ValueError:
            return raw_value

    def _parse_optional_int_input(self, input_id: str) -> object:
        raw_value = self.query_one(f"#{input_id}", Input).value.strip()
        if not raw_value:
            return None
        try:
            return int(raw_value)
        except ValueError:
            return raw_value

    def _parse_optional_text_input(self, input_id: str) -> str | None:
        raw_value = self.query_one(f"#{input_id}", Input).value.strip()
        return raw_value or None

    def _parse_optional_choice_input(self, input_id: str) -> str | None:
        raw_value = self._parse_optional_text_input(input_id)
        return raw_value.lower() if raw_value else None
