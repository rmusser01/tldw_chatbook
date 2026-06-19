"""Console session settings modal."""

from __future__ import annotations

from typing import Mapping

from textual import events
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static

from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.Chat.console_provider_endpoints import (
    first_configured_endpoint,
    normalize_generic_endpoint_for_compare,
)
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


MODEL_INPUT_PLACEHOLDER = "Enter model id"
MODAL_LABEL_WIDTH = 16
MODEL_CUSTOM_BUTTON_WIDTH = 18


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
    ) -> None:
        super().__init__()
        self._settings = settings
        self._app_config = app_config
        self._providers_models = providers_models
        self._context_estimate = context_estimate
        self._can_save = can_save
        self._focus_model = focus_model
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
                "",
                id="console-settings-error",
                classes="console-settings-error console-settings-error-summary",
                markup=False,
            )

            with Vertical(id="console-settings-body", classes="console-settings-body"):
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
                            disabled=not has_model_options,
                            classes="console-settings-control",
                        )
                        model_select.styles.width = "1fr"
                        model_select.styles.min_width = 0
                        model_select.display = has_model_options
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
                        model_input.display = not has_model_options
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
                        yield self._modal_label("Streaming")
                        yield Checkbox(
                            value=self._settings.streaming,
                            id="console-settings-streaming",
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
        """Open a settings select when an input-held click lands on the select.

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
            select_region = getattr(select, "screen_region", select.region)
            if select_region.contains(event.screen_x, event.screen_y):
                select.focus()
                select.action_show_overlay()
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
        draft = self._build_draft()
        errors = [
            *self._required_sampling_errors(),
            *validate_console_session_settings(draft, app_config=self._app_config),
        ]
        if errors:
            self.query_one("#console-settings-error", Static).update("\n".join(errors))
            return
        self.dismiss(draft)

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
            streaming=self.query_one("#console-settings-streaming", Checkbox).value,
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
            model_select.disabled = False
            model_select.display = True
            model_input.disabled = True
            model_input.display = False
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
            provider = str(self.query_one("#console-settings-provider", Select).value or "")
            current_model = normalize_console_model_value(model_input.value)
            self._sync_model_controls(provider, current_model)
            self._sync_readiness_display()
            model_select.focus()
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
        options = [(option.label, option.value) for option in build_console_provider_options(self._providers_models)]
        if self._settings.provider and self._settings.provider not in {value for _, value in options}:
            options.append((self._settings.provider, self._settings.provider))
        return options or [(self._settings.provider, self._settings.provider)]

    def _model_select_options(self, provider: str, current_model: str | None) -> list[tuple[str, str]]:
        return [
            (option.label, option.value)
            for option in build_console_model_options(provider, self._providers_models, current_model)
        ]

    def _configured_model_select_options(self, provider: str) -> list[tuple[str, str]]:
        return [
            (option.label, option.value)
            for option in build_console_model_options(provider, self._providers_models, None)
        ]

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
