"""Settings destination shell for global app preferences."""

import logging
import os
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.events import Key
from textual.reactive import reactive
from textual.widgets import Button, Input, Rule, Static

from ...Sync_Interop.sync_promotion_state import SyncPromotionState, build_sync_promotion_state
from ...Sync_Interop.sync_readiness import DEFAULT_SYNC_ELIGIBILITY_REGISTRY, build_sync_readiness_report
from ...Widgets.destination_workbench import DestinationModeStrip
from ...config import DEFAULT_CONFIG_PATH, coerce_bool_setting, save_setting_to_cli_config
from ..Navigation.base_app_screen import BaseAppScreen
from .provider_model_resolution import resolve_effective_provider_model
from .settings_config_adapter import SettingsConfigAdapter, redact_secret_text
from .settings_config_models import SettingsCategoryId, SettingsCategorySummary, SettingsDraft
from ..Navigation.main_navigation import NavigateToScreen


logger = logging.getLogger(__name__)


class SettingsScreen(BaseAppScreen):
    """Global preferences, appearance, accounts, storage, and app behavior."""

    BINDINGS = [
        ("s", "settings_save_category", "Save Settings category"),
        ("r", "settings_revert_category", "Revert Settings category"),
        ("t", "settings_test_category", "Test Settings category"),
    ]

    active_category = reactive(SettingsCategoryId.OVERVIEW.value, recompose=True)

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "settings", **kwargs)
        self._settings_drafts: dict[SettingsCategoryId, SettingsDraft] = {}
        self._provider_test_result = "Provider test has not run."

    def _category_summaries(self) -> tuple[SettingsCategorySummary, ...]:
        return (
            SettingsCategorySummary(
                SettingsCategoryId.OVERVIEW,
                "Overview",
                "Readiness, storage, privacy, Console behavior, diagnostics.",
                "Active",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.PROVIDERS_MODELS,
                "Providers & Models",
                "Default provider, model, and readiness shared with Console.",
                "Shared",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.APPEARANCE,
                "Appearance",
                "Theme, density, and visual customization surface.",
                "Customize",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.STORAGE,
                "Storage",
                "Config path, local databases, and file locations.",
                "Local",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.PRIVACY_SECURITY,
                "Privacy & Security",
                "Secrets, encryption, redaction, and local privacy boundaries.",
                "Local",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.CONSOLE_BEHAVIOR,
                "Console Behavior",
                "Composer, large paste handling, and chat-flow defaults.",
                "Console",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.DIAGNOSTICS,
                "Diagnostics",
                "Config validation, logs, and troubleshooting signals.",
                "Validate",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.ADVANCED_CONFIG,
                "Advanced Config",
                "Raw TOML view and expert configuration editing.",
                "Advanced",
            ),
        )

    def _active_summary(self) -> SettingsCategorySummary:
        for summary in self._category_summaries():
            if summary.category.value == self.active_category:
                return summary
        return self._category_summaries()[0]

    def _console_settings(self) -> dict:
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            self.app_instance.app_config = {}
            app_config = self.app_instance.app_config
        console_settings = app_config.setdefault("console", {})
        if not isinstance(console_settings, dict):
            console_settings = {}
            app_config["console"] = console_settings
        return console_settings

    def _chat_defaults(self) -> dict:
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            self.app_instance.app_config = {}
            app_config = self.app_instance.app_config
        chat_defaults = app_config.setdefault("chat_defaults", {})
        if not isinstance(chat_defaults, dict):
            chat_defaults = {}
            app_config["chat_defaults"] = chat_defaults
        return chat_defaults

    def _loaded_collapse_large_pastes_enabled(self) -> bool:
        return coerce_bool_setting(
            self._console_settings().get("collapse_large_pastes", True),
            True,
        )

    def _collapse_large_pastes_enabled(self) -> bool:
        draft = self._settings_drafts.get(SettingsCategoryId.CONSOLE_BEHAVIOR)
        if draft is not None and "collapse_large_pastes" in draft.values:
            return coerce_bool_setting(
                draft.values.get("collapse_large_pastes"),
                True,
            )
        return coerce_bool_setting(
            self._console_settings().get("collapse_large_pastes", True),
            True,
        )

    def _collapse_large_pastes_label(self) -> str:
        state = "Enabled" if self._collapse_large_pastes_enabled() else "Disabled"
        return f"{state}: collapse large pastes"

    def _update_console_paste_summary(self) -> None:
        try:
            summary = self.query_one("#settings-overview-console-paste-collapse", Static)
        except QueryError:
            return
        summary.update(f"Console paste collapse: {self._collapse_large_pastes_label()}")

    def _category_has_unsaved_changes(self, category: SettingsCategoryId) -> bool:
        draft = self._settings_drafts.get(category)
        return bool(draft and draft.is_dirty)

    def _category_status(self, summary: SettingsCategorySummary) -> str:
        if self._category_has_unsaved_changes(summary.category):
            return "Unsaved"
        return summary.status

    def _active_category_id(self) -> SettingsCategoryId:
        return SettingsCategoryId(self.active_category)

    def _update_draft_status_widgets(self, category: SettingsCategoryId) -> None:
        has_unsaved_changes = self._category_has_unsaved_changes(category)
        status = "Unsaved changes" if has_unsaved_changes else "No unsaved changes"
        try:
            self.query_one("#settings-selected-category-draft-status", Static).update(status)
        except QueryError:
            pass
        try:
            category_status = "Unsaved" if has_unsaved_changes else self._category_summary_by_id(category).status
            self.query_one(f"#settings-category-{category.value}-status", Static).update(
                f"Status: {category_status}"
            )
        except QueryError:
            pass

    def _category_summary_by_id(self, category: SettingsCategoryId) -> SettingsCategorySummary:
        for summary in self._category_summaries():
            if summary.category is category:
                return summary
        return self._category_summaries()[0]

    def _stage_console_large_paste_value(self, value: bool) -> None:
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        draft.set_value(
            "collapse_large_pastes",
            self._loaded_collapse_large_pastes_enabled(),
            value,
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _sync_safety_states(self) -> tuple[SyncPromotionState, ...]:
        labels = {
            "library_collections": "Collections",
            "workspaces": "Workspaces",
        }
        sync_scope_service = getattr(self.app_instance, "sync_scope_service", None)
        list_states = getattr(sync_scope_service, "list_write_sync_promotion_states", None)
        if callable(list_states):
            try:
                return tuple(
                    list_states(
                        domains=list(labels),
                        surface_labels=labels,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load Settings sync safety states; using local fallback. error_type=%s",
                    type(exc).__name__,
                )
        return tuple(
            build_sync_promotion_state(
                domain=domain,
                surface_label=label,
                readiness=build_sync_readiness_report(
                    domain=domain,
                    server_profile_id=None,
                    workspace_id=None,
                    registry=DEFAULT_SYNC_ELIGIBILITY_REGISTRY,
                ),
            )
            for domain, label in labels.items()
        )

    @staticmethod
    def _column_divider(identifier: str) -> Rule:
        divider = Rule(orientation="vertical", id=identifier)
        divider.add_class("destination-pane-divider")
        return divider

    def _config_path(self) -> Path:
        override = os.environ.get("TLDW_CONFIG_PATH")
        return Path(override).expanduser() if override else DEFAULT_CONFIG_PATH

    def _config_writable_status(self) -> str:
        config_path = self._config_path()
        target = config_path if config_path.exists() else config_path.parent
        writable = os.access(target, os.W_OK) if target.exists() else os.access(target.parent, os.W_OK)
        return "writable" if writable else "not writable"

    def _provider_readiness_label(self) -> str:
        resolved = self._resolve_provider_model_for_settings()
        provider = str(resolved.provider or "not selected").strip()
        model = str(resolved.model or "not selected").strip()
        if provider and provider != "not selected":
            return f"Provider readiness: {provider} / {model}"
        return "Provider readiness: needs provider and model"

    def _provider_draft(self) -> SettingsDraft | None:
        return self._settings_drafts.get(SettingsCategoryId.PROVIDERS_MODELS)

    def _provider_draft_value(self, key: str):
        draft = self._provider_draft()
        return draft.values.get(key) if draft is not None and key in draft.values else None

    def _resolve_provider_model_for_settings(self):
        return resolve_effective_provider_model(
            self.app_instance,
            settings_provider=self._provider_draft_value("provider"),
            settings_model=self._provider_draft_value("model"),
        )

    def _provider_setting_values(self) -> dict[str, object]:
        defaults = self._chat_defaults()
        resolved = self._resolve_provider_model_for_settings()
        return {
            "provider": str(resolved.provider or "").strip(),
            "model": str(resolved.model or "").strip(),
            "streaming": coerce_bool_setting(
                self._provider_draft_value("streaming")
                if self._provider_draft_value("streaming") is not None
                else defaults.get("streaming", True),
                True,
            ),
            "temperature": self._provider_draft_value("temperature")
            if self._provider_draft_value("temperature") is not None
            else defaults.get("temperature", 0.7),
        }

    @staticmethod
    def _normalise_temperature(value: object) -> float:
        return float(str(value).strip())

    def _provider_form_values_from_widgets(self) -> dict[str, object]:
        provider = self.query_one("#settings-provider-value", Input).value.strip()
        model = self.query_one("#settings-model-value", Input).value.strip()
        streaming = coerce_bool_setting(
            self.query_one("#settings-streaming-default", Input).value,
            True,
        )
        temperature = self._normalise_temperature(
            self.query_one("#settings-temperature-default", Input).value
        )
        return {
            "provider": provider,
            "model": model,
            "streaming": streaming,
            "temperature": temperature,
        }

    def _stage_provider_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.PROVIDERS_MODELS
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        original = self._chat_defaults().get(key)
        draft.set_value(key, original, value)
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    @staticmethod
    def _provider_key_env_name(provider: str) -> str | None:
        provider_key = provider.strip().lower()
        if provider_key == "openai":
            return "OPENAI_API_KEY"
        if provider_key == "anthropic":
            return "ANTHROPIC_API_KEY"
        if provider_key == "cohere":
            return "COHERE_API_KEY"
        if provider_key == "mistral":
            return "MISTRAL_API_KEY"
        return None

    @staticmethod
    def _provider_endpoint_summary(provider: str, config: dict) -> str:
        provider_key = provider.strip()
        provider_config = config.get(provider_key, {}) if isinstance(config, dict) else {}
        if not isinstance(provider_config, dict):
            provider_config = {}
        endpoint = (
            provider_config.get("api_base")
            or provider_config.get("base_url")
            or provider_config.get("api_url")
            or provider_config.get("endpoint")
        )
        if endpoint:
            return f"Endpoint: {endpoint}"
        if provider_key.lower() in {"llama_cpp", "ollama", "koboldcpp", "local"}:
            return "Endpoint: local/default or provider-specific config"
        return "Endpoint: provider default"

    def _provider_key_status(self, provider: str) -> str:
        env_name = self._provider_key_env_name(provider)
        if env_name is None:
            return "API key: not required for this provider"
        if os.environ.get(env_name):
            return f"{env_name}=<redacted>"
        return f"{env_name}=missing"

    def _run_provider_readiness_test(self) -> str:
        try:
            values = self._provider_form_values_from_widgets()
        except QueryError:
            values = self._provider_setting_values()
        provider = str(values.get("provider") or "").strip()
        model = str(values.get("model") or "").strip()

        findings: list[str] = ["Provider test"]
        if not provider:
            findings.append("provider=missing")
        else:
            findings.append(f"provider={provider}")
        if not model:
            findings.append("model=missing")
        else:
            findings.append(f"model={model}")

        env_name = self._provider_key_env_name(provider)
        if env_name is not None:
            raw_value = os.environ.get(env_name)
            findings.append(f"{env_name}={raw_value if raw_value else 'missing'}")
        else:
            findings.append("api_key=not required")

        if provider.lower() in {"llama_cpp", "ollama", "koboldcpp", "local"}:
            findings.append("network=not checked")

        status = "ready" if provider and model else "blocked"
        findings.append(f"status={status}")
        return redact_secret_text(" | ".join(findings))

    def _update_provider_test_result(self) -> None:
        try:
            self.query_one("#settings-provider-test-result", Static).update(self._provider_test_result)
        except QueryError:
            pass

    def _render_category_buttons(self) -> ComposeResult:
        for summary in self._category_summaries():
            button = Button(
                summary.title,
                id=f"settings-category-{summary.category.value}",
                classes="settings-category-button",
                tooltip=summary.description,
            )
            if summary.category.value == self.active_category:
                button.add_class("settings-active-section")
            yield button
            yield Static(summary.description, classes="destination-section")
            if summary.status:
                yield Static(
                    f"Status: {self._category_status(summary)}",
                    id=f"settings-category-{summary.category.value}-status",
                    classes="destination-section",
                )

    def _render_overview_detail(self) -> ComposeResult:
        yield Static("Overview", classes="destination-section settings-column-title")
        with Vertical(id="settings-overview-card", classes="settings-focus-card"):
            yield Static("Provider readiness", classes="destination-section")
            yield Static(self._provider_readiness_label(), id="settings-overview-provider-readiness")
            yield Static("Storage", classes="destination-section")
            yield Static(
                f"Storage: config path {self._config_path()} ({self._config_writable_status()})",
                id="settings-overview-storage",
            )
            yield Static("Privacy", classes="destination-section")
            yield Static(
                "Privacy: local config by default; secret-looking diagnostics are redacted.",
                id="settings-overview-privacy",
            )
            yield Static(
                f"Console paste collapse: {self._collapse_large_pastes_label()}",
                id="settings-overview-console-paste-collapse",
            )
            yield Static("Diagnostics: validate config before saving raw TOML changes.")

    def _render_provider_detail(self) -> ComposeResult:
        resolved = self._resolve_provider_model_for_settings()
        values = self._provider_setting_values()
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        yield Static("Providers & Models", classes="destination-section settings-column-title")
        with Vertical(id="settings-providers-models-card", classes="settings-focus-card"):
            yield Static("Provider readiness", classes="destination-section")
            yield Static(self._provider_readiness_label())
            yield Static(f"Source: {resolved.provider_source}")
            yield Static(f"Provider source: {resolved.provider_source}")
            yield Static(f"Model source: {resolved.model_source}")
            yield Static("Changes here will share the same provider/model resolution path as Console.")
            yield Static("Provider", classes="destination-section")
            yield Input(
                value=str(values["provider"]),
                id="settings-provider-value",
                placeholder="Provider, e.g. OpenAI or llama_cpp",
            )
            yield Static("Model", classes="destination-section")
            yield Input(
                value=str(values["model"]),
                id="settings-model-value",
                placeholder="Model name",
            )
            yield Static("Streaming default", classes="destination-section")
            yield Input(
                value=str(values["streaming"]).lower(),
                id="settings-streaming-default",
                placeholder="true or false",
            )
            yield Static("Temperature default", classes="destination-section")
            yield Input(
                value=str(values["temperature"]),
                id="settings-temperature-default",
                placeholder="0.0 - 2.0",
            )
            yield Static(
                self._provider_endpoint_summary(str(values["provider"]), app_config),
                id="settings-provider-endpoint",
            )
            yield Static(
                self._provider_key_status(str(values["provider"])),
                id="settings-provider-key-status",
            )
            yield Button(
                "Test Provider",
                id="settings-test-provider",
                tooltip="Run a local, non-network readiness check for this provider configuration.",
            )
            yield Static(self._provider_test_result, id="settings-provider-test-result")

    def _render_console_behavior_card(self, *, compact: bool = False) -> ComposeResult:
        with Vertical(id="settings-console-behavior-card", classes="settings-secondary-card"):
            title = "Console paste collapse" if compact else "Console Behavior"
            yield Static(title, classes="destination-section")
            yield Static(
                "Collapse large pasted chunks over 50 characters.",
                id="settings-console-collapse-large-pastes-label",
            )
            yield Button(
                self._collapse_large_pastes_label(),
                id="settings-console-collapse-large-pastes-toggle",
                tooltip="Toggle compact display for large pasted Console chunks.",
            )
            yield Static(
                "Keeps large paste chunks compact in Console. Disable to keep pasted text literal.",
                id="settings-console-collapse-large-pastes-help",
            )

    def _render_detail_pane(self) -> ComposeResult:
        category = SettingsCategoryId(self.active_category)
        if category is SettingsCategoryId.OVERVIEW:
            yield from self._render_overview_detail()
        elif category is SettingsCategoryId.PROVIDERS_MODELS:
            yield from self._render_provider_detail()
        elif category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            yield Static("Console Behavior", classes="destination-section settings-column-title")
            with Vertical(id="settings-console-behavior-detail", classes="settings-focus-card"):
                yield Static("Collapse large pasted chunks", classes="destination-section")
                yield Static("Large paste chunks over 50 characters display as compact placeholders.")
                yield Static("Normal typing remains literal and does not transform unexpectedly.")
                yield from self._render_console_behavior_card(compact=False)
        elif category is SettingsCategoryId.APPEARANCE:
            yield Static("Appearance", classes="destination-section settings-column-title")
            with Vertical(id="settings-appearance-card", classes="settings-focus-card"):
                yield Static("Theme and display customization live in the Appearance surface.")
                yield Static("Use the inspector action to open the dedicated customization surface.")
        elif category is SettingsCategoryId.STORAGE:
            yield Static("Storage", classes="destination-section settings-column-title")
            with Vertical(id="settings-storage-card", classes="settings-focus-card"):
                yield Static(f"Config path: {self._config_path()}")
                yield Static(f"Config directory status: {self._config_writable_status()}")
                yield Static("Database and media paths remain local unless a server handoff is explicit.")
        elif category is SettingsCategoryId.PRIVACY_SECURITY:
            yield Static("Privacy & Security", classes="destination-section settings-column-title")
            with Vertical(id="settings-privacy-security-card", classes="settings-focus-card"):
                yield Static("Secrets are read from environment/config and are not shown in diagnostics.")
                yield Static("Validation errors redact API key, token, password, and secret assignments.")
                yield Static("Encryption status: use local profile protections where configured.")
        elif category is SettingsCategoryId.DIAGNOSTICS:
            yield Static("Diagnostics", classes="destination-section settings-column-title")
            with Vertical(id="settings-diagnostics-card", classes="settings-focus-card"):
                yield Static("Validate config", classes="destination-section")
                yield Static("Raw TOML validation is available before applying advanced edits.")
                yield Static("Logs and troubleshooting should expose actionable errors without secrets.")
        else:
            yield Static("Advanced Config", classes="destination-section settings-column-title")
            with Vertical(id="settings-advanced-config-card", classes="settings-focus-card"):
                yield Static("Raw TOML", classes="destination-section")
                yield Static("Expert-only raw configuration editing with validation before save.")
                yield Static("Invalid top-level values are blocked; table-shaped TOML is required.")

    def _render_impact_pane(self) -> ComposeResult:
        summary = self._active_summary()
        yield Static("Scope Inspector", classes="destination-section settings-column-title")
        yield Static(f"Selected category: {summary.title}", classes="destination-section")
        yield Static(
            "Unsaved changes" if self._category_has_unsaved_changes(summary.category) else "No unsaved changes",
            id="settings-selected-category-draft-status",
            classes="destination-section",
        )
        if summary.category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            yield Static("Affects Console", classes="destination-section")
            yield Static("Composer paste display, chat-flow defaults, and user input feedback.")
        elif summary.category is SettingsCategoryId.PROVIDERS_MODELS:
            yield Static("Affects Console and provider-backed generation.", classes="destination-section")
            yield Static(self._provider_readiness_label())
        elif summary.category is SettingsCategoryId.APPEARANCE:
            yield Static("Affects visual presentation only.", classes="destination-section")
            yield Button(
                "Open Appearance",
                id="settings-open-appearance",
                tooltip="Open appearance customization settings.",
            )
        else:
            yield Static("Impact and boundaries", classes="destination-section")
            yield Static(summary.description)
        yield Static(
            "MCP and tool-control settings live under MCP, not global Settings.",
            id="settings-boundary-note",
        )
        yield Static("Mutation replay: disabled", id="settings-sync-mutation-disabled")
        yield Static(
            "Writes remain blocked until explicit review, conflict, rollback, and audit gates are implemented.",
            id="settings-sync-write-gates",
        )
        if summary.category is SettingsCategoryId.OVERVIEW:
            yield Button(
                "Open Appearance",
                id="settings-open-appearance",
                tooltip="Open appearance customization settings.",
            )
        yield Button(
            "Save",
            id="settings-save-category",
            tooltip="Save changes for the selected Settings category.",
        )
        yield Button(
            "Revert",
            id="settings-revert-category",
            tooltip="Discard unsaved changes for the selected Settings category.",
        )

    def compose_content(self) -> ComposeResult:
        active_summary = self._active_summary()
        with Vertical(id="settings-shell"):
            yield Static(
                "Settings | Global preferences, appearance, accounts, storage | Local",
                id="settings-title",
                classes="ds-destination-header",
            )
            with DestinationModeStrip(id="settings-category-strip", classes="destination-mode-strip"):
                yield Static(
                    f"Mode: {active_summary.title} | Runtime controls stay in MCP and ACP",
                    id="settings-category-label",
                    classes="destination-section",
                )
            with Horizontal(id="settings-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="settings-category-pane", classes="destination-workbench-pane"):
                    yield Static("Settings Sections", classes="destination-section settings-column-title")
                    yield from self._render_category_buttons()
                yield self._column_divider("settings-category-detail-divider")
                with Vertical(id="settings-detail-pane", classes="destination-workbench-pane"):
                    yield Static("Preference Detail", classes="destination-section settings-column-title")
                    yield from self._render_detail_pane()
                yield self._column_divider("settings-detail-impact-divider")
                with Vertical(id="settings-impact-pane", classes="destination-workbench-pane ds-inspector"):
                    yield from self._render_impact_pane()

    def _category_value_from_button(self, button: Button) -> str | None:
        if not button.id or not button.has_class("settings-category-button"):
            return None
        prefix = "settings-category-"
        if not button.id.startswith(prefix):
            return None
        value = button.id.removeprefix(prefix)
        if value not in {summary.category.value for summary in self._category_summaries()}:
            return None
        return value

    def _focused_category_value(self) -> str | None:
        focused = self.app.focused
        if isinstance(focused, Button):
            return self._category_value_from_button(focused)
        return None

    def _focus_category(self, category_value: str) -> None:
        try:
            self.query_one(f"#settings-category-{category_value}", Button).focus()
        except QueryError:
            logger.debug("Unable to focus Settings category button: %s", category_value)

    def _move_category_focus(self, delta: int) -> None:
        category_values = [summary.category.value for summary in self._category_summaries()]
        current_value = self._focused_category_value() or self.active_category
        try:
            current_index = category_values.index(current_value)
        except ValueError:
            current_index = 0
        next_index = max(0, min(len(category_values) - 1, current_index + delta))
        self._focus_category(category_values[next_index])

    def _select_category(self, category_value: str, *, restore_focus: bool = False) -> None:
        self.active_category = category_value
        if restore_focus:
            self.call_after_refresh(self._focus_category, category_value)

    @on(Button.Pressed, "#settings-open-appearance")
    def open_appearance_settings(self) -> None:
        self.post_message(NavigateToScreen("customize"))

    @on(Button.Pressed, ".settings-category-button")
    def handle_category_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        category_value = self._category_value_from_button(event.button)
        if category_value is not None:
            self._select_category(category_value, restore_focus=event.button.has_focus)

    @on(Button.Pressed, "#settings-console-collapse-large-pastes-toggle")
    def handle_console_collapse_large_pastes_changed(self, event: Button.Pressed) -> None:
        event.stop()
        next_value = not self._collapse_large_pastes_enabled()
        self._stage_console_large_paste_value(next_value)
        event.button.label = self._collapse_large_pastes_label()
        self._update_console_paste_summary()
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Input.Changed, "#settings-provider-value")
    def handle_provider_value_changed(self, event: Input.Changed) -> None:
        self._stage_provider_value("provider", event.value.strip())
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-model-value")
    def handle_model_value_changed(self, event: Input.Changed) -> None:
        self._stage_provider_value("model", event.value.strip())
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-streaming-default")
    def handle_streaming_default_changed(self, event: Input.Changed) -> None:
        self._stage_provider_value("streaming", coerce_bool_setting(event.value, True))
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-temperature-default")
    def handle_temperature_default_changed(self, event: Input.Changed) -> None:
        try:
            value = self._normalise_temperature(event.value)
        except ValueError:
            value = event.value
        self._stage_provider_value("temperature", value)
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Button.Pressed, "#settings-save-category")
    def handle_save_category(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_save_category()

    @on(Button.Pressed, "#settings-revert-category")
    def handle_revert_category(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_revert_category()

    @on(Button.Pressed, "#settings-test-provider")
    def handle_test_provider(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_test_category()

    def action_settings_save_category(self) -> None:
        category = self._active_category_id()
        if category is SettingsCategoryId.PROVIDERS_MODELS:
            try:
                values = self._provider_form_values_from_widgets()
            except ValueError:
                self.app.notify("Temperature must be a number.", severity="error")
                return
            defaults = self._chat_defaults()
            dirty_values = {
                key: value
                for key, value in values.items()
                if defaults.get(key) != value
            }
            if not dirty_values:
                self.app.notify("No Settings changes to save.", severity="information")
                return
            saved = SettingsConfigAdapter().save_values("chat_defaults", values)
            if saved:
                defaults.update(values)
                self._settings_drafts.pop(category, None)
                self._update_draft_status_widgets(category)
                self.app.notify("Provider and model settings saved.", severity="information")
            else:
                self.app.notify("Failed to save provider and model settings.", severity="error")
            return

        draft = self._settings_drafts.get(category)
        if not draft or not draft.is_dirty:
            self.app.notify("No Settings changes to save.", severity="information")
            return

        if category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            dirty_values = {key: draft.values[key] for key in draft.dirty_keys}
            saved = True
            for key, value in dirty_values.items():
                if not save_setting_to_cli_config("console", key, value):
                    saved = False
            if saved:
                self._console_settings().update(dirty_values)
                self._settings_drafts.pop(category, None)
                self._sync_console_behavior_widgets()
                self.app.notify("Console behavior settings saved.", severity="information")
            else:
                self.app.notify("Failed to save Console behavior settings.", severity="error")
            return

        self.app.notify("This Settings category has no save action yet.", severity="warning")

    def action_settings_revert_category(self) -> None:
        category = self._active_category_id()
        self._settings_drafts.pop(category, None)
        if category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            self._sync_console_behavior_widgets()
        elif category is SettingsCategoryId.PROVIDERS_MODELS:
            values = self._provider_setting_values()
            try:
                self.query_one("#settings-provider-value", Input).value = str(values["provider"])
                self.query_one("#settings-model-value", Input).value = str(values["model"])
                self.query_one("#settings-streaming-default", Input).value = str(values["streaming"]).lower()
                self.query_one("#settings-temperature-default", Input).value = str(values["temperature"])
            except QueryError:
                pass
            self._update_draft_status_widgets(category)
        else:
            self._update_draft_status_widgets(category)
        self.app.notify("Settings category changes reverted.", severity="information")

    def action_settings_test_category(self) -> None:
        if self._active_category_id() is SettingsCategoryId.PROVIDERS_MODELS:
            self._provider_test_result = self._run_provider_readiness_test()
            self._update_provider_test_result()
            self.app.notify("Provider test finished.", severity="information")
            return
        self.app.notify("No test action is available for this Settings category yet.", severity="warning")

    def _sync_console_behavior_widgets(self) -> None:
        try:
            self.query_one("#settings-console-collapse-large-pastes-toggle", Button).label = (
                self._collapse_large_pastes_label()
            )
        except QueryError:
            pass
        self._update_console_paste_summary()
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    def on_key(self, event: Key) -> None:
        if event.key == "tab":
            focused = self.app.focused
            if focused is None or getattr(focused, "has_class", lambda *_: False)("nav-button"):
                self._focus_category(SettingsCategoryId.OVERVIEW.value)
                event.stop()
                event.prevent_default()
            return
        if event.key in {"down", "j"} and self._focused_category_value() is not None:
            self._move_category_focus(1)
            event.stop()
            event.prevent_default()
            return
        if event.key in {"up", "k"} and self._focused_category_value() is not None:
            self._move_category_focus(-1)
            event.stop()
            event.prevent_default()
            return
        if event.key == "enter":
            focused = self.app.focused
            if isinstance(focused, Button) and focused.id in {
                "settings-console-collapse-large-pastes-toggle",
                "settings-test-provider",
            }:
                focused.press()
                event.stop()
                event.prevent_default()
                return
            category_value = self._focused_category_value()
            if category_value is not None:
                self._select_category(category_value, restore_focus=True)
                event.stop()
                event.prevent_default()
