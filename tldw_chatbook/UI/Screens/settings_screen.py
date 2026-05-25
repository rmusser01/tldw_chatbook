"""Settings destination shell for global app preferences."""

from collections.abc import Mapping
import logging
import os
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.events import Key
from textual.reactive import reactive
from textual.widgets import Button, Input, Rule, Static, TextArea

from ...Chat.provider_readiness import get_provider_readiness, provider_config_key
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
        self._diagnostics_validation_result = "Config validation: not run"
        self._diagnostics_reload_result = "Config reload: not run"
        self._advanced_config_result = "Advanced config validation: not run"

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
            category_status_widget = self.query_one(
                f"#settings-category-{category.value}-status", Static
            )
            category_status_widget.update(f"Status: {category_status}")
            if has_unsaved_changes:
                category_status_widget.add_class("settings-dirty-category")
            else:
                category_status_widget.remove_class("settings-dirty-category")
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

    def _raw_config_text(self) -> str:
        config_path = self._config_path()
        if config_path.exists():
            try:
                return config_path.read_text(encoding="utf-8")
            except OSError as exc:
                return f"# Unable to read {config_path}: {type(exc).__name__}"
        return "# Config file does not exist yet.\n"

    def _known_storage_paths(self) -> tuple[str, ...]:
        paths = [f"Config path: {self._config_path()}"]
        user_data_dir = getattr(self.app_instance, "user_data_dir", None)
        if user_data_dir:
            paths.append(f"User data directory: {user_data_dir}")
        for attr_name, label in (
            ("notifications_db_path", "Notifications DB"),
            ("subscriptions_db_path", "Watchlists DB"),
            ("workspaces_db_path", "Workspaces DB"),
        ):
            value = getattr(self.app_instance, attr_name, None)
            if value:
                paths.append(f"{label}: {value}")
        return tuple(paths)

    def _appearance_theme_summary(self) -> str:
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        if not isinstance(app_config, Mapping):
            return "Theme: default"
        for section_name in ("appearance", "ui", "theme"):
            section = app_config.get(section_name, {})
            if isinstance(section, Mapping):
                theme = section.get("theme") or section.get("name")
                if theme:
                    return f"Theme: {theme} from [{section_name}]"
        return "Theme: default"

    def _set_static_text(self, selector: str, text: str) -> None:
        try:
            self.query_one(selector, Static).update(text)
        except QueryError:
            pass

    def _validate_current_config(self) -> str:
        try:
            SettingsConfigAdapter().load()
        except Exception as exc:
            return f"Config validation: invalid - {redact_secret_text(str(exc))}"
        return "Config validation: valid"

    def _reload_current_config(self) -> str:
        try:
            loaded = SettingsConfigAdapter().load()
        except Exception as exc:
            return f"Config reload: failed - {redact_secret_text(str(exc))}"
        if isinstance(loaded, dict):
            self.app_instance.app_config = loaded
            return "Config reload: loaded"
        return "Config reload: failed - loaded config was not a table"

    def _advanced_editor_text(self) -> str:
        try:
            return self.query_one("#settings-advanced-config-editor", TextArea).text
        except QueryError:
            return ""

    def _validate_advanced_config_text(self, text: str) -> str:
        result = SettingsConfigAdapter().validate_raw_toml(text)
        status = "valid" if result.valid else "invalid"
        return f"Advanced config validation: {status} - {redact_secret_text(result.message)}"

    def _save_advanced_config_text(self, text: str) -> str:
        validation = SettingsConfigAdapter().validate_raw_toml(text)
        if not validation.valid:
            return f"Advanced config save: blocked - {redact_secret_text(validation.message)}"

        config_path = self._config_path()
        tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
        backup_path = config_path.with_suffix(config_path.suffix + ".bak")
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if config_path.exists():
                backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
            tmp_path.write_text(text, encoding="utf-8")
            tmp_path.replace(config_path)
            return f"Advanced config save: saved; backup: {backup_path}"
        except OSError as exc:
            return f"Advanced config save: failed - {redact_secret_text(str(exc))}"

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

    def _provider_loaded_setting_values(self) -> dict[str, object]:
        defaults = self._chat_defaults()
        resolved = resolve_effective_provider_model(self.app_instance)
        return {
            "provider": str(resolved.provider or "").strip(),
            "model": str(resolved.model or "").strip(),
            "streaming": coerce_bool_setting(
                defaults.get("streaming", True),
                True,
            ),
            "temperature": defaults.get("temperature", 0.7),
        }

    def _provider_setting_values(self) -> dict[str, object]:
        loaded = self._provider_loaded_setting_values()
        return {
            key: self._provider_draft_value(key)
            if self._provider_draft_value(key) is not None
            else value
            for key, value in loaded.items()
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
        original = self._provider_loaded_setting_values().get(key)
        draft.set_value(key, original, value)
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _provider_config(self, provider: str) -> Mapping[str, object]:
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        api_settings = app_config.get("api_settings", {}) if isinstance(app_config, Mapping) else {}
        if not isinstance(api_settings, Mapping):
            return {}
        provider_config = api_settings.get(provider_config_key(provider), {})
        return provider_config if isinstance(provider_config, Mapping) else {}

    def _provider_endpoint_summary(self, provider: str) -> str:
        provider_key = provider_config_key(provider)
        provider_config = self._provider_config(provider)
        endpoint = (
            provider_config.get("api_base")
            or provider_config.get("base_url")
            or provider_config.get("api_url")
            or provider_config.get("endpoint")
        )
        if endpoint:
            return f"Endpoint: api_settings.{provider_key}={endpoint}"
        if provider_key in {"llama_cpp", "ollama", "koboldcpp", "oobabooga", "vllm"}:
            return "Endpoint: provider default or not configured"
        return "Endpoint: provider default"

    def _provider_key_status(self, provider: str) -> str:
        readiness = get_provider_readiness(
            provider,
            getattr(self.app_instance, "app_config", {}) or {},
        )
        if readiness.api_key_source:
            return f"API key: {readiness.api_key_source}"
        if readiness.env_var:
            return f"{readiness.env_var}=missing"
        if not readiness.requires_api_key:
            return "API key: not required for this provider"
        return "API key: missing"

    def _run_provider_readiness_test(self) -> str:
        try:
            provider = self.query_one("#settings-provider-value", Input).value.strip()
            model = self.query_one("#settings-model-value", Input).value.strip()
        except QueryError:
            values = self._provider_setting_values()
            provider = str(values.get("provider") or "").strip()
            model = str(values.get("model") or "").strip()

        readiness = get_provider_readiness(
            provider,
            getattr(self.app_instance, "app_config", {}) or {},
        )

        findings: list[str] = ["Provider test"]
        findings.append(readiness.user_message)
        if not model:
            findings.append("model=missing")
        else:
            findings.append(f"model={model}")

        if readiness.api_key_source:
            findings.append(f"api_key_source={readiness.api_key_source}")
        if readiness.env_var:
            raw_value = os.environ.get(readiness.env_var)
            findings.append(f"{readiness.env_var}={raw_value if raw_value else 'missing'}")
        elif not readiness.requires_api_key:
            findings.append("api_key=not required")
        findings.append(self._provider_endpoint_summary(provider))

        status = "ready" if readiness.ready and model else "blocked"
        findings.append(f"status={status}")
        return redact_secret_text(" | ".join(findings))

    def _update_provider_test_result(self) -> None:
        try:
            self.query_one("#settings-provider-test-result", Static).update(self._provider_test_result)
        except QueryError:
            pass

    def _update_provider_dynamic_widgets(self) -> None:
        try:
            provider = self.query_one("#settings-provider-value", Input).value.strip()
        except QueryError:
            provider = str(self._provider_setting_values().get("provider") or "")
        try:
            self.query_one("#settings-provider-readiness", Static).update(
                self._provider_readiness_label()
            )
            self.query_one("#settings-provider-endpoint", Static).update(
                self._provider_endpoint_summary(provider)
            )
            self.query_one("#settings-provider-key-status", Static).update(
                self._provider_key_status(provider)
            )
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
                status = Static(
                    f"Status: {self._category_status(summary)}",
                    id=f"settings-category-{summary.category.value}-status",
                    classes="destination-section settings-status-row",
                )
                if self._category_has_unsaved_changes(summary.category):
                    status.add_class("settings-dirty-category")
                yield status

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
        yield Static("Providers & Models", classes="destination-section settings-column-title")
        with Vertical(id="settings-providers-models-card", classes="settings-focus-card"):
            yield Static("Provider readiness", classes="destination-section")
            yield Static(self._provider_readiness_label(), id="settings-provider-readiness")
            yield Static(f"Source: {resolved.provider_source}", id="settings-provider-source")
            yield Static(f"Provider source: {resolved.provider_source}")
            yield Static(f"Model source: {resolved.model_source}", id="settings-model-source")
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
                self._provider_endpoint_summary(str(values["provider"])),
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
                yield Static(self._appearance_theme_summary())
                yield Static("Theme and display customization live in the Appearance surface.")
                yield Static("Use the inspector action to open the dedicated customization surface.")
                yield Static("Open Appearance from the inspector to change themes, density, and visual polish.")
        elif category is SettingsCategoryId.STORAGE:
            yield Static("Storage", classes="destination-section settings-column-title")
            with Vertical(id="settings-storage-card", classes="settings-focus-card"):
                for path_summary in self._known_storage_paths():
                    yield Static(path_summary)
                yield Static(f"Config directory status: {self._config_writable_status()}")
                yield Static("Database and media paths remain local unless a server handoff is explicit.")
        elif category is SettingsCategoryId.PRIVACY_SECURITY:
            yield Static("Privacy & Security", classes="destination-section settings-column-title")
            with Vertical(id="settings-privacy-security-card", classes="settings-focus-card"):
                yield Static("Secrets are read from environment/config and are not shown in diagnostics.")
                yield Static("Validation errors redact API key, token, password, and secret assignments.")
                yield Static("Encryption: not configured from this first-slice Settings screen.")
                yield Static("Secret redaction: enabled for diagnostics and validation errors.")
        elif category is SettingsCategoryId.DIAGNOSTICS:
            yield Static("Diagnostics", classes="destination-section settings-column-title")
            with Vertical(id="settings-diagnostics-card", classes="settings-focus-card"):
                yield Static("Validate config", classes="destination-section")
                yield Static(f"Config path: {self._config_path()}")
                yield Static("Raw TOML validation is available before applying advanced edits.")
                yield Static("Logs and troubleshooting should expose actionable errors without secrets.")
                with Horizontal(id="settings-diagnostics-actions", classes="settings-action-row"):
                    yield Button(
                        "Validate Config",
                        id="settings-validate-config",
                        tooltip="Validate the current Settings config file.",
                    )
                    yield Button(
                        "Reload Config",
                        id="settings-reload-config",
                        tooltip="Reload the current Settings config into the running app.",
                    )
                yield Static(
                    self._diagnostics_validation_result,
                    id="settings-diagnostics-validation-result",
                    classes="settings-status-row",
                )
                yield Static(
                    self._diagnostics_reload_result,
                    id="settings-diagnostics-reload-result",
                    classes="settings-status-row",
                )
        else:
            yield Static("Advanced Config", classes="destination-section settings-column-title")
            with Vertical(id="settings-advanced-config-card", classes="settings-focus-card"):
                yield Static("Raw TOML", classes="destination-section")
                yield Static("Expert-only raw configuration editing with validation before save.")
                yield Static("Raw TOML bypasses guided validation and should be used only for expert edits.")
                yield Static("Invalid top-level values are blocked; table-shaped TOML is required.")
                yield TextArea(
                    self._raw_config_text(),
                    id="settings-advanced-config-editor",
                )
                with Horizontal(id="settings-advanced-config-actions", classes="settings-action-row"):
                    yield Button(
                        "Validate Raw TOML",
                        id="settings-advanced-validate-config",
                        tooltip="Validate raw TOML before writing it to disk.",
                    )
                    yield Button(
                        "Save Raw TOML",
                        id="settings-advanced-save-config",
                        tooltip="Atomically save raw TOML after validation.",
                    )
                yield Static(
                    self._advanced_config_result,
                    id="settings-advanced-config-result",
                    classes="settings-status-row",
                )

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
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-model-value")
    def handle_model_value_changed(self, event: Input.Changed) -> None:
        self._stage_provider_value("model", event.value.strip())
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-streaming-default")
    def handle_streaming_default_changed(self, event: Input.Changed) -> None:
        self._stage_provider_value("streaming", coerce_bool_setting(event.value, True))
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-temperature-default")
    def handle_temperature_default_changed(self, event: Input.Changed) -> None:
        try:
            value = self._normalise_temperature(event.value)
        except ValueError:
            value = event.value
        self._stage_provider_value("temperature", value)
        self._update_provider_dynamic_widgets()
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

    @on(Button.Pressed, "#settings-validate-config")
    def handle_validate_config(self, event: Button.Pressed) -> None:
        event.stop()
        self._diagnostics_validation_result = self._validate_current_config()
        self._set_static_text(
            "#settings-diagnostics-validation-result",
            self._diagnostics_validation_result,
        )

    @on(Button.Pressed, "#settings-reload-config")
    def handle_reload_config(self, event: Button.Pressed) -> None:
        event.stop()
        self._diagnostics_reload_result = self._reload_current_config()
        self._set_static_text(
            "#settings-diagnostics-reload-result",
            self._diagnostics_reload_result,
        )

    @on(Button.Pressed, "#settings-advanced-validate-config")
    def handle_advanced_validate_config(self, event: Button.Pressed) -> None:
        event.stop()
        self._advanced_config_result = self._validate_advanced_config_text(
            self._advanced_editor_text()
        )
        self._set_static_text("#settings-advanced-config-result", self._advanced_config_result)

    @on(Button.Pressed, "#settings-advanced-save-config")
    def handle_advanced_save_config(self, event: Button.Pressed) -> None:
        event.stop()
        self._advanced_config_result = self._save_advanced_config_text(
            self._advanced_editor_text()
        )
        self._set_static_text("#settings-advanced-config-result", self._advanced_config_result)

    def action_settings_save_category(self) -> None:
        category = self._active_category_id()
        if category is SettingsCategoryId.PROVIDERS_MODELS:
            try:
                values = self._provider_form_values_from_widgets()
            except ValueError:
                self.app.notify("Temperature must be a number.", severity="error")
                return
            loaded_values = self._provider_loaded_setting_values()
            dirty_values = {
                key: value
                for key, value in values.items()
                if loaded_values.get(key) != value
            }
            if not dirty_values:
                self.app.notify("No Settings changes to save.", severity="information")
                return
            saved = SettingsConfigAdapter().save_values("chat_defaults", dirty_values)
            if saved:
                defaults = self._chat_defaults()
                defaults.update(dirty_values)
                self._settings_drafts.pop(category, None)
                self._update_provider_dynamic_widgets()
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
                "settings-validate-config",
                "settings-reload-config",
                "settings-advanced-validate-config",
                "settings-advanced-save-config",
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
