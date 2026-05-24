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
from textual.widgets import Button, Rule, Static

from ...Sync_Interop.sync_promotion_state import SyncPromotionState, build_sync_promotion_state
from ...Sync_Interop.sync_readiness import DEFAULT_SYNC_ELIGIBILITY_REGISTRY, build_sync_readiness_report
from ...Widgets.destination_workbench import DestinationModeStrip
from ...config import DEFAULT_CONFIG_PATH, coerce_bool_setting, save_setting_to_cli_config
from ..Navigation.base_app_screen import BaseAppScreen
from .provider_model_resolution import resolve_effective_provider_model
from .settings_config_models import SettingsCategoryId, SettingsCategorySummary
from ..Navigation.main_navigation import NavigateToScreen


logger = logging.getLogger(__name__)


class SettingsScreen(BaseAppScreen):
    """Global preferences, appearance, accounts, storage, and app behavior."""

    active_category = reactive(SettingsCategoryId.OVERVIEW.value, recompose=True)

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "settings", **kwargs)

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

    def _collapse_large_pastes_enabled(self) -> bool:
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
        resolved = resolve_effective_provider_model(self.app_instance)
        provider = str(resolved.provider or "not selected").strip()
        model = str(resolved.model or "not selected").strip()
        if provider and provider != "not selected":
            return f"Provider readiness: {provider} / {model}"
        return "Provider readiness: needs provider and model"

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
                yield Static(f"Status: {summary.status}", classes="destination-section")

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
        yield from self._render_console_behavior_card(compact=True)

    def _render_provider_detail(self) -> ComposeResult:
        resolved = resolve_effective_provider_model(self.app_instance)
        yield Static("Providers & Models", classes="destination-section settings-column-title")
        with Vertical(id="settings-providers-models-card", classes="settings-focus-card"):
            yield Static("Provider readiness", classes="destination-section")
            yield Static(self._provider_readiness_label())
            yield Static(f"Provider source: {resolved.provider_source}")
            yield Static(f"Model source: {resolved.model_source}")
            yield Static("Changes here will share the same provider/model resolution path as Console.")

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
        self._console_settings()["collapse_large_pastes"] = next_value
        event.button.label = self._collapse_large_pastes_label()
        self._update_console_paste_summary()
        if save_setting_to_cli_config("console", "collapse_large_pastes", next_value):
            self.app.notify("Console paste display setting saved.", severity="information")
        else:
            self.app.notify("Failed to save Console paste display setting.", severity="error")

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
            if isinstance(focused, Button) and focused.id == "settings-console-collapse-large-pastes-toggle":
                focused.press()
                event.stop()
                event.prevent_default()
                return
            category_value = self._focused_category_value()
            if category_value is not None:
                self._select_category(category_value, restore_focus=True)
                event.stop()
                event.prevent_default()
