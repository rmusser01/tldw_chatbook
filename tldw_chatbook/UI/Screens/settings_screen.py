"""Settings destination shell for global app preferences."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Rule, Static

from ...Sync_Interop.sync_promotion_state import SyncPromotionState, build_sync_promotion_state
from ...Sync_Interop.sync_readiness import DEFAULT_SYNC_ELIGIBILITY_REGISTRY, build_sync_readiness_report
from ...Widgets.destination_workbench import DestinationModeStrip
from ...config import coerce_bool_setting, save_setting_to_cli_config
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen


class SettingsScreen(BaseAppScreen):
    """Global preferences, appearance, accounts, storage, and app behavior."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "settings", **kwargs)

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
            except Exception:
                pass
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

    def compose_content(self) -> ComposeResult:
        sync_safety_states = self._sync_safety_states()
        with Vertical(id="settings-shell"):
            yield Static(
                "Settings | Global preferences, appearance, accounts, storage | Local",
                id="settings-title",
                classes="ds-destination-header",
            )
            with DestinationModeStrip(id="settings-category-strip", classes="destination-mode-strip"):
                yield Static(
                    "Mode: Global / Console behavior / Appearance | Runtime controls stay in MCP and ACP",
                    id="settings-category-label",
                    classes="destination-section",
                )
            with Horizontal(id="settings-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="settings-category-pane", classes="destination-workbench-pane"):
                    yield Static("Settings Sections", classes="destination-section settings-column-title")
                    yield Static("Global preferences", classes="destination-section")
                    yield Static("Appearance", classes="destination-section")
                    yield Static("Accounts/Auth", classes="destination-section")
                    yield Static("Storage", classes="destination-section")
                    yield Static("App-level behavior", classes="destination-section")
                    yield Static("Console behavior", classes="destination-section")
                    yield Static("Sync Safety", classes="destination-section settings-active-section")
                yield self._column_divider("settings-category-detail-divider")
                with Vertical(id="settings-detail-pane", classes="destination-workbench-pane"):
                    yield Static("Preference Detail", classes="destination-section settings-column-title")
                    with Vertical(id="settings-sync-safety-card", classes="settings-focus-card"):
                        yield Static(
                            "Write Sync Safety",
                            id="settings-sync-safety-title",
                            classes="destination-section",
                        )
                        yield Static(
                            "This screen is a visibility contract, not an enablement panel.",
                            id="settings-sync-safety-framing",
                        )
                        yield Static(
                            "Write sync promotion is display-only: dry-run, review, conflict, and rollback gates are visible before any replay path exists.",
                            id="settings-sync-safety-summary",
                        )
                        for state in sync_safety_states:
                            yield Static(
                                f"{state.surface_label}: {state.sync_label}",
                                id=f"settings-sync-safety-{state.domain}",
                                classes="settings-sync-safety-row",
                            )
                        yield Static(
                            "No write-sync controls are available here.",
                            id="settings-sync-safety-no-controls",
                        )
                    with Vertical(id="settings-console-behavior-card", classes="settings-secondary-card"):
                        yield Static("Console behavior", classes="destination-section")
                        yield Static(
                            "Large paste display: collapse paste chunks over 50 characters.",
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
                    yield Static("Other global settings", classes="destination-section")
                    yield Static("Appearance controls are available in the customization surface.")
                    yield Static("Accounts/Auth and storage defaults remain global app settings.")
                    yield Static("Runtime-specific MCP and ACP controls stay with their destinations.")
                yield self._column_divider("settings-detail-impact-divider")
                with Vertical(id="settings-impact-pane", classes="destination-workbench-pane ds-inspector"):
                    yield Static("Scope Inspector", classes="destination-section settings-column-title")
                    yield Static("Impact and boundaries", classes="destination-section")
                    yield Static(
                        "MCP and tool-control settings live under MCP, not global Settings.",
                        id="settings-boundary-note",
                    )
                    yield Static("Mutation replay: disabled", id="settings-sync-mutation-disabled")
                    yield Static(
                        "Writes remain blocked until explicit review, conflict, rollback, and audit gates are implemented.",
                        id="settings-sync-write-gates",
                    )
                    yield Button(
                        "Open Appearance",
                        id="settings-open-appearance",
                        tooltip="Open appearance customization settings.",
                    )

    @on(Button.Pressed, "#settings-open-appearance")
    def open_appearance_settings(self) -> None:
        self.post_message(NavigateToScreen("customize"))

    @on(Button.Pressed, "#settings-console-collapse-large-pastes-toggle")
    def handle_console_collapse_large_pastes_changed(self, event: Button.Pressed) -> None:
        event.stop()
        next_value = not self._collapse_large_pastes_enabled()
        self._console_settings()["collapse_large_pastes"] = next_value
        event.button.label = self._collapse_large_pastes_label()
        if save_setting_to_cli_config("console", "collapse_large_pastes", next_value):
            self.app.notify("Console paste display setting saved.", severity="information")
        else:
            self.app.notify("Failed to save Console paste display setting.", severity="error")
