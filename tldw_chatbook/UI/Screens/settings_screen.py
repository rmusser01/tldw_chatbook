"""Settings destination shell for global app preferences."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Rule, Static

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

    @staticmethod
    def _column_divider(identifier: str) -> Rule:
        divider = Rule(orientation="vertical", id=identifier)
        divider.add_class("destination-pane-divider")
        return divider

    def compose_content(self) -> ComposeResult:
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
                    yield Static("Column 1: Settings Sections", classes="destination-section settings-column-title")
                    yield Static("Global preferences", classes="destination-section")
                    yield Static("Appearance", classes="destination-section")
                    yield Static("Accounts/Auth", classes="destination-section")
                    yield Static("Storage", classes="destination-section")
                    yield Static("App-level behavior", classes="destination-section")
                    yield Static("Console behavior", classes="destination-section")
                yield self._column_divider("settings-category-detail-divider")
                with Vertical(id="settings-detail-pane", classes="destination-workbench-pane"):
                    yield Static("Column 2: Preference Detail", classes="destination-section settings-column-title")
                    yield Static("Global Preferences", classes="destination-section")
                    yield Static("Appearance controls are available in the customization surface.")
                    yield Static("Accounts/Auth and storage defaults remain global app settings.")
                    yield Static("Runtime-specific MCP and ACP controls stay with their destinations.")
                    yield Static("Console Behavior", classes="destination-section")
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
                yield self._column_divider("settings-detail-impact-divider")
                with Vertical(id="settings-impact-pane", classes="destination-workbench-pane ds-inspector"):
                    yield Static("Column 3: Scope Inspector", classes="destination-section settings-column-title")
                    yield Static("Impact and boundaries", classes="destination-section")
                    yield Static(
                        "MCP and tool-control settings live under MCP, not global Settings.",
                        id="settings-boundary-note",
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
