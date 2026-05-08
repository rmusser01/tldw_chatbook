"""Settings destination shell for global app preferences."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from ...Widgets.destination_workbench import DestinationModeStrip
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen


class SettingsScreen(BaseAppScreen):
    """Global preferences, appearance, accounts, storage, and app behavior."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "settings", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="settings-shell"):
            yield Static("Settings", id="settings-title", classes="ds-destination-header")
            yield Static(
                "Settings owns global preferences, appearance, accounts/auth, storage, and app-level behavior.",
                id="settings-purpose",
                classes="destination-purpose",
            )
            with DestinationModeStrip(id="settings-category-strip", classes="destination-filter-strip"):
                yield Static(
                    "Categories: Global | Appearance | Accounts | Storage | Behavior",
                    id="settings-category-label",
                    classes="destination-section",
                )
            with Horizontal(id="settings-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="settings-category-pane", classes="destination-workbench-pane"):
                    yield Static("Settings Categories", classes="destination-section")
                    yield Static("Global preferences", classes="destination-section")
                    yield Static("Appearance", classes="destination-section")
                    yield Static("Accounts/Auth", classes="destination-section")
                    yield Static("Storage", classes="destination-section")
                    yield Static("App-level behavior", classes="destination-section")
                with Vertical(id="settings-detail-pane", classes="destination-workbench-pane"):
                    yield Static("Global Preferences", classes="destination-section")
                    yield Static("Appearance controls are available in the customization surface.")
                    yield Static("Accounts/Auth and storage defaults remain global app settings.")
                    yield Static("Runtime-specific MCP and ACP controls stay with their destinations.")
                with Vertical(id="settings-impact-pane", classes="destination-workbench-pane ds-inspector"):
                    yield Static("Impact And Boundaries", classes="destination-section")
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
