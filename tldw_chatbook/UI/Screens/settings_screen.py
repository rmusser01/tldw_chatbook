"""Settings destination shell for global app preferences."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ..Navigation.base_app_screen import BaseAppScreen


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
            with Vertical(id="settings-sections", classes="ds-panel"):
                yield Static("Global preferences", classes="destination-section")
                yield Static("Appearance", classes="destination-section")
                yield Static("Accounts/Auth", classes="destination-section")
                yield Static("Storage", classes="destination-section")
                yield Static("App-level behavior", classes="destination-section")
                yield Static("MCP and tool-control settings live under MCP, not global Settings.")
                yield Button("Open Appearance", id="settings-open-appearance")
