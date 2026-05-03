"""Settings destination shell for global app preferences."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


class SettingsScreen(BaseAppScreen):
    """Global preferences, appearance, accounts, storage, and app behavior."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "settings", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="settings-shell"):
            yield Static("Settings", id="settings-title", classes="ds-destination-header")
            yield Static(
                "Global app preferences, appearance, accounts, storage, and behavior.",
                id="settings-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="settings-sections", classes="ds-panel"):
                yield Static("Preferences | Appearance | Accounts | Storage | Behavior")
