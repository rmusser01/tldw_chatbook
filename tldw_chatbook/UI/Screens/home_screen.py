"""Home dashboard screen for the master shell."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


class HomeScreen(BaseAppScreen):
    """Dashboard, notifications, readiness, and next-best action surface."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "home", **kwargs)

    def compose_content(self) -> ComposeResult:
        """Compose the lightweight Home route anchor."""
        with Vertical(id="home-dashboard"):
            yield Static("Home", id="home-title", classes="ds-destination-header")
            yield Static(
                "Dashboard, notifications, status, active work, and next actions.",
                id="home-purpose",
                classes="destination-purpose",
            )
