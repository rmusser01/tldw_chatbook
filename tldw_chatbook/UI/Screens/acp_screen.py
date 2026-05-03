"""ACP destination shell for agent sessions and runtimes."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


class ACPScreen(BaseAppScreen):
    """Agent Client Protocol agents, sessions, runtimes, diffs, and terminals."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "acp", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="acp-shell"):
            yield Static("ACP", id="acp-title", classes="ds-destination-header")
            yield Static(
                "Agent Client Protocol agents, sessions, runtimes, diffs, and terminals.",
                id="acp-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="acp-sections", classes="ds-panel"):
                yield Static("No ACP runtime is configured yet. Sessions will appear here when available.")
