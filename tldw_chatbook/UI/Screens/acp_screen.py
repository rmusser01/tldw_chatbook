"""ACP destination shell for agent sessions and runtimes."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

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
                yield Static("Installed agents", classes="destination-section")
                yield Static("Sessions", classes="destination-section")
                yield Static("Resume", classes="destination-section")
                yield Static("Diffs", classes="destination-section")
                yield Static("Terminal/Shell", classes="destination-section")
                yield Static(
                    "ACP runtime is not configured yet. Install or configure an ACP-compatible agent before launch.",
                    id="acp-empty-state",
                )
                yield Button("Launch ACP Agent", id="acp-launch-agent", disabled=True)
