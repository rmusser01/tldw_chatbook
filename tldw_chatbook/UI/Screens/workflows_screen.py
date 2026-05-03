"""Workflows destination shell for reusable agent procedures."""

from textual.app import ComposeResult
from textual import on
from textual.containers import Vertical
from textual.widgets import Button, Static

from ..Navigation.base_app_screen import BaseAppScreen


class WorkflowsScreen(BaseAppScreen):
    """Reusable procedures, recipes, dry-runs, and outputs."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "workflows", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="workflows-shell"):
            yield Static("Workflows", id="workflows-title", classes="ds-destination-header")
            yield Static(
                "Workflows own what procedure runs.",
                id="workflows-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="workflows-sections", classes="ds-panel"):
                yield Static("Recipes", classes="destination-section")
                yield Static("Inputs", classes="destination-section")
                yield Static("Steps", classes="destination-section")
                yield Static("Dry Run", classes="destination-section")
                yield Static("Approvals", classes="destination-section")
                yield Static("Outputs", classes="destination-section")
                yield Static("Launch in Console", classes="destination-section")
                yield Static("No workflow service is wired yet.", id="workflows-empty-state")
                yield Button("Launch in Console", id="workflows-launch-in-console")

    @on(Button.Pressed, "#workflows-launch-in-console")
    def launch_in_console(self) -> None:
        self.app_instance.open_console_for_live_work(
            source="workflows",
            title="Workflows",
        )
