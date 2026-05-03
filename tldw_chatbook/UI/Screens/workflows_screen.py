"""Workflows destination shell for reusable agent procedures."""

from textual.app import ComposeResult
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
                yield Static("Console launch unavailable", classes="destination-section")
                yield Static("No workflow service is wired yet.", id="workflows-empty-state")
                yield Static(
                    "Console launch is unavailable until workflow execution payloads are wired.",
                    id="workflows-console-unavailable",
                )
                yield Button(
                    "Console launch unavailable",
                    id="workflows-launch-in-console",
                    disabled=True,
                    tooltip="Unavailable until Workflows can pass actionable execution context to Console.",
                )
