"""Workflows destination shell for reusable agent procedures."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


class WorkflowsScreen(BaseAppScreen):
    """Reusable procedures, recipes, dry-runs, and outputs."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "workflows", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="workflows-shell"):
            yield Static("Workflows", id="workflows-title", classes="ds-destination-header")
            yield Static(
                "Reusable procedures, recipes, dry-runs, and workflow outputs.",
                id="workflows-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="workflows-sections", classes="ds-panel"):
                yield Static("Recipes | Dry-runs | Launch in Console | Outputs")
