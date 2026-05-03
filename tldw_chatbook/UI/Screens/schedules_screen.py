"""Schedules destination shell for run timing and recovery."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


class SchedulesScreen(BaseAppScreen):
    """When jobs, watchlists, and workflows run."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "schedules", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="schedules-shell"):
            yield Static("Schedules", id="schedules-title", classes="ds-destination-header")
            yield Static(
                "Timing, triggers, pauses, retries, and recovery for active work.",
                id="schedules-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="schedules-sections", classes="ds-panel"):
                yield Static("Runs | Triggers | Paused work | Recovery")
