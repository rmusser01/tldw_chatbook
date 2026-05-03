"""Schedules destination shell for run timing and recovery."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ..Navigation.base_app_screen import BaseAppScreen


class SchedulesScreen(BaseAppScreen):
    """When jobs, watchlists, and workflows run."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "schedules", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="schedules-shell"):
            yield Static("Schedules", id="schedules-title", classes="ds-destination-header")
            yield Static(
                "Schedules own when things run.",
                id="schedules-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="schedules-sections", classes="ds-panel"):
                yield Static("Next Run", classes="destination-section")
                yield Static("No scheduler data is available yet.", id="schedules-empty-state")
                yield Static("Paused", classes="destination-section")
                yield Static("Failed", classes="destination-section")
                yield Static("Retry", classes="destination-section")
                yield Static("Console recovery unavailable", classes="destination-section")
                yield Static(
                    "Console recovery is unavailable until schedule run payloads are wired.",
                    id="schedules-console-unavailable",
                )
                yield Button(
                    "Console recovery unavailable",
                    id="schedules-follow-in-console",
                    disabled=True,
                    tooltip="Unavailable until Schedules can pass actionable run context to Console.",
                )
