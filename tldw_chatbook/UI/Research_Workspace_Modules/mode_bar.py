"""Shared Workspace/Runs mode navigation for the Research destination."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.widgets import Button, Static

from ...Widgets.destination_workbench import DestinationModeStrip
from ..Navigation.main_navigation import NavigateToScreen


RESEARCH_MODE_CHIPS: tuple[tuple[str, str, str, str], ...] = (
    (
        "workspace",
        "Workspace",
        "research_workspace",
        "Open the Sources, Grounded Chat, and Studio workspace.",
    ),
    ("runs", "Runs", "research", "Open durable Research Runs."),
)


class ResearchModeStrip(DestinationModeStrip):
    """Compact navigation between the two real Research screens."""

    def __init__(self, active_route: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.active_route = active_route

    def compose(self) -> ComposeResult:
        yield Static("Research:", classes="research-mode-label destination-section")
        for mode_id, label, route, tooltip in RESEARCH_MODE_CHIPS:
            classes = "research-mode-chip"
            if route == self.active_route:
                classes += " is-active"
            yield Button(
                label,
                id=f"research-mode-{mode_id}",
                classes=classes,
                tooltip=tooltip,
                compact=True,
            )

    @on(Button.Pressed, ".research-mode-chip")
    def navigate_mode(self, event: Button.Pressed) -> None:
        """Post navigation only when the pressed chip owns another route."""
        event.stop()
        for mode_id, _label, route, _tooltip in RESEARCH_MODE_CHIPS:
            if event.button.id == f"research-mode-{mode_id}":
                if route != self.active_route:
                    self.post_message(NavigateToScreen(route))
                return
