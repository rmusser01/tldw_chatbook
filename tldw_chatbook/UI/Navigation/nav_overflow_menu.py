"""Overflow destination menu for the main navigation bar (NV-01, TASK-2154.21).

The nav strip's static "More: Ctrl+P" hint used to be the only overflow
affordance: at widths where the destination buttons clip, the clipped
destinations (Lab/Logs/Settings at ~140 cols, more below that) had no
reachable home. This modal is that home -- every destination listed with the
same digit-prefixed label the strip shows (so the menu doubles as shortcut
documentation), a press navigating exactly like the strip button does.
"""

from __future__ import annotations

from typing import Any

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from .main_navigation import NavigateToScreen, nav_button_label
from .shell_destinations import SHELL_DESTINATION_ORDER


class NavOverflowMenu(ModalScreen[None]):
    """List every shell destination; a press posts ``NavigateToScreen``."""

    DEFAULT_CSS = """
    NavOverflowMenu {
        align: center middle;
    }

    #nav-overflow-menu {
        width: 44;
        height: auto;
        max-height: 80%;
        border: tall $primary;
        background: $surface;
        padding: 1 2;
    }

    #nav-overflow-menu-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .nav-overflow-destination {
        width: 100%;
        min-width: 0;
        height: 1;
        min-height: 1;
        border: none;
        background: $surface;
        content-align: left middle;
    }

    .nav-overflow-destination:hover,
    .nav-overflow-destination:focus {
        background: $primary-darken-1;
        text-style: bold;
    }
    """

    def __init__(self, active_destination_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._active_destination_id = active_destination_id

    def compose(self) -> ComposeResult:
        with Vertical(id="nav-overflow-menu"):
            yield Static("All destinations", id="nav-overflow-menu-title")
            for index, destination in enumerate(SHELL_DESTINATION_ORDER):
                label = nav_button_label(
                    index,
                    destination.accessible_label,
                    destination_id=destination.destination_id,
                )
                if destination.destination_id == self._active_destination_id:
                    label = f"{label} (current)"
                button = Button(
                    label,
                    id=f"nav-overflow-{destination.destination_id}",
                    classes="nav-overflow-destination",
                    tooltip=destination.tooltip,
                    compact=True,
                )
                button._overflow_target_route = destination.primary_route
                yield button

    @on(Button.Pressed, ".nav-overflow-destination")
    def handle_destination(self, event: Button.Pressed) -> None:
        """Navigate exactly as the strip's own button would."""
        event.stop()
        route = getattr(event.button, "_overflow_target_route", None)
        if not route:
            return
        self.post_message(NavigateToScreen(route))
        self.dismiss()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss()


__all__ = ["NavOverflowMenu"]
