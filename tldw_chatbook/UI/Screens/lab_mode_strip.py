"""Shared mode strip for the Lab destination's screens.

The Lab destination seats three screens -- Models (``llm``), Speech
(``stts``), and Evals (``evals``) -- that previously had no way to reach
each other: nothing posted ``NavigateToScreen("evals")`` and the Models
screen had no Speech/Evals link. Each Lab screen mounts this strip directly
under its DestinationHeader, mirroring the Personas mode-chip pattern
(``DestinationModeStrip`` + chip buttons), except Lab modes are separate
screens: a chip press posts ``NavigateToScreen`` for that mode's route.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.widgets import Button, Static

from ...Widgets.destination_workbench import DestinationModeStrip
from ..Navigation.main_navigation import NavigateToScreen

#: (mode id, chip label, shell route, chip tooltip) in strip order. The
#: routes are the Lab destination's own screens; tooltips mirror each
#: screen's DestinationHeader subtitle.
LAB_MODE_CHIPS: tuple[tuple[str, str, str, str], ...] = (
    ("models", "Models", "llm", "Manage providers, models, and endpoints."),
    ("speech", "Speech", "stts", "Speech-to-text and text-to-speech tools."),
    ("evals", "Evals", "evals", "Run and review evaluation jobs."),
)

LAB_MODE_CHIP_IDS: tuple[str, ...] = tuple(
    f"lab-mode-{mode_id}" for mode_id, *_ in LAB_MODE_CHIPS
)


class LabModeStrip(DestinationModeStrip):
    """Models | Speech | Evals chip strip shared by the Lab screens.

    The chip for the screen that owns the strip carries ``is-active`` and
    is inert; the other chips navigate to their screen's route.
    """

    # Height/border guard for harness apps that do not load the app CSS
    # bundle; mirrors the Personas screen's #personas-mode-strip rules.
    DEFAULT_CSS = """
    LabModeStrip {
        height: 1;
        min-height: 1;
        padding: 0 1;
        overflow: hidden;
    }

    LabModeStrip .lab-mode-label {
        width: 8;
        min-width: 8;
        height: 1;
        min-height: 1;
    }

    LabModeStrip Button.lab-mode-chip {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    LabModeStrip .lab-mode-chip.is-active {
        border: none;
        text-style: bold underline;
    }
    """

    def __init__(self, active_route: str, **kwargs: Any) -> None:
        """Create a Lab mode strip for the screen whose route is ``active_route``.

        Args:
            active_route: The shell route id of the screen that owns this strip.
                The matching mode chip is marked ``is-active`` and is inert.
            kwargs: Forwarded to ``DestinationModeStrip``.
        """
        super().__init__(**kwargs)
        self.active_route = active_route

    def compose(self) -> ComposeResult:
        """Build the ``Modes:`` label and the Lab mode chip buttons."""
        yield Static("Modes:", classes="lab-mode-label destination-section")
        for mode_id, label, route, tooltip in LAB_MODE_CHIPS:
            classes = "lab-mode-chip"
            if route == self.active_route:
                classes = f"{classes} is-active"
            yield Button(
                label,
                id=f"lab-mode-{mode_id}",
                classes=classes,
                tooltip=tooltip,
            )

    @on(Button.Pressed, ".lab-mode-chip")
    def _handle_mode_chip(self, event: Button.Pressed) -> None:
        event.stop()
        for mode_id, _label, route, _tooltip in LAB_MODE_CHIPS:
            if event.button.id == f"lab-mode-{mode_id}":
                if route != self.active_route:
                    self.post_message(NavigateToScreen(route))
                return
