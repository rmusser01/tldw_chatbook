"""Personas destination shell for behavior profiles and prompt context."""

from textual.app import ComposeResult
from textual import on
from textual.containers import Vertical
from textual.widgets import Button, Static

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen


class PersonasScreen(BaseAppScreen):
    """Characters, personas, prompts, dictionaries, and behavior profiles."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "personas", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="personas-shell"):
            yield Static("Personas", id="personas-title", classes="ds-destination-header")
            yield Static(
                "Characters, personas, prompts, dictionaries, lore, and behavior profiles.",
                id="personas-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="personas-sections", classes="ds-panel"):
                yield Button("Open Personas", id="personas-open-profiles")
                yield Static(
                    "Characters, prompts, dictionaries, and lore stay here; Library owns saved conversation browsing.",
                    id="personas-boundary",
                    classes="destination-purpose",
                )

    @on(Button.Pressed, "#personas-open-profiles")
    def open_profiles(self) -> None:
        self.post_message(NavigateToScreen("ccp"))
