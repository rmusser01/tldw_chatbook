"""Personas destination shell for behavior profiles and prompt context."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


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
                yield Static("Characters | Personas | Prompts | Dictionaries | Lore")
