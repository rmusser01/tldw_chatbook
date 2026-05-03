"""Personas destination shell for behavior profiles and prompt context."""

from textual.app import ComposeResult
from textual import on
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
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
                yield Button(
                    "Open Personas",
                    id="personas-open-profiles",
                    tooltip="Open character, prompt, dictionary, and lore management.",
                )
                yield Static(
                    "Characters, prompts, dictionaries, and lore stay here; Library owns saved conversation browsing.",
                    id="personas-boundary",
                    classes="destination-purpose",
                )
                yield Button(
                    "Attach to Console",
                    id="personas-attach-to-console",
                    tooltip="Stage persona context in Console.",
                )

    @on(Button.Pressed, "#personas-open-profiles")
    def open_profiles(self) -> None:
        self.post_message(NavigateToScreen("ccp"))

    @on(Button.Pressed, "#personas-attach-to-console")
    def attach_to_console(self) -> None:
        self.app_instance.open_chat_with_handoff(
            ChatHandoffPayload(
                source="personas",
                item_type="persona-context",
                title="Persona context",
                body="Stage characters, prompts, dictionaries, lore, or behavior profiles for Console.",
            )
        )
