"""Persona card display widget for the CCP screen."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static, TextArea

if TYPE_CHECKING:
    from ...UI.Screens.ccp_screen import CCPScreen

logger = logger.bind(module="CCPPersonaCardWidget")


class PersonaCardMessage(Message):
    """Base persona card message."""


class EditPersonaRequested(PersonaCardMessage):
    """User requested persona editing."""

    def __init__(self, persona_id: str) -> None:
        super().__init__()
        self.persona_id = persona_id


class StartPersonaChatRequested(PersonaCardMessage):
    """User requested to start a chat with the persona."""

    def __init__(self, persona_id: str) -> None:
        super().__init__()
        self.persona_id = persona_id


class CCPPersonaCardWidget(Container):
    """Read-only persona profile view."""

    DEFAULT_CSS = """
    CCPPersonaCardWidget.hidden {
        display: none !important;
    }
    """

    def __init__(self, parent_screen: Optional["CCPScreen"] = None, **kwargs: Any) -> None:
        super().__init__(id="ccp-persona-card-view", **kwargs)
        self.parent_screen = parent_screen
        self.persona_data: Dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Static("Persona Profile", classes="pane-title")
        with VerticalScroll():
            yield Static("", id="ccp-persona-name-display")
            yield Static("", id="ccp-persona-mode-display")
            yield TextArea("", id="ccp-persona-system-prompt-display", read_only=True)
            yield Button("Start Chat", id="ccp-persona-start-chat-button")
            yield Button("Edit Persona", id="ccp-persona-edit-button")

    def load_persona(self, persona_data: Dict[str, Any]) -> None:
        """Populate widget with persona profile data."""
        self.persona_data = dict(persona_data)
        self.query_one("#ccp-persona-name-display", Static).update(persona_data.get("name", ""))
        mode_label = str(persona_data.get("mode", "session_scoped") or "session_scoped").replace("_", " ")
        self.query_one("#ccp-persona-mode-display", Static).update(f"Mode: {mode_label}")
        system_prompt = persona_data.get("system_prompt") or persona_data.get("description") or ""
        self.query_one("#ccp-persona-system-prompt-display", TextArea).text = str(system_prompt)

    async def _emit_message_to_app(self, message: Message, handler_name: str) -> None:
        """Post the message through Textual and mirror it to app-level test callbacks."""
        self.post_message(message)
        handler = getattr(self.app, handler_name, None)
        if callable(handler):
            result = handler(message)
            if inspect.isawaitable(result):
                await result

    @on(Button.Pressed, "#ccp-persona-edit-button")
    async def handle_edit(self, event: Button.Pressed) -> None:
        """Emit an edit request for the loaded persona."""
        event.stop()
        persona_id = self.persona_data.get("id")
        if persona_id:
            await self._emit_message_to_app(
                EditPersonaRequested(str(persona_id)),
                "on_edit_persona_requested",
            )

    @on(Button.Pressed, "#ccp-persona-start-chat-button")
    async def handle_start_chat(self, event: Button.Pressed) -> None:
        """Emit a start-chat request for the loaded persona."""
        event.stop()
        persona_id = self.persona_data.get("id")
        if persona_id:
            await self._emit_message_to_app(
                StartPersonaChatRequested(str(persona_id)),
                "on_start_persona_chat_requested",
            )
