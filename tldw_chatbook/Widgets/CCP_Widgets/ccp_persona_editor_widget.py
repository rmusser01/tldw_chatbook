"""Persona editor widget for the CCP screen."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Select, TextArea

if TYPE_CHECKING:
    from ...UI.Screens.ccp_screen import CCPScreen


class PersonaEditorMessage(Message):
    """Base persona editor message."""


class PersonaSaveRequested(PersonaEditorMessage):
    """User requested persona save."""

    def __init__(self, persona_data: Dict[str, Any]) -> None:
        super().__init__()
        self.persona_data = persona_data


class CCPPersonaEditorWidget(Container):
    """Minimal persona profile editor."""

    DEFAULT_CSS = """
    CCPPersonaEditorWidget.hidden {
        display: none !important;
    }
    """

    def __init__(self, parent_screen: Optional["CCPScreen"] = None, **kwargs: Any) -> None:
        super().__init__(id="ccp-persona-editor-view", **kwargs)
        self.parent_screen = parent_screen
        self.persona_data: Dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Input(id="ccp-persona-name", placeholder="Persona name")
            yield Select(
                [
                    ("Session Scoped", "session_scoped"),
                    ("Persistent Scoped", "persistent_scoped"),
                ],
                value="session_scoped",
                id="ccp-persona-mode",
            )
            yield TextArea("", id="ccp-persona-system-prompt")
            yield Button("Save Persona", id="ccp-persona-save-button")

    def load_persona(self, persona_data: Dict[str, Any]) -> None:
        """Populate editor fields from persona data."""
        self.persona_data = dict(persona_data)
        self.query_one("#ccp-persona-name", Input).value = persona_data.get("name", "")
        self.query_one("#ccp-persona-mode", Select).value = persona_data.get("mode", "session_scoped")
        self.query_one("#ccp-persona-system-prompt", TextArea).text = (
            persona_data.get("system_prompt")
            or persona_data.get("description")
            or ""
        )

    async def _emit_message_to_app(self, message: Message, handler_name: str) -> None:
        """Post the message through Textual and mirror it to app-level test callbacks."""
        self.post_message(message)
        handler = getattr(self.app, handler_name, None)
        if callable(handler):
            result = handler(message)
            if inspect.isawaitable(result):
                await result

    @on(Button.Pressed, "#ccp-persona-save-button")
    async def handle_save(self, event: Button.Pressed) -> None:
        """Emit persona save request with current editor values."""
        event.stop()
        payload = dict(self.persona_data)
        payload["name"] = self.query_one("#ccp-persona-name", Input).value
        mode_value = self.query_one("#ccp-persona-mode", Select).value
        payload["mode"] = str(mode_value) if mode_value != Select.BLANK else "session_scoped"
        payload["system_prompt"] = self.query_one("#ccp-persona-system-prompt", TextArea).text
        await self._emit_message_to_app(
            PersonaSaveRequested(payload),
            "on_persona_save_requested",
        )
