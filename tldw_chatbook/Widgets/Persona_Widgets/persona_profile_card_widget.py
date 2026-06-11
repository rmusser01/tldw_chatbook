"""Read-only persona profile view for the Personas workbench."""

from __future__ import annotations

from typing import Any, Dict

from textual import on
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Button, Static

from .personas_pane_messages import EditPersonaRequested


class PersonaProfileCardWidget(Container):
    """Read-only persona profile card with an Edit action."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("id", "ccp-persona-card-view")
        super().__init__(**kwargs)
        self._persona_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("Persona Profile", classes="destination-section")
        yield Static("", id="personas-card-name")
        yield Static("", id="personas-card-description")
        yield Static("System prompt", classes="destination-section")
        yield Static("", id="personas-card-system-prompt")
        yield Button("Edit", id="personas-card-edit", disabled=True)

    def show_persona(self, data: Dict[str, Any]) -> None:
        self._persona_id = str(data.get("id", "")) or None
        self.query_one("#personas-card-name", Static).update(str(data.get("name", "Unnamed")))
        self.query_one("#personas-card-description", Static).update(
            str(data.get("description", ""))
        )
        self.query_one("#personas-card-system-prompt", Static).update(
            str(data.get("system_prompt", ""))
        )
        self.query_one("#personas-card-edit", Button).disabled = self._persona_id is None

    @on(Button.Pressed, "#personas-card-edit")
    def _edit_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._persona_id is not None:
            self.post_message(EditPersonaRequested(self._persona_id))


__all__ = ["PersonaProfileCardWidget"]
