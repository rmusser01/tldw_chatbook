"""Read-only persona profile view for the Personas workbench.

Styled to match ``PersonasCharacterCardWidget``: labeled inline rows
("Name: x"), empty rows hidden, and the Edit action in a bottom
``.ds-toolbar`` using the shared flat button vocabulary.
"""

from __future__ import annotations

from typing import Any, Dict

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static

from .personas_pane_messages import EditPersonaRequested


class PersonaProfileCardWidget(Container):
    """Read-only persona profile card with an Edit action."""

    # Structure only: colors come from the app stylesheet ($ds-* tokens do not
    # resolve in bare-App harnesses, so DEFAULT_CSS must not reference them).
    DEFAULT_CSS = """
    PersonaProfileCardWidget {
        width: 100%;
        height: 100%;
    }

    PersonaProfileCardWidget .ds-field-row {
        height: auto;
    }

    PersonaProfileCardWidget .ds-toolbar {
        height: 1;
        min-height: 1;
    }

    PersonaProfileCardWidget .ds-toolbar Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        margin-right: 1;
    }
    """

    #: (label, value-Static id) pairs, each rendered as ONE Static with the
    #: label inline ("Name: Archivist"), matching the character card.
    _FIELD_ROWS: tuple[tuple[str, str], ...] = (
        ("Name", "personas-card-name"),
        ("Description", "personas-card-description"),
        ("System prompt", "personas-card-system-prompt"),
    )

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("id", "ccp-persona-card-view")
        super().__init__(**kwargs)
        self._persona_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("Persona Profile", classes="destination-section")
        # markup=False: these Statics render profile content, which must
        # display literally (an unmatched [/tag] would raise MarkupError at
        # render time with markup enabled).
        for _label, dom_id in self._FIELD_ROWS:
            yield Static("", id=dom_id, classes="ds-field-row", markup=False)
        with Horizontal(classes="ds-toolbar"):
            yield Button(
                "Edit",
                id="personas-card-edit",
                disabled=True,
                classes="console-action-secondary",
            )

    def show_persona(self, data: Dict[str, Any]) -> None:
        self._persona_id = str(data.get("id", "")) or None
        values = {
            "personas-card-name": str(data.get("name", "") or "Unnamed"),
            "personas-card-description": str(data.get("description", "") or ""),
            "personas-card-system-prompt": str(data.get("system_prompt", "") or ""),
        }
        for label, dom_id in self._FIELD_ROWS:
            widget = self.query_one(f"#{dom_id}", Static)
            value = values[dom_id]
            # Empty rows are hidden outright - a bare "Label:" line is noise,
            # not information (same rule as the character card).
            widget.display = bool(value)
            widget.update(f"{label}: {value}" if value else f"{label}:")
        self.query_one("#personas-card-edit", Button).disabled = self._persona_id is None

    @on(Button.Pressed, "#personas-card-edit")
    def _edit_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._persona_id is not None:
            self.post_message(EditPersonaRequested(self._persona_id))


__all__ = ["PersonaProfileCardWidget"]
