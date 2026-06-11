"""Persona profile create/edit form for the Personas workbench."""

from __future__ import annotations

from typing import Any, Dict

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Input, Label, Static, TextArea

from .personas_pane_messages import PersonaProfileEditCancelled, PersonaProfileSaveRequested


class PersonaProfileEditorWidget(Container):
    """ds-field-row form: name, description, system prompt."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("id", "ccp-persona-editor-view")
        super().__init__(**kwargs)
        self._persona_id: str | None = None
        self._version: int | None = None

    def compose(self) -> ComposeResult:
        yield Static("Persona Editor", classes="destination-section")
        with Vertical(classes="ds-field-row"):
            yield Label("Name")
            yield Input(id="personas-editor-name", placeholder="Persona name")
        with Vertical(classes="ds-field-row"):
            yield Label("Description")
            yield TextArea(id="personas-editor-description")
        with Vertical(classes="ds-field-row"):
            yield Label("System prompt")
            yield TextArea(id="personas-editor-system-prompt")
        yield Static("", id="personas-editor-validation")
        with Horizontal(classes="ds-toolbar"):
            yield Button("Save", id="personas-editor-save")
            yield Button("Cancel", id="personas-editor-cancel")

    def load_persona(self, data: Dict[str, Any]) -> None:
        """Push persona data into the editor form fields.

        CCPPersonaHandler calls this method when it queries ``#ccp-persona-editor-view``
        and finds a ``load_persona`` attribute (see ccp_persona_handler._load_editor).
        """
        self._persona_id = str(data.get("id", "")) or None
        # Kept for optimistic locking: the save path passes it back as
        # ``expected_version`` (None for new personas).
        self._version = data.get("version")
        self.query_one("#personas-editor-name", Input).value = str(data.get("name", ""))
        self.query_one("#personas-editor-description", TextArea).text = str(
            data.get("description", "")
        )
        self.query_one("#personas-editor-system-prompt", TextArea).text = str(
            data.get("system_prompt", "")
        )
        self.query_one("#personas-editor-validation", Static).update("")

    def new_persona(self) -> None:
        """Clear the form for a new (unsaved) persona."""
        self.load_persona({})

    def collect(self) -> Dict[str, Any]:
        """Return the current form values as a dict.

        The ``id`` and ``version`` keys are omitted when no persona has been
        loaded (new form).
        """
        data: Dict[str, Any] = {
            "name": self.query_one("#personas-editor-name", Input).value.strip(),
            "description": self.query_one("#personas-editor-description", TextArea).text,
            "system_prompt": self.query_one("#personas-editor-system-prompt", TextArea).text,
        }
        if self._persona_id is not None:
            data["id"] = self._persona_id
        if self._version is not None:
            data["version"] = self._version
        return data

    def validate(self) -> tuple[str, ...]:
        """Return a tuple of validation error strings, empty if valid."""
        errors: list[str] = []
        if not self.query_one("#personas-editor-name", Input).value.strip():
            errors.append("name: required")
        return tuple(errors)

    @on(Button.Pressed, "#personas-editor-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        errors = self.validate()
        validation = self.query_one("#personas-editor-validation", Static)
        if errors:
            validation.update("Validation errors:\n" + "\n".join(errors))
            return
        validation.update("")
        self.post_message(PersonaProfileSaveRequested(self.collect()))

    @on(Button.Pressed, "#personas-editor-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaProfileEditCancelled())


__all__ = ["PersonaProfileEditorWidget"]
