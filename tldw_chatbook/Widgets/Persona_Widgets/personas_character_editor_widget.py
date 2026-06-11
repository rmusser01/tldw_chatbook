"""ds-native character editor form for the Personas workbench.

Replaces ``CCPCharacterEditorWidget`` on the Personas screen only. It keeps
the legacy widget's external contract — the ``ccp-character-editor-view``
default id, the ``load_character``/``new_character``/``get_character_data``
API, and the legacy ``CharacterSaveRequested``/``CharacterEditorCancelled``
messages — while rendering with the workbench's flat ds vocabulary (primary
fields up top, an Advanced section for the long tail, a read-only avatar
status line, no image box).
"""

from __future__ import annotations

from typing import Any, Dict, List

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Static, TextArea

from ..CCP_Widgets.ccp_character_editor_widget import (
    CharacterEditorCancelled,
    CharacterSaveRequested,
)
from .personas_pane_messages import EditorContentChanged


class PersonasCharacterEditorWidget(Container):
    """ds-field-row character form with an Advanced section and avatar status."""

    # Structure only: colors come from the app stylesheet ($ds-* tokens do not
    # resolve in bare-App harnesses, so DEFAULT_CSS must not reference them).
    DEFAULT_CSS = """
    PersonasCharacterEditorWidget {
        width: 100%;
        height: 100%;
    }

    PersonasCharacterEditorWidget #personas-char-editor-body {
        height: 1fr;
    }

    PersonasCharacterEditorWidget .ds-field-row {
        height: auto;
    }

    PersonasCharacterEditorWidget #personas-char-editor-first-message {
        height: 3;
    }

    PersonasCharacterEditorWidget #personas-char-editor-description {
        height: 4;
    }

    PersonasCharacterEditorWidget #personas-char-editor-personality {
        height: 3;
    }

    PersonasCharacterEditorWidget #personas-char-editor-system-prompt {
        height: 3;
    }

    PersonasCharacterEditorWidget #personas-char-editor-advanced {
        height: auto;
        display: none;
    }

    PersonasCharacterEditorWidget #personas-char-editor-scenario,
    PersonasCharacterEditorWidget #personas-char-editor-post-history,
    PersonasCharacterEditorWidget #personas-char-editor-creator-notes {
        height: 2;
    }

    PersonasCharacterEditorWidget #personas-char-editor-alt-greetings {
        height: 3;
    }

    PersonasCharacterEditorWidget #personas-char-editor-advanced-toggle {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    PersonasCharacterEditorWidget #personas-char-editor-avatar-row {
        height: 1;
        min-height: 1;
        padding: 0 1;
    }

    PersonasCharacterEditorWidget #personas-char-editor-avatar-status {
        width: auto;
        margin-right: 2;
    }

    PersonasCharacterEditorWidget .ds-toolbar {
        height: 1;
        min-height: 1;
    }

    PersonasCharacterEditorWidget .ds-toolbar Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        margin-right: 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "ccp-character-editor-view")
        super().__init__(**kwargs)
        # Base copy of the loaded record: get_character_data starts from it so
        # id/version (and any keys the form does not edit, e.g.
        # character_book) survive a load -> save round trip.
        self._character_data: Dict[str, Any] = {}
        # Greeting fidelity: the loaded greetings list and its joined TextArea
        # form. An untouched save must return the original list verbatim —
        # re-splitting the joined text would corrupt any greeting that itself
        # contains newlines (one multi-paragraph greeting becomes N greetings).
        self._loaded_greetings: List[str] = []
        self._loaded_greetings_text: str = ""
        # Dirty tracking (UX-E3): ``_loading`` suppresses Changed events that
        # are dispatched while a programmatic population is in progress;
        # ``_loaded_snapshot`` is the authoritative suppressor for the ones
        # Textual delivers AFTER ``load_character`` returns (programmatic
        # ``value``/``text`` sets post Changed asynchronously). ``None`` means
        # no editing session has started yet. ``_dirty_posted`` makes the
        # EditorContentChanged announcement once-per-session.
        self._loading: bool = False
        self._loaded_snapshot: tuple | None = None
        self._dirty_posted: bool = False

    def compose(self) -> ComposeResult:
        yield Static("Character Editor", classes="destination-section")
        with VerticalScroll(id="personas-char-editor-body"):
            with Vertical(classes="ds-field-row"):
                yield Label("Name")
                yield Input(id="personas-char-editor-name", placeholder="Character name")
            with Vertical(classes="ds-field-row"):
                yield Label("First message")
                yield TextArea(id="personas-char-editor-first-message")
            with Vertical(classes="ds-field-row"):
                yield Label("Description")
                yield TextArea(id="personas-char-editor-description")
            with Vertical(classes="ds-field-row"):
                yield Label("Personality")
                yield TextArea(id="personas-char-editor-personality")
            with Vertical(classes="ds-field-row"):
                yield Label("System prompt")
                yield TextArea(id="personas-char-editor-system-prompt")
            yield Button(
                "Advanced ▸",
                id="personas-char-editor-advanced-toggle",
                classes="console-action-subdued",
            )
            with Vertical(id="personas-char-editor-advanced"):
                with Vertical(classes="ds-field-row"):
                    yield Label("Scenario")
                    yield TextArea(id="personas-char-editor-scenario")
                with Vertical(classes="ds-field-row"):
                    yield Label("Post-history instructions")
                    yield TextArea(id="personas-char-editor-post-history")
                with Vertical(classes="ds-field-row"):
                    yield Label("Creator notes")
                    yield TextArea(id="personas-char-editor-creator-notes")
                with Vertical(classes="ds-field-row"):
                    yield Label("Creator")
                    yield Input(id="personas-char-editor-creator", placeholder="Creator name")
                with Vertical(classes="ds-field-row"):
                    yield Label("Version")
                    yield Input(id="personas-char-editor-version", value="1.0")
                with Vertical(classes="ds-field-row"):
                    yield Label("Tags (comma-separated)")
                    yield Input(id="personas-char-editor-tags", placeholder="tag, another tag")
                with Vertical(classes="ds-field-row"):
                    yield Label("Alternate greetings (one per line)")
                    yield TextArea(id="personas-char-editor-alt-greetings")
            with Horizontal(id="personas-char-editor-avatar-row"):
                yield Static("Avatar: none", id="personas-char-editor-avatar-status")
        yield Static("", id="personas-char-editor-validation")
        with Horizontal(classes="ds-toolbar"):
            yield Button(
                "Save", id="personas-char-editor-save", classes="console-action-primary"
            )
            yield Button(
                "Cancel",
                id="personas-char-editor-cancel",
                classes="console-action-secondary",
            )

    # ===== Field accessors =====

    def _input(self, suffix: str) -> Input:
        return self.query_one(f"#personas-char-editor-{suffix}", Input)

    def _area(self, suffix: str) -> TextArea:
        return self.query_one(f"#personas-char-editor-{suffix}", TextArea)

    # ===== Public API =====

    def load_character(self, data: Dict[str, Any]) -> None:
        """Fill the form from ``data`` (tolerant of legacy key aliases)."""
        self._loading = True
        try:
            self._populate_form(data)
        finally:
            self._loading = False
        # Re-arm dirty tracking for the new session. The Changed events fired
        # by the programmatic sets above are delivered after this method
        # returns, so the handler compares against this snapshot (taken from
        # the just-populated form) and ignores events that match it.
        self._loaded_snapshot = self._form_snapshot()
        self._dirty_posted = False

    def _populate_form(self, data: Dict[str, Any]) -> None:
        self._character_data = dict(data or {})
        record = self._character_data
        self._input("name").value = str(record.get("name") or "")
        self._area("first-message").text = str(
            record.get("first_mes", record.get("first_message", "")) or ""
        )
        self._area("description").text = str(record.get("description") or "")
        self._area("personality").text = str(record.get("personality") or "")
        self._area("system-prompt").text = str(
            record.get("system_prompt", record.get("system", "")) or ""
        )
        self._area("scenario").text = str(record.get("scenario") or "")
        self._area("post-history").text = str(record.get("post_history_instructions") or "")
        self._area("creator-notes").text = str(record.get("creator_notes") or "")
        self._input("creator").value = str(record.get("creator") or "")
        self._input("version").value = str(
            record.get("character_version", record.get("version", "1.0")) or "1.0"
        )
        self._input("tags").value = ", ".join(
            str(tag) for tag in (record.get("tags") or [])
        )
        self._loaded_greetings = [
            str(greeting) for greeting in (record.get("alternate_greetings") or [])
        ]
        self._loaded_greetings_text = "\n".join(self._loaded_greetings)
        self._area("alt-greetings").text = self._loaded_greetings_text
        avatar = "embedded" if (record.get("image") or record.get("avatar")) else "none"
        self.query_one("#personas-char-editor-avatar-status", Static).update(
            f"Avatar: {avatar}"
        )
        self.query_one("#personas-char-editor-validation", Static).update("")
        self._set_advanced_open(False)

    def new_character(self) -> None:
        """Clear the form for a new (unsaved) character; version defaults 1.0."""
        self.load_character({})

    def show_validation(self, errors: tuple[str, ...]) -> None:
        """Render screen-side validation errors in the editor footer.

        The footer Static is the single in-editor validation surface: the
        editor's own name-required check and the screen's ``_validate_character``
        results (e.g. character_book errors) both land here, in the same
        format. An empty tuple clears it.
        """
        validation = self.query_one("#personas-char-editor-validation", Static)
        if errors:
            validation.update("Validation errors:\n" + "\n".join(errors))
        else:
            validation.update("")

    def get_character_data(self) -> Dict[str, Any]:
        """Current form values, in the legacy editor's key structure.

        Starts from the loaded record copy (preserving ``id``/``version`` and
        unedited keys) and overrides the editor-owned keys. ``first_mes`` is
        the legacy alias the save path normalizes; when the loaded record also
        carried ``first_message`` it is kept in sync so the stale loaded value
        cannot win in the DB save.
        """
        data = dict(self._character_data)
        data["name"] = self._input("name").value
        data["description"] = self._area("description").text
        data["personality"] = self._area("personality").text
        data["scenario"] = self._area("scenario").text
        first_message = self._area("first-message").text
        data["first_mes"] = first_message
        if "first_message" in data:
            data["first_message"] = first_message
        data["creator_notes"] = self._area("creator-notes").text
        data["system_prompt"] = self._area("system-prompt").text
        data["post_history_instructions"] = self._area("post-history").text
        data["creator"] = self._input("creator").value
        # Empty/whitespace Version falls back to the new_character default.
        version = self._input("version").value
        data["character_version"] = version if version.strip() else "1.0"
        # Greeting fidelity rule: if the TextArea text is exactly the joined
        # form of the loaded list, the user did not edit it — return the
        # ORIGINAL list verbatim so multi-line greetings survive the round
        # trip. Only when the text was edited do we re-parse one greeting per
        # non-blank line.
        greetings_text = self._area("alt-greetings").text
        if greetings_text == self._loaded_greetings_text:
            data["alternate_greetings"] = list(self._loaded_greetings)
        else:
            data["alternate_greetings"] = [
                line.strip() for line in greetings_text.splitlines() if line.strip()
            ]
        data["tags"] = [
            tag.strip() for tag in self._input("tags").value.split(",") if tag.strip()
        ]
        return data

    # ===== Internals =====

    def _form_snapshot(self) -> tuple:
        """Raw field values, for change detection (cheap, no parsing)."""
        return (
            self._input("name").value,
            self._area("first-message").text,
            self._area("description").text,
            self._area("personality").text,
            self._area("system-prompt").text,
            self._area("scenario").text,
            self._area("post-history").text,
            self._area("creator-notes").text,
            self._input("creator").value,
            self._input("version").value,
            self._input("tags").value,
            self._area("alt-greetings").text,
        )

    def _set_advanced_open(self, open_: bool) -> None:
        """Show/hide the Advanced section and keep the toggle label in sync."""
        self.query_one("#personas-char-editor-advanced").display = open_
        self.query_one("#personas-char-editor-advanced-toggle", Button).label = (
            "Advanced ▾" if open_ else "Advanced ▸"
        )

    # ===== Events =====

    @on(Input.Changed)
    @on(TextArea.Changed)
    def _field_changed(self, event: Input.Changed | TextArea.Changed) -> None:
        """Announce the first real user modification of the session.

        All Inputs/TextAreas that bubble here are this editor's own fields.
        Programmatic population also fires Changed; those events either land
        while ``_loading`` is set or (the usual case, since Textual posts them
        asynchronously) after ``load_character`` returned, where the snapshot
        comparison filters them out because the form still matches what was
        loaded. Paste and undo also fire Changed, so the comparison covers
        them too.
        """
        if self._loading or self._dirty_posted or self._loaded_snapshot is None:
            return
        if self._form_snapshot() == self._loaded_snapshot:
            return
        self._dirty_posted = True
        self.post_message(EditorContentChanged())

    @on(Button.Pressed, "#personas-char-editor-advanced-toggle")
    def _toggle_advanced(self, event: Button.Pressed) -> None:
        event.stop()
        self._set_advanced_open(
            not self.query_one("#personas-char-editor-advanced").display
        )

    @on(Button.Pressed, "#personas-char-editor-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        validation = self.query_one("#personas-char-editor-validation", Static)
        if not self._input("name").value.strip():
            validation.update("Validation errors:\nname: required")
            return
        validation.update("")
        self.post_message(CharacterSaveRequested(self.get_character_data()))

    @on(Button.Pressed, "#personas-char-editor-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(CharacterEditorCancelled())


__all__ = ["PersonasCharacterEditorWidget"]
