"""Roleplay P1f: an I/O-free panel listing a character's embedded dictionaries.

The panel renders what the screen feeds via ``load_character_dictionaries`` and
posts intent messages; the screen owns all service/DB work. Each embedded
dictionary is a snapshot (an embedded copy — editing the source dictionary does
not update it).
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, DataTable, Static


class CharacterDictionaryAttachRequested(Message):
    """Request the attach-dictionary picker for the current character."""


class CharacterDictionaryDetachRequested(Message):
    """Detach one embedded dictionary from the current character.

    Args:
        dictionary_name: The embedded dictionary to remove (by name).
    """

    def __init__(self, dictionary_name: str) -> None:
        super().__init__()
        self.dictionary_name = dictionary_name


class PersonasCharacterDictionariesWidget(Container):
    """List + attach/detach a character's embedded dictionaries (snapshots)."""

    DEFAULT_CSS = """
    PersonasCharacterDictionariesWidget #personas-char-dicts-table { height: auto; max-height: 8; }
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "personas-character-dictionaries")
        super().__init__(**kwargs)
        self._rows: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Static("Dictionaries (embedded copies)", classes="destination-section")
        yield Static(
            "No dictionaries attached to this character yet.",
            id="personas-char-dicts-empty",
            markup=False,
        )
        yield DataTable(id="personas-char-dicts-table", cursor_type="row")
        with Horizontal(classes="personas-dict-form-row"):
            yield Button("Attach dictionary…", id="personas-char-dicts-add", classes="console-action-secondary")
            yield Button("Detach", id="personas-char-dicts-detach", classes="console-action-secondary")

    def on_mount(self) -> None:
        self.query_one("#personas-char-dicts-table", DataTable).add_columns("dictionary", "entries")
        self.load_character_dictionaries([])

    def load_character_dictionaries(self, rows: list[dict[str, Any]]) -> None:
        """Render the character's embedded dictionaries.

        Args:
            rows: ``{"name": str, "entry_count": int, "enabled": bool}`` entries.

        A hostile/crafted card import can produce two embedded blocks with
        the same name (``attach_to_character`` dedups by name so it never
        creates this, but nothing stops a crafted ``extensions`` payload from
        having it). ``DataTable.add_row`` keys rows by ``str(name)``, so a
        second same-named row would raise ``DuplicateKey`` — which would
        propagate uncaught through the import worker and exit the app. Dedup
        by name (first occurrence wins) before touching the table so that
        can never happen, regardless of what the screen feeds this panel.
        """
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            name = str(row.get("name"))
            if name in seen:
                continue
            seen.add(name)
            deduped.append(row)
        self._rows = deduped
        table = self.query_one("#personas-char-dicts-table", DataTable)
        table.clear()
        for row in self._rows:
            table.add_row(
                Text(str(row.get("name") or "(unnamed)")),
                Text(str(row.get("entry_count") if row.get("entry_count") is not None else "")),
                key=str(row.get("name")),
            )
        empty = self.query_one("#personas-char-dicts-empty", Static)
        empty.display = not self._rows
        table.display = bool(self._rows)

    def _selected_name(self) -> str | None:
        table = self.query_one("#personas-char-dicts-table", DataTable)
        if table.row_count == 0 or table.cursor_row is None or table.cursor_row < 0:
            return None
        try:
            return str(table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value)
        except Exception:
            return None

    @on(Button.Pressed, "#personas-char-dicts-add")
    def _attach_pressed(self, event: Button.Pressed) -> None:
        # Note: named `_attach_pressed` (not `_attach`) to avoid shadowing
        # Textual's internal `DOMNode._attach(self, parent)` used during mounting.
        event.stop()
        self.post_message(CharacterDictionaryAttachRequested())

    @on(Button.Pressed, "#personas-char-dicts-detach")
    def _detach_pressed(self, event: Button.Pressed) -> None:
        # Note: named `_detach_pressed` (not `_detach`) to avoid shadowing
        # Textual's internal `DOMNode._detach` used during unmounting.
        event.stop()
        name = self._selected_name()
        if name is not None:
            self.post_message(CharacterDictionaryDetachRequested(name))


__all__ = [
    "PersonasCharacterDictionariesWidget",
    "CharacterDictionaryAttachRequested",
    "CharacterDictionaryDetachRequested",
]
