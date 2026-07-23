"""Roleplay P2f: an I/O-free panel listing a character's embedded world books.

The panel renders what the screen feeds via ``load_world_books`` and posts
intent messages; the screen owns all service/DB work. Each embedded world book
is a snapshot (an embedded copy — editing the source book does not update it).
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, DataTable, Static


class CharacterWorldBookAttachRequested(Message):
    """Request the attach-world-book picker for the current character."""


class CharacterWorldBookDetachRequested(Message):
    """Detach one embedded world book from the current character.

    Args:
        name: The embedded world book to remove (by name).
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class PersonasCharacterWorldBooksWidget(Container):
    """List + attach/detach a character's embedded world books (snapshots)."""

    DEFAULT_CSS = """
    PersonasCharacterWorldBooksWidget #personas-char-worldbooks-table { height: auto; max-height: 8; }
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "personas-character-world-books")
        super().__init__(**kwargs)
        self._rows: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Static("World Books (embedded copies)", classes="destination-section")
        yield Static(
            "No world books attached to this character yet.",
            id="personas-char-worldbooks-empty",
            markup=False,
        )
        yield DataTable(id="personas-char-worldbooks-table", cursor_type="row")
        with Horizontal(classes="personas-dict-form-row"):
            yield Button(
                "Attach world book…",
                id="personas-char-worldbooks-add",
                classes="console-action-secondary",
            )
            yield Button(
                "Detach",
                id="personas-char-worldbooks-detach",
                classes="console-action-secondary",
            )

    def on_mount(self) -> None:
        self.query_one("#personas-char-worldbooks-table", DataTable).add_columns(
            "world book", "entries"
        )
        self.load_world_books([])

    def load_world_books(self, rows: list[dict[str, Any]]) -> None:
        """Render the character's embedded world books.

        Args:
            rows: ``{"name": str, "entry_count": int, "enabled": bool}`` entries.

        Dedup by name (first wins) before touching the table: ``DataTable`` keys
        rows by ``str(name)``, so a hostile/imported card with two same-named
        embedded blocks would otherwise raise ``DuplicateKey`` — which would
        propagate uncaught through the import worker and exit the app.
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
        table = self.query_one("#personas-char-worldbooks-table", DataTable)
        table.clear()
        for row in self._rows:
            table.add_row(
                Text(str(row.get("name") or "(unnamed)")),
                Text(
                    str(
                        row.get("entry_count")
                        if row.get("entry_count") is not None
                        else ""
                    )
                ),
                key=str(row.get("name")),
            )
        empty = self.query_one("#personas-char-worldbooks-empty", Static)
        empty.display = not self._rows
        table.display = bool(self._rows)

    def _selected_name(self) -> str | None:
        table = self.query_one("#personas-char-worldbooks-table", DataTable)
        if table.row_count == 0 or table.cursor_row is None or table.cursor_row < 0:
            return None
        try:
            return str(
                table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value
            )
        except Exception:
            return None

    @on(Button.Pressed, "#personas-char-worldbooks-add")
    def _attach_pressed(self, event: Button.Pressed) -> None:
        # `_attach_pressed` (not `_attach`) avoids shadowing DOMNode._attach.
        event.stop()
        self.post_message(CharacterWorldBookAttachRequested())

    @on(Button.Pressed, "#personas-char-worldbooks-detach")
    def _detach_pressed(self, event: Button.Pressed) -> None:
        # `_detach_pressed` (not `_detach`) avoids shadowing DOMNode._detach.
        event.stop()
        name = self._selected_name()
        if name is not None:
            self.post_message(CharacterWorldBookDetachRequested(name))


__all__ = [
    "PersonasCharacterWorldBooksWidget",
    "CharacterWorldBookAttachRequested",
    "CharacterWorldBookDetachRequested",
]
