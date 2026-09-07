"""Roleplay P1f: an I/O-free panel listing a character's embedded dictionaries.

The panel renders what the screen feeds via ``load_character_dictionaries`` and
posts intent messages; the screen owns all service/DB work. Each embedded
dictionary is a snapshot (an embedded copy — editing the source dictionary does
not update it).

task-2231: the panel is a collapsible section (the preview pane's disclosure
idiom - a full-width toggle button doubling as the section header, carrying the
live count). Collapsed by default so an empty section costs exactly one line;
the collapsed state is widget state that ``load_character_dictionaries`` never
resets, so the user's choice survives attach/detach refreshes for the session.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
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

    # Structure only (no $ds-* tokens: they do not resolve in bare-App
    # harnesses). ``height: auto`` everywhere is load-bearing: the button row
    # is a Horizontal (Textual default ``height: 1fr``), and a 1fr descendant
    # of an auto-height container makes the measurement resolve to "fill all
    # available space" instead of "sum of children" - the old bottom-dock CSS
    # hid that behind an explicit wrapper max-height.
    BUNDLED_CSS = """
    PersonasCharacterDictionariesWidget {
        height: auto;
        width: 100%;
    }

    PersonasCharacterDictionariesWidget #personas-char-dicts-toggle {
        width: 100%;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    PersonasCharacterDictionariesWidget #personas-char-dicts-body {
        height: auto;
    }

    PersonasCharacterDictionariesWidget .personas-dict-form-row {
        height: auto;
    }

    PersonasCharacterDictionariesWidget #personas-char-dicts-table { height: auto; max-height: 8; }
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "personas-character-dictionaries")
        super().__init__(**kwargs)
        self._rows: list[dict[str, Any]] = []
        # Session-persistent disclosure state; load_character_dictionaries
        # updates the count but never this flag.
        self._collapsed: bool = True

    def compose(self) -> ComposeResult:
        # The toggle doubles as the section header (count included), so the
        # old standalone "Dictionaries (copied into this character)" descriptor
        # line is folded into its tooltip.
        yield Button(
            self._toggle_label(),
            id="personas-char-dicts-toggle",
            classes="console-action-subdued",
            tooltip="Dictionaries are copied into this character as snapshots.",
        )
        with Vertical(id="personas-char-dicts-body"):
            yield Static(
                "No dictionaries attached to this character yet.",
                id="personas-char-dicts-empty",
                markup=False,
            )
            yield DataTable(id="personas-char-dicts-table", cursor_type="row")
            with Horizontal(classes="personas-dict-form-row"):
                yield Button(
                    "Attach dictionary…",
                    id="personas-char-dicts-add",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Detach",
                    id="personas-char-dicts-detach",
                    classes="console-action-secondary",
                )

    def on_mount(self) -> None:
        self.query_one("#personas-char-dicts-body").display = False
        self.query_one("#personas-char-dicts-table", DataTable).add_columns(
            "dictionary", "entries"
        )
        self.load_character_dictionaries([])

    def _toggle_label(self) -> str:
        """Header line: disclosure arrow plus the live row count."""
        arrow = "▸" if self._collapsed else "▾"
        return f"{arrow} Dictionaries ({len(self._rows)})"

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
                Text(
                    str(
                        row.get("entry_count")
                        if row.get("entry_count") is not None
                        else ""
                    )
                ),
                key=str(row.get("name")),
            )
        empty = self.query_one("#personas-char-dicts-empty", Static)
        empty.display = not self._rows
        table.display = bool(self._rows)
        # The count lives in the section header; the collapsed state does not
        # (a data refresh must not re-collapse what the user expanded).
        self.query_one("#personas-char-dicts-toggle", Button).label = (
            self._toggle_label()
        )

    def _selected_name(self) -> str | None:
        table = self.query_one("#personas-char-dicts-table", DataTable)
        if table.row_count == 0 or table.cursor_row is None or table.cursor_row < 0:
            return None
        try:
            return str(
                table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value
            )
        except Exception:
            return None

    @on(Button.Pressed, "#personas-char-dicts-toggle")
    def _toggle_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._collapsed = not self._collapsed
        self.query_one("#personas-char-dicts-body").display = not self._collapsed
        self.query_one("#personas-char-dicts-toggle", Button).label = (
            self._toggle_label()
        )

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
