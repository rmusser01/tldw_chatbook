"""A small modal for picking a world book to attach to a character (Roleplay P2f).

Distinct from ``ConversationAttachPicker`` (which picks a conversation and
returns a string id); this one lists world books and returns the picked int
world_book id.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static


class WorldBookPicker(ModalScreen[int | None]):
    """Pick one world book (by int id) to attach to the current character.

    Args:
        world_books: ``{"world_book_id": int, "name": str}`` rows to choose from
            (already filtered to those not yet attached to the character).
    """

    DEFAULT_CSS = """
    WorldBookPicker { align: center middle; }
    WorldBookPicker > Vertical {
        width: 60%; max-width: 80; height: auto; max-height: 80%;
        padding: 1 2; border: round $panel;
    }
    WorldBookPicker #worldbook-pick-list { height: auto; max-height: 16; }
    """

    def __init__(
        self,
        world_books: list[dict[str, Any]],
        *,
        title: str = "Attach world book",
        confirm_label: str = "Attach",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._world_books = list(world_books)
        self._row_ids: list[int] = []
        self._title = title
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, markup=False)
            yield Input(placeholder="Search world books…", id="worldbook-pick-search")
            yield ListView(id="worldbook-pick-list")
            with Vertical(id="worldbook-pick-actions"):
                yield Button(
                    self._confirm_label,
                    id="worldbook-pick-confirm",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Cancel", id="worldbook-pick-cancel", classes="console-action-secondary"
                )

    def on_mount(self) -> None:
        self._populate(self._world_books)

    def _populate(self, rows: list[dict[str, Any]]) -> None:
        listing = self.query_one("#worldbook-pick-list", ListView)
        listing.clear()
        self._row_ids = []
        for row in rows:
            listing.append(
                ListItem(Static(str(row.get("name") or "(unnamed)"), markup=False))
            )
            self._row_ids.append(int(row.get("world_book_id")))
        listing.index = None

    @on(Input.Changed, "#worldbook-pick-search")
    def _filter(self, event: Input.Changed) -> None:
        event.stop()
        needle = event.value.strip().lower()
        rows = (
            [b for b in self._world_books if needle in str(b.get("name") or "").lower()]
            if needle
            else self._world_books
        )
        self._populate(rows)

    def _selected_id(self) -> int | None:
        listing = self.query_one("#worldbook-pick-list", ListView)
        index = listing.index
        if index is None or not 0 <= index < len(self._row_ids):
            return None
        return self._row_ids[index]

    @on(Button.Pressed, "#worldbook-pick-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(self._selected_id())

    @on(Button.Pressed, "#worldbook-pick-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)


__all__ = ["WorldBookPicker"]
