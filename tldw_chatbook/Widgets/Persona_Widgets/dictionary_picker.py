"""A small modal for picking a dictionary to attach to a character (Roleplay P1f).

Distinct from ``DictionaryAttachPicker`` (which picks a conversation and returns a
string id); this one lists dictionaries and returns the picked int dictionary id.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static


class DictionaryPicker(ModalScreen[int | None]):
    """Pick one dictionary (by int id) to attach to the current character.

    Args:
        dictionaries: ``{"dictionary_id": int, "name": str}`` rows to choose from
            (already filtered to those not yet attached to the character).
    """

    DEFAULT_CSS = """
    DictionaryPicker { align: center middle; }
    DictionaryPicker > Vertical {
        width: 60%; max-width: 80; height: auto; max-height: 80%;
        padding: 1 2; border: round $panel;
    }
    DictionaryPicker #dict-pick-list { height: auto; max-height: 16; }
    """

    def __init__(
        self,
        dictionaries: list[dict[str, Any]],
        *,
        title: str = "Attach dictionary",
        confirm_label: str = "Attach",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._dictionaries = list(dictionaries)
        self._row_ids: list[int] = []
        self._title = title
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, markup=False)
            yield Input(placeholder="Search dictionaries…", id="dict-pick-search")
            yield ListView(id="dict-pick-list")
            with Vertical(id="dict-pick-actions"):
                yield Button(self._confirm_label, id="dict-pick-confirm", classes="console-action-secondary")
                yield Button("Cancel", id="dict-pick-cancel", classes="console-action-secondary")

    def on_mount(self) -> None:
        self._populate(self._dictionaries)

    def _populate(self, rows: list[dict[str, Any]]) -> None:
        listing = self.query_one("#dict-pick-list", ListView)
        listing.clear()
        self._row_ids = []
        for row in rows:
            listing.append(ListItem(Static(str(row.get("name") or "(unnamed)"), markup=False)))
            self._row_ids.append(int(row.get("dictionary_id")))
        listing.index = None

    @on(Input.Changed, "#dict-pick-search")
    def _filter(self, event: Input.Changed) -> None:
        event.stop()
        needle = event.value.strip().lower()
        rows = [d for d in self._dictionaries if needle in str(d.get("name") or "").lower()] if needle else self._dictionaries
        self._populate(rows)

    def _selected_id(self) -> int | None:
        listing = self.query_one("#dict-pick-list", ListView)
        index = listing.index
        if index is None or not 0 <= index < len(self._row_ids):
            return None
        return self._row_ids[index]

    @on(Button.Pressed, "#dict-pick-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(self._selected_id())

    @on(Button.Pressed, "#dict-pick-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)


__all__ = ["DictionaryPicker"]
