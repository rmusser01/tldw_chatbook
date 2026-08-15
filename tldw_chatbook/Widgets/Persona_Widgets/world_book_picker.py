"""A small modal for picking a world book to attach to a character (Roleplay P2f).

Distinct from ``ConversationAttachPicker`` (which picks a conversation and
returns a string id); this one lists world books and returns the picked int
world_book id.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

#: Debounce for the search `Input` -- mirrors the console picker family's
#: 0.2 s shape (`console_prompt_picker_modal.py`). A full refresh clears
#: and re-appends every matching `ListItem` into the `ListView`, which
#: should not happen on every keystroke (task-15476).
SEARCH_DEBOUNCE_SECONDS = 0.2


class WorldBookPicker(SafeModalDismissMixin, ModalScreen[int | None]):
    """Pick one world book (by int id) to attach to the current character.

    Args:
        world_books: ``{"world_book_id": int, "name": str}`` rows to choose from
            (already filtered to those not yet attached to the character).
    """

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#world-book-picker-dialog"

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
        self._filter_debounce_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="world-book-picker-dialog"):
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
        needle = event.value
        if self._filter_debounce_timer is not None:
            self._filter_debounce_timer.stop()
        self._filter_debounce_timer = self.set_timer(
            SEARCH_DEBOUNCE_SECONDS, lambda: self._apply_filter_debounced(needle)
        )

    def _apply_filter_debounced(self, raw_value: str) -> None:
        self._filter_debounce_timer = None
        needle = raw_value.strip().lower()
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
        self.dismiss_safe_once(None)


__all__ = ["WorldBookPicker"]
