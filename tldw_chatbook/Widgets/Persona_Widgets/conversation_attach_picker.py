"""A small modal for picking a conversation (by string id) to attach the
current entity to. Generic — used by the Roleplay Lore Attachments flow (P2e);
the dictionary flow keeps its own DictionaryAttachPicker.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static


class ConversationAttachPicker(ModalScreen[str | None]):
    """Pick one conversation (by string id) to attach the current entity to.

    Args:
        conversations: ``{"conversation_id": str, "title": str}`` rows to choose from.
    """

    DEFAULT_CSS = """
    ConversationAttachPicker { align: center middle; }
    ConversationAttachPicker > Vertical {
        width: 60%; max-width: 80; height: auto; max-height: 80%;
        padding: 1 2; border: round $panel;
    }
    ConversationAttachPicker #conversation-attach-list { height: auto; max-height: 16; }
    """

    def __init__(self, conversations: list[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._conversations = list(conversations)
        self._row_ids: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Attach to conversation", markup=False)
            yield Input(placeholder="Search conversations…", id="conversation-attach-search")
            yield ListView(id="conversation-attach-list")
            with Vertical(id="conversation-attach-actions"):
                yield Button(
                    "Attach",
                    id="conversation-attach-confirm",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Cancel",
                    id="conversation-attach-cancel",
                    classes="console-action-secondary",
                )

    def on_mount(self) -> None:
        self._populate(self._conversations)

    def _populate(self, rows: list[dict[str, Any]]) -> None:
        listing = self.query_one("#conversation-attach-list", ListView)
        listing.clear()
        self._row_ids = []
        for row in rows:
            item = ListItem(Static(str(row.get("title") or "(untitled)"), markup=False))
            listing.append(item)
            self._row_ids.append(str(row.get("conversation_id")))
        # A filter change must require an explicit re-select: don't let a
        # previously-highlighted index silently carry over onto a rebuilt,
        # differently-ordered row set.
        listing.index = None

    @on(Input.Changed, "#conversation-attach-search")
    def _filter(self, event: Input.Changed) -> None:
        event.stop()
        needle = event.value.strip().lower()
        rows = (
            [
                c
                for c in self._conversations
                if needle in str(c.get("title") or "").lower()
            ]
            if needle
            else self._conversations
        )
        self._populate(rows)

    def _selected_id(self) -> str | None:
        listing = self.query_one("#conversation-attach-list", ListView)
        index = listing.index
        if index is None or not 0 <= index < len(self._row_ids):
            return None
        return self._row_ids[index]

    @on(Button.Pressed, "#conversation-attach-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(self._selected_id())

    @on(Button.Pressed, "#conversation-attach-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)


__all__ = ["ConversationAttachPicker"]
