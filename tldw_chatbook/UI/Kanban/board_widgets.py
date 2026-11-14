"""Kanban board widgets (TASK-18934, ADR-237).

Adapted from backlog-md-py's ``tui/widgets.py`` (BoardHeader /
BoardColumn / TaskCard / TaskInspector): same widget decomposition and
markup-free rendering discipline — card and inspector text render as
plain strings (``markup=False`` / literal ``Text``) so card content
containing Rich-markup metacharacters can neither swallow styling nor
raise MarkupError.
"""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Static

from tldw_chatbook.UI.Kanban.board_models import BoardSnapshot, CardView, FilterState


class BoardHeader(Horizontal):
    """One-line board title plus filter/source metadata."""

    def __init__(self, board_name: str) -> None:
        super().__init__(id="kanban-board-header")
        self.board_name = board_name

    def compose(self) -> ComposeResult:
        yield Static(self.board_name, id="kanban-board-title", markup=False)
        yield Static("", id="kanban-board-meta", markup=False)

    def update_snapshot(self, snapshot: BoardSnapshot, filters: FilterState) -> None:
        suffix = "local"
        if filters.text.strip():
            suffix = f"{suffix} | filter: {filters.text.strip()}"
        self.query_one("#kanban-board-title", Static).update(snapshot.board_name)
        self.query_one("#kanban-board-meta", Static).update(suffix)


class BoardColumn(VerticalScroll):
    """One status column: title, cards, and an honest empty state."""

    def __init__(
        self,
        *,
        status: str,
        cards: tuple[CardView, ...],
        selected_card_id: int | None,
        id: str,
    ) -> None:
        super().__init__(id=id, classes="kanban-board-column")
        self.status = status
        self.cards = cards
        self.selected_card_id = selected_card_id

    def compose(self) -> ComposeResult:
        yield Static(
            f"{self.status} ({len(self.cards)})",
            classes="kanban-column-title",
            markup=False,
        )
        if not self.cards:
            yield Static("No cards", classes="kanban-empty-column", markup=False)
            return
        for card in self.cards:
            yield CardWidget(card, selected=card.id == self.selected_card_id)


class CardWidget(Static):
    """One card on the board; click selects it."""

    def __init__(self, card: CardView, *, selected: bool) -> None:
        classes = "kanban-card"
        if selected:
            classes = f"{classes} selected"
        super().__init__(
            _card_text(card),
            id=f"kanban-card-{card.id}",
            classes=classes,
            markup=False,
        )
        self.card_view = card

    def on_click(self, event: events.Click) -> None:
        screen = self.screen
        select = getattr(screen, "select_card", None)
        if callable(select):
            select(self.card_view.id)
        event.stop()


class CardInspector(VerticalScroll):
    """Keyboard-scrollable full card details; Escape returns to the board."""

    def compose(self) -> ComposeResult:
        self._card_id: int | None = None
        yield Static("No card selected", id="kanban-card-inspector-text", markup=False)

    def update_card(self, card: CardView | None) -> None:
        card_id = card.id if card is not None else None
        self.query_one(Static).update(
            _inspector_text(card) if card is not None else "No card selected"
        )
        if card_id != self._card_id:
            self.scroll_home(animate=False)
        self._card_id = card_id


def _card_text(card: CardView) -> str:
    priority = f" [{card.priority}]" if card.priority else ""
    lines = [f"{card.display_id}{priority}", card.title]
    if card.due_date:
        lines.append(f"Due: {card.due_date}")
    return "\n".join(lines)


def _inspector_text(card: CardView) -> str:
    lines = [
        f"{card.display_id} - {card.title}",
        f"Column: {card.status}",
    ]
    if card.priority:
        lines.append(f"Priority: {card.priority}")
    if card.due_date:
        lines.append(f"Due: {card.due_date}")
    if card.updated_at:
        lines.append(f"Updated: {card.updated_at}")
    if card.description:
        lines.extend(["", card.description])
    return "\n".join(lines)
