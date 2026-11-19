"""Kanban board view-models (TASK-18934, ADR-237).

Pure logic, no Textual and no store imports: adapted from backlog-md-py's
``tui/models.py`` (filtering, refresh-stable selection) with the
markdown-project fields (file paths, orchestration queue, run history)
replaced by kanban card fields. The screen's snapshot loader is the only
code that maps store rows onto these models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class CardView:
    """One kanban card as the board renders it."""

    id: int
    list_id: int
    title: str
    status: str
    description: str = ""
    priority: str | None = None
    due_date: str | None = None
    updated_at: str | None = None

    @property
    def display_id(self) -> str:
        return f"#{self.id}"


@dataclass(frozen=True)
class BoardSnapshot:
    """One board's columns of cards, ready to render.

    ``list_ids`` maps each column name to the store list id behind it, so
    card mutations (move/new) work off the snapshot alone — whoever built
    the snapshot (default store loader or a test loader) owns the mapping.
    """

    board_id: int | None
    board_name: str
    statuses: tuple[str, ...]
    columns: Mapping[str, tuple[CardView, ...]]
    list_ids: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class FilterState:
    """The board's free-text filter (the only filter v1 ships)."""

    text: str = ""


@dataclass(frozen=True)
class SelectionState:
    """The board's cursor: which card, in which column, at which row."""

    card_id: int | None = None
    status: str | None = None
    row: int = 0


def filter_snapshot(snapshot: BoardSnapshot, filters: FilterState) -> BoardSnapshot:
    """Return ``snapshot`` with each column narrowed by the text filter."""
    needle = filters.text.strip().casefold()
    if not needle:
        return snapshot
    columns = {
        status: tuple(card for card in cards if _matches(card, needle))
        for status, cards in snapshot.columns.items()
    }
    return replace(snapshot, columns=columns)


def _matches(card: CardView, needle: str) -> bool:
    return any(
        needle in value.casefold()
        for value in (str(card.id), card.title, card.description)
        if value
    )


def select_after_refresh(
    previous: BoardSnapshot,
    current: BoardSnapshot,
    old_selection: SelectionState,
) -> SelectionState:
    """Pick the selection a refresh should keep (upstream's contract).

    The still-existing selected card wins; otherwise the same row of the
    same column; otherwise the nearest following (then preceding) column
    that holds a card; otherwise the first card on the board.
    """
    if old_selection.card_id is not None:
        preserved = _find_card_selection(current, old_selection.card_id)
        if preserved is not None:
            return preserved

    if old_selection.status in current.columns:
        cards = current.columns[old_selection.status]
        if cards:
            row = _bounded_row(old_selection.row, len(cards))
            return SelectionState(card_id=cards[row].id, status=old_selection.status, row=row)

    if old_selection.status in previous.statuses:
        start = previous.statuses.index(old_selection.status)
    elif old_selection.status in current.statuses:
        start = current.statuses.index(old_selection.status)
    else:
        return _first_available_selection(current)

    for index in range(start + 1, len(current.statuses)):
        selection = _first_in_column(current, index)
        if selection is not None:
            return selection
    for index in range(start - 1, -1, -1):
        selection = _first_in_column(current, index)
        if selection is not None:
            return selection
    return SelectionState()


def _find_card_selection(
    snapshot: BoardSnapshot, card_id: int
) -> SelectionState | None:
    for status in snapshot.statuses:
        for row, card in enumerate(snapshot.columns.get(status, ())):
            if card.id == card_id:
                return SelectionState(card_id=card.id, status=status, row=row)
    return None


def _bounded_row(row: int, length: int) -> int:
    if length <= 0:
        return 0
    return max(0, min(row, length - 1))


def _first_in_column(snapshot: BoardSnapshot, index: int) -> SelectionState | None:
    status = snapshot.statuses[index]
    cards = snapshot.columns.get(status, ())
    if not cards:
        return None
    return SelectionState(card_id=cards[0].id, status=status, row=0)


def _first_available_selection(snapshot: BoardSnapshot) -> SelectionState:
    for index in range(len(snapshot.statuses)):
        selection = _first_in_column(snapshot, index)
        if selection is not None:
            return selection
    return SelectionState()
