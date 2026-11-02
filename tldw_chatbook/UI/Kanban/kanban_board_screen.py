"""Kanban board screen (TASK-18934, ADR-237).

A keyboard-first destination screen hosting the ported board UI:
columns ←/→ (arrows or ``h``/``l``), cards ↑/↓ (or ``k``/``j``), Enter
focuses the card detail inspector, ``shift+h``/``shift+l`` move the
selected card to the adjacent column, ``n`` creates a card in the focused
column, ``/`` focuses the filter, ``r`` refreshes. Adapted from
backlog-md-py's ``tui/app.py`` controller logic; per ADR-031 the
upstream ``q``/``ctrl+q`` quit bindings are dropped (app-global
``ctrl+q`` owns quit) and refresh runs as an exclusive async worker on
the screen instead of upstream's thread + fixed-interval polling — the
store is local SQLite on the app loop, so action-driven refresh is the
honest cadence.

The snapshot loader is injectable (``snapshot_loader``) so the board's
logic tests run without the store; the default loader is the only code
that maps ``LocalKanbanService`` rows onto the board view-models.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, HorizontalScroll, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from tldw_chatbook.UI.Kanban.board_models import (
    BoardSnapshot,
    CardView,
    FilterState,
    SelectionState,
    filter_snapshot,
    select_after_refresh,
)
from tldw_chatbook.UI.Kanban.board_widgets import (
    BoardColumn,
    BoardHeader,
    CardInspector,
    CardWidget,
)
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen

SnapshotLoader = Callable[[], Awaitable[BoardSnapshot]]

_BOARD_LOADING_MESSAGE = "Loading board..."
_NO_BOARDS_HINT = "No kanban board yet — press n to create one (To Do / Doing / Done)."

_DEFAULT_BOARD_NAME = "Personal"
_DEFAULT_BOARD_LISTS = ("To Do", "Doing", "Done")


class NewCardModal(ModalScreen[str | None]):
    """Title-only new-card dialog: Enter confirms, Escape cancels."""

    BINDINGS = (
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "confirm", "Add", show=False),
    )

    def __init__(self, column_name: str) -> None:
        super().__init__(id="kanban-new-card-modal")
        self.column_name = column_name

    def compose(self) -> ComposeResult:
        with Vertical(id="kanban-new-card-dialog"):
            yield Static(
                f"New card in {self.column_name}",
                id="kanban-new-card-label",
                markup=False,
            )
            yield Input(placeholder="Card title", id="kanban-new-card-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # The auto-focused Input's own enter→submit binding outranks this
        # screen's BINDINGS entry, so Enter arrives here as Input.Submitted
        # (the same pattern every Input-dialog screen in this app uses).
        # Without this handler Enter does nothing at all.
        event.stop()
        self.action_confirm()

    def action_confirm(self) -> None:
        title = self.query_one("#kanban-new-card-input", Input).value.strip()
        self.dismiss(title or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class KanbanBoardScreen(BaseAppScreen):
    """The Kanban destination: a board of columns and cards."""

    BUNDLED_CSS = """
    #kanban-board-root {
        height: 100%;
        min-height: 0;
    }
    #kanban-board-header {
        height: auto;
        min-height: 0;
        padding: $ds-space-0 $ds-space-1;
    }
    #kanban-board-title {
        width: auto;
        padding: $ds-space-0 $ds-space-1 $ds-space-0 $ds-space-0;
    }
    #kanban-board-meta {
        width: 1fr;
        color: $ds-text-muted;
    }
    #kanban-filter-text {
        height: auto;
        min-height: 0;
    }
    #kanban-board-columns {
        height: 1fr;
        min-height: 0;
    }
    #kanban-column-strip {
        height: 100%;
        width: auto;
    }
    .kanban-board-column {
        width: $ds-size-34;
        min-width: $ds-size-26;
        height: 100%;
        border: round $ds-control-edge;
        margin: $ds-space-0 $ds-space-1 $ds-space-0 $ds-space-0;
        padding: $ds-space-0 $ds-space-1;
    }
    .kanban-column-title {
        padding: $ds-space-0 $ds-space-0 $ds-space-1 $ds-space-0;
    }
    .kanban-empty-column {
        color: $ds-text-muted;
    }
    .kanban-card {
        height: auto;
        margin: $ds-space-0 $ds-space-0 $ds-space-1 $ds-space-0;
        padding: $ds-space-0 $ds-space-1;
    }
    .kanban-card.selected {
        border: heavy $ds-focus-accent;
    }
    #kanban-card-inspector {
        height: $ds-size-10;
        min-height: $ds-control-height;
        border: round $ds-control-edge;
        padding: $ds-space-0 $ds-space-1;
    }
    """

    BINDINGS = (
        Binding("left", "cursor_left", "Left", show=False, priority=True),
        Binding("right", "cursor_right", "Right", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("h", "cursor_left", "Left", show=False),
        Binding("l", "cursor_right", "Right", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("H", "move_card_left", "Move card ←", key_display="Shift+H"),
        Binding("L", "move_card_right", "Move card →", key_display="Shift+L"),
        Binding("enter", "focus_inspector", "Card detail"),
        Binding("escape", "focus_board", "Board", show=False),
        Binding("n", "new_card", "New card"),
        Binding("/", "focus_filter", "Filter", show=False),
        Binding("r", "refresh_board", "Refresh"),
    )

    def __init__(
        self, app_instance: Any, screen_name: str = "kanban", **kwargs
    ) -> None:
        super().__init__(app_instance, screen_name, **kwargs)
        #: Injectable for tests; None means the default store loader.
        self.snapshot_loader: SnapshotLoader | None = None
        self.snapshot: BoardSnapshot | None = None
        self.visible_snapshot: BoardSnapshot | None = None
        self.filter_state = FilterState()
        self.selection = SelectionState()
        self._reselect_card_id: int | None = None
        self._bootstrap_in_flight = False
        #: Column name -> store list id (from the last loaded snapshot).
        self._list_ids_by_status: dict[str, int] = {}
        self._column_widget_counter: dict[str, int] = {}

    # -- compose / lifecycle --------------------------------------------------

    def compose_content(self) -> ComposeResult:
        with Container(id="kanban-board-root"):
            yield BoardHeader("Kanban")
            yield Input(placeholder="Filter cards…", id="kanban-filter-text")
            with (
                HorizontalScroll(id="kanban-board-columns"),
                HorizontalScroll(id="kanban-column-strip"),
            ):
                yield Static(_BOARD_LOADING_MESSAGE, id="kanban-board-loading")
            yield CardInspector(id="kanban-card-inspector")

    def on_mount(self) -> None:
        self.refresh_board()
        strip = self.query_one("#kanban-board-columns")
        self.call_after_refresh(lambda: self.set_focus(strip))

    def on_screen_resume(self) -> None:
        self.refresh_board()

    # -- data -------------------------------------------------------------------

    def refresh_board(self) -> None:
        """Reload the snapshot off the render path (exclusive worker)."""
        self._load_and_apply()

    @work(exclusive=True, group="kanban-board-refresh")
    async def _load_and_apply(self) -> None:
        loader = self.snapshot_loader or self._default_snapshot_loader
        try:
            snapshot = await loader()
            await self._apply_snapshot(snapshot)
        except Exception as exc:  # noqa: BLE001 — a failed load toasts, never crashes
            self.notify(f"Kanban board load failed: {exc}", severity="error", markup=False)

    async def _apply_snapshot(self, snapshot: BoardSnapshot) -> None:
        previous = self.visible_snapshot or snapshot
        self.snapshot = snapshot
        self.visible_snapshot = filter_snapshot(snapshot, self.filter_state)
        self._list_ids_by_status = dict(snapshot.list_ids)
        self._bootstrap_in_flight = False
        if self._reselect_card_id is not None:
            self.selection = select_after_refresh(
                previous,
                self.visible_snapshot,
                SelectionState(
                    card_id=self._reselect_card_id,
                    status=self.selection.status,
                    row=self.selection.row,
                ),
            )
            self._reselect_card_id = None
        else:
            self.selection = select_after_refresh(
                previous, self.visible_snapshot, self.selection
            )
        await self._render_board()

    async def _default_snapshot_loader(self) -> BoardSnapshot:
        """Map ``LocalKanbanService`` rows onto the board view-models."""
        service = self.app_instance.local_kanban_service
        if service is None:
            return BoardSnapshot(
                board_id=None, board_name="Kanban", statuses=(), columns={}
            )
        boards_payload = await service.list_boards()
        boards = boards_payload.get("boards") or []
        if not boards:
            return BoardSnapshot(
                board_id=None, board_name="Kanban", statuses=(), columns={}
            )
        board = boards[0]
        board_id = int(board["id"])
        lists_payload = await service.list_lists(board_id)
        board_lists = sorted(
            lists_payload.get("lists") or [],
            key=lambda item: item.get("position") or 0,
        )
        # The board keys columns by NAME, but nothing stops two lists from
        # sharing one (the store has no UNIQUE(board_id, name)); suffix
        # duplicates instead of letting the second silently swallow the
        # first's cards.
        list_ids: dict[str, int] = {}
        columns: dict[str, tuple[CardView, ...]] = {}
        # Reserve literal names before allocating suffixes, including names
        # that already look generated (e.g. "Done (2)").
        literal_names = {
            str(item.get("name") or f"List {int(item['id'])}") for item in board_lists
        }
        for board_list in board_lists:
            list_id = int(board_list["id"])
            name = str(board_list.get("name") or f"List {list_id}")
            if name in list_ids:
                stem = name
                suffix = 2
                name = f"{stem} ({suffix})"
                while name in list_ids or name in literal_names:
                    suffix += 1
                    name = f"{stem} ({suffix})"
            list_ids[name] = list_id
            cards_payload = await service.list_cards(list_id)
            columns[name] = tuple(
                CardView(
                    id=int(card["id"]),
                    list_id=int(card.get("list_id") or list_id),
                    title=str(card.get("title") or ""),
                    status=name,
                    description=str(card.get("description") or ""),
                    priority=card.get("priority"),
                    due_date=card.get("due_date"),
                    updated_at=card.get("updated_at"),
                )
                for card in (cards_payload.get("cards") or [])
            )
        return BoardSnapshot(
            board_id=board_id,
            board_name=str(board.get("name") or "Board"),
            statuses=tuple(columns),
            columns=columns,
            list_ids=list_ids,
        )

    # -- render -------------------------------------------------------------------

    async def _render_board(self) -> None:
        visible = self.visible_snapshot
        if visible is None:
            return
        self.query_one(BoardHeader).update_snapshot(visible, self.filter_state)
        strip = self.query_one("#kanban-column-strip")
        await strip.remove_children()
        self._column_widget_counter = {}
        if visible.statuses:
            await strip.mount(
                *[
                    BoardColumn(
                        status=status,
                        cards=visible.columns.get(status, ()),
                        selected_card_id=self.selection.card_id,
                        id=self._column_widget_id(status),
                    )
                    for status in visible.statuses
                ]
            )
        elif self.snapshot is not None and self.snapshot.board_id is None:
            await strip.mount(Static(_NO_BOARDS_HINT, id="kanban-board-loading"))
        else:
            await strip.mount(
                Static(
                    "No cards match the filter."
                    if self.filter_state.text.strip()
                    else "No cards yet — press n to add one.",
                    id="kanban-board-loading",
                )
            )
        self._render_selection()

    def _render_selection(self) -> None:
        """Keep mounted columns and their scroll offsets when selection changes."""
        for widget in self.query(CardWidget):
            widget.set_class(widget.card_view.id == self.selection.card_id, "selected")
        self.query_one(CardInspector).update_card(self._selected_card())
        self.call_after_refresh(self._reveal_selection)

    def _reveal_selection(self) -> None:
        for widget in self.query(".kanban-card.selected"):
            widget.scroll_visible(animate=False)

    def _column_widget_id(self, status: str) -> str:
        # Upstream's widget_id discipline: Textual ids are ASCII-only, so a
        # status like "To Do" (space) must be sanitized before it can key a
        # DOM node.
        sanitized = "".join(
            character if character.isascii() and character.isalnum() else "-"
            for character in status
        ).strip("-")
        base = f"kanban-column-{sanitized or 'column'}"
        candidate = base
        suffix = 1
        # Uniqueness across this render pass, not just per distinct name:
        # a pathological set ("To Do", "To Do", "To-Do-2") can still
        # collide after suffixing, and a duplicate id raises DuplicateIds
        # inside the refresh worker.
        while candidate in self._column_widget_counter:
            suffix += 1
            candidate = f"{base}-{suffix}"
        self._column_widget_counter[candidate] = 1
        return candidate

    # -- selection ----------------------------------------------------------------

    def select_card(self, card_id: int) -> None:
        if self.visible_snapshot is None:
            return
        for status in self.visible_snapshot.statuses:
            for row, card in enumerate(self.visible_snapshot.columns.get(status, ())):
                if card.id == card_id:
                    self.selection = SelectionState(
                        card_id=card.id, status=status, row=row
                    )
                    self._render_selection()
                    return

    def _selected_card(self) -> CardView | None:
        if self.visible_snapshot is None:
            return None
        return _selected_card(self.visible_snapshot, self.selection.card_id)

    # -- actions --------------------------------------------------------------------

    def action_cursor_left(self) -> None:
        self._move_columns(-1)

    def action_cursor_right(self) -> None:
        self._move_columns(1)

    def action_cursor_up(self) -> None:
        self._move_rows(-1)

    def action_cursor_down(self) -> None:
        self._move_rows(1)

    def _move_columns(self, delta: int) -> None:
        visible = self.visible_snapshot
        if visible is None or not visible.statuses:
            return
        selection = next_column_selection(visible, self.selection, delta)
        if selection is not None:
            self.selection = selection
            self._render_selection()

    def _move_rows(self, delta: int) -> None:
        visible = self.visible_snapshot
        if visible is None:
            return
        selection = moved_row_selection(visible, self.selection, delta)
        if selection is not None and selection != self.selection:
            self.selection = selection
            self._render_selection()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        focused = self.app.focused
        navigation = action.startswith("cursor_") or action.startswith("move_card_")
        if navigation and isinstance(focused, (Input, CardInspector)):
            return False
        return super().check_action(action, parameters)

    def action_focus_board(self) -> None:
        self.query_one("#kanban-board-columns").focus()

    def action_focus_inspector(self) -> None:
        self.query_one(CardInspector).focus()

    def action_focus_filter(self) -> None:
        self.query_one("#kanban-filter-text", Input).focus()

    def action_refresh_board(self) -> None:
        self.refresh_board()

    def action_move_card_left(self) -> None:
        self._move_selected_card(-1)

    def action_move_card_right(self) -> None:
        self._move_selected_card(1)

    def action_new_card(self) -> None:
        if self.snapshot is None:
            return
        if self.snapshot.board_id is None:
            self._bootstrap_first_board()
            return
        status = self.selection.status or (
            self.visible_snapshot.statuses[0]
            if self.visible_snapshot and self.visible_snapshot.statuses
            else None
        )
        if status is None:
            self.notify(
                "This board has no columns yet (v1 columns come from the "
                "board's lists; column management is future work).",
                severity="warning",
            )
            return
        # Bind the RESOLVED status into the callback: re-reading
        # ``self.selection`` on dismiss is wrong on a card-less board,
        # where selection carries no status at all (the modal's own title
        # is the contract the user saw).
        self.app.push_screen(
            NewCardModal(status),
            lambda title: self._create_card_in_status(status, title),
        )

    def _create_card_in_status(self, status: str, title: str | None) -> None:
        if not title:
            return
        list_id = self._list_ids_by_status.get(status)
        if list_id is None:
            return
        self._run_mutation(self._create_card_operation(list_id, title), title)

    async def _create_card_operation(self, list_id: int, title: str) -> None:
        service = self.app_instance.local_kanban_service
        if service is not None:
            await service.create_card(
                list_id, {"title": title, "client_id": str(uuid4())}
            )

    def _move_selected_card(self, delta: int) -> None:
        card = self._selected_card()
        visible = self.visible_snapshot
        if card is None or visible is None or not visible.statuses:
            return
        try:
            current_index = visible.statuses.index(card.status)
        except ValueError:
            return
        target_index = current_index + delta
        if target_index < 0 or target_index >= len(visible.statuses):
            return
        target_status = visible.statuses[target_index]
        target_list_id = self._list_ids_by_status.get(target_status)
        if target_list_id is None:
            return
        self._run_mutation(self._move_card_operation(card.id, target_list_id), card.id)

    async def _move_card_operation(self, card_id: int, target_list_id: int) -> None:
        service = self.app_instance.local_kanban_service
        if service is not None:
            await service.move_card(card_id, {"target_list_id": target_list_id})

    def _bootstrap_first_board(self) -> None:
        """User-initiated (n on an empty store): create the default board.

        Re-entrancy-guarded: until the post-mutation refresh lands, the
        snapshot still reports no board, so an impatient double-tap (or
        key repeat) would otherwise create a second "Personal" board.
        """
        if self._bootstrap_in_flight:
            return
        self._bootstrap_in_flight = True
        self._run_mutation(self._bootstrap_first_board_operation(), None)

    async def _bootstrap_first_board_operation(self) -> None:
        service = self.app_instance.local_kanban_service
        if service is None:
            return
        board = await service.create_board(
            {"name": _DEFAULT_BOARD_NAME, "client_id": str(uuid4())}
        )
        board_id = int(board["id"])
        for position, name in enumerate(_DEFAULT_BOARD_LISTS):
            await service.create_list(
                board_id,
                {"name": name, "position": position, "client_id": str(uuid4())},
            )

    def _run_mutation(self, operation: Any, reselect_hint: Any) -> None:
        self._run_mutation_worker(operation, reselect_hint)

    @work(exclusive=False, group="kanban-board-mutation")
    async def _run_mutation_worker(self, operation: Any, reselect_hint: Any) -> None:
        try:
            await operation
        except Exception as exc:  # noqa: BLE001 — mutations toast, never crash
            self.notify(f"Kanban change failed: {exc}", severity="error", markup=False)
            # A failed mutation triggers no refresh, so any in-flight
            # bootstrap guard would otherwise stick forever.
            self._bootstrap_in_flight = False
            return
        if isinstance(reselect_hint, int):
            self._reselect_card_id = reselect_hint
        self.refresh_board()

    # -- filter ----------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "kanban-filter-text":
            return
        self.filter_state = FilterState(text=event.value)
        if self.snapshot is None:
            return
        previous = self.visible_snapshot or self.snapshot
        self.visible_snapshot = filter_snapshot(self.snapshot, self.filter_state)
        self.selection = select_after_refresh(
            previous, self.visible_snapshot, self.selection
        )
        self.call_later(self._render_board)


# -- pure movement helpers (tested) ---------------------------------------------


def next_column_selection(
    snapshot: BoardSnapshot, selection: SelectionState, delta: int
) -> SelectionState | None:
    """The selection after moving one column in ``delta`` direction.

    Empty columns are skipped (upstream's behavior): the cursor lands on
    the nearest column in that direction that holds a card, keeping the
    current row when it fits.
    """
    if not snapshot.statuses:
        return None
    current_status = selection.status or snapshot.statuses[0]
    try:
        current_index = snapshot.statuses.index(current_status)
    except ValueError:
        current_index = 0
    ordered = range(
        current_index + delta,
        len(snapshot.statuses) if delta > 0 else -1,
        delta,
    )
    for index in ordered:
        cards = snapshot.columns.get(snapshot.statuses[index], ())
        if cards:
            row = max(0, min(selection.row, len(cards) - 1))
            return SelectionState(
                card_id=cards[row].id, status=snapshot.statuses[index], row=row
            )
    return None


def moved_row_selection(
    snapshot: BoardSnapshot, selection: SelectionState, delta: int
) -> SelectionState | None:
    """The selection after moving one row within the current column."""
    if selection.status is None:
        return None
    cards = snapshot.columns.get(selection.status, ())
    if not cards:
        return None
    row = max(0, min(selection.row + delta, len(cards) - 1))
    return SelectionState(card_id=cards[row].id, status=selection.status, row=row)


def _selected_card(snapshot: BoardSnapshot, card_id: int | None) -> CardView | None:
    if card_id is None:
        return None
    for cards in snapshot.columns.values():
        for card in cards:
            if card.id == card_id:
                return card
    return None
