"""Kanban board view-models, widgets, and screen (TASK-18934).

Pins the ported-from-backlog-md-py pure logic: filtering, selection
preservation across refreshes, and column/row movement math — plus the
screen's registration contract (shell destination, lockstep shortcut,
route resolution).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import App, ComposeResult

from Tests.private_profile import private_profile_test
from tldw_chatbook.UI.Kanban.board_models import (
    BoardSnapshot,
    CardView,
    FilterState,
    SelectionState,
    filter_snapshot,
    select_after_refresh,
)
from tldw_chatbook.UI.Kanban.kanban_board_screen import (
    KanbanBoardScreen,
    SnapshotLoader,
)


def _card(card_id: str, title: str = "Card", description: str = "") -> CardView:
    return CardView(
        id=card_id, list_id=0, title=title, status="", description=description
    )


def _snapshot(
    columns: dict[str, tuple[CardView, ...]],
    *,
    board_name: str = "Personal",
    list_ids: dict[str, int] | None = None,
) -> BoardSnapshot:
    # Match the real loader's contract: a card's ``status`` IS its column.
    normalized = {
        status: tuple(replace(card, status=status) for card in cards)
        for status, cards in columns.items()
    }
    return BoardSnapshot(
        board_id=1,
        board_name=board_name,
        statuses=tuple(normalized),
        columns=normalized,
        list_ids=list_ids or {},
    )


class TestFilterSnapshot:
    def test_text_filter_matches_id_title_and_description(self) -> None:
        snapshot = _snapshot(
            {
                "To Do": (
                    _card("task-1", "Alpha search"),
                    _card("task-2", "Beta"),
                    _card("task-3", "Gamma", description="mentions search term"),
                )
            }
        )

        visible = filter_snapshot(snapshot, FilterState(text="search"))

        assert [card.id for card in visible.columns["To Do"]] == ["task-1", "task-3"]

    def test_empty_filter_returns_everything(self) -> None:
        snapshot = _snapshot({"To Do": (_card("task-1"),)})

        visible = filter_snapshot(snapshot, FilterState())

        assert visible.columns == snapshot.columns


class TestSelectAfterRefresh:
    def test_selected_task_is_preserved_when_it_survives(self) -> None:
        before = _snapshot({"To Do": (_card("task-1"), _card("task-2"))})
        after = _snapshot({"To Do": (_card("task-2"), _card("task-3"))})

        selection = select_after_refresh(
            before, after, SelectionState(card_id="task-2", status="To Do", row=0)
        )

        assert selection == SelectionState(card_id="task-2", status="To Do", row=0)

    def test_vanished_task_falls_back_to_same_row_same_column(self) -> None:
        before = _snapshot({"To Do": (_card("task-1"), _card("task-2"))})
        after = _snapshot({"To Do": (_card("task-9", "a"), _card("task-8", "b"))})

        selection = select_after_refresh(
            before, after, SelectionState(card_id="task-2", status="To Do", row=1)
        )

        assert selection.card_id == "task-8"
        assert selection.row == 1

    def test_vanished_column_walks_forward_then_backward(self) -> None:
        before = _snapshot(
            {"To Do": (_card("task-1"),), "Doing": (), "Done": (_card("task-9"),)}
        )
        after = _snapshot({"Other": (_card("task-5"),), "Doing": (), "Done": (_card("task-9"),)})

        selection = select_after_refresh(
            before, after, SelectionState(card_id="task-1", status="To Do", row=0)
        )

        # "To Do" is gone; the walk finds the next column WITH a card.
        assert selection.card_id == "task-9"
        assert selection.status == "Done"


class TestColumnRowMath:
    def test_column_movement_skips_empty_columns(self) -> None:
        from tldw_chatbook.UI.Kanban.kanban_board_screen import next_column_selection

        snapshot = _snapshot(
            {
                "To Do": (_card("task-1"),),
                "Doing": (),
                "Done": (_card("task-9"),),
            }
        )

        selection = next_column_selection(
            snapshot, SelectionState(card_id="task-1", status="To Do", row=0), +1
        )

        assert selection is not None
        assert selection.status == "Done"
        assert selection.card_id == "task-9"

    def test_row_movement_clamps_to_column_bounds(self) -> None:
        from tldw_chatbook.UI.Kanban.kanban_board_screen import (
            moved_row_selection,
        )

        snapshot = _snapshot({"To Do": (_card("task-1"), _card("task-2"))})

        down = moved_row_selection(
            snapshot, SelectionState(card_id="task-1", status="To Do", row=0), +1
        )
        clamped = moved_row_selection(snapshot, down, +5)

        assert down.card_id == "task-2"
        assert clamped.card_id == "task-2"
        assert clamped.row == 1


class TestRegistration:
    def test_kanban_destination_is_registered_in_lockstep(self) -> None:
        from tldw_chatbook.UI.Navigation.shell_destinations import (
            SHELL_DESTINATION_ORDER,
            SHELL_DESTINATION_SHORTCUTS,
        )

        ids = [destination.destination_id for destination in SHELL_DESTINATION_ORDER]
        assert "kanban" in ids
        kanban_position = ids.index("kanban")
        # Appended last: no existing destination's strip position moved.
        assert kanban_position == len(ids) - 1
        assert SHELL_DESTINATION_SHORTCUTS["kanban"] == "f8"
        # Lockstep: every destination has a shortcut and vice versa.
        assert set(ids) == set(SHELL_DESTINATION_SHORTCUTS)

    def test_route_resolves_to_the_kanban_screen_class(self) -> None:
        from tldw_chatbook.UI.Navigation.screen_registry import _SCREEN_ROUTES

        route = _SCREEN_ROUTES.get("kanban")
        assert route is not None
        assert route.module_path == "tldw_chatbook.UI.Kanban.kanban_board_screen"
        assert route.class_name == "KanbanBoardScreen"

    def test_tab_constant_exists_and_matches_route(self) -> None:
        from tldw_chatbook.Constants import TAB_KANBAN

        assert TAB_KANBAN == "kanban"


class _KanbanHarness(App[None]):
    """Mounts the board screen with an injected snapshot loader."""

    def __init__(self, loader: SnapshotLoader) -> None:
        super().__init__()
        self._loader = loader

    def compose(self) -> ComposeResult:
        screen = KanbanBoardScreen(
            app_instance=_FakeAppInstance(), screen_name="kanban"
        )
        screen.snapshot_loader = self._loader
        yield screen


class _FakeAppInstance:
    """Minimal app stand-in: no kanban service (loader is injected)."""

    local_kanban_service = None


@pytest.mark.asyncio
@private_profile_test
async def test_board_screen_renders_columns_and_navigates_by_keyboard(
    request,
) -> None:
    """AC#2/AC#6: keyboard-first navigation with no mouse dependency —
    rows move within a column, column moves skip empty columns, and the
    inspector tracks the selection."""
    from textual.widgets import Static

    board = _snapshot(
        {
            "To Do": (_card(1, "first"), _card(2, "second")),
            "Doing": (),
            "Done": (_card(3, "Done card"),),
        }
    )

    async def loader() -> BoardSnapshot:
        return board

    app = _KanbanHarness(loader)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.1)
        screen = app.query_one(KanbanBoardScreen)
        # Wait for the async load worker to apply its snapshot.
        for _ in range(50):
            if screen.visible_snapshot is not None:
                break
            await pilot.pause(0.05)
        assert screen.visible_snapshot is not None
        columns = app.query(".kanban-board-column")
        assert len(columns) == 3

        # Selection starts on the first card of the first column.
        assert screen.selection.card_id == 1

        screen.action_cursor_down()
        await pilot.pause()
        assert screen.selection.card_id == 2

        # "Doing" is empty: moving right skips it to "Done".
        screen.action_cursor_right()
        await pilot.pause()
        await pilot.pause()
        assert screen.selection.status == "Done"
        assert screen.selection.card_id == 3

        inspector_text = str(
            app.query_one("#kanban-card-inspector", Static).renderable
        )
        assert "Done card" in inspector_text


@pytest.mark.asyncio
@private_profile_test
async def test_board_screen_moves_card_to_adjacent_column_via_store(
    request,
) -> None:
    """shift+h/shift+l move the selected card through the store."""
    moves: list[tuple[int, dict]] = []
    board = _snapshot(
        {
            "To Do": (_card(1, "first"),),
            "Doing": (_card(2, "second"),),
        },
        list_ids={"To Do": 70, "Doing": 71},
    )

    class _RecordingService:
        async def move_card(self, card_id: int, request_data: dict) -> dict:
            moves.append((card_id, dict(request_data)))
            return {"id": card_id}

        async def list_boards(self) -> dict:
            return {"boards": [{"id": 7, "name": "Board"}]}

        async def list_lists(self, board_id: int) -> dict:
            return {
                "lists": [
                    {"id": 70, "name": "To Do", "position": 0},
                    {"id": 71, "name": "Doing", "position": 1},
                ]
            }

        async def list_cards(self, list_id: int) -> dict:
            if list_id == 70:
                return {"cards": [{"id": 1, "list_id": 70, "title": "first"}]}
            return {"cards": [{"id": 2, "list_id": 71, "title": "second"}]}

    service = _RecordingService()

    async def loader() -> BoardSnapshot:
        return board

    app = _KanbanHarness(loader)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.1)
        screen = app.query_one(KanbanBoardScreen)
        for _ in range(50):
            if screen.visible_snapshot is not None:
                break
            await pilot.pause(0.05)
        # Swap in the recording service for the mutation path.
        screen.app_instance.local_kanban_service = service
        assert screen.selection.card_id == 1

        screen.action_move_card_right()
        await pilot.pause(0.2)

        assert moves == [(1, {"target_list_id": 71})]
