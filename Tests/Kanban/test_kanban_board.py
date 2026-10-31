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
from textual.widgets import Input

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp
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
        after = _snapshot(
            {"Other": (_card("task-5"),), "Doing": (), "Done": (_card("task-9"),)}
        )

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
        assert SHELL_DESTINATION_SHORTCUTS["kanban"] == "f9"
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


class _KanbanHarness(ConsolidatedCSSApp[None]):
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

        inspector_text = str(app.query_one("#kanban-card-inspector", Static).renderable)
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


@pytest.mark.asyncio
@private_profile_test
async def test_new_card_flow_bootstraps_and_creates_through_the_store(request) -> None:
    """The honest-empty-start story end to end: on an empty store `n`
    bootstraps exactly one board (double-tap guarded), the next `n` opens
    the modal in the focused column, typing + Enter creates the card
    through the store, and the refreshed board shows it."""
    created_boards: list[dict] = []
    created_lists: list[tuple[int, dict]] = []
    created_cards: list[tuple[int, dict]] = []

    class _RecordingService:
        async def create_board(self, request_data: dict) -> dict:
            created_boards.append(dict(request_data))
            return {"id": 7, "name": request_data.get("name")}

        async def create_list(self, board_id: int, request_data: dict) -> dict:
            created_lists.append((board_id, dict(request_data)))
            return {"id": 70 + len(created_lists), "name": request_data.get("name")}

        async def create_card(self, list_id: int, request_data: dict) -> dict:
            created_cards.append((list_id, dict(request_data)))
            return {"id": 500 + len(created_cards), "title": request_data.get("title")}

    service = _RecordingService()

    def _snapshot() -> BoardSnapshot:
        if not created_boards:
            return BoardSnapshot(
                board_id=None, board_name="Kanban", statuses=(), columns={}
            )
        names = ("To Do", "Doing", "Done")
        list_ids = {name: 70 + index for index, name in enumerate(names)}
        columns = {name: () for name in names}
        todo_cards = list(columns["To Do"])
        for index, (list_id, data) in enumerate(created_cards):
            todo_cards.append(
                CardView(
                    id=500 + index,
                    list_id=list_id,
                    title=str(data.get("title") or ""),
                    status="To Do",
                )
            )
        columns["To Do"] = tuple(todo_cards)
        return BoardSnapshot(
            board_id=7,
            board_name="Personal",
            statuses=names,
            columns=columns,
            list_ids=list_ids,
        )

    async def loader() -> BoardSnapshot:
        return _snapshot()

    app = _KanbanHarness(loader)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.1)
        screen = app.query_one(KanbanBoardScreen)
        for _ in range(50):
            if screen.visible_snapshot is not None:
                break
            await pilot.pause(0.05)
        screen.app_instance.local_kanban_service = service

        # Empty store: n bootstraps; a double-tap must not create two boards.
        screen.action_new_card()
        screen.action_new_card()
        await pilot.pause(0.3)
        assert len(created_boards) == 1
        assert created_boards[0]["name"] == "Personal"
        assert created_boards[0]["client_id"]
        assert [data["name"] for _board, data in created_lists] == [
            "To Do",
            "Doing",
            "Done",
        ]
        # The post-bootstrap refresh landed the new board.
        for _ in range(50):
            if screen.visible_snapshot and screen.visible_snapshot.statuses:
                break
            await pilot.pause(0.05)
        assert screen.visible_snapshot is not None
        assert screen.visible_snapshot.statuses == ("To Do", "Doing", "Done")

        # Populated board: n opens the modal focused on the first column.
        screen.action_new_card()
        await pilot.pause(0.2)
        # The modal is the current screen now; query through it (an
        # app-level query_one resolves against the default screen).
        from tldw_chatbook.UI.Kanban.kanban_board_screen import NewCardModal

        modal = app.screen
        assert isinstance(modal, NewCardModal)
        assert modal.column_name == "To Do"
        modal.query_one("#kanban-new-card-input", Input).focus()
        await pilot.pause(0.1)
        for character in "Hello":
            await pilot.press(character)
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert len(created_cards) == 1
        assert created_cards[0][0] == 70
        assert created_cards[0][1]["title"] == "Hello"
        assert created_cards[0][1]["client_id"]


@pytest.mark.asyncio
async def test_real_store_columns_keep_cards_when_names_and_suffixes_collide(tmp_path):
    from tldw_chatbook.Kanban_Interop.local_kanban_service import LocalKanbanService

    service = LocalKanbanService(db_path=tmp_path / "kanban.db")
    board = await service.create_board(
        {"name": "Personal", "client_id": "board-control"}
    )
    stored_lists = []
    stored_cards = []
    for position, name in enumerate(
        ("Done", "Done", "Done (2)", "Done (3)", "Done (2)")
    ):
        column = await service.create_list(
            board["id"],
            {"name": name, "position": position, "client_id": f"list-{position}"},
        )
        stored_lists.append(column)
        stored_cards.append(
            await service.create_card(
                column["id"],
                {"title": f"Card {position}", "client_id": f"card-{position}"},
            )
        )
    app_instance = _FakeAppInstance()
    app_instance.local_kanban_service = service
    screen = KanbanBoardScreen(app_instance)

    snapshot = await screen._default_snapshot_loader()

    assert len(snapshot.statuses) == len(stored_lists)
    assert list(snapshot.list_ids.values()) == [column["id"] for column in stored_lists]
    assert [snapshot.columns[label][0].id for label in snapshot.statuses] == [
        card["id"] for card in stored_cards
    ]
    # A generated label cannot take a literal name owned by a later list.
    assert snapshot.list_ids["Done (2)"] == stored_lists[2]["id"]
    assert snapshot.list_ids["Done (3)"] == stored_lists[3]["id"]
    destination = snapshot.list_ids[snapshot.statuses[1]]
    await screen._move_card_operation(stored_cards[0]["id"], destination)
    await screen._create_card_operation(destination, "New in duplicate column")
    refreshed = await screen._default_snapshot_loader()
    assert {card.title for card in refreshed.columns[refreshed.statuses[1]]} == {
        "Card 0",
        "Card 1",
        "New in duplicate column",
    }


@pytest.mark.asyncio
@private_profile_test
async def test_new_card_modal_renders_the_column_name_literally(request):
    from textual.widgets import Static
    from tldw_chatbook.UI.Kanban.kanban_board_screen import NewCardModal

    app = App()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(NewCardModal("[bold]Done[/bold]"))
        await pilot.pause()
        label = app.screen.query_one("#kanban-new-card-label", Static)
        assert label.visual.plain == "New card in [bold]Done[/bold]"


@pytest.mark.asyncio
@private_profile_test
async def test_keyboard_bootstrap_and_create_use_real_local_store(request, tmp_path):
    from tldw_chatbook.Kanban_Interop.local_kanban_service import LocalKanbanService
    from tldw_chatbook.UI.Kanban.kanban_board_screen import NewCardModal

    service = LocalKanbanService(db_path=tmp_path / "kanban.db")

    class RealKanbanApp(ConsolidatedCSSApp[None]):
        def compose(self):
            app_instance = _FakeAppInstance()
            app_instance.local_kanban_service = service
            yield KanbanBoardScreen(app_instance)

    app = RealKanbanApp()
    async with app.run_test(size=(120, 30)) as pilot:
        screen = app.query_one(KanbanBoardScreen)
        await screen.workers.wait_for_complete()
        await pilot.press("n")
        await screen.workers.wait_for_complete()
        await pilot.pause()
        boards = (await service.list_boards())["boards"]
        assert len(boards) == 1
        columns = (await service.list_lists(boards[0]["id"]))["lists"]
        assert [column["name"] for column in columns] == ["To Do", "Doing", "Done"]
        await pilot.press("n")
        await pilot.pause()
        assert isinstance(app.screen, NewCardModal)
        app.screen.query_one(Input).value = "Real card"
        await pilot.press("enter")
        await screen.workers.wait_for_complete()
        await pilot.pause()
        cards = (await service.list_cards(columns[0]["id"]))["cards"]
        assert [card["title"] for card in cards] == ["Real card"]
        client_ids = [
            boards[0]["client_id"],
            *(column["client_id"] for column in columns),
            cards[0]["client_id"],
        ]
        assert len(set(client_ids)) == 5
