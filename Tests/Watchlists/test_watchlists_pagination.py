"""Transactional pagination regressions for the Watchlists Read screen."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.article_list import (
    ArticleListPane,
    NextItemsPageRequested,
    PreviousItemsPageRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemSelected


def _item(index: int, *, day: int = 13) -> dict[str, object]:
    return {
        "id": str(index),
        "title": f"Item {index}",
        "source_name": "Pagination feed",
        "status": "new",
        "created_at": f"2026-08-{day:02d}T12:{index % 60:02d}:00+00:00",
        "content": f"Body {index}",
    }


def _items(start: int, count: int, *, day: int = 13) -> list[dict[str, object]]:
    return [_item(index, day=day) for index in range(start, start + count)]


@asynccontextmanager
async def _open_screen(controller: AsyncMock):
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        screen._controller = controller
        yield screen, pilot


async def _wait_until(pilot, predicate: Callable[[], bool]) -> None:
    for _ in range(80):
        if predicate():
            return
        await pilot.pause(0.025)
    assert predicate(), "condition did not become true"


@pytest.mark.asyncio
async def test_first_page_requests_lookahead_but_mounts_only_fifty():
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 51)

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._load_items() is True

        controller.list_items.assert_awaited_once_with(
            runtime_backend="local",
            limit=51,
            offset=0,
            statuses=["new", "reviewed", "ingested"],
        )
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert len(screen._loaded_items) == 50
        assert len(pane.items) == 50
        assert "50" not in {str(item["id"]) for item in screen._loaded_items}
        assert "50" not in {str(item["id"]) for item in pane.items}
        assert screen._items_page_index == 0
        assert screen._items_has_next is True
        assert pane.page_number == 1
        assert pane.has_previous is False
        assert pane.has_next is True


@pytest.mark.asyncio
async def test_first_page_without_lookahead_disables_next():
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 50)

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._load_items() is True

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert len(screen._loaded_items) == 50
        assert screen._items_has_next is False
        assert pane.has_next is False


@pytest.mark.asyncio
async def test_next_commits_offset_fifty_only_after_success_and_keeps_content():
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 51)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        open_item = screen._loaded_items[0]
        screen._selected_content_item = open_item
        screen._selected_content_page_key = screen._items_committed_page_key
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item

        entered = asyncio.Event()
        result: asyncio.Future[list[dict[str, object]]] = (
            asyncio.get_running_loop().create_future()
        )

        async def pending_page(**kwargs):
            assert kwargs["offset"] == 50
            entered.set()
            return await result

        controller.list_items.reset_mock()
        controller.list_items.side_effect = pending_page
        screen.post_message(NextItemsPageRequested())
        await _wait_until(pilot, entered.is_set)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert screen._items_page_index == 0
        assert pane.page_number == 1
        assert pane.page_loading is True
        assert screen._loaded_items[0] is open_item
        assert content.item is open_item

        result.set_result(_items(100, 51))
        await _wait_until(pilot, lambda: screen._items_page_index == 1)

        controller.list_items.assert_awaited_once()
        assert controller.list_items.await_args.kwargs["offset"] == 50
        assert len(screen._loaded_items) == 50
        assert len(pane.items) == 50
        assert pane.page_number == 2
        assert pane.has_previous is True
        assert pane.has_next is True
        assert pane.page_loading is False
        assert screen._selected_content_item is open_item
        assert content.item is open_item
        assert str(open_item["id"]) not in {
            str(item["id"]) for item in screen._loaded_items
        }


@pytest.mark.asyncio
async def test_previous_returns_to_offset_zero():
    controller = AsyncMock()

    async with _open_screen(controller) as (screen, pilot):
        controller.list_items.return_value = _items(100, 51)
        assert await screen._load_items(
            target_page_index=1, explicit_page_change=True
        ) is True

        controller.list_items.reset_mock()
        controller.list_items.return_value = _items(0, 50)
        screen.post_message(PreviousItemsPageRequested())
        await _wait_until(pilot, lambda: screen._items_page_index == 0)

        controller.list_items.assert_awaited_once()
        assert controller.list_items.await_args.kwargs["offset"] == 0
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert pane.page_number == 1
        assert pane.has_previous is False
        assert pane.has_next is False


@pytest.mark.asyncio
async def test_failed_explicit_transition_preserves_committed_reader_state():
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 51)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        prior_rows = screen._loaded_items
        open_item = prior_rows[0]
        screen._selected_content_item = open_item
        screen._selected_content_page_key = screen._items_committed_page_key
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item

        async def fail_next(**kwargs):
            assert kwargs["offset"] == 50
            raise RuntimeError("page unavailable")

        controller.list_items.reset_mock()
        controller.list_items.side_effect = fail_next
        screen.post_message(NextItemsPageRequested())
        await _wait_until(
            pilot,
            lambda: controller.list_items.await_count == 1
            and screen._items_page_loading is False,
        )

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert screen._loaded_items is prior_rows
        assert pane.items is prior_rows
        assert screen._items_page_index == 0
        assert screen._items_has_next is True
        assert pane.page_number == 1
        assert pane.has_next is True
        assert pane.page_loading is False
        assert screen._selected_content_item is open_item
        assert content.item is open_item


@pytest.mark.asyncio
async def test_repeated_next_while_loading_starts_only_one_request():
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 51)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        entered = asyncio.Event()
        release = asyncio.Event()

        async def pending_next(**kwargs):
            assert kwargs["offset"] == 50
            entered.set()
            await release.wait()
            return _items(100, 50)

        controller.list_items.reset_mock()
        controller.list_items.side_effect = pending_next
        screen.post_message(NextItemsPageRequested())
        await _wait_until(pilot, entered.is_set)
        screen.post_message(NextItemsPageRequested())
        await pilot.pause(0.2)

        assert controller.list_items.await_count == 1
        release.set()
        await _wait_until(pilot, lambda: screen._items_page_index == 1)


@pytest.mark.asyncio
async def test_same_page_refresh_pins_open_item_without_exceeding_fifty():
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 51, day=13)

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._load_items() is True
        open_item = dict(screen._loaded_items[-1])
        open_item["created_at"] = "2026-08-01T00:00:00+00:00"
        screen._selected_content_item = open_item
        screen._selected_content_page_key = screen._items_committed_page_key

        controller.list_items.return_value = _items(100, 50, day=14)
        assert await screen._load_items() is True

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        loaded_ids = [str(item["id"]) for item in screen._loaded_items]
        assert len(loaded_ids) == 50
        assert str(open_item["id"]) in loaded_ids
        assert len(pane.items) == 50
        assert str(open_item["id"]) in {str(item["id"]) for item in pane.items}


@pytest.mark.asyncio
async def test_selection_records_the_page_that_was_committed_before_detail_fetch():
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 1)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        selection_key = screen._items_committed_page_key
        item = screen._loaded_items[0]
        detail_started = asyncio.Event()
        detail_result: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def pending_content(**_kwargs):
            detail_started.set()
            return await detail_result

        controller.get_item_content.side_effect = pending_content
        screen._mark_item_read_on_open = Mock()
        task = asyncio.create_task(screen.handle_item_selected(ItemSelected(item)))
        await _wait_until(pilot, detail_started.is_set)
        screen._items_committed_page_key = ("different",)
        detail_result.set_result("Fetched body")
        await task

        assert screen._selected_content_item is item
        assert screen._selected_content_page_key == selection_key
