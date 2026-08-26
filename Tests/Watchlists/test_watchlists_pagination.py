"""Transactional cached-snapshot pagination regressions for Watchlists Read."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.Subscriptions.watchlist_item_page import (
    WatchlistItemCursor,
    WatchlistItemPage,
)
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.article_list import (
    ArticleListPane,
    NextItemsPageRequested,
    PreviousItemsPageRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
from tldw_chatbook.UI.Watchlists_Modules.items_pane import (
    ItemSelected,
    ItemsFilterChanged,
    RefreshItemsRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.reader_item_snapshot import (
    ReaderItemSnapshot,
)
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope


def _item(index: int, *, day: int = 13, status: str = "new") -> dict[str, object]:
    effective = f"2026-08-{day:02d} 12:{index % 60:02d}:00"
    return {
        "id": f"local:watchlist_item:{index}",
        "item_id": index,
        "title": f"Item {index}",
        "source_name": "Pagination feed",
        "status": status,
        "created_at": effective,
        "effective_date": effective,
        "content": f"Body {index}",
    }


def _items(ids: Iterable[int], *, day: int = 13) -> tuple[dict[str, object], ...]:
    return tuple(_item(index, day=day) for index in ids)


def _page(
    ids: Iterable[int],
    *,
    high_water: int,
    snapshot_count: int | None = None,
    has_more: bool = False,
    cursor: WatchlistItemCursor | None = None,
) -> WatchlistItemPage:
    rows = _items(ids)
    if cursor is None and has_more and rows:
        last = rows[-1]
        cursor = WatchlistItemCursor(
            str(last["effective_date"]), int(last["item_id"])
        )
    return WatchlistItemPage(
        items=rows,
        has_more=has_more,
        snapshot_max_item_id=high_water,
        snapshot_count=snapshot_count,
        next_cursor=cursor,
    )


@asynccontextmanager
async def _open_screen(controller: AsyncMock):
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        controller.get_overview_data.return_value = {}
        screen._controller = controller
        yield screen, pilot


async def _wait_until(pilot, predicate: Callable[[], bool]) -> None:
    for _ in range(80):
        if predicate():
            return
        await pilot.pause(0.025)
    assert predicate(), "condition did not become true"


@pytest.mark.asyncio
async def test_first_page_uses_typed_reader_query_without_cursor_or_watermark():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        range(50, 0, -1), high_water=50, snapshot_count=50
    )

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True

        controller.list_reader_items_page.assert_awaited_once_with(
            runtime_backend="local",
            limit=50,
            statuses=["new", "reviewed", "ingested"],
        )
        assert controller.list_items.await_count == 0
        assert isinstance(screen._items_snapshot, ReaderItemSnapshot)
        assert screen._items_snapshot.watermark == 50
        assert screen._items_snapshot.snapshot_count == 50
        assert [row["item_id"] for row in screen._loaded_items] == list(
            range(50, 0, -1)
        )


@pytest.mark.asyncio
async def test_next_uses_committed_watermark_and_cursor_and_deduplicates():
    controller = AsyncMock()
    first_cursor = WatchlistItemCursor("2026-08-13 12:01:00", 1)
    controller.list_reader_items_page.side_effect = [
        _page(
            range(50, 0, -1),
            high_water=50,
            snapshot_count=51,
            has_more=True,
            cursor=first_cursor,
        ),
        _page([1, 0], high_water=50, has_more=False),
    ]

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        assert await screen._load_next_items_page() is True

        assert controller.list_reader_items_page.await_args_list[1].kwargs == {
            "runtime_backend": "local",
            "limit": 50,
            "statuses": ["new", "reviewed", "ingested"],
            "snapshot_max_item_id": 50,
            "after": first_cursor,
        }
        assert [row["item_id"] for row in screen._loaded_items] == [0]
        assert screen._items_snapshot.page_count == 2
        assert sum(
            row["item_id"] == 1
            for page in screen._items_snapshot.pages
            for row in page
        ) == 1


@pytest.mark.asyncio
async def test_previous_and_cached_forward_present_without_io():
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([4, 3], high_water=4, snapshot_count=4, has_more=True),
        _page([2, 1], high_water=4),
    ]

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        assert await screen._load_next_items_page() is True
        controller.list_reader_items_page.reset_mock()

        screen.post_message(PreviousItemsPageRequested())
        await _wait_until(pilot, lambda: screen._items_page_index == 0)
        screen.post_message(NextItemsPageRequested())
        await _wait_until(pilot, lambda: screen._items_page_index == 1)

        assert controller.list_reader_items_page.await_count == 0
        assert [row["item_id"] for row in screen._loaded_items] == [2, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["failure", "cancel"])
async def test_interrupted_next_keeps_committed_page_reader_and_snapshot(interruption):
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4, 3], high_water=4, snapshot_count=4, has_more=True
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        prior_snapshot = screen._items_snapshot
        prior_rows = screen._loaded_items
        open_item = prior_rows[0]
        screen._selected_content_item = open_item
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item
        entered = asyncio.Event()
        pending: asyncio.Future[WatchlistItemPage] = (
            asyncio.get_running_loop().create_future()
        )

        async def next_page(**_kwargs):
            entered.set()
            return await pending

        controller.list_reader_items_page.reset_mock()
        controller.list_reader_items_page.side_effect = next_page
        task = asyncio.create_task(screen._load_next_items_page())
        await _wait_until(pilot, entered.is_set)
        if interruption == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            pending.set_exception(RuntimeError("unavailable"))
            assert await task is False

        assert screen._items_snapshot is prior_snapshot
        assert screen._loaded_items is prior_rows
        assert screen._items_page_index == 0
        assert screen._selected_content_item is open_item
        assert content.item is open_item


@pytest.mark.asyncio
async def test_failed_next_presentation_rolls_back_rows_and_snapshot(monkeypatch):
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([4, 3], high_water=4, snapshot_count=4, has_more=True),
        _page([2, 1], high_water=4),
    ]

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        prior_snapshot = screen._items_snapshot
        prior_rows = screen._loaded_items
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        original_apply = pane.apply_page_items

        async def apply_then_fail(items, *, focus_first=False):
            await original_apply(items, focus_first=focus_first)
            if items and items[0]["item_id"] == 2:
                raise RuntimeError("presentation failed")

        monkeypatch.setattr(pane, "apply_page_items", apply_then_fail)

        assert await screen._load_next_items_page() is False
        assert screen._items_snapshot is prior_snapshot
        assert screen._loaded_items is prior_rows
        assert pane.items is prior_rows
        assert screen._items_page_index == 0


@pytest.mark.asyncio
async def test_repeated_next_while_loading_coalesces_to_one_request():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4, 3], high_water=4, snapshot_count=4, has_more=True
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        entered = asyncio.Event()
        release = asyncio.Event()

        async def pending(**_kwargs):
            entered.set()
            await release.wait()
            return _page([2, 1], high_water=4)

        controller.list_reader_items_page.reset_mock()
        controller.list_reader_items_page.side_effect = pending
        first = asyncio.create_task(screen._load_next_items_page())
        await _wait_until(pilot, entered.is_set)
        second = asyncio.create_task(screen._load_next_items_page())
        await pilot.pause(0.1)
        assert controller.list_reader_items_page.await_count == 1
        release.set()
        assert await first is True
        assert await second is True


@pytest.mark.asyncio
async def test_duplicate_only_continuations_are_bounded_and_not_cached_blank():
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([4], high_water=4, snapshot_count=2, has_more=True),
        _page([4], high_water=4, has_more=True),
        _page([4], high_water=4, has_more=True),
        _page([3], high_water=4),
    ]

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        assert await screen._load_next_items_page() is True

        assert controller.list_reader_items_page.await_count == 4
        assert screen._items_snapshot.page_count == 2
        assert [row["item_id"] for row in screen._loaded_items] == [3]


@pytest.mark.asyncio
async def test_late_continuation_from_superseded_query_cannot_append():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=2, has_more=True
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        entered = asyncio.Event()
        old_result: asyncio.Future[WatchlistItemPage] = (
            asyncio.get_running_loop().create_future()
        )

        async def controlled(**kwargs):
            if kwargs.get("snapshot_max_item_id") is not None:
                entered.set()
                return await old_result
            return _page([9], high_water=9, snapshot_count=1)

        controller.list_reader_items_page.reset_mock()
        controller.list_reader_items_page.side_effect = controlled
        old_next = asyncio.create_task(screen._load_next_items_page())
        await _wait_until(pilot, entered.is_set)
        screen._items_search_query = "new query"
        assert await screen._replace_items_snapshot(reason="search") is True
        replacement = screen._items_snapshot
        old_result.set_result(_page([3], high_water=4))

        assert await old_next is False
        assert screen._items_snapshot is replacement
        assert [row["item_id"] for row in screen._loaded_items] == [9]


@pytest.mark.asyncio
async def test_refresh_requests_a_new_first_page_and_preserves_reader():
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([4], high_water=4, snapshot_count=2, has_more=True),
        _page([3], high_water=4),
        _page([9], high_water=9, snapshot_count=1),
    ]

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        assert await screen._load_next_items_page() is True
        open_item = screen._loaded_items[0]
        screen._selected_content_item = open_item
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item
        controller.list_reader_items_page.reset_mock()

        screen.handle_refresh_items_requested(RefreshItemsRequested())
        await _wait_until(pilot, lambda: controller.list_reader_items_page.await_count == 1)
        await _wait_until(pilot, lambda: not screen._items_page_loading)

        kwargs = controller.list_reader_items_page.await_args.kwargs
        assert "snapshot_max_item_id" not in kwargs
        assert "after" not in kwargs
        assert screen._items_page_index == 0
        assert screen._selected_content_item is open_item
        assert content.item is open_item
        assert [row["item_id"] for row in screen._loaded_items] == [9]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_filter", "search", "reason"),
    [("unread", "", "filter"), ("all", "needle", "search")],
)
async def test_query_replacement_keeps_old_rows_and_reader_until_commit(
    status_filter, search, reason
):
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4, 3], high_water=4, snapshot_count=2
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        prior_rows = screen._loaded_items
        open_item = prior_rows[0]
        screen._selected_content_item = open_item
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item
        entered = asyncio.Event()
        release = asyncio.Event()

        async def pending(**_kwargs):
            entered.set()
            await release.wait()
            return _page([8], high_water=8, snapshot_count=1)

        controller.list_reader_items_page.reset_mock()
        controller.list_reader_items_page.side_effect = pending
        if reason == "filter":
            screen.handle_items_filter_changed(ItemsFilterChanged(status_filter, search))
        else:
            screen._items_search_query = search
            task = asyncio.create_task(screen._replace_items_snapshot(reason="search"))
        await _wait_until(pilot, entered.is_set)

        assert screen._loaded_items is prior_rows
        assert screen._selected_content_item is open_item
        assert content.item is open_item
        release.set()
        await _wait_until(pilot, lambda: not screen._items_page_loading)
        if reason == "search":
            assert await task is True

        assert screen._items_page_index == 0
        assert screen._items_snapshot.query.context_key[-2:] == (
            status_filter,
            search.casefold(),
        )
        assert open_item in screen._loaded_items, "filter/search pins the open article"


@pytest.mark.asyncio
async def test_failed_query_replacement_keeps_committed_query_rows_and_reader():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=1
    )

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        prior_snapshot = screen._items_snapshot
        prior_rows = screen._loaded_items
        open_item = prior_rows[0]
        screen._selected_content_item = open_item
        controller.list_reader_items_page.side_effect = RuntimeError("offline")
        screen._items_search_query = "new"

        assert await screen._replace_items_snapshot(reason="search") is False

        assert screen._items_snapshot is prior_snapshot
        assert screen._loaded_items is prior_rows
        assert screen._selected_content_item is open_item


@pytest.mark.asyncio
async def test_same_query_pane_rebuild_uses_cache_and_preserves_selected_article():
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([4], high_water=4, snapshot_count=2, has_more=True),
        _page([3], high_water=4),
    ]

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        assert await screen._load_next_items_page() is True
        selected = screen._loaded_items[0]
        screen._selected_content_item = selected
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        pane.selected_item = selected
        controller.list_reader_items_page.reset_mock()

        await screen.query_one("#wl-workbench").refresh_region_content(Region.ITEMS)
        replacement = screen.query_one("#watchlists-items-pane", ArticleListPane)

        assert controller.list_reader_items_page.await_count == 0
        assert replacement.items is screen._loaded_items
        assert replacement.selected_item is selected
        assert replacement.page_number == 2


@pytest.mark.asyncio
async def test_selection_records_committed_query_while_detail_fetch_is_pending():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=1
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        committed_query = screen._items_snapshot.query
        item = screen._loaded_items[0]
        started = asyncio.Event()
        detail: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def pending_content(**_kwargs):
            started.set()
            return await detail

        controller.get_item_content.side_effect = pending_content
        screen._mark_item_read_on_open = Mock()
        task = asyncio.create_task(screen.handle_item_selected(ItemSelected(item)))
        await _wait_until(pilot, started.is_set)
        screen._items_search_query = "pending"
        detail.set_result("Fetched")
        await task

        assert screen._selected_content_item is item
        assert screen._selected_content_page_key == committed_query.context_key


@pytest.mark.asyncio
async def test_one_mutation_path_patches_every_cached_projection_without_query():
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([4, 3], high_water=4, snapshot_count=3, has_more=True),
        _page([2], high_water=4),
    ]

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        selected = screen._loaded_items[0]
        screen._selected_content_item = selected
        assert await screen._load_next_items_page() is True
        controller.list_reader_items_page.reset_mock()

        screen._patch_committed_items_after_mutation(4, status="reviewed", is_flagged=True)

        cached = screen._items_snapshot.page(0)[0]
        assert cached["status"] == "reviewed"
        assert cached["is_flagged"] is True
        assert selected["status"] == "reviewed"
        assert selected["is_flagged"] is True
        assert controller.list_reader_items_page.await_count == 0


def test_query_identity_uses_explicit_scope_and_omits_page_index():
    screen = WatchlistsCollectionsScreen(Mock())
    screen.__dict__["_reactive_runtime_backend"] = "local"
    scope = TreeScope(kind="watchlist", watchlist_id=7)

    first = screen._items_page_key(scope=scope, status="unread", search=" Needle ")
    second = screen._items_page_key(scope=scope, status="unread", search="needle")

    assert first == second
    assert first == ("local", "watchlist", 7, None, "unread", "needle")


def test_production_has_no_legacy_items_loader_or_offset_reader_calls():
    source = inspect.getsource(WatchlistsCollectionsScreen)

    assert "def _load_items(" not in source
    assert "self._load_items(" not in source
    assert "offset=" not in source[source.index("def _items_page_key") :]
    assert "list_reader_items_page" in source
