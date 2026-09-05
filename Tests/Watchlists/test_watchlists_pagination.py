"""Transactional cached-snapshot pagination regressions for Watchlists Read."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness, _static_text
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
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
    BreadcrumbScopeSelected,
    InspectorPane,
)
from tldw_chatbook.UI.Watchlists_Modules.items_pane import (
    ItemSelected,
    ItemsFilterChanged,
    RefreshItemsRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.reader_item_snapshot import (
    ReaderItemSnapshot,
)
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
    TreeScope,
    TreeScopeChanged,
    WatchlistTree,
)


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
    day: int = 13,
) -> WatchlistItemPage:
    rows = _items(ids, day=day)
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
        assert [row["item_id"] for row in screen._loaded_items] == [
            9,
            open_item["item_id"],
        ]
        assert screen._items_snapshot_count == 1


@pytest.mark.asyncio
async def test_arrivals_use_the_committed_query_and_only_update_the_pill():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=17, has_more=True
    )
    controller.count_reader_item_arrivals.return_value = 3

    async with _open_screen(controller) as (screen, _pilot):
        screen._items_status_filter = "unread"
        screen._items_search_query = "Needle"
        assert await screen._replace_items_snapshot(reason="search") is True
        snapshot = screen._items_snapshot
        committed_kwargs = snapshot.query.as_kwargs()
        rows = screen._loaded_items
        screen._items_page_index = 2
        screen._selected_content_item = rows[0]

        # These are attempted, mutable controls. Arrival authority belongs to
        # the mounted snapshot, so none of them may leak into the count query.
        screen._items_status_filter = "all"
        screen._items_search_query = "Different attempted search"
        screen._pending_tree_scope = TreeScope(kind="source", source_id=999)

        assert await screen._refresh_items_pending_arrivals() is True

        controller.count_reader_item_arrivals.assert_awaited_once_with(
            runtime_backend="local",
            snapshot_max_item_id=snapshot.watermark,
            **committed_kwargs,
        )
        assert screen._items_snapshot is snapshot
        assert screen._loaded_items is rows
        assert screen._items_page_index == 2
        assert screen._selected_content_item is rows[0]
        assert screen._items_snapshot_count == 17
        assert screen._items_pending_arrivals == 3
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert pane.snapshot_count == 17
        assert pane.new_items_note == "3 new items"


@pytest.mark.asyncio
async def test_stale_arrivals_result_cannot_cross_snapshot_replacement():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=1
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        entered = asyncio.Event()
        pending: asyncio.Future[int] = asyncio.get_running_loop().create_future()

        async def count(**_kwargs):
            entered.set()
            return await pending

        controller.count_reader_item_arrivals.side_effect = count
        task = asyncio.create_task(screen._refresh_items_pending_arrivals())
        await _wait_until(pilot, entered.is_set)
        prior = screen._items_snapshot
        controller.list_reader_items_page.return_value = _page(
            [9], high_water=9, snapshot_count=1
        )
        assert await screen._replace_items_snapshot(reason="refresh") is True
        assert screen._items_snapshot is not prior
        pending.set_result(7)

        assert await task is False
        assert screen._items_pending_arrivals == 0


@pytest.mark.asyncio
async def test_only_the_latest_arrivals_reconciliation_can_publish():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=1
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        pending = [
            asyncio.get_running_loop().create_future(),
            asyncio.get_running_loop().create_future(),
        ]
        started = 0

        async def count(**_kwargs):
            nonlocal started
            index = started
            started += 1
            return await pending[index]

        controller.count_reader_item_arrivals.side_effect = count
        old = asyncio.create_task(screen._refresh_items_pending_arrivals())
        await _wait_until(pilot, lambda: started == 1)
        newest = asyncio.create_task(screen._refresh_items_pending_arrivals())
        await _wait_until(pilot, lambda: started == 2)
        pending[1].set_result(2)
        assert await newest is True
        pending[0].set_result(7)
        assert await old is False
        assert screen._items_pending_arrivals == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state_name", "replacement"),
    [("_reactive_active_section", "sources"), ("_reactive_runtime_backend", "server")],
)
async def test_arrivals_cannot_publish_after_section_or_backend_changes(
    state_name, replacement
):
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=1
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        entered = asyncio.Event()
        pending: asyncio.Future[int] = asyncio.get_running_loop().create_future()

        async def count(**_kwargs):
            entered.set()
            return await pending

        controller.count_reader_item_arrivals.side_effect = count
        task = asyncio.create_task(screen._refresh_items_pending_arrivals())
        await _wait_until(pilot, entered.is_set)
        screen.__dict__[state_name] = replacement
        pending.set_result(4)

        assert await task is False
        assert screen._items_pending_arrivals == 0


@pytest.mark.asyncio
async def test_refresh_failure_keeps_arrival_notice_and_success_clears_it():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=17
    )

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        screen._items_pending_arrivals = 3
        screen._push_items_pager_state()
        prior = screen._items_snapshot
        controller.list_reader_items_page.side_effect = RuntimeError("offline")

        assert await screen._replace_items_snapshot(reason="refresh") is False
        assert screen._items_snapshot is prior
        assert screen._items_snapshot_count == 17
        assert screen._items_pending_arrivals == 3

        controller.list_reader_items_page.side_effect = None
        controller.list_reader_items_page.return_value = _page(
            [9], high_water=9, snapshot_count=21
        )
        assert await screen._replace_items_snapshot(reason="refresh") is True
        assert screen._items_snapshot.watermark == 9
        assert screen._items_snapshot_count == 21
        assert screen._items_pending_arrivals == 0
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert pane.snapshot_count == 21
        assert pane.new_items_note == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failure", "cancel", "success"])
async def test_refresh_preserves_search_authority_until_transactional_outcome(
    outcome,
):
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=1
    )

    async with _open_screen(controller) as (screen, pilot):
        screen._items_search_query = "server-only-match"
        assert await screen._replace_items_snapshot(reason="search") is True
        prior_snapshot = screen._items_snapshot
        prior_rows = screen._loaded_items
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert pane.search_results_authoritative is True
        assert [row["item_id"] for row in pane._filtered_items()] == [4]
        entered = asyncio.Event()
        pending: asyncio.Future[WatchlistItemPage] = (
            asyncio.get_running_loop().create_future()
        )

        async def refresh_page(**_kwargs):
            entered.set()
            return await pending

        controller.list_reader_items_page.side_effect = refresh_page
        screen._supersede_items_query_intent()
        task = asyncio.create_task(
            screen._replace_items_snapshot(reason="refresh")
        )
        await _wait_until(pilot, entered.is_set)

        assert screen._items_snapshot is prior_snapshot
        assert screen._loaded_items is prior_rows
        assert screen._items_search_results_authoritative is True
        assert pane.search_results_authoritative is True
        assert [row["item_id"] for row in pane._filtered_items()] == [4]

        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "failure":
            pending.set_exception(RuntimeError("offline"))
            assert await task is False
        else:
            pending.set_result(_page([9], high_water=9, snapshot_count=1))
            assert await task is True

        if outcome == "success":
            assert screen._items_snapshot is not prior_snapshot
            assert [row["item_id"] for row in screen._loaded_items] == [9]
        else:
            assert screen._items_snapshot is prior_snapshot
            assert screen._loaded_items is prior_rows
        assert screen._items_search_results_authoritative is True
        assert pane.search_results_authoritative is True
        expected_id = 9 if outcome == "success" else 4
        assert [row["item_id"] for row in pane._filtered_items()] == [expected_id]


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
@pytest.mark.parametrize("reason", ["filter", "search"])
async def test_query_pin_is_cached_and_deduplicated_across_page_replay(reason):
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([4], high_water=4, snapshot_count=1),
        _page([3], high_water=4, snapshot_count=2, has_more=True),
        _page([4, 2], high_water=4),
    ]

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        open_item = screen._loaded_items[0]
        screen._selected_content_item = open_item
        if reason == "filter":
            screen._items_status_filter = "unread"
        else:
            screen._items_search_query = "needle"

        assert await screen._replace_items_snapshot(reason=reason) is True
        assert [row["item_id"] for row in screen._items_snapshot.page(0)] == [4, 3]
        assert 4 in screen._items_snapshot.seen_ids
        assert await screen._load_next_items_page() is True
        assert [row["item_id"] for row in screen._loaded_items] == [2]
        assert sum(
            row["item_id"] == 4
            for page in screen._items_snapshot.pages
            for row in page
        ) == 1

        controller.list_reader_items_page.reset_mock()
        assert await screen._present_cached_items_page(0) is True
        assert [row["item_id"] for row in screen._loaded_items] == [4, 3]
        assert await screen._load_next_items_page() is True
        assert [row["item_id"] for row in screen._loaded_items] == [2]
        assert controller.list_reader_items_page.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["filter", "search"])
async def test_full_query_pin_caches_displaced_terminal_row(reason):
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([100], high_water=100, snapshot_count=1),
        _page(range(50, 0, -1), high_water=100, snapshot_count=50),
    ]

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        screen._selected_content_item = screen._loaded_items[0]
        if reason == "filter":
            screen._items_status_filter = "unread"
        else:
            screen._items_search_query = "needle"

        assert await screen._replace_items_snapshot(reason=reason) is True
        snapshot = screen._items_snapshot
        assert len(snapshot.page(0)) == 50
        assert snapshot.page_count == 1
        assert snapshot.has_next(0) is True
        assert snapshot.cursor is None
        assert snapshot.has_more is False
        assert [row["item_id"] for row in snapshot.pending_items] == [1]
        assert snapshot.seen_ids == frozenset({100, *range(2, 51)})

        controller.list_reader_items_page.reset_mock()
        assert await screen._load_next_items_page() is True
        assert [row["item_id"] for row in screen._loaded_items] == [1]
        assert screen._items_snapshot.pending_items == ()
        assert sum(
            row["item_id"] == 1
            for page in screen._items_snapshot.pages
            for row in page
        ) == 1
        assert await screen._load_next_items_page() is False
        assert controller.list_reader_items_page.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["filter", "search"])
async def test_query_pin_merges_displaced_row_with_service_continuation(reason):
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([100], high_water=100, snapshot_count=1, day=10),
        _page(
            range(50, 0, -1),
            high_water=100,
            snapshot_count=51,
            has_more=True,
            day=13,
        ),
        _page([0], high_water=100, day=12),
    ]

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        screen._selected_content_item = screen._loaded_items[0]
        if reason == "filter":
            screen._items_status_filter = "unread"
        else:
            screen._items_search_query = "needle"

        assert await screen._replace_items_snapshot(reason=reason) is True
        assert screen._loaded_items[-1]["item_id"] == 100

        controller.list_reader_items_page.reset_mock()
        assert await screen._load_next_items_page() is True
        assert [row["item_id"] for row in screen._loaded_items] == [1, 0]
        assert controller.list_reader_items_page.await_count == 1


@pytest.mark.asyncio
async def test_mark_all_read_rebuilds_multi_page_unread_snapshot():
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page(
            range(60, 10, -1),
            high_water=60,
            snapshot_count=60,
            has_more=True,
        ),
        _page([], high_water=60, snapshot_count=0),
    ]
    controller.mark_all_read.return_value = list(range(1, 61))

    async with _open_screen(controller) as (screen, _pilot):
        screen._items_status_filter = "unread"
        assert await screen._replace_items_snapshot(reason="filter") is True
        assert screen._items_snapshot.snapshot_count == 60
        assert screen._items_snapshot.has_next(0) is True

        await screen._mark_all_read_worker()

        assert controller.list_reader_items_page.await_count == 2
        assert screen._items_snapshot.snapshot_count == 0
        assert screen._loaded_items == []
        assert screen._items_snapshot.has_next(0) is False
        assert await screen._load_next_items_page() is False


@pytest.mark.asyncio
async def test_mark_all_read_closes_uncached_tail_when_rebuild_fails():
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page(
            range(60, 10, -1),
            high_water=60,
            snapshot_count=60,
            has_more=True,
        ),
        RuntimeError("offline"),
    ]
    controller.mark_all_read.return_value = list(range(1, 61))

    async with _open_screen(controller) as (screen, _pilot):
        screen._items_status_filter = "unread"
        assert await screen._replace_items_snapshot(reason="filter") is True

        await screen._mark_all_read_worker()

        assert controller.list_reader_items_page.await_count == 2
        assert screen._items_snapshot.snapshot_count == 50
        assert screen._items_snapshot.has_next(0) is False
        assert await screen._load_next_items_page() is False


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
        screen._items_pending_arrivals = 3
        screen._push_items_pager_state()
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        pane.selected_item = selected
        controller.list_reader_items_page.reset_mock()

        await screen.query_one("#wl-workbench").refresh_region_content(Region.ITEMS)
        replacement = screen.query_one("#watchlists-items-pane", ArticleListPane)

        assert controller.list_reader_items_page.await_count == 0
        assert replacement.items is screen._loaded_items
        assert replacement.selected_item is selected
        assert replacement.page_number == 2
        assert replacement.snapshot_count == 2
        assert replacement.new_items_note == "3 new items"


@pytest.mark.asyncio
async def test_explicit_refresh_retains_the_open_out_of_predicate_pinned_row():
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([4], high_water=4, snapshot_count=1),
        _page([3], high_water=4, snapshot_count=1),
        _page([3], high_water=4, snapshot_count=1),
    ]

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        pinned = screen._loaded_items[0]
        screen._selected_content_item = pinned
        screen._items_search_query = "needle"
        assert await screen._replace_items_snapshot(reason="search") is True
        assert [row["item_id"] for row in screen._loaded_items] == [4, 3]

        assert await screen._replace_items_snapshot(reason="refresh") is True
        assert [row["item_id"] for row in screen._loaded_items] == [4, 3]
        assert screen._items_snapshot_count == 1
        assert screen._selected_content_item is pinned


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
async def test_detail_result_cannot_select_from_replaced_snapshot():
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([4], high_water=4, snapshot_count=1),
        _page([9], high_water=9, snapshot_count=1),
    ]

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        stale_item = screen._loaded_items[0]
        started = asyncio.Event()
        detail: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def pending_content(**_kwargs):
            started.set()
            return await detail

        controller.get_item_content.side_effect = pending_content
        screen._mark_item_read_on_open = Mock()
        selection = asyncio.create_task(
            screen.handle_item_selected(ItemSelected(stale_item))
        )
        await _wait_until(pilot, started.is_set)
        old_snapshot = screen._items_snapshot
        screen._items_search_query = "replacement"
        assert await screen._replace_items_snapshot(reason="search") is True
        assert screen._items_snapshot is not old_snapshot
        assert [row["item_id"] for row in screen._loaded_items] == [9]

        detail.set_result("Fetched stale body")
        await selection

        content = screen.query_one("#watchlists-content-pane", ContentPane)
        assert screen._selected_content_item is None
        assert screen._selected_content_page_key is None
        assert content.item is None
        screen._mark_item_read_on_open.assert_not_called()


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


@pytest.mark.parametrize(
    ("scope", "expected"),
    (
        (
            TreeScope(kind="source", source_id=9, parent_context="all"),
            {"source_id": 9},
        ),
        (
            TreeScope(kind="source", source_id=9, parent_context="unassigned"),
            {"source_id": 9, "unassigned_only": True},
        ),
        (
            TreeScope(kind="source", source_id=9, parent_context="unread"),
            {"source_id": 9, "status": "new"},
        ),
        (
            TreeScope(
                kind="source",
                source_id=9,
                watchlist_id=7,
                parent_context="watchlist",
            ),
            {"source_id": 9, "watchlist_id": 7},
        ),
    ),
)
def test_contextual_source_scope_emits_its_exact_reader_predicates(
    scope: TreeScope, expected: dict[str, object]
) -> None:
    screen = WatchlistsCollectionsScreen(Mock())

    assert screen._items_scope_query(scope) == expected


def test_query_identity_uses_explicit_contextual_scope_and_omits_page_index():
    screen = WatchlistsCollectionsScreen(Mock())
    screen.__dict__["_reactive_runtime_backend"] = "local"
    scope = TreeScope(
        kind="source",
        source_id=9,
        watchlist_id=7,
        parent_context="watchlist",
    )

    first = screen._items_page_key(scope=scope, status="unread", search=" Needle ")
    second = screen._items_page_key(scope=scope, status="unread", search="needle")

    assert first == second
    assert first == (
        "local",
        "source",
        "watchlist",
        7,
        9,
        "unread",
        "needle",
    )


def test_same_source_under_different_parents_has_distinct_query_identity():
    screen = WatchlistsCollectionsScreen(Mock())
    screen.__dict__["_reactive_runtime_backend"] = "local"
    scopes = (
        TreeScope(kind="source", source_id=9, parent_context="all"),
        TreeScope(kind="source", source_id=9, parent_context="unassigned"),
        TreeScope(kind="source", source_id=9, parent_context="unread"),
        TreeScope(
            kind="source",
            source_id=9,
            watchlist_id=7,
            parent_context="watchlist",
        ),
    )

    keys = {
        screen._items_page_key(scope=scope, status="all", search="")
        for scope in scopes
    }

    assert len(keys) == len(scopes)


def test_production_has_no_legacy_items_loader_or_offset_reader_calls():
    source = inspect.getsource(WatchlistsCollectionsScreen)

    assert "def _load_items(" not in source
    assert "self._load_items(" not in source
    assert "offset=" not in source[source.index("def _items_page_key") :]
    assert "list_reader_items_page" in source


@pytest.mark.asyncio
async def test_atomic_scope_keeps_committed_reader_until_first_page_mounts(
    monkeypatch,
):
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=1
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        screen._wc_loaded = True
        screen._local_watchlist_count = 1
        screen._refresh_centre_header_for_scope()
        await _wait_until(
            pilot, lambda: bool(screen.query("#wc-watchlists-summary"))
        )
        prior_scope = screen.tree_scope
        prior_rows = screen._loaded_items
        open_item = prior_rows[0]
        screen._selected_content_item = open_item
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item
        content.position = "1 of 1"
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        screen._items_status_filter = "all"
        pane.status_filter = "all"
        pane.selected_item = open_item
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def replacement(**_kwargs):
            entered.set()
            await release.wait()
            return _page(
                [9], high_water=9, snapshot_count=23, has_more=True
            )

        controller.list_reader_items_page.reset_mock()
        controller.list_reader_items_page.side_effect = replacement
        candidate_source_id = screen._watchlist_bundle_service()._db.add_subscription(
            name="Candidate feed",
            type="rss",
            source="https://atomic-candidate.example/feed",
        )
        await screen._load_tree_data().wait()
        candidate = TreeScope(
            kind="source",
            source_id=candidate_source_id,
            parent_context="unread",
        )
        screen._tree_all_source_rows = [
            {"id": candidate_source_id, "name": "Candidate feed"}
        ]
        tree = screen.query_one("#wl-tree", WatchlistTree)
        screen.post_message(TreeScopeChanged(candidate))
        await _wait_until(pilot, entered.is_set)

        assert screen.tree_scope == prior_scope
        assert tree.active_scope == prior_scope
        assert screen._loaded_items is prior_rows
        assert screen._selected_content_item is open_item
        assert content.item is open_item

        committed_paint: dict[str, object] = {}
        original_publish = screen._publish_items_rows

        async def observe_atomic_paint(*args, **kwargs):
            result = await original_publish(*args, **kwargs)
            if kwargs.get("atomic_batch"):
                committed_paint.update(
                    heading=_static_text(
                        screen.query_one("#wc-watchlists-summary")
                    ),
                    inspector_scope=inspector.scope,
                    inspector_labels=list(inspector.breadcrumb_labels),
                    tree_scope=screen.query_one(
                        "#wl-tree", WatchlistTree
                    ).active_scope,
                    rows=[row["item_id"] for row in pane.items],
                    snapshot_count=screen._items_snapshot_count,
                    page_number=pane.page_number,
                    has_next=pane.has_next,
                    page_loading=pane.page_loading,
                    reader=content.item,
                )
            return result

        monkeypatch.setattr(screen, "_publish_items_rows", observe_atomic_paint)
        release.set()
        await _wait_until(pilot, lambda: bool(committed_paint))
        assert "All Unread / Candidate feed" in str(committed_paint["heading"])
        assert committed_paint["inspector_scope"] == candidate
        assert committed_paint["inspector_labels"] == [
            "All Unread",
            "Candidate feed",
        ]
        assert committed_paint["tree_scope"] == candidate
        assert committed_paint["rows"] == [9]
        assert committed_paint["snapshot_count"] == 23
        assert committed_paint["page_number"] == 1
        assert committed_paint["has_next"] is True
        assert committed_paint["page_loading"] is False
        assert committed_paint["reader"] is None
        assert screen.query_one("#wl-tree", WatchlistTree).active_scope == candidate
        assert [row["item_id"] for row in screen._loaded_items] == [9]
        assert screen._items_snapshot_count == 23
        assert screen._selected_content_item is None
        assert content.item is None
        assert content.position == ""
        assert pane.selected_item is None
        assert screen._items_status_filter == "all"
        assert pane.status_filter == "unread"
        assert pane.status_filter_disabled_reason == (
            "All Unread always shows unread items."
        )

        controller.list_reader_items_page.side_effect = None
        controller.list_reader_items_page.return_value = _page(
            [11], high_water=11, snapshot_count=1
        )
        restored_scope = TreeScope(kind="all")
        screen.post_message(TreeScopeChanged(restored_scope))
        await _wait_until(pilot, lambda: screen.tree_scope == restored_scope)

        assert screen._items_status_filter == "all"
        assert pane.status_filter == "all"
        assert pane.status_filter_disabled_reason is None


@pytest.mark.asyncio
async def test_pending_scope_failure_retains_committed_scope_and_names_both():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=17, has_more=True
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        screen._wc_loaded = True
        screen._local_watchlist_count = 1
        screen._refresh_centre_header_for_scope()
        await _wait_until(
            pilot, lambda: bool(screen.query("#wc-watchlists-summary"))
        )
        prior_scope = screen.tree_scope
        prior_selected_scope = screen.selected_scope
        prior_snapshot = screen._items_snapshot
        prior_rows = screen._loaded_items
        prior_count = screen._items_snapshot_count
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        open_item = prior_rows[0]
        screen._selected_content_item = open_item
        screen._selected_content_page_key = prior_snapshot.query.context_key
        pane.selected_item = open_item
        content.item = open_item
        content.position = "1 of 17"
        await pilot.pause()
        prior_inspector_scope = inspector.scope
        prior_inspector_labels = list(inspector.breadcrumb_labels)
        prior_page_number = pane.page_number
        prior_has_next = pane.has_next
        prior_content_position = content.position
        screen._items_status_filter = "all"
        pane.status_filter = "all"
        candidate_source_id = screen._watchlist_bundle_service()._db.add_subscription(
            name="Candidate [A]",
            type="rss",
            source="https://failed-candidate.example/feed",
        )
        await screen._load_tree_data().wait()
        screen._tree_all_source_rows = [
            {"id": candidate_source_id, "name": "Candidate [A]"}
        ]
        tree = screen.query_one("#wl-tree", WatchlistTree)
        await _wait_until(
            pilot,
            lambda: "1 source"
            in _static_text(screen.query_one("#wc-watchlists-summary")),
        )
        prior_heading = _static_text(screen.query_one("#wc-watchlists-summary"))
        screen.app_instance.notify = Mock()
        controller.list_reader_items_page.reset_mock()
        controller.list_reader_items_page.side_effect = RuntimeError("offline")

        screen.post_message(
            TreeScopeChanged(
                TreeScope(
                    kind="source",
                    source_id=candidate_source_id,
                    parent_context="unread",
                )
            )
        )
        await _wait_until(
            pilot, lambda: controller.list_reader_items_page.await_count == 1
        )
        await _wait_until(pilot, lambda: not screen._items_page_loading)

        assert screen.tree_scope == prior_scope
        assert screen.selected_scope == prior_selected_scope
        assert tree.active_scope == prior_scope
        assert screen._items_status_filter == "all"
        assert pane.status_filter == "all"
        assert pane.status_filter_disabled_reason is None
        assert _static_text(screen.query_one("#wc-watchlists-summary")) == prior_heading
        assert inspector.scope == prior_inspector_scope
        assert list(inspector.breadcrumb_labels) == prior_inspector_labels
        assert screen._items_snapshot is prior_snapshot
        assert screen._loaded_items is prior_rows
        assert pane.items is prior_rows
        assert screen._items_snapshot_count == prior_count == 17
        assert pane.page_number == prior_page_number == 1
        assert pane.has_next == prior_has_next is True
        assert pane.page_loading is False
        assert screen._selected_content_item is open_item
        assert pane.selected_item is open_item
        assert content.item is open_item
        assert content.position == prior_content_position
        screen.app_instance.notify.assert_called_once_with(
            "Couldn't open Candidate [A] under All Unread; still showing "
            "All Sources.",
            severity="error",
            markup=False,
        )


@pytest.mark.asyncio
async def test_pending_scope_presentation_failure_does_not_commit(monkeypatch):
    controller = AsyncMock()
    controller.list_reader_items_page.side_effect = [
        _page([4], high_water=4, snapshot_count=1),
        _page([9], high_water=9, snapshot_count=1),
    ]

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        prior_snapshot = screen._items_snapshot
        prior_rows = screen._loaded_items
        screen._tree_watchlists = [{"id": 7, "name": "Candidate [A]"}]
        screen.app_instance.notify = Mock()
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        original_apply = pane.apply_page_items

        async def fail_candidate(items, *, focus_first=False):
            await original_apply(items, focus_first=focus_first)
            if items and items[0]["item_id"] == 9:
                raise RuntimeError("paint failed")

        monkeypatch.setattr(pane, "apply_page_items", fail_candidate)
        screen.post_message(
            BreadcrumbScopeSelected(TreeScope(kind="watchlist", watchlist_id=7))
        )
        await _wait_until(
            pilot, lambda: controller.list_reader_items_page.await_count == 2
        )
        await _wait_until(pilot, lambda: not screen._items_page_loading)

        assert screen.tree_scope == TreeScope(kind="all")
        assert screen._items_snapshot is prior_snapshot
        assert screen._loaded_items is prior_rows
        assert pane.items is prior_rows
        screen.app_instance.notify.assert_called_once_with(
            "Couldn't open Candidate [A]; still showing All Sources.",
            severity="error",
            markup=False,
        )


@pytest.mark.asyncio
async def test_non_scope_replacement_supersedes_orphaned_pending_scope():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=1
    )

    async with _open_screen(controller) as (screen, _pilot):
        orphaned = TreeScope(kind="watchlist", watchlist_id=7)
        screen._pending_tree_scope = orphaned

        screen._supersede_items_query_intent()

        assert screen._pending_tree_scope is None


@pytest.mark.asyncio
async def test_management_scope_invalidates_reader_without_hidden_item_io():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=1
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        open_item = screen._loaded_items[0]
        screen._selected_content_item = open_item
        screen._selected_content_page_key = ("page", 1)
        screen._items_pending_arrivals = 3
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        pane.selected_item = open_item
        pane.show_new_items_pill(3)

        screen.__dict__["_reactive_active_section"] = "sources"
        controller.list_reader_items_page.reset_mock()
        candidate = TreeScope(
            kind="source",
            source_id=9,
            parent_context="unassigned",
        )
        screen._request_tree_scope(candidate)

        assert controller.list_reader_items_page.await_count == 0
        assert screen.tree_scope == candidate
        assert screen._items_snapshot is None
        assert screen._loaded_items == []
        assert screen._selected_content_item is None
        assert screen._selected_content_page_key is None
        assert screen._items_snapshot_count == 0
        assert screen._items_pending_arrivals == 0
        assert content.item is None


@pytest.mark.asyncio
async def test_pending_scope_only_newest_request_can_publish():
    controller = AsyncMock()
    controller.list_reader_items_page.return_value = _page(
        [4], high_water=4, snapshot_count=1
    )

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._replace_items_snapshot(reason="initial") is True
        calls: list[str] = []
        returned: list[str] = []
        releases = {
            parent: asyncio.Event() for parent in ("all", "unassigned", "unread")
        }
        item_ids = {"all": 7, "unassigned": 8, "unread": 9}

        async def replacement(**kwargs):
            parent = (
                "unread"
                if kwargs.get("status") == "new"
                else "unassigned"
                if kwargs.get("unassigned_only")
                else "all"
            )
            calls.append(parent)
            while not releases[parent].is_set():
                try:
                    await releases[parent].wait()
                except asyncio.CancelledError:
                    continue
            returned.append(parent)
            item_id = item_ids[parent]
            return _page(
                [item_id],
                high_water=item_id,
                snapshot_count=item_id,
            )

        controller.list_reader_items_page.reset_mock()
        controller.list_reader_items_page.side_effect = replacement
        for parent in ("all", "unassigned", "unread"):
            screen.post_message(
                TreeScopeChanged(
                    TreeScope(kind="source", source_id=9, parent_context=parent)
                )
            )
            await _wait_until(pilot, lambda: parent in calls)

        assert screen._pending_tree_scope == TreeScope(
            kind="source", source_id=9, parent_context="unread"
        )
        releases["unread"].set()
        await _wait_until(
            pilot, lambda: screen.tree_scope.parent_context == "unread"
        )
        assert returned == ["unread"]
        assert screen._pending_tree_scope is None
        assert screen.tree_scope == TreeScope(
            kind="source", source_id=9, parent_context="unread"
        )
        assert [row["item_id"] for row in screen._loaded_items] == [9]
        assert screen._items_snapshot_count == 9

        releases["unassigned"].set()
        releases["all"].set()
        await _wait_until(
            pilot,
            lambda: set(returned) == {"all", "unassigned", "unread"},
        )
        await pilot.pause(0.1)

        assert screen._pending_tree_scope is None
        assert screen.tree_scope == TreeScope(
            kind="source", source_id=9, parent_context="unread"
        )
        assert [row["item_id"] for row in screen._loaded_items] == [9]
        assert screen._items_snapshot_count == 9
