"""Transactional pagination regressions for the Watchlists Read screen."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Button, Static

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
from tldw_chatbook.UI.Watchlists_Modules.items_pane import (
    ItemSelected,
    ItemsFilterChanged,
    NextUnreadRequested,
    RefreshItemsRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope


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
        await _wait_until(
            pilot,
            lambda: screen._items_page_index == 1
            and not screen._items_page_loading
            and {str(item["id"]) for item in pane.items}
            == {str(index) for index in range(100, 150)},
        )
        await pilot.pause()

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
        assert {str(item["id"]) for item in pane.displayed_items()} == {
            str(index) for index in range(100, 150)
        }


@pytest.mark.asyncio
async def test_page_state_commits_only_after_mounted_rows_finish_applying(monkeypatch):
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 51)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        prior_rows = screen._loaded_items
        prior_key = screen._items_committed_page_key
        open_item = prior_rows[0]
        screen._selected_content_item = open_item
        screen._selected_content_page_key = prior_key
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        original_apply = pane.apply_page_items
        presentation_started = asyncio.Event()
        release_presentation = asyncio.Event()

        async def blocking_apply(items, *, focus_first=False):
            presentation_started.set()
            await release_presentation.wait()
            await original_apply(items, focus_first=focus_first)

        monkeypatch.setattr(pane, "apply_page_items", blocking_apply)
        controller.list_items.return_value = _items(100, 51)
        controller.list_items.reset_mock()
        screen.post_message(NextItemsPageRequested())
        await _wait_until(pilot, presentation_started.is_set)

        assert screen._items_page_index == 0
        assert screen._items_committed_page_key == prior_key
        assert screen._loaded_items is prior_rows
        assert pane.items is prior_rows
        assert screen._items_page_loading is True
        assert pane.page_loading is True
        assert str(pane.query_one("#items-page-label", Static).renderable) == "Page 1"
        assert pane.query_one("#items-page-previous", Button).disabled is True
        assert pane.query_one("#items-page-next", Button).disabled is True
        assert screen._selected_content_item is open_item
        assert content.item is open_item

        release_presentation.set()
        await _wait_until(
            pilot,
            lambda: screen._items_page_index == 1
            and not screen._items_page_loading
            and {str(item["id"]) for item in pane.items}
            == {str(index) for index in range(100, 150)},
        )
        await pilot.pause()

        assert screen._items_committed_page_key == screen._items_page_key(1)
        assert str(pane.query_one("#items-page-label", Static).renderable) == "Page 2"
        assert pane.query_one("#items-page-previous", Button).disabled is False
        assert pane.query_one("#items-page-next", Button).disabled is False
        assert {str(item["id"]) for item in pane.displayed_items()} == {
            str(index) for index in range(100, 150)
        }
        assert screen._selected_content_item is open_item
        assert content.item is open_item


@pytest.mark.asyncio
async def test_selection_waits_for_presented_page_to_commit(monkeypatch):
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 51)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        original_apply = pane.apply_page_items
        presentation_started = asyncio.Event()
        release_presentation = asyncio.Event()

        async def apply_then_block(items, *, focus_first=False):
            await original_apply(items, focus_first=focus_first)
            if str(items[0]["id"]) == "100":
                presentation_started.set()
                await release_presentation.wait()

        monkeypatch.setattr(pane, "apply_page_items", apply_then_block)
        controller.list_items.return_value = _items(100, 51)
        controller.get_item_content.return_value = "Fetched body"
        screen._mark_item_read_on_open = Mock()
        load_task = asyncio.create_task(
            screen._load_items(target_page_index=1, explicit_page_change=True)
        )
        await _wait_until(pilot, presentation_started.is_set)

        item = pane.items[0]
        selection_task = asyncio.create_task(
            screen.handle_item_selected(ItemSelected(item))
        )
        await pilot.pause(0.05)
        selection_waited_for_commit = not selection_task.done()

        release_presentation.set()
        assert await load_task is True
        await selection_task

        assert selection_waited_for_commit
        assert screen._selected_content_item is item
        assert screen._selected_content_page_key == screen._items_page_key(1)


@pytest.mark.asyncio
async def test_cancelled_presentation_rolls_back_before_failed_successor(monkeypatch):
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 51)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        prior_rows = screen._loaded_items
        prior_key = screen._items_committed_page_key
        open_item = prior_rows[0]
        screen._selected_content_item = open_item
        screen._selected_content_page_key = prior_key
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        original_apply = pane.apply_page_items
        presentation_started = asyncio.Event()
        never_release = asyncio.Event()

        async def apply_then_block(items, *, focus_first=False):
            await original_apply(items, focus_first=focus_first)
            if str(items[0]["id"]) == "100":
                presentation_started.set()
                await never_release.wait()

        monkeypatch.setattr(pane, "apply_page_items", apply_then_block)
        calls = 0

        async def page_then_failure(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                assert kwargs["offset"] == 50
                return _items(100, 51)
            raise RuntimeError("successor failed")

        controller.list_items.reset_mock()
        controller.list_items.side_effect = page_then_failure
        screen.post_message(NextItemsPageRequested())
        await _wait_until(pilot, presentation_started.is_set)
        assert {str(item["id"]) for item in pane.items} == {
            str(index) for index in range(100, 150)
        }, "precondition: presentation mutated the mounted page before cancellation"

        screen.run_worker(
            screen._load_items(target_page_index=1, explicit_page_change=True),
            exclusive=True,
            group="wc_items",
        )
        await _wait_until(
            pilot,
            lambda: controller.list_items.await_count == 2
            and not screen._items_page_loading
            and pane.items is prior_rows,
        )
        await pilot.pause()

        assert screen._items_page_index == 0
        assert screen._items_committed_page_key == prior_key
        assert screen._loaded_items is prior_rows
        assert pane.items is prior_rows
        assert {str(item["id"]) for item in pane.displayed_items()} == {
            str(item["id"]) for item in prior_rows
        }
        assert screen._items_has_next is True
        assert str(pane.query_one("#items-page-label", Static).renderable) == "Page 1"
        assert pane.query_one("#items-page-previous", Button).disabled is True
        assert pane.query_one("#items-page-next", Button).disabled is False
        assert screen._selected_content_item is open_item
        assert content.item is open_item


@pytest.mark.asyncio
async def test_presentation_failure_restores_prior_rows_and_page(monkeypatch):
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 51)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        prior_rows = screen._loaded_items
        prior_key = screen._items_committed_page_key
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        original_apply = pane.apply_page_items

        async def apply_then_fail(items, *, focus_first=False):
            await original_apply(items, focus_first=focus_first)
            if str(items[0]["id"]) == "100":
                raise RuntimeError("row presentation failed")

        monkeypatch.setattr(pane, "apply_page_items", apply_then_fail)
        screen.app_instance.notify = Mock()
        controller.list_items.return_value = _items(100, 51)
        controller.list_items.reset_mock()
        screen.post_message(NextItemsPageRequested())
        await _wait_until(
            pilot,
            lambda: controller.list_items.await_count == 1
            and not screen._items_page_loading
            and pane.items is prior_rows,
        )
        await pilot.pause()

        assert screen._items_page_index == 0
        assert screen._items_committed_page_key == prior_key
        assert screen._loaded_items is prior_rows
        assert {str(item["id"]) for item in pane.displayed_items()} == {
            str(item["id"]) for item in prior_rows
        }
        assert screen._items_has_next is True
        assert str(pane.query_one("#items-page-label", Static).renderable) == "Page 1"
        assert screen.app_instance.notify.called


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


@pytest.mark.asyncio
async def test_status_change_resets_to_first_page_before_loading():
    controller = AsyncMock()
    controller.list_items.return_value = _items(100, 51)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items(
            target_page_index=2, explicit_page_change=True
        ) is True
        entered = asyncio.Event()
        release = asyncio.Event()

        async def pending_first_page(**kwargs):
            assert kwargs["offset"] == 0
            assert kwargs["status"] == "new"
            entered.set()
            await release.wait()
            return _items(0, 3)

        controller.list_items.reset_mock()
        controller.list_items.side_effect = pending_first_page
        screen.handle_items_filter_changed(ItemsFilterChanged("unread", ""))

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert screen._items_page_index == 0
        assert screen._items_has_next is False
        assert screen._items_page_loading is True
        assert pane.page_number == 1
        assert pane.query_one("#items-page-previous", Button).disabled is True
        assert pane.query_one("#items-page-next", Button).disabled is True
        await _wait_until(pilot, entered.is_set)
        release.set()
        await _wait_until(pilot, lambda: not screen._items_page_loading)


@pytest.mark.asyncio
async def test_search_edit_stays_on_logical_page_one_through_debounce_and_load():
    controller = AsyncMock()
    old_rows = _items(100, 51)
    old_rows[0]["title"] = "Needle provisional"
    controller.list_items.return_value = old_rows

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items(
            target_page_index=2, explicit_page_change=True
        ) is True
        prior_rows = screen._loaded_items
        open_item = prior_rows[0]
        screen._selected_content_item = open_item
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item
        entered = asyncio.Event()
        release = asyncio.Event()

        async def pending_search(**kwargs):
            assert kwargs["offset"] == 0
            assert kwargs["limit"] == 51
            assert kwargs["search"] == "needle"
            entered.set()
            await release.wait()
            return _items(0, 1)

        controller.list_items.reset_mock()
        controller.list_items.side_effect = pending_search
        screen.handle_items_filter_changed(ItemsFilterChanged("all", "needle"))

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert screen._items_page_index == 0
        assert screen._items_page_loading is True
        assert screen._items_has_next is False
        assert screen._loaded_items is prior_rows
        assert content.item is open_item
        assert pane.search_results_authoritative is False
        assert pane.query_one("#items-page-previous", Button).disabled is True
        assert pane.query_one("#items-page-next", Button).disabled is True
        await pilot.pause(0.1)
        assert controller.list_items.await_count == 0
        await _wait_until(pilot, entered.is_set)
        assert screen._items_page_index == 0
        assert screen._items_page_loading is True
        assert content.item is open_item
        release.set()
        await _wait_until(pilot, lambda: not screen._items_page_loading)


@pytest.mark.asyncio
async def test_refresh_reloads_the_committed_current_page():
    controller = AsyncMock()
    controller.list_items.return_value = _items(100, 51)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items(
            target_page_index=2, explicit_page_change=True
        ) is True
        controller.list_items.reset_mock()
        controller.list_items.return_value = _items(200, 1)

        screen.handle_refresh_items_requested(RefreshItemsRequested())
        await _wait_until(pilot, lambda: controller.list_items.await_count == 1)
        await _wait_until(pilot, lambda: not screen._items_page_loading)

        assert controller.list_items.await_args.kwargs["offset"] == 100
        assert screen._items_page_index == 2


@pytest.mark.asyncio
async def test_tree_scope_change_resets_and_reloads_page_one():
    controller = AsyncMock()
    controller.list_items.return_value = _items(100, 51)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items(
            target_page_index=2, explicit_page_change=True
        ) is True
        controller.list_items.reset_mock()
        controller.list_items.return_value = _items(0, 1)

        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=7)
        await _wait_until(pilot, lambda: controller.list_items.await_count == 1)
        await _wait_until(pilot, lambda: not screen._items_page_loading)

        assert controller.list_items.await_args.kwargs["offset"] == 0
        assert controller.list_items.await_args.kwargs["watchlist_id"] == 7
        assert screen._items_page_index == 0


@pytest.mark.asyncio
async def test_backend_change_reloads_read_but_not_a_hidden_read_section():
    controller = AsyncMock()
    controller.list_items.return_value = _items(100, 51)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items(
            target_page_index=2, explicit_page_change=True
        ) is True
        controller.list_items.reset_mock()
        controller.list_items.return_value = _items(0, 1)

        screen.runtime_backend = "server"
        await _wait_until(pilot, lambda: controller.list_items.await_count == 1)
        await _wait_until(pilot, lambda: not screen._items_page_loading)
        assert controller.list_items.await_args.kwargs["runtime_backend"] == "server"
        assert controller.list_items.await_args.kwargs["offset"] == 0
        assert screen._items_page_index == 0

        screen.active_section = "sources"
        await pilot.pause(0.1)
        screen._items_page_index = 2
        screen._items_has_next = True
        controller.list_items.reset_mock()
        screen.runtime_backend = "local"
        await pilot.pause(0.2)
        assert controller.list_items.await_count == 0
        assert screen._items_page_index == 0
        assert screen._items_page_loading is False


@pytest.mark.asyncio
async def test_selection_during_search_debounce_keeps_prior_committed_key():
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 2)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        prior_key = screen._items_committed_page_key
        item = screen._loaded_items[0]
        item["title"] = "Needle"
        screen._mark_item_read_on_open = Mock()
        controller.get_item_content.return_value = "Fetched"

        screen.handle_items_filter_changed(ItemsFilterChanged("all", "needle"))
        await screen.handle_item_selected(ItemSelected(item))

        assert screen._items_page_index == 0
        assert screen._items_committed_page_key == prior_key
        assert screen._selected_content_page_key == prior_key
        await pilot.pause(0.35)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ["backend", "scope", "status", "search"])
async def test_query_context_changes_do_not_pin_the_open_item(changed_field):
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 2)

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._load_items() is True
        open_item = screen._loaded_items[0]
        screen._selected_content_item = open_item
        screen._selected_content_page_key = screen._items_committed_page_key
        if changed_field == "backend":
            screen.__dict__["_reactive_runtime_backend"] = "server"
        elif changed_field == "scope":
            screen.__dict__["_reactive_tree_scope"] = TreeScope(
                kind="watchlist", watchlist_id=7
            )
        elif changed_field == "status":
            screen._items_status_filter = "unread"
        else:
            screen._items_search_query = "different"
        screen._reset_items_paging_for_context(loading=True)
        controller.list_items.return_value = _items(100, 2)

        assert await screen._load_items() is True

        assert str(open_item["id"]) not in {
            str(item["id"]) for item in screen._loaded_items
        }


@pytest.mark.asyncio
async def test_content_only_backend_match_survives_authoritative_search_load():
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 1)

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._load_items() is True
        result = _item(77)
        result.pop("content")
        controller.list_items.return_value = [result]
        screen._items_search_query = "full-body-only-token"
        screen._reset_items_paging_for_context(loading=True)

        assert await screen._load_items() is True

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert pane.search_results_authoritative is True
        assert [str(item["id"]) for item in pane.displayed_items()] == ["77"]
        article_rows = list(pane.query(".article-row"))
        assert len(article_rows) == 1
        # task-15776: `.article-row` is the ListItem itself now, so the
        # display check reads the row directly, not a wrapper parent.
        assert article_rows[0].display is True


@pytest.mark.asyncio
async def test_backend_search_is_authoritative_before_row_presentation(monkeypatch):
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 1)

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._load_items() is True
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        original_apply = pane.apply_page_items
        presentation_started = asyncio.Event()

        async def assert_authoritative_before_apply(items, *, focus_first=False):
            if str(items[0]["id"]) == "77":
                assert pane.search_results_authoritative is True
                presentation_started.set()
            await original_apply(items, focus_first=focus_first)

        monkeypatch.setattr(
            pane, "apply_page_items", assert_authoritative_before_apply
        )
        result = _item(77)
        result.pop("content")
        controller.list_items.return_value = [result]
        screen._items_search_query = "full-body-only-token"
        screen._reset_items_paging_for_context(loading=True)

        assert await screen._load_items() is True
        assert presentation_started.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["cancel", "exception", "stale"])
async def test_interrupted_search_presentation_restores_provisional_authority(
    monkeypatch, interruption
):
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 1)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        prior_rows = screen._loaded_items
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        original_apply = pane.apply_page_items
        presentation_started = asyncio.Event()
        release_presentation = asyncio.Event()

        async def interrupt_target_apply(items, *, focus_first=False):
            if str(items[0]["id"]) == "77":
                assert pane.search_results_authoritative is True
                presentation_started.set()
                if interruption == "exception":
                    await original_apply(items, focus_first=focus_first)
                    raise RuntimeError("presentation failed")
                await release_presentation.wait()
            await original_apply(items, focus_first=focus_first)

        monkeypatch.setattr(pane, "apply_page_items", interrupt_target_apply)
        controller.list_items.return_value = [_item(77)]
        screen._items_search_query = "new search"
        screen._reset_items_paging_for_context(loading=True)
        load = asyncio.create_task(screen._load_items())
        await _wait_until(pilot, presentation_started.is_set)

        if interruption == "cancel":
            load.cancel()
            with pytest.raises(asyncio.CancelledError):
                await load
        elif interruption == "stale":
            screen._items_search_query = "newer search"
            screen._reset_items_paging_for_context(loading=True)
            release_presentation.set()
            assert await load is False
        else:
            assert await load is False

        assert screen._items_search_results_authoritative is False
        assert pane.search_results_authoritative is False
        assert pane.items is prior_rows


@pytest.mark.asyncio
async def test_stale_refresh_rollback_keeps_new_context_provisional(monkeypatch):
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 1)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        prior_rows = screen._loaded_items
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        original_apply = pane.apply_page_items
        presentation_started = asyncio.Event()
        release_presentation = asyncio.Event()

        async def block_refresh_presentation(items, *, focus_first=False):
            await original_apply(items, focus_first=focus_first)
            if str(items[0]["id"]) == "77":
                presentation_started.set()
                await release_presentation.wait()

        monkeypatch.setattr(pane, "apply_page_items", block_refresh_presentation)
        controller.list_items.return_value = [_item(77)]
        load = asyncio.create_task(screen._load_items())
        await _wait_until(pilot, presentation_started.is_set)
        assert screen._items_search_results_authoritative is True
        assert pane.search_results_authoritative is True

        screen._items_search_query = "new context"
        screen._reset_items_paging_for_context(loading=True)
        assert screen._items_search_results_authoritative is False
        assert pane.search_results_authoritative is False
        release_presentation.set()

        assert await load is False
        assert screen._items_search_results_authoritative is False
        assert pane.search_results_authoritative is False
        assert pane.items is prior_rows


@pytest.mark.asyncio
async def test_context_round_trip_does_not_coalesce_with_cancelled_rollback(
    monkeypatch,
):
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 1)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        prior_rows = screen._loaded_items
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        original_apply = pane.apply_page_items
        presentation_started = asyncio.Event()
        rollback_started = asyncio.Event()
        release_rollback = asyncio.Event()
        never_release_presentation = asyncio.Event()
        replacement_worker_started = asyncio.Event()
        replacement_results = []

        async def block_presentation_and_rollback(items, *, focus_first=False):
            first_id = str(items[0]["id"])
            if first_id == "100":
                await original_apply(items, focus_first=focus_first)
                presentation_started.set()
                await never_release_presentation.wait()
                return
            if items is prior_rows and presentation_started.is_set():
                rollback_started.set()
                await release_rollback.wait()
            await original_apply(items, focus_first=focus_first)

        monkeypatch.setattr(pane, "apply_page_items", block_presentation_and_rollback)
        a_backend_calls = 0
        backend_calls = []

        async def context_pages(**kwargs):
            nonlocal a_backend_calls
            backend_calls.append(kwargs)
            a_backend_calls += 1
            return _items(100 if a_backend_calls == 1 else 200, 1)

        async def replacement_context_a_load():
            replacement_worker_started.set()
            replacement_results.append(await screen._load_items())

        controller.list_items.reset_mock()
        controller.list_items.side_effect = context_pages
        screen.run_worker(
            screen._load_items(), exclusive=True, group="wc_items"
        )
        await _wait_until(pilot, presentation_started.is_set)

        screen._items_search_query = "temporary"
        screen._reset_items_paging_for_context(loading=True)
        screen._items_search_query = ""
        screen._reset_items_paging_for_context(loading=True)
        screen.run_worker(
            replacement_context_a_load(), exclusive=True, group="wc_items"
        )
        await _wait_until(pilot, rollback_started.is_set)
        await _wait_until(pilot, replacement_worker_started.is_set)
        release_rollback.set()

        await _wait_until(
            pilot,
            lambda: not screen._items_page_loading
            and [str(item["id"]) for item in screen._loaded_items] == ["200"],
        )
        assert replacement_results == [True]
        assert a_backend_calls == 2
        assert all("search" not in call for call in backend_calls)


@pytest.mark.asyncio
async def test_older_context_result_cannot_paint_after_newer_result():
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 1)

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items() is True
        old_started = asyncio.Event()
        old_result = asyncio.get_running_loop().create_future()

        async def controlled(**kwargs):
            if "search" not in kwargs:
                old_started.set()
                return await old_result
            return _items(200, 1)

        controller.list_items.reset_mock()
        controller.list_items.side_effect = controlled
        older = asyncio.create_task(screen._load_items())
        await _wait_until(pilot, old_started.is_set)
        screen._items_search_query = "new context"
        screen._reset_items_paging_for_context(loading=True)
        assert await screen._load_items() is True
        old_result.set_result(_items(100, 1))
        assert await older is False

        assert [str(item["id"]) for item in screen._loaded_items] == ["200"]


@pytest.mark.asyncio
async def test_failed_search_keeps_provisional_rows_content_and_disabled_next():
    controller = AsyncMock()
    rows = _items(100, 51)
    rows[0]["title"] = "Needle provisional"
    controller.list_items.return_value = rows

    async with _open_screen(controller) as (screen, pilot):
        assert await screen._load_items(
            target_page_index=2, explicit_page_change=True
        ) is True
        prior_rows = screen._loaded_items
        open_item = prior_rows[0]
        screen._selected_content_item = open_item
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item

        controller.list_items.reset_mock()
        controller.list_items.side_effect = RuntimeError("search unavailable")
        screen.handle_items_filter_changed(ItemsFilterChanged("all", "needle"))
        await _wait_until(pilot, lambda: controller.list_items.await_count == 1)
        await _wait_until(pilot, lambda: not screen._items_page_loading)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert screen._items_page_index == 0
        assert screen._items_has_next is False
        assert screen._loaded_items is prior_rows
        assert pane.items is prior_rows
        assert pane.search_results_authoritative is False
        assert pane.has_previous is False
        assert pane.has_next is False
        assert content.item is open_item


@pytest.mark.asyncio
async def test_empty_nonfirst_refresh_walks_back_to_nearest_nonempty_page():
    controller = AsyncMock()
    controller.list_items.return_value = _items(100, 10)

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._load_items(
            target_page_index=2, explicit_page_change=True
        ) is True

        async def pages(**kwargs):
            return _items(0, 4) if kwargs["offset"] == 0 else []

        controller.list_items.reset_mock()
        controller.list_items.side_effect = pages
        assert await screen._load_items() is True

        assert [call.kwargs["offset"] for call in controller.list_items.await_args_list] == [
            100,
            50,
            0,
        ]
        assert screen._items_page_index == 0
        assert [str(item["id"]) for item in screen._loaded_items] == [
            str(index) for index in range(4)
        ]


@pytest.mark.asyncio
async def test_page_navigation_keys_never_fetch_or_reach_the_lookahead_row():
    controller = AsyncMock()
    controller.list_items.return_value = _items(0, 51)

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._load_items() is True
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        last_visible = screen._loaded_items[-1]
        screen._selected_content_item = last_visible
        pane.selected_item = last_visible
        controller.list_items.reset_mock()

        screen.action_next_item()
        screen.action_previous_item()
        screen.handle_next_unread_requested(NextUnreadRequested())

        assert controller.list_items.await_count == 0
        assert "50" not in {str(item["id"]) for item in pane.displayed_items()}


@pytest.mark.asyncio
async def test_rebuilt_read_pane_is_seeded_with_committed_page_state():
    controller = AsyncMock()
    controller.list_items.return_value = _items(100, 51)

    async with _open_screen(controller) as (screen, _pilot):
        assert await screen._load_items(
            target_page_index=2, explicit_page_change=True
        ) is True
        original = screen.query_one("#watchlists-items-pane", ArticleListPane)

        await screen.query_one("#wl-workbench").refresh_region_content(Region.ITEMS)
        replacement = screen.query_one("#watchlists-items-pane", ArticleListPane)

        assert replacement is not original
        assert replacement.items is screen._loaded_items
        assert replacement.page_number == 3
        assert replacement.has_previous is True
        assert replacement.has_next is True
        assert replacement.page_loading is False
