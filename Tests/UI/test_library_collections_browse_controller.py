"""Non-visual Library Collections page-owner contracts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from tldw_chatbook.Library.library_collections_state import CollectionBrowseScope
from tldw_chatbook.UI.Library_Modules.library_collections_browse_controller import (
    LibraryCollectionsBrowseController,
)


def _item(collection_id: int | str) -> dict[str, object]:
    return {
        "collection_id": f"collection-{collection_id}",
        "name": f"Collection {collection_id}",
        "description": "",
        "item_count": 0,
        "created_at": "2026-08-16T00:00:00+00:00",
        "updated_at": "2026-08-16T00:00:00+00:00",
    }


def _page(page: int, total: int) -> dict[str, object]:
    offset = (page - 1) * 20
    count = min(20, max(total - offset, 0))
    return {
        "items": [_item(offset + index + 1) for index in range(count)],
        "total": total,
        "limit": 20,
        "offset": offset,
    }


def _locator(target_id: str, *, page: int, rank: int, total: int) -> dict[str, object]:
    payload = _page(page, total)
    target_index = rank - payload["offset"]
    payload.update(
        page=page,
        target_id=target_id,
        target_rank=rank,
        target_index=target_index,
    )
    payload["items"][target_index] = _item(target_id.removeprefix("collection-"))
    return payload


class _Screen:
    def __init__(self) -> None:
        self.pending: list[Awaitable[None]] = []

    def run_worker(self, work: Awaitable[None], **_kwargs: Any) -> Awaitable[None]:
        self.pending.append(work)
        return work


class _Service:
    def __init__(
        self,
        *pages: object,
        locator_outcomes: tuple[object, ...] = (),
    ) -> None:
        self.pages = list(pages)
        self.locator_outcomes = list(locator_outcomes)
        self.page_calls: list[dict[str, Any]] = []
        self.locator_calls: list[tuple[str, dict[str, Any]]] = []

    async def list_library_collections(self, **kwargs: Any) -> object:
        self.page_calls.append(kwargs)
        outcome = self.pages.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def locate_library_collection_page(
        self, collection_id: str, **kwargs: Any
    ) -> object:
        self.locator_calls.append((collection_id, kwargs))
        outcome = self.locator_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def _call(fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
    kwargs.pop("isolate_in_worker")
    return await fn(**kwargs)


def _controller(
    screen: _Screen,
    service: _Service,
    *,
    sync: Callable[..., None] = lambda *_args: None,
    active: Callable[[], bool] = lambda: True,
) -> LibraryCollectionsBrowseController:
    return LibraryCollectionsBrowseController(
        screen=screen,
        run_service_call=lambda: _call,
        collections_service=lambda: service,
        sync_view=lambda: sync,
        request_is_active=active,
    )


@pytest.mark.asyncio
async def test_page_request_uses_exact_fixed_coordinates_and_forwards_focus() -> None:
    screen = _Screen()
    service = _Service(_page(2, 21))
    synced: list[str | None] = []
    controller = _controller(screen, service, sync=synced.append)

    controller.request(CollectionBrowseScope(page=2), focus_identity="next")
    await screen.pending.pop()

    assert service.page_calls == [{"limit": 20, "offset": 20}]
    assert controller.applied_scope == CollectionBrowseScope(page=2)
    assert controller.scope_for_page(1) == CollectionBrowseScope(page=1)
    assert synced == ["next", "next"]


@pytest.mark.asyncio
async def test_page_failure_retains_last_good_rows_and_retry_targets_request() -> None:
    screen = _Screen()
    service = _Service(_page(1, 40), RuntimeError("private"), _page(2, 40))
    controller = _controller(screen, service)
    controller.request(CollectionBrowseScope(), focus_identity=None)
    await screen.pending.pop()
    retained = controller.retained_items

    controller.request(CollectionBrowseScope(page=2), focus_identity="next")
    assert controller.loading is True
    assert controller.retained_items is retained
    assert controller.pager.status_copy == "Loading page 2…"
    await screen.pending.pop()

    assert controller.retained_items is retained
    assert controller.freshness == "fresh"
    assert controller.error_copy == "Couldn't load page 2."
    assert controller.pager.range_copy == "1-20 of 40"

    controller.retry(focus_identity="retry")
    await screen.pending.pop()
    assert service.page_calls[-1] == {"limit": 20, "offset": 20}
    assert controller.applied_scope == CollectionBrowseScope(page=2)


@pytest.mark.asyncio
async def test_initial_failure_does_not_fabricate_total() -> None:
    screen = _Screen()
    controller = _controller(screen, _Service(RuntimeError("private")))

    controller.request(CollectionBrowseScope(), focus_identity=None)
    await screen.pending.pop()

    assert controller.applied_result is None
    assert controller.freshness == "uninitialized"
    assert controller.pager.title_count is None
    assert controller.pager.range_copy == "No page loaded · Total unavailable"
    assert controller.pager.retry_visible is True


@pytest.mark.asyncio
async def test_late_generation_and_inactive_route_cannot_publish() -> None:
    screen = _Screen()
    active = True
    controller = _controller(screen, _Service(_page(1, 1)), active=lambda: active)
    controller.request(CollectionBrowseScope(), focus_identity=None)
    late = screen.pending.pop()
    controller.begin(CollectionBrowseScope(page=2))
    await late

    assert controller.applied_result is None
    assert controller.requested_scope == CollectionBrowseScope(page=2)

    active = False
    controller.invalidate()
    assert controller.request(CollectionBrowseScope(), focus_identity=None) is None
    assert screen.pending == []


@pytest.mark.asyncio
async def test_controller_clamps_once_to_ranked_final_page() -> None:
    screen = _Screen()
    service = _Service(_page(99, 45), _page(3, 45))
    controller = _controller(screen, service)
    requested = CollectionBrowseScope(page=99)

    controller.request(requested, focus_identity="next")
    await screen.pending.pop()

    assert [call["offset"] for call in service.page_calls] == [1960, 40]
    assert controller.requested_scope == requested
    assert controller.applied_scope == CollectionBrowseScope(page=3)
    assert controller.pager.page_copy == "Page 3 of 3"


@pytest.mark.asyncio
async def test_second_shrink_retains_stale_rows_without_third_read() -> None:
    screen = _Screen()
    service = _Service(_page(2, 40), _page(99, 45), _page(3, 20))
    controller = _controller(screen, service)
    controller.request(CollectionBrowseScope(page=2), focus_identity=None)
    await screen.pending.pop()
    retained = controller.retained_items

    controller.request(CollectionBrowseScope(page=99), focus_identity="next")
    await screen.pending.pop()

    assert [call["offset"] for call in service.page_calls] == [20, 1960, 40]
    assert controller.retained_items is retained
    assert controller.freshness == "stale"
    assert controller.stale_copy == "Source changed again; try again."
    assert controller.pager.title_count is None
    assert controller.pager.range_copy == "List may be out of date"
    assert controller.pager.retry_visible is True


@pytest.mark.asyncio
async def test_locator_applies_owning_page_and_exposes_target_identity() -> None:
    screen = _Screen()
    payload = _locator("collection-21", page=2, rank=20, total=45)
    service = _Service(locator_outcomes=(payload,))
    controller = _controller(screen, service)

    controller.request_locator("collection-21", focus_identity="create")
    await screen.pending.pop()

    assert service.locator_calls == [("collection-21", {"limit": 20})]
    assert controller.applied_scope == CollectionBrowseScope(page=2)
    assert controller.located_target_id == "collection-21"
    assert controller.freshness == "fresh"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [None, RuntimeError("private"), {"items": []}],
)
async def test_locator_failure_fails_closed_without_replacing_applied_page(
    outcome: object,
) -> None:
    screen = _Screen()
    service = _Service(_page(1, 1), locator_outcomes=(outcome,))
    controller = _controller(screen, service)
    controller.request(CollectionBrowseScope(), focus_identity=None)
    await screen.pending.pop()
    applied = controller.applied_result

    controller.request_locator("collection-99", focus_identity="rename")
    await screen.pending.pop()

    assert controller.applied_result is applied
    assert controller.located_target_id is None
    assert controller.error_copy == "Couldn't locate that Collection."


@pytest.mark.asyncio
async def test_retry_after_locator_failure_retries_the_stable_id() -> None:
    screen = _Screen()
    located = _locator("collection-21", page=2, rank=20, total=45)
    service = _Service(
        locator_outcomes=(RuntimeError("private"), located),
    )
    controller = _controller(screen, service)

    controller.request_locator("collection-21", focus_identity="rename")
    await screen.pending.pop()
    controller.retry(focus_identity="retry")
    await screen.pending.pop()

    assert service.locator_calls == [
        ("collection-21", {"limit": 20}),
        ("collection-21", {"limit": 20}),
    ]
    assert controller.applied_scope == CollectionBrowseScope(page=2)
    assert controller.located_target_id == "collection-21"


@pytest.mark.asyncio
async def test_begin_mutation_fences_older_page_before_write() -> None:
    screen = _Screen()
    controller = _controller(screen, _Service(_page(1, 1)))
    controller.request(CollectionBrowseScope(), focus_identity=None)
    late = screen.pending.pop()

    assert controller.begin_mutation() == CollectionBrowseScope()
    await late

    assert controller.applied_result is None


@pytest.mark.asyncio
async def test_committed_mutation_reconciles_known_rows_as_stale() -> None:
    screen = _Screen()
    service = _Service(_page(2, 40))
    controller = _controller(screen, service)
    controller.request(CollectionBrowseScope(page=2), focus_identity=None)
    await screen.pending.pop()
    applied = controller.applied_result

    controller.begin_mutation()
    controller.reconcile_committed_mutation(
        remove_ids=("collection-21",),
        upsert_items=(_item(99),),
    )

    assert controller.applied_result is applied
    assert [item["collection_id"] for item in controller.retained_items] == [
        "collection-99",
        *(f"collection-{value}" for value in range(22, 41)),
    ]
    assert controller.freshness == "stale"
    assert controller.pager.title_count is None
    assert controller.pager.retry_visible is True


def test_reconciliation_rejects_malformed_known_rows() -> None:
    controller = _controller(_Screen(), _Service())

    with pytest.raises(ValueError, match="exact summary keys"):
        controller.reconcile_committed_mutation(upsert_items=({"collection_id": "x"},))
