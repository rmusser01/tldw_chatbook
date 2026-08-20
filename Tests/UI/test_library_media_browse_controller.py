"""Non-visual Library Media page-owner contracts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from tldw_chatbook.Library.library_media_state import MediaBrowseScope
from tldw_chatbook.UI.Library_Modules.library_media_browse_controller import (
    LibraryMediaBrowseController,
)


def _item(media_id: int) -> dict[str, object]:
    return {
        "id": f"local:media:{media_id}",
        "backing_media_id": media_id,
        "title": f"Media {media_id}",
        "media_type": "video",
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


class _Screen:
    def __init__(self) -> None:
        self.pending: list[Awaitable[None]] = []

    def run_worker(self, work: Awaitable[None], **_kwargs: Any) -> Awaitable[None]:
        self.pending.append(work)
        return work


class _Service:
    def __init__(self, *pages: object, types: object = ("audio", "video")) -> None:
        self.pages = list(pages)
        self.types = types
        self.search_calls: list[dict[str, Any]] = []
        self.type_calls: list[dict[str, Any]] = []

    async def search_media(self, **kwargs: Any) -> object:
        self.search_calls.append(kwargs)
        outcome = self.pages.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def list_library_media_types(self, **kwargs: Any) -> object:
        self.type_calls.append(kwargs)
        if isinstance(self.types, Exception):
            raise self.types
        return self.types


async def _call(fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
    kwargs.pop("isolate_in_worker")
    return await fn(**kwargs)


def _controller(
    screen: _Screen,
    service: _Service,
    *,
    sync: Callable[..., None] = lambda *_args: None,
    active: Callable[[], bool] = lambda: True,
) -> LibraryMediaBrowseController:
    return LibraryMediaBrowseController(
        screen=screen,
        run_service_call=lambda: _call,
        media_service=lambda: service,
        sync_view=lambda: sync,
        request_is_active=active,
    )


@pytest.mark.asyncio
async def test_controller_sends_exact_summary_coordinates_and_full_scope() -> None:
    screen = _Screen()
    service = _Service(_page(2, 21))
    controller = _controller(screen, service)
    scope = MediaBrowseScope(
        query="needle", media_type="video", sort_by="title_asc", page=2
    )

    controller.request(scope, focus_identity=None)
    await screen.pending.pop()

    assert service.search_calls == [
        {
            "mode": "local",
            "query": "needle",
            "limit": 20,
            "offset": 20,
            "library_summary": True,
            "sort_by": "title_asc",
            "media_types": ["video"],
        }
    ]
    assert controller.applied_scope == scope
    assert controller.mutation_refresh_scope == scope
    assert controller.scope_for_page(1) == scope.with_page(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("media_type", ["All", "all", "ALL"])
async def test_controller_filters_literal_all_type_values(media_type: str) -> None:
    screen = _Screen()
    service = _Service(_page(1, 1))
    controller = _controller(screen, service)

    controller.request(MediaBrowseScope(media_type=media_type), focus_identity=None)
    await screen.pending.pop()

    assert service.search_calls[0]["media_types"] == [media_type]


@pytest.mark.asyncio
async def test_controller_omits_type_filter_for_unfiltered_scope() -> None:
    screen = _Screen()
    service = _Service(_page(1, 1))
    controller = _controller(screen, service)

    controller.request(MediaBrowseScope(), focus_identity=None)
    await screen.pending.pop()

    assert "media_types" not in service.search_calls[0]


@pytest.mark.asyncio
async def test_controller_retains_applied_rows_while_loading_and_after_page_failure() -> (
    None
):
    screen = _Screen()
    service = _Service(_page(1, 40), RuntimeError("private"))
    controller = _controller(screen, service)
    controller.request(MediaBrowseScope(), focus_identity=None)
    await screen.pending.pop()
    retained = controller.retained_items

    controller.request(MediaBrowseScope(page=2), focus_identity=None)
    assert controller.loading is True
    assert controller.retained_items is retained
    assert controller.pager.status_copy == "Loading page 2…"
    await screen.pending.pop()

    assert controller.retained_items is retained
    assert controller.freshness == "fresh"
    assert controller.error_copy == "Couldn't load page 2."
    assert controller.pager.range_copy == "1-20 of 40"


@pytest.mark.asyncio
async def test_scope_failure_copy_and_retry_use_requested_scope() -> None:
    screen = _Screen()
    requested = MediaBrowseScope(query="new")
    service = _Service(_page(1, 1), RuntimeError("private"), _page(1, 1))
    controller = _controller(screen, service)
    controller.request(MediaBrowseScope(), focus_identity=None)
    await screen.pending.pop()
    controller.request(requested, focus_identity=None)
    await screen.pending.pop()

    assert controller.error_copy == "Filter wasn't applied; showing previous results."
    assert controller.mutation_refresh_scope == MediaBrowseScope()
    controller.retry(focus_identity=None)
    await screen.pending.pop()
    assert controller.requested_scope == requested
    assert service.search_calls[-1]["query"] == "new"


@pytest.mark.asyncio
async def test_controller_clamps_once_and_keeps_original_requested_scope() -> None:
    screen = _Screen()
    service = _Service(_page(99, 45), _page(3, 45))
    controller = _controller(screen, service)
    requested = MediaBrowseScope(page=99)

    controller.request(requested, focus_identity=None)
    await screen.pending.pop()

    assert len(service.search_calls) == 2
    assert service.search_calls[-1]["offset"] == 40
    assert controller.requested_scope == requested
    assert controller.applied_scope == MediaBrowseScope(page=3)
    assert controller.pager.page_copy == "Page 3 of 3"


@pytest.mark.asyncio
async def test_second_shrink_goes_stale_without_a_third_read() -> None:
    screen = _Screen()
    service = _Service(_page(2, 40), _page(99, 45), _page(3, 20))
    controller = _controller(screen, service)
    controller.request(MediaBrowseScope(page=2), focus_identity=None)
    await screen.pending.pop()
    retained = controller.retained_items

    controller.request(MediaBrowseScope(page=99), focus_identity=None)
    await screen.pending.pop()

    assert len(service.search_calls) == 3
    assert controller.retained_items is retained
    assert controller.freshness == "stale"
    assert controller.stale_copy
    assert controller.pager.retry_visible is True


@pytest.mark.asyncio
async def test_late_page_generation_cannot_replace_current_request() -> None:
    screen = _Screen()
    service = _Service(_page(1, 1))
    controller = _controller(screen, service)
    controller.request(MediaBrowseScope(query="old"), focus_identity=None)
    late = screen.pending.pop()
    controller.begin(MediaBrowseScope(query="new"))
    await late

    assert controller.requested_scope.query == "new"
    assert controller.applied_result is None
    assert controller.loading is True


@pytest.mark.asyncio
async def test_facets_are_complete_sorted_unique_and_independently_fenced() -> None:
    screen = _Screen()
    service = _Service(
        _page(1, 1), types=("video", "All", "all", "ALL", "audio", "video")
    )
    controller = _controller(screen, service)
    controller.request_facets(fingerprint="old")
    late = screen.pending.pop()
    controller.invalidate_facets(fingerprint="new")
    await late
    assert controller.type_options == ()

    controller.request_facets(fingerprint="new")
    await screen.pending.pop()
    assert controller.type_options == ("ALL", "All", "all", "audio", "video")
    assert controller.facet_fingerprint == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [("video",), RuntimeError("private")])
async def test_accepted_facet_transitions_publish_loading_and_outcome(
    outcome: object,
) -> None:
    screen = _Screen()
    synced: list[str] = []
    controller = _controller(
        screen,
        _Service(types=outcome),
        sync=lambda _focus: synced.append(
            "loading" if controller.facet_loading else "settled"
        ),
    )

    controller.request_facets(fingerprint="current")
    assert synced == ["loading"]
    await screen.pending.pop()

    assert synced == ["loading", "settled"]
    if isinstance(outcome, Exception):
        assert controller.facet_error_copy
    else:
        assert controller.type_options == ("video",)


@pytest.mark.asyncio
async def test_stale_facet_outcome_does_not_publish() -> None:
    screen = _Screen()
    synced: list[str] = []
    controller = _controller(
        screen,
        _Service(types=("video",)),
        sync=lambda _focus: synced.append("sync"),
    )

    controller.request_facets(fingerprint="old")
    stale = screen.pending.pop()
    controller.invalidate_facets(fingerprint="new")
    await stale

    assert synced == ["sync"]


def test_inactive_facet_request_does_not_publish() -> None:
    screen = _Screen()
    synced: list[str] = []
    controller = _controller(
        screen,
        _Service(types=("video",)),
        sync=lambda _focus: synced.append("sync"),
        active=lambda: False,
    )

    assert controller.request_facets(fingerprint="inactive") is None
    assert synced == []
    assert screen.pending == []


@pytest.mark.asyncio
async def test_navigation_invalidation_fences_page_and_facets() -> None:
    screen = _Screen()
    service = _Service(_page(1, 1), types=("video",))
    controller = _controller(screen, service)
    controller.request(MediaBrowseScope(), focus_identity=None)
    page_work = screen.pending.pop()
    controller.request_facets(fingerprint="before")
    facet_work = screen.pending.pop()
    controller.invalidate()
    await page_work
    await facet_work

    assert controller.applied_result is None
    assert controller.type_options == ()


def test_retain_stale_items_does_not_forge_exact_metadata() -> None:
    screen = _Screen()
    controller = _controller(screen, _Service(_page(1, 1)))
    with pytest.raises(ValueError, match="before"):
        controller.retain_stale_items((), stale_copy="Changed")


@pytest.mark.asyncio
async def test_committed_mutation_reconciles_retained_without_forging_envelope() -> (
    None
):
    screen = _Screen()
    service = _Service(_page(2, 40), RuntimeError("refresh failed"))
    controller = _controller(screen, service)
    scope = MediaBrowseScope(page=2)
    controller.request(scope, focus_identity=None)
    await screen.pending.pop()
    applied = controller.applied_result

    assert controller.begin_mutation() == scope
    controller.reconcile_committed_mutation(
        remove_ids=("local:media:21",),
        upsert_items=(_item(99),),
    )

    assert controller.applied_result is applied
    assert [item["id"] for item in controller.retained_items] == [
        "local:media:99",
        *(f"local:media:{media_id}" for media_id in range(22, 41)),
    ]
    assert controller.freshness == "stale"
    assert controller.pager.title_count is None
    assert controller.pager.range_copy == "List may be out of date"

    controller.request(scope, focus_identity=None)
    await screen.pending.pop()

    assert service.search_calls[-1]["offset"] == 20
    assert controller.applied_result is applied
    assert controller.freshness == "stale"
    assert controller.pager.retry_visible is True


@pytest.mark.asyncio
async def test_mutation_begin_fences_page_and_facet_results_before_write() -> None:
    screen = _Screen()
    service = _Service(_page(1, 1), types=("obsolete",))
    controller = _controller(screen, service)
    controller.request(MediaBrowseScope(), focus_identity=None)
    late_page = screen.pending.pop()
    controller.request_facets(fingerprint="before")
    late_facets = screen.pending.pop()

    assert controller.begin_mutation() == MediaBrowseScope()
    await late_page
    await late_facets

    assert controller.applied_result is None
    assert controller.type_options == ()
