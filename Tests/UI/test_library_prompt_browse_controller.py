"""Contracts for non-visual Library Prompt browse orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from tldw_chatbook.Library.library_prompts_state import PromptBrowseScope
from tldw_chatbook.UI.Library_Modules.library_prompt_browse_controller import (
    LibraryPromptBrowseController,
)


class _WorkerScreen:
    def __init__(self) -> None:
        self.pending: list[Awaitable[None]] = []
        self.worker_kwargs: list[dict[str, Any]] = []

    def run_worker(self, work: Awaitable[None], **kwargs: Any) -> Awaitable[None]:
        self.pending.append(work)
        self.worker_kwargs.append(kwargs)
        return work


class _BrowseService:
    def __init__(self, *, name: str = "Prompt", error: bool = False) -> None:
        self.name = name
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def browse_prompts(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error:
            raise RuntimeError("private service failure")
        page_size = kwargs["page_size"]
        return {
            "items": [
                {
                    "id": "local:prompt:one",
                    "local_id": 1,
                    "name": self.name,
                    "backend": "local",
                }
            ],
            "total_items": 1,
            "total_pages": 1,
            "current_page": 1,
            "page": 1,
            "per_page": page_size,
        }


class _ScriptedBrowseService:
    def __init__(self, *outcomes: dict[str, Any] | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def browse_prompts(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _page_record(
    *,
    page: int,
    total: int,
    page_size: int = 20,
    name: str = "Prompt",
) -> dict[str, Any]:
    count = min(page_size, max(0, total - (page - 1) * page_size))
    first_id = (page - 1) * page_size + 1
    return {
        "items": [
            {
                "id": f"local:prompt:{local_id}",
                "local_id": local_id,
                "name": f"{name} {local_id}",
                "backend": "local",
            }
            for local_id in range(first_id, first_id + count)
        ],
        "total_items": total,
        "total_pages": (total + page_size - 1) // page_size if total else 0,
        "current_page": page,
        "page": page,
        "per_page": page_size,
    }


async def _run_service_call(
    callable_obj: Callable[..., Awaitable[Any]],
    **kwargs: Any,
) -> Any:
    kwargs.pop("isolate_in_worker")
    return await callable_obj(**kwargs)


def _controller(
    *,
    screen: _WorkerScreen,
    service: Callable[[], Any],
    sync_view: Callable[[], Callable[..., None]],
    active: Callable[[], bool] = lambda: True,
    run_service_call: Callable[[], Callable[..., Awaitable[Any]]] = lambda: (
        _run_service_call
    ),
) -> LibraryPromptBrowseController:
    return LibraryPromptBrowseController(
        screen=screen,
        run_service_call=run_service_call,
        prompt_service=service,
        sync_view=sync_view,
        request_is_active=active,
    )


@pytest.mark.asyncio
async def test_controller_sends_explicit_library_twenty_row_default() -> None:
    screen = _WorkerScreen()
    service = _BrowseService()
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
    )

    controller.request(PromptBrowseScope(), focus_identity=None)
    await screen.pending.pop()

    assert service.calls == [
        {
            "mode": "local",
            "query": "",
            "collection_id": None,
            "sort_by": "last_modified",
            "sort_order": "desc",
            "page": 1,
            "page_size": 20,
        }
    ]


def test_controller_starts_without_an_applied_prompt_page() -> None:
    controller = _controller(
        screen=_WorkerScreen(),
        service=lambda: _BrowseService(),
        sync_view=lambda: lambda _result, _focus: None,
    )

    assert controller.applied_result is None
    assert controller.retained_items == ()
    assert controller.visible_result is controller.result
    assert controller.freshness == "uninitialized"
    assert controller.pager.title_count is None
    assert controller.pager.range_copy == "Loading page 1…"


@pytest.mark.asyncio
async def test_controller_retains_applied_rows_during_loading_and_page_failure() -> (
    None
):
    screen = _WorkerScreen()
    service = _ScriptedBrowseService(
        _page_record(page=2, total=60),
        RuntimeError("private page failure"),
    )
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
    )
    controller.request(PromptBrowseScope(page=2), focus_identity=None)
    await screen.pending.pop()
    applied = controller.applied_result
    retained = controller.retained_items

    controller.request(PromptBrowseScope(page=3), focus_identity=None)

    assert controller.scope.page == 3
    assert controller.result.status == "loading"
    assert controller.applied_result is applied
    assert controller.retained_items is retained
    assert controller.visible_result is applied
    assert controller.pager.status_copy == "Loading page 3…"

    await screen.pending.pop()

    assert controller.result.status == "error"
    assert controller.applied_result is applied
    assert controller.retained_items is retained
    assert controller.freshness == "fresh"
    assert controller.pager.range_copy == "21-40 of 60"
    assert controller.pager.status_copy == "Couldn't load page 3."
    assert controller.pager.retry_visible is True


@pytest.mark.asyncio
async def test_controller_failed_scope_change_keeps_full_applied_scope_for_paging() -> (
    None
):
    screen = _WorkerScreen()
    applied_scope = PromptBrowseScope(
        query="old",
        collection_id=7,
        sort_by="name",
        sort_order="asc",
        page=3,
    )
    requested_scope = PromptBrowseScope(
        query="new",
        collection_id=9,
        sort_by="last_modified",
        sort_order="desc",
        page=1,
    )
    service = _ScriptedBrowseService(
        _page_record(page=3, total=60, name="Old"),
        RuntimeError("private filter failure"),
    )
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
    )
    controller.request(applied_scope, focus_identity=None)
    await screen.pending.pop()

    controller.request(requested_scope, focus_identity=None)
    await screen.pending.pop()

    previous = controller.scope_for_page(2)
    assert controller.scope == requested_scope
    assert controller.applied_result is not None
    assert controller.applied_result.scope == applied_scope
    assert previous == PromptBrowseScope(
        query="old",
        collection_id=7,
        sort_by="name",
        sort_order="asc",
        page=2,
    )
    assert controller.mutation_refresh_scope == applied_scope
    assert (
        controller.pager.status_copy
        == "Filter wasn't applied; showing previous results."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested_scope",
    [
        PromptBrowseScope(query="new", page=1),
        PromptBrowseScope(collection_id=9, page=1),
        PromptBrowseScope(sort_by="name", sort_order="asc", page=1),
    ],
)
async def test_controller_classifies_each_prompt_scope_failure(
    requested_scope: PromptBrowseScope,
) -> None:
    screen = _WorkerScreen()
    service = _ScriptedBrowseService(
        _page_record(page=2, total=40),
        RuntimeError("private scope failure"),
    )
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
    )
    controller.request(PromptBrowseScope(page=2), focus_identity=None)
    await screen.pending.pop()

    controller.request(requested_scope, focus_identity=None)
    await screen.pending.pop()

    assert (
        controller.pager.status_copy
        == "Filter wasn't applied; showing previous results."
    )


@pytest.mark.asyncio
async def test_controller_applies_coherent_prompt_clamp_once() -> None:
    screen = _WorkerScreen()
    service = _ScriptedBrowseService(_page_record(page=3, total=45))
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
    )
    requested = PromptBrowseScope(page=99)

    controller.request(requested, focus_identity=None)
    await screen.pending.pop()

    assert len(service.calls) == 1
    assert controller.scope == requested
    assert controller.result.page == 3
    assert controller.applied_result is controller.result
    assert controller.scope_for_page(2) == PromptBrowseScope(page=2)
    assert controller.mutation_refresh_scope == PromptBrowseScope(page=3)
    assert controller.pager.page_copy == "Page 3 of 3"


@pytest.mark.asyncio
async def test_controller_initial_failure_is_uninitialized_and_retry_uses_requested_scope() -> (
    None
):
    screen = _WorkerScreen()
    service = _ScriptedBrowseService(
        RuntimeError("private failure"),
        _page_record(page=1, total=1, name="Recovered"),
    )
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
    )
    requested = PromptBrowseScope(query="retry me")
    controller.request(requested, focus_identity=None)
    await screen.pending.pop()
    failed_token = controller.result.request_token

    assert controller.freshness == "uninitialized"
    assert controller.applied_result is None
    assert controller.pager.status_copy
    assert controller.pager.retry_visible is True

    controller.retry(focus_identity=None)
    await screen.pending.pop()

    assert controller.result.request_token > failed_token
    assert controller.scope == requested
    assert controller.applied_result is controller.result
    assert controller.freshness == "fresh"
    assert controller.error_copy == ""


@pytest.mark.asyncio
async def test_controller_stale_retained_page_clears_only_after_success() -> None:
    screen = _WorkerScreen()
    service = _ScriptedBrowseService(
        _page_record(page=1, total=2),
        RuntimeError("refresh failed"),
        _page_record(page=1, total=1, name="Recovered"),
    )
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
    )
    controller.request(PromptBrowseScope(), focus_identity=None)
    await screen.pending.pop()
    [retained] = controller.retained_items[:1]
    controller.retain_stale_items((retained,), stale_copy="List may be out of date")

    controller.request(controller.mutation_refresh_scope, focus_identity=None)
    await screen.pending.pop()

    assert controller.freshness == "stale"
    assert controller.retained_items == (retained,)
    assert controller.stale_copy == "List may be out of date"
    assert controller.pager.retry_visible is True

    controller.retry(focus_identity=None)
    await screen.pending.pop()

    assert controller.freshness == "fresh"
    assert controller.stale_copy == ""
    assert len(controller.retained_items) == 1
    assert controller.retained_items[0]["name"] == "Recovered 1"


@pytest.mark.asyncio
@pytest.mark.parametrize("late_kind", ["success", "error"])
async def test_controller_rejects_late_success_and_error(late_kind: str) -> None:
    screen = _WorkerScreen()
    service = _BrowseService(error=late_kind == "error")
    synced: list[str] = []
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda result, _focus: synced.append(result.status),
    )
    stale_scope = PromptBrowseScope(query="stale")
    controller.request(stale_scope, focus_identity="library-prompts-filter")
    stale_work = screen.pending.pop()

    current_scope = PromptBrowseScope(query="current")
    controller.begin(current_scope)
    await stale_work

    assert controller.scope == current_scope
    assert controller.result.status == "loading"
    assert synced == ["loading"]


@pytest.mark.asyncio
@pytest.mark.parametrize("late_kind", ["success", "error"])
async def test_controller_late_result_cannot_replace_last_applied_page(
    late_kind: str,
) -> None:
    screen = _WorkerScreen()
    late: dict[str, Any] | Exception = (
        _page_record(page=2, total=40)
        if late_kind == "success"
        else RuntimeError("private late failure")
    )
    service = _ScriptedBrowseService(_page_record(page=1, total=40), late)
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
    )
    controller.request(PromptBrowseScope(page=1), focus_identity=None)
    await screen.pending.pop()
    applied = controller.applied_result
    retained = controller.retained_items

    controller.request(PromptBrowseScope(page=2), focus_identity=None)
    late_work = screen.pending.pop()
    current_scope = PromptBrowseScope(query="current")
    controller.begin(current_scope)
    await late_work

    assert controller.scope == current_scope
    assert controller.result.status == "loading"
    assert controller.applied_result is applied
    assert controller.retained_items is retained
    assert controller.freshness == "fresh"


@pytest.mark.asyncio
async def test_controller_rejects_result_after_navigation_invalidation() -> None:
    screen = _WorkerScreen()
    service = _BrowseService()
    active = True
    synced: list[str] = []
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda result, _focus: synced.append(result.status),
        active=lambda: active,
    )
    controller.request(PromptBrowseScope(), focus_identity=None)
    stale_work = screen.pending.pop()

    active = False
    invalidated_token = controller.invalidate()
    await stale_work

    assert controller.result.status == "loading"
    assert controller.result.request_token == invalidated_token
    assert synced == ["loading"]


@pytest.mark.asyncio
async def test_controller_navigation_rejection_preserves_last_applied_page() -> None:
    screen = _WorkerScreen()
    service = _ScriptedBrowseService(
        _page_record(page=1, total=40),
        _page_record(page=2, total=40),
    )
    active = True
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
        active=lambda: active,
    )
    controller.request(PromptBrowseScope(page=1), focus_identity=None)
    await screen.pending.pop()
    applied = controller.applied_result
    retained = controller.retained_items

    controller.request(PromptBrowseScope(page=2), focus_identity=None)
    late_work = screen.pending.pop()
    active = False
    controller.invalidate()
    await late_work

    assert controller.applied_result is applied
    assert controller.retained_items is retained
    assert controller.freshness == "fresh"


@pytest.mark.asyncio
async def test_controller_retry_uses_fresh_token_and_same_scope() -> None:
    screen = _WorkerScreen()
    failing = _BrowseService(error=True)
    recovered = _BrowseService(name="Recovered")
    service: Any = failing
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
    )
    scope = PromptBrowseScope(query="same")
    controller.request(scope, focus_identity="library-prompts-retry")
    await screen.pending.pop()
    failed_token = controller.result.request_token
    assert controller.result.status == "error"

    service = recovered
    controller.retry(focus_identity="library-prompts-retry")
    retry_token = controller.result.request_token
    await screen.pending.pop()

    assert retry_token > failed_token
    assert controller.scope == scope
    assert controller.result.status == "ready"
    assert controller.result.items[0]["name"] == "Recovered"


@pytest.mark.asyncio
async def test_controller_live_reads_replaced_service_seam() -> None:
    screen = _WorkerScreen()
    first = _BrowseService(name="Old")
    replacement = _BrowseService(name="Replacement")
    service: Any = first
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
    )
    controller.request(PromptBrowseScope(), focus_identity=None)
    work = screen.pending.pop()

    service = replacement
    await work

    assert first.calls == []
    assert len(replacement.calls) == 1
    assert controller.result.items[0]["name"] == "Replacement"
    assert screen.worker_kwargs == [
        {
            "exclusive": True,
            "group": "library-prompt-browse",
        }
    ]


@pytest.mark.asyncio
async def test_controller_live_reads_replaced_service_call_seam() -> None:
    screen = _WorkerScreen()
    service = _BrowseService()
    first_calls: list[str] = []
    replacement_calls: list[str] = []

    async def first_call(*_args: Any, **_kwargs: Any) -> Any:
        first_calls.append("called")
        raise AssertionError("early-bound service-call seam was used")

    async def replacement_call(
        callable_obj: Callable[..., Awaitable[Any]], **kwargs: Any
    ) -> Any:
        replacement_calls.append("called")
        return await _run_service_call(callable_obj, **kwargs)

    run_service_call: Callable[..., Awaitable[Any]] = first_call
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: lambda _result, _focus: None,
        run_service_call=lambda: run_service_call,
    )
    controller.request(PromptBrowseScope(), focus_identity=None)
    work = screen.pending.pop()

    run_service_call = replacement_call
    await work

    assert first_calls == []
    assert replacement_calls == ["called"]
    assert controller.result.status == "ready"


@pytest.mark.asyncio
async def test_controller_live_reads_replaced_sync_callback_on_settle() -> None:
    screen = _WorkerScreen()
    service = _BrowseService()
    first_sync: list[str] = []
    replacement_sync: list[str] = []

    def first_sync_callback(result: Any, _focus: str | None) -> None:
        first_sync.append(result.status)

    sync: Callable[..., None] = first_sync_callback
    controller = _controller(
        screen=screen,
        service=lambda: service,
        sync_view=lambda: sync,
    )
    controller.request(PromptBrowseScope(), focus_identity=None)
    work = screen.pending.pop()

    def replacement_sync_callback(result: Any, _focus: str | None) -> None:
        replacement_sync.append(result.status)

    sync = replacement_sync_callback
    await work

    assert first_sync == ["loading"]
    assert replacement_sync == ["ready"]
