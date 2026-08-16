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
