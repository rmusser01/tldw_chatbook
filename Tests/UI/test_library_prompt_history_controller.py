from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from tldw_chatbook.UI.Library_Modules import LibraryPromptHistoryController


class _ScreenStub:
    def __init__(self) -> None:
        self.started: list[Awaitable[Any]] = []

    def run_worker(self, work: Awaitable[Any], **_kwargs: Any) -> None:
        self.started.append(work)


class _HistoryServiceStub:
    def __init__(self) -> None:
        self.count_calls: list[str] = []
        self.page_calls: list[int | None] = []

    async def count_prompt_versions(self, **kwargs: Any) -> int:
        self.count_calls.append(kwargs["prompt_identifier"])
        return 7 if kwargs["prompt_identifier"] == "prompt-b" else 3

    async def list_prompt_versions(self, **kwargs: Any) -> dict[str, Any]:
        self.page_calls.append(kwargs["before_change_id"])
        return {
            "items": [
                {
                    "prompt_uuid": kwargs["prompt_identifier"],
                    "change_id": 1,
                    "version": 1,
                    "timestamp": "2026-08-08T12:00:00+00:00",
                    "artifact_type": "prompt",
                    "artifact_type_raw": "",
                    "name": "Prompt",
                    "author": "",
                    "details": "",
                    "system_prompt": "",
                    "user_prompt": "User",
                    "keywords": [],
                    "keywords_captured": True,
                    "compatibility_state": "compatible",
                    "compatibility_reason": "",
                    "restore_eligible": True,
                    "changed_fields": [],
                    "change_summary": "Created",
                }
            ],
            "total_count": 1,
            "has_more": False,
            "next_before_change_id": None,
        }


async def _call_service(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    kwargs.pop("isolate_in_worker")
    result = function(*args, **kwargs)
    if isinstance(result, Awaitable):
        return await result
    return result


@pytest.mark.asyncio
async def test_prompt_history_controller_live_reads_replaced_screen_seams() -> None:
    screen = _ScreenStub()
    service = _HistoryServiceStub()
    started_after_replacement: list[Awaitable[Any]] = []
    synced_after_replacement: list[Any] = []
    service_calls_after_replacement: list[str] = []
    service_call: dict[str, Callable[..., Awaitable[Any]]] = {"current": _call_service}
    sync_view: dict[str, Callable[[Any], None]] = {"current": lambda _state: None}
    controller = LibraryPromptHistoryController(
        screen=screen,
        run_service_call=lambda *args, **kwargs: service_call["current"](
            *args, **kwargs
        ),
        prompt_service=lambda: service,
        sync_view=lambda state: sync_view["current"](state),
    )

    def replacement_worker(work: Awaitable[Any], **_kwargs: Any) -> None:
        started_after_replacement.append(work)

    async def replacement_service_call(
        function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        service_calls_after_replacement.append(function.__name__)
        return await _call_service(function, *args, **kwargs)

    screen.run_worker = replacement_worker  # type: ignore[method-assign]
    service_call["current"] = replacement_service_call
    sync_view["current"] = synced_after_replacement.append

    controller.initialize({"uuid": "prompt-a", "version": 4})
    assert not screen.started
    assert len(started_after_replacement) == 1
    await started_after_replacement.pop()

    assert service_calls_after_replacement == ["count_prompt_versions"]
    assert controller.state is not None
    assert controller.state.retained_count == 3
    assert synced_after_replacement[-1] is controller.state


@pytest.mark.asyncio
async def test_prompt_history_controller_ignores_late_count_after_prompt_switch() -> (
    None
):
    screen = _ScreenStub()
    service = _HistoryServiceStub()
    controller = LibraryPromptHistoryController(
        screen=screen,
        run_service_call=_call_service,
        prompt_service=lambda: service,
        sync_view=lambda _state: None,
    )

    controller.initialize({"uuid": "prompt-a", "version": 4})
    prompt_a_count = screen.started.pop()
    controller.initialize({"uuid": "prompt-b", "version": 9})
    prompt_b_count = screen.started.pop()

    await prompt_a_count
    assert controller.state is not None
    assert controller.state.prompt_uuid == "prompt-b"
    assert controller.state.retained_count is None

    await prompt_b_count
    assert controller.state.prompt_uuid == "prompt-b"
    assert controller.state.retained_count == 7


@pytest.mark.asyncio
async def test_prompt_history_controller_reload_resets_page_without_refetching_count():
    screen = _ScreenStub()
    service = _HistoryServiceStub()
    controller = LibraryPromptHistoryController(
        screen=screen,
        run_service_call=_call_service,
        prompt_service=lambda: service,
        sync_view=lambda _state: None,
    )

    controller.initialize({"uuid": "prompt-a", "version": 1})
    await screen.started.pop()
    controller.request_page()
    await screen.started.pop()

    controller.reload_page()

    assert service.count_calls == ["prompt-a"]
    assert service.page_calls == [None]
    assert len(screen.started) == 1
    assert controller.state is not None
    assert controller.state.rows == ()
    assert controller.state.selected is None
    await screen.started.pop()
    assert service.count_calls == ["prompt-a"]
    assert service.page_calls == [None, None]
    assert controller.state.page_status == "loaded"


@pytest.mark.asyncio
async def test_prompt_history_controller_ignores_absent_row_selection() -> None:
    """A stale row identity within the current scope is a reducer no-op."""
    screen = _ScreenStub()
    service = _HistoryServiceStub()
    controller = LibraryPromptHistoryController(
        screen=screen,
        run_service_call=_call_service,
        prompt_service=lambda: service,
        sync_view=lambda _state: None,
    )
    controller.initialize({"uuid": "prompt-a", "version": 1})
    await screen.started.pop()
    controller.request_page()
    await screen.started.pop()
    before = controller.state

    controller.select(change_id=999, source_version=999)

    assert controller.state is before
