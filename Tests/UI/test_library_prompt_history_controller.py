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
    async def count_prompt_versions(self, **kwargs: Any) -> int:
        return 7 if kwargs["prompt_identifier"] == "prompt-b" else 3


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
