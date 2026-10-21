"""Regression coverage for bounded Console Library startup (TASK-35992)."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_automatic_library_preparation import (
    _context,
    _controller_for_preparation,
    _preparation,
)
from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_controller import _DispatchRecoveryRefusal

pytestmark = pytest.mark.bootstrap_profile


class ColdLibrary:
    def __init__(self, *, stalled=False):
        self.ready = False
        self.stalled = stalled
        self.started = asyncio.Event()
        self.never = asyncio.Event()

    async def prewarm(self):
        self.started.set()
        if self.stalled:
            await self.never.wait()
        else:
            await asyncio.sleep(0.03)
            self.ready = True

    async def search(self, *_args, **_kwargs):
        if not self.ready:
            await self.prewarm()
        return {"runtime_backend": "local", "results": []}


@pytest.mark.asyncio
async def test_cold_initialization_has_a_separate_budget_from_search():
    service = ColdLibrary()
    controller, _ = _controller_for_preparation(_preparation(), service, timeout=0.001)
    controller._library_initialization_timeout = 1.0

    outcome = await controller.prepare_library_for_turn("preparation-1")

    assert outcome.state is ConsoleTurnPreparationState.READY
    assert outcome.error_code is None


@pytest.mark.asyncio
async def test_stalled_initialization_pauses_with_an_actionable_timeout():
    service = ColdLibrary(stalled=True)
    controller, store = _controller_for_preparation(
        _preparation(), service, timeout=0.001
    )
    controller._library_initialization_timeout = 0.01

    outcome = await asyncio.wait_for(
        controller.prepare_library_for_turn("preparation-1"), 1
    )

    assert outcome.state is ConsoleTurnPreparationState.PAUSED
    assert outcome.error_code == "library_retrieval_timeout"
    assert (
        store.preparation_for_session("session-1").state
        is ConsoleTurnPreparationState.PAUSED
    )


@pytest.mark.asyncio
async def test_cancelling_cold_initialization_preserves_task_cancellation():
    service = ColdLibrary(stalled=True)
    controller, _ = _controller_for_preparation(_preparation(), service)
    task = asyncio.create_task(controller.prepare_library_for_turn("preparation-1"))
    await service.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled", [False, True])
async def test_dispatch_retry_uses_the_same_bounded_initialization(
    monkeypatch, stalled
):
    service = ColdLibrary(stalled=stalled)
    preparation = _preparation()
    controller, store = _controller_for_preparation(preparation, service, timeout=0.001)
    controller._library_initialization_timeout = 0.01 if stalled else 1.0
    context = _context()
    user = store.append_message(
        "session-1",
        role=ConsoleMessageRole.USER,
        content="retry question",
        persist=False,
    )
    resolution = await controller.provider_gateway.resolve_for_send(None)

    async def authority(*_args):
        return context.library_authority

    async def destination(*_args):
        return resolution

    monkeypatch.setattr(
        controller,
        "resolve_turn_configuration_snapshot",
        lambda _: context.configuration,
    )
    monkeypatch.setattr(controller, "_capture_turn_library_authority", authority)
    monkeypatch.setattr(controller, "_resolve_for_send_bounded", destination)
    monkeypatch.setattr(
        controller, "_finalize_turn_execution_context", lambda *_: context
    )
    monkeypatch.setattr(
        controller, "_provider_messages_for_session", lambda *_args, **_kwargs: []
    )
    recovery = SimpleNamespace(
        assistant_message_id="assistant-retry",
        checkpoint=SimpleNamespace(
            user_message_id=user.id,
            frozen_authority=context.library_authority,
            resolved_destination=context.resolved_destination,
        ),
    )

    pending = controller._resolve_dispatch_retry_context("session-1", recovery)
    if stalled:
        with pytest.raises(_DispatchRecoveryRefusal, match="timed out"):
            await asyncio.wait_for(pending, 1)
    else:
        result = await asyncio.wait_for(pending, 1)
        assert result.turn_context is context
        assert service.ready
