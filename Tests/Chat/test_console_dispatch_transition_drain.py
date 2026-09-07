"""Cancellation and failure ownership at the dispatch handoff, without a DB."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


@pytest.mark.parametrize("outcome", ["complete", "failed", "cancelled"])
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_dispatch_drain_preserves_work_and_cleans_terminal_owner(
    outcome: str, cancel_waiter: bool
) -> None:
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=SimpleNamespace()
    )
    release = asyncio.Event()
    other_release = asyncio.Event()

    async def handoff() -> None:
        await release.wait()
        if outcome == "failed":
            raise RuntimeError("handoff failed")
        if outcome == "cancelled":
            raise asyncio.CancelledError

    transition = asyncio.create_task(handoff())
    other = asyncio.create_task(other_release.wait())
    controller._pending_dispatch_transitions.update(owner=transition, other=other)
    waiter = asyncio.create_task(controller._drain_dispatch_transition("owner"))
    try:
        await asyncio.sleep(0)
        if cancel_waiter:
            for _ in range(2):
                waiter.cancel()
                await asyncio.sleep(0)
        assert not waiter.done()
        assert not transition.done()
        assert not other.done()
        release.set()
        if outcome == "failed":
            with pytest.raises(RuntimeError, match="handoff failed"):
                await waiter
        elif outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await waiter
        else:
            assert await waiter is cancel_waiter
        assert controller._pending_dispatch_transitions == {"other": other}
        assert not other.done()
        assert await controller._drain_dispatch_transition("missing") is False
    finally:
        release.set()
        other_release.set()
        await asyncio.gather(waiter, transition, other, return_exceptions=True)


async def test_dispatch_drain_does_not_remove_a_replacement_owner() -> None:
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=SimpleNamespace()
    )
    release = asyncio.Event()
    newer_release = asyncio.Event()
    original = asyncio.create_task(release.wait())
    newer = asyncio.create_task(newer_release.wait())
    controller._pending_dispatch_transitions["owner"] = original
    waiter = asyncio.create_task(controller._drain_dispatch_transition("owner"))
    try:
        await asyncio.sleep(0)
        controller._pending_dispatch_transitions["owner"] = newer
        release.set()
        assert await waiter is False
        assert controller._pending_dispatch_transitions["owner"] is newer
        assert not newer.done()
    finally:
        release.set()
        newer_release.set()
        await asyncio.gather(waiter, original, newer, return_exceptions=True)
