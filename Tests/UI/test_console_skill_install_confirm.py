from __future__ import annotations

import asyncio
import time

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class _FakeApp:
    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _controller():
    store = ConsoleChatStore()
    return ConsoleChatController(store=store, provider_gateway=object()), store


@pytest.mark.asyncio
async def test_confirm_round_trip_allow():
    controller, _ = _controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_skill_install = received.append

    async def resolve_soon():
        await asyncio.sleep(0.05)
        assert received and received[0] is not None
        assert received[0]["url"] == "https://github.com/o/r"
        controller.resolve_pending_skill_install(True)

    task = asyncio.create_task(
        asyncio.to_thread(controller.request_skill_install_confirm, "https://github.com/o/r")
    )
    await resolve_soon()
    allowed = await task
    assert allowed is True
    assert received[-1] is None  # card cleared afterwards


@pytest.mark.asyncio
async def test_confirm_round_trip_deny():
    controller, _ = _controller()
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None

    async def resolve_soon():
        await asyncio.sleep(0.05)
        controller.resolve_pending_skill_install(False)

    task = asyncio.create_task(
        asyncio.to_thread(controller.request_skill_install_confirm, "https://x/y")
    )
    await resolve_soon()
    assert (await task) is False


def test_confirm_timeout_denies():
    controller, _ = _controller()
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None
    controller.skill_install_confirm_timeout_seconds = lambda: 0.05
    started = time.monotonic()
    allowed = controller.request_skill_install_confirm("https://x/y")
    assert allowed is False
    assert time.monotonic() - started < 2.5


def test_context_change_denies_pending_confirm():
    controller, _ = _controller()
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None
    import threading

    result = {}

    def worker():
        result["allowed"] = controller.request_skill_install_confirm("https://x/y")

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.1)
    controller._deny_pending_skill_install_on_context_change()
    t.join(timeout=3.0)
    assert result["allowed"] is False
