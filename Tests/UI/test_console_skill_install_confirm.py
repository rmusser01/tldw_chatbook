from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState


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


def test_task_resume_state_pending_skill_install_roundtrip():
    s = TaskResumeState()
    assert s.has_pending_skill_install() is False
    s2 = replace(s, pending_skill_install={"url": "https://x/y", "timeout_seconds": 120.0})
    assert s2.has_pending_skill_install() is True
    assert TaskResumeState.from_dict(s2.to_dict()).pending_skill_install == {
        "url": "https://x/y", "timeout_seconds": 120.0,
    }


@pytest.mark.asyncio
async def test_skill_install_card_allow_and_deny():
    from textual import on
    from textual.app import App, ComposeResult
    from textual.widgets import Button
    from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
        SkillInstallConfirmCard,
    )

    class _Host(App[None]):
        def __init__(self):
            super().__init__()
            self.decided = []

        def compose(self) -> ComposeResult:
            yield SkillInstallConfirmCard()

        @on(SkillInstallConfirmCard.InstallDecided)
        def _cap(self, event: SkillInstallConfirmCard.InstallDecided) -> None:
            self.decided.append(event.allow)

    app = _Host()
    async with app.run_test() as pilot:
        card = app.query_one(SkillInstallConfirmCard)
        # A URL containing Rich-markup-like text must render literally.
        card.set_install(
            {"url": "https://github.com/o/[bold]r[/]", "timeout_seconds": 120.0}
        )
        await pilot.pause()
        assert card.display is True
        from textual.widgets import Static
        url_text = str(app.query_one("#skill-install-url", Static).render())
        assert "[bold]" in url_text  # not interpreted as markup
        app.query_one("#skill-install-allow", Button).press()
        await pilot.pause()
        assert app.decided == [True]
        card.set_install({"url": "https://github.com/o/r", "timeout_seconds": 120.0})
        await pilot.pause()
        app.query_one("#skill-install-deny", Button).press()
        await pilot.pause()
        assert app.decided == [True, False]
