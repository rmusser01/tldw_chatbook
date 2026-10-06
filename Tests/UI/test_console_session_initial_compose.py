"""A real Textual strip completes its initial composition before tab updates."""

import asyncio
from types import SimpleNamespace

import pytest
from textual.app import App

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Widgets.Console.console_session_surface import (
    ConsoleSessionSurface,
    ConsoleSessionTabStrip,
)

pytestmark = pytest.mark.bootstrap_profile


class InitialComposeHost(App[None]):
    def __init__(self):
        super().__init__()
        self.surface = ConsoleSessionSurface(SimpleNamespace())

    def compose(self):
        yield self.surface


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_context", ("app", "strip-inherited"))
@pytest.mark.parametrize("task_factory", ("normal", "eager"))
async def test_session_update_waits_for_actual_initial_strip_composition(
    monkeypatch, task_factory, caller_context
):
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    if task_factory == "eager":
        loop.set_task_factory(asyncio.eager_task_factory)
    app = InitialComposeHost()
    entered = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    observation = {}
    original_compose = ConsoleSessionTabStrip._compose
    original_sync = ConsoleSessionSurface.sync_sessions
    sessions = [ConsoleChatSession(title="Initial", id="initial")]

    async def hold_original_compose(strip):
        assert type(strip) is ConsoleSessionTabStrip
        assert strip.is_attached and not strip._mounted_event.is_set()
        observation["strip"] = strip
        entered.set()
        await asyncio.wait_for(release.wait(), 5)
        await original_compose(strip)

    monkeypatch.setattr(ConsoleSessionTabStrip, "_compose", hold_original_compose)

    async def drive_original_update():
        pending = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            strip = observation["strip"]
            assert app.surface.is_attached and not strip._mounted_event.is_set()
            if caller_context == "strip-inherited":
                with strip._context():
                    pending = asyncio.create_task(
                        original_sync(
                            app.surface, sessions=sessions, active_session_id="initial"
                        )
                    )
            else:
                pending = asyncio.create_task(
                    original_sync(
                        app.surface, sessions=sessions, active_session_id="initial"
                    )
                )
            await asyncio.sleep(0.05)
            observation["pending_before_release"] = not pending.done()
            observation["ids_before_release"] = [child.id for child in strip.children]
        finally:
            release.set()
            if pending is not None:
                await asyncio.wait_for(pending, 5)
            completed.set()

    with app._context():
        controller = asyncio.create_task(drive_original_update())
    try:
        async with app.run_test(size=(80, 24)):
            await asyncio.wait_for(completed.wait(), 5)
            await controller
            strip = observation["strip"]
            assert strip._mounted_event.is_set()
            assert observation["pending_before_release"]
            assert (
                observation["ids_before_release"] == []
            ), "Session publication raced the original pending composed New tab control"
            assert [
                child.id for child in strip.children
            ] == app.surface._desired_tab_child_ids(
                sessions=sessions, active_session_id="initial"
            )
    finally:
        release.set()
        await asyncio.gather(controller, return_exceptions=True)
        loop.set_task_factory(previous_factory)
