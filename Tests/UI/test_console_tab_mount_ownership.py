"""Original tab publication waits for its real strip composition owner."""

import asyncio
from types import SimpleNamespace

import pytest
from textual._context import active_message_pump
from textual.app import App

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Widgets.Console import console_session_surface as source

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("eager", [False, True], ids=["normal", "eager"])
async def test_original_tab_sync_waits_for_actual_pending_strip_mount(monkeypatch, eager):
    entered, release = asyncio.Event(), asyncio.Event()
    seen = {}
    original_strip = source.ConsoleSessionTabStrip

    class HeldStrip(original_strip):
        async def mount_composed_widgets(self, widgets):
            # Textual's original _compose has gathered its actual children and
            # awaits this documented mount hook before setting _mounted_event.
            seen["strip"] = self
            seen["mount_task"] = asyncio.current_task()
            entered.set()
            await release.wait()
            await super().mount_composed_widgets(widgets)

    class Host(App):
        def compose(self):
            yield source.ConsoleSessionSurface(SimpleNamespace(notify=lambda *_: None))

    host = Host()
    loop = asyncio.get_running_loop()
    old_factory = loop.get_task_factory()
    async with host.run_test(size=(100, 30)):
        surface = host.query_one(source.ConsoleSessionSurface)
        monkeypatch.setattr(source, "ConsoleSessionTabStrip", HeldStrip)
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        publication = None
        recomposing = asyncio.create_task(surface.recompose())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            strip = seen["strip"]
            assert strip.is_attached and not strip._mounted_event.is_set()
            assert not strip.children
            token = active_message_pump.set(strip)
            try:
                publication = asyncio.create_task(surface.sync_sessions(
                    sessions=[ConsoleChatSession(title="Current", id="current")],
                    active_session_id="current",
                ))
            finally:
                active_message_pump.reset(token)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            seen["waited"] = not publication.done()
            seen["untouched"] = not strip.children
        finally:
            release.set()
            outcomes = await asyncio.wait_for(asyncio.gather(
                recomposing, *(() if publication is None else (publication,)),
                return_exceptions=True,
            ), 10)
            loop.set_task_factory(old_factory)
        assert seen["waited"] and seen["untouched"], {
            "waited": seen["waited"], "untouched": seen["untouched"],
            "outcomes": [type(item).__name__ for item in outcomes],
        }
        assert all(item is None for item in outcomes), outcomes
        assert strip._mounted_event.is_set()
        assert [child.id for child in strip.children] == [
            "console-session-tab-current", "console-close-session-tab-current",
            "console-new-chat-tab", "console-new-temporary-tab",
        ]
        assert publication is not seen["mount_task"]


@pytest.mark.parametrize("eager", [False, True], ids=["normal", "eager"])
async def test_actual_strip_mount_owner_can_publish_before_mounted_event(monkeypatch, eager):
    completed = asyncio.Event()
    observed = {}
    original_strip = source.ConsoleSessionTabStrip

    class PublishingStrip(original_strip):
        async def on_mount(self):
            observed["event_before"] = self._mounted_event.is_set()
            observed["owner"] = asyncio.current_task()
            surface = self.parent.parent
            assert isinstance(surface, source.ConsoleSessionSurface)
            await surface.sync_sessions(
                sessions=[ConsoleChatSession(title="Mounted", id="mounted")],
                active_session_id="mounted",
            )
            observed["strip"] = self
            completed.set()

    class Host(App):
        def compose(self):
            yield source.ConsoleSessionSurface(SimpleNamespace(notify=lambda *_: None))

    host = Host()
    loop = asyncio.get_running_loop()
    old_factory = loop.get_task_factory()
    async with host.run_test(size=(100, 30)):
        surface = host.query_one(source.ConsoleSessionSurface)
        monkeypatch.setattr(source, "ConsoleSessionTabStrip", PublishingStrip)
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        try:
            await asyncio.wait_for(surface.recompose(), 10)
            await asyncio.wait_for(completed.wait(), 5)
        finally:
            loop.set_task_factory(old_factory)
        strip = observed["strip"]
        assert observed["event_before"] is False
        assert observed["owner"] is strip._task
        assert strip._mounted_event.is_set()
        assert strip.query_one("#console-session-tab-mounted")


@pytest.mark.parametrize("eager", [False, True], ids=["normal", "eager"])
async def test_pending_tab_publication_retires_when_actual_parent_is_removed(monkeypatch, eager):
    entered, release = asyncio.Event(), asyncio.Event()
    seen = {}
    original_strip = source.ConsoleSessionTabStrip

    class HeldStrip(original_strip):
        async def mount_composed_widgets(self, widgets):
            seen["strip"] = self
            entered.set()
            await release.wait()
            await super().mount_composed_widgets(widgets)

    class Host(App):
        def compose(self):
            yield source.ConsoleSessionSurface(SimpleNamespace(notify=lambda *_: None))

    host = Host()
    loop = asyncio.get_running_loop()
    old_factory = loop.get_task_factory()
    async with host.run_test(size=(100, 30)):
        surface = host.query_one(source.ConsoleSessionSurface)
        monkeypatch.setattr(source, "ConsoleSessionTabStrip", HeldStrip)
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        publication = removing = None
        recomposing = asyncio.create_task(surface.recompose())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            strip = seen["strip"]
            publication = asyncio.create_task(surface.sync_sessions(
                sessions=[ConsoleChatSession(title="Retired", id="retired")],
                active_session_id="retired",
            ))
            await asyncio.sleep(0)
            assert not publication.done()
            async def remove_parent():
                await surface.remove()
            removing = asyncio.create_task(remove_parent())
            for _ in range(10):
                if surface._pruning or surface._closing or surface._closed:
                    break
                await asyncio.sleep(0)
            assert surface._pruning or surface._closing or surface._closed
        finally:
            release.set()
            results = await asyncio.wait_for(asyncio.gather(
                recomposing, *(task for task in (publication, removing) if task is not None),
                return_exceptions=True,
            ), 10)
            loop.set_task_factory(old_factory)
        assert all(result is None for result in results), results
        assert not surface.is_attached and not strip.is_attached
        assert not strip.query("#console-session-tab-retired")
