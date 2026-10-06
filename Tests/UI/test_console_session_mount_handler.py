"""Custom strip mount handlers keep their original session publication route."""

import asyncio

import pytest
from textual._context import active_message_pump

from Tests.UI.test_console_session_initial_compose import InitialComposeHost
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Widgets.Console import console_session_surface as surfaces

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("task_factory", ("normal", "eager"))
async def test_custom_strip_mount_can_publish_its_own_session_tabs(
    monkeypatch, task_factory
):
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    sessions = [ConsoleChatSession(title="Mount handler", id="mount-handler")]
    called = []

    class MountPublishingStrip(surfaces.ConsoleSessionTabStrip):
        async def on_mount(self):
            assert active_message_pump.get(None) is self
            assert not self._mounted_event.is_set()
            called.append(self)
            surface = self.app.surface
            await surface.sync_sessions(
                sessions=sessions, active_session_id="mount-handler"
            )

    monkeypatch.setattr(surfaces, "ConsoleSessionTabStrip", MountPublishingStrip)
    if task_factory == "eager":
        loop.set_task_factory(asyncio.eager_task_factory)
    app = InitialComposeHost()
    try:
        async with asyncio.timeout(5):
            async with app.run_test(size=(80, 24)):
                strip = app.surface.query_one("#console-native-tab-strip")
                assert called == [strip]
                assert strip._mounted_event.is_set()
                assert [child.id for child in strip.children] == (
                    app.surface._desired_tab_child_ids(
                        sessions=sessions, active_session_id="mount-handler"
                    )
                )
    finally:
        loop.set_task_factory(previous_factory)
