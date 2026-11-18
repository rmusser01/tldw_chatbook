"""Actual Textual navigation waits for selected Chat's resume publication.

The publication/transport below is an orchestration fixture, not native
retirement evidence. ScreenResume, navigation and the strict getter are real.
"""

import asyncio
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.screen import Screen

from tldw_chatbook.app_navigation import NavigationMixin
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
from tldw_chatbook.UI.Console_Modules import execution_state
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


@pytest.mark.asyncio
@pytest.mark.parametrize("writer_pending", [False, True])
async def test_reused_chat_navigation_waits_before_original_return_callbacks(
    monkeypatch, writer_pending,
):
    entered, release, write_release = (asyncio.Event() for _ in range(3))
    acknowledgements = []

    class Chat(Screen):
        screen_name = "chat"

        @property
        def app_instance(self):
            return self.app

        def on_screen_resume(self):
            if self.app.console_execution_client is not None:
                return execution_state.resume_selected_view(self)

        def _start_console_attach_reconciliation(self):
            pass  # This test owns the resume publication, not widget rendering.

    class Harness(NavigationMixin, App):
        BINDINGS = [("ctrl+2", "return_chat", "Chat")]
        _LEGACY_ROUTE_LIBRARY_NAV_CONTEXT = {}

        def __init__(self):
            super().__init__()
            self.console_execution_client = None
            self.console_runtime = None
            self.chat = Chat()
            self.settings = Screen()
            self.current_tab = "settings"
            self.conversation_local_marks_service = object()
            self._console_manual_read_departure = self.conversation_local_marks_service

        async def on_mount(self):
            self.install_screen(self.chat, "chat")
            self.install_screen(self.settings, "settings")
            await self.push_screen(self.chat)

        def _current_runtime_identity(self):
            return "test"

        def _ensure_screen_owned_css(self, _route):
            pass

        def _reusable_navigation_screen(self, *_args):
            return self.chat

        async def _dismiss_navigation_overlays(self, _route):
            return True

        def _clear_focus_if_leaving_console(self, _route):
            pass

        def _resync_navigation_bar_active(self, _screen):
            pass

        def action_return_chat(self):
            self.navigation = self.run_worker(self._complete_screen_navigation(
                message=NavigateToScreen("chat"), requested_screen="chat",
                screen_name="chat", current_tab_value="chat", screen_class=Chat,
                current_screen=self.screen,
            ))

    app = Harness()
    monkeypatch.setattr(execution_state, "ChatScreen", Chat)
    monkeypatch.setattr(execution_state, "ConsoleExecutionClient", SimpleNamespace)
    monkeypatch.setattr(execution_state, "_require_sources", lambda: None)
    monkeypatch.setattr(execution_state, "_require_methods", lambda *_: None)
    from tldw_chatbook.Chat import console_execution_readers

    monkeypatch.setattr(console_execution_readers, "require_reader_sources", lambda: None)
    for name in ("ConsoleProjectionView", "ConsoleProjection", "ConsoleSessionComposition"):
        monkeypatch.setattr(execution_state, name, SimpleNamespace)

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.switch_screen(app.settings)
        await pilot.pause()
        screen = app.chat
        row = SimpleNamespace(id="chat", incarnation_id="incarnation", settings_revision=0)
        frame = SimpleNamespace(source_revision=1, session_id=row.id, incarnation_id=row.incarnation_id)
        view = SimpleNamespace(source_revision=1, active_session_id=row.id,
                               projections=(frame,), session=lambda _: row)
        client = SimpleNamespace(
            _app=app, _view=screen, _visit=1, _source_revision=1 if writer_pending else 2,
            _attachment_id="attachment", _intake_open=not writer_pending,
            _channel_error=None, _disposed=False, _shutdown_task=None,
            projection_view=view, _projection=frame,
            session_composition=SimpleNamespace(session_id=row.id,
                incarnation_id=row.incarnation_id, settings_revision=0),
            _workspace_binding_task=None, _workspace_operations=set(),
        )
        app.console_execution_client = client
        screen._console_process_visit = screen._console_process_mounted_visit = 1
        screen._console_process_surface_reconciled = True

        def require_view(actual, visit):
            if actual is not screen or app.screen is not screen or visit != client._visit:
                raise ConsoleChannelLost("original visible visit refused")

        def attach(actual):
            assert actual is screen
            client._visit += 1
            return client._visit

        async def read(actual, visit, session_id, incarnation, *, checkpoint):
            require_view(actual, visit)
            assert (session_id, incarnation, checkpoint) == (row.id, row.incarnation_id, "mount")
            assert client._intake_open
            entered.set()
            await release.wait()
            require_view(actual, visit)
            view.source_revision = frame.source_revision = client._source_revision
            return frame

        async def acknowledge():
            assert execution_state.selected_client(screen) is client
            acknowledgements.append(client._source_revision)

        async def write():
            await write_release.wait()
            client._source_revision = 2
            client._intake_open = True

        client.require_view, client.attach_view, client.read_view_display = require_view, attach, read
        screen._session = SimpleNamespace(acknowledge_explicit_console_return=acknowledge)
        if writer_pending:
            operation = client._workspace_binding_task = asyncio.create_task(write())
            client._workspace_operations.add(operation)
        press = asyncio.create_task(pilot.press("ctrl+2"))
        try:
            if writer_pending:
                # The registry may already show rw while Apply is still owned.
                await asyncio.sleep(0.05)
                assert not entered.is_set() and not acknowledgements
                write_release.set()
            await asyncio.wait_for(entered.wait(), 2)
            assert not press.done() and not app.navigation.is_finished
            assert not acknowledgements
            with pytest.raises(RuntimeError, match="selected projection"):
                execution_state.selected_client(screen)
            release.set()
            await asyncio.wait_for(press, 3)
            await app.navigation.wait()
            await pilot.pause()
            assert acknowledgements == [2]
            assert execution_state.selected_client(screen) is client
        finally:
            write_release.set()
            release.set()
            await asyncio.gather(press, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [
    "hidden", "changed_source", "changed_after_callback", "pump_ended", "cancelled",
])
async def test_navigation_resume_barrier_keeps_currentness_and_pump_ownership(monkeypatch, outcome):
    from Tests.UI.test_console_selected_refresh_coverage import _resumed_publication

    screen, client, app = _resumed_publication(monkeypatch)
    client.projection_view.source_revision = client._projection.source_revision = client._source_revision
    pump_release = asyncio.Event()
    pump = screen._task = asyncio.create_task(pump_release.wait())
    callbacks = []
    screen.call_later = lambda callback: callbacks.append(callback) or True
    waiting = asyncio.create_task(execution_state.wait_selected_navigation_resume(screen))
    await asyncio.sleep(0)
    assert callbacks and not waiting.done()
    try:
        if outcome == "pump_ended":
            pump_release.set()
            with pytest.raises(RuntimeError, match="pump ended"):
                await waiting
        elif outcome == "cancelled":
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert not pump.done()
        else:
            if outcome == "changed_after_callback":
                callbacks[0]()
            if outcome == "hidden":
                app.screen = Screen()
            else:
                client._source_revision += 1
            callbacks[0]()
            with pytest.raises((RuntimeError, ConsoleChannelLost)):
                await waiting
            assert not pump.done()
        callbacks[0]()  # A late callback cannot publish after observer settlement.
    finally:
        pump_release.set()
        await pump
