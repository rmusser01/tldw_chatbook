"""Decision-worker scheduling controls; native read retirement is not simulated."""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
from tldw_chatbook.Constants import TAB_CHAT
from tldw_chatbook.UI.Console_Modules import execution_decisions, execution_state
from tldw_chatbook.UI.Navigation import base_app_screen


class _Chat(SimpleNamespace):
    pass


class _Destination(SimpleNamespace):
    pass


class _Client(SimpleNamespace):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        None,
        "visible",
        "source",
        "attachment",
        "session",
        "incarnation",
        "client",
        "cache",
        "stack",
        "destination",
        "channel",
        "read_failure",
        "retirement_failure",
        "definition",
        "source_definition",
        "publication",
        "before_read",
    ],
)
async def test_only_a_successful_same_scope_passive_read_defers_navigation(
    monkeypatch, changed
):
    app = SimpleNamespace(console_runtime=None)
    screen = _Chat(
        app=app,
        app_instance=app,
        is_attached=True,
        is_mounted=True,
        is_current=True,
        _console_process_surface_reconciled=True,
    )
    destination = _Destination(app=app, is_attached=True, is_mounted=True)
    row = SimpleNamespace(id="session", incarnation_id="incarnation")
    published = SimpleNamespace(active_session_id=row.id, session=lambda _: row)
    client = _Client(
        _app=app,
        _view=screen,
        _visit=1,
        _source_revision=3,
        _attachment_id="attachment",
        _view_preparation=None,
        _project_modal=None,
        _disposed=False,
        _shutdown_task=None,
        _channel_error=None,
        projection_view=published,
    )
    app.console_execution_client = client
    app.screen, app.screen_stack = screen, [screen]
    app._reusable_screen_instances = {TAB_CHAT: (object(), screen)}
    actions = []
    source_checks = []
    completed = False

    def require_view(view, visit):
        if (
            view is not screen
            or visit != 1
            or app.screen is not screen
            or client._channel_error is not None
        ):
            raise ConsoleChannelLost("original view refused")

    def selected(_screen):
        require_view(_screen, client._visit)
        return client

    def require_methods(*_):
        if completed and changed == "definition":
            raise RuntimeError("original decision reader changed")

    def require_sources():
        source_checks.append(True)
        if changed == "source_definition":
            raise RuntimeError("original source changed")

    def require_publication(*_):
        if changed == "publication":
            raise RuntimeError("original publication refused")

    def switch():
        screen.is_current = False
        app.screen, app.screen_stack = destination, [destination]

    async def read(*_):
        nonlocal completed
        actions.append("read")
        await asyncio.sleep(0)
        if changed != "visible":
            switch()
        if changed == "read_failure":
            raise RuntimeError("original read failed")
        if changed == "retirement_failure":
            client._channel_error = ConsoleChannelLost("original retirement refused")
            raise client._channel_error
        if changed == "source":
            client._source_revision += 1
        elif changed == "attachment":
            client._attachment_id = "other"
        elif changed == "session":
            published.active_session_id = "other"
        elif changed == "incarnation":
            row.incarnation_id = "other"
        elif changed == "client":
            app.console_execution_client = object()
        elif changed == "cache":
            app._reusable_screen_instances[TAB_CHAT] = (object(), screen)
        elif changed == "stack":
            app.screen_stack = [screen, destination]
        elif changed == "destination":
            destination.is_mounted = False
        elif changed == "channel":
            client._channel_error = ConsoleChannelLost("channel failed")
        completed = True  # Only this successful return permits the new branch.

    async def present(*_):
        actions.append("present")
        return True

    client.require_view, client.read_decision_head = require_view, read
    monkeypatch.setattr(execution_state, "ChatScreen", _Chat)
    monkeypatch.setattr(execution_state, "ConsoleExecutionClient", _Client)
    monkeypatch.setattr(base_app_screen, "BaseAppScreen", _Destination)
    monkeypatch.setattr(execution_state, "_require_sources", require_sources)
    monkeypatch.setattr(execution_state, "_require_methods", require_methods)
    monkeypatch.setattr(
        execution_state, "_require_selected_publication", require_publication
    )
    monkeypatch.setattr(execution_state, "selected_client", selected)
    monkeypatch.setattr(execution_state, "_defer_selected_refresh", lambda *_: False)
    monkeypatch.setattr(
        execution_state,
        "_decision_refresh_rendered",
        lambda *_: actions.append("rendered"),
    )
    monkeypatch.setattr(execution_state, "_decision_refresh_due", lambda *_: False)
    monkeypatch.setattr(
        execution_decisions, "pending_project_visibility", lambda *_: None
    )
    monkeypatch.setattr(execution_decisions, "present_selected_decision", present)
    if changed == "before_read":
        switch()
    if changed in (None, "visible"):
        await execution_state._refresh_selected_decisions(screen, client, 1)
        assert actions == (
            ["read"] if changed is None else ["read", "present", "rendered"]
        )
        assert screen._console_process_surface_reconciled is (changed == "visible")
        assert len(source_checks) == (0 if changed == "visible" else 1)
    else:
        with pytest.raises((ConsoleChannelLost, RuntimeError)):
            await execution_state._refresh_selected_decisions(screen, client, 1)
        assert actions == ([] if changed == "before_read" else ["read"])
        assert screen._console_process_surface_reconciled
