"""Passive refresh suspension is distinct from active view/channel authority."""

import asyncio
from types import SimpleNamespace

import pytest
from textual.screen import ModalScreen, Screen

from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
from tldw_chatbook.UI.Console_Modules import execution_decisions, execution_state


def _visit(monkeypatch):
    """Orchestration values only; no process or native receipt is simulated."""
    app = SimpleNamespace(console_runtime=None)
    screen = SimpleNamespace(
        app=app,
        app_instance=app,
        is_attached=True,
        is_mounted=True,
        is_current=True,
        _console_process_visit=1,
        _console_process_mounted_visit=1,
        _console_process_surface_reconciled=True,
        _console_attach_reconcile_running=False,
        _console_credential_poll_timer=None,
    )
    row = SimpleNamespace(id="chat", incarnation_id="incarnation")
    publication = SimpleNamespace(active_session_id=row.id, session=lambda _: row)
    client = SimpleNamespace(
        _app=app,
        _view=screen,
        _visit=1,
        _source_revision=4,
        _disposed=False,
        _shutdown_task=None,
        _channel_error=None,
        _project_modal=None,
        _readiness_deadline=float("inf"),
        projection_view=publication,
    )
    app.console_execution_client = client
    app.screen, app.screen_stack = screen, [screen]
    monkeypatch.setattr(execution_state, "ChatScreen", SimpleNamespace)
    monkeypatch.setattr(execution_state, "ConsoleExecutionClient", SimpleNamespace)
    monkeypatch.setattr(execution_state, "_require_sources", lambda: None)
    monkeypatch.setattr(execution_state, "_require_methods", lambda *_: None)
    monkeypatch.setattr(execution_state, "selected_client", lambda _: client)
    return screen, client, app


def _cover(screen, app):
    modal = ModalScreen()
    app.screen, app.screen_stack = modal, [screen, modal]
    screen.is_current = False


def test_only_actual_modal_coverage_or_ended_visit_defers_refresh(monkeypatch):
    screen, client, app = _visit(monkeypatch)
    assert not execution_state._defer_selected_refresh(screen, client, 1)
    _cover(screen, app)
    assert execution_state._defer_selected_refresh(screen, client, 1)
    assert not screen._console_process_surface_reconciled

    # A known original project modal has its own strict decision currentness.
    with monkeypatch.context() as owned:
        owned.setattr(execution_decisions, "is_owned_project_modal", lambda *_: True)
        assert not execution_state._defer_selected_refresh(screen, client, 1)
    app.screen = Screen()
    app.screen_stack = [screen, app.screen]
    assert not execution_state._defer_selected_refresh(screen, client, 1)
    client._view, client._visit = None, 2
    assert execution_state._defer_selected_refresh(screen, client, 1)
    client._channel_error = ConsoleChannelLost("actual channel loss")
    assert not execution_state._defer_selected_refresh(screen, client, 1)


@pytest.mark.asyncio
async def test_refresh_covered_during_read_keeps_values_and_defers_adoption(
    monkeypatch,
):
    screen, client, app = _visit(monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    previous = client.projection_view

    async def read(*_, **__):
        entered.set()
        await release.wait()
        raise ConsoleChannelLost("Console attachment visit changed")

    client.read_projection = read
    task = asyncio.create_task(execution_state.refresh_selected_view(screen))
    await entered.wait()
    _cover(screen, app)
    release.set()
    assert await task is False
    assert client.projection_view is previous
    assert not screen._console_process_surface_reconciled


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["active", "channel", "source", "producer"])
async def test_refresh_does_not_hide_active_source_or_producer_failures(
    monkeypatch, failure
):
    screen, client, app = _visit(monkeypatch)
    error = (
        RuntimeError("producer failure")
        if failure == "producer"
        else ConsoleChannelLost("failure")
    )

    async def read(*_, **__):
        if failure != "active":
            _cover(screen, app)
        if failure == "channel":
            client._channel_error = error
        if failure == "source":
            client._source_revision += 1
        raise error

    client.read_projection = read
    with pytest.raises(type(error)) as caught:
        await execution_state.refresh_selected_view(screen)
    assert caught.value is error


@pytest.mark.asyncio
async def test_new_visit_reconciles_after_the_superseded_worker_releases_its_flag(
    monkeypatch,
):
    screen, client, app = _visit(monkeypatch)
    scheduled = []
    screen._start_console_attach_reconciliation = lambda: None
    screen.call_after_refresh = scheduled.append

    async def refresh(*_, **__):
        assert screen._console_attach_reconcile_running
        client._visit = screen._console_process_visit = 3
        screen._console_process_surface_reconciled = False
        return False

    monkeypatch.setattr(execution_state, "refresh_selected_view", refresh)
    await execution_state.reconcile_selected_view(screen)
    assert not screen._console_attach_reconcile_running
    assert scheduled == [screen._start_console_attach_reconciliation]
    assert not screen._console_process_surface_reconciled
