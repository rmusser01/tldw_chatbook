"""Decision hints schedule existing reads; they never publish decision values."""

import asyncio
from functools import partial
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_selected_refresh_coverage import _visit
from tldw_chatbook.UI.Console_Modules import execution_decisions, execution_state


def _scope(monkeypatch):
    # Orchestration doubles only, following the existing coverage fixture.
    # Neither these values nor the worker stand-in certify native retirement.
    screen, client, app = _visit(monkeypatch)
    client._attachment_id = "attachment"
    client._decision_hint = client._decision_rendered_hint = None
    client.projection_view.source_revision = client._source_revision
    client.require_view = lambda view, visit: (
        view is screen and visit == client._visit
    ) or pytest.fail("wrong visit")
    screen._console_process_view_started = True
    screen._console_credential_poll_timer = object()
    scheduled, workers = [], []
    screen.call_later = (
        lambda callback, *args: scheduled.append(partial(callback, *args)) or True
    )

    def start(callback, **kwargs):
        assert kwargs == {"group": "console-attach-reconciliation", "exclusive": False}
        worker = SimpleNamespace(is_finished=False, work=callback)
        workers.append(worker)
        return worker

    screen.run_worker = start
    monkeypatch.setattr(execution_decisions, "is_owned_project_modal", lambda *_: False)
    return screen, client, app, scheduled, workers


def _hint(client, revision=1):
    row = client.projection_view.session(client.projection_view.active_session_id)
    return (
        client._source_revision,
        row.id,
        row.incarnation_id,
        client._attachment_id,
        client._visit,
        revision,
    )


def _deliver(scheduled):
    callbacks = scheduled[:]
    scheduled.clear()
    for callback in callbacks:
        callback()


def test_current_decision_wakes_before_ttl_and_coalesces_original_worker(monkeypatch):
    screen, client, _, scheduled, workers = _scope(monkeypatch)
    assert client._readiness_deadline == float("inf")
    assert execution_state.request_selected_decision_refresh(client, _hint(client))
    assert execution_state.request_selected_decision_refresh(client, _hint(client, 2))
    assert not execution_state.request_selected_decision_refresh(
        client, _hint(client, 2)
    )
    assert not execution_state.request_selected_decision_refresh(client, _hint(client))
    assert not workers  # The authenticated receiver enters the actual screen pump.
    _deliver(scheduled)
    assert len(workers) == 1
    assert screen._console_attach_reconcile_worker is workers[0]
    assert client._decision_rendered_hint is None
    assert execution_state._decision_refresh_due(client)


@pytest.mark.parametrize("field", range(5))
def test_wrong_decision_scope_never_requests_a_projection(monkeypatch, field):
    _, client, _, scheduled, workers = _scope(monkeypatch)
    hint = list(_hint(client))
    hint[field] = hint[field] + 1 if type(hint[field]) is int else hint[field] + "-old"
    assert not execution_state.request_selected_decision_refresh(client, tuple(hint))
    assert client._decision_hint is None and not scheduled and not workers


@pytest.mark.parametrize(
    "state", ["detached", "unmounted", "source-unpublished", "different-app"]
)
def test_late_decision_hint_cannot_wake_an_obsolete_view(monkeypatch, state):
    screen, client, _, scheduled, workers = _scope(monkeypatch)
    hint = _hint(client)
    if state == "detached":
        client._view = None
    elif state == "unmounted":
        screen.is_mounted = False
    elif state == "source-unpublished":
        client.projection_view.source_revision -= 1
    else:
        screen.app_instance = object()
    assert not execution_state.request_selected_decision_refresh(client, hint)
    assert client._decision_hint is None and not scheduled and not workers


@pytest.mark.asyncio
async def test_hint_during_checked_read_survives_and_requests_one_followup(monkeypatch):
    screen, client, _, scheduled, workers = _scope(monkeypatch)
    first, later = _hint(client), _hint(client, 2)
    assert execution_state.request_selected_decision_refresh(client, first)
    _deliver(scheduled)
    entered, complete = asyncio.Event(), asyncio.Event()

    async def render(*_):
        entered.set()
        await complete.wait()
        return True

    monkeypatch.setattr(execution_state, "_refresh_current_selected_view", render)
    task = asyncio.create_task(workers[0].work())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert screen._console_attach_reconcile_running
        assert execution_state.request_selected_decision_refresh(client, later)
        _deliver(scheduled)
        assert len(workers) == 1
        complete.set()
        await task
    finally:
        complete.set()
        await asyncio.gather(task, return_exceptions=True)
    workers[0].is_finished = True
    assert client._decision_rendered_hint == first
    assert client._decision_hint == later and execution_state._decision_refresh_due(
        client
    )
    _deliver(scheduled)
    assert len(workers) == 2
    await workers[1].work()
    workers[1].is_finished = True
    assert client._decision_rendered_hint == later
    assert not execution_state._decision_refresh_due(client) and not scheduled


@pytest.mark.asyncio
async def test_hint_during_direct_refresh_waits_for_that_existing_read(monkeypatch):
    screen, client, _, scheduled, workers = _scope(monkeypatch)
    first, later = _hint(client), _hint(client, 2)
    assert execution_state.request_selected_decision_refresh(client, first)
    entered, complete = asyncio.Event(), asyncio.Event()

    async def render(*_):
        entered.set()
        await complete.wait()
        return True

    monkeypatch.setattr(execution_state, "_refresh_current_selected_view", render)
    task = asyncio.create_task(execution_state.refresh_selected_view(screen))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        _deliver(scheduled)
        assert not workers and screen._console_selected_refresh_count == 1
        assert execution_state.request_selected_decision_refresh(client, later)
        _deliver(scheduled)
        assert not workers
        complete.set()
        assert await task
    finally:
        complete.set()
        await asyncio.gather(task, return_exceptions=True)
    assert screen._console_selected_refresh_count == 0
    assert client._decision_rendered_hint == first
    assert execution_state._decision_refresh_due(client)
    _deliver(scheduled)
    assert len(workers) == 1


@pytest.mark.asyncio
async def test_deferred_projection_does_not_consume_the_hint(monkeypatch):
    screen, client, _, _, _ = _scope(monkeypatch)
    hint = _hint(client)
    client._decision_hint = hint

    async def deferred(*_):
        return False

    monkeypatch.setattr(execution_state, "_refresh_current_selected_view", deferred)
    assert not await execution_state.refresh_selected_view(screen)
    assert client._decision_rendered_hint is None
    assert execution_state._decision_refresh_due(client)


def test_dropped_hint_leaves_original_periodic_reconciliation(monkeypatch):
    screen, client, _, _, workers = _scope(monkeypatch)
    execution_state.refresh_selected_when_due(screen)
    assert not workers
    client._readiness_deadline = 0
    execution_state.refresh_selected_when_due(screen)
    assert len(workers) == 1 and client._decision_hint is None
