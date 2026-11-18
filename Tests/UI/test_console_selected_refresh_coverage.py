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


def _resumed_publication(monkeypatch):
    """Exercise real selection predicates with inert publication fixtures."""
    original_selection = execution_state.selected_client
    screen, client, app = _visit(monkeypatch)
    monkeypatch.setattr(execution_state, "selected_client", original_selection)
    from tldw_chatbook.Chat import console_execution_readers

    monkeypatch.setattr(console_execution_readers, "require_reader_sources", lambda: None)
    for name in ("ConsoleProjectionView", "ConsoleProjection", "ConsoleSessionComposition"):
        monkeypatch.setattr(execution_state, name, SimpleNamespace)
    row = client.projection_view.session("chat")
    row.settings_revision = 2
    client._projection = SimpleNamespace(
        source_revision=3, session_id=row.id, incarnation_id=row.incarnation_id,
    )
    client.projection_view.source_revision = 3
    client.projection_view.projections = (client._projection,)
    client.session_composition = SimpleNamespace(
        session_id=row.id, incarnation_id=row.incarnation_id, settings_revision=2,
    )
    client._visit = screen._console_process_visit = 3
    client._intake_open = True

    def require_view(view, visit):
        if (view is not client._view or visit != client._visit
            or app.screen is not view or client._channel_error is not None):
            raise ConsoleChannelLost("original view boundary refused")

    client.require_view = require_view
    return screen, client, app


@pytest.mark.asyncio
async def test_resumed_fenced_publication_refreshes_without_rendering_old_values(monkeypatch):
    screen, client, _ = _resumed_publication(monkeypatch)
    with pytest.raises(RuntimeError, match="selected projection is unavailable"):
        execution_state.selected_client(screen)
    assert execution_state.defer_selected_passive_render(screen)
    scheduled = []
    screen.run_worker = lambda callback, **_: scheduled.append(callback) or SimpleNamespace(is_finished=False)
    execution_state.refresh_selected_when_due(screen)
    execution_state.refresh_selected_when_due(screen)
    assert len(scheduled) == 1
    calls = []

    async def refresh(current_screen, *, checkpoint=None):
        calls.append((current_screen, checkpoint))
        return False  # No renderer can claim an interrupted refresh succeeded.

    monkeypatch.setattr(execution_state, "refresh_selected_view", refresh)
    await scheduled[0]()
    assert calls == [(screen, "mount")]
    assert not screen._console_attach_reconcile_running
    with pytest.raises(RuntimeError, match="selected projection is unavailable"):
        execution_state.selected_client(screen)


@pytest.mark.asyncio
async def test_resumed_refresh_only_exposes_new_current_publication(monkeypatch):
    screen, client, _ = _resumed_publication(monkeypatch)
    calls = []

    async def read_and_present(current_screen, current_client, visit, checkpoint):
        calls.append((current_screen, current_client, visit, checkpoint))
        # The transport's actual checked adoption is covered by its own tests;
        # here the new coherent revision models its successful return only.
        client.projection_view.source_revision = client._source_revision
        client._projection.source_revision = client._source_revision
        assert execution_state.selected_client(screen) is client
        return True

    monkeypatch.setattr(execution_state, "_refresh_current_selected_view", read_and_present)
    screen.call_later = lambda *_: None
    assert await execution_state.refresh_selected_view(screen, checkpoint="mount")
    assert calls == [(screen, client, 3, "mount")]
    assert not execution_state.defer_selected_passive_render(screen)


@pytest.mark.parametrize("changed", ["same_visit", "future", "frame", "composition", "channel", "view"])
def test_resume_does_not_accept_mixed_or_unrelated_publications(monkeypatch, changed):
    screen, client, app = _resumed_publication(monkeypatch)
    if changed == "same_visit":
        screen._console_process_mounted_visit = client._visit
    elif changed == "future":
        client.projection_view.source_revision = client._source_revision + 1
    elif changed == "frame":
        client._projection.source_revision = 2
    elif changed == "composition":
        client.session_composition.settings_revision += 1
    elif changed == "channel":
        client._channel_error = ConsoleChannelLost("actual channel failure")
    else:
        app.screen = Screen()
    if changed in {"same_visit", "future"}:
        assert not execution_state._resumed_source_refresh(screen, client)
    else:
        with pytest.raises((RuntimeError, ConsoleChannelLost)):
            execution_state._resumed_source_refresh(screen, client)
    with pytest.raises((RuntimeError, ConsoleChannelLost)):
        execution_state.selected_client(screen)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [None, "navigation", "marks", "source", "visit", "session"])
async def test_explicit_return_waits_for_this_visits_checked_display(monkeypatch, changed):
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController

    screen, client, app = _resumed_publication(monkeypatch)
    app.conversation_local_marks_service = object()
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._screen = screen
    controller.app_instance = app
    controller._manual_navigation_generation = 5
    reads = []

    def store():
        # This is the original g31 failure when the return races the mount.
        reads.append(execution_state.selected_client(screen))
        return SimpleNamespace(active_session_id=None)

    controller._chat_store_accessor = store
    await controller.acknowledge_explicit_console_return()
    assert not reads

    async def mount(current_screen, *, checkpoint):
        assert current_screen is screen and checkpoint == "mount"
        client._projection.source_revision = client.projection_view.source_revision = client._source_revision
        return True

    work = []
    screen._console_process_view_started = True
    screen._console_credential_poll_timer = object()
    screen.run_worker = lambda callback, **options: work.append((callback, options))
    monkeypatch.setattr(execution_state, "refresh_selected_view", mount)
    await execution_state.reconcile_selected_view(screen)
    assert screen._console_pending_manual_return is None
    (resume, options), = work
    assert options == {"group": "console-manual-read-return", "exclusive": True}
    if changed == "navigation":
        controller._manual_navigation_generation += 1
    elif changed == "marks":
        app.conversation_local_marks_service = object()
    elif changed == "source":
        client._source_revision += 1
    elif changed == "visit":
        client._visit += 1
    elif changed == "session":
        row = client.projection_view.session("chat")
        row.incarnation_id = client._projection.incarnation_id = client.session_composition.incarnation_id = "another"
    await resume()
    assert reads == ([client] if changed is None else [])


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

    client.read_view_display = read
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

    client.read_view_display = read
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


@pytest.mark.parametrize("state", ["covered", "revoked", "teardown", "unmounted", "failed"])
def test_passive_timers_leave_previous_progress_untouched(monkeypatch, state):
    from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail

    screen, client, app = _visit(monkeypatch)
    client._readiness_deadline = 0
    if state == "covered":
        _cover(screen, app)
    elif state == "revoked":
        client._view, client._visit = None, 2
    elif state == "teardown":
        screen._console_chat_tearing_down = True
    elif state == "unmounted":
        screen.is_mounted = False
    else:
        screen._console_process_content_failed = True

    def unexpected(*_, **__):
        raise AssertionError("Passive obsolete view requested or painted values")

    monkeypatch.setattr(execution_state, "selected_client", unexpected)
    screen.run_worker = unexpected
    button = SimpleNamespace(is_attached=True, label="Progress: 3 queued")
    counts = {"chat": 3}
    rail = SimpleNamespace(
        is_attached=True, screen=screen,
        _agent_progress_state=unexpected,
        _progress_button=button, _progress_counts=counts,
        _refresh_progress_navigation=unexpected,
    )
    execution_state.refresh_selected_when_due(screen)
    ConsoleLeftRail._sync_progress_count(rail)
    assert button.label == "Progress: 3 queued"
    assert rail._progress_counts is counts


@pytest.mark.parametrize("owned_modal", [False, True])
@pytest.mark.parametrize("failure", ["channel", "source"])
def test_current_passive_timers_keep_original_reads_and_failures(monkeypatch, owned_modal, failure):
    from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail

    screen, client, app = _visit(monkeypatch)
    if owned_modal:
        _cover(screen, app)
        monkeypatch.setattr(execution_decisions, "is_owned_project_modal", lambda *_: True)
    client._readiness_deadline = 0
    scheduled = []
    screen.run_worker = lambda *args, **kwargs: scheduled.append((args, kwargs))
    execution_state.refresh_selected_when_due(screen)
    assert len(scheduled) == 1  # Owned modal uses only the independent decision read.
    reads = []
    rail = SimpleNamespace(
        is_attached=True, screen=screen,
        _agent_progress_state=lambda: (reads.append(True) or (5, {"chat": 5})),
        _progress_button=SimpleNamespace(is_attached=True, label="old"),
        _progress_counts={}, _refresh_progress_navigation=None,
    )
    ConsoleLeftRail._sync_progress_count(rail)
    assert reads == [True]
    assert rail._progress_button.label == "Progress: 5 queued"

    error = (
        ConsoleChannelLost("actual active channel loss")
        if failure == "channel" else RuntimeError("actual active source change")
    )
    if failure == "channel":
        client._channel_error = error

    def failed(*_, **__):
        raise error

    monkeypatch.setattr(execution_state, "selected_client", failed)
    if failure == "source":
        monkeypatch.setattr(execution_state, "_require_sources", failed)
    rail._agent_progress_state = failed
    with pytest.raises(type(error)) as refreshed:
        execution_state.refresh_selected_when_due(screen)
    assert refreshed.value is error
    with pytest.raises(type(error)) as progress:
        ConsoleLeftRail._sync_progress_count(rail)
    assert progress.value is error
