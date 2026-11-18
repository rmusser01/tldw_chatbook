"""A resumed screen publishes the current source before handing out its readers."""

import asyncio
from types import SimpleNamespace

import pytest
from textual.screen import ModalScreen, Screen

from Tests.UI.test_console_selected_refresh_coverage import _resumed_publication
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
from tldw_chatbook.UI.Console_Modules import execution_state


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [
    "current", "failed_write", "lost_visit", "wrong_frame", "moved", "covered",
    "moved_channel_failure", "moved_source_change", "moved_attachment_change",
    "moved_wrong_client_owner", "rebound",
])
async def test_resume_waits_for_actual_workspace_write_then_visible_mount(monkeypatch, outcome):
    screen, client, app = _resumed_publication(monkeypatch)
    release_write, read_started, release_read = (asyncio.Event() for _ in range(3))
    renders, reads = [], []
    client._intake_open = False
    client._attachment_id = "original-attachment"
    client.attach_view = lambda actual: client._visit if actual is screen else pytest.fail()
    screen._start_console_attach_reconciliation = lambda: None
    screen.call_after_refresh = renders.append

    async def write():
        await release_write.wait()
        if outcome == "failed_write":
            raise RuntimeError("binding failed")
        client._intake_open = True

    async def read(actual, visit, session_id, incarnation, *, checkpoint):
        client.require_view(actual, visit)
        assert client._intake_open
        reads.append((session_id, incarnation, checkpoint))
        read_started.set()
        await release_read.wait()
        client.require_view(actual, visit)
        client.projection_view.source_revision = client._source_revision
        client._projection.source_revision = client._source_revision
        return object() if outcome == "wrong_frame" else client._projection

    client.read_view_display = read
    operation = client._workspace_binding_task = asyncio.create_task(write())
    client._workspace_operations = {operation}
    resumed = asyncio.create_task(execution_state.resume_selected_view(screen))
    await asyncio.sleep(0)
    assert not resumed.done() and not reads
    assert execution_state.defer_selected_passive_render(screen)
    with pytest.raises(RuntimeError, match="selected projection"):
        execution_state.selected_client(screen)
    release_write.set()
    if outcome == "failed_write":
        with pytest.raises(RuntimeError, match="binding failed"):
            await resumed
        assert not reads
    else:
        await asyncio.wait_for(read_started.wait(), 2)
        assert not resumed.done() and not renders
        if outcome == "lost_visit":
            app.screen = SimpleNamespace()
        elif outcome == "rebound":
            client._visit += 1
        elif outcome.startswith("moved") or outcome == "covered":
            _transfer(app, screen, covered=outcome == "covered")
            if outcome == "moved_channel_failure":
                client._channel_error = ConsoleChannelLost("actual channel failed")
            elif outcome == "moved_source_change":
                client._source_revision += 1
            elif outcome == "moved_attachment_change":
                client._attachment_id = "replacement-attachment"
            elif outcome == "moved_wrong_client_owner":
                client._view = object()
            if outcome in {"moved", "covered"}:
                assert execution_state._awaiting_resume_publication(screen)
        release_read.set()
        if outcome == "current":
            await resumed
            assert execution_state.selected_client(screen) is client
            assert screen._console_resumed_publication == (
                client, client._visit, client.projection_view, client._projection,
            )
            assert renders == [screen._start_console_attach_reconciliation]
        elif outcome in {"moved", "covered", "rebound"}:
            await resumed
        elif outcome == "lost_visit" or outcome.startswith("moved_"):
            with pytest.raises(ConsoleChannelLost):
                await resumed
        else:
            with pytest.raises(RuntimeError, match="resumed publication changed"):
                await resumed
        assert reads == [("chat", "incarnation", "mount")]
    if outcome != "current":
        assert not renders and not hasattr(screen, "_console_resumed_publication")
    assert screen._console_resuming_source is None


def _transfer(app, previous, *, covered=False):
    """Model stack ownership only, never process/native retirement authority."""
    base = ModalScreen if covered else Screen

    class Destination(base):
        @property
        def app(self):
            return app

        @property
        def is_attached(self):
            return True

        @property
        def is_mounted(self):
            return True

    destination = Destination()
    app.screen = destination
    app.screen_stack = [previous, destination] if covered else [destination]
    previous.is_current = False
    return destination


@pytest.mark.asyncio
async def test_resume_cancel_leaves_actual_binding_task_owned(monkeypatch):
    screen, client, _ = _resumed_publication(monkeypatch)
    release = asyncio.Event()
    operation = client._workspace_binding_task = asyncio.create_task(release.wait())
    client._workspace_operations = {operation}
    resumed = asyncio.create_task(execution_state._publish_resumed_source(
        screen, client, client._visit, operation,
    ))
    await asyncio.sleep(0)
    resumed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await resumed
    assert not operation.done() and operation in client._workspace_operations
    assert screen._console_resuming_source is None
    release.set()
    await operation


@pytest.mark.asyncio
async def test_navigation_during_binding_join_never_issues_display(monkeypatch):
    screen, client, app = _resumed_publication(monkeypatch)
    release = asyncio.Event()
    operation = client._workspace_binding_task = asyncio.create_task(release.wait())
    client._workspace_operations = {operation}
    client.read_view_display = lambda *_args, **_kwargs: pytest.fail("old visit read issued")
    resumed = asyncio.create_task(execution_state._publish_resumed_source(
        screen, client, client._visit, operation,
    ))
    await asyncio.sleep(0)
    _transfer(app, screen)
    release.set()
    await resumed
    assert operation.done() and screen._console_resuming_source is None
    assert not hasattr(screen, "_console_resumed_publication")
