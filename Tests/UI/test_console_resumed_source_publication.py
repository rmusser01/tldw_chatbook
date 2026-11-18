"""A resumed screen publishes the current source before handing out its readers."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_selected_refresh_coverage import _resumed_publication
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
from tldw_chatbook.UI.Console_Modules import execution_state


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["current", "failed_write", "lost_visit", "wrong_frame"])
async def test_resume_waits_for_actual_workspace_write_then_visible_mount(monkeypatch, outcome):
    screen, client, app = _resumed_publication(monkeypatch)
    release_write, read_started, release_read = (asyncio.Event() for _ in range(3))
    renders, reads = [], []
    client._intake_open = False
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
        release_read.set()
        if outcome == "current":
            await resumed
            assert execution_state.selected_client(screen) is client
            assert screen._console_resumed_publication == (
                client, client._visit, client.projection_view, client._projection,
            )
            assert renders == [screen._start_console_attach_reconciliation]
        elif outcome == "lost_visit":
            with pytest.raises(ConsoleChannelLost):
                await resumed
        else:
            with pytest.raises(RuntimeError, match="resumed publication changed"):
                await resumed
        assert reads == [("chat", "incarnation", "mount")]
    if outcome != "current":
        assert not renders and not hasattr(screen, "_console_resumed_publication")
    assert screen._console_resuming_source is None
