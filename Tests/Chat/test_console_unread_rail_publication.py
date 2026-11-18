"""Scheduling controls for a completed unread read during a source fence.

These unit controls do not qualify native retirement or source authority; the
real mounted rail-publication test exercises those original producers.
"""

import asyncio
from types import SimpleNamespace

import pytest


def _case(monkeypatch, *, selected=True, read_failure=False):
    from tldw_chatbook.UI.Console_Modules import execution_state
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController

    screen = object()
    client = SimpleNamespace(_view=screen, _visit=1)
    service = SimpleNamespace(manual_revision=2, unread_ids_for=lambda ids: set(ids))
    app = SimpleNamespace(
        console_execution_client=client if selected else None,
        conversation_local_marks_service=service,
    )
    state = {"key": (service, 2, ("profile", "owner")), "pending": {"chat"}, "values": {}}
    entered, release, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls, paints = [], []
    current = {"scope": None, "ready": not selected, "hidden": False, "authority": ("profile", "owner")}

    async def read(callback, ids):
        calls.append((callback, ids))
        entered.set()
        await release.wait()
        if read_failure:
            raise OSError("optional read failed")
        return callback(ids)

    def authority():
        assert current["ready"], "old projection was read during its source fence"
        return current["authority"]

    controller = SimpleNamespace(
        app_instance=app, _screen=screen, _manual_unread_cache=state,
        _console_switcher_authority=authority,
        _sync_console_workspace_context=lambda: paints.append(True),
    )
    monkeypatch.setattr(asyncio, "to_thread", read)
    monkeypatch.setattr(execution_state, "selected_rail_publication", lambda *_: current["scope"])
    monkeypatch.setattr(execution_state, "defer_selected_passive_render", lambda _: current["hidden"])

    async def publish():
        await finish.wait()
        current["ready"] = True
        current["scope"] = None
        return True

    publication = asyncio.create_task(publish()) if selected else None
    if publication is not None:
        current["scope"] = SimpleNamespace(task=publication)
    worker = asyncio.create_task(ConsoleWorkspaceController._load_manual_unread_rows(
        controller, service, state, ("chat",),
    ))
    return SimpleNamespace(**locals())


@pytest.mark.asyncio
@pytest.mark.parametrize("read_failure", [False, True])
async def test_completed_unread_read_waits_for_rail_without_rereading(monkeypatch, read_failure):
    case = _case(monkeypatch, read_failure=read_failure)
    await case.entered.wait()
    case.release.set()
    await asyncio.sleep(0)
    assert not case.worker.done()
    assert not case.paints and not case.state["values"]
    case.finish.set()
    await case.worker
    assert case.publication.result() is True
    assert len(case.calls) == 1
    assert case.state["values"] == {"chat": None if read_failure else True}
    assert not case.state["pending"] and case.paints == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["service", "revision", "cache", "authority", "visit", "hidden"])
async def test_unread_completion_rechecks_original_owners_after_rail(monkeypatch, change):
    case = _case(monkeypatch)
    await case.entered.wait()
    case.release.set()
    await asyncio.sleep(0)
    if change == "service":
        case.app.conversation_local_marks_service = object()
    elif change == "revision":
        case.service.manual_revision += 1
    elif change == "cache":
        case.controller._manual_unread_cache = None
    elif change == "authority":
        case.current["authority"] = ("different", "owner")
    elif change == "visit":
        case.client._visit += 1
    else:
        case.current["hidden"] = True
    case.finish.set()
    await case.worker
    assert case.publication.result() is True
    assert len(case.calls) == 1
    assert not case.state["values"] and not case.paints


@pytest.mark.asyncio
async def test_unread_observer_cancellation_does_not_cancel_rail(monkeypatch):
    case = _case(monkeypatch)
    await case.entered.wait()
    case.release.set()
    await asyncio.sleep(0)
    case.worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await case.worker
    assert not case.publication.done()
    assert not case.state["pending"] and not case.state["values"] and not case.paints
    case.finish.set()
    assert await case.publication is True


@pytest.mark.asyncio
async def test_unowned_stale_unread_projection_still_refuses(monkeypatch):
    case = _case(monkeypatch)
    await case.entered.wait()
    case.current["scope"] = None
    case.release.set()
    with pytest.raises(AssertionError, match="old projection"):
        await case.worker
    assert not case.state["values"] and not case.paints
    case.finish.set()
    await case.publication


@pytest.mark.asyncio
async def test_local_unread_read_retains_original_completion(monkeypatch):
    case = _case(monkeypatch, selected=False)
    await case.entered.wait()
    case.release.set()
    await case.worker
    assert len(case.calls) == 1
    assert case.state["values"] == {"chat": True} and case.paints == [True]
