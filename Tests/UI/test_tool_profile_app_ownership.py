"""Tool Profile shutdown admission and the existing process watchdog order."""

import asyncio
import threading

import pytest
from textual.app import App

from Tests.private_profile import private_profile_test
from Tests.Tool_Packs.test_operations import result_for
from tldw_chatbook.Tool_Packs.operations import ToolProfileWriteUnavailable


@pytest.mark.asyncio
@private_profile_test
async def test_tool_writes_drain_before_a_later_owner_can_fail(request, monkeypatch):
    from tldw_chatbook.app import TldwCli

    app = object.__new__(TldwCli)
    owner = app._get_tool_profile_operations()
    entered, release = threading.Event(), threading.Event()
    later = []

    async def fail_later(self):
        later.append("recovery")
        raise RuntimeError("later owner failed")

    monkeypatch.setattr(TldwCli, "_shutdown_recovery_service", fail_later)

    def write(cancelled):
        entered.set()
        assert release.wait(5)
        return result_for("remove")

    task = owner.start("remove", "research", write)
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        shutdown = asyncio.create_task(app._shutdown_app_owned_lifecycles())
        await asyncio.sleep(0)
        assert not shutdown.done() and later == []
        with pytest.raises(ToolProfileWriteUnavailable, match="shutdown"):
            owner.start("import", "another", write)
        shutdown.cancel()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        assert not task.done()
    finally:
        release.set()
        await task
    with pytest.raises(RuntimeError, match="later owner failed"):
        await app._shutdown_app_owned_lifecycles()
    assert (
        later == ["recovery"]
        and task.result().result.tombstone.profile_id == "research"
    )


@pytest.mark.asyncio
@private_profile_test
async def test_shutdown_cannot_lazily_construct_an_unused_owner(request, monkeypatch):
    from tldw_chatbook.app import TldwCli

    app = object.__new__(TldwCli)

    async def stop(self):
        raise RuntimeError("stop before unrelated owners")

    monkeypatch.setattr(TldwCli, "_shutdown_recovery_service", stop)
    with pytest.raises(RuntimeError, match="stop before"):
        await app._shutdown_app_owned_lifecycles()
    assert getattr(app, "_tool_profile_operations", None) is None
    with pytest.raises(ToolProfileWriteUnavailable, match="shutdown"):
        app._get_tool_profile_operations()


@pytest.mark.asyncio
@private_profile_test
async def test_existing_watchdog_is_armed_before_first_app_owned_drain(
    request, monkeypatch
):
    import tldw_chatbook.app as app_module

    app = object.__new__(app_module.TldwCli)
    events = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def drain():
        events.append("drain")
        entered.set()
        await release.wait()

    async def textual_shutdown(self):
        events.append("textual")

    app._shutdown_app_owned_lifecycles = drain
    monkeypatch.setattr(
        app_module, "arm_exit_watchdog", lambda **kwargs: events.append("watchdog")
    )
    monkeypatch.setattr(App, "_shutdown", textual_shutdown)
    task = asyncio.create_task(app._shutdown())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert events == ["watchdog", "drain"]
    finally:
        release.set()
        await task
    assert events == ["watchdog", "drain", "textual"]


@pytest.mark.asyncio
@private_profile_test
async def test_real_embedded_app_remains_outside_process_exit_authority(
    request, monkeypatch
):
    # Run the existing assertion unchanged with its config source selected
    # before importing the real app (the process-isolation fixture contract).
    from Tests.App.test_app_shutdown import (
        test_mounting_the_real_app_under_test_arms_no_watchdog,
    )

    await test_mounting_the_real_app_under_test_arms_no_watchdog(monkeypatch)
