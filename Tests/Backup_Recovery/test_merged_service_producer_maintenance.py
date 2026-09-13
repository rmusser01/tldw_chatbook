"""Backup maintenance settles the upstream thread-offloaded owner services."""

import asyncio
import sqlite3
import threading
import time

import pytest

from Tests.Backup_Recovery.test_participant_lifetimes import local_root as _local_root

local_root = _local_root


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["writing", "research"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_scope_drains_actual_worker_before_closing_cache(
    tmp_path, monkeypatch, local_root, route, cancel
):
    if route == "writing":
        from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService
        from tldw_chatbook.Writing_Interop.writing_scope_service import WritingScopeService

        local = LocalWritingService(tmp_path / "writing.sqlite")
        scope = WritingScopeService(local_service=local, server_service=None)
        method = "list_projects"
    else:
        from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService
        from tldw_chatbook.Research_Interop.research_scope_service import ResearchScopeService

        local = LocalResearchService(tmp_path / "research.sqlite")
        scope = ResearchScopeService(local_service=local, server_service=None)
        method = "list_sessions"
    entered, release = threading.Event(), threading.Event()
    connections = []
    original = getattr(local, method)

    def blocked(*args, **kwargs):
        result = original(*args, **kwargs)
        connections.append(local._connect())
        entered.set()
        assert release.wait(5)
        assert connections[-1].execute("SELECT 1").fetchone()[0] == 1
        return result

    monkeypatch.setattr(local, method, blocked)
    task = asyncio.create_task(getattr(scope, method)())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        scope._maintenance_close_admission()
        with pytest.raises(Exception, match="runtime_producer_paused"):
            await getattr(scope, method)()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not await scope._maintenance_drain(time.monotonic() + 0.02)
        release.set()
        if not cancel:
            await task
        assert await scope._maintenance_drain(time.monotonic() + 2)
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("SELECT 1")
        scope._maintenance_resume()
        monkeypatch.setattr(local, method, original)
        assert await getattr(scope, method)() == []
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        local.close()


@pytest.mark.asyncio
async def test_notes_runtime_pause_settles_admitted_work_and_resumes(
    tmp_path, local_root
):
    from Tests.Notes.test_notes_sync_runtime import _Adapter, _input, _owner, _store

    store = _store(tmp_path)
    adapter = _Adapter([_input()])
    owner, _coordinator, _watcher = _owner(
        store=store, admitted=True, adapter=adapter
    )
    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    owner._watcher_factory = lambda schedule: PollingNotesSyncWatcher(
        lambda: (), schedule, interval_seconds=10
    )
    await owner.start()
    initial_watcher = owner._watcher
    started, release = asyncio.Event(), asyncio.Event()
    original = adapter.observe_root

    async def blocked(root):
        started.set()
        await release.wait()
        return await original(root)

    adapter.observe_root = blocked
    task = asyncio.create_task(owner.check_root("root-1"))
    try:
        await asyncio.wait_for(started.wait(), 2)
        owner._maintenance_close_admission()
        with pytest.raises(Exception, match="runtime_producer_paused"):
            await owner.check_root("root-1")
        assert not await owner._maintenance_drain(time.monotonic() + 0.02)
        release.set()
        await task
        assert await owner._maintenance_drain(time.monotonic() + 2)
        assert not owner._closing
        owner._maintenance_resume()
        assert owner._watcher is not initial_watcher
        assert not owner._watcher_task.done()
        await owner.check_root("root-1")
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await owner.shutdown()


@pytest.mark.asyncio
async def test_cancelled_notes_command_keeps_native_worker_owned(
    tmp_path, monkeypatch, local_root
):
    from Tests.Notes.test_notes_sync_runtime import _Adapter, _input, _owner, _store

    store = _store(tmp_path)
    owner, _coordinator, _watcher = _owner(
        store=store, admitted=True, adapter=_Adapter([_input()])
    )
    await owner.start()
    entered, release = threading.Event(), threading.Event()
    connections = []
    original = store.get_root

    def blocked(root_id):
        result = original(root_id)
        connections.append(store._get_connection())
        entered.set()
        assert release.wait(5)
        assert connections[-1].execute("SELECT 1").fetchone()[0] == 1
        return result

    monkeypatch.setattr(store, "get_root", blocked)
    command = asyncio.create_task(owner.check_root("root-1"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        owner._maintenance_close_admission()
        command.cancel()
        with pytest.raises(asyncio.CancelledError):
            await command
        assert not await owner._maintenance_drain(time.monotonic() + 0.02)
        release.set()
        assert await owner._maintenance_drain(time.monotonic() + 2)
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("SELECT 1")
        owner._maintenance_resume()
    finally:
        release.set()
        await asyncio.gather(command, return_exceptions=True)
        monkeypatch.setattr(store, "get_root", original)
        await owner.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_notes_start_admitted_before_pause_settles_its_child(
    tmp_path, local_root, cancel
):
    from Tests.Notes.test_notes_sync_runtime import _Adapter, _input, _owner, _store

    store = _store(tmp_path)
    owner, _coordinator, _watcher = _owner(
        store=store, admitted=True, adapter=_Adapter([_input()])
    )
    caller = asyncio.create_task(owner.start())
    try:
        await asyncio.sleep(0)
        assert owner._start_task is not None
        owner._maintenance_close_admission()
        if cancel:
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
        else:
            await caller
        assert await owner._maintenance_drain(time.monotonic() + 3)
        await owner._start_task
        owner._maintenance_resume()
        await owner.check_root("root-1")
    finally:
        await asyncio.gather(caller, return_exceptions=True)
        await owner.shutdown()


@pytest.mark.asyncio
async def test_notes_hint_admitted_before_pause_completes_one_round(
    tmp_path, local_root
):
    from Tests.Notes.test_notes_sync_runtime import _Adapter, _input, _owner, _store

    adapter = _Adapter([_input()])
    owner, _coordinator, _watcher = _owner(
        store=_store(tmp_path), admitted=True, adapter=adapter
    )
    await owner.start()
    observations = adapter.observe_calls
    try:
        hint = owner.schedule_hint("root-1")
        assert hint is not None
        owner._maintenance_close_admission()
        assert owner.schedule_hint("root-1") is None
        assert await owner._maintenance_drain(time.monotonic() + 3)
        await hint
        assert adapter.observe_calls == observations + 1
        owner._maintenance_resume()
    finally:
        await owner.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["writing", "research", "notes"])
@pytest.mark.parametrize("borrower", ["transaction", "operation"])
async def test_runtime_close_defers_for_a_foreign_native_borrower(
    tmp_path, local_root, route, borrower
):
    from tldw_chatbook.Backup_Recovery.participants import _core_operation

    if route == "writing":
        from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService
        from tldw_chatbook.Writing_Interop.writing_scope_service import WritingScopeService

        local = LocalWritingService(tmp_path / "writing.sqlite")
        runtime = WritingScopeService(local_service=local, server_service=None)
        connection = local._connect()
    elif route == "research":
        from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService
        from tldw_chatbook.Research_Interop.research_scope_service import ResearchScopeService

        local = LocalResearchService(tmp_path / "research.sqlite")
        runtime = ResearchScopeService(local_service=local, server_service=None)
        connection = local._connect()
    else:
        from Tests.Notes.test_notes_sync_runtime import _Adapter, _input, _owner, _store

        local = _store(tmp_path)
        runtime, _coordinator, _watcher = _owner(
            store=local, admitted=True, adapter=_Adapter([_input()])
        )
        await runtime.start()
        connection = local._get_connection()
    operation = _core_operation(local) if borrower == "operation" else None
    try:
        if operation is not None:
            operation.__enter__()
        else:
            connection.execute("BEGIN")
        runtime._maintenance_close_admission()
        assert not await runtime._maintenance_drain(time.monotonic() + 1)
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        if operation is not None:
            operation.__exit__(None, None, None)
            operation = None
        else:
            connection.rollback()
        assert await runtime._maintenance_drain(time.monotonic() + 2)
        runtime._maintenance_resume()
    finally:
        if operation is not None:
            operation.__exit__(None, None, None)
        if route == "notes":
            await runtime.shutdown()
        else:
            local.close()
