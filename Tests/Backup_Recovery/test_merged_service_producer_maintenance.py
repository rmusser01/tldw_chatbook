"""Backup maintenance settles the upstream thread-offloaded owner services."""

import asyncio
import sqlite3
import threading
import time
from pathlib import Path

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
        from tldw_chatbook.Writing_Interop.local_writing_service import (
            LocalWritingService,
        )
        from tldw_chatbook.Writing_Interop.writing_scope_service import (
            WritingScopeService,
        )

        local = LocalWritingService(tmp_path / "writing.sqlite")
        scope = WritingScopeService(local_service=local, server_service=None)
        method = "list_projects"
    else:
        from tldw_chatbook.Research_Interop.local_research_service import (
            LocalResearchService,
        )
        from tldw_chatbook.Research_Interop.research_scope_service import (
            ResearchScopeService,
        )

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
@pytest.mark.parametrize("method", ["check_root", "binding_labels"])
async def test_cancelled_notes_command_keeps_native_worker_owned(
    tmp_path, monkeypatch, local_root, method
):
    from Tests.Notes.test_notes_sync_runtime import _Adapter, _input, _owner, _store

    store = _store(tmp_path)
    owner, _coordinator, _watcher = _owner(
        store=store, admitted=True, adapter=_Adapter([_input()])
    )
    await owner.start()
    await owner.check_root("root-1")
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
    args = ("root-1", ()) if method == "binding_labels" else ("root-1",)
    command = asyncio.create_task(getattr(owner, method)(*args))
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
@pytest.mark.parametrize("already_admitted", [False, True])
async def test_notes_binding_labels_obeys_maintenance(
    tmp_path, local_root, already_admitted
):
    from Tests.Notes.test_notes_sync_runtime import _Adapter, _input, _owner, _store

    class EmptyLabelsAdapter(_Adapter):
        async def build_binding_labels(self, root, plan, binding_ids, *, root_name):
            if binding_ids:
                raise AssertionError("this lifecycle fixture requests no labels")
            return ()

    adapter = EmptyLabelsAdapter([_input()])
    owner, _coordinator, _watcher = _owner(
        store=_store(tmp_path), admitted=True, adapter=adapter
    )
    await owner.start()
    await owner.check_root("root-1")
    command = None
    entered, release = asyncio.Event(), asyncio.Event()
    original = adapter.observe_root

    async def blocked(record):
        observed = await original(record)
        if asyncio.current_task() is command:
            entered.set()
            await release.wait()
        return observed

    try:
        if already_admitted:
            adapter.observe_root = blocked
            command = asyncio.create_task(owner.binding_labels("root-1", ()))
            await asyncio.wait_for(entered.wait(), 2)
        owner._maintenance_close_admission()
        with pytest.raises(Exception, match="runtime_producer_paused"):
            await owner.binding_labels("root-1", ())
        if command is not None:
            assert not await owner._maintenance_drain(time.monotonic() + 0.02)
            release.set()
            await command
        assert await owner._maintenance_drain(time.monotonic() + 2)
        adapter.observe_root = original
        owner._maintenance_resume()
        await owner.binding_labels("root-1", ())
    finally:
        release.set()
        if command is not None:
            await asyncio.gather(command, return_exceptions=True)
        owner._maintenance_resume()
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
        from tldw_chatbook.Writing_Interop.local_writing_service import (
            LocalWritingService,
        )
        from tldw_chatbook.Writing_Interop.writing_scope_service import (
            WritingScopeService,
        )

        local = LocalWritingService(tmp_path / "writing.sqlite")
        runtime = WritingScopeService(local_service=local, server_service=None)
        connection = local._connect()
    elif route == "research":
        from tldw_chatbook.Research_Interop.local_research_service import (
            LocalResearchService,
        )
        from tldw_chatbook.Research_Interop.research_scope_service import (
            ResearchScopeService,
        )

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


async def _receipt_owner(tmp_path, *, history):
    """Use the native receipt store/projection with an inert startup watcher."""
    from dataclasses import replace

    from Tests.Notes.test_notes_device_state_store import _binding
    from Tests.Notes.test_notes_sync_runtime import (
        _Adapter,
        _Folders,
        _input,
        _LocalNotes,
        _owner,
        _store,
    )
    from tldw_chatbook.Notes.notes_device_state_store import NotesSyncOperationRecord
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
    from tldw_chatbook.Notes.notes_sync_models import NotesSyncOperationState
    from tldw_chatbook.Notes.notes_sync_runtime import _ProductionRuntimeAdapter

    store = _store(tmp_path)
    adapter = _Adapter([_input()])
    projection = _ProductionRuntimeAdapter(
        store, NotesScopeService(_LocalNotes("body"), None, folder_repository=_Folders()),
        local_user_id="user-1", recovery_capacity_bytes=1024 * 1024,
    )
    adapter.build_receipt_labels = projection.build_receipt_labels
    owner, _, _ = _owner(store=store, admitted=True, adapter=adapter)
    await owner.start()
    if history:
        store.create_binding(replace(_binding(), note_scope_id="local_note"))
        store.create_operation(NotesSyncOperationRecord(
            operation_id="receipt-operation", root_id="root-1", binding_id="binding-1",
            kind="update_note", state=NotesSyncOperationState.PENDING,
            reason_code=None, observation_token="receipt-observation",  # nosec B106 - synthetic journal ID, not a credential
            expected_note_version=None, expected_file_digest=None,
        ))
        with store.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE notes_sync_operations SET state = 'completed' "
                "WHERE operation_id = ?", ("receipt-operation",),
            )
    return owner, store


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("finish", ["maintenance", "shutdown"])
@pytest.mark.parametrize("method,history", [
    ("list_completed_operations", False),
    ("list_completed_operations", True),
    ("get_root", True),
    ("list_bindings", True),
    ("active_binding_path_for_note", True),
])
async def test_notes_receipt_native_read_finishes_before_close(
    tmp_path, monkeypatch, local_root, method, history, cancel, finish
):
    """A receipt waiter ending must not let shutdown close its live SQLite worker."""
    owner, store = await _receipt_owner(tmp_path, history=history)
    entered, release, survived = threading.Event(), threading.Event(), threading.Event()
    connections = []
    original = getattr(store, method)

    def blocked(*args, **kwargs):
        result = original(*args, **kwargs)
        connection = store._get_connection()
        connections.append(connection)
        entered.set()
        assert release.wait(5)
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        survived.set()
        return result

    monkeypatch.setattr(store, method, blocked)
    location_read = method == "active_binding_path_for_note"
    command = asyncio.create_task(
        owner.note_file_location("note-1")
        if location_read
        else owner.write_receipts("root-1")
    )
    shutdown = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if cancel:
            command.cancel()
            with pytest.raises(asyncio.CancelledError):
                await command
        if finish == "maintenance":
            owner._maintenance_close_admission()
            assert not await owner._maintenance_drain(time.monotonic() + 0.02)
        else:
            shutdown = asyncio.create_task(owner.shutdown())
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(shutdown), 0.02)
        release.set()
        if not cancel:
            result = await command
            if location_read:
                assert result == str(
                    Path(owner._root_paths["root-1"]) / "folder/note.md"
                )
            else:
                assert [(row.kind, row.relative_path) for row in result] == (
                    [("update_note", "folder/note.md")] if history else []
                )
        if finish == "maintenance":
            assert await owner._maintenance_drain(time.monotonic() + 2)
        else:
            await asyncio.wait_for(shutdown, 2)
        assert survived.is_set()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("SELECT 1")
    finally:
        release.set()
        await asyncio.gather(command, return_exceptions=True)
        monkeypatch.setattr(store, method, original)
        if shutdown is not None:
            await asyncio.gather(shutdown, return_exceptions=True)
        owner._maintenance_resume()
        await owner.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("location_read", [False, True])
async def test_paused_notes_receipts_refuse_backup_pause_and_resume(
    tmp_path, local_root, history, location_read
):
    """Normal root pause allows history, but global maintenance fences new reads."""
    owner, _ = await _receipt_owner(tmp_path, history=history)

    async def read():
        return (
            await owner.note_file_location("note-1")
            if location_read
            else await owner.write_receipts("root-1")
        )

    try:
        await owner.pause_root("root-1")
        before = await read()
        owner._maintenance_close_admission()
        with pytest.raises(Exception, match="runtime_producer_paused"):
            await read()
        assert await owner._maintenance_drain(time.monotonic() + 2)
        owner._maintenance_resume()
        assert await read() == before
    finally:
        owner._maintenance_resume()
        await owner.shutdown()


@pytest.mark.asyncio
async def test_closed_notes_location_does_not_reopen_native_store(
    tmp_path, monkeypatch, local_root
):
    owner, store = await _receipt_owner(tmp_path, history=True)
    await owner.shutdown()
    reads = []
    original = store.active_binding_path_for_note

    def observed(note_id):
        reads.append(note_id)
        return original(note_id)

    monkeypatch.setattr(store, "active_binding_path_for_note", observed)
    try:
        assert await owner.note_file_location("note-1") == ""
        assert not reads
    finally:
        store.close()
