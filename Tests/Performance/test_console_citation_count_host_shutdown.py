"""Original citation-count worker must retain its actual database callback."""

import asyncio
import inspect
import sqlite3
import sys
import threading
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
from types import CodeType, SimpleNamespace

import pytest
from textual.app import App
from textual.screen import Screen
from textual.worker import WorkerState

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@private_profile_test
async def test_original_citation_count_worker_retires_before_host_drain(
    request, tmp_path
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.citation_provenance_runtime import (
        CitationProvenanceRuntimePolicy,
    )
    from tldw_chatbook.Chat.citation_trace_repository import CitationTraceRepository
    from tldw_chatbook.DB import base_db
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.citation_counts import (
        _read_with_first_use_key,
    )
    from tldw_chatbook.UI.Console_Modules.view_workers import (
        capture_console_view_workers,
        drain_console_view_workers,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    database = CharactersRAGDB(tmp_path / "chat.db", "citation-retirement")
    repository = CitationTraceRepository(
        database,
        policy=CitationProvenanceRuntimePolicy(),
        identity_context=None,
        fingerprint_codec=None,
    )
    eligible = (("native-message", "missing-persisted-message", "Reply", "complete"),)
    signature = ("conversation", eligible)
    host, screen = App(), Screen()
    published = []
    controller = SimpleNamespace(
        _read_console_citation_counts=ChatScreen._read_console_citation_counts,
        _apply_console_citation_counts=lambda *args: published.append(args) or False,
    )
    original = ChatScreen._discover_console_citation_counts
    query = ChatScreen._read_console_citation_counts
    reader = CitationTraceRepository.get_active_trace_for_current_message
    monitor = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for owner, name in (
        (ChatScreen, "_discover_console_citation_counts"),
        (ChatScreen, "_read_console_citation_counts"),
        (CitationTraceRepository, "get_active_trace_for_current_message"),
        (base_db, "run_owned_db_call"),
        (_WorkItem, "run"),
    ):
        callback = getattr(owner, name)
        monitor._pin(callback)
        monitor.slots.append((owner, name, inspect.getattr_static(owner, name)))
    invoke_code = next(
        code
        for code in base_db.run_owned_db_call.__code__.co_consts
        if isinstance(code, CodeType) and code.co_name == "invoke"
    )
    entered, release = threading.Event(), threading.Event()
    held, errors = {}, []
    worker = drain = None

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def hold(code, _offset, _value):
        frame = ancestor = invocation = None
        try:
            frame = sys._getframe(1)
            if frame.f_locals.get("self") is not repository or held:
                return
            assert code is reader.__code__ and frame.f_globals is reader.__globals__
            ancestor = frame.f_back
            invocation = None
            while (
                ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__
            ):
                if ancestor.f_code is invoke_code:
                    invocation = ancestor
                ancestor = ancestor.f_back
            assert invocation is not None
            assert invocation.f_globals is base_db.run_owned_db_call.__globals__
            assert invocation.f_locals["database"] is database
            # TASK-33620.8: the reader runs inside the first-use key wrapper,
            # in the same owned worker call.
            assert invocation.f_locals["operation"] is _read_with_first_use_key
            assert invocation.f_locals["args"] == (query, repository, eligible)
            assert invocation.f_locals["kwargs"] == {}
            invocation = None
            assert ancestor is not None
            item = ancestor.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            connection = database._local.conn
            participant = database._maintenance_participant
            with storage._lock:
                lease = participant.connections[connection]
                assert lease in storage._live_leases and not closed(connection)
                assert lease.resource_thread is threading.current_thread()
            held.update(
                future=item.future,
                connection=connection,
                participant=participant,
                lease=lease,
            )
            entered.set()
            assert release.wait(10), "Original citation-count query was not released"
        except BaseException as error:  # noqa: BLE001 - passive observer records every failure before unconditional native release/join
            errors.append(type(error).__name__)
            entered.set()
        finally:
            del frame, ancestor, invocation

    for candidate in range(5, 0, -1):
        if candidate == sys.monitoring.DEBUGGER_ID:
            continue
        try:
            sys.monitoring.use_tool_id(candidate, "citation-count-host-native")
        except ValueError:
            continue
        tool = candidate
        break
    else:
        raise AssertionError("No local monitoring slot")
    assert sys.monitoring.get_events(tool) == 0
    assert (
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, hold)
        is None
    )
    sys.monitoring.set_local_events(
        tool, reader.__code__, sys.monitoring.events.PY_RETURN
    )
    with host._context():
        try:
            issued = original(
                controller,
                repository,
                signature,
                1,
                repository_token=("ready", 1, 1, 1),
            )
            worker = screen.run_worker(
                issued,
                group="console-citation-counts",
                exit_on_error=False,
            )
            assert await asyncio.to_thread(entered.wait, 10)
            assert held and not errors, errors
            assert worker._node is screen and worker in host.workers
            task = worker._task
            assert worker._work is issued and type(task) is asyncio.Task
            assert issued.cr_code is original.__code__ and not task.done()
            worker.cancel()
            captured = capture_console_view_workers(host)
            selected = any(
                row[0] is worker and row[1] is screen for row in captured[-1]
            )
            drain = asyncio.create_task(drain_console_view_workers(captured))
            await asyncio.wait({drain}, timeout=0.05)
            retained = not drain.done()
            for _ in range(2):
                drain.cancel()
                await asyncio.sleep(0.01)
                retained = retained and not drain.done()
            assert not held["future"].done() and not closed(held["connection"])
        finally:
            release.set()
            try:
                if held:
                    outcome = await asyncio.gather(
                        asyncio.wrap_future(held["future"]), return_exceptions=True
                    )
                outcomes = await asyncio.gather(
                    *(
                        task
                        for task in (drain, worker._task if worker else None)
                        if task is not None
                    ),
                    return_exceptions=True,
                )
            finally:
                sys.monitoring.set_local_events(tool, reader.__code__, 0)
                assert (
                    sys.monitoring.register_callback(
                        tool, sys.monitoring.events.PY_RETURN, None
                    )
                    is hold
                )
                sys.monitoring.free_tool_id(tool)
                try:
                    receipt = monitor.close()
                finally:
                    database.close()
    assert receipt["original_source_current"] and not errors, (receipt, errors)
    assert outcome == [{"native-message": 0}], outcome
    assert worker._task is task and worker._work is issued and task.done()
    assert isinstance(outcomes[0], asyncio.CancelledError), outcomes
    assert worker.state is WorkerState.CANCELLED and not published
    assert held["future"].done() and closed(held["connection"])
    with storage._lock:
        assert held["lease"] not in storage._live_leases
        assert held["connection"] not in held["participant"].connections
    assert selected, "Host drain omitted the original citation-count worker"
    assert retained, "Host drain released original native-live citation-count query"


@pytest.mark.asyncio
@private_profile_test
async def test_count_host_closes_while_key_only_thread_remains_observable(
    request, tmp_path
):
    from Tests.Chat.test_citation_key_first_use import _Keyring, _compose
    from tldw_chatbook.Chat.citation_trace_repository import (
        load_local_citation_identity_context,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.view_workers import (
        capture_console_view_workers,
        drain_console_view_workers,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    entered, release = threading.Event(), threading.Event()
    database = CharactersRAGDB(tmp_path / "key-host.db", "key-host")
    original_identity = load_local_citation_identity_context(database)

    class HeldKeyring(_Keyring):
        __module__ = "keyring.backends.macOS"

        def get_password(self, service, account):
            # The physical key target has no owned SQLite callback ancestor.
            frame = sys._getframe()
            while frame:
                assert frame.f_code.co_name != "invoke"
                frame = frame.f_back
            assert getattr(database._local, "conn", None) is None
            entered.set()
            assert release.wait(10)
            return super().get_password(service, account)

    repository = _compose(database, tmp_path, HeldKeyring())
    host, screen = App(), Screen()
    read, published = [], []
    controller = SimpleNamespace(
        _read_console_citation_counts=lambda *_: read.append(True) or {},
        _apply_console_citation_counts=lambda *args: published.append(args),
    )
    worker = drain = None
    with host._context():
        try:
            worker = screen.run_worker(
                ChatScreen._discover_console_citation_counts(
                    controller,
                    repository,
                    ("conversation", ()),
                    1,
                    repository_token=("ready", 1, 1, 1),
                ),
                group="console-citation-counts",
                exit_on_error=False,
            )
            assert await asyncio.to_thread(entered.wait, 5)
            database.add_note(
                title="Concurrent writer", content="Key has no DB custody"
            )
            worker.cancel()
            worker.cancel()
            drain = asyncio.create_task(
                drain_console_view_workers(capture_console_view_workers(host))
            )
            done, _ = await asyncio.wait({drain}, timeout=0.2)
            assert drain in done
            assert worker._task.done()
            assert repository._key_acquisition.thread.is_alive()
            assert not repository._key_acquisition.future.done()
            assert not read and not published
        finally:
            release.set()
            if repository._key_acquisition is not None:
                await asyncio.to_thread(repository._key_acquisition.thread.join, 5)
                assert not repository._key_acquisition.thread.is_alive()
            await asyncio.gather(
                *(
                    item
                    for item in (drain, worker._task if worker else None)
                    if item is not None
                ),
                return_exceptions=True,
            )
            assert load_local_citation_identity_context(database) == original_identity
            assert not read and not published
            database.close()
