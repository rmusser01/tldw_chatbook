"""Original retrieval-display workers retain their native database callbacks."""

import asyncio
import inspect
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
import sqlite3
import sys
import threading
from types import CodeType, SimpleNamespace

import pytest
from textual.app import App
from textual.screen import Screen
from textual.worker import WorkerState

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["scope", "world-books", "dictionaries"])
@private_profile_test
async def test_original_retrieval_worker_retires_before_host_drain(
    request, tmp_path, kind
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from Tests.UI.test_console_retrieval_controller import _controller
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Character_Chat import world_info_resolver
    from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
        LocalChatDictionaryService,
    )
    from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
        ChatDictionaryScopeService,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events
    from tldw_chatbook.Backup_Recovery.dictionary_source_job import _DictionaryJob
    from tldw_chatbook.UI.Console_Modules import retrieval
    from tldw_chatbook.UI.Console_Modules.view_workers import (
        capture_console_view_workers,
        drain_console_view_workers,
    )
    from tldw_chatbook.DB import base_db

    database = CharactersRAGDB(tmp_path / "chat.db", "retrieval-retirement")
    conversation_id = database.add_conversation({"title": "Retrieval lifetime"})
    controller, _state = _controller()
    controller.app_instance.chachanotes_db = database
    controller._current_conversation_id = lambda: conversation_id
    host, screen = App(), Screen()
    published = []
    controller._request_control_bar_sync = lambda: published.append(True)
    local = service = None
    session = ConsoleChatStore().ensure_session()
    session.persisted_conversation_id = conversation_id
    later_scope_reads = []
    if kind == "scope":
        from tldw_chatbook.Chat.rag_scope import (
            RagScope,
            ScopeItem,
            write_conversation_scope,
        )

        write_conversation_scope(
            database,
            conversation_id,
            RagScope(
                items=(ScopeItem("note", "missing-note"),), updated_at="before-cancel"
            ),
        )
        session.workspace_id = "custom-workspace"
        controller.app_instance.workspace_registry_service = SimpleNamespace(
            get_workspace_scope=lambda _workspace: later_scope_reads.append(True),
            db=None,
        )
        reader = chat_rag_events._read_cached_conversation_scope_sync
        original = (
            retrieval.ConsoleRetrievalController._resolve_console_effective_scope_state
        )
        args = (session,)
    elif kind == "world-books":
        reader = world_info_resolver.summarize_active_world_books
        original = (
            retrieval.ConsoleRetrievalController.refresh_active_world_books_summary
        )
        args = ()
    else:
        local = LocalChatDictionaryService(database)
        service = ChatDictionaryScopeService(local_service=local, server_service=None)
        controller._dictionary_scope_service = lambda: service
        reader = inspect.unwrap(
            LocalChatDictionaryService.summarize_active_dictionaries
        )
        original = (
            retrieval.ConsoleRetrievalController.refresh_active_dictionaries_summary
        )
        args = ()
    query = reader
    monitor = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for subject, name in (
        (retrieval.ConsoleRetrievalController, original.__name__),
        (retrieval, "resolve_scope_for_session"),
        (chat_rag_events, "_await_scope_read"),
        (chat_rag_events, "_resolve_scope_with_current_ids"),
        (retrieval, "_run_dictionary_summary_off_thread"),
        (retrieval, "_await_finite_display"),
        (LocalChatDictionaryService, "summarize_active_dictionaries"),
        (_DictionaryJob, "_work"),
        (base_db, "run_owned_db_call"),
        (_WorkItem, "run"),
    ):
        monitor._pin(getattr(subject, name))
        monitor.slots.append((subject, name, inspect.getattr_static(subject, name)))
    monitor._pin(reader)
    invoke_code = next(
        code
        for code in base_db.run_owned_db_call.__code__.co_consts
        if isinstance(code, CodeType) and code.co_name == "invoke"
    )
    entered, release = threading.Event(), threading.Event()
    held, errors = {}, []
    worker = drain = None
    outer = {}
    retired = False

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def observe_outer(code, _offset):
        frame = ancestor = None
        try:
            frame = sys._getframe(1)
            if frame.f_locals.get("service") is not service:
                return
            assert code is retrieval._run_dictionary_summary_off_thread.__code__
            assert frame.f_locals["conversation_id"] == conversation_id
            ancestor = frame.f_back
            while (
                ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__
            ):
                ancestor = ancestor.f_back
            assert ancestor is not None
            item = ancestor.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            outer["future"] = item.future
        finally:
            del frame, ancestor

    def hold(code, _offset, _value):
        frame = ancestor = invocation = None
        try:
            frame = sys._getframe(1)
            selected = (
                frame.f_locals.get("self") is local
                if kind == "dictionaries"
                else frame.f_locals.get("db") is database
            )
            if not selected or held:
                return
            assert code is reader.__code__ and frame.f_globals is reader.__globals__
            ancestor = frame.f_back
            expected = (
                _DictionaryJob._work.__code__ if kind == "dictionaries" else invoke_code
            )
            while (
                ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__
            ):
                if ancestor.f_code is expected:
                    invocation = ancestor
                ancestor = ancestor.f_back
            assert invocation is not None and ancestor is not None
            if kind == "dictionaries":
                job = invocation.f_locals["self"]
                assert (
                    type(job) is _DictionaryJob
                    and job._source is local
                    and job._db is database
                )
                assert (
                    job._function
                    is LocalChatDictionaryService.summarize_active_dictionaries
                )
                assert outer and not outer["future"].done()
            else:
                assert invocation.f_globals is base_db.run_owned_db_call.__globals__
                assert invocation.f_locals["database"] is database
                assert invocation.f_locals["operation"] is query
                assert invocation.f_locals["args"] == (
                    (database, conversation_id)
                    if kind == "scope"
                    else (database, conversation_id, None)
                )
                assert invocation.f_locals["kwargs"] == {}
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
            assert release.wait(10), "Original retrieval query was not released"
        except BaseException as error:
            errors.append(type(error).__name__)
            entered.set()
        finally:
            del frame, ancestor, invocation

    for candidate in range(5, 0, -1):
        if candidate == sys.monitoring.DEBUGGER_ID:
            continue
        try:
            sys.monitoring.use_tool_id(candidate, "retrieval-host-native")
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
    if kind == "dictionaries":
        assert (
            sys.monitoring.register_callback(
                tool, sys.monitoring.events.PY_START, observe_outer
            )
            is None
        )
        sys.monitoring.set_local_events(
            tool,
            retrieval._run_dictionary_summary_off_thread.__code__,
            sys.monitoring.events.PY_START,
        )
    with host._context():
        try:
            issued = original(controller, *args)
            worker = screen.run_worker(
                issued,
                group="console-sync",
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
                if outer:
                    await asyncio.wrap_future(outer["future"])
                outcomes = await asyncio.gather(
                    *(
                        task
                        for task in (drain, worker._task if worker else None)
                        if task is not None
                    ),
                    return_exceptions=True,
                )
                if held:
                    with storage._lock:
                        retired = (
                            closed(held["connection"])
                            and held["lease"] not in storage._live_leases
                            and held["connection"]
                            not in held["participant"].connections
                        )
            finally:
                if kind == "dictionaries":
                    sys.monitoring.set_local_events(
                        tool, retrieval._run_dictionary_summary_off_thread.__code__, 0
                    )
                    assert (
                        sys.monitoring.register_callback(
                            tool, sys.monitoring.events.PY_START, None
                        )
                        is observe_outer
                    )
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
    assert all(not isinstance(value, BaseException) for value in outcome), outcome
    assert worker._task is task and worker._work is issued and task.done()
    assert worker.state is WorkerState.CANCELLED and not published
    assert (
        retired
    ), "Original retrieval callback left its native connection or lease open"
    assert controller._active_dictionaries_summary is None
    assert controller._active_world_books_summary is None
    assert controller._console_effective_scope_cache == {}
    assert not later_scope_reads, "Cancellation started a later custom workspace read"
    assert getattr(controller.app_instance, "_console_rag_scope_cache", None) is None
    assert selected, "Host drain omitted the original retrieval worker"
    assert retained, "Host drain released original native-live retrieval callback"
    assert isinstance(outcomes[0], asyncio.CancelledError), outcomes


@pytest.mark.asyncio
@pytest.mark.parametrize("display", [False, True], ids=["default", "display"])
@private_profile_test
async def test_custom_workspace_scope_preserves_direct_cancellation(
    request, tmp_path, display
):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        resolve_scope_for_session,
    )

    database = CharactersRAGDB(tmp_path / "chat.db", "custom-scope-cancel")
    entered, release = threading.Event(), threading.Event()
    callback_future = []

    def read(_workspace_id):
        frame = sys._getframe()
        try:
            while frame.f_code is not _WorkItem.run.__code__:
                frame = frame.f_back
            callback_future.append(frame.f_locals["self"].future)
        finally:
            del frame
        entered.set()
        assert release.wait(10)
        return None

    app = SimpleNamespace(
        chachanotes_db=database,
        workspace_registry_service=SimpleNamespace(
            db=database, get_workspace_scope=read
        ),
    )
    session = SimpleNamespace(persisted_conversation_id=None, workspace_id="custom")
    task = asyncio.create_task(
        resolve_scope_for_session(app, session, retain_display_reads=display)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.wait({task}, timeout=0.1)
        assert (
            task.done() and task.cancelled()
        ), "Custom scope read acquired stock retention semantics"
        assert not callback_future[0].done()
    finally:
        release.set()
        if callback_future:
            await asyncio.wrap_future(callback_future[0])
        await asyncio.gather(task, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("display", [False, True], ids=["default", "display"])
@pytest.mark.parametrize("cached", [False, True], ids=["fresh", "cached"])
async def test_custom_scope_getter_failure_preserves_existing_error_boundary(
    display, cached
):
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        resolve_scope_for_session,
    )

    accesses = []

    class CustomRegistry:
        db = None

        @property
        def get_workspace_scope(self):
            accesses.append(True)
            raise LookupError("custom getter unavailable")

    app = SimpleNamespace(
        chachanotes_db=None, workspace_registry_service=CustomRegistry()
    )
    session = SimpleNamespace(persisted_conversation_id=None, workspace_id="custom")
    result = await resolve_scope_for_session(
        app, session, use_cache=cached, retain_display_reads=display
    )
    assert result.effective.state == "empty"
    assert result.effective.cause == "workspace-scope-unavailable"
    assert len(accesses) == int(cached)
