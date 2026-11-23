"""Quit must revoke queued browser admission before yielding to cleanup."""

import asyncio
import json
import sqlite3
import sys
import threading

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["queued", "native", "late", "detached", "stay"])
@pytest.mark.parametrize("route", ["cache", "flat"])
@private_profile_test
async def test_approved_quit_cancels_browser_at_terminal_fence(
    request, tmp_path, phase, route
):
    from loguru import logger
    from textual.app import App
    from textual.worker import Worker

    from Tests.UI.test_console_workspace_controller import _workspace_controller
    from tldw_chatbook.app_lifecycle import LifecycleMixin
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules import console_browser_read

    class QuitHost(App):
        _confirm_console_runtime_quit = LifecycleMixin._confirm_console_runtime_quit
        _confirm_workflow_session_quit = LifecycleMixin._confirm_workflow_session_quit

        def _close_boot_worker_gate(self, reason):
            assert reason == "quit"

        async def _run_approved_quit_cleanup(self):
            assert self.console_runtime._disposed and self._shutting_down
            if phase == "detached":
                await LifecycleMixin._shutdown_console_runtime(self)
                assert self.console_runtime is None
            if phase in {"late", "detached"}:
                # The screen can remain live after runtime disposal/detachment.
                issue()
            if phase not in {"late", "detached"}:
                facts["cancelled_at_fence"] = worker._task.cancelling() > 0
            if other is not None:
                assert other._task.cancelling() == 0, (
                    "Quit cancelled unrelated queued work early"
                )
            await asyncio.sleep(0.05)
            if other is not None:
                assert facts["other_started"]
            if phase == "native":
                assert not worker._task.done()
                (read,) = self.console_runtime._preparation_reads
                facts["read"] = read
                assert not read.retired.done() and not read._producer.done()
                with storage._lock:
                    assert (
                        database._maintenance_participant.connections[
                            facts["connection"]
                        ]
                        in storage._live_leases
                    )
                release.set()
            await asyncio.gather(
                *(
                    item._task
                    for item in tuple(self.workers)
                    if item.group
                    in {
                        "console-persisted-browser-cache",
                        "console-flat-conversation-search",
                    }
                ),
                return_exceptions=True,
            )

    database = CharactersRAGDB(tmp_path / "quit.sqlite", client_id="quit-browser")
    service = ChatConversationService(database)
    service.create_conversation(title="Global", scope_type="global")
    service.create_conversation(
        title="Default", scope_type="workspace", workspace_id="workspace-default"
    )
    database.close_connection()
    host = QuitHost()
    host.local_chat_conversation_service = service
    host.chat_conversation_scope_service = None
    runtime = host.console_runtime = ConsoleRuntime(app=host)
    host._quit_in_progress = True
    host._shutting_down = False
    host._console_runtime_shutdown_task = None
    host._plugin_service = None
    controller = _workspace_controller(screen=host, app_instance=host)

    def issue():
        if route == "cache":
            controller._sync_persisted_console_browser_rows()
        else:
            controller._console_conversation_browser_query = "Global"
            controller._start_console_conversation_browser_search(
                "Global", controller._console_conversation_browser_search_token
            )

    facts, errors = {}, []
    entered, release = threading.Event(), threading.Event()
    code = ChatConversationService.list_conversations.__code__
    starts = []

    def started(observed_code, _offset):
        assert observed_code is code
        starts.append(True)

    def returned(observed_code, _offset, result):
        if phase != "native" or entered.is_set():
            return
        frame = sys._getframe(1)
        if observed_code is not code or frame.f_locals.get("self") is not service:
            return
        assert result["items"] and frame.f_locals["scope_type"] == "global"
        facts["connection"] = database._local.conn
        # Holding even an observational frame across a native wait changes
        # SQLite ownership; retain only the selected connection reference.
        del frame
        entered.set()
        assert release.wait(10)

    changed_code = console_browser_read._StockBrowserRead._changed.__code__

    def changed(observed_code, _offset):
        assert observed_code is changed_code
        frame = sys._getframe(2)  # selected_start delegates this callback.
        assert frame.f_code is changed_code
        read = frame.f_locals["self"]
        caller_is_loop_check = (
            frame.f_back.f_code
            is console_browser_read._StockBrowserRead.require_loop_current.__code__
        )
        del frame
        if read.controller is not controller:
            return
        facts.setdefault("source_refusals", []).append(
            {
                "caller_is_loop_check": caller_is_loop_check,
                "read_cancelled": read.cancelled,
                "runtime_is_original": read._runtime is runtime,
                "runtime_disposed": runtime._disposed,
                "issuer_new_cancellation": read._issuer.cancelling()
                > read._initial_cancellation_count,
                "app_same": read.controller.app_instance is read.app,
                "local_service_same": read.app.local_chat_conversation_service
                is read.local_service,
                "captured_functions_current": console_browser_read._sources_current(),
            }
        )

    monitor = sys.monitoring
    tool = next(value for value in range(1, 6) if monitor.get_tool(value) is None)
    monitor.use_tool_id(tool, "quit-browser-fence")

    def selected_start(observed_code, offset):
        if observed_code is changed_code:
            changed(observed_code, offset)
        else:
            started(observed_code, offset)

    monitor.register_callback(tool, monitor.events.PY_START, selected_start)
    monitor.set_local_events(tool, changed_code, monitor.events.PY_START)
    monitor.register_callback(tool, monitor.events.PY_RETURN, returned)
    monitor.set_local_events(
        tool, code, monitor.events.PY_START | monitor.events.PY_RETURN
    )
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    worker = other = None

    async def readiness():
        facts["other_started"] = True
        try:
            await asyncio.Event().wait()
        finally:
            facts["other_retired"] = True

    other_work = None
    try:
        async with host.run_test():
            if phase == "stay":
                host.screen.confirm_quit = lambda: False
            if phase not in {"late", "detached"}:
                issue()
                (worker,) = tuple(host.workers)
                assert type(worker) is Worker and worker._node is host
                assert not worker._task.done() and worker._task.cancelling() == 0
            if phase == "queued":
                other_work = readiness()
                other = host.run_worker(other_work, group="console-readiness-config")
            if phase == "native":
                assert await asyncio.to_thread(entered.wait, 10)
            await LifecycleMixin._confirm_and_quit(host)
            if phase == "stay":
                assert not runtime._disposed and not host._shutting_down
                await worker.wait()
                assert not worker.is_cancelled and len(starts) == 2
            elif phase not in {"late", "detached"}:
                assert facts["cancelled_at_fence"], (
                    "Quit yielded with an uncancelled browser"
                )
                assert worker._task.done() and worker.is_cancelled
            else:
                assert not tuple(host.workers), (
                    "Disposed runtime admitted a new browser worker"
                )
            assert not errors, errors
            if phase not in {"native", "stay"}:
                assert not starts, "Browser queried after terminal revocation"
            elif phase == "native":
                read = facts["read"]
                assert read.retired.done() and read._producer.done()
                assert not runtime._preparation_reads
                with pytest.raises(sqlite3.ProgrammingError):
                    sqlite3.Connection.in_transaction.__get__(facts["connection"])
            await LifecycleMixin._shutdown_console_runtime(host)
            assert host.console_runtime is None
            if other is not None:
                assert other.is_cancelled and other._task.done()
                assert facts["other_retired"]
    finally:
        release.set()
        if other is not None:
            other.cancel()
            await asyncio.gather(other._task, return_exceptions=True)
            if other_work.cr_frame is not None:
                other_work.close()
        if worker is not None:
            await asyncio.gather(worker._task, return_exceptions=True)
        monitor.set_local_events(tool, code, 0)
        monitor.set_local_events(tool, changed_code, 0)
        monitor.register_callback(tool, monitor.events.PY_START, None)
        monitor.register_callback(tool, monitor.events.PY_RETURN, None)
        monitor.free_tool_id(tool)
        logger.remove(sink)
        await runtime.dispose()
        database.close_connection()
        print(
            json.dumps(
                {
                    "phase": phase,
                    "route": route,
                    "cancelled_at_fence": facts.get("cancelled_at_fence"),
                    "source_refusals": facts.get("source_refusals", []),
                    "error_count": len(errors),
                },
                sort_keys=True,
            )
        )


def _close_console(agent, route):
    """Close the way an exit does: quit approval latches the runtime, and
    every exit route marks the view tearing down before its drain."""
    if route == "latched":
        agent._screen._console_runtime()._disposed = True
    else:
        agent._screen._console_chat_tearing_down = True


def _view_agent_runs_reader(agent, bridge, reader, conversation_id="conv"):
    """One render's call into the view's AgentRuns reader, and its empty value."""
    from types import SimpleNamespace

    if reader == "counts":
        rows = (SimpleNamespace(conversation_id=conversation_id),)
        return (lambda: agent._console_subagent_counts_for_rows(bridge, rows)), {}
    agent._current_rail_conversation_id = lambda: conversation_id
    return agent._console_agent_fleet_rows, ()


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("route", ["latched", "tearing_down"])
@pytest.mark.parametrize("reader", ["counts", "history"])
async def test_disposing_runtime_admits_no_new_view_agent_runs_read(
    tmp_path, monkeypatch, reader, route
):
    """TASK-42426: once the Console closes, a render starts no AgentRuns read.

    Quit drains (cancels) the view's reads before it disposes the runtime,
    and a cancelled read keeps no fresh refresh time. The view stays live
    while the runtime retires its AgentRuns file, so a render would start a
    new read there and race that retirement.
    """
    from Tests.UI.test_console_refresh_read_batching import (
        _agent,
        _historical_bridge,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    database = AgentRunsDB(tmp_path / "quit-reads.sqlite", "quit-reads")
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()
    tasks = []
    try:
        bridge, _child = _historical_bridge(database)
        agent, tasks = _agent(bridge)
        reads = []
        render, shown = _view_agent_runs_reader(agent, bridge, reader)
        if reader == "counts":
            name, owner = "count_subagents_by_conversation", database
        else:
            name, owner = "_derive_historical_snapshot", bridge
        original = getattr(owner, name)

        def held(*args, **kwargs):
            reads.append(threading.get_ident())
            entered.set()
            try:
                assert release.wait(5)
                return original(*args, **kwargs)
            finally:
                retired.set()

        monkeypatch.setattr(owner, name, held)
        assert render() == shown
        assert await asyncio.to_thread(entered.wait, 5)
        # Quit's view drain cancels the read; it retires on its own thread.
        tasks[0].cancel()
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert await asyncio.to_thread(retired.wait, 5)
        _close_console(agent, route)
        for _ in range(3):
            assert render() == shown
        assert len(tasks) == 1 and len(reads) == 1, (
            "A render admitted a new AgentRuns read while the Console closed"
        )
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("reader", ["counts", "history"])
async def test_disposing_runtime_keeps_showing_published_agent_runs_values(
    tmp_path, monkeypatch, reader
):
    """TASK-42426: a closing Console shows the values a read already published.

    Only the conversation that read them gets them back: a render for another
    conversation shows nothing rather than borrowed rows. No render reads.
    """
    from Tests.UI.test_console_refresh_read_batching import (
        _agent,
        _finish,
        _historical_bridge,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.UI.Screens import chat_screen

    database = AgentRunsDB(tmp_path / "quit-values.sqlite", "quit-values")
    tasks = []
    try:
        bridge, child = _historical_bridge(database)
        agent, tasks = _agent(bridge)
        # Every render would read again if nothing stopped it.
        monkeypatch.setattr(chat_screen, "CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS", 0)
        render, empty = _view_agent_runs_reader(agent, bridge, reader)
        assert render() == empty
        await _finish(tasks)
        def shown():
            value = render()
            return value if reader == "counts" else [row.row_id for row in value]

        published = shown()
        assert published == ({"conv": 1} if reader == "counts" else [child])
        await _finish(tasks)
        issued = len(tasks)
        _close_console(agent, "latched")
        for _ in range(3):
            assert shown() == published
        other, _empty = _view_agent_runs_reader(agent, bridge, reader, "other-conv")
        assert other() == empty, "A closing Console showed another chat's rows"
        assert len(tasks) == issued, "A render read AgentRuns while the Console closed"
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()
