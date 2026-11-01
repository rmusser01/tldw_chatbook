"""Actual Screen caller ownership for finite Change Review projection."""

import asyncio
import sys
import threading

import pytest

from Tests.UI.test_console_hook_review_send_freeze import _until
from Tests.UI.test_console_received_intent_feedback import _received_console_case

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("closing", [False, True])
async def test_mounted_marker_refresh_rechecks_screen_after_projection_lock(
    monkeypatch, closing, record_property
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    async with _received_console_case(
        monkeypatch, f"marker-lock-{closing}", durable=True
    ) as case:
        screen = case.console
        bridge = screen._ensure_console_agent_bridge()
        database = bridge._db
        assert type(database) is AgentRunsDB and not database.is_memory_db
        assert case.runtime.change_review_coordinator is not None
        run_id = f"marker-run-{closing}"
        message = case.store.append_message(
            case.session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="reply",
            persist=True,
        )
        conversation_id = case.session.persisted_conversation_id
        assert conversation_id and message.persisted_message_id
        database.create_run(
            run_id=run_id,
            conversation_id=conversation_id,
            agent_kind="primary",
            assistant_message_id=message.persisted_message_id,
        )
        database.record_change_snapshot(
            run_id=run_id,
            root="/project",
            baseline_sha="before",
            end_sha="after",
            files_changed=1,
            adds=2,
        )
        projection = screen._change_review_projection
        code = AgentRunsDB.list_change_review_run_anchors.__code__
        observations = []

        def entered(observed_code, _offset):
            frame = sys._getframe(1)
            if observed_code is code and frame.f_locals.get("self") is database:
                observations.append(threading.current_thread())

        monitoring = sys.monitoring
        tool = next(
            value for value in range(1, 6) if monitoring.get_tool(value) is None
        )
        monitoring.use_tool_id(tool, "mounted-marker-lock")
        task = None
        was_closing = screen._closing
        await projection._prepare_lock.acquire()
        try:
            monitoring.register_callback(tool, monitoring.events.PY_START, entered)
            monitoring.set_local_events(tool, code, monitoring.events.PY_START)
            task = asyncio.create_task(screen._sync_native_console_transcript())
            assert await _until(lambda: bool(projection._prepare_lock._waiters), 5)
            assert not observations
            screen._closing = closing
            projection._prepare_lock.release()
            await asyncio.wait_for(task, 10)
            if closing:
                assert not observations, (
                    "queued closed Screen issued a new native marker read"
                )
            else:
                assert observations and all(
                    thread is not threading.current_thread() for thread in observations
                )
                transcript = screen.query_one(
                    "#console-native-transcript", ConsoleTranscript
                )
                assert any(
                    row.change_review_run_id == run_id for row in transcript._messages
                )
                assert not projection._preparation_reads
                assert not any(
                    read.creator is projection
                    for read in case.runtime._preparation_reads
                )
            record_property(
                "other_runtime_read_callbacks",
                [
                    getattr(read.callback, "__qualname__", type(read.callback).__name__)
                    for read in case.runtime._preparation_reads
                    if read.creator is not projection
                ],
            )
            record_property(
                "original_marker_read_threads", [thread.name for thread in observations]
            )
        finally:
            screen._closing = was_closing
            if projection._prepare_lock.locked():
                projection._prepare_lock.release()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_START, None)
            monitoring.free_tool_id(tool)


@pytest.mark.parametrize("closure", ["_closing", "_closed", "shutdown"])
async def test_mounted_marker_held_read_retires_without_closed_publication(
    monkeypatch, closure
):
    import sqlite3

    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    async with _received_console_case(
        monkeypatch, f"marker-held-{closure}", durable=True
    ) as case:
        screen = case.console
        database = screen._ensure_console_agent_bridge()._db
        message = case.store.append_message(
            case.session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="persisted reply",
            persist=True,
        )
        run_id = f"marker-held-{closure}"
        database.create_run(
            run_id=run_id,
            conversation_id=case.session.persisted_conversation_id,
            agent_kind="primary",
            assistant_message_id=message.persisted_message_id,
        )
        database.record_change_snapshot(
            run_id=run_id,
            root="/project",
            baseline_sha="before",
            end_sha="after",
            files_changed=1,
            adds=2,
        )
        projection = screen._change_review_projection
        participant = _repository_participant(database)
        code = AgentRunsDB.list_change_review_run_anchors.__code__
        entered, release = threading.Event(), threading.Event()
        captured = {}

        def held(observed_code, _offset, result):
            frame = sys._getframe(1)
            if (
                observed_code is not code
                or frame.f_locals.get("self") is not database
                or captured
            ):
                return
            assert result == [
                {"id": run_id, "assistant_message_id": message.persisted_message_id}
            ]
            connection = database._thread_local.conn
            with storage._lock:
                lease = participant.connections[connection]
                assert lease in storage._live_leases
            captured.update(connection=connection, lease=lease)
            entered.set()
            assert release.wait(15)

        monitoring = sys.monitoring
        tool = next(
            value for value in range(1, 6) if monitoring.get_tool(value) is None
        )
        monitoring.use_tool_id(tool, "mounted-marker-close-held")
        task = shutdown = None
        app = screen.app_instance
        old_closing, old_closed = screen._closing, screen._closed
        old_shutdown = getattr(app, "_console_runtime_shutdown_task", None)
        try:
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, held)
            monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
            task = asyncio.create_task(screen._sync_native_console_transcript())
            assert await _until(entered.is_set, 10)
            assert projection._preparation_reads and case.runtime._preparation_reads
            if closure == "shutdown":
                shutdown = asyncio.create_task(asyncio.Event().wait())
                app._console_runtime_shutdown_task = shutdown
            else:
                setattr(screen, closure, True)
            release.set()
            await asyncio.wait_for(task, 10)
            assert projection._key is None and projection._marker_blocks == []
            assert not projection._preparation_reads and not any(
                read.creator is projection for read in case.runtime._preparation_reads
            )
            with pytest.raises(sqlite3.ProgrammingError):
                sqlite3.Connection.in_transaction.__get__(captured["connection"])
            with storage._lock:
                assert captured["connection"] not in participant.connections
                assert captured["lease"] not in storage._live_leases
            transcript = screen.query_one(
                "#console-native-transcript", ConsoleTranscript
            )
            assert not any(
                row.change_review_run_id == run_id for row in transcript._messages
            )
        finally:
            release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            screen._closing, screen._closed = old_closing, old_closed
            app._console_runtime_shutdown_task = old_shutdown
            if shutdown is not None:
                shutdown.cancel()
                await asyncio.gather(shutdown, return_exceptions=True)
            monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            monitoring.free_tool_id(tool)
