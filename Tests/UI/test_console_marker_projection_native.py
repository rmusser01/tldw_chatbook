"""Original marker SQL must leave its caller's loop responsive."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import sys
import threading
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@private_profile_test
async def test_original_marker_read_leaves_shared_loop_responsive(request):
    from tldw_chatbook import config
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.UI.Console_Modules.transcript import (
        ConsoleChangeReviewProjection,
    )
    from tldw_chatbook.Workspaces.change_review_finalization import (
        ChangeReviewPublicationSignal,
    )

    database = AgentRunsDB(config.get_user_data_dir() / "marker-loop.sqlite")
    database.create_run(
        run_id="run",
        conversation_id="chat",
        agent_kind="primary",
        assistant_message_id="assistant",
    )
    database.record_change_snapshot(
        run_id="run",
        root="/project",
        baseline_sha="before",
        end_sha="after",
        files_changed=1,
        adds=2,
    )
    database.close()
    participant = _repository_participant(database)
    bridge = ConsoleAgentBridge(
        agent_runs_db=database,
        store=None,
        provider_gateway=None,
        registry=ToolCatalogRegistry(),
    )
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime._agent_bridge = bridge
    runtime._change_review_coordinator = SimpleNamespace(
        publication_signal=ChangeReviewPublicationSignal()
    )
    projection = ConsoleChangeReviewProjection(
        runtime_accessor=lambda: runtime, conversation_id_accessor=lambda: "chat"
    )
    messages = [
        ConsoleChatMessage(
            id="assistant",
            persisted_message_id="assistant",
            role=ConsoleMessageRole.ASSISTANT,
            content="reply",
        )
    ]
    callback = AgentRunsDB.list_change_review_run_anchors
    code = callback.__code__
    entered, release = threading.Event(), threading.Event()
    captured = {}
    progress = []
    stop = False
    loop = asyncio.get_running_loop()

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def hold_original_return(observed_code, _offset, result):
        frame = sys._getframe(1)
        if (
            observed_code is not code
            or captured
            or frame.f_locals.get("self") is not database
        ):
            return
        assert result == [{"id": "run", "assistant_message_id": "assistant"}]
        connection = database._thread_local.conn
        with storage._lock:
            lease = participant.connections[connection]
            assert lease in storage._live_leases
            assert lease.resource_thread is threading.current_thread()
        captured.update(
            connection=connection, lease=lease, thread=threading.current_thread()
        )
        entered.set()
        assert release.wait(15), "test did not release the original marker SELECT"

    def release_after_observation():
        if entered.wait(12):
            release.wait(0.2)
        release.set()

    async def heartbeat():
        while not stop:
            if entered.is_set() and not release.is_set():
                progress.append(loop.time())
            await asyncio.sleep(0.01)

    def snapshot_and_cleanup():
        assert threading.current_thread() is captured["thread"]
        connection = captured["connection"]
        with storage._lock:
            result = (
                closed(connection),
                connection in participant.connections,
                captured["lease"] in storage._live_leases,
            )
        if not result[0]:
            database.close()
        assert closed(connection)
        return result

    monitoring = sys.monitoring
    tool = next(value for value in range(1, 6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(tool, "marker-original-native-loop")
    releaser = threading.Thread(
        target=release_after_observation, name="marker-test-release"
    )
    ticker = asyncio.create_task(heartbeat())
    result = None
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="marker-read"
    ) as executor:
        loop.set_default_executor(executor)
        try:
            monitoring.register_callback(
                tool, monitoring.events.PY_RETURN, hold_original_return
            )
            monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
            releaser.start()
            # Compare the original synchronous route with the new caller's
            # preparation seam; neither route substitutes its actual reader.
            prepare = getattr(projection, "prepare", None)
            if prepare is not None:
                assert await prepare()
            result = projection.project(messages)
        finally:
            release.set()
            releaser.join(2)
            stop = True
            await ticker
            if captured:
                after_product = (
                    snapshot_and_cleanup()
                    if captured["thread"] is threading.current_thread()
                    else await loop.run_in_executor(executor, snapshot_and_cleanup)
                )
            monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            monitoring.free_tool_id(tool)
            database.close()
    assert not releaser.is_alive()
    assert (
        callback is AgentRunsDB.list_change_review_run_anchors
        and callback.__code__ is code
    )
    assert result[0] is messages[0]
    assert [message.change_review_run_id for message in result[1:]] == ["run"]
    assert captured and not runtime._preparation_reads
    assert (
        progress
    ), "original marker SELECT blocked its caller loop while native work was held"
    assert after_product == (
        True,
        False,
        False,
    ), "marker worker did not retire its owned connection"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    [
        "cancel",
        "repeated_cancel",
        "conversation",
        "revision",
        "dispose",
        "borrowed",
        "coalesced",
        "foreign_cache",
        "database",
        "reader_defaults",
    ],
)
@private_profile_test
async def test_marker_read_keeps_native_custody_and_current_publication(
    request, action
):
    from tldw_chatbook import config
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.UI.Console_Modules.transcript import (
        ConsoleChangeReviewProjection,
    )
    from tldw_chatbook.Workspaces.change_review_finalization import (
        ChangeReviewPublicationSignal,
    )

    database = AgentRunsDB(config.get_user_data_dir() / "marker-custody.sqlite")
    database.create_run(
        run_id="run",
        conversation_id="chat",
        agent_kind="primary",
        assistant_message_id="assistant",
    )
    database.record_change_snapshot(
        run_id="run",
        root="/project",
        baseline_sha="before",
        end_sha="after",
        files_changed=1,
        adds=2,
    )
    database.close()
    participant = _repository_participant(database)
    bridge = ConsoleAgentBridge(
        agent_runs_db=database,
        store=None,
        provider_gateway=None,
        registry=ToolCatalogRegistry(),
    )
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime._agent_bridge = bridge
    signal = ChangeReviewPublicationSignal()
    runtime._change_review_coordinator = SimpleNamespace(publication_signal=signal)
    selected = ["chat"]
    projection = ConsoleChangeReviewProjection(
        runtime_accessor=lambda: runtime, conversation_id_accessor=lambda: selected[0]
    )
    code = AgentRunsDB.list_change_review_run_anchors.__code__
    entered, release = threading.Event(), threading.Event()
    captured = {}
    calls = []
    loop = asyncio.get_running_loop()
    task = other = dispose = None
    original_reader = AgentRunsDB.change_snapshots_for_conversation
    original_defaults = original_reader.__defaults__
    replacement_db = None

    def hold_return(observed_code, _offset, result):
        frame = sys._getframe(1)
        if observed_code is not code or frame.f_locals.get("self") is not database:
            return
        calls.append(result)
        if captured:
            return
        connection = database._thread_local.conn
        with storage._lock:
            lease = participant.connections[connection]
            assert (
                lease in storage._live_leases
                and lease.resource_thread is threading.current_thread()
            )
        captured.update(
            connection=connection, lease=lease, thread=threading.current_thread()
        )
        entered.set()
        assert release.wait(15)
        if action == "foreign_cache":
            foreign = database._get_connection()
            foreign.execute("BEGIN")
            database._thread_local.conn = foreign
            captured["foreign"] = foreign

    async def held():
        while not entered.is_set():
            if task.done():
                task.result()
                raise AssertionError("original marker query was not reached")
            await asyncio.sleep(0.01)

    def after_read():
        assert threading.current_thread() is captured["thread"]
        connection = captured["connection"]
        try:
            transaction = sqlite3.Connection.in_transaction.__get__(connection)
            closed = False
        except sqlite3.ProgrammingError:
            closed, transaction = True, None
        with storage._lock:
            result = (
                closed,
                transaction,
                connection in participant.connections,
                captured["lease"] in storage._live_leases,
            )
        if action == "foreign_cache":
            foreign = captured["foreign"]
            assert database._thread_local.conn is foreign
            assert sqlite3.Connection.in_transaction.__get__(foreign)
            with storage._lock:
                assert participant.connections[foreign] in storage._live_leases
            foreign.rollback()
            database.close()
        if not closed:
            connection.rollback()
            database.close()
        return result

    monitoring = sys.monitoring
    tool = next(value for value in range(1, 6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(tool, "marker-original-custody")
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="marker-custody"
    ) as executor:
        loop.set_default_executor(executor)
        try:
            if action == "borrowed":

                def borrow():
                    connection = database._held_connection()
                    connection.execute("BEGIN")
                    return connection

                borrowed = await loop.run_in_executor(executor, borrow)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, hold_return)
            monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
            task = asyncio.create_task(projection.prepare())
            await asyncio.wait_for(held(), 12)
            assert runtime._preparation_reads and projection._preparation_reads
            if action in {"cancel", "repeated_cancel"}:
                task.cancel()
                await asyncio.sleep(0)
                if action == "repeated_cancel":
                    task.cancel()
                assert not (await asyncio.wait({task}, timeout=0.05))[0]
            elif action == "conversation":
                selected[0] = "other-chat"
            elif action == "revision":
                signal.anchor_published()
            elif action == "database":
                replacement_db = AgentRunsDB(":memory:")
                bridge._db = replacement_db
            elif action == "reader_defaults":
                original_reader.__defaults__ = (None,)
            elif action == "dispose":
                dispose = asyncio.create_task(runtime.dispose())
                await asyncio.sleep(0.05)
                assert runtime._disposed and not dispose.done()
            elif action == "coalesced":
                other = asyncio.create_task(projection.prepare())
                await asyncio.sleep(0.05)
                assert len(runtime._preparation_reads) == 1 and not other.done()
            elif action == "borrowed":
                assert captured["connection"] is borrowed
            assert projection._marker_blocks == []
        finally:
            release.set()
            outcomes = await asyncio.gather(
                *(value for value in (task, other, dispose) if value is not None),
                return_exceptions=True,
            )
            after = (
                await loop.run_in_executor(executor, after_read) if captured else None
            )
            original_reader.__defaults__ = original_defaults
            if replacement_db is not None:
                replacement_db.close()
            monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            monitoring.free_tool_id(tool)
            database.close()
    assert not runtime._preparation_reads and not projection._preparation_reads
    assert after == (
        (False, True, True, True)
        if action == "borrowed"
        else (True, None, False, False)
    )
    assert len(calls) == 1
    if action in {"cancel", "repeated_cancel"}:
        assert task.cancelled() and projection._key is None
    elif action in {
        "conversation",
        "revision",
        "dispose",
        "foreign_cache",
        "database",
        "reader_defaults",
    }:
        assert outcomes[0] is False and projection._key is None
        if dispose is not None:
            assert dispose.exception() is None
    else:
        assert all(outcome is True for outcome in outcomes)
        assert len(projection._marker_blocks) == 1
