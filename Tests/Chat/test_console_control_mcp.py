"""Actual advertised builtin invocation and its original worker custody."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from contextlib import contextmanager, ExitStack
from http.server import ThreadingHTTPServer

import pytest

from Tests.Chat.test_console_control_trace import _observe, _prepare
from Tests.Chat.test_console_domain_boundary import domain_case as _domain_case
from Tests.Chat.test_console_first_send_atomicity import _until
from Tests.Chat.test_console_refused_stream_retry import _Handler, _MODEL

domain_case = _domain_case
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.loopback_network,
]


class _CharacterHandler(_Handler):
    def do_POST(self):  # noqa: N802
        from tldw_chatbook.MCP.tool_naming import llm_tool_name

        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.payloads.append(payload)
        name = llm_tool_name(
            "builtin:tldw_chatbook",
            getattr(self.server, "tool_name", "list_characters"),
        )
        if len(self.server.payloads) == 2:
            self.server.tool_request.set()
            assert self.server.tool_response.wait(10), "Tool response was not released"
        elif len(self.server.payloads) == 3:
            assert self.server.terminal_response.wait(
                10
            ), "Terminal response was not released"
        message = {"role": "assistant", "content": "Character list received."}
        if len(self.server.payloads) in {1, 2}:
            message["content"] = None
            message["tool_calls"] = [
                {
                    "id": "character-list",
                    "type": "function",
                    "function": {
                        "name": "load_tools"
                        if len(self.server.payloads) == 1
                        else name,
                        "arguments": json.dumps({"ids": [name]})
                        if len(self.server.payloads) == 1
                        else json.dumps(getattr(self.server, "tool_arguments", {})),
                    },
                }
            ]
        body = json.dumps(
            {
                "id": "character-call",
                "object": "chat.completion",
                "model": _MODEL,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ).encode()
        self._send(200, "application/json", body)


@contextmanager
def _character_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CharacterHandler)
    server.payloads = []
    server.tool_request = threading.Event()
    server.tool_response = threading.Event()
    server.terminal_response = threading.Event()
    server.terminal_response.set()
    server.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.tool_response.set()
        server.terminal_response.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def _assert_retained_trace(case, source):
    with case.store._provider_trace_settlement_lock:
        failed = dict(case.store._provider_trace_settlement_failed_work)
        assert failed
        assert set(failed) <= case.store._provider_trace_settlement_owned_call_ids
        assert not case.store._provider_trace_settlement_work
        assert not case.store._provider_trace_settlement_worker_active
    for call_id, (handoff, _message) in failed.items():
        prepared = handoff._prepared
        owner = prepared._control_owner
        assert owner.dispatch.source is source
        assert owner.call_id == call_id
        with owner.coordinator._queue_lock:
            database, queued = owner.coordinator._pending[call_id]
            assert database is case.database
            assert queued.call_id == call_id and queued._control_owner is owner


@pytest.mark.parametrize("domain_case", ["control_mcp"], indirect=True)
@pytest.mark.parametrize("mode", ["normal", "cancel", "fence", "start_failure"])
async def test_actual_character_worker_outlives_cancelled_bridge_and_drain(
    domain_case, mode
):
    from tldw_chatbook.Chat.console_control_dispatch import Drain, FenceSource
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    case = domain_case
    from Tests.Chat.test_console_domain_boundary import _selection
    from tldw_chatbook.Agents.plain_chat_tool_surface import full_agent_tools_enabled
    from tldw_chatbook.Chat.console_configuration_capture import (
        CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
    )
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    assert (
        case.controller._agent_runtime_enabled
        and _selection(case).agent_runtime_enabled
    )
    assert type(case.runtime.agent_bridge) is ConsoleAgentBridge
    assert case.controller._agent_bridge is case.runtime.agent_bridge
    assert type(case.runtime._agent_runs_db) is AgentRunsDB
    assert full_agent_tools_enabled() and case.domain.native_tools_enabled()
    plane = case.sources.control_plane
    delegate = plane.local_service.runtime_delegate
    assert delegate._require_chachanotes_db() is case.database
    assert delegate._require_media_db() is case.sources.storage.media
    assert "list_characters" not in CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS
    operations = case.runtime._execution_service.ensure_permission_operations()
    await operations.set_tool_state("builtin:tldw_chatbook", "list_characters", "allow")
    entered, release = threading.Event(), threading.Event()
    thread_entered, thread_release = threading.Event(), threading.Event()
    retained_threads = []
    diagnostics = []

    def observe_permission(frame):
        from tldw_chatbook.Chat.console_control_dispatch import _command_source
        from tldw_chatbook.MCP.local_permission_operation import active

        callers = []
        caller = frame.f_back
        while caller is not None and len(callers) < 12:
            callers.append(caller.f_code.co_name)
            caller = caller.f_back
        diagnostics.append(
            {
                "operation": frame.f_locals.get("operation"),
                "selected_source": _command_source.get() is not None,
                "local_operation": active() is not None,
                "callers": callers,
            }
        )

    def observe_blocked(frame):
        diagnostics.append({"blocked": str(frame.f_locals.get("error", ""))[:180]})

    def observe_result(_frame, values):
        result = values[-1]
        diagnostics.append(
            {
                "tool_ok": result.ok,
                "tool_error": result.error,
                "dispatch": result.dispatch_state,
            }
        )

    def hold_reader(frame):
        if (
            frame.f_locals.get("self") is case.database
            and threading.current_thread() is not threading.main_thread()
        ):
            entered.set()
            assert release.wait(10), "Original character reader was not released"

    def observe_thread(frame):
        thread = frame.f_locals["self"]
        if thread.name != "tool-mcp__tldw_chatbook__list_characters":
            return
        owners = [
            item for item in journey.controls._tool_threads if item.thread is thread
        ]
        assert len(owners) == 1
        retained_threads.extend(owners)
        if mode == "start_failure":
            raise RuntimeError("task60 original tool Thread start refused")
        if mode == "fence":
            thread_entered.set()
            assert thread_release.wait(10), "Original tool Thread start not released"

    def hold_thread_return(frame):
        thread = frame.f_locals["self"]
        if thread.name == "tool-mcp__tldw_chatbook__list_characters":
            thread_entered.set()
            assert thread_release.wait(10), "Original tool Thread return not released"

    with _character_server() as server:
        journey = await _prepare(case, server, kill_switch=False)
        from tldw_chatbook.MCP.permission_process import PermissionProcess
        from tldw_chatbook.Agents.agent_models import ToolResult
        from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

        with ExitStack() as observations:
            observations.enter_context(
                _observe(
                    threading.Thread.start.__code__,
                    sys.monitoring.events.PY_START,
                    observe_thread,
                )
            )
            if mode == "cancel":
                observations.enter_context(
                    _observe(
                        threading.Thread.run.__code__,
                        sys.monitoring.events.PY_RETURN,
                        hold_thread_return,
                    )
                )
            observations.enter_context(
                _observe(
                    CharactersRAGDB.list_character_cards.__code__,
                    sys.monitoring.events.PY_RETURN,
                    hold_reader,
                )
            )
            observations.enter_context(
                _observe(
                    MCPToolProvider.invoke.__code__,
                    sys.monitoring.events.PY_RETURN,
                    observe_result,
                    values=True,
                )
            )
            observations.enter_context(
                _observe(
                    PermissionProcess.call.__code__,
                    sys.monitoring.events.PY_START,
                    observe_permission,
                )
            )
            observations.enter_context(
                _observe(
                    ToolResult.blocked.__func__.__code__,
                    sys.monitoring.events.PY_START,
                    observe_blocked,
                )
            )
            receipt = await journey.start()
            try:
                # Separate actual disclosure/provider setup from physical tool
                # custody. Each phase retains the original bounded wait.
                assert await _until(
                    lambda: bool(server.payloads) or receipt.work.done()
                )
                assert server.payloads, diagnostics
                assert await _until(
                    lambda: server.tool_request.is_set() or receipt.work.done()
                )
                assert server.tool_request.is_set(), diagnostics
                assert any(
                    tool["function"]["name"] == "mcp__tldw_chatbook__list_characters"
                    for tool in server.payloads[1]["tools"]
                )
                server.tool_response.set()
                if mode == "fence":
                    assert await _until(
                        lambda: thread_entered.is_set() or receipt.work.done()
                    )
                    assert thread_entered.is_set(), diagnostics
                    await journey.controls.dispatch(
                        FenceSource(journey.identity(), "configuration")
                    ).applied
                    thread_release.set()
                    await asyncio.gather(receipt.work, return_exceptions=True)
                    assert not entered.is_set()
                    with pytest.raises(RuntimeError, match="trace.*unresolved"):
                        await journey.controls.dispatch(
                            Drain(journey.identity())
                        ).applied
                    _assert_retained_trace(case, retained_threads[0].source)
                    assert any(
                        "source is fenced" in row.get("blocked", "")
                        for row in diagnostics
                    ), diagnostics
                    assert retained_threads and all(
                        not row.thread.is_alive() for row in retained_threads
                    )
                    return
                if mode == "start_failure":
                    assert (
                        await asyncio.wait_for(asyncio.shield(receipt.work), 20)
                    ).accepted
                    assert (
                        len(retained_threads) == 1 and retained_threads[0].never_started
                    )
                    assert retained_threads[0].thread.ident is None
                    assert (
                        not entered.is_set() and not journey.controls._mcp_submissions
                    )
                    await journey.controls.dispatch(Drain(journey.identity())).applied
                    return
                assert await _until(
                    lambda: entered.is_set() or receipt.work.done()
                ), diagnostics
                assert entered.is_set(), {
                    "provider_requests": len(server.payloads),
                    "turn_done": receipt.work.done(),
                    "outcome": repr(receipt.work.result())
                    if receipt.work.done()
                    else "pending",
                    "diagnostic": diagnostics,
                }
                (submission,) = journey.controls._mcp_submissions
                assert submission.invocation.tool.name == "list_characters"
                assert submission.invocation.handed_off
                assert len(submission.workers) == 1
                worker = submission.workers[0]
                if mode == "normal":
                    release.set()
                    assert (
                        await asyncio.wait_for(asyncio.shield(receipt.work), 20)
                    ).accepted
                    assert await asyncio.wrap_future(submission.retired) is True
                    assert worker.retired and worker.native_binding.retirement.confirmed
                    assert any(row.get("tool_ok") for row in diagnostics), diagnostics
                    await journey.controls.dispatch(Drain(journey.identity())).applied
                    assert retained_threads and all(
                        not row.thread.is_alive() for row in retained_threads
                    )
                    return
                submission.task.cancel()
                await asyncio.gather(submission.task, return_exceptions=True)
                assert not submission.retired.done() and not worker.finished
                assert await _until(thread_entered.is_set)
                assert retained_threads[0].thread.is_alive()
                await journey.controls.dispatch(
                    FenceSource(journey.identity(), "configuration")
                ).applied
                drain = journey.controls.dispatch(Drain(journey.identity()))
                await asyncio.sleep(0)
                assert not drain.applied.done()
                release.set()
                assert await asyncio.wrap_future(submission.retired) is True
                await asyncio.sleep(0)
                assert (
                    not drain.applied.done() and retained_threads[0].thread.is_alive()
                )
                thread_release.set()
                with pytest.raises(RuntimeError, match="trace.*unresolved"):
                    await drain.applied
                _assert_retained_trace(case, submission.invocation.source)
                assert not retained_threads[0].thread.is_alive()
                assert worker.retired and worker.native_binding.retirement.confirmed
                assert submission.retired.result() is True
            finally:
                server.tool_response.set()
                release.set()
                thread_release.set()
                await asyncio.gather(receipt.work, return_exceptions=True)
                print("original MCP gate observations", diagnostics)


async def _await_advertised_tool(case, receipt, server, raw_name):
    from tldw_chatbook.MCP.tool_naming import llm_tool_name

    assert await _until(
        lambda: case.session.id in case.controller._active_cancel_events
        or receipt.work.done()
    )
    assert isinstance(
        case.controller._active_cancel_events.get(case.session.id), threading.Event
    )
    assert any(
        record.task is receipt.work for record in case.runtime._turn_custody.values()
    )
    assert await _until(lambda: bool(server.payloads) or receipt.work.done())
    assert server.payloads
    assert await _until(lambda: server.tool_request.is_set() or receipt.work.done())
    assert server.tool_request.is_set()
    assert any(
        tool["function"]["name"] == llm_tool_name("builtin:tldw_chatbook", raw_name)
        for tool in server.payloads[1]["tools"]
    )


@pytest.mark.parametrize("domain_case", ["control_mcp"], indirect=True)
@pytest.mark.parametrize("tool_name", ["create_character", "update_character"])
async def test_advertised_character_mutation_retires_actual_database(
    domain_case, tool_name
):
    import sqlite3
    from tldw_chatbook.Chat.console_control_dispatch import Drain
    from tldw_chatbook.Chat.console_control_mcp import _worker
    from tldw_chatbook.MCP.tools import MCPTools
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_configuration_capture import (
        CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
    )

    case = domain_case
    assert tool_name not in CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS
    operations = case.runtime._execution_service.ensure_permission_operations()
    await operations.set_tool_state("builtin:tldw_chatbook", tool_name, "allow")
    if tool_name == "create_character":
        arguments = {
            "name": "Task60 native character",
            "fields": {"description": "Created by original MCP worker"},
        }
        expected = arguments["fields"]
    else:
        original = case.database.list_character_cards()[0]
        arguments = {
            "character_id": original["id"],
            "expected_version": original["version"],
            "fields": {"creator_notes": "Updated by original MCP worker"},
        }
        expected = arguments["fields"]
    captured = []

    def observe_write(frame, values):
        tools = frame.f_locals["self"]
        assert tools is case.sources.control_plane.local_service.runtime_delegate._tools
        worker = _worker.get()
        assert worker is not None and worker.service is tools
        captured.append((worker, values[-1]))

    with _character_server() as server:
        server.tool_name, server.tool_arguments = tool_name, arguments
        journey = await _prepare(case, server, kill_switch=False)
        receipt = await journey.start()
        try:
            await _await_advertised_tool(case, receipt, server, tool_name)
            with _observe(
                MCPTools._write_character.__code__,
                sys.monitoring.events.PY_RETURN,
                observe_write,
                values=True,
            ):
                server.tool_response.set()
                assert (
                    await asyncio.wait_for(asyncio.shield(receipt.work), 20)
                ).accepted
            assert len(captured) == 1
            worker, result = captured[0]
            assert "error_code" not in result and result["id"] > 0
            assert worker.invocation.tool.name == tool_name
            assert worker.invocation.source.sources is case.sources
            assert await asyncio.wrap_future(worker.invocation.bridge.retired) is True
            retirement = worker.native_binding.retirement
            assert worker.finished and worker.retired and retirement.confirmed
            assert retirement.borrowed is None and retirement.connection is not None
            assert retirement.creator is not threading.current_thread()
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                sqlite3.Connection.in_transaction.__get__(retirement.connection)
            with storage._lock:
                assert retirement.lease not in storage._live_leases
                assert retirement.lease._key is None
                assert retirement.operation not in storage._operations
            saved = case.database.get_character_card_by_id(result["id"])
            assert all(saved[key] == value for key, value in expected.items())
            if tool_name == "update_character":
                assert saved["version"] == original["version"] + 1
            assert (
                await journey.controls.dispatch(Drain(journey.identity())).applied
                is True
            )
        finally:
            server.tool_response.set()
            await asyncio.gather(receipt.work, return_exceptions=True)


@pytest.mark.parametrize("domain_case", ["control_mcp"], indirect=True)
@pytest.mark.parametrize(
    "phase",
    [
        "before_callback",
        "before_body",
        "submission_failure",
        "creation_failure",
    ],
)
async def test_original_mcp_submission_scheduling_retains_exact_task(
    domain_case, phase
):
    from tldw_chatbook.Chat.console_control_dispatch import Drain
    from tldw_chatbook.Chat.console_control_mcp import MCPBridgeSubmission
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    case = domain_case
    operations = case.runtime._execution_service.ensure_permission_operations()
    await operations.set_tool_state("builtin:tldw_chatbook", "list_characters", "allow")
    submissions, tasks, native_entries = [], [], []

    def observe_submission(frame):
        submission = frame.f_locals["self"]
        assert submission.controls is journey.controls
        assert submission.invocation.tool.name == "list_characters"
        submissions.append(submission)
        if phase == "before_callback":
            assert submission.task is None
            assert submission.future.cancel()

    def refuse_submission(frame):
        caller = frame.f_back
        if (
            caller is not None
            and caller.f_code is MCPBridgeSubmission.submit.__code__
            and caller.f_locals["self"] in submissions
        ):
            raise RuntimeError("task60 original loop submission refused")

    def observe_task(frame, values=()):
        if (
            not submissions
            or frame.f_locals["coro_or_future"] is not submissions[0].coroutine
        ):
            return
        if phase == "creation_failure":
            raise RuntimeError("task60 original Task creation refused")
        task = values[-1]
        assert isinstance(task, asyncio.Task)
        assert task.get_coro() is submissions[0].coroutine
        tasks.append(task)
        assert task.cancel()

    def observe_native(frame):
        if frame.f_locals["self"] is case.database:
            native_entries.append(threading.current_thread())

    with _character_server() as server:
        journey = await _prepare(case, server, kill_switch=False)
        with ExitStack() as observations:
            receipt = await journey.start()
            try:
                assert await _until(
                    lambda: case.session.id in case.controller._active_cancel_events
                    or receipt.work.done()
                )
                assert isinstance(
                    case.controller._active_cancel_events.get(case.session.id),
                    threading.Event,
                )
                assert any(
                    record.task is receipt.work
                    for record in case.runtime._turn_custody.values()
                )
                assert await _until(
                    lambda: bool(server.payloads) or receipt.work.done()
                )
                assert server.payloads
                assert await _until(
                    lambda: server.tool_request.is_set() or receipt.work.done()
                )
                assert server.tool_request.is_set()
                assert any(
                    tool["function"]["name"] == "mcp__tldw_chatbook__list_characters"
                    for tool in server.payloads[1]["tools"]
                )
                # The original advertised-tool response is held. Arm only the
                # intended submission; do not instrument unrelated setup.
                observations.enter_context(
                    _observe(
                        MCPBridgeSubmission.submit.__code__,
                        sys.monitoring.events.PY_START,
                        observe_submission,
                    )
                )
                observations.enter_context(
                    _observe(
                        CharactersRAGDB.list_character_cards.__code__,
                        sys.monitoring.events.PY_START,
                        observe_native,
                    )
                )
                if phase == "submission_failure":
                    observations.enter_context(
                        _observe(
                            asyncio.BaseEventLoop.call_soon_threadsafe.__code__,
                            sys.monitoring.events.PY_START,
                            refuse_submission,
                        )
                    )
                elif phase in {"before_body", "creation_failure"}:
                    observations.enter_context(
                        _observe(
                            asyncio.ensure_future.__code__,
                            sys.monitoring.events.PY_RETURN
                            if phase == "before_body"
                            else sys.monitoring.events.PY_START,
                            observe_task,
                            values=phase == "before_body",
                        )
                    )
                server.tool_response.set()
                assert (
                    await asyncio.wait_for(asyncio.shield(receipt.work), 20)
                ).accepted
                assert len(submissions) == 1
                submission = submissions[0]
                assert await asyncio.wrap_future(submission.retired) is True
                if phase in {"creation_failure", "submission_failure"}:
                    assert (
                        submission.task is None
                        and submission.coroutine.cr_frame is None
                    )
                    assert not native_entries and not submission.workers
                else:
                    assert type(submission.task) is asyncio.Task
                    assert submission.task.done()
                    if phase == "before_body":
                        assert tasks == [submission.task]
                        assert submission.task.cancelled()
                        assert not native_entries and not submission.workers
                    else:
                        assert submission.future.cancelled()
                        # Original run_coroutine_threadsafe chaining can start
                        # its Task before propagating a pre-callback cancel.
                        assert all(
                            worker.finished and worker.retired
                            for worker in submission.workers
                        )
                        assert all(
                            worker.native_binding.retirement.confirmed
                            for worker in submission.workers
                        )
                assert (
                    await journey.controls.dispatch(Drain(journey.identity())).applied
                    is True
                )
                assert (
                    not journey.controls._mcp_submissions
                    and not journey.controls._tool_threads
                )
            finally:
                server.tool_response.set()
                await asyncio.gather(receipt.work, return_exceptions=True)
