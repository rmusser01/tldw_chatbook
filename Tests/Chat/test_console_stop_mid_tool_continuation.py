"""A Stop that lands while a native tool executes (TASK-37653 macos-final60).

The live qualification's primary run ended with "Provider continuation could
not be persisted" when the 600-second deadline Stop landed during a tool call:
the Stop's terminal write (owner loop) and the tool result's continuation
write (agent worker) touched the same owner message without sharing its
generation lock.
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import threading
import time
from unittest.mock import patch

import httpx
import pytest
import urllib3

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.owned_http_client]

MODEL = "synthetic-qwen"


def _sse(delta: dict, finish: str) -> bytes:
    chunks = [
        {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    events = [
        {"id": "s", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
        | chunk
        for chunk in chunks
    ]
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return (body + "data: [DONE]\n\n").encode()


@pytest.mark.asyncio
async def test_stop_racing_a_tool_result_ends_cancelled_not_persistence_error(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    profile = {
        "custom_endpoints": {
            "stopmid": {
                "display_name": "Stop fixture",
                "family": "openai_compatible",
                "base_url": "http://127.0.0.1:8010/v1",
                "models": [MODEL],
            }
        }
    }
    chat = CharactersRAGDB(str(tmp_path / "chat.db"), client_id="stopmid")
    gateway = ConsoleProviderGateway(config_provider=lambda: profile, environ={})
    store = ConsoleChatStore(persistence=ChatPersistenceService(chat))
    runs = AgentRunsDB(tmp_path / "runs.db", client_id="stopmid")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="custom-ep:stopmid",
        model=MODEL,
        agent_bridge=ConsoleAgentBridge(
            agent_runs_db=runs, store=store, provider_gateway=gateway
        ),
        agent_runtime_enabled=True,
    )
    session = store.ensure_session()
    loop = asyncio.get_running_loop()
    stopping = threading.Event()
    stop_committed = threading.Event()
    finish_entered = threading.Event()
    finish_done = threading.Event()
    persist_errors: list[str] = []
    wire: list[dict] = []

    original_invoke = AgentService._make_invoke_tool

    def stopping_invoke(service, *args, **kwargs):
        invoke = original_invoke(service, *args, **kwargs)

        def deadline_stop_lands_mid_call(call):
            result = invoke(call)
            # The qualification's deadline watchdog: Stop on the owner loop
            # while this tool call is still in flight.
            stopping.set()
            loop.call_soon_threadsafe(
                lambda: controller.stop_active_run(record_user_stop=False)
            )
            assert stop_committed.wait(5), "Stop never reached its terminal write"
            return result

        return deadline_stop_lands_mid_call

    monkeypatch.setattr(AgentService, "_make_invoke_tool", stopping_invoke)
    original_variant = ConsoleChatStore._persist_generation_variant

    def paused_after_commit(self, *args, **kwargs):
        committed = original_variant(self, *args, **kwargs)
        if stopping.is_set() and not stop_committed.is_set():
            # Hold the Stop between its durable commit and its in-memory
            # version publication -- the window the tool result raced into.
            stop_committed.set()
            assert finish_entered.wait(5), "tool result never reached persistence"
            finish_done.wait(1.0)
        return committed

    monkeypatch.setattr(
        ConsoleChatStore, "_persist_generation_variant", paused_after_commit
    )
    original_persist = ConsoleChatStore.persist_provider_continuation_event

    def observed_persist(self, event, **kwargs):
        finishing = stopping.is_set() and type(event).__name__ == "ToolCallFinished"
        if finishing:
            finish_entered.set()
        try:
            return original_persist(self, event, **kwargs)
        except Exception as exc:
            persist_errors.append(f"{type(event).__name__}: {exc}")
            raise
        finally:
            if finishing:
                finish_done.set()

    monkeypatch.setattr(
        ConsoleChatStore, "persist_provider_continuation_event", observed_persist
    )

    async def metadata(transport, request):
        return httpx.Response(
            200, json={"data": [{"id": MODEL, "context_length": 32768}]}
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", metadata)

    def native(pool, connection, method, url, *args, **kwargs):
        wire.append(json.loads(kwargs.get("body", args[0] if args else None)))
        assert (pool.host, pool.port, url) == (
            "127.0.0.1",
            8010,
            "/v1/chat/completions",
        )
        call = {
            "index": 0,
            "id": "call_1_0",
            "type": "function",
            "function": {"name": "get_current_datetime", "arguments": "{}"},
        }
        return urllib3.response.HTTPResponse(
            body=io.BytesIO(_sse({"tool_calls": [call]}, "tool_calls")),
            status=200,
            headers={"Content-Type": "text/event-stream"},
            preload_content=False,
        )

    def denied(*args, **kwargs):
        raise AssertionError("network prohibited")

    try:
        with (
            patch.object(socket.socket, "connect", denied),
            patch.object(socket.socket, "connect_ex", denied),
            patch.object(socket, "getaddrinfo", denied),
            patch.object(
                urllib3.connectionpool.HTTPConnectionPool, "_make_request", native
            ),
        ):
            result = await controller.submit_draft(
                "What time is it?", session_id=session.id
            )
        assert result.accepted and stop_committed.is_set()
        # Stop cancels this submit_draft task, but the agent worker runs on
        # under asyncio.shield (_run_maintenance_agent_call) until it writes
        # the terminal status; reading before it finished saw "running".
        await asyncio.gather(
            *tuple(controller._maintenance_background), return_exceptions=True
        )
        connection = runs._held_connection()
        run_id, status = connection.execute(
            "SELECT id, status FROM agent_runs"
        ).fetchone()
        errors = [
            json.loads(row[0])["summary"]
            for row in connection.execute(
                "SELECT payload FROM agent_run_steps WHERE run_id = ? ORDER BY seq",
                (run_id,),
            )
            if json.loads(row[0])["kind"] == "error"
        ]
        assert (status, errors) == ("cancelled", [])
        assert persist_errors == []
        assert len(wire) == 1
    finally:
        await controller.shutdown()
        await gateway.aclose()
        with chat.quiesce_connections(timeout_seconds=2):
            assert chat.registered_connection_count() == 0
        runs.close()


def _continuation_store(tmp_path, *, persist_owner: bool, content: str = ""):
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(str(tmp_path / "owner.db"), client_id="owner-order")
    store = ConsoleChatStore(persistence=ChatPersistenceService(database))
    session = store.create_session(title="Owner order")
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="hi", persist=True
    )
    owner = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content=content,
        persist=persist_owner,
    )
    return database, store, session, owner


def test_close_racing_a_continuation_write_cannot_deadlock(tmp_path):
    """TASK-37653 review: owner scope -> first persistence is the one order.

    The continuation write holds the owner's generation scope and then
    persists the session; close used to take first persistence and then the
    owner scope while purging -- an AB/BA deadlock between the agent worker
    and the UI thread.
    """
    from Tests.Chat.test_console_provider_continuation import (
        _MOONSHOT_RUN_TARGET,
        _active_checkpoint,
    )
    from tldw_chatbook.Agents.agent_models import (
        ContinuationEventContext,
        ToolBatchReady,
    )

    database, store, session, owner = _continuation_store(tmp_path, persist_owner=False)
    original_persist = store.persist_session_if_needed
    outcomes: dict[str, object] = {}
    entered = threading.Event()

    def persist_after_close_waits(session_id, **kwargs):
        if threading.current_thread().name == "continuation-worker":
            # The worker holds the owner scope here. Let close reach it --
            # with or without first persistence held, which is the fix.
            entered.set()
            deadline = time.monotonic() + 5
            while store._generation_runtime_counts(owner.id)[1] < 2:
                assert time.monotonic() < deadline, "close never reached the owner"
                time.sleep(0.001)
        return original_persist(session_id, **kwargs)

    store.persist_session_if_needed = persist_after_close_waits

    def run(name, operation):
        try:
            outcomes[name] = operation()
        except BaseException as exc:  # noqa: BLE001 - reported below
            outcomes[name] = exc
        finally:
            database.close_connection()

    event = ToolBatchReady(
        ContinuationEventContext(owner.id, "run-1", "primary", "persistent"),
        _active_checkpoint(),
        None,
    )
    worker = threading.Thread(
        target=run,
        args=(
            "persist",
            lambda: store.persist_provider_continuation_event(
                event, continuation_target=_MOONSHOT_RUN_TARGET
            ),
        ),
        name="continuation-worker",
        daemon=True,
    )
    closer = threading.Thread(
        target=run,
        args=("close", lambda: store.close_session(session.id)),
        name="console-close",
        daemon=True,
    )
    try:
        worker.start()
        assert entered.wait(5), "the continuation write never took its owner"
        closer.start()
        worker.join(10)
        closer.join(10)
        assert not worker.is_alive() and not closer.is_alive(), (
            "close and the continuation write deadlocked"
        )
        assert outcomes["persist"] is None
        assert not isinstance(outcomes["close"], BaseException), outcomes["close"]
        assert session.id not in {item.id for item in store.sessions()}
        assert database.get_message_by_id(owner.id)["provider_continuation_json"]
    finally:
        database.close_connection()


def _sealed(checkpoint, message_id="message"):
    from dataclasses import replace

    envelope = {
        "schema_version": 1,
        "namespace_id": "a" * 64,
        "session_id": "session",
        "run_id": "run-1",
        "conversation_id": "conversation",
        "message_id": message_id,
        "installations": [],
        "checkpoint_digest": "b" * 64,
        "mac": "c" * 64,
    }
    return replace(checkpoint, schema_version=2, managed_resume=json.dumps(envelope))


def test_managed_seal_runs_outside_the_owner_scope_and_rechecks_its_owner(
    tmp_path,
):
    """TASK-37653 review: Stop must not wait behind a blocked plugin seal.

    The finalizer blocks on the plugin worker loop with no timeout. Sealed
    under the owner's generation scope, it froze every Stop on the owner
    loop; the seal is pure, so the locked write only rechecks its inputs.
    """
    from dataclasses import replace

    from Tests.Chat.test_console_provider_continuation import (
        _MOONSHOT_RUN_TARGET,
        _active_checkpoint,
    )
    from tldw_chatbook.Agents.agent_models import (
        ContinuationEventContext,
        ToolBatchReady,
        ToolCallExecuting,
    )

    database, store, _session, owner = _continuation_store(tmp_path, persist_owner=True)
    context = ContinuationEventContext(owner.id, "run-1", "primary", "persistent")
    stop_got_owner: list[bool] = []

    def finalize(checkpoint, conversation_id, message_id, run_id):
        stopped = threading.Event()

        def stop_on_owner_loop():
            with store._generation_owner_scope(owner.id):
                stopped.set()

        stopper = threading.Thread(target=stop_on_owner_loop, daemon=True)
        stopper.start()
        stop_got_owner.append(stopped.wait(2))
        stopper.join(5)
        return _sealed(checkpoint)

    try:
        store.persist_provider_continuation_event(
            ToolBatchReady(context, _active_checkpoint(), None),
            continuation_target=_MOONSHOT_RUN_TARGET,
            checkpoint_finalizer=finalize,
        )
        assert stop_got_owner == [True]
        assert store.get_message(owner.id).provider_continuation.managed_resume

        moves: list[int] = []

        def finalize_while_owner_changes(checkpoint, *args):
            # Another writer moves the owner after every seal's inputs were read.
            moves.append(len(moves))
            message = store._message_or_raise(owner.id)
            current = message.provider_continuation
            message.provider_continuation = replace(
                current,
                rounds=(
                    replace(current.rounds[0], assistant_content=f"moved {moves}"),
                ),
            )
            return _sealed(checkpoint)

        with pytest.raises(RuntimeError, match="changed while sealing"):
            store.persist_provider_continuation_event(
                ToolCallExecuting(context, "PRIVATE-CALL-ID", 1),
                continuation_target=_MOONSHOT_RUN_TARGET,
                checkpoint_finalizer=finalize_while_owner_changes,
            )
        assert moves == [0, 1]  # one fresh re-seal, then the refusal
    finally:
        database.close_connection()


def test_stop_persisting_the_owner_during_the_seal_reseals_once_and_writes(
    tmp_path,
):
    """TASK-37653 review: a Stop may give the owner its durable id mid-seal.

    The seal covers ``persisted_message_id or message.id``. A Stop that
    persists the still-unpersisted assistant while the seal runs outside the
    owner scope changes that input, and the recheck surfaced "Provider
    continuation could not be persisted". The write now re-seals once over
    the fresh inputs.
    """
    from Tests.Chat.test_console_provider_continuation import (
        _MOONSHOT_RUN_TARGET,
        _active_checkpoint,
    )
    from tldw_chatbook.Agents.agent_models import (
        ContinuationEventContext,
        ToolBatchReady,
    )
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )

    database, store, _session, owner = _continuation_store(
        tmp_path, persist_owner=False, content="partial answer"
    )
    sealed_for: list[str] = []

    def finalize_while_stop_persists(checkpoint, conversation_id, message_id, run_id):
        if not sealed_for:
            # The Stop's terminal write, on the owner loop, while this seal
            # runs outside the owner scope: the row gets a DB-assigned id.
            store.persist_message_if_needed(owner.id)
        sealed_for.append(message_id)
        return _sealed(checkpoint, message_id)

    try:
        store.persist_provider_continuation_event(
            ToolBatchReady(
                ContinuationEventContext(owner.id, "run-1", "primary", "persistent"),
                _active_checkpoint(),
                None,
            ),
            continuation_target=_MOONSHOT_RUN_TARGET,
            checkpoint_finalizer=finalize_while_stop_persists,
        )
        durable_id = store.get_message(owner.id).persisted_message_id
        assert durable_id not in {None, owner.id}
        assert sealed_for == [owner.id, durable_id]
        stored = parse_provider_continuation_json(
            database.get_message_by_id(durable_id)["provider_continuation_json"]
        )
        assert json.loads(stored.managed_resume)["message_id"] == durable_id
    finally:
        database.close_connection()


def test_close_refuses_a_message_that_joined_after_its_owners_were_claimed(
    tmp_path,
):
    """TASK-37653 review: close invalidates only owners whose scope it holds.

    Close claims the session's owner scopes from one snapshot; a message that
    joins afterwards would be invalidated by the purge without its scope
    (re-opening the lock-order deadlock for it), so close refuses instead.
    """
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    database, store, session, owner = _continuation_store(tmp_path, persist_owner=True)
    original_invalidate = store.invalidate_response_rule_source

    def join_after_claims(session_id, reason):
        if reason == "closed":
            store.append_message(
                session_id, role=ConsoleMessageRole.TOOL, content="late tool"
            )
        return original_invalidate(session_id, reason)

    store.invalidate_response_rule_source = join_after_claims
    try:
        with pytest.raises(RuntimeError, match="joined this chat while closing"):
            store.close_session(session.id)
        assert session.id in {item.id for item in store.sessions()}
        assert store.get_message(owner.id) is not None
    finally:
        database.close_connection()
