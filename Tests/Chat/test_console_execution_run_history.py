"""Finite original history reads are independent of unrelated display expiry."""

import asyncio
import io
import sys
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        None,
        "source",
        "visit",
        "attachment",
        "session",
        "conversation",
        "retirement",
        "failure",
    ],
)
async def test_history_transport_joins_retirement_without_full_or_readiness(changed):
    """Transport ordering only; the backend test below proves real native use."""
    from Tests.Chat.test_console_execution_process import (
        _ready_client,
        _reply,
        _retrieve,
    )
    from Tests.Chat.test_console_execution_workspace import _until
    from tldw_chatbook.Chat.console_execution_process import (
        ConsoleChannelLost,
        ConsoleRemoteFailure,
    )

    client = _ready_client()
    screen = object()
    client._view, client._visit = screen, 1
    client._readiness_deadline = -1
    row = SimpleNamespace(
        id="chat", incarnation_id="incarnation", persisted_conversation_id="saved"
    )
    published = SimpleNamespace(active_session_id=row.id, session=lambda _: row)
    client.projection_view = published
    client.store_reader = SimpleNamespace(_current=lambda: published)

    def require_view(view, visit):
        if client._view is not view or client._visit != visit:
            raise ConsoleChannelLost("original visit changed")

    client.require_view = require_view
    request = dict(
        conversation_id="saved", include_superseded=False, limit=3, agent_kind="primary"
    )
    await client._projection_lock.acquire()
    task = asyncio.create_task(client.read_run_history(request))
    receipt = None
    try:
        await _until(lambda: len(client._channel.messages) == 1)
        receipt = next(iter(client._requests.values()))
        assert receipt.operation == "ReadRunHistory"
        assert dict(client._channel.messages[0].payload["request"]) == request
        client._receive(
            _reply(
                client, receipt, 1, "received", {"sequence": receipt.identity.sequence}
            )
        )
        payload = {
            "source_revision": 0,
            "session_id": "chat",
            "incarnation_id": "incarnation",
            "attachment_id": client._attachment_id,
            "visit": 1,
            "run_history": [
                {
                    "id": "run",
                    "conversation_id": "saved",
                    "parent_run_id": None,
                    "agent_kind": "primary",
                    "status": "done",
                    "created_at": "created",
                    "updated_at": "updated",
                    "assistant_message_id": None,
                    "steps": "[]",
                }
            ],
        }
        client._receive(
            _reply(
                client,
                receipt,
                2,
                "failed" if changed == "failure" else "applied",
                {"code": "command_failed", "stage": "applied"}
                if changed == "failure"
                else payload,
            )
        )
        await asyncio.sleep(0)
        assert not task.done()
        if changed == "source":
            client._source_revision += 1
        elif changed == "visit":
            client._visit += 1
        elif changed == "attachment":
            client._attachment_id = "other"
        elif changed == "session":
            published.active_session_id = "other"
        elif changed == "conversation":
            row.persisted_conversation_id = "other"
        client._receive(
            _reply(
                client,
                receipt,
                3,
                "retired",
                {
                    "operation_id": receipt.identity.request_id,
                    "retired": changed != "retirement",
                },
            )
        )
        if changed is None:
            assert (await task)[0]["steps"] == []
            assert (
                client.projection_view is published and client._readiness_deadline == -1
            )
        else:
            with pytest.raises(
                ConsoleRemoteFailure if changed == "failure" else ConsoleChannelLost
            ):
                await task
        assert client._projection_lock.locked()
        assert len(client._channel.messages) == 1
    finally:
        client._projection_lock.release()
        client._fail(RuntimeError("test cleanup"))
        await asyncio.gather(task, *client._projection_reads, return_exceptions=True)
        if receipt is not None:
            await _retrieve(receipt)


@private_profile_test
@pytest.mark.asyncio
async def test_actual_history_query_keeps_filters_sources_and_retirement_without_full(
    request, monkeypatch
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_backend_run_history import (
        ConsoleBackendRunHistoryRead,
    )
    from tldw_chatbook.Chat.console_domain_display import ConsoleDisplayFull
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleWireMessage,
        WireIdentity,
        decode_message,
        encode_message,
    )
    from tldw_chatbook.Chat.console_execution_run_history_values import (
        decode_run_history,
    )
    from tldw_chatbook.Chat.console_execution_worker import (
        _BackendChannel,
        _BackendService,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.base_db import run_owned_db_call

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="history"
    )
    held = False
    try:
        backend.start()
        session = backend.store.create_session(title="Original history")
        lifecycle = backend.presentation.view_lifecycle
        await lifecycle.prepare(lifecycle.admit("client", 1, "mount"))
        assert backend.store.persist_session_if_needed(session.id)
        database = backend.runtime._agent_bridge._db
        assert type(database) is AgentRunsDB

        def seed():
            for kind, status in (
                ("primary", "done"),
                ("primary", "superseded"),
                ("subagent", "done"),
            ):
                run = database.create_run(
                    conversation_id=session.persisted_conversation_id, agent_kind=kind
                )
                database.append_steps(run, [{"kind": "result", "ok": True}])
                assert database.set_status(run, status)

        await run_owned_db_call(database, seed)
        channel = _BackendChannel(io.BytesIO(), io.BytesIO())
        service = _BackendService(backend, channel)

        async def command(query, *, attachment="client", visit=1, source=0):
            sequence = backend._sequence + 1
            message = ConsoleWireMessage(
                WireIdentity(
                    "profile",
                    "history",
                    sequence,
                    f"history-{sequence}",
                    session.id,
                    session.incarnation_id,
                    source,
                ),
                "command",
                "ReadRunHistory",
                {"attachment_id": attachment, "visit": visit, "request": query},
            )
            handler = service.accept(message)
            if handler is not None:
                await handler
            frames = []
            while not channel.outbox.empty():
                frame = decode_message(channel.outbox.get_nowait())
                if frame.identity.request_id == message.identity.request_id:
                    frames.append(frame)
            return frames

        reads, full = [], []

        def read_return(frame):
            owner = frame.f_locals["self"]
            assert owner.database is database and owner.checkpoint is lifecycle._current
            reads.append(owner)

        await backend.controls._projection_read_lock.acquire()
        held = True
        with (
            _observe(
                ConsoleBackendRunHistoryRead.read.__code__,
                sys.monitoring.events.PY_RETURN,
                read_return,
            ),
            _observe(
                ConsoleDisplayFull.run.__code__,
                sys.monitoring.events.PY_START,
                lambda _: full.append(True),
            ),
        ):
            for superseded, limit, kind in (
                (True, None, None),
                (False, None, None),
                (True, 1, "primary"),
            ):
                query = dict(
                    conversation_id=session.persisted_conversation_id,
                    include_superseded=superseded,
                    limit=limit,
                    agent_kind=kind,
                )
                expected = await run_owned_db_call(
                    database, database.list_runs, **query
                )
                async with asyncio.timeout(10):
                    frames = await command(query)
                assert [frame.kind for frame in frames] == [
                    "received",
                    "applied",
                    "retired",
                ]
                assert frames[-1].payload["retired"] is True
                rows = decode_run_history(
                    encode_message(frames[1])["payload"]["run_history"]
                )
                assert [(row["id"], row["steps"]) for row in rows] == [
                    (row["id"], row["steps"]) for row in expected
                ]
                assert reads[-1].observation.confirmed()
                assert not reads[-1].reads and not backend.controls.admitted_effects
            assert len(reads) == 3 and not full
            for mismatch in ({"attachment": "other"}, {"visit": 2}, {"source": 1}):
                frames = await command(query, **mismatch)
                assert (
                    frames[0].kind == "failed"
                    and frames[0].payload["stage"] == "admission"
                )
            frames = await command({**query, "conversation_id": "other"})
            assert [frame.kind for frame in frames] == ["received", "failed", "retired"]
            assert frames[-1].payload["retired"] is True
            with monkeypatch.context() as changed:
                changed.setattr(database, "list_runs", lambda **_: [])
                frames = await command(query)
                assert [frame.kind for frame in frames] == [
                    "received",
                    "failed",
                    "retired",
                ]
            await lifecycle.prepare(lifecycle.admit("client", 1, "hidden"))
            frames = await command(query)
            assert [frame.kind for frame in frames] == ["failed"]
            assert len(reads) == 3 and not full
            assert backend.controls._projection_read_lock.locked()
    finally:
        if held:
            backend.controls._projection_read_lock.release()
        await backend.aclose()
