"""Finite decision reads retain exact transport scope and original owners."""

import asyncio
import io
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_execution_process import _ready_client, _reply, _retrieve
from Tests.Chat.test_console_execution_workspace import _until
from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_execution_decision_head import ConsoleDecisionHead, encode_decision_head
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost, ConsoleRemoteFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["current", "source", "visit", "attachment", "incarnation", "negative", "admission", "cancelled"])
async def test_decision_read_waits_for_its_receipt_without_full_lock(case):
    client = _ready_client()
    screen = object()
    client._view, client._visit = screen, 1
    row = SimpleNamespace(id="session", incarnation_id="incarnation")
    client.projection_view = SimpleNamespace(
        source_revision=0, active_session_id=row.id, session=lambda _: row,
    )
    client.require_view = lambda view, visit: (
        view is client._view and visit == client._visit
    ) or (_ for _ in ()).throw(ConsoleChannelLost("changed view"))
    await client._projection_lock.acquire()
    task = asyncio.create_task(client.read_decision_head(screen, 1, row.id, row.incarnation_id))
    receipt = None
    try:
        await _until(lambda: bool(client._channel.messages))
        receipt = next(iter(client._requests.values()))
        assert receipt.operation == "ReadDecisionHead"
        if case == "admission":
            client._receive(_reply(client, receipt, 1, "failed", {"code": "command_refused", "stage": "admission"}))
            with pytest.raises(ConsoleRemoteFailure):
                await task
            assert not receipt.retired.done() and client.decision_head is None
            return
        client._receive(_reply(client, receipt, 1, "received", {"sequence": receipt.identity.sequence}))
        head = ConsoleDecisionHead(0, row.id, row.incarnation_id, client._attachment_id, 1, None, None)
        client._receive(_reply(client, receipt, 2, "applied", encode_decision_head(head)))
        if case == "source":
            client._source_revision += 1
        elif case == "visit":
            client._visit += 1
        elif case == "attachment":
            client._attachment_id = "replacement"
        elif case == "incarnation":
            row.incarnation_id = "replacement"
        await asyncio.sleep(0)
        assert not task.done() and client.decision_head is None
        if case == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(client._projection_reads) == 1
        client._receive(_reply(client, receipt, 3, "retired", {
            "operation_id": receipt.identity.request_id, "retired": case != "negative",
        }))
        if case == "current":
            assert await task == head and client._projection_lock.locked()
        elif case == "cancelled":
            await _until(lambda: not client._projection_reads)
            assert client.decision_head == head
        else:
            with pytest.raises(ConsoleChannelLost):
                await task
            assert client.decision_head is None
        assert (client._channel_error is not None) == (case == "negative")
    finally:
        client._projection_lock.release()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if receipt is not None:
            await _retrieve(receipt)


@private_profile_test
@pytest.mark.asyncio
async def test_original_decision_heads_bypass_full_lock_without_claiming_answerability(request, monkeypatch):
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
    from tldw_chatbook.Chat.console_chat_controller import ProjectInstructionDispatchNotice
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import open_console_backend_profile
    from tldw_chatbook.Chat.console_execution_protocol import ConsoleWireMessage, WireIdentity, decode_message, encode_message
    from tldw_chatbook.Chat.console_execution_decision_head import decode_decision_head
    from tldw_chatbook.Chat.console_execution_worker import _BackendChannel, _BackendService

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(expected_config_path=path, expected_data_directory=directory)
    backend = ConsoleExecutionBackend(profile, profile_epoch="profile", service_epoch="decision-read")
    pending = []
    lock_held = False
    try:
        backend.start()
        session = backend.store.create_session(title="Original decision heads")
        lifecycle = backend.presentation.view_lifecycle
        await lifecycle.prepare(lifecycle.admit("actual-client", 1, "mount"))
        channel = _BackendChannel(io.BytesIO(), io.BytesIO())
        service = _BackendService(backend, channel)

        async def command(operation, payload, *, source=0):
            sequence = backend._sequence + 1
            message = ConsoleWireMessage(WireIdentity(
                "profile", "decision-read", sequence, f"decision-{sequence}",
                session.id, session.incarnation_id, source,
            ), "command", operation, payload)
            handler = service.accept(message)
            if handler is not None:
                await handler
            frames = []
            while not channel.outbox.empty():
                frame = decode_message(channel.outbox.get_nowait())
                if frame.identity.request_id == message.identity.request_id:
                    frames.append(frame)
            return frames

        async def read():
            frames = await command("ReadDecisionHead", {"attachment_id": "actual-client", "visit": 1})
            assert [frame.kind for frame in frames] == ["received", "applied", "retired"]
            assert frames[-1].payload["retired"] is True
            return decode_decision_head(encode_message(frames[1])["payload"])

        notice = ProjectInstructionDispatchNotice(session.id, "Local endpoint", "AGENTS.md", ".", 42, (), ())
        pending.append(asyncio.create_task(asyncio.to_thread(backend.runtime._confirm_project_instruction_dispatch, notice)))
        await _until(lambda: bool(backend.runtime._project_dispatch_decisions))
        # Hold the exact FULL serialization lock. The resident decision read
        # neither waits on it nor constructs a substitute display/native owner.
        await backend.controls._projection_read_lock.acquire()
        lock_held = True
        head = await read()
        assert head.project.notice == notice
        original = backend.runtime._project_dispatch_decisions[head.project.decision_id]
        assert original.answerable_since is None and not original.ever_answerable
        assert not pending[0].done()
        mount = await command("ProjectDecisionPresentation", {
            "attachment_id": "actual-client", "visit": 1,
            "decision_id": head.project.decision_id, "action": "mounted",
        })
        assert mount[1].payload["applied"] and original.answerable_since is not None
        assert (await read()).project.decision_id == head.project.decision_id
        await command("ResolveDecision", {"decision": {
            "kind": "ProjectDispatchDecision", "decision_id": head.project.decision_id, "action": "proceed",
        }})
        assert await pending[0] == "proceed"
        assert (await read()).project is None  # Discover completion under the actual old modal.
        await lifecycle.prepare(lifecycle.admit("actual-client", 1, "visible"))
        host = backend.controller._interrupt_host
        for index in range(2):
            call = MCPPendingCall(
                llm_name="fs_read", server_key="local:__local__", tool_name="fs_read",
                server_label="Local tools", arguments={"path": "README.md"},
                reason="ask", options=("approve_once", "deny"), call_id=f"call-{index}",
            )
            pending.append(asyncio.create_task(asyncio.to_thread(
                host.request_mcp_approvals, [call], session_id=session.id,
            )))
            await _until(lambda count=index + 1: len(host.registries["approval"]) == count)
            await _until(lambda count=index + 1: len(host.payloads["approval"]) == count)
        first = await read()
        assert first.project is None and first.decision.decision_type == "approval"
        assert backend.controller._answerable_decision_by_session.get(session.id) is None
        for index in range(2):
            head = await read()
            call_ids = [item["call_id"] for item in head.decision.payload["calls"]]
            assert call_ids == [f"call-{index}"]
            shown = await command("ResolveDecision", {"decision": {
                "kind": "ApprovalPresentationDecision", "round_id": head.decision.decision_id,
                "call_ids": call_ids, "attachment_id": "actual-client", "visit": 1,
            }})
            assert shown[1].payload["applied"]
            await command("ResolveDecision", {"decision": {
                "kind": "ApprovalRoundDecision", "round_id": head.decision.decision_id,
                "call_ids": call_ids, "decisions": {call_ids[0]: "approve_once"}, "denial_reasons": {},
            }})
            assert await pending[index + 1] == {call_ids[0]: "approve_once"}
        assert (await read()).decision is None
        with monkeypatch.context() as changed:
            changed.setattr(backend.presentation.project_decisions, "read_head", lambda *_: head)
            frames = await command("ReadDecisionHead", {"attachment_id": "actual-client", "visit": 1})
            assert [frame.kind for frame in frames] == ["failed"]
            assert frames[0].payload["stage"] == "admission"
        await lifecycle.prepare(lifecycle.admit("actual-client", 1, "hidden"))
        frames = await command("ReadDecisionHead", {"attachment_id": "actual-client", "visit": 1})
        assert [frame.kind for frame in frames] == ["failed"]
        assert backend.controls._projection_read_lock.locked()
    finally:
        if lock_held:
            backend.controls._projection_read_lock.release()
        for decision_id in tuple(backend.runtime._project_dispatch_decisions):
            backend.runtime.resolve_project_instruction_dispatch(decision_id, "cancel")
        backend.controller._signal_stop(session_id=session.id)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await backend.aclose()
