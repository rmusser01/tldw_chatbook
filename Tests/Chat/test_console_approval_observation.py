"""Original approval observer reads the retired head, not an unrelated FULL.

These tests use the installed client/reader and framed receipt dispatcher.
The transport is controlled; they do not certify native backend retirement.
"""

import asyncio

import pytest
import pytest_asyncio
from textual.app import App
from textual.screen import Screen

from Tests.Chat.test_console_execution_decision_values import _approval
from Tests.Chat.test_console_execution_process import _ready_client, _reply, _retrieve
from Tests.Chat.test_console_execution_readers import _publication
from Tests.Chat.test_console_execution_workspace import _until
from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Chat.console_execution_decision_head import ConsoleDecisionHead, encode_decision_head
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost, ConsoleRemoteFailure
from tldw_chatbook.Chat.console_execution_projection_view import ConsoleProjectionUnavailable
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


@pytest_asyncio.fixture
async def bound_reader():
    # Exercise the real require_view properties without composing/starting a
    # Console runtime. The original App/Screen relationship is installed here.
    app = App()
    app._observation_clients = []
    screen = object.__new__(ChatScreen)
    Screen.__init__(screen)
    screen.app_instance = app
    screen._attach(app)
    app._screen_stacks[app._current_mode].append(screen)
    try:
        yield app, screen
    finally:
        for client in app._observation_clients:
            client._fail(ConsoleChannelLost("test cleanup"))
            await asyncio.gather(*client._projection_reads, return_exceptions=True)
            for receipt in tuple(client._requests.values()):
                await _retrieve(receipt)


def _client(app, screen):
    client = _ready_client()
    app._observation_clients.append(client)
    client._app = app
    app.console_execution_client = client
    app.console_runtime = None
    client._view, client._visit = screen, 1
    session = ConsoleChatSession(id="session")
    client.projection_view = _publication(session)
    client._projection = client.projection_view.projections[0]
    return client, session


def _deliver(client, receipt, kind, payload):
    client._receive(_reply(client, receipt, client._received_sequence + 1, kind, payload))


async def _read_head(client, session, decision, *, retire=True):
    task = asyncio.create_task(client.read_decision_head(
        client._view, client._visit, session.id, session.incarnation_id,
    ))
    await _until(lambda: any(item.operation == "ReadDecisionHead" for item in client._requests.values()))
    receipt = next(item for item in client._requests.values() if item.operation == "ReadDecisionHead")
    _deliver(client, receipt, "received", {"sequence": receipt.identity.sequence})
    head = ConsoleDecisionHead(
        0, session.id, session.incarnation_id, client._attachment_id, 1, decision, None,
    )
    _deliver(client, receipt, "applied", encode_decision_head(head))
    await asyncio.sleep(0)
    assert not task.done()
    if retire:
        _deliver(client, receipt, "retired", {"operation_id": receipt.identity.request_id, "retired": True})
        await task
    return task, receipt


@pytest.mark.asyncio
async def test_original_approval_observer_reads_retired_head_while_full_is_held(bound_reader):
    client, session = _client(*bound_reader)
    await client._projection_lock.acquire()
    try:
        assert client.controller_reader._parked_approval_payloads is None
        task, receipt = await _read_head(client, session, _approval(), retire=False)
        assert client.controller_reader._parked_approval_payloads is None
        _deliver(client, receipt, "retired", {"operation_id": receipt.identity.request_id, "retired": True})
        await task
        reader = client.controller_reader
        observed = reader._parked_approval_payloads
        assert observed is client.decision_head.decision
        payload = reader._head_round_payload(observed, session.id)
        assert payload["calls"][0]["call_id"] == "call"
        assert client._projection.decision is None and client._projection_lock.locked()
        assert [message.operation for message in client._channel.messages] == ["ReadDecisionHead"]
        client.submit("ResolveDecision", {"decision": {
            "kind": "ApprovalPresentationDecision", "round_id": "round", "call_ids": ["call"],
            "attachment_id": client._attachment_id, "visit": client._visit,
        }}, session_id=session.id, incarnation_id=session.incarnation_id)
        assert reader._parked_approval_payloads is observed
        await _read_head(client, session, None)
        assert reader._parked_approval_payloads is None
        with pytest.raises(ConsoleProjectionUnavailable, match="approval"):
            reader._head_round_payload(observed, session.id)
    finally:
        client._projection_lock.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "rollback", "admission", "negative"])
async def test_resolve_invalidates_old_and_inflight_head_until_fresh_retired_read(bound_reader, outcome):
    client, session = _client(*bound_reader)
    await _read_head(client, session, _approval())
    reader = client.controller_reader
    old = reader._parked_approval_payloads
    task, pending = await _read_head(client, session, _approval(), retire=False)
    resolution = client.submit("ResolveDecision", {"decision": {
        "kind": "ApprovalRoundDecision", "round_id": "round", "call_ids": ["call"],
        "decisions": {"call": "approve_once"},
    }}, session_id=session.id, incarnation_id=session.incarnation_id)
    assert reader._parked_approval_payloads is None
    _deliver(client, pending, "retired", {"operation_id": pending.identity.request_id, "retired": True})
    await task
    assert reader._parked_approval_payloads is None
    if outcome == "admission":
        _deliver(client, resolution, "failed", {"code": "command_refused", "stage": "admission"})
    else:
        _deliver(client, resolution, "received", {"sequence": resolution.identity.sequence})
        if outcome == "rollback":
            _deliver(client, resolution, "failed", {"code": "command_failed", "stage": "applied"})
        else:
            _deliver(client, resolution, "applied", {"applied": True})
        assert reader._parked_approval_payloads is None
        _deliver(client, resolution, "retired", {"operation_id": resolution.identity.request_id, "retired": outcome != "negative"})
    assert reader._parked_approval_payloads is None
    with pytest.raises(ConsoleProjectionUnavailable, match="approval"):
        reader._head_round_payload(old, session.id)
    assert client._decision_resolutions == (
        {resolution.identity.request_id} if outcome == "negative" else set()
    )
    await _read_head(client, session, None if outcome == "success" else _approval())
    assert (reader._parked_approval_payloads is None) == (outcome in {"success", "negative"})
    if outcome in {"rollback", "admission"}:
        with pytest.raises(ConsoleRemoteFailure):
            resolution.applied.result()
    await _retrieve(resolution)


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [None, "next-round"])
async def test_later_full_wins_over_an_older_head_still_awaiting_retirement(bound_reader, replacement):
    client, session = _client(*bound_reader)
    await _read_head(client, session, _approval())
    task, pending = await _read_head(client, session, _approval(), retire=False)
    decision = _approval() if replacement is not None else None
    if decision is not None:
        # A detached original next-round value; publication qualification is
        # modeled here, while the independent head uses actual transport above.
        from dataclasses import replace

        decision = replace(decision, decision_id=replacement,
                           payload={**decision.payload, "round_id": replacement})
    client.projection_view = _publication(session, decision=decision)
    client._projection = client.projection_view.projections[0]
    client._projection_sequence = pending.identity.sequence + 1
    client._decision_full_observation = (client._projection, client._decision_observation_context())
    reader = client.controller_reader
    assert reader._parked_approval_payloads is decision
    _deliver(client, pending, "retired", {"operation_id": pending.identity.request_id, "retired": True})
    await task
    assert reader._parked_approval_payloads is decision


@pytest.mark.asyncio
async def test_hint_and_later_full_clearing_do_not_resurrect_old_approval(bound_reader):
    client, session = _client(*bound_reader)
    await _read_head(client, session, _approval())
    reader = client.controller_reader
    task, receipt = await _read_head(client, session, _approval(), retire=False)
    # Same authenticated hint tuple used by the original receiver. This test
    # isolates observation invalidation; hint authentication has separate tests.
    client._decision_hint = (0, session.id, session.incarnation_id, client._attachment_id, 1, 1)
    assert reader._parked_approval_payloads is None
    _deliver(client, receipt, "retired", {"operation_id": receipt.identity.request_id, "retired": True})
    await task
    assert reader._parked_approval_payloads is None
    # Model a later, positively adopted FULL's existing observer metadata.
    # Neither this assignment nor the value fixture claims native execution.
    client._projection_sequence = client._decision_head_sequence + 1
    client._decision_full_observation = (client._projection, client._decision_observation_context())
    assert reader._parked_approval_payloads is None
    # A queued old head must not override that later FULL's explicit clear.
    client._decision_head_observation = (client.decision_head, client._decision_observation_context())
    assert reader._parked_approval_payloads is None


@pytest.mark.asyncio
async def test_later_admission_refusal_cannot_hide_an_earlier_unretired_resolution(bound_reader):
    client, session = _client(*bound_reader)
    await _read_head(client, session, _approval())
    payload = {"decision": {
        "kind": "ApprovalRoundDecision", "round_id": "round", "call_ids": ["call"],
        "decisions": {"call": "approve_once"},
    }}
    first = client.submit("ResolveDecision", payload, session_id=session.id, incarnation_id=session.incarnation_id)
    _deliver(client, first, "received", {"sequence": first.identity.sequence})
    second = client.submit("ResolveDecision", payload, session_id=session.id, incarnation_id=session.incarnation_id)
    _deliver(client, second, "failed", {"code": "command_refused", "stage": "admission"})
    assert client._decision_resolutions == {first.identity.request_id}
    assert client._decision_resolution_sequence == second.identity.sequence
    await _read_head(client, session, _approval())
    assert client.controller_reader._parked_approval_payloads is None
    _deliver(client, first, "applied", {"applied": True})
    assert client.controller_reader._parked_approval_payloads is None
    _deliver(client, first, "retired", {"operation_id": first.identity.request_id, "retired": True})
    assert not client._decision_resolutions
    assert client.controller_reader._parked_approval_payloads is None
    await _read_head(client, session, None)
    assert client.controller_reader._parked_approval_payloads is None
    await _retrieve(second)


@private_profile_test
@pytest.mark.asyncio
async def test_actual_adopted_full_is_invalidated_before_the_first_independent_head(request, bound_reader):
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import open_console_backend_profile
    from tldw_chatbook.Chat.console_execution_projection_result import encode_projection_payload
    from tldw_chatbook.Chat.console_execution_projection_values import encode_message_projection

    client, _ = _client(*bound_reader)
    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(expected_config_path=path, expected_data_directory=directory)
    backend = ConsoleExecutionBackend(profile, profile_epoch=client.profile_epoch, service_epoch=client.service_epoch)
    pending = None
    try:
        backend.start()
        session = backend.store.create_session(title="Actual approval observation")
        client.projection_view = _publication(session)
        client._projection = client.projection_view.projections[0]
        lifecycle = backend.presentation.view_lifecycle
        await lifecycle.prepare(lifecycle.admit(client._attachment_id, 1, "mount"))
        call = MCPPendingCall(
            llm_name="fs_read", server_key="local:__local__", tool_name="fs_read",
            server_label="Local tools", arguments={"path": "README.md"},
            reason="ask", options=("approve_once", "deny"), call_id="call",
        )
        host = backend.controller._interrupt_host
        pending = asyncio.create_task(asyncio.to_thread(host.request_mcp_approvals, [call], session_id=session.id))
        await _until(lambda: bool(host.payloads["approval"]))
        task = asyncio.create_task(client.read_projection(session.id, session.incarnation_id))
        await _until(lambda: bool(client._channel.messages))
        message = client._channel.messages[-1]
        parent_receipt = client._requests[message.identity.request_id]
        original = backend.dispatch(message)
        result = await original.applied
        await original.work
        assert await original.retired is True
        command = backend.adapter._commands[message.identity.request_id]
        body = encode_projection_payload(result, messages=[encode_message_projection(item) for item in result.messages])
        body["session_view"] = backend.presentation.projection_view(command.identity, result)
        _deliver(client, parent_receipt, "received", {"sequence": parent_receipt.identity.sequence})
        _deliver(client, parent_receipt, "applied", body)
        _deliver(client, parent_receipt, "retired", {"operation_id": parent_receipt.identity.request_id, "retired": True})
        await task
        assert client.decision_head is None
        observed = client.controller_reader._parked_approval_payloads
        assert observed is client._projection.decision and observed is not None
        client.submit("ResolveDecision", {"decision": {
            "kind": "ApprovalRoundDecision", "round_id": observed.decision_id,
            "call_ids": ["call"], "decisions": {"call": "approve_once"},
        }}, session_id=session.id, incarnation_id=session.incarnation_id)
        assert client.controller_reader._parked_approval_payloads is None
    finally:
        if pending is not None:
            # The standalone approval has no running turn's Stop event.
            # Retire its actual session round before joining the waiting worker.
            backend.controller._interrupt_host._cancel_pending_decisions_for_session(session.id)
            await asyncio.gather(pending, return_exceptions=True)
        await backend.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["visit", "source", "session", "incarnation", "attachment", "channel", "producer", "hidden"])
async def test_approval_head_never_crosses_original_scope_or_sources(bound_reader, monkeypatch, change):
    client, session = _client(*bound_reader)
    await _read_head(client, session, _approval())
    if change == "visit":
        client._visit += 1
    elif change == "source":
        client._source_revision += 1
    elif change == "attachment":
        client._attachment_id = "replacement"
    elif change in {"session", "incarnation"}:
        altered = ConsoleChatSession(id="other" if change == "session" else session.id)
        client.projection_view = _publication(altered)
        client._projection = client.projection_view.projections[0]
    elif change == "channel":
        client._channel_error = ConsoleChannelLost("test")
    elif change == "hidden":
        client._app._screen_stacks[client._app._current_mode].append(Screen())
    else:
        monkeypatch.setattr(client, "read_decision_head", lambda *_: None)
    if change in {"source", "channel", "producer"}:
        with pytest.raises(ConsoleProjectionUnavailable):
            _ = client.controller_reader._parked_approval_payloads
    elif change == "hidden":
        with pytest.raises(ConsoleChannelLost):
            _ = client.controller_reader._parked_approval_payloads
    else:
        assert client.controller_reader._parked_approval_payloads is None
