"""Authenticated tickets retain durable, truthful native command outcomes."""

import asyncio
import base64
import json
import threading
import time

import pytest

from Tests.agent_runs_owner import owned_agent_runs_db
from tldw_chatbook.Chat.console_remote_models import ConsoleHostIdentity, ConsoleTarget
from tldw_chatbook.Chat.console_remote_receipts import (
    ConsoleRemoteReceipts,
    ExpiredConsoleCommand,
)
from tldw_chatbook.DB.console_command_receipts import ConsoleCommandReceiptRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class RecordingCommands:
    def __init__(self):
        self.calls = 0
        self.result = {"state": "applied", "native_id": "native-a"}
        self.error = None

    async def execute(self, action, target, payload, command_id):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def receipt_owner(tmp_path):
    database = owned_agent_runs_db(tmp_path / "agent_runs.db")
    commands = RecordingCommands()
    clock = [1_000_000_000]
    host = ConsoleHostIdentity(server_epoch="epoch-a", console_id="console-a")
    owner = ConsoleRemoteReceipts(
        commands,
        ConsoleCommandReceiptRepository(database),
        host,
        monotonic_ns=lambda: clock[0],
    )

    def advance(seconds):
        clock[0] += int(seconds * 1_000_000_000)

    yield owner, commands, advance
    database.close()


def target():
    return ConsoleTarget(
        server_epoch="epoch-a",
        console_id="console-a",
        session_id="chat-a",
        incarnation_id="incarnation-a",
    )


async def test_duplicate_submission_is_not_reissued_after_expiry(receipt_owner):
    owner, commands, advance = receipt_owner
    payload = {"text": "one exact correction"}
    ticket = owner.issue("send", target(), payload)
    first = await owner.submit(ticket.token, payload)
    assert (await owner.submit(ticket.token, payload)).command_id == first.command_id
    advance(301)
    assert owner.query(first.command_id) is not None
    with pytest.raises(ExpiredConsoleCommand):
        await owner.submit(ticket.token, payload)
    assert commands.calls == 1


async def test_payload_change_and_unsigned_ticket_changes_never_dispatch(receipt_owner):
    owner, commands, _advance = receipt_owner
    ticket = owner.issue("send", target(), {"text": "secret"})
    with pytest.raises(ValueError):
        await owner.submit(ticket.token, {"text": "changed"})
    encoded, signature = ticket.token.split(".")
    original = json.loads(base64.urlsafe_b64decode(encoded))
    for key, value in [
        ("expires_ns", 999999999999999999),
        ("action", "stop"),
        ("target", owner.host_identity.model_dump()),
    ]:
        changed = dict(original, **{key: value})
        forged = base64.urlsafe_b64encode(json.dumps(changed).encode()).decode()
        with pytest.raises(ValueError):
            await owner.submit(f"{forged}.{signature}", {"text": "secret"})
    assert commands.calls == 0


async def test_prior_console_or_server_epoch_ticket_is_invalid(receipt_owner):
    owner, commands, _advance = receipt_owner
    ticket = owner.issue("send", target(), {})
    for host in [
        ConsoleHostIdentity(server_epoch="epoch-b", console_id="console-a"),
        ConsoleHostIdentity(server_epoch="epoch-a", console_id="console-b"),
    ]:
        successor = ConsoleRemoteReceipts(commands, owner.repository, host)
        with pytest.raises(ValueError):
            await successor.submit(ticket.token, {})
    assert commands.calls == 0


async def test_interrupted_handoff_is_unknown_and_never_replayed(receipt_owner):
    owner, commands, _advance = receipt_owner
    ticket = owner.issue("send", target(), {})
    commands.error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await owner.submit(ticket.token, {})
    assert owner.query(ticket.command_id).state == "outcome_unknown"
    commands.error = None
    assert (await owner.submit(ticket.token, {})).state == "outcome_unknown"
    assert commands.calls == 1


async def test_existing_pending_claim_does_not_grant_dispatch(receipt_owner):
    owner, commands, _advance = receipt_owner
    ticket = owner.issue("send", target(), {})
    commands.result = {"state": "pending", "native_id": ticket.command_id}
    first = await owner.submit(ticket.token, {})
    assert first.state == "pending"
    assert (await owner.submit(ticket.token, {})).state == "pending"
    assert commands.calls == 1


async def test_capacity_refuses_effect_but_exact_stop_remains_available(
    receipt_owner, monkeypatch
):
    from tldw_chatbook.DB import console_command_receipts as module

    monkeypatch.setattr(module, "MAX_COMMAND_RECEIPTS", 1)
    owner, commands, _advance = receipt_owner
    first = owner.issue("send", target(), {})
    await owner.submit(first.token, {})
    second = owner.issue("send", target(), {})
    with pytest.raises(ValueError, match="capacity"):
        await owner.submit(second.token, {})
    stop = owner.issue("stop", target().model_copy(update={"run_id": "run-a"}), {})
    commands.result = {"state": "accepted", "native_id": "run-a"}
    assert (await owner.submit(stop.token, {})).state == "accepted"
    assert commands.calls == 2


async def test_receipt_body_and_payload_never_persist(receipt_owner):
    owner, _commands, _advance = receipt_owner
    payload = {"text": "DO NOT STORE THIS BODY"}
    ticket = owner.issue("send", target(), payload)
    receipt = await owner.submit(ticket.token, payload)
    assert "DO NOT STORE" not in repr(receipt)
    with owner.repository.runs_db.connection() as connection:
        row = connection.execute("SELECT * FROM console_command_receipts").fetchone()
        assert "DO NOT STORE" not in repr(tuple(row))


async def test_before_claim_fault_has_no_effect_and_can_retry(
    receipt_owner, monkeypatch
):
    owner, commands, _advance = receipt_owner
    ticket = owner.issue("send", target(), {})
    original = owner.repository.claim

    def interrupted(_receipt):
        raise OSError("before claim")

    monkeypatch.setattr(owner.repository, "claim", interrupted)
    with pytest.raises(OSError):
        await owner.submit(ticket.token, {})
    assert commands.calls == 0 and owner.query(ticket.command_id) is None
    monkeypatch.setattr(owner.repository, "claim", original)
    assert (await owner.submit(ticket.token, {})).state == "applied"
    assert commands.calls == 1


async def test_lost_ack_queries_committed_outcome_after_restart(receipt_owner):
    owner, commands, _advance = receipt_owner
    ticket = owner.issue("send", target(), {})
    await owner.submit(ticket.token, {})
    successor = ConsoleRemoteReceipts(
        commands,
        owner.repository,
        ConsoleHostIdentity(server_epoch="new-server", console_id="new-console"),
    )
    assert successor.query(ticket.command_id).state == "applied"
    with pytest.raises(ValueError):
        await successor.submit(ticket.token, {})
    assert commands.calls == 1


async def test_prior_owner_pending_is_discoverable_unknown(receipt_owner):
    owner, commands, _advance = receipt_owner
    ticket = owner.issue("send", target(), {})
    commands.result = {"state": "pending"}
    await owner.submit(ticket.token, {})
    successor = ConsoleRemoteReceipts(
        commands,
        owner.repository,
        ConsoleHostIdentity(server_epoch="new-server", console_id="new-console"),
    )
    assert successor.query(ticket.command_id).state == "outcome_unknown"
    assert (
        successor.repository.list_unresolved().items[0].command_id == ticket.command_id
    )
    assert commands.calls == 1


async def test_native_acceptance_wins_over_lost_transport_ack(receipt_owner):
    owner, commands, _advance = receipt_owner
    ticket = owner.issue("send", target(), {})

    async def accepted_then_interrupted(action, target, payload, command_id):
        commands.calls += 1
        owner.observe(command_id, state="accepted", native_id=command_id)
        raise asyncio.CancelledError()

    commands.execute = accepted_then_interrupted
    with pytest.raises(asyncio.CancelledError):
        await owner.submit(ticket.token, {})
    assert owner.query(ticket.command_id).state == "accepted"
    assert (await owner.submit(ticket.token, {})).state == "accepted"
    assert commands.calls == 1


async def test_full_capacity_stop_expiry_and_query_are_honest(
    receipt_owner, monkeypatch
):
    from tldw_chatbook.DB import console_command_receipts as module

    monkeypatch.setattr(module, "MAX_COMMAND_RECEIPTS", 0)
    owner, commands, advance = receipt_owner
    stop = owner.issue("stop", target().model_copy(update={"run_id": "run-a"}), {})
    commands.result = {"state": "accepted", "native_id": "run-a"}
    result = await owner.submit(stop.token, {})
    assert (
        result.reason == "stop_requested_exit_unconfirmed_receipt_unavailable_capacity"
    )
    assert owner.query(stop.command_id) is None
    with owner.repository.runs_db.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM console_command_receipts"
            ).fetchone()[0]
            == 0
        )
    advance(300)
    with pytest.raises(ExpiredConsoleCommand):
        await owner.submit(stop.token, {})
    assert commands.calls == 1


# Reuse the native, admitted configuration fixture rather than replacing Send.
from Tests.Chat import test_console_received_intent_custody as custody_controls
from Tests.Chat import test_console_received_intent_transitions as native_controls

catalog_store = native_controls.catalog_store
local_root = native_controls.local_root
mcp_sources = native_controls.mcp_sources
snapshot_case = native_controls.snapshot_case
runtime_case = native_controls.runtime_case
configuration_workspace = native_controls.configuration_workspace
received_workspace = native_controls.received_workspace
intent_case = custody_controls.intent_case


async def test_native_send_preparing_acceptance_queue_and_no_duplicate_dispatch(
    receipt_owner,
    received_workspace,
    tmp_path,
):
    case = received_workspace
    gateway = native_controls._configure_command_case(case, tmp_path)
    commands, native_target, revisions = native_controls._command_owner(case)
    owner = ConsoleRemoteReceipts(
        commands, receipt_owner[0].repository, commands.host_identity
    )
    case.runtime._remote_receipts = owner
    payload = {"text": "first phone input", "expected_revisions": revisions}
    ticket = owner.issue("send", native_target, payload)
    first = await owner.submit(ticket.token, payload)
    assert first.state == "pending" and first.reason == "Preparing"
    record = case.runtime._turn_custody[ticket.command_id]
    try:
        await native_controls._wait_for_seam(record, gateway.started[0])
        assert owner.query(ticket.command_id).state == "accepted"
        assert (await owner.submit(ticket.token, payload)).state == "accepted"
        queued_payload = {"text": "second phone input", "expected_revisions": revisions}
        queued_ticket = owner.issue("send", native_target, queued_payload)
        queued_receipt = await owner.submit(queued_ticket.token, queued_payload)
        assert queued_receipt.state == "pending"
        queued_record = case.runtime._turn_custody[queued_ticket.command_id]
        outcome = await native_controls._outcome(queued_record)
        assert outcome.applied
        assert owner.query(queued_ticket.command_id).state == "queued"
        assert (
            await owner.submit(queued_ticket.token, queued_payload)
        ).state == "queued"
        assert gateway.user_turns == ["first phone input"]
        for event in gateway.release:
            event.set()
        await native_controls._outcome(record)
        assert gateway.user_turns == ["first phone input", "second phone input"]
        assert owner.query(queued_ticket.command_id).state == "accepted"
        assert owner.query(ticket.command_id).state == "accepted"
    finally:
        for event in gateway.release:
            event.set()


async def test_native_preparing_stop_at_capacity_is_repeatable_without_rows(
    receipt_owner,
    intent_case,
    monkeypatch,
):
    from Tests.Chat.test_console_remote_commands import _commands
    from tldw_chatbook.DB import console_command_receipts as module

    monkeypatch.setattr(module, "MAX_COMMAND_RECEIPTS", 0)
    runtime, store, session, _request = intent_case
    commands, native_target = _commands(intent_case)
    owner = ConsoleRemoteReceipts(
        commands, receipt_owner[0].repository, commands.host_identity
    )
    intent = custody_controls._intent(intent_case)
    runtime.accept_received_intent(intent)
    record = runtime._turn_custody[intent.turn_id]
    exact = native_target.model_copy(update={"turn_id": intent.turn_id})
    ticket = owner.issue("stop", exact, {})
    before = store.message_count(session.id)
    first = await owner.submit(ticket.token, {})
    second = await owner.submit(ticket.token, {})
    assert first.state == second.state == "accepted"
    assert record.task.cancelling()
    assert store.message_count(session.id) == before
    assert owner.query(ticket.command_id) is None
    with owner.repository.runs_db.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM console_command_receipts"
            ).fetchone()[0]
            == 0
        )


async def test_native_streaming_stop_at_capacity_does_not_duplicate_history(
    receipt_owner,
    received_workspace,
    tmp_path,
    monkeypatch,
):
    from tldw_chatbook.DB import console_command_receipts as module

    case = received_workspace
    gateway = native_controls._configure_command_case(case, tmp_path)
    commands, native_target, revisions = native_controls._command_owner(case)
    owner = ConsoleRemoteReceipts(
        commands, receipt_owner[0].repository, commands.host_identity
    )
    case.runtime._remote_receipts = owner
    payload = {"text": "one real turn", "expected_revisions": revisions}
    ticket = owner.issue("send", native_target, payload)
    await owner.submit(ticket.token, payload)
    record = case.runtime._turn_custody[ticket.command_id]
    try:
        await native_controls._wait_for_seam(record, gateway.started[0])
        monkeypatch.setattr(module, "MAX_COMMAND_RECEIPTS", 1)
        stop = owner.issue(
            "stop", native_target.model_copy(update={"turn_id": ticket.command_id}), {}
        )
        assert (await owner.submit(stop.token, {})).state == "accepted"
        count = case.store.message_count(case.session.id)
        assert (await owner.submit(stop.token, {})).state in {"accepted", "refused"}
        assert case.store.message_count(case.session.id) == count
        assert gateway.user_turns == ["one real turn"]
        assert owner.query(stop.command_id) is None
        with owner.repository.runs_db.connection() as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM console_command_receipts"
                ).fetchone()[0]
                == 1
            )
    finally:
        for event in gateway.release:
            event.set()
        await native_controls._outcome(record)


async def test_native_subagent_stop_at_capacity_is_exact_and_does_not_spawn(
    receipt_owner,
    intent_case,
    tmp_path,
    monkeypatch,
):
    import threading

    from Tests.Agents.test_primary_redirect import _service
    from Tests.Chat.test_console_agent_bridge_steering import (
        _bridge_with_fleet,
        _live_child,
    )
    from Tests.Chat.test_console_remote_commands import _commands
    from tldw_chatbook.DB import console_command_receipts as module

    monkeypatch.setattr(module, "MAX_COMMAND_RECEIPTS", 0)
    runtime, store, session, _request = intent_case
    commands, native_target = _commands(intent_case)
    owner = ConsoleRemoteReceipts(
        commands, receipt_owner[0].repository, commands.host_identity
    )
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    runtime._chat_controller._agent_bridge = bridge
    session.persisted_conversation_id = "conv-1"
    handle = _live_child(coordinator, run_id="child-run")
    service = _service()
    service._fleet = coordinator
    cancelled = threading.Event()
    service._fleet_cancels[handle] = cancelled
    service._revoke_approvals = lambda _run: None
    bridge._fleet_services["conv-1"] = service
    exact = native_target.model_copy(
        update={"handle_id": handle, "run_id": "child-run"}
    )
    ticket = owner.issue("subagent_stop", exact, {})
    before = store.message_count(session.id)
    assert (await owner.submit(ticket.token, {})).state == "accepted"
    assert (await owner.submit(ticket.token, {})).state == "accepted"
    assert cancelled.is_set() and coordinator.get(handle).status == "running"
    assert len(bridge.fleet_snapshot("conv-1")) == 1
    assert store.message_count(session.id) == before
    assert owner.query(ticket.command_id) is None
    stale = owner.issue(
        "subagent_stop", exact.model_copy(update={"run_id": "successor"}), {}
    )
    assert (await owner.submit(stale.token, {})).state == "refused"
    bridge._fleet_services.clear()


async def test_private_child_ticket_submit_query_and_discovery(
    receipt_owner, intent_case
):
    from types import SimpleNamespace

    from tldw_chatbook import app_served
    from tldw_chatbook.Canvas.control_protocol import (
        CONTROL_PROTOCOL_VERSION,
        ControlMessage,
        encode_control_frame,
    )
    from tldw_chatbook.Chat.console_remote_projection import ConsoleRemoteProjection
    from tldw_chatbook.Web_Server.served_console_events import ConsoleEventBuffer

    runtime, _store, _session, _request = intent_case
    owner, commands, _advance = receipt_owner
    runtime._agent_runs_db = owner.repository.runs_db
    runtime._remote_receipts = owner
    runtime._remote_projection = ConsoleRemoteProjection(
        runtime, owner.host_identity, ConsoleEventBuffer()
    )
    app = SimpleNamespace(
        console_runtime=runtime,
        served_canvas_control=SimpleNamespace(host_identity=owner.host_identity),
    )

    runtime._app = app

    async def request(method, data):
        response = await app_served._handle_served_control_request(
            app,
            ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                f"console.{method}.request",
                "correlation",
                None,
                {"request": data},
            ),
        )
        assert response.request_id == "correlation"
        encode_control_frame(response)
        return response.payload

    ticket = (
        await request(
            "ticket", {"action": "send", "target": target().model_dump(), "payload": {}}
        )
    )["ticket"]
    commands.result = {"state": "pending"}
    receipt = (await request("command", {"ticket": ticket["token"], "payload": {}}))[
        "receipt"
    ]
    assert receipt["command_id"] == ticket["command_id"]
    assert (await request("receipt", {"command_id": ticket["command_id"]}))["receipt"][
        "state"
    ] == "pending"
    snapshot = (await request("snapshot", {}))["snapshot"]
    assert snapshot["command_receipts"][0]["command_id"] == ticket["command_id"]
    assert snapshot["next_receipt_cursor"] is None
    assert (await request("receipt", {"command_id": "unavailable"}))["receipt"] is None
    runtime._agent_runs_db = None


from Tests.Chat import test_console_remote_projection as projection_controls

remote_case = projection_controls.remote_case
remote_round = projection_controls.remote_round


async def test_receipt_records_native_decision_once_and_keeps_raw_shell_pending(
    receipt_owner, remote_round
):
    projection, host, exact = remote_round
    owner = ConsoleRemoteReceipts(
        projection._commands, receipt_owner[0].repository, projection.host_identity
    )
    projection.runtime._remote_receipts = owner
    payload = {"kind": "approval", "answer": {"decisions": {"write_file": "deny"}}}
    with host.lock:
        call = host.payloads["approval"][exact.round_id]["calls"][0]
        call.update(
            server_key="local:__local__",
            tool_name="shell_exec",
            full_command="private body",
        )
    refused = owner.issue("decision", exact, payload)
    assert (await owner.submit(refused.token, payload)).state == "refused"
    assert not host.registries["approval"][exact.round_id]["event"].is_set()
    call.update(tool_name="write_file", full_command=None)
    ticket = owner.issue("decision", exact, payload)
    receipt = await owner.submit(ticket.token, payload)
    assert receipt.state == "accepted" and receipt.reason == "answer_recorded"
    assert (await owner.submit(ticket.token, payload)).state in {"accepted", "applied"}
    for _index in range(100):
        if owner.query(ticket.command_id).state == "applied":
            break
        await asyncio.sleep(0.01)
    assert owner.query(ticket.command_id).state == "applied"
    loser = owner.issue("decision", exact, payload)
    assert (await owner.submit(loser.token, payload)).state == "refused"
    with pytest.raises(ValueError):
        await owner.submit(
            ticket.token,
            {
                "kind": "approval",
                "answer": {"decisions": {"write_file": "approve_once"}},
            },
        )


async def test_snapshot_discovers_bounded_prior_epoch_receipts(
    receipt_owner, remote_case
):
    owner, commands, _advance = receipt_owner
    commands.result = {"state": "pending"}
    for _index in range(25):
        ticket = owner.issue("send", target(), {})
        await owner.submit(ticket.token, {})
    projection, _host, _target = remote_case
    successor = ConsoleRemoteReceipts(
        commands, owner.repository, projection.host_identity
    )
    projection.runtime._remote_receipts = successor
    first = projection.snapshot(None)
    assert len(first["command_receipts"]) == 20 and first["next_receipt_cursor"]
    assert all(row["state"] == "outcome_unknown" for row in first["command_receipts"])
    second = projection.snapshot(None, receipt_cursor=first["next_receipt_cursor"])
    assert (
        len(second["command_receipts"]) == 5 and second["next_receipt_cursor"] is None
    )
    assert (
        len(
            {
                row["command_id"]
                for row in first["command_receipts"] + second["command_receipts"]
            }
        )
        == 25
    )


@pytest.mark.parametrize("boundary", ["decided", "timeout", "cancelled", "interrupted"])
async def test_native_decision_final_boundary_controls_receipt(
    receipt_owner, remote_case, boundary
):
    projection, host, exact = remote_case
    owner = ConsoleRemoteReceipts(
        projection._commands, receipt_owner[0].repository, projection.host_identity
    )
    projection.runtime._remote_receipts = owner
    exact = exact.model_copy(update={"round_id": "question-race", "run_id": "run-race"})
    ready, release, cancelled = threading.Event(), threading.Event(), threading.Event()
    state = {
        "event": threading.Event(),
        "decision": {},
        "cancel_event": cancelled,
        "visit_event": threading.Event(),
        "session_id": exact.session_id,
        "run_id": exact.run_id,
    }
    payload = {
        "request_id": exact.round_id,
        "session_id": exact.session_id,
        "questions": [
            {
                "question": "Pick",
                "header": "Choice",
                "options": [{"label": "A"}, {"label": "B"}],
            }
        ],
    }
    native = []
    deadline = time.monotonic() + (0.3 if boundary == "timeout" else 30)

    def before_wait():
        ready.set()
        assert release.wait(3)

    def outcome(value):
        if boundary == "interrupted":
            raise OSError("native outcome callback interrupted")
        native.append(value)

    def request():
        try:
            host.run_round(
                "question",
                exact.round_id,
                payload,
                state,
                session_id=exact.session_id,
                owning_session_id=exact.session_id,
                deadline=None,
                hard_deadline=deadline,
                is_parked=False,
                before_wait=before_wait,
                on_outcome=outcome,
            )
        except OSError:
            native.append("interrupted")

    worker = threading.Thread(target=request, daemon=True)
    worker.start()
    try:
        assert await asyncio.to_thread(ready.wait, 3)
        answer = {
            "kind": "question",
            "answer": {"answers": [{"question": "Pick", "selected": ["A"]}]},
        }
        ticket = owner.issue("decision", exact, answer)
        accepted = await owner.submit(ticket.token, answer)
        assert accepted.state == "accepted" and accepted.reason == "answer_recorded"
        other = owner.issue(
            "decision",
            exact,
            {
                "kind": "question",
                "answer": {"answers": [{"question": "Pick", "selected": ["B"]}]},
            },
        )
        assert (
            await owner.submit(
                other.token,
                {
                    "kind": "question",
                    "answer": {"answers": [{"question": "Pick", "selected": ["B"]}]},
                },
            )
        ).state == "refused"
        assert state["decision"]["answers"][0]["selected"] == ["A"]
        if boundary == "timeout":
            await asyncio.sleep(max(0, deadline - time.monotonic()) + 0.02)
        elif boundary == "cancelled":
            cancelled.set()
    finally:
        release.set()
        await asyncio.to_thread(worker.join, 3)
        assert not worker.is_alive()
    await asyncio.sleep(
        0
    )  # Receive the worker's final native outcome on the child loop.
    expected = {
        "decided": "applied",
        "timeout": "refused",
        "cancelled": "refused",
        "interrupted": "outcome_unknown",
    }[boundary]
    assert owner.query(ticket.command_id).state == expected
    assert native == [boundary]
    assert (await owner.submit(ticket.token, answer)).state == expected
    assert exact.round_id not in host.registries["question"]


async def test_native_settlement_loser_after_validation_cannot_claim_winner(
    receipt_owner, remote_case, monkeypatch
):
    projection, host, exact = remote_case
    owner = ConsoleRemoteReceipts(
        projection._commands, receipt_owner[0].repository, projection.host_identity
    )
    projection.runtime._remote_receipts = owner
    exact = exact.model_copy(
        update={"round_id": "question-winner", "run_id": "run-winner"}
    )
    state = {
        "event": threading.Event(),
        "decision": {},
        "session_id": exact.session_id,
        "run_id": exact.run_id,
    }
    host.register_round("question", exact.round_id, state)
    host.payloads["question"][exact.round_id] = {
        "questions": [
            {
                "question": "Pick",
                "header": "Choice",
                "options": [{"label": "A"}, {"label": "B"}],
            }
        ]
    }
    original = host.resolve_pending_question

    def concurrent_desktop_winner(answers, *, request_id, _on_recorded=None):
        original([{"question": "Pick", "selected": ["B"]}], request_id=request_id)
        original(answers, request_id=request_id, _on_recorded=_on_recorded)

    monkeypatch.setattr(host, "resolve_pending_question", concurrent_desktop_winner)
    answer = {
        "kind": "question",
        "answer": {"answers": [{"question": "Pick", "selected": ["A"]}]},
    }
    ticket = owner.issue("decision", exact, answer)
    assert (await owner.submit(ticket.token, answer)).state == "refused"
    assert state["decision"]["answers"][0]["selected"] == ["B"]
    assert "remote_receipt_outcome" not in state


async def test_preclaim_pruning_recovers_only_expired_completed_capacity(
    receipt_owner, monkeypatch
):
    from tldw_chatbook.DB import console_command_receipts as module

    monkeypatch.setattr(module, "MAX_COMMAND_RECEIPTS", 1)
    owner, commands, advance = receipt_owner
    first = owner.issue("send", target(), {})
    await owner.submit(first.token, {})
    advance(300)
    second = owner.issue("send", target(), {})
    assert (await owner.submit(second.token, {})).state == "applied"
    assert owner.query(first.command_id) is None
    with pytest.raises(ExpiredConsoleCommand):
        await owner.submit(first.token, {})
    assert commands.calls == 2


@pytest.mark.parametrize("state", ["accepted", "queued", "applied", "refused"])
async def test_duplicate_crossing_expiry_during_retention_never_dispatches_again(
    receipt_owner, state
):
    owner, commands, _advance = receipt_owner
    commands.result = {"state": state, "native_id": "native-a"}
    ticket = owner.issue("send", target(), {})
    original = await owner.submit(ticket.token, {})
    readings = iter([original.expires_ns - 1, original.expires_ns])
    owner.monotonic_ns = lambda: next(readings, original.expires_ns)
    assert await owner.submit(ticket.token, {}) == original
    assert owner.query(ticket.command_id) == original
    assert commands.calls == 1


@pytest.mark.parametrize("full_stop", [False, True])
async def test_new_effect_crossing_expiry_inside_claim_is_refused(
    receipt_owner, monkeypatch, full_stop
):
    from tldw_chatbook.DB import console_command_receipts as module

    owner, commands, _advance = receipt_owner
    if full_stop:
        monkeypatch.setattr(module, "MAX_COMMAND_RECEIPTS", 0)
    exact = target().model_copy(update={"run_id": "run-a"})
    ticket = owner.issue("stop" if full_stop else "send", exact, {})
    now = [1_000_000_000]
    owner.monotonic_ns = lambda: now[0]
    original_claim = owner.repository.claim

    def claim_crossing_deadline(receipt):
        try:
            return original_claim(receipt)
        finally:
            now[0] = receipt.expires_ns

    monkeypatch.setattr(owner.repository, "claim", claim_crossing_deadline)
    with pytest.raises(ExpiredConsoleCommand):
        await owner.submit(ticket.token, {})
    assert commands.calls == 0
    stored = owner.query(ticket.command_id)
    if full_stop:
        assert stored is None
    else:
        assert stored.state == "refused" and stored.reason == "command_expired"
    with pytest.raises(ExpiredConsoleCommand):
        await owner.submit(ticket.token, {})
    assert commands.calls == 0


async def test_prior_owner_unknown_discovery_preserves_current_native_decision(
    receipt_owner, remote_round
):
    old, commands, _advance = receipt_owner
    commands.result = {"state": "pending"}
    pending = old.issue("send", target(), {})
    assert (await old.submit(pending.token, {})).state == "pending"
    projection, host, exact = remote_round
    successor = ConsoleRemoteReceipts(
        projection._commands, old.repository, projection.host_identity
    )
    projection.runtime._remote_receipts = successor
    discovered = projection.snapshot(None)["command_receipts"]
    assert len(discovered) == 1
    assert discovered[0]["command_id"] == pending.command_id
    assert discovered[0]["state"] == "outcome_unknown"
    assert discovered[0]["target"] == target().model_dump()
    assert commands.calls == 1
    projection.register_client("current-phone")
    with pytest.raises(ValueError):
        projection.set_answerable(
            "current-phone",
            exact.model_copy(update={"server_epoch": "epoch-a"}),
            answerable=True,
        )
    projection.set_answerable("current-phone", exact, answerable=True)
    assert any(
        row["answerable"] and row["round_id"] == exact.round_id
        for row in host.remote_rounds(exact.session_id)
    )
    current = successor.issue(
        "decision",
        exact,
        {"kind": "approval", "answer": {"decisions": {"write_file": "deny"}}},
    )
    result = await successor.submit(
        current.token,
        {"kind": "approval", "answer": {"decisions": {"write_file": "deny"}}},
    )
    assert result.state in {"accepted", "applied"}
    assert successor.query(pending.command_id).state == "outcome_unknown"
    assert commands.calls == 1
    with pytest.raises(ValueError):
        await successor.submit(pending.token, {})
