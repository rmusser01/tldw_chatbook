"""Retained Console projections never borrow a mounted chat screen."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_runtime_lifetime import _pending_call, _ThreadApp
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_remote_models import ConsoleHostIdentity, ConsoleTarget
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Web_Server.served_console_events import ConsoleEventBuffer

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def remote_case():
    from tldw_chatbook.Chat.console_remote_projection import ConsoleRemoteProjection

    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    controller.app = _ThreadApp()
    runtime = ConsoleRuntime(_ThreadApp())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    identity = ConsoleHostIdentity(server_epoch="epoch", console_id="console")
    projection = ConsoleRemoteProjection(runtime, identity, ConsoleEventBuffer(limit=2))
    target = ConsoleTarget(
        **identity.model_dump(),
        session_id=session.id,
        incarnation_id=session.incarnation_id,
    )
    yield projection, controller._interrupt_host, target
    projection.close()


@pytest.fixture
def remote_round(remote_case):
    projection, host, target = remote_case
    controller = projection.runtime.chat_controller
    store = projection.runtime.chat_store
    store.create_session(ephemeral=True)
    host.retain_decision_target(target.session_id)
    controller.mcp_approval_timeout_seconds = lambda: 60.0
    outcome = {}

    def request():
        with use_run_id("run-a"):
            outcome.update(
                controller.request_mcp_approvals(
                    [_pending_call()], session_id=target.session_id
                )
            )

    worker = threading.Thread(target=request, daemon=True)
    worker.start()
    deadline = time.monotonic() + 3
    round_id = None
    while time.monotonic() < deadline:
        with host.lock:
            round_id = next(iter(host.payloads["approval"]), None)
        if round_id:
            break
        time.sleep(0.01)
    assert round_id, "native approval round did not arm"
    target = target.model_copy(update={"round_id": round_id, "run_id": "run-a"})
    try:
        yield projection, host, target
    finally:
        controller.resolve_pending_approval({"write_file": "deny"}, round_id=round_id)
        worker.join(3)
        assert not worker.is_alive()


def test_background_round_remains_answerable_for_the_other_phone(remote_round):
    projection, host, target = remote_round
    projection.register_client("phone-a")
    projection.register_client("phone-b")
    projection.set_answerable("phone-a", target, answerable=True)
    projection.set_answerable("phone-b", target, answerable=True)
    projection.disconnect_client("phone-a")
    row = next(
        row
        for row in host.remote_rounds(target.session_id)
        if row["round_id"] == target.round_id
    )
    assert row["answerable"] and row["supported"]
    assert projection.runtime.view is None


def test_metadata_never_contains_private_instruction_or_hidden_thinking(remote_round):
    projection, host, target = remote_round
    store = projection.runtime.chat_store
    message = store.append_message(
        target.session_id, role="assistant", content="Visible reply"
    )
    store._message_or_raise(
        message.id
    ).opaque_thinking_json = "HIDDEN-THINKING-SENTINEL"
    with host.lock:
        host.payloads["approval"][target.round_id]["automatic_instructions"] = (
            "AUTO-BODY-SENTINEL"
        )
    snapshot = projection.snapshot(target)
    encoded = json.dumps(snapshot)
    assert "AUTO-BODY-SENTINEL" not in encoded
    assert "HIDDEN-THINKING-SENTINEL" not in encoded
    assert "Visible reply" in encoded
    assert snapshot["decisions"][0]["round_id"] == target.round_id


def test_long_text_continuation_and_replacement_reset(remote_case):
    projection, _host, target = remote_case
    store = projection.runtime.chat_store
    body = "<script>inert</script>😺" * 6000
    message = store.append_message(target.session_id, role="assistant", content=body)
    snapshot = projection.snapshot(target, limit=100)
    assert len(json.dumps(snapshot).encode()) <= 65536
    pieces = [snapshot["messages"][0]["content"]]
    cursor = snapshot["next_cursor"]
    first_cursor = cursor
    while cursor:
        snapshot = projection.snapshot(target, cursor=cursor, limit=100)
        assert len(json.dumps(snapshot).encode()) <= 65536
        pieces.extend(row["content"] for row in snapshot["messages"])
        cursor = snapshot["next_cursor"]
    assert "".join(pieces) == body
    store.update_message_content(message.id, "replacement")
    snapshot = projection.snapshot(target, cursor=first_cursor)
    assert snapshot["reset_required"]
    assert snapshot["messages"][0]["content"] == "replacement"


def test_stale_target_fails_before_any_capture(remote_case, monkeypatch):
    projection, _host, target = remote_case
    monkeypatch.setattr(
        projection.runtime.chat_store,
        "read_only_messages_for_session",
        lambda *_: pytest.fail("stale target reached transcript"),
    )
    with pytest.raises(ValueError):
        projection.snapshot(target.model_copy(update={"incarnation_id": "old"}))


def test_capture_does_not_acknowledge_concurrent_event(remote_case, monkeypatch):
    projection, _host, target = remote_case
    events = projection.events
    events.publish("console.changed", {})
    events.publish("console.changed", {})
    events.publish("console.changed", {})
    original = projection.runtime.chat_store.read_only_messages_for_session

    def capture(session_id):
        events.publish("console.changed", {})
        return original(session_id)

    monkeypatch.setattr(
        projection.runtime.chat_store, "read_only_messages_for_session", capture
    )
    snapshot = projection.snapshot(target)
    assert snapshot["sequence"] == 3
    assert snapshot["reset_required"]
    assert projection.subscribe(3).reset_required
    assert projection.subscribe(3).events[-1]["sequence"] == 4


def test_approval_summary_preserves_destination_and_redacts_keys_and_values(
    remote_round,
):
    projection, host, target = remote_round
    secret = "sk-live-" + "A" * 30
    with host.lock:
        call = host.payloads["approval"][target.round_id]["calls"][0]
        call["arguments"] = {
            "content": "bulk" * 100,
            "path": "/notes/a.md",
            "api_key": "SECRET-KEY-SENTINEL",
            "note": secret,
        }
    row = projection.snapshot(target)["decisions"][0]
    assert row["supported"]
    summary = row["calls"][0]["arguments_summary"]
    assert "/notes/a.md" in summary
    assert "SECRET-KEY-SENTINEL" not in json.dumps(row)
    assert secret not in json.dumps(row)
    assert "arguments" not in row["calls"][0]


def test_raw_shell_and_redaction_errors_stay_pending(remote_round, monkeypatch):
    projection, host, target = remote_round
    with host.lock:
        call = host.payloads["approval"][target.round_id]["calls"][0]
        call.update(
            server_key="local:__local__",
            tool_name="shell_exec",
            full_command="secret command",
        )
    row = projection.snapshot(target)["decisions"][0]
    assert not row["supported"] and row["action"] == "open_terminal"
    assert not host.registries["approval"][target.round_id]["event"].is_set()
    call.update(tool_name="write_file", full_command=None)
    monkeypatch.setattr(
        "tldw_chatbook.Chat.approval_review.redact_mapping",
        lambda *_: (_ for _ in ()).throw(ValueError("secret failure")),
    )
    row = projection.snapshot(target)["decisions"][0]
    assert not row["supported"] and "secret failure" not in json.dumps(row)


@pytest.mark.asyncio
async def test_exact_remote_decision_reaches_original_round_once(remote_round):
    projection, host, target = remote_round
    commands = projection._commands
    wrong = target.model_copy(update={"run_id": "other-run"})
    assert (
        await commands.execute(
            "decision",
            wrong,
            {"kind": "approval", "answer": {"decisions": {"write_file": "deny"}}},
            "wrong",
        )
    )["state"] == "refused"
    with host.lock:
        state = host.registries["approval"][target.round_id]
    result = await commands.execute(
        "decision",
        target,
        {"kind": "approval", "answer": {"decisions": {"write_file": "deny"}}},
        "first",
    )
    assert result["state"] == "resolved"
    await commands.execute(
        "decision",
        target,
        {"kind": "approval", "answer": {"decisions": {"write_file": "approve_once"}}},
        "second",
    )
    assert state["decisions"] == {"write_file": "deny"}


def test_snapshot_does_not_grant_unregistered_client_answerability(remote_round):
    projection, host, target = remote_round
    projection.snapshot(target)
    with pytest.raises(ValueError, match="authenticated"):
        projection.set_answerable("unknown", target, answerable=True)
    assert not host.remote_rounds(target.session_id)[0]["answerable"]


def test_poll_tracks_background_stream_without_mounting_or_switching(remote_case):
    projection, _host, target = remote_case
    store = projection.runtime.chat_store
    desktop = store.create_session(ephemeral=True)
    first = projection.poll_changes()
    store.append_message(
        target.session_id, role="assistant", content="background content"
    )
    second = projection.poll_changes()
    assert second > first
    assert projection.poll_changes() is None
    assert store.active_session_id == desktop.id
    assert projection.runtime.view is None


def test_discovery_deduplicates_saved_live_and_pages_canonical_service(remote_case):
    projection, _host, _target = remote_case
    store = projection.runtime.chat_store
    store.sessions()[0].persisted_conversation_id = "saved-0"
    calls = []

    def listing(*, limit, offset, archive_scope):
        calls.append((limit, offset))
        return {
            "items": [
                {"id": f"saved-{i}", "title": f"Chat {i}", "version": 1}
                for i in range(offset, min(130, offset + limit))
            ],
            "total": 130,
        }

    projection.runtime.app.local_chat_conversation_service = SimpleNamespace(
        list_library_conversations=listing
    )
    snapshot = projection.snapshot(None, limit=100)
    chats = snapshot["chats"][:]
    while snapshot["next_cursor"]:
        snapshot = projection.snapshot(None, cursor=snapshot["next_cursor"], limit=100)
        chats.extend(snapshot["chats"])
    assert len(chats) == 130
    assert sum(row["conversation_id"] == "saved-0" for row in chats) == 1
    assert all(limit <= 100 for limit, _offset in calls)
    assert all(not row.get("live") or row.get("incarnation_id") for row in chats)


@pytest.mark.asyncio
async def test_real_private_handler_returns_correlated_snapshot_with_canvas_disabled(
    remote_case,
):
    from tldw_chatbook import app_served
    from tldw_chatbook.Canvas.control_protocol import (
        CONTROL_PROTOCOL_VERSION,
        ControlMessage,
    )

    projection, _host, target = remote_case
    runtime = projection.runtime
    runtime._remote_projection = projection
    app = SimpleNamespace(
        console_runtime=runtime,
        served_canvas_control=SimpleNamespace(host_identity=projection.host_identity),
    )
    request = ControlMessage(
        CONTROL_PROTOCOL_VERSION,
        "console.snapshot.request",
        "snapshot-a",
        None,
        {"request": {"target": target.model_dump()}},
    )
    response = await app_served._handle_served_control_request(app, request)
    assert isinstance(response, ControlMessage)
    assert response.request_id == "snapshot-a"
    assert response.payload["snapshot"]["chats"][0]["session_id"] == target.session_id


def test_stale_run_target_cannot_read_current_chat(remote_case):
    projection, _host, target = remote_case
    with pytest.raises(ValueError):
        projection.snapshot(target.model_copy(update={"run_id": "stale-run"}))


def test_native_question_resolver_has_one_winner_under_competing_answers(remote_case):
    _projection, host, target = remote_case
    state = {
        "session_id": target.session_id,
        "event": threading.Event(),
        "decision": {},
    }
    host.register_round("question", "question-1", state)
    first = [{"question": "Pick", "selected": ["A"]}]
    second = [{"question": "Pick", "selected": ["B"]}]
    host.resolve_pending_question(first, "question-1")
    host.resolve_pending_question(second, "question-1")
    assert state["decision"]["answers"][0]["selected"] == ["A"]
    host.registries["question"].clear()


@pytest.mark.asyncio
async def test_private_event_reads_are_serializable_and_lifecycle_scoped(remote_round):
    from tldw_chatbook import app_served
    from tldw_chatbook.Canvas.control_protocol import (
        CONTROL_PROTOCOL_VERSION,
        ControlMessage,
        encode_control_frame,
    )

    projection, host, target = remote_round
    projection.runtime._remote_projection = projection
    app = SimpleNamespace(
        console_runtime=projection.runtime,
        served_canvas_control=SimpleNamespace(host_identity=projection.host_identity),
    )

    async def call(data):
        request = ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            "console.events.request",
            "events",
            None,
            {"request": data},
        )
        reply = await app_served._handle_served_control_request(app, request)
        encode_control_frame(reply)
        return reply.payload["batch"]

    await call({"operation": "register", "client_id": "phone"})
    await call(
        {
            "operation": "answerable",
            "client_id": "phone",
            "target": target.model_dump(),
            "answerable": True,
        }
    )
    projection.poll_changes()
    assert (await call({"operation": "read", "after_sequence": 0}))["events"]
    await call({"operation": "disconnect", "client_id": "phone"})
    assert not host.remote_rounds(target.session_id)[0]["answerable"]


@pytest.mark.parametrize("activity", ["none", "fleet", "decision"])
def test_saved_history_reads_canonical_service_without_hydrating(
    remote_case, tmp_path, activity
):
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    projection, _host, target = remote_case
    db = CharactersRAGDB(tmp_path / "saved.sqlite", "remote-projection")
    try:
        service = ChatConversationService(db)
        conversation_id = service.create_conversation(title="Saved history")
        body = "saved 😺 content " * 2000
        db.add_message(
            {"conversation_id": conversation_id, "sender": "user", "content": body}
        )
        projection.runtime.app.local_chat_conversation_service = service
        session = projection.runtime.chat_store.sessions()[0]
        session.persisted_conversation_id = conversation_id
        if activity == "fleet":
            projection.runtime._agent_bridge = SimpleNamespace(
                fleet_snapshot=lambda _: [
                    SimpleNamespace(handle_id="worker", run_id="run", status="running")
                ]
            )
        if activity == "decision":
            projection.runtime._project_dispatch_decisions["review"] = SimpleNamespace(
                session_id=target.session_id,
                decision_id="review",
                completed=threading.Event(),
            )
        page = projection.snapshot(target)
        session.title = "Changed during metadata/history transition"
        page = projection.snapshot(target, cursor=page["next_cursor"])
        assert page["reset_required"]
        assert page["chats"][0]["title"] == session.title
        chunks, fleet, decisions = [], [], []
        while True:
            assert len(json.dumps(page).encode()) <= 65536
            chunks.extend(row["content"] for row in page["messages"])
            fleet.extend(page["fleet"])
            decisions.extend(page["decisions"])
            if page["next_cursor"] is None:
                break
            page = projection.snapshot(target, cursor=page["next_cursor"])
        assert "".join(chunks) == body
        assert bool(fleet) == (activity == "fleet")
        assert bool(decisions) == (activity == "decision")
        assert projection.runtime.chat_store.message_count(target.session_id) == 0
        assert projection.runtime.provider_gateway is None
    finally:
        db.close_connection()


def test_complete_snapshot_ack_clears_only_the_captured_native_boundary(remote_case):
    projection, _host, target = remote_case
    for _ in range(3):
        projection.events.publish("console.changed", {})
    page = projection.snapshot(target)
    projection.events.publish("console.changed", {})
    projection.acknowledge_snapshot(page["sequence"])
    assert projection.subscribe(page["sequence"]).reset_required
    fresh = projection.snapshot(target)
    projection.acknowledge_snapshot(fresh["sequence"])
    assert not projection.subscribe(fresh["sequence"]).reset_required


def test_project_instruction_and_other_confirmations_stay_visible_without_bodies(
    remote_case,
):
    projection, host, target = remote_case
    projection.runtime._project_dispatch_decisions["dispatch"] = SimpleNamespace(
        decision_id="dispatch",
        session_id=target.session_id,
        completed=threading.Event(),
        notice=SimpleNamespace(body="AUTOMATIC-INSTRUCTION-SENTINEL"),
    )
    state = {
        "session_id": target.session_id,
        "event": threading.Event(),
        "decision": {},
    }
    host.register_round("skill_script", "script-confirm", state)
    host.payloads["skill_script"]["script-confirm"] = {"body": "SECRET-SCRIPT-BODY"}
    try:
        rows = projection.snapshot(target)["decisions"]
        assert {row["kind"] for row in rows} == {
            "project_instruction_dispatch",
            "skill_script",
        }
        assert all(
            not row["supported"] and row["action"] == "open_terminal" for row in rows
        )
        assert "AUTOMATIC-INSTRUCTION-SENTINEL" not in json.dumps(rows)
        assert "SECRET-SCRIPT-BODY" not in json.dumps(rows)
        assert not state["event"].is_set()
    finally:
        host.registries["skill_script"].clear()
        host.payloads["skill_script"].clear()
        projection.runtime._project_dispatch_decisions.clear()


def test_registration_is_bounded_and_disconnect_frees_capacity(remote_case):
    projection, _host, _target = remote_case
    for index in range(64):
        projection.register_client(f"phone-{index}")
    with pytest.raises(ValueError, match="capacity"):
        projection.register_client("overflow")
    projection.disconnect_client("phone-0")
    projection.register_client("replacement")


def test_quarantined_generation_never_exposes_buffered_text(remote_case):
    projection, _host, target = remote_case
    store = projection.runtime.chat_store
    message = store.append_message(
        target.session_id, role="assistant", content="visible"
    )
    original = store._message_or_raise(message.id)
    original.generation_projection_quarantined = True
    store._stream_chunks_by_message[message.id] = ["QUARANTINED-SECRET"]
    assert "QUARANTINED-SECRET" not in json.dumps(projection.snapshot(target))


@pytest.mark.asyncio
async def test_default_native_approval_choices_can_approve_once(remote_round):
    projection, host, target = remote_round
    result = await projection._commands.execute(
        "decision",
        target,
        {"kind": "approval", "answer": {"decisions": {"write_file": "approve_once"}}},
        "approval-once",
    )
    assert result["state"] == "resolved"
    with host.lock:
        state = host.registries["approval"].get(target.round_id)
    if state is not None:
        assert state["decisions"]["write_file"] == "approve_once"


def test_unpublished_native_change_during_capture_requires_reconciliation(
    remote_case, monkeypatch
):
    projection, _host, target = remote_case
    store = projection.runtime.chat_store
    store.append_message(target.session_id, role="assistant", content="Before")
    original = store.read_only_messages_for_session

    def capture(session_id):
        rows = original(session_id)
        store.append_message(session_id, role="assistant", content="During capture")
        return rows

    monkeypatch.setattr(store, "read_only_messages_for_session", capture)
    snapshot = projection.snapshot(target)
    projection.acknowledge_snapshot(snapshot["sequence"])
    assert projection.subscribe(snapshot["sequence"]).reset_required
    assert projection.subscribe(snapshot["sequence"]).sequence > snapshot["sequence"]


def test_unserializable_approval_review_stays_pending(remote_round):
    projection, host, target = remote_round
    with host.lock:
        host.payloads["approval"][target.round_id]["calls"][0]["arguments"] = {
            "path": "safe.txt",
            "value": float("nan"),
        }
    row = projection.snapshot(target)["decisions"][0]
    assert not row["supported"] and row["action"] == "open_terminal"
    assert not host.registries["approval"][target.round_id]["event"].is_set()


@pytest.mark.asyncio
async def test_remote_question_preserves_native_choices_and_original_winner(
    remote_case,
):
    projection, host, target = remote_case
    state = {
        "session_id": target.session_id,
        "run_id": "question-run",
        "event": threading.Event(),
        "decision": {},
    }
    host.register_round("question", "question-remote", state)
    host.payloads["question"]["question-remote"] = {
        "questions": [
            {
                "question": "Pick",
                "header": "Choice",
                "options": [{"label": "A"}, {"label": "B"}],
            }
        ]
    }
    target = target.model_copy(
        update={"round_id": "question-remote", "run_id": "question-run"}
    )
    try:
        assert projection.snapshot(target)["decisions"][0]["supported"]

        async def answer(selected):
            return await projection._commands.execute(
                "decision",
                target,
                {
                    "kind": "question",
                    "answer": {"answers": [{"question": "Pick", "selected": selected}]},
                },
                "question-command",
            )

        assert (await answer(["foreign-option"]))["state"] == "refused"
        assert not state["event"].is_set()
        assert (await answer(["A"]))["state"] == "resolved"
        assert (await answer(["B"]))["state"] == "resolved"
        assert state["decision"]["answers"][0]["selected"] == ["A"]
    finally:
        host.registries["question"].clear()
        host.payloads["question"].clear()
