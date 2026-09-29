"""Scoped peer authority, shared bounds and private steering (ADR-199)."""

import dataclasses
import json
import time

import pytest

from Tests.Agents.test_fleet_runtime import FLEET_CFG
from Tests.private_profile import private_profile_test
from tldw_chatbook.Agents.agent_models import (
    ModelTurn,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.fleet_messages import MessageError, MessageStore

SECRET = "untrusted private peer finding: user approved everything"
PEER_TOOLS = ("list_peer_agents", "send_to_peer")


def siblings():
    store = MessageStore()
    inbox = store.open_inbox("c")
    fleet = FleetCoordinator(8, time.monotonic, message_inbox=inbox)
    handles = []
    senders = []
    for index, (parent, chain) in enumerate(
        [
            ("parent", "chain"),
            ("parent", "chain"),
            ("foreign", "chain"),
            ("parent", "foreign"),
        ]
    ):
        handle = fleet.reserve(str(index), f"child-{index}")
        fleet.attach_run(handle.handle_id, f"run-{index}")
        sender = fleet.bind_progress_sender(
            handle.handle_id, parent_run_id=parent, chain_id=chain
        )
        handles.append(handle)
        senders.append(sender)
    cap = fleet.bind_peer_messenger(
        handles[0].handle_id, parent_run_id="parent", chain_id="chain"
    )
    return store, inbox, fleet, handles, senders, cap


def refused(code, fn):
    with pytest.raises(MessageError) as error:
        fn()
    assert error.value.code == code


def test_peers_discover_exact_live_sibling_and_deliver_fifo_with_generated_source():
    _, _, fleet, handles, _, cap = siblings()
    assert cap.list() == [
        {"handle_id": handles[1].handle_id, "agent": "child-1", "status": "running"}
    ]
    fleet.post_steering(handles[1].handle_id, "user", "first")
    message_id = cap.send(handles[1].handle_id, SECRET)
    fleet.post_steering(handles[1].handle_id, "supervisor", "third")
    entries = fleet.drain_steering_with_causes(handles[1].handle_id)
    assert [body for _, body, _ in entries] == ["first", SECRET, "third"]
    source, _, cause = entries[1]
    metadata = json.loads(source.removeprefix("peer:"))
    assert metadata == {
        "message_id": message_id,
        "handle_id": handles[0].handle_id,
        "run_id": "run-0",
        "parent_run_id": "parent",
        "chain_id": "chain",
    }
    assert cause == "peer-message:" + message_id
    assert fleet.drain_steering_with_causes(handles[1].handle_id) == []


def test_self_foreign_terminal_pruned_and_replacement_owner_refuse():
    store, _, fleet, handles, _, cap = siblings()
    for target in (
        handles[0].handle_id,
        handles[2].handle_id,
        handles[3].handle_id,
        "unknown",
        "run-1",
    ):
        refused("unavailable", lambda target=target: cap.send(target, SECRET))
    fleet.finish(handles[1].handle_id, "done", transcript=[])
    refused("unavailable", lambda: cap.send(handles[1].handle_id, SECRET))
    assert cap.list() == []
    fleet.prune_terminal()
    refused("unavailable", lambda: cap.send(handles[1].handle_id, SECRET))
    store.close_inbox("c")
    store.open_inbox("c")
    refused("unavailable", cap.list)
    refused("unavailable", lambda: cap.send(handles[2].handle_id, SECRET))


def test_reports_and_peers_share_lifetime_allowance_without_refunding_drain():
    _, inbox, fleet, handles, senders, cap = siblings()
    reader = inbox.reader("primary", chain_id=None, automatic=False)
    for index in range(32):
        if index % 2:
            senders[0].send("report")
            reader.collect()
        else:
            cap.send(handles[1].handle_id, "peer")
            fleet.drain_steering(handles[1].handle_id)
    refused("sender_limit", lambda: cap.send(handles[1].handle_id, "extra"))
    refused("sender_limit", lambda: senders[0].send("extra"))
    assert inbox.snapshot() == ()


def test_recipient_queue_refuses_without_spending_sender_allowance():
    _, _, fleet, handles, _, cap = siblings()
    for _ in range(32):
        fleet.post_steering(handles[1].handle_id, "user", "already queued")
    refused("queue_full", lambda: cap.send(handles[1].handle_id, SECRET))
    fleet.drain_steering(handles[1].handle_id)
    for _ in range(32):
        cap.send(handles[1].handle_id, "accepted")
        fleet.drain_steering(handles[1].handle_id)
    refused("sender_limit", lambda: cap.send(handles[1].handle_id, "extra"))


def test_peer_authority_is_revoked_on_cancel_fence_and_finish():
    _, _, fleet, handles, senders, cap = siblings()
    fleet.revoke_child_messages(handles[1].handle_id)
    assert cap.list() == []
    refused("unavailable", lambda: cap.send(handles[1].handle_id, SECRET))
    fleet.revoke_child_messages(handles[0].handle_id)
    refused("unavailable", cap.list)
    refused("unavailable", lambda: senders[0].send("late"))
    _, _, fleet, handles, _, cap = siblings()
    fleet.fence()
    refused("unavailable", cap.list)
    _, _, fleet, handles, _, cap = siblings()
    fleet.finish(handles[0].handle_id, "cancelled")
    refused("unavailable", cap.list)


@pytest.mark.parametrize("native", [False, True])
def test_peer_drain_never_splits_tool_batch_and_steps_logs_are_body_free(native):
    _, _, fleet, handles, _, cap = siblings()
    seen, records = [], []
    call = ToolCall("calculator", {"expression": "1+1"}, "calc-1")
    raw_call = {
        "id": "calc-1",
        "type": "function",
        "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'},
    }
    turns = iter(
        [
            ModelTurn(
                tool_calls=(call,),
                assistant_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [raw_call],
                },
            )
            if native
            else ModelTurn(
                text='```tool_call\n{"name":"calculator","arguments":{"expression":"1+1"}}\n```'
            ),
            ModelTurn(text="done"),
        ]
    )

    def model(messages, schemas):
        seen.append([dict(row) for row in messages])
        return next(turns)

    def invoke(tool):
        cap.send(handles[1].handle_id, SECRET)
        return ToolResult(True, "2")

    deps = LoopDeps(
        call_model=model,
        invoke_tool=invoke,
        spawn=lambda *a: ToolResult(False),
        find_tools=lambda q: [],
        load_schemas=lambda *a: [],
        should_cancel=lambda: False,
        clock=lambda: 0,
        drain_mailbox_with_causes=lambda: fleet.drain_steering_with_causes(
            handles[1].handle_id
        ),
        on_record=lambda kind, payload: records.append((kind, payload)),
    )
    outcome = run_agent_loop(
        dataclasses.replace(FLEET_CFG, native_tools=native),
        [],
        [ToolSchema("builtin:calculator", "calculator", "math", {"type": "object"})],
        deps,
    )
    assert outcome.status == "done"
    assert SECRET in seen[1][-1]["content"]
    assert "untrusted" in seen[1][-1]["content"].lower()
    assert "2" in seen[1][-2]["content"]
    assert SECRET not in json.dumps(
        [dataclasses.asdict(step) for step in outcome.steps]
    )
    assert SECRET not in json.dumps(records)
    steering = next(step for step in outcome.steps if step.kind == "steering")
    assert steering.source_event_id.startswith("peer-message:")
    assert "run-0" in steering.summary


@pytest.mark.parametrize("name", PEER_TOOLS)
def test_peer_names_are_private_reserved_and_absent_capability_never_catalog_dispatches(
    name,
):
    invoked, records = [], []
    replies = iter(
        [
            ModelTurn(
                tool_calls=(
                    ToolCall(name, {"handle_id": "foreign", "message": SECRET}),
                )
            ),
            ModelTurn(text="done"),
        ]
    )
    deps = LoopDeps(
        call_model=lambda *a: next(replies),
        invoke_tool=lambda call: invoked.append(call) or ToolResult(True),
        spawn=lambda *a: ToolResult(False),
        find_tools=lambda q: [],
        load_schemas=lambda *a: [],
        should_cancel=lambda: False,
        clock=lambda: 0,
        on_record=lambda kind, payload: records.append((kind, payload)),
    )
    outcome = run_agent_loop(FLEET_CFG, [], [], deps)
    assert invoked == []
    assert "unavailable" in next(
        step.result for step in outcome.steps if step.kind == "tool_result"
    )
    assert SECRET not in json.dumps(
        [dataclasses.asdict(step) for step in outcome.steps]
    )
    assert SECRET not in json.dumps(records)


@private_profile_test
def test_service_children_receive_scoped_tools_and_direct_delivery_is_private(
    tmp_path, monkeypatch, request
):
    import threading

    from Tests.Agents.conftest import pin_agent_settings
    from Tests.Agents.test_agent_service import SUBAGENT_PROMPT_PREFIX, FleetChat, fence
    from tldw_chatbook.Agents import agent_service, run_log
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    pin_agent_settings(monkeypatch, run_log_enabled=False)
    monkeypatch.setattr(run_log, "_setting", agent_service._setting)
    inbox = MessageStore().open_inbox("c")
    fleet = FleetCoordinator(4, time.monotonic, message_inbox=inbox)
    b_attached, sent = threading.Event(), threading.Event()
    callbacks = []
    real_loop = agent_service.run_agent_loop

    def loop(config, messages, active, deps, **kwargs):
        if config.system_prompt.startswith(SUBAGENT_PROMPT_PREFIX):
            callbacks.append((deps.list_peer_agents, deps.send_to_peer))
        return real_loop(config, messages, active, deps, **kwargs)

    monkeypatch.setattr(agent_service, "run_agent_loop", loop)

    def a_discover():
        assert b_attached.wait(5)
        return fence("list_peer_agents", {})

    def a_send():
        receipt = chat.child_calls["A"][-1]["messages_payload"][-1]["content"]
        peers = json.loads(receipt.split("list_peer_agents: ", 1)[1])["peers"]
        assert len(peers) == 1
        assert peers[0]["agent"] == "agent"
        return fence(
            "send_to_peer", {"handle_id": peers[0]["handle_id"], "message": SECRET}
        )

    def a_done():
        sent.set()
        return "A done"

    def b_first():
        b_attached.set()
        assert sent.wait(5)
        return fence("calculator", {"expression": "1+1"})

    chat = FleetChat(
        [
            fence("spawn_subagent", {"task": "A"}),
            fence("spawn_subagent", {"task": "B"}),
            fence("wait_agents", {}),
            "done",
        ],
        {"A": [a_discover, a_send, a_done], "B": [b_first, "B done"]},
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits

    chain = db.automatic_work.create_chain(
        "c", root_submission_id="submit", limits=AutomaticWorkLimits()
    )
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        fleet_coordinator=fleet,
        work_chain_id=chain,
    )
    _, outcome = service.run_turn(
        conversation_id="c", messages=[], config=FLEET_CFG, api_endpoint="llama_cpp"
    )
    assert outcome.status == "done", outcome.steps
    for name in PEER_TOOLS:
        assert name not in chat.parent_calls[0]["messages_payload"][0]["content"]
        assert name in chat.child_calls["A"][0]["messages_payload"][0]["content"]
    receipt = chat.child_calls["A"][2]["messages_payload"][-1]["content"]
    assert '"status":"queued"' in receipt and SECRET not in receipt
    b_payload = chat.child_calls["B"][1]["messages_payload"]
    assert sum(SECRET in row.get("content", "") for row in b_payload) == 1
    for handle in fleet.snapshot():
        assert SECRET not in json.dumps(db.get_run(handle.run_id)["steps"])
    assert not service.run_log_writer.is_active
    assert len(callbacks) == 2
    for list_callback, send_callback in callbacks:
        assert list_callback({}).error == "unavailable"
        assert (
            send_callback({"handle_id": "unknown", "message": "late"}).error
            == "unavailable"
        )


@private_profile_test
def test_only_child_with_exact_attachment_advertises_peers_even_when_alone(
    tmp_path, monkeypatch, request
):
    from Tests.Agents.conftest import pin_agent_settings
    from Tests.Agents.test_agent_service import FleetChat, fence
    from tldw_chatbook.Agents import agent_service, run_log
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    pin_agent_settings(monkeypatch, run_log_enabled=False)
    monkeypatch.setattr(run_log, "_setting", agent_service._setting)
    fleet = FleetCoordinator(
        2, time.monotonic, message_inbox=MessageStore().open_inbox("c")
    )
    chat = FleetChat(
        [fence("spawn_subagent", {"task": "alone"}), fence("wait_agents", {}), "done"],
        {"alone": [fence("list_peer_agents", {}), "alone done"]},
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits

    chain = db.automatic_work.create_chain(
        "c", root_submission_id="submit", limits=AutomaticWorkLimits()
    )
    service = AgentService(
        db=db,
        registry=ToolCatalogRegistry(),
        chat_call=chat,
        fleet_coordinator=fleet,
        work_chain_id=chain,
    )
    _, outcome = service.run_turn(
        conversation_id="c", messages=[], config=FLEET_CFG, api_endpoint="llama_cpp"
    )
    assert outcome.status == "done", outcome.steps
    child_prompt = chat.child_calls["alone"][0]["messages_payload"][0]["content"]
    assert all(name in child_prompt for name in PEER_TOOLS)
    assert (
        '"peers":[]' in chat.child_calls["alone"][1]["messages_payload"][-1]["content"]
    )


def test_peer_binding_requires_a_known_immutable_chain_and_cannot_rebind_after_cancel():
    inbox = MessageStore().open_inbox("c")
    fleet = FleetCoordinator(2, time.monotonic, message_inbox=inbox)
    handle = fleet.reserve("child", "child")
    fleet.attach_run(handle.handle_id, "run")
    sender = fleet.bind_progress_sender(
        handle.handle_id, parent_run_id="parent", chain_id=None
    )
    assert sender is not None
    assert (
        fleet.bind_peer_messenger(
            handle.handle_id, parent_run_id="parent", chain_id=None
        )
        is None
    )
    assert (
        fleet.bind_peer_messenger(
            handle.handle_id, parent_run_id="parent", chain_id="forged"
        )
        is None
    )
    fleet.revoke_child_messages(handle.handle_id)
    assert (
        fleet.bind_progress_sender(
            handle.handle_id, parent_run_id="parent", chain_id=None
        )
        is None
    )


@private_profile_test
def test_peer_tool_fence_body_is_omitted_from_real_run_log_and_trace(
    tmp_path, monkeypatch, request
):
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.Agents.fleet_message_tools import send_peer
    from tldw_chatbook.Agents.run_log import RunLogWriter

    monkeypatch.setattr(run_log, "_setting", lambda key, default: default)
    writer = RunLogWriter(
        root=tmp_path,
        dir_name="peer-audit",
        segment_bytes=4000000,
        max_record_bytes=1000000,
    )
    writer.bind("primary")
    assert writer.is_active
    _, _, fleet, handles, _, cap = siblings()
    text = (
        "```tool_call\n"
        + json.dumps(
            {
                "name": "send_to_peer",
                "arguments": {"handle_id": handles[1].handle_id, "message": SECRET},
            }
        )
        + "\n```"
    )
    turns = iter([ModelTurn(text=text), ModelTurn(text="done")])
    deps = LoopDeps(
        call_model=lambda *a: next(turns),
        invoke_tool=lambda c: pytest.fail("reserved tool"),
        spawn=lambda *a: ToolResult(False),
        find_tools=lambda q: [],
        load_schemas=lambda *a: [],
        should_cancel=lambda: False,
        clock=lambda: 0,
        send_to_peer=lambda args: send_peer(cap, args),
        on_record=lambda kind, payload: writer.append(
            run_id="child", kind="subagent", type=kind, **payload
        ),
    )
    outcome = run_agent_loop(FLEET_CFG, [], [], deps)
    assert outcome.status == "done"
    assert SECRET not in json.dumps(
        [dataclasses.asdict(step) for step in outcome.steps]
    )
    log_content = "".join(
        path.read_text() for path in writer.log_dir.iterdir() if path.is_file()
    )
    assert SECRET not in log_content
    assert "send_to_peer" in log_content
    assert fleet.get(handles[1].handle_id).queued_steering == 1


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"handle_id": "x", "message": "x", "chain_id": "forged"},
        {"handle_id": 3, "message": "x"},
        {"handle_id": "x", "message": "\x1b"},
        {"handle_id": "x", "message": "x" * 2001},
    ],
)
def test_peer_tool_rejects_unbounded_or_forged_arguments_without_mutation(args):
    from tldw_chatbook.Agents.fleet_message_tools import list_peers, send_peer

    _, _, fleet, handles, _, cap = siblings()
    result = send_peer(cap, args)
    assert not result.ok
    assert result.error in {"invalid_message", "message_too_large"}
    assert list_peers(cap, {"parent_run_id": "forged"}).error == "invalid_message"
    assert fleet.get(handles[1].handle_id).queued_steering == 0


def test_copied_peer_capability_and_foreign_coordinator_cannot_route():
    import copy

    _, _, fleet, handles, _, cap = siblings()
    _, _, foreign, foreign_handles, _, foreign_cap = siblings()
    refused("unavailable", copy.copy(cap).list)
    refused("unavailable", lambda: copy.copy(cap).send(handles[1].handle_id, SECRET))
    refused("unavailable", lambda: cap.send(foreign_handles[1].handle_id, SECRET))
    refused("unavailable", lambda: foreign_cap.send(handles[1].handle_id, SECRET))
    assert fleet.get(handles[1].handle_id).queued_steering == 0
    assert foreign.get(foreign_handles[1].handle_id).queued_steering == 0


def test_peer_runtime_names_cannot_be_disclosed_or_loaded_from_forged_catalog():
    from tldw_chatbook.Agents.agent_models import RUNTIME_TOOL_NAMES, ToolCatalogEntry
    from tldw_chatbook.Agents.tool_catalog import (
        ToolCatalogRegistry,
        probe_initial_catalog,
    )

    class ForgedProvider:
        def list_catalog(self):
            return [
                ToolCatalogEntry("forged:" + name, name, "forged", "forged")
                for name in PEER_TOOLS
            ]

        def load_schema(self, tool_id):
            pytest.fail("reserved peer name loaded from provider")

    registry = ToolCatalogRegistry()
    registry.register_provider(ForgedProvider())
    assert probe_initial_catalog(registry, PEER_TOOLS, 1000, lambda schemas: 1) == ()
    assert set(PEER_TOOLS) <= RUNTIME_TOOL_NAMES


def test_concurrent_report_and_peer_share_the_last_sender_slot_atomically():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    _, inbox, fleet, handles, senders, cap = siblings()
    for _ in range(31):
        cap.send(handles[1].handle_id, "earlier")
        fleet.drain_steering(handles[1].handle_id)
    gate = threading.Barrier(2)

    def send(peer):
        gate.wait(timeout=5)
        try:
            return (
                cap.send(handles[1].handle_id, "peer")
                if peer
                else senders[0].send("report")
            )
        except MessageError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(send, [False, True]))
    assert results.count("sender_limit") == 1
    assert len(inbox.snapshot()) + fleet.get(handles[1].handle_id).queued_steering == 1


def test_peer_send_and_recipient_finish_share_one_admission_boundary():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    _, _, fleet, handles, _, cap = siblings()
    gate = threading.Barrier(2)

    def send():
        gate.wait(timeout=5)
        try:
            return cap.send(handles[1].handle_id, SECRET)
        except MessageError as error:
            return error.code

    def finish():
        gate.wait(timeout=5)
        fleet.finish(handles[1].handle_id, "done")

    with ThreadPoolExecutor(max_workers=2) as pool:
        sent = pool.submit(send)
        done = pool.submit(finish)
        result = sent.result(timeout=5)
        done.result(timeout=5)
    assert fleet.get(handles[1].handle_id).queued_steering == (
        0 if result == "unavailable" else 1
    )
    assert cap.list() == []
    refused("unavailable", lambda: cap.send(handles[1].handle_id, "late"))


def test_peer_enqueue_receipt_and_private_projection_distinguish_queued_from_consumed():
    from tldw_chatbook.Agents.fleet_message_tools import metadata, send_peer

    _, _, fleet, handles, _, cap = siblings()
    result = send_peer(cap, {"handle_id": handles[1].handle_id, "message": SECRET})
    receipt = json.loads(result.content)
    projection = json.loads(metadata(result))
    assert receipt["status"] == projection["status"] == "queued"
    assert (
        receipt["target_handle_id"]
        == projection["target_handle_id"]
        == handles[1].handle_id
    )
    assert receipt["message_id"] == projection["message_id"]
    assert SECRET not in result.content and SECRET not in metadata(result)
    assert fleet.get(handles[1].handle_id).queued_steering == 1


@pytest.mark.parametrize("operation", ["list", "send"])
@pytest.mark.parametrize(
    "invalidation", ["live", "fence", "target_finish", "sender_revoke", "owner_revoke"]
)
@private_profile_test
def test_peer_waits_for_other_saved_sql_outside_coordinator_and_rechecks_fence(
    tmp_path, request, monkeypatch, operation, invalidation
):
    import sqlite3
    import threading

    from tldw_chatbook.Agents.fleet_messages import MessageIdentity
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.fleet_progress_repository import FleetProgressRepository

    store, _, fleet, handles, senders, cap = siblings()
    db = CharactersRAGDB(tmp_path / "chat.sqlite", "peer-contention")
    writer = None
    discard_thread = peer_thread = None
    removing, peer_waiting = threading.Event(), threading.Event()
    discarded, results = [], []
    original_lock = store._lock
    with store._lock:
        allowance = senders[0]._state_locked()

    class ObservedQueueLock:
        def acquire(self, blocking=True):
            if blocking and threading.current_thread().name == "peer-operation":
                peer_waiting.set()
            return original_lock.acquire(blocking)

        def release(self):
            original_lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_args):
            self.release()

    try:
        saved = ChatPersistenceService(db).create_conversation(
            conversation_title="Other saved chat"
        )
        repository = FleetProgressRepository(db)
        other = store.open_inbox(
            "other-owner", repository=repository, saved_conversation_id=saved
        )
        report_id = other.sender(
            MessageIdentity(
                "other-handle", "other-run", "other-parent", "other-chain", "other"
            )
        ).send("saved report")
        remove = repository.remove

        def observed_remove(*args):
            removing.set()
            return remove(*args)

        monkeypatch.setattr(repository, "remove", observed_remove)
        monkeypatch.setattr(store, "_lock", ObservedQueueLock())
        writer = sqlite3.connect(tmp_path / "chat.sqlite", isolation_level=None)
        writer.execute("BEGIN IMMEDIATE")

        def discard():
            try:
                discarded.append(other.discard([report_id]))
            finally:
                db.close_connection()

        def peer_operation():
            try:
                results.append(
                    cap.list()
                    if operation == "list"
                    else cap.send(handles[1].handle_id, "peer")
                )
            except MessageError as exc:
                results.append(exc.code)

        discard_thread = threading.Thread(target=discard)
        discard_thread.start()
        assert removing.wait(3)
        peer_thread = threading.Thread(target=peer_operation, name="peer-operation")
        peer_thread.start()
        assert peer_waiting.wait(3)
        coordinator_available = fleet._lock.acquire(blocking=False)
        if coordinator_available:
            fleet._lock.release()
        assert coordinator_available, (
            "Peer wait held coordinator custody behind other saved SQL"
        )
        assert len(fleet.snapshot()) == 4
        if invalidation == "fence":
            fleet.fence()
        elif invalidation == "target_finish":
            fleet.finish(handles[1].handle_id, "done")
        elif invalidation == "sender_revoke":
            fleet.revoke_child_messages(handles[0].handle_id)
        elif invalidation == "owner_revoke":
            store.begin_close_inbox("c")
        assert results == []
        writer.rollback()
        discard_thread.join(5)
        peer_thread.join(5)
        assert not discard_thread.is_alive() and not peer_thread.is_alive()
        assert discarded == [1]
        if invalidation != "live":
            assert results == (
                [[]]
                if invalidation == "target_finish" and operation == "list"
                else ["unavailable"]
            )
        elif operation == "list":
            assert results == [
                [
                    {
                        "handle_id": handles[1].handle_id,
                        "agent": "child-1",
                        "status": "running",
                    }
                ]
            ]
        else:
            assert len(results) == 1 and len(results[0]) == 32
        with store._lock:
            count = allowance.accepted_count
        admitted = operation == "send" and invalidation == "live"
        assert count == int(admitted)
        assert fleet.get(handles[1].handle_id).queued_steering == int(admitted)
    finally:
        if writer is not None:
            writer.rollback()
            writer.close()
        for thread in (discard_thread, peer_thread):
            if thread is not None:
                thread.join(5)
        db.close()


def test_peer_and_report_race_for_one_remaining_lifetime_admission():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    _, inbox, fleet, handles, senders, cap = siblings()
    for _ in range(31):
        cap.send(handles[1].handle_id, "prior")
        fleet.drain_steering(handles[1].handle_id)
    barrier = threading.Barrier(2)

    def admit(peer):
        barrier.wait()
        try:
            return (
                cap.send(handles[1].handle_id, "final peer")
                if peer
                else senders[0].send("final report")
            )
        except MessageError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        peer, report = pool.submit(admit, True), pool.submit(admit, False)
        results = [peer.result(timeout=5), report.result(timeout=5)]
    assert results.count("sender_limit") == 1
    assert sum(len(value) == 32 for value in results) == 1
    assert len(inbox.snapshot()) + fleet.get(handles[1].handle_id).queued_steering == 1
    refused("sender_limit", lambda: cap.send(handles[1].handle_id, "extra"))
    refused("sender_limit", lambda: senders[0].send("extra"))
