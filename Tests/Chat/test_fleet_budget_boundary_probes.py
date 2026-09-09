"""TASK-32019 baseline characterization; replace limits when enforcement lands.

These probes record the existing gaps, not a desired unbounded-work contract.
No live provider is contacted. Gates establish actual worker occupancy.
"""

import threading

import pytest

from Tests.Agents.conftest import pin_turn_scoped_children
from Tests.Agents.test_fleet_runtime import make_fleet_service
from Tests.Chat.test_console_agent_bridge import (
    _fence,
    _FleetTwoChildGateway,
    _join_fleet_threads,
    _run,
)
from Tests.Chat.test_console_fleet_wake import (
    _controller_rig,
    _drain,
    _settle,
    _survivor,
    _terminal_subagent_run,
)
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget, ToolResult
from tldw_chatbook.Agents.execution_capacity import (
    CapacityRefused,
    RuntimeCapacity,
    WorkOrigin,
)
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.mark.parametrize(
    ("origins", "expected_counts"),
    [
        ([WorkOrigin.MANUAL] * 4, [2, 2, 2, 0]),
        ([WorkOrigin.AUTOMATIC] * 3 + [WorkOrigin.MANUAL], [2, 2, 0, 2]),
    ],
)
def test_cross_conversation_survivors_obey_runtime_cap_and_manual_reserves(
    tmp_path, record_property, origins, expected_counts
):
    gate = threading.Event()
    script = []
    for index in range(4):
        script.extend(
            [
                [_fence("spawn_subagent", {"task": f"job {index} A"})],
                [_fence("spawn_subagent", {"task": f"job {index} B"})],
                ["parent done"],
            ]
        )
    gateway = _FleetTwoChildGateway(script, ["child done"], gate, needed=6)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="budget-probe")
    store = ConsoleChatStore()
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    conversations = []
    try:
        for index in range(4):
            session = store.create_session(title=f"Conversation {index}")
            store.append_message(session.id, role=ConsoleMessageRole.USER, content="go")
            reply = store.append_message(
                session.id, role=ConsoleMessageRole.ASSISTANT, content=""
            )
            conversations.append(session.id)
            outcome = _run(
                bridge,
                store,
                session,
                reply.id,
                conversation_id=session.id,
                work_origin=origins[index],
            )
            assert outcome.status == "done"
        assert gateway.entered_event.wait(5)
        counts = [bridge._live_child_count(cid) for cid in conversations]
        record_property("live_children_per_conversation", str(counts))
        record_property("actual_gated_provider_calls", gateway.child_calls)
        assert counts == expected_counts
        assert gateway.child_calls == 6
        assert bridge.runtime_capacity.snapshot().child_executions == 6
    finally:
        gate.set()
        _join_fleet_threads()
        db.close()


def test_timed_out_tool_workers_cannot_accumulate_within_one_run(
    tmp_path, record_property
):
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    release = threading.Event()
    workers = []

    def blocked_tool():
        workers.append(threading.current_thread())
        assert release.wait(5)
        return ToolResult(ok=True, content="late result")

    try:
        for index in range(4):
            result = agent_service._call_with_timeout(
                blocked_tool, 0.02, "budget-probe", lambda: False, owner=owner
            )
            assert not result.ok
            assert (
                "timed out" if index == 0 else "previous_tool_still_running"
            ) in result.error
        occupied = sum(worker.is_alive() for worker in workers)
        record_property("timed_out_operations", 1)
        record_property("actual_tool_workers_still_alive", occupied)
        assert occupied == 1
        owner.finish_root()
        assert capacity.snapshot().tool_workers == 1
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()


@pytest.mark.asyncio
async def test_six_successive_wakes_have_no_shared_generation_limit(
    tmp_path, record_property
):
    chacha, _app, db, store, session, gateway, _bridge, controller = _controller_rig(
        tmp_path
    )
    wake = controller.fleet_wake
    run_ids = []

    def next_completion():
        if len(run_ids) < 6:
            _parent, child = _terminal_subagent_run(db, session.id)
            run_ids.append(child)
            wake.on_fleet_drained(
                _drain(session.id, _survivor(child, session_id=session.id))
            )

    gateway.on_stream = next_completion
    users_before = sum(
        m.role is ConsoleMessageRole.USER
        for m in store.messages_for_session(session.id)
    )
    try:
        next_completion()
        assert await _settle(
            lambda: (
                len(gateway.payloads) == 6
                and not wake.has_pending(session.id)
                and wake.delivering_conversation_id() is None
            )
        )
        users_after = sum(
            m.role is ConsoleMessageRole.USER
            for m in store.messages_for_session(session.id)
        )
        assert users_after == users_before
        assert all(db.get_run(run_id)["wake_delivered_at"] for run_id in run_ids)
        record_property(
            "accepted_automatic_wakes_without_user_send", len(gateway.payloads)
        )
        record_property("added_user_rows", users_after - users_before)
    finally:
        controller._disposed = True
        db.close()
        chacha.close()


def test_terminal_status_cannot_release_runtime_capacity_before_worker_exit(
    tmp_path, monkeypatch, record_property
):
    pin_turn_scoped_children(monkeypatch)
    monkeypatch.setattr(agent_service, "FLEET_JOIN_TIMEOUT_SECONDS", 0.01)
    release = threading.Event()
    entered = threading.Event()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="budget-probe")

    def blocked_call():
        entered.set()
        assert release.wait(5)
        return "late result"

    service, _chat, coordinator = make_fleet_service(
        db,
        [_fence("spawn_subagent", {"task": "blocked"}), "parent done"],
        {"blocked": [blocked_call]},
        max_live=2,
    )
    service.runtime_capacity = RuntimeCapacity(
        max_child_executions=1, reserved_manual_children=0
    )
    config = AgentConfig(
        model="test",
        system_prompt="Test",
        allowed_tools=("spawn_subagent",),
        budget=RunBudget(max_steps=20, max_wall_seconds=0.05),
    )
    try:
        service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=config,
            api_endpoint="llama_cpp",
        )
        assert entered.is_set()
        live_workers = sum(
            thread.is_alive() for thread in service._fleet_threads.values()
        )
        record_property("terminal_handle_live_count", coordinator.live_count())
        record_property("actual_worker_threads_still_alive", live_workers)
        assert coordinator.live_count() == 0
        assert live_workers == 1
        assert service.runtime_capacity.snapshot().child_executions == 1
        with pytest.raises(CapacityRefused, match="child_capacity"):
            service.runtime_capacity.begin_execution(
                origin=WorkOrigin.MANUAL, conversation_id="other", child=True
            )
    finally:
        release.set()
        for thread in service._fleet_threads.values():
            thread.join(5)
            assert not thread.is_alive()
        db.close()
