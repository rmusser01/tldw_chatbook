"""Child admission uses physical runtime ownership before handles or rows."""

import threading
import time
from contextlib import contextmanager

import pytest

from Tests.Agents.conftest import pin_agent_settings
from Tests.Agents.test_agent_service import FleetChat, fence
from Tests.Agents.test_tool_worker_capacity import BlockingTool
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def service_for(
    db,
    capacity,
    replies,
    children,
    *,
    origin=WorkOrigin.MANUAL,
    tool=None,
    allow_unconsumed=False,
):
    registry = ToolCatalogRegistry()
    if tool is not None:
        registry.register_provider(tool)
    chat = FleetChat(replies, children, allow_unconsumed=allow_unconsumed)
    coordinator = FleetCoordinator(max_live=3, clock=time.monotonic)
    service = AgentService(
        db,
        registry,
        chat_call=chat,
        fleet_coordinator=coordinator,
        runtime_capacity=capacity,
        work_origin=origin,
    )
    return service, chat, coordinator


def run(service, conversation, *, max_subagents=2):
    return service.run_turn(
        conversation_id=conversation,
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="test",
            system_prompt="Test",
            allowed_tools=("spawn_subagent", "whoami"),
            budget=RunBudget(
                max_steps=40,
                max_model_turns=12,
                max_subagents=max_subagents,
                max_tool_call_seconds=0.02,
            ),
        ),
        api_endpoint="llama_cpp",
    )


def test_simultaneous_conversations_launch_only_six_real_children(
    tmp_path, monkeypatch
):
    pin_agent_settings(monkeypatch, subagents_outlive_turn=True)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="admission")
    capacity = RuntimeCapacity()
    release = threading.Event()
    ready = threading.Barrier(4)
    entered = []
    all_children_entered = threading.Event()
    errors = []

    def child():
        entered.append(threading.current_thread())
        if len(entered) == 6:
            all_children_entered.set()
        assert release.wait(10)
        return "child done"

    services = [
        service_for(
            db,
            capacity,
            [
                fence("spawn_subagent", {"task": "a"}),
                fence("spawn_subagent", {"task": "b"}),
                "done",
            ],
            {"a": [child], "b": [child]},
            allow_unconsumed=True,  # Two candidates must be refused before dispatch.
        )[0]
        for _ in range(4)
    ]

    def parent(i):
        try:
            ready.wait(3)
            assert run(services[i], f"c{i}")[1].status == "done"
        except BaseException as exc:  # noqa: BLE001 -- propagate thread failures to test
            errors.append(exc)

    parents = [threading.Thread(target=parent, args=(i,)) for i in range(4)]
    try:
        for thread in parents:
            thread.start()
        for thread in parents:
            thread.join(5)
            assert not thread.is_alive()
        assert not errors
        assert all_children_entered.wait(5)
        assert capacity.snapshot().child_executions == 6
        rows = [
            row
            for i in range(4)
            for row in db.list_runs(f"c{i}")
            if row["agent_kind"] == "subagent"
        ]
        assert len(rows) == 6
        assert sum(len(service._fleet_threads) for service in services) == 6
    finally:
        release.set()
        for service in services:
            for thread in service._fleet_threads.values():
                thread.join(5)
        db.close()
    assert len(entered) == 6
    assert capacity.snapshot().executions == ()


def test_terminal_child_with_timed_out_tool_holds_slot_until_tool_finishes(tmp_path):
    capacity = RuntimeCapacity(max_child_executions=1, reserved_manual_children=0)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="admission")
    tool = BlockingTool()
    first, _, _ = service_for(
        db,
        capacity,
        [fence("spawn_subagent", {"task": "child"}), fence("wait_agents", {}), "done"],
        {"child": [fence("whoami", {}), "child done"]},
        tool=tool,
    )

    def release_worker():
        tool.release.set()
        for worker in tool.workers:
            worker.join(5)
        return fence("spawn_subagent", {"task": "retry"})

    try:
        assert run(first, "first", max_subagents=1)[1].status == "done"
        for thread in first._fleet_threads.values():
            thread.join(5)
        assert all(row["status"] == "done" for row in db.list_runs("first"))
        assert (
            capacity.snapshot().child_executions
            == capacity.snapshot().stopping_children
            == 1
        )
        second, chat, _ = service_for(
            db,
            capacity,
            [
                fence("spawn_subagent", {"task": "refused"}),
                release_worker,
                fence("wait_agents", {}),
                "done",
            ],
            {"retry": ["retry done"]},
        )
        assert run(second, "second", max_subagents=1)[1].status == "done"
        assert "runtime sub-agent limit reached" in str(chat.parent_calls)
        rows = [
            row for row in db.list_runs("second") if row["agent_kind"] == "subagent"
        ]
        assert len(rows) == 1
        assert (
            rows[0]["task"] == "retry"
        )  # Refusal creates no row or spent spawn allowance.
        for thread in second._fleet_threads.values():
            thread.join(5)
    finally:
        tool.release.set()
        for worker in tool.workers:
            worker.join(5)
        db.close()
    assert capacity.snapshot().executions == ()


@pytest.mark.parametrize("failure_site", ["construct", "start"])
def test_failed_thread_start_releases_capacity_and_spawn_allowance(
    tmp_path, monkeypatch, failure_site
):
    capacity = RuntimeCapacity(max_child_executions=1, reserved_manual_children=0)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="admission")
    service, _, _ = service_for(
        db,
        capacity,
        [
            fence("spawn_subagent", {"task": "failed"}),
            fence("spawn_subagent", {"task": "retry"}),
            fence("wait_agents", {}),
            "done",
        ],
        {"retry": ["done"]},
    )
    original_thread = threading.Thread
    original_start = threading.Thread.start
    failed = []

    def construct(*args, **kwargs):
        if kwargs.get("name", "").startswith("fleet-") and not failed:
            failed.append(True)
            raise RuntimeError("thread unavailable")
        return original_thread(*args, **kwargs)

    def start(thread):
        if thread.name.startswith("fleet-") and not failed:
            failed.append(True)
            raise RuntimeError("thread unavailable")
        return original_start(thread)

    if failure_site == "construct":
        monkeypatch.setattr(threading, "Thread", construct)
    else:
        monkeypatch.setattr(threading.Thread, "start", start)
    try:
        assert run(service, "c", max_subagents=1)[1].status == "done"
        assert failed == [True]
        rows = [row for row in db.list_runs("c") if row["agent_kind"] == "subagent"]
        assert len(rows) == 1 and rows[0]["task"] == "retry"
    finally:
        for thread in service._fleet_threads.values():
            thread.join(5)
        db.close()
    assert capacity.snapshot().executions == ()


def test_inline_model_start_failure_releases_capacity_and_spawn_allowance(
    tmp_path, monkeypatch
):
    pin_agent_settings(monkeypatch, max_live_subagents=1)
    capacity = RuntimeCapacity(max_child_executions=1, reserved_manual_children=0)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="inline-admission")
    starts = []

    @contextmanager
    def model_scope():
        starts.append(True)
        if len(starts) == 1:
            raise RuntimeError("driver cannot start")
        yield

    chat = FleetChat(
        [
            fence("spawn_subagent", {"task": "failed"}),
            fence("spawn_subagent", {"task": "retry"}),
            "done",
        ],
        {"retry": ["child done"]},
    )
    service = AgentService(
        db,
        ToolCatalogRegistry(),
        chat_call=chat,
        runtime_capacity=capacity,
        inline_child_model_scope=model_scope,
    )
    try:
        assert run(service, "c", max_subagents=1)[1].status == "done"
        rows = [row for row in db.list_runs("c") if row["agent_kind"] == "subagent"]
        assert len(rows) == 1 and rows[0]["task"] == "retry"
    finally:
        db.close()
    assert capacity.snapshot().executions == ()


def test_inline_mode_cannot_bypass_runtime_admission(tmp_path, monkeypatch):
    pin_agent_settings(monkeypatch, max_live_subagents=1)
    capacity = RuntimeCapacity(max_child_executions=1, reserved_manual_children=0)
    blocker = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="other", child=True
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="inline-admission")

    def release_then_retry():
        assert [
            row for row in db.list_runs("c") if row["agent_kind"] == "subagent"
        ] == []
        blocker.finish_root()
        return fence("spawn_subagent", {"task": "retry"})

    chat = FleetChat(
        [fence("spawn_subagent", {"task": "refused"}), release_then_retry, "done"],
        {"retry": ["child done"]},
    )
    service = AgentService(
        db, ToolCatalogRegistry(), chat_call=chat, runtime_capacity=capacity
    )
    try:
        assert run(service, "c", max_subagents=1)[1].status == "done"
        rows = [row for row in db.list_runs("c") if row["agent_kind"] == "subagent"]
        assert len(rows) == 1 and rows[0]["task"] == "retry"
    finally:
        blocker.finish_root()
        db.close()
    assert capacity.snapshot().executions == ()
