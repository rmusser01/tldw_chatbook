"""Capacity follows live operations, including workers whose callers left."""

import threading

import pytest

from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Agents.agent_service import _call_with_timeout
from tldw_chatbook.Agents.execution_capacity import (
    CapacityRefused,
    RuntimeCapacity,
    WorkOrigin,
)


def test_automatic_workers_leave_two_slots_for_manual_work_across_conversations():
    capacity = RuntimeCapacity()
    owners = [
        capacity.begin_execution(origin=WorkOrigin.AUTOMATIC, conversation_id=str(i))
        for i in range(7)
    ]
    operations = [owner.reserve_tool() for owner in owners[:6]]
    with pytest.raises(CapacityRefused, match="automatic_tool_capacity"):
        owners[6].reserve_tool()
    for i in range(2):
        owner = capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id=f"m{i}"
        )
        owners.append(owner)
        operations.append(owner.reserve_tool())
    with pytest.raises(CapacityRefused, match="tool_capacity"):
        owners[6].reserve_tool()
    for owner in owners:
        owner.finish_root()
    assert capacity.snapshot().tool_workers == 8
    for operation in operations:
        operation.finish()
        operation.finish()
    assert capacity.snapshot().executions == ()


@pytest.mark.parametrize("cancel", [False, True])
def test_abandoned_worker_keeps_slot_and_blocks_its_run_until_real_completion(cancel):
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.AUTOMATIC, conversation_id="c")
    owner.bind_run("r")
    gate = threading.Event()
    workers = []

    def tool():
        workers.append(threading.current_thread())
        assert gate.wait(5)
        return ToolResult(ok=True, content="late")

    try:
        result = _call_with_timeout(tool, 0.02, "slow", lambda: cancel, owner=owner)
        assert not result.ok
        assert ("cancelled" if cancel else "timed out") in result.error
        snapshot = capacity.snapshot()
        assert snapshot.tool_workers == snapshot.stopping_tool_workers == 1
        assert snapshot.executions[0].run_id == "r"
        assert not _call_with_timeout(
            lambda: pytest.fail("retry ran"), 1, "retry", owner=owner
        ).ok
        owner.finish_root()
        assert capacity.snapshot().tool_workers == 1
    finally:
        gate.set()
        for worker in workers:
            worker.join(5)
    assert capacity.snapshot().executions == ()


@pytest.mark.parametrize("root_first", [True, False])
def test_drain_callback_runs_once_outside_lock_after_physical_completion(root_first):
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    operation = owner.reserve_tool()
    observed = []
    owner.on_drained(lambda proven: observed.append((proven, capacity.snapshot())))
    if root_first:
        owner.finish_root()
        assert observed == []
        operation.finish()
    else:
        operation.finish()
        assert observed == []
        owner.finish_root()
    owner.finish_root()
    operation.finish()
    assert len(observed) == 1
    assert observed[0][0] is True
    assert observed[0][1].executions == ()


def test_drain_callbacks_isolate_failures_and_late_registration_sees_outcome():
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    observed = []

    def fail(_proven):
        raise RuntimeError("callback failed")

    owner.on_drained(fail)
    owner.on_drained(observed.append)
    owner.finish_root()
    owner.on_drained(lambda proven: observed.append(("late", proven)))
    assert observed == [True, ("late", True)]


def test_cleanup_unproven_is_sticky_through_actual_worker_finally():
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    gate = threading.Event()
    workers = []
    observed = []

    def tool():
        workers.append(threading.current_thread())
        assert gate.wait(5)
        return ToolResult(ok=True, content="late")

    try:
        result = _call_with_timeout(tool, 0.02, "slow", lambda: False, owner=owner)
        assert not result.ok
        owner.on_drained(observed.append)
        owner.mark_cleanup_unproven()
        owner.mark_cleanup_unproven()
        owner.finish_root()
        assert observed == []
    finally:
        gate.set()
        for worker in workers:
            worker.join(5)
    assert observed == [False]
    owner.on_drained(lambda proven: observed.append(proven))
    assert observed == [False, False]


def test_thread_start_failure_and_repeated_release_do_not_lose_capacity(monkeypatch):
    capacity = RuntimeCapacity(max_tool_workers=1, reserved_manual_tool_workers=0)
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")

    def fail_start(self):
        raise RuntimeError("thread exhausted")

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", fail_start)
        result = _call_with_timeout(
            lambda: pytest.fail("not started"), 1, "tool", owner=owner
        )
    assert not result.ok
    assert capacity.snapshot().tool_workers == 0
    assert _call_with_timeout(lambda: ToolResult(ok=True), 1, "tool", owner=owner).ok
    owner.finish_root()
    owner.finish_root()
    assert capacity.snapshot().executions == ()


def test_closing_admission_retains_occupied_operations():
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    operation = owner.reserve_tool()
    capacity.close()
    with pytest.raises(CapacityRefused, match="runtime_closed"):
        capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="new")
    with pytest.raises(CapacityRefused, match="runtime_closed"):
        owner.reserve_tool()
    owner.finish_root()
    assert capacity.snapshot().tool_workers == 1
    operation.finish()
    assert capacity.snapshot().executions == ()


def test_simultaneous_admission_cannot_oversubscribe_the_last_slots():
    capacity = RuntimeCapacity()
    barrier = threading.Barrier(12)
    operations = []
    owners = []
    refused = []

    def reserve(i):
        owner = capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id=str(i)
        )
        owners.append(owner)
        barrier.wait(3)
        try:
            operations.append(owner.reserve_tool())
        except CapacityRefused as exc:
            refused.append(str(exc))

    threads = [threading.Thread(target=reserve, args=(i,)) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert len(operations) == 8
    assert refused == ["tool_capacity"] * 4
    for operation in operations:
        operation.finish()
    for owner in owners:
        owner.finish_root()


def test_lowered_limits_hold_until_actual_release_and_invalid_reserves_are_safe():
    capacity = RuntimeCapacity(max_tool_workers=2, reserved_manual_tool_workers=False)
    automatic = capacity.begin_execution(
        origin=WorkOrigin.AUTOMATIC, conversation_id="a"
    )
    with pytest.raises(CapacityRefused):  # bool uses the default reserve of two
        automatic.reserve_tool()
    manual = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="m")
    operation = manual.reserve_tool()
    capacity.set_tool_limits(1, -1)  # clamp reserve to zero
    with pytest.raises(CapacityRefused):
        automatic.reserve_tool()
    assert capacity.snapshot().tool_workers == 1
    operation.finish()
    automatic.reserve_tool().finish()
    automatic.finish_root()
    manual.finish_root()


def test_runtime_reads_changed_environment_limits_before_next_tool(monkeypatch):
    monkeypatch.setenv("TLDW_AGENTS_MAX_RUNTIME_TOOL_WORKERS", "2")
    monkeypatch.setenv("TLDW_AGENTS_RESERVED_MANUAL_TOOL_WORKERS", "0")
    capacity = RuntimeCapacity.from_settings()
    owner = capacity.begin_execution(origin=WorkOrigin.AUTOMATIC, conversation_id="c")
    first = owner.reserve_tool()
    monkeypatch.setenv("TLDW_AGENTS_MAX_RUNTIME_TOOL_WORKERS", "1")
    with pytest.raises(CapacityRefused, match="tool_capacity"):
        owner.reserve_tool()
    first.finish()
    owner.reserve_tool().finish()
    owner.finish_root()


def test_child_limit_reserves_manual_slots_and_retains_terminal_cleanup():
    capacity = RuntimeCapacity()
    children = [
        capacity.begin_execution(
            origin=WorkOrigin.AUTOMATIC, conversation_id=str(i), child=True
        )
        for i in range(4)
    ]
    with pytest.raises(CapacityRefused, match="automatic_child_capacity"):
        capacity.begin_execution(
            origin=WorkOrigin.AUTOMATIC, conversation_id="blocked", child=True
        )
    children.extend(
        capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id=str(i), child=True
        )
        for i in range(2)
    )
    operation = children[0].reserve_model()
    operation.mark_stopping()
    children[0].finish_root()
    assert capacity.snapshot().child_executions == 6
    assert capacity.snapshot().stopping_children == 1
    with pytest.raises(CapacityRefused, match="child_capacity"):
        capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id="blocked", child=True
        )
    # Primary runs consume no child slot, even when all six are occupied.
    capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="parent"
    ).finish_root()
    operation.finish()
    replacement = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="replacement", child=True
    )
    replacement.finish_root()
    for child in children:
        child.finish_root()
    assert capacity.snapshot().executions == ()


def test_concurrent_child_admission_is_atomic_and_lowering_keeps_owners():
    capacity = RuntimeCapacity()
    barrier = threading.Barrier(10)
    admitted = []
    refused = []

    def launch(i):
        barrier.wait(3)
        try:
            admitted.append(
                capacity.begin_execution(
                    origin=WorkOrigin.MANUAL, conversation_id=str(i), child=True
                )
            )
        except CapacityRefused as exc:
            refused.append(str(exc))

    threads = [threading.Thread(target=launch, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert len(admitted) == 6
    assert refused == ["child_capacity"] * 4
    capacity.set_child_limits(1, 0)
    assert capacity.snapshot().child_executions == 6
    for child in admitted[:-1]:
        child.finish_root()
    with pytest.raises(CapacityRefused):
        capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id="still full", child=True
        )
    admitted[-1].finish_root()
    capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="free", child=True
    ).finish_root()


@pytest.mark.parametrize(
    "invalid", [True, False, 0, -1, 1.5, float("inf"), "1.5", "NaN", None]
)
def test_invalid_total_uses_default_instead_of_disabling_limits(invalid):
    capacity = RuntimeCapacity(max_tool_workers=invalid)
    owners = [
        capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id=str(i))
        for i in range(9)
    ]
    operations = [owner.reserve_tool() for owner in owners[:8]]
    with pytest.raises(CapacityRefused):
        owners[8].reserve_tool()
    for operation in operations:
        operation.finish()
    for owner in owners:
        owner.finish_root()
