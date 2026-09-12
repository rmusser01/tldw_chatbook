"""First-use admission stays shared and fenced without startup allocation."""

import asyncio
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from Tests.Chat.test_console_agent_bridge import _bridge
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


def test_parallel_first_use_and_replacement_share_one_capacity(tmp_path):
    bridge, db, store, _, _ = _bridge(tmp_path, [["done"]])
    runtime = ConsoleRuntime(app=None)
    runtime.set_agent_bridge(bridge)
    assert runtime._execution_capacity is bridge._runtime_capacity is None
    barrier = threading.Barrier(8)

    def allocate(index):
        barrier.wait()
        return bridge.runtime_capacity if index % 2 else runtime.execution_capacity

    with ThreadPoolExecutor(max_workers=8) as pool:
        capacities = list(pool.map(allocate, range(8)))
    assert all(capacity is capacities[0] for capacity in capacities)
    replacement = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=object()
    )
    runtime.set_agent_bridge(replacement)
    assert replacement.runtime_capacity is bridge.runtime_capacity is capacities[0]
    asyncio.run(runtime.dispose())
    assert capacities[0].snapshot().closed


def test_active_capacity_cannot_be_rebound(tmp_path):
    from tldw_chatbook.Agents.agent_models import WorkOrigin

    bridge, _, _, _, _ = _bridge(tmp_path, [["done"]])
    capacity = bridge.runtime_capacity
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="conversation"
    )
    try:
        with pytest.raises(ValueError, match="active bridge"):
            ConsoleRuntime(app=None).set_agent_bridge(bridge)
        assert bridge.runtime_capacity is capacity
    finally:
        owner.finish_root()


def test_unused_replaced_bridge_and_disposed_runtime_never_allocate(tmp_path):
    bridge, db, store, _, _ = _bridge(tmp_path, [["done"]])
    runtime = ConsoleRuntime(app=None)
    runtime.set_agent_bridge(bridge)
    replacement = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=object()
    )
    runtime.set_agent_bridge(replacement)
    with pytest.raises(RuntimeError, match="closed"):
        _ = bridge.runtime_capacity
    assert bridge._runtime_capacity is runtime._execution_capacity is None
    asyncio.run(runtime.dispose())
    with pytest.raises(RuntimeError, match="disposed"):
        _ = runtime.execution_capacity
    with pytest.raises(RuntimeError, match="closed"):
        _ = replacement.runtime_capacity
    assert runtime._execution_capacity is replacement._runtime_capacity is None


def test_core_bridge_runtime_construction_does_not_import_capacity():
    code = """
import sys
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
runtime = ConsoleRuntime(app=None)
bridge = ConsoleAgentBridge(agent_runs_db=AgentRunsDB(":memory:", client_id="lazy"), store=ConsoleChatStore(), provider_gateway=object())
runtime.set_agent_bridge(bridge)
assert "tldw_chatbook.Agents.execution_capacity" not in sys.modules
assert runtime._execution_capacity is bridge._runtime_capacity is None
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_disposal_wins_against_blocked_first_allocation():
    runtime = ConsoleRuntime(app=None)
    entered = threading.Event()
    errors = []

    def allocate():
        entered.set()
        try:
            _ = runtime.execution_capacity
        except RuntimeError as exc:
            errors.append(exc)

    with runtime._execution_capacity_lock:
        worker = threading.Thread(target=allocate)
        worker.start()
        assert entered.wait(2)
        asyncio.run(runtime.dispose())
    worker.join(2)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert runtime._execution_capacity is None


def test_bridge_close_wins_against_blocked_first_allocation(tmp_path):
    bridge, _, _, _, _ = _bridge(tmp_path, [["done"]])
    entered = threading.Event()
    errors = []

    def allocate():
        entered.set()
        try:
            _ = bridge.runtime_capacity
        except RuntimeError as exc:
            errors.append(exc)

    with bridge._runtime_capacity_lock:
        worker = threading.Thread(target=allocate)
        worker.start()
        assert entered.wait(2)
        bridge.close_all_progress()
    worker.join(2)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert bridge._runtime_capacity is None


@pytest.mark.parametrize("entrypoint", ["begin_dispose", "dispose"])
def test_shutdown_latch_waits_for_canvas_publication_lock(entrypoint):
    runtime = ConsoleRuntime(app=None)
    original_lock = runtime._canvas_native_lock
    waiting = threading.Event()
    errors = []

    class ObservedLock:
        def __enter__(self):
            waiting.set()
            original_lock.acquire()

        def __exit__(self, *_args):
            original_lock.release()

    runtime._canvas_native_lock = ObservedLock()

    def shutdown():
        try:
            if entrypoint == "dispose":
                asyncio.run(runtime.dispose())
            else:
                runtime.begin_dispose()
        except Exception as exc:  # noqa: BLE001 - surface worker failures on the test thread
            errors.append(exc)

    with original_lock:
        worker = threading.Thread(target=shutdown)
        worker.start()
        observed_publication_lock = waiting.wait(2)
        published_disposal_early = runtime._disposed
    worker.join(2)
    assert not worker.is_alive()
    assert errors == []
    assert observed_publication_lock
    assert not published_disposal_early
    assert runtime._disposed
    assert runtime._execution_capacity is None
    with pytest.raises(RuntimeError, match="disposed"):
        _ = runtime.execution_capacity
