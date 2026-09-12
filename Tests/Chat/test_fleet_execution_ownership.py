"""Physical model cleanup remains accounted after a bounded join returns."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
from tldw_chatbook.Chat import console_agent_bridge
from tldw_chatbook.Chat.console_agent_bridge import _ModelCallLifeline
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


def test_slow_lifeline_cleanup_remains_owned_after_root_and_bounded_shutdown(
    monkeypatch,
):
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.AUTOMATIC, conversation_id="c")
    entered = threading.Event()
    release = threading.Event()

    async def close_pool():
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)

    monkeypatch.setattr(console_agent_bridge, "_LOOP_THREAD_JOIN_SECONDS", 0.01)
    lifeline = _ModelCallLifeline("owned-model", close_current_loop=close_pool)
    try:
        lifeline.start(owner=owner)
        lifeline.shutdown()
        assert entered.wait(1)
        owner.finish_root()
        lifeline.shutdown()
        snapshot = capacity.snapshot()
        assert snapshot.model_lifelines == snapshot.stopping_model_lifelines == 1
        assert snapshot.executions[0].root_finished
    finally:
        release.set()
        lifeline.shutdown()
        if lifeline._thread.ident is not None:
            lifeline._thread.join(5)
    assert capacity.snapshot().executions == ()


def test_failed_model_driver_start_releases_without_waiting_for_shutdown(monkeypatch):
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    lifeline = _ModelCallLifeline("failed-model")

    def fail_start(self):
        raise RuntimeError("thread exhausted")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    try:
        with pytest.raises(RuntimeError, match="thread exhausted"):
            lifeline.start(owner=owner)
        assert capacity.snapshot().model_lifelines == 0
    finally:
        lifeline.shutdown()
        owner.finish_root()
    assert capacity.snapshot().executions == ()


def test_duplicate_model_start_does_not_replace_the_live_operation():
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    lifeline = _ModelCallLifeline("duplicate-model")
    try:
        lifeline.start(owner=owner)
        with pytest.raises(RuntimeError, match="cannot be restarted"):
            lifeline.start(owner=owner)
        assert capacity.snapshot().model_lifelines == 1
    finally:
        lifeline.shutdown()
        owner.finish_root()
    assert capacity.snapshot().executions == ()


def test_inline_child_keeps_its_delayed_model_cleanup_without_fleet_counters(
    tmp_path, monkeypatch
):
    from Tests.Agents.conftest import pin_agent_settings
    from Tests.Agents.test_agent_service import fence
    from Tests.Chat.test_console_agent_bridge import (
        _bridge_with_gateway,
        _FleetTwoChildGateway,
        _run,
    )

    pin_agent_settings(monkeypatch, max_live_subagents=1)
    gate = threading.Event()
    gate.set()
    release = threading.Event()
    drivers = []
    gateway = _FleetTwoChildGateway(
        [fence("spawn_subagent", {"task": "inline"}), "done"],
        "child done",
        gate,
        needed=1,
    )

    async def close_pool():
        thread = threading.current_thread()
        if thread.name.startswith("child-loop-"):
            drivers.append(thread)
            while not release.is_set():
                await asyncio.sleep(0.005)

    gateway.aclose_current_loop = close_pool
    monkeypatch.setattr(console_agent_bridge, "_LOOP_THREAD_JOIN_SECONDS", 0.01)
    bridge, db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    try:
        assert _run(bridge, store, session, aid).status == "done"
        snapshot = bridge.runtime_capacity.snapshot()
        assert snapshot.model_lifelines == snapshot.stopping_model_lifelines == 1
        assert snapshot.executions[0].root_finished
        assert db.get_run(snapshot.executions[0].run_id)["agent_kind"] == "subagent"
        assert bridge._unsettled_child_counts.get("conv-1", 0) == 0
        assert bridge._live_child_count("conv-1") == 0
    finally:
        release.set()
        for driver in drivers:
            driver.join(5)
        db.close()
    assert bridge.runtime_capacity.snapshot().executions == ()


@pytest.mark.parametrize("origin", [WorkOrigin.MANUAL, WorkOrigin.AUTOMATIC])
def test_survivor_inherits_origin_and_stays_owned_across_bridge_replacement(
    tmp_path, monkeypatch, origin
):
    from Tests.Agents.conftest import pin_agent_settings
    from Tests.Chat.test_console_agent_bridge import (
        _bridge_with_gateway,
        _CrossTurnGateway,
        _join_fleet_threads,
        _run,
    )

    pin_agent_settings(monkeypatch, subagents_outlive_turn=True)
    release = threading.Event()
    gateway = _CrossTurnGateway(release)
    bridge, db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_agent_bridge(bridge)
    try:
        if origin is WorkOrigin.AUTOMATIC:
            from Tests.Agents.test_automatic_child_scope import accepted_context

            context = accepted_context(db)
            chain_id = context.chain_id
            with context.scope():
                outcome = _run(
                    bridge,
                    store,
                    session,
                    aid,
                    work_origin=origin,
                    work_chain_id=chain_id,
                    conversation_id="conversation",
                )
        else:
            outcome = _run(bridge, store, session, aid, work_origin=origin)
        assert outcome.status == "done"
        snapshot = runtime.execution_capacity.snapshot()
        assert snapshot.model_lifelines == 1
        assert len(snapshot.executions) == 1
        survivor = snapshot.executions[0]
        assert survivor.origin is origin
        assert db.get_run(survivor.run_id)["agent_kind"] == "subagent"
        runtime.set_agent_bridge(None)
        replacement = console_agent_bridge.ConsoleAgentBridge(
            agent_runs_db=db, store=store, provider_gateway=gateway
        )
        runtime.set_agent_bridge(replacement)
        assert replacement.runtime_capacity.snapshot() == snapshot
        assert bridge.runtime_capacity is replacement.runtime_capacity
        asyncio.run(runtime.dispose())
        assert runtime.execution_capacity.snapshot().closed
        assert runtime.execution_capacity.snapshot().model_lifelines == 1
    finally:
        release.set()
        _join_fleet_threads(timeout=10)
        db.close()
    assert runtime.execution_capacity.snapshot().executions == ()
