"""Explicit design measurements, not a regression gate for the old policy.

Run: pytest Tests/Chat/fleet_delivery_probe.py -q -s
The REPORT lines measure current behavior without asserting that its gaps persist.
TASK-32020/32021; prospective regression cases are in ADR-135's crash matrix.
"""

import asyncio
import json
import threading
import time

import pytest

from Tests.Chat.test_child_run_scope_ordering import _survivor_bridge
from Tests.Chat.test_console_agent_bridge import (
    SUBAGENT_PROMPT_PREFIX,
    _fence,
    _join_fleet_threads,
    _run,
)
from Tests.Chat.test_console_agent_swap import (
    FakeMCPService,
    _catalog_record,
    _controller,
    _disable_project_instructions_for_legacy_agent_swap_tests,  # noqa: F401
    _fake_app,
    _tool_dict,
)
from Tests.Chat.test_console_fleet_wake import (
    _AppStub,
    _controller_rig,
    _drain,
    _RecordingWakeGateway,
    _settle,
    _survivor,
    _terminal_subagent_run,
)
from Tests.Chat.test_console_fleet_wake_safety import (
    _mcp_tests_keep_a_small_catalog,  # noqa: F401
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def report(scenario, **measurements):
    print("REPORT " + json.dumps({"scenario": scenario, **measurements}))


@pytest.mark.asyncio
async def test_measure_ready_child_delay(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="probe")
    gates = [threading.Event(), threading.Event()]
    _, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "first"})],
            [_fence("spawn_subagent", {"task": "second"})],
            ["parent returned"],
        ],
        needed=2,
    )
    original = gateway.stream_chat

    async def stream(resolution, messages, **kwargs):
        if messages and str(messages[0].get("content", "")).startswith(
            SUBAGENT_PROMPT_PREFIX
        ):
            with gateway._count_lock:
                index = gateway.child_calls
                gateway.child_calls += 1
                if gateway.child_calls == 2:
                    gateway.entered_event.set()
            await asyncio.get_running_loop().run_in_executor(None, gates[index].wait)
            yield f"child {index} result"
        else:
            async for chunk in original(resolution, messages, **kwargs):
                yield chunk

    gateway.stream_chat = stream
    wake_gateway = _RecordingWakeGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=wake_gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=False,
    )
    controller.fleet_wake.wire(app=_AppStub(chacha))
    drained = []
    bridge.on_fleet_drained("probe", drained.append)
    try:
        assert (
            _run(bridge, store, session, aid, conversation_id=session.id).status
            == "done"
        )
        assert gateway.entered_event.wait(5)
        gates[0].set()
        assert await _settle(
            lambda: any(
                row["status"] == "done"
                for row in db.list_runs(session.id, agent_kind="subagent")
            )
        )
        start = time.monotonic()
        await asyncio.sleep(0.35)
        before = len(wake_gateway.payloads)
        drain_before = len(drained)
        held = time.monotonic() - start
        gates[1].set()
        assert await _settle(lambda: wake_gateway.payloads)
        report(
            "slow_sibling",
            observed_hold_seconds=round(held, 3),
            wake_calls_while_sibling_held=before,
            drain_events_while_sibling_held=drain_before,
            wake_calls_after_release=len(wake_gateway.payloads),
        )
        assert await _settle(lambda: not controller.fleet_wake.has_pending(session.id))
    finally:
        for gate in gates:
            gate.set()
        _join_fleet_threads()
        controller._disposed = True
        chacha.close()


@pytest.mark.asyncio
async def test_measure_other_conversation_delay_during_real_approval(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="probe")
    controller, store, db = _controller(
        tmp_path,
        [
            [_fence("mcp__srv__run", {"x": 1})],
            ["first reply"],
            ["second reply"],
        ],
    )
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])]
    )
    controller.app = _fake_app(service)
    cards = []
    controller.set_pending_approval = cards.append
    controller.mcp_approval_timeout_seconds = lambda: 10.0
    wake = controller.fleet_wake
    wake.wire(app=_AppStub(chacha))
    first = store.ensure_session()
    round_id = None
    try:
        _, first_run = _terminal_subagent_run(db, first.id)
        wake.on_fleet_drained(
            _drain(first.id, _survivor(first_run, session_id=first.id))
        )
        assert await _settle(lambda: cards and cards[-1])
        round_id = cards[-1]["round_id"]
        second = store.create_session(title="second")
        _, second_run = _terminal_subagent_run(db, second.id)
        start = time.monotonic()
        wake.on_fleet_drained(
            _drain(second.id, _survivor(second_run, session_id=second.id))
        )
        await asyncio.sleep(0.35)
        before = len(store.messages_for_session(second.id))
        held = time.monotonic() - start
        with controller._approval_state_lock:
            undecided = not controller._pending_approval_rounds[round_id]["decisions"]
        controller.resolve_pending_approval(
            {"mcp__srv__run": "approve_once"}, round_id=round_id
        )
        round_id = None
        assert await _settle(lambda: db.get_run(second_run).get("wake_delivered_at"))
        report(
            "blocked_approval",
            observed_hold_seconds=round(held, 3),
            second_conversation_rows_while_approval_held=before,
            approval_undecided_until_explicit_resolution=undecided,
            second_conversation_rows_after_release=len(
                store.messages_for_session(second.id)
            ),
        )
    finally:
        if round_id is not None:
            controller.resolve_pending_approval(
                {"mcp__srv__run": "deny"}, round_id=round_id
            )
        controller._disposed = True
        chacha.close()


@pytest.mark.asyncio
async def test_measure_failed_stamp_replay(tmp_path, monkeypatch):
    chacha, _app, db, _, session, gateway, _, controller = _controller_rig(tmp_path)
    wake = controller.fleet_wake
    controller.wake_conversation_in_view = lambda *_: False
    _, run_id = _terminal_subagent_run(db, session.id)
    stamp = db.mark_wake_delivered

    def fail_stamp(*_):
        raise OSError("injected ledger failure")

    monkeypatch.setattr(db, "mark_wake_delivered", fail_stamp)
    try:
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(
            lambda: gateway.payloads and not wake.has_pending(session.id)
        )
        first_calls = len(gateway.payloads)
        unstamped = not db.get_run(run_id).get("wake_delivered_at")
        monkeypatch.setattr(db, "mark_wake_delivered", stamp)
        # Reconstruct from durable rows/marks via the same mount/restart claim.
        seeded = wake.seed_from_marks()
        wake.retry_soon()
        assert await _settle(lambda: not wake.has_pending(session.id))
        report(
            "failed_stamp",
            calls_before_claim=first_calls,
            unstamped_after_finished_turn=unstamped,
            seeded_conversations=seeded,
            calls_after_claim=len(gateway.payloads),
        )
    finally:
        controller._disposed = True
        chacha.close()
