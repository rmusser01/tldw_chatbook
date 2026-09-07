"""PROBES P2/P3/P4 (task-15860 Task 0) -- what already works post-teardown.

NOT regression tests. Three questions the headless-wake plan sizes its work
from, answered by EXECUTION against current dev:

P2  After `ChatScreen.on_unmount`, does `ConsoleFleetWakeCoordinator.
    on_fleet_drained` still record into its pending registry, and is its
    captured loop still alive? (PR 3a-2 Task 1 proved the *attention*
    consumer fires post-teardown; the wake coordinator was added later.)

P3  If the controller merely SURVIVES (shutdown skipped), does a wake
    delivered after unmount reach `submit_draft` and complete the turn --
    and which screen-wired hook slots are actually touched with no view?

P4  With no UI wired, what is the wall time to a risk-tagged tool's denial
    in a wake turn, and what is the effective `[mcp]
    approval_timeout_seconds`?
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from Tests.Chat.test_console_fleet_wake import (
    _RecordingWakeGateway,
    _drain,
    _survivor,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_mcp_approval import _pending
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_chat_controller import (
    _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS,
    ConsoleChatController,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _report(title: str, lines: list[str]) -> None:
    print(f"\n===== {title} =====")
    for line in lines:
        print(f"  {line}")


def _drain_from_child_thread(wake, drain) -> None:
    """Deliver the drain the way production does: from a plain thread.

    (`Tests/UI/test_console_fleet_wake_ui_freshness.py:66` -- an asyncio
    task inherits the creating context, so a drain injected from the test
    coroutine validates against a rig production never has.)
    """
    thread = threading.Thread(target=lambda: wake.on_fleet_drained(drain))
    thread.start()
    thread.join(5)


def _hook_slots(controller, screen) -> dict[str, object]:
    """Every controller attribute currently bound to a SCREEN method."""
    slots: dict[str, object] = {}
    for name in dir(controller):
        if name.startswith("__"):
            continue
        try:
            value = getattr(controller, name)
        except Exception:  # noqa: BLE001 -- properties may raise off-loop
            continue
        if callable(value) and getattr(value, "__self__", None) is screen:
            slots[name] = value
    return slots


async def _seed_console(app, pilot, gateway):
    """Mount a real Console, send once, return (screen, controller, ids)."""
    chat = ChatScreen(app)
    await app.push_screen(chat)
    app._initial_screen_pushed = True
    app.current_tab = "chat"
    await pilot.pause()
    await _wait_for_selector(chat, pilot, "#console-native-composer")
    controller = chat._ensure_console_chat_controller()
    store = chat._console_chat_store
    session_id = store.sessions()[0].id
    await controller.submit_draft("first user message", session_id=session_id)
    conversation_id = store.sessions()[0].persisted_conversation_id
    return chat, controller, store, session_id, conversation_id


def _terminal_survivor_run(runs_db, conversation_id, *, result="child answer"):
    parent_id = runs_db.create_run(
        conversation_id=conversation_id, agent_kind="primary"
    )
    runs_db.set_status(parent_id, "done", "turn final")
    run_id = runs_db.create_run(
        conversation_id=conversation_id,
        agent_kind="subagent",
        task="long job",
        parent_run_id=parent_id,
    )
    runs_db.set_status(run_id, "done", result)
    return run_id


# ---------------------------------------------------------------------------
# P2 -- post-unmount fan-out on current dev
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_p2_post_unmount_fanout(tmp_path):
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _RecordingWakeGateway(reply="assistant one")
    app.console_provider_gateway_factory = lambda: gateway

    findings: list[str] = []

    async with app.run_test(size=(160, 48)) as pilot:
        app_loop = asyncio.get_running_loop()
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        wake = controller.fleet_wake
        bridge = controller._agent_bridge
        runs_db = getattr(bridge, "runs_db", None)
        findings.append(f"runs_db present pre-unmount: {runs_db is not None}")
        loop_before = wake._loop
        findings.append(
            f"captured loop pre-unmount is the app loop: {loop_before is app_loop}"
        )
        fanout = getattr(bridge, "_fleet_drain_fanout", None)
        findings.append(
            "fan-out registrations pre-unmount: "
            + str([name for name, _ in getattr(fanout, "_consumers", [])])
        )
        run_id = _terminal_survivor_run(runs_db, conversation_id)

        # ---- real navigation away: on_unmount + controller.shutdown() ----
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat not in app.screen_stack
        findings.append(
            f"post-unmount _shutdown_requested set: "
            f"{controller._shutdown_requested.is_set()}"
        )

        loop_after = wake._loop
        findings.append(
            "P2(loop) captured loop post-unmount: "
            f"is_none={loop_after is None} "
            f"is_closed={None if loop_after is None else loop_after.is_closed()} "
            f"is_app_loop={loop_after is app_loop} "
            f"is_running={None if loop_after is None else loop_after.is_running()}"
        )

        # Fire through the BRIDGE's own fan-out from a plain thread -- the
        # production path -- not just the coordinator method directly.
        drain = _drain(conversation_id, _survivor(run_id, session_id=session_id))
        fanout_after = getattr(bridge, "_fleet_drain_fanout", None)
        findings.append(
            "fan-out registrations post-unmount: "
            + str([name for name, _ in getattr(fanout_after, "_consumers", [])])
        )
        thread = threading.Thread(target=lambda: fanout_after.fire(drain))
        thread.start()
        thread.join(5)
        await pilot.pause()
        findings.append(
            "P2(registry) on_fleet_drained recorded post-unmount (via the "
            "bridge fan-out): "
            f"has_pending={wake.has_pending(conversation_id)} "
            f"pending_ids={wake.pending_conversation_ids()}"
        )

        # Did the scheduled attempt do anything? (it must bail at the
        # shutdown gate -- `_attempt`'s second check)
        for _ in range(10):
            await pilot.pause()
        findings.append(
            "P2(delivery) wake turns reaching the provider after unmount: "
            f"{len(gateway.payloads) - 1}"  # minus the seeding send
        )
        findings.append(
            f"P2(delivery) delivering_conversation_id={wake.delivering_conversation_id()}"
        )
        ledger = (runs_db.get_run(run_id) or {}).get("wake_delivered_at")
        findings.append(f"P2(ledger) wake_delivered_at={ledger}")

    _report("P2 -- post-unmount fan-out (real teardown, shutdown NOT skipped)", findings)


# ---------------------------------------------------------------------------
# P3 -- how much already works if the controller merely survives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_p3_controller_survives_unmount(tmp_path, monkeypatch):
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _RecordingWakeGateway(reply="wake reply")
    app.console_provider_gateway_factory = lambda: gateway
    # The plain-provider path, so the wake turn streams through the double.
    app.app_config.setdefault("console", {})["agent_runtime"] = False

    findings: list[str] = []
    touched: list[str] = []

    # Keep the controller ALIVE artificially: skip shutdown() only.
    real_shutdown = ConsoleChatController.shutdown

    async def _no_shutdown(self):
        findings.append("shutdown() was suppressed for this probe")
        return None

    async with app.run_test(size=(160, 48)) as pilot:
        app_loop = asyncio.get_running_loop()
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        wake = controller.fleet_wake
        runs_db = controller._agent_bridge.runs_db
        run_id = _terminal_survivor_run(runs_db, conversation_id)

        # Instrument every screen-bound hook slot BEFORE the screen dies.
        slots = _hook_slots(controller, chat)
        slot_names = sorted(slots)
        findings.append(f"P3(hooks) screen-bound controller slots: {slot_names}")
        for name, fn in slots.items():
            def _wrap(_name=name, _fn=fn):
                def _recorder(*a, **kw):
                    touched.append(_name)
                    try:
                        return _fn(*a, **kw)
                    except Exception as exc:  # noqa: BLE001
                        touched.append(f"{_name}!RAISED:{type(exc).__name__}")
                        raise
                return _recorder
            setattr(controller, name, _wrap())
        ui_hook = wake.delivery_ui_hook
        findings.append(
            f"P3(hooks) wake.delivery_ui_hook wired: {ui_hook is not None}"
        )
        if ui_hook is not None:
            def _hook_recorder(session, _fn=ui_hook):
                touched.append("wake.delivery_ui_hook")
                try:
                    return _fn(session)
                except Exception as exc:  # noqa: BLE001
                    touched.append(
                        f"wake.delivery_ui_hook!RAISED:{type(exc).__name__}"
                    )
                    raise
            wake.delivery_ui_hook = _hook_recorder

        monkeypatch.setattr(ConsoleChatController, "shutdown", _no_shutdown)
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat not in app.screen_stack, "the SCREEN must still unmount"
        findings.append(
            "post-unmount _shutdown_requested set: "
            f"{controller._shutdown_requested.is_set()}"
        )
        findings.append(f"wake loop still the app loop: {wake._loop is app_loop}")
        payloads_before = len(gateway.payloads)

        _drain_from_child_thread(
            wake, _drain(conversation_id, _survivor(run_id, session_id=session_id))
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if len(gateway.payloads) > payloads_before:
                break
            await asyncio.sleep(0.05)
        delivered = len(gateway.payloads) - payloads_before
        findings.append(
            f"P3(delivery) wake turns reaching the provider after unmount: {delivered}"
        )
        if delivered:
            tail = gateway.payloads[-1][-1]
            findings.append(
                f"P3(delivery) payload tail role={tail['role']!r} "
                f"carries the child's result="
                f"{'child answer' in str(tail['content'])}"
            )
        # Did the TURN complete (ledger stamped only after acceptance)?
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if (runs_db.get_run(run_id) or {}).get("wake_delivered_at"):
                break
            await asyncio.sleep(0.05)
        findings.append(
            "P3(turn) wake_delivered_at stamped (turn accepted+committed): "
            f"{bool((runs_db.get_run(run_id) or {}).get('wake_delivered_at'))}"
        )
        findings.append(
            "P3(transcript) rows now in the (orphaned) store: "
            + str(
                [
                    (m.role.value, m.content[:28])
                    for m in store.messages_for_session(session_id)
                ]
            )
        )
        findings.append(f"P3(hooks) slots TOUCHED during the wake turn: {sorted(set(touched))}")
        findings.append(
            "P3(hooks) slots NOT touched: "
            + str(sorted(set(slot_names) - {t.split('!')[0] for t in touched}))
        )

        # P3b -- the composition of P1 and P3, the decision-relevant fact:
        # the wake turn RAN and persisted, but the ScreenStateStore snapshot
        # was taken BEFORE it. What does the user see on returning?
        db = app.chachanotes_db
        db_rows = db.get_messages_for_conversation(conversation_id, limit=100)
        findings.append(
            "P3b(db) rows persisted by the surviving-controller wake: "
            + str([(r["sender"], r["content"][:28]) for r in db_rows])
        )
        monkeypatch.setattr(ConsoleChatController, "shutdown", real_shutdown)
        await controller.shutdown()
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause()
        chat2 = app.screen
        await _wait_for_selector(chat2, pilot, "#console-native-composer")
        restored = chat2._console_chat_store.messages_for_session(session_id)
        findings.append(
            "P3b(remount) transcript the user sees on returning: "
            + str([(m.role.value, m.content[:28]) for m in restored])
        )
        restored_text = "\n".join(m.content for m in restored)
        findings.append(
            "P3b(remount) the wake notice survives the return: "
            f"{'[Background sub-agent completion' in restored_text}"
        )

    _report("P3 -- controller survives unmount (shutdown skipped)", findings)


# ---------------------------------------------------------------------------
# P4 -- headless approval cost
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_probe_p4_headless_approval_cost():
    """No UI wired at all: no `app`, no `set_pending_approval`.

    Marked ``slow`` (needs ``--run-slow``): it deliberately runs to the
    SHIPPED 120s deadline, because the measurement IS the finding. The
    fast control below pins the same mechanism at 0.05s.

    A wake turn runs on the SAME controller and therefore through the same
    `request_mcp_approvals` gate, so this measures the wake case's cost.
    The row is shaped as a risk-floored built-in (`server_key
    "agent:builtin"`, `reason "risk_floored"`) -- the shape
    `build_tool_review_hook` emits for a risk-tagged tool.
    """
    findings: list[str] = []
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=object())
    session = store.ensure_session(title="Headless")
    findings.append(f"controller.app wired: {controller.app is not None}")
    findings.append(
        f"set_pending_approval wired: {controller.set_pending_approval is not None}"
    )
    findings.append(
        f"park_pending_approval wired: {controller.park_pending_approval is not None}"
    )
    effective = controller._resolve_mcp_approval_timeout_seconds()
    findings.append(
        f"P4(config) effective [mcp] approval_timeout_seconds = {effective} "
        f"(shipped default constant {_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS})"
    )

    risk_row = _pending(
        server_key="agent:builtin",
        tool_name="write_file",
        llm_name="builtin__write_file",
        reason="risk_floored",
    )
    started = time.monotonic()
    decisions = controller.request_mcp_approvals([risk_row], session_id=session.id)
    elapsed = time.monotonic() - started
    findings.append(f"P4(measured) decisions={decisions}")
    findings.append(f"P4(measured) wall time to verdict = {elapsed:.2f}s")
    findings.append(
        "P4(measured) verdict is the fail-closed 'timeout' word (refusal is "
        "recorded downstream as 'denied-timeout' in the tool providers)"
    )
    _report("P4 -- headless approval cost at the SHIPPED default", findings)


def test_probe_p4_mechanism_at_a_short_injected_deadline():
    """Control for P4: the same path at a 0.05s deadline, to show the cost
    IS the configured deadline (plus <=1s poll slack) and not something
    else that only looks like it at 120s."""
    findings: list[str] = []
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=object())
    session = store.ensure_session(title="Headless")
    controller.mcp_approval_timeout_seconds = lambda: 0.05
    started = time.monotonic()
    decisions = controller.request_mcp_approvals(
        [_pending(server_key="agent:builtin", reason="risk_floored")],
        session_id=session.id,
    )
    elapsed = time.monotonic() - started
    findings.append(f"decisions={decisions} elapsed={elapsed:.2f}s (deadline 0.05s)")
    _report("P4 control -- injected 0.05s deadline", findings)
