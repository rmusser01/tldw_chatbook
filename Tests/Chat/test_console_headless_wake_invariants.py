"""Headless-wake invariants and caps (task-15860, plan Task 1 AC#1/AC#3).

The end-to-end fire is pinned through the real app and the real navigation
API in `Tests/UI/test_console_headless_wake_fires.py`. This file pins the
properties that must hold once it fires, at the controller/runtime seam
where each one can be driven deterministically:

* **disposed vs visit-ended are genuinely different states.**
  `_attempt`'s gate used to read `controller._shutdown_requested`, which
  after the lifetime landing means BOTH "this visit ended" (every
  navigation away) and "the app is exiting". It now reads `_disposed`,
  which means only the second. Both directions are asserted by
  consequence -- a wake that delivers, and a wake that never does -- not
  by reading the flag.
* **AC#2 is not widened.** Relaxing the wake gate must not revive the
  visit's cancellation Event for anything else: a round armed during the
  visit stays denied on the very controller that then delivers a headless
  wake.
* **AC#3 headless**: exactly-once via the `wake_delivered_at` ledger
  across refusal, retry and restart; no phantom wake after a restart for
  a crash-killed child swept to `error`; the `autowake_enabled` kill
  switch read fresh at the headless fire point, losing nothing while OFF.
* **Caps still apply**: `max_parallel_runs`, per-session busy refusal,
  and the shared dispatch that carries the wall clock and token ceiling.

Rig note: "headless" here is produced by the production seam
`ConsoleRuntime.leave_console(view)` -- what `ChatScreen.on_unmount`
calls through `leave_console_runtime`. Every test asserts, as a harness
precondition, that this really did set the visit's Event and really did
NOT dispose the controller; otherwise a green could mean the test never
reached the state the old gate refused.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_fleet_wake import (
    _FakeWakeBridge,
    _RecordingWakeGateway,
    _controller_rig,
    _drain,
    _quiet,
    _settle,
    _survivor,
    _terminal_subagent_run,
)
from Tests.Chat.test_console_runtime_lifetime import _pending_call
from Tests.Chat.test_console_viewless_hooks import _marked, _mounted_view, _runtime_for
from tldw_chatbook.Agents.agent_models import RUN_DONE, RunOutcome
from tldw_chatbook.Chat.console_agent_bridge import (
    CONSOLE_RUN_BUDGET,
    ConsoleAgentBridge,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_fleet_wake import WAKE_NOTICE_HEADER
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


async def _leave_console(rig) -> ConsoleRuntime:
    """Attach a view, then end the visit through the production seam.

    `ConsoleRuntime.leave_console` is exactly what `ChatScreen.on_unmount`
    reaches through `leave_console_runtime`. The two preconditions are the
    point: the visit Event MUST be set (that is the state the old gate
    refused, so a test that skipped it would prove nothing) and the
    controller must NOT be disposed (a navigation is not an app exit).
    """
    controller = rig[-1]
    runtime = _runtime_for(rig)
    view = _mounted_view()
    runtime.attach_view(view)
    assert await runtime.leave_console(view) is True, "the visit never ended"
    assert controller._shutdown_requested.is_set(), (
        "harness precondition: leaving Console must set the visit's Event"
    )
    assert controller._disposed is False, (
        "harness precondition: a navigation must not dispose the runtime"
    )
    return runtime


def _notice_rows(store, session_id):
    return [
        message
        for message in store.messages_for_session(session_id)
        if getattr(message.metadata, "origin", "") == "agent_wake"
    ]


# ---------------------------------------------------------------------------
# 1. Disposed vs visit-ended: two states, two outcomes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_visit_that_merely_ended_does_not_refuse_the_wake(tmp_path):
    """AC#1 at the runtime seam: leaving Console does not silence a wake.

    RED before the gate change -- the visit's Event was the gate, so this
    delivered nothing.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        await _leave_console(rig)
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        app.conversation_local_marks_service.set_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        )
        wake = controller.fleet_wake

        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )

        assert await _settle(lambda: gateway.payloads), (
            "a survivor settled after the Console visit ended and no wake turn "
            "ever reached the provider"
        )
        assert await _settle(
            lambda: bool((runs_db.get_run(run_id) or {}).get("wake_delivered_at"))
        ), "the headless delivery never committed to the ledger"
        assert len(_notice_rows(store, session.id)) == 1
        assert _marked(app, session.id), (
            "nobody could have watched this delivery; the ◈ mark must survive"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_disposed_runtime_refuses_the_wake_and_loses_nothing(tmp_path):
    """The other direction: app exit still refuses -- and refusal is not loss.

    `dispose()` closes the provider gateway and cancels/awaits every
    session's stream task, so a turn started here could reach nobody. The
    pending entry, the durable mark and the unstamped ledger all survive
    for the next process.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        runtime = _runtime_for(rig)
        runtime.attach_view(_mounted_view())
        await runtime.dispose()
        assert controller._disposed is True, "harness precondition: disposed"

        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        app.conversation_local_marks_service.set_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        )
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )

        assert await _quiet(lambda: gateway.payloads, seconds=1.0), (
            "a DISPOSED runtime delivered a wake turn -- its gateway is closed "
            "and its stream tasks are cancelled; nothing it produced could "
            "reach anyone"
        )
        assert wake.has_pending(session.id), "a refused wake keeps its pending bit"
        assert not (runs_db.get_run(run_id) or {}).get("wake_delivered_at"), (
            "a refused wake must never stamp the delivered ledger"
        )
        assert _marked(app, session.id), "a refused wake must not clear the mark"
        assert _notice_rows(store, session.id) == [], (
            "a refused wake left an orphaned notice row"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_disposed_controller_never_reopens_for_a_new_view(tmp_path):
    """`_disposed` is permanent; a visit boundary is not.

    Attaching a fresh view to a disposed runtime must not hand it a new
    visit (`begin_visit` refuses on `_disposed`), so the wake stays
    refused. This is what makes the two signals genuinely different rather
    than two spellings of the same latch.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        runtime = _runtime_for(rig)
        runtime.attach_view(_mounted_view())
        await runtime.dispose()

        runtime.attach_view(_mounted_view())
        assert controller._disposed is True, "dispose must be permanent"
        assert controller._shutdown_requested.is_set(), (
            "a disposed controller must never receive a fresh, unset Event"
        )

        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads, seconds=1.0), (
            "re-attaching a view to a DISPOSED runtime revived the wake"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_the_relaxed_wake_gate_does_not_revive_the_visits_cancellation(
    tmp_path,
):
    """AC#2 is not widened: the same controller denies AND delivers.

    The gate change must not be reachable as "leaving Console no longer
    means anything". On ONE controller, in one run: a round armed during
    the visit resolves to `deny` when the visit ends (AC#2's documented
    semantics, untouched), and a survivor settling immediately afterwards
    still gets its full wake turn (AC#1). If a future edit ever relaxed
    the visit Event itself instead of the gate, the first half goes red.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        controller.mcp_approval_timeout_seconds = lambda: 60.0
        runtime = _runtime_for(rig)
        view = _mounted_view()
        runtime.attach_view(view)

        decisions: dict[str, str] = {}
        armed = threading.Event()

        def _run_round() -> None:
            armed.set()
            decisions.update(
                controller.request_mcp_approvals(
                    [_pending_call()], session_id=session.id
                )
            )

        worker = threading.Thread(target=_run_round, daemon=True)
        worker.start()
        assert armed.wait(timeout=2), "the round never armed"
        await asyncio.sleep(0.2)  # let the poll loop reach its first wait

        assert await runtime.leave_console(view) is True
        worker.join(timeout=10)
        assert not worker.is_alive(), "the round never resolved after leaving"
        assert decisions == {"write_file": "deny"}, (
            "leaving Console must still DENY a parked approval round: "
            f"{decisions}"
        )

        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        controller.fleet_wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: gateway.payloads), (
            "the same controller that denied the round must still deliver the "
            "headless wake"
        )
    finally:
        chacha.close()


# ---------------------------------------------------------------------------
# 2. AC#3 headless: exactly-once, no phantom, the kill switch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exactly_once_across_a_refusal_a_retry_and_a_restart_headless(
    tmp_path,
):
    """The ledger is the exactly-once bit, headless as well as mounted.

    Three legs on one run: a REFUSED headless attempt commits nothing; the
    RETRY delivers once and stamps; and a RESTART -- a fresh store,
    controller and coordinator over the same `agent_runs` file -- claims
    nothing back from the still-set mark, because the ledger says it was
    delivered. The control that stops that last leg being vacuous: a
    second, undelivered run in the same conversation IS claimed, and the
    notice the restarted process composes carries only its result.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        await _leave_console(rig)
        marks = app.conversation_local_marks_service
        marks.set_mark(session.id, ConversationLocalMarksService.FLEET_UNSEEN)
        _parent, run_id = _terminal_subagent_run(
            runs_db, session.id, result="first child result"
        )
        wake = controller.fleet_wake

        # (a) REFUSED: the provider is not ready. Nothing commits.
        gateway.ready = False
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "a not-ready provider must refuse the headless wake before any stream"
        )
        assert wake.has_pending(session.id)
        assert not (runs_db.get_run(run_id) or {}).get("wake_delivered_at")
        assert _marked(app, session.id)
        assert _notice_rows(store, session.id) == []

        # (b) RETRY: delivered exactly once, stamped once.
        gateway.ready = True
        wake.retry_soon()
        assert await _settle(lambda: gateway.payloads), "the retry never delivered"
        assert await _settle(lambda: not wake.has_pending(session.id))
        stamp = (runs_db.get_run(run_id) or {}).get("wake_delivered_at")
        assert stamp, "the retried headless delivery never stamped the ledger"
        wake.retry_soon()
        assert await _quiet(lambda: len(gateway.payloads) > 1, seconds=1.0)
        assert len(_notice_rows(store, session.id)) == 1

        # (c) RESTART: a fresh process over the same durable state.
        restart_store = ConsoleChatStore()
        restart_session = restart_store.ensure_session(title="Research")
        # What `restore_persisted_session` does: bind the session to the
        # conversation it resumes, so `_resolve_session_id` can find it.
        restart_session.persisted_conversation_id = session.id
        restart_gateway = _RecordingWakeGateway(reply="restarted reply")
        restart_controller = ConsoleChatController(
            store=restart_store,
            provider_gateway=restart_gateway,
            agent_bridge=_FakeWakeBridge(runs_db),
            agent_runtime_enabled=False,
        )
        restart_wake = restart_controller.fleet_wake
        restart_wake.wire(app=app)
        assert marks.has_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        ), "harness precondition: the mark outlives the delivery (off-view)"

        assert restart_wake.seed_from_marks() == 0, (
            "the mount claim re-announced a run the ledger already shows "
            "delivered"
        )
        assert restart_wake.pending_conversation_ids() == ()
        restart_wake.retry_soon()
        assert await _quiet(lambda: restart_gateway.payloads, seconds=1.0), (
            "a restart woke the supervisor for an already-delivered completion"
        )

        # ...and the control: an UNDELIVERED run in the same conversation is
        # claimed, and its notice carries only its own result.
        _p2, run_two = _terminal_subagent_run(
            runs_db, session.id, result="second child result"
        )
        assert restart_wake.seed_from_marks() == 1, (
            "the mount claim missed a genuinely undelivered completion"
        )
        restart_wake.retry_soon()
        assert await _settle(lambda: restart_gateway.payloads)
        notice = str(restart_gateway.payloads[-1][-1]["content"])
        assert "second child result" in notice
        assert "first child result" not in notice, (
            "the restarted process re-announced the already-delivered result"
        )
        assert (runs_db.get_run(run_two) or {}).get("wake_delivered_at")
        assert (runs_db.get_run(run_id) or {}).get("wake_delivered_at") == stamp, (
            "the first run's delivered stamp moved across the restart"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_crash_killed_child_swept_to_error_wakes_nobody_after_a_restart(
    tmp_path,
):
    """No phantom wake: a run reconciled to `error` carries no mark.

    A hard crash leaves a child stuck `running`; the next process's
    `AgentRunsDB.__init__` sweeps it to `error` (`reconcile_orphaned_runs`).
    That row never settled through the fan-out, so no FLEET_UNSEEN mark was
    ever written for it and the mount claim -- the ONLY headless path a
    restarted process has -- must find nothing to announce.

    The non-vacuous control: the same conversation's genuinely settled,
    marked run IS claimed by the same call.

    **What the orphan is NOT exempt from, measured rather than assumed.**
    The first version of this test also asserted the swept orphan could
    never appear in a later legitimate wake's notice. It failed, and the
    code is right: `AgentRunsDB.undelivered_wake_runs` deliberately
    includes `error` runs and its docstring records choosing `>=` on the
    parent/child timestamps *"so a restart-reconcile sweep that stamps an
    orphaned child and its parent in the same pass still reports the
    child"*. An interrupted child is genuinely owed to the supervisor --
    the news is "it died", and saying so once with its honest status is
    the contract. So the property pinned below is the true one: the orphan
    wakes NOBODY on its own (no mark, nothing claimed), and when a real
    completion in the same conversation does trigger a wake it is
    announced with its real `error` status, exactly once, and never again.
    """
    chacha_path = tmp_path / "chacha.sqlite"
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        crashed_parent = runs_db.create_run(
            conversation_id=session.id, agent_kind="primary"
        )
        runs_db.set_status(crashed_parent, "done", "turn final")
        crashed = runs_db.create_run(
            conversation_id=session.id,
            agent_kind="subagent",
            task="killed mid-flight",
            parent_run_id=crashed_parent,
        )
        assert (runs_db.get_run(crashed) or {}).get("status") == "running"

        # Restart: the sweep guard is per-PROCESS, so drop this path from it
        # the way `Tests/DB/test_agent_runs_db.py` does to model a new one.
        AgentRunsDB._swept_paths.discard(runs_db.db_path_str)
        swept_db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
        assert (swept_db.get_run(crashed) or {}).get("status") == "error", (
            "harness precondition: the restart must reconcile the orphan"
        )
        assert not (swept_db.get_run(crashed) or {}).get("wake_delivered_at")
        assert not app.conversation_local_marks_service.has_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        ), "a crash-killed child never settled, so nothing marked it"

        restart_store = ConsoleChatStore()
        restart_session = restart_store.ensure_session(title="Research")
        restart_session.persisted_conversation_id = session.id
        restart_gateway = _RecordingWakeGateway()
        restart_controller = ConsoleChatController(
            store=restart_store,
            provider_gateway=restart_gateway,
            agent_bridge=_FakeWakeBridge(swept_db),
            agent_runtime_enabled=False,
        )
        restart_wake = restart_controller.fleet_wake
        restart_wake.wire(app=app)

        assert restart_wake.seed_from_marks() == 0, (
            "a crash-killed child swept to `error` was claimed as a wake"
        )
        restart_wake.retry_soon()
        assert await _quiet(lambda: restart_gateway.payloads, seconds=1.0), (
            "a crash-killed child woke the supervisor after a restart"
        )

        # Control: a real, marked, undelivered completion IS claimed.
        _p, settled = _terminal_subagent_run(
            swept_db, session.id, result="a genuinely settled result"
        )
        app.conversation_local_marks_service.set_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        )
        assert restart_wake.seed_from_marks() == 1
        restart_wake.retry_soon()
        assert await _settle(lambda: restart_gateway.payloads)
        notice = str(restart_gateway.payloads[-1][-1]["content"])
        assert "a genuinely settled result" in notice
        # The orphan rides along in this legitimate wake -- by design -- and
        # must be announced HONESTLY, never as a success.
        assert f"[{crashed}]" in notice, (
            "an interrupted child is owed to the supervisor and must be "
            f"announced when a wake next runs: {notice[:400]}"
        )
        assert f"[{crashed}] " in notice and "— error" in notice, (
            "the swept orphan was announced without its real `error` status: "
            f"{notice[:400]}"
        )
        # ...and exactly once: the stamp closes it out for good.
        assert (swept_db.get_run(crashed) or {}).get("wake_delivered_at"), (
            "an announced run must be stamped, or the next wake repeats it"
        )
        _p3, third = _terminal_subagent_run(
            swept_db, session.id, result="a third result"
        )
        app.conversation_local_marks_service.set_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        )
        assert restart_wake.seed_from_marks() == 1
        restart_wake.retry_soon()
        assert await _settle(lambda: len(restart_gateway.payloads) > 1)
        second_notice = str(restart_gateway.payloads[-1][-1]["content"])
        assert "a third result" in second_notice
        assert f"[{crashed}]" not in second_notice, (
            "the swept orphan was announced a SECOND time -- the ledger stamp "
            f"must close it out: {second_notice[:400]}"
        )
    finally:
        AgentRunsDB._swept_paths.discard(str(tmp_path / "runs.db"))
        chacha.close()
        assert chacha_path.exists()


@pytest.mark.asyncio
async def test_a_wake_racing_app_exit_leaves_consistent_durable_state(tmp_path):
    """The window this slice opens: quit while a headless wake is in flight.

    Before the gate change a wake could not START between visits, so
    `dispose()` never raced one. Now it can: a survivor settling after the
    user navigated away, with the app quitting a moment later. `dispose()`
    -> `shutdown()` cancels EVERY session's stream task, wake turns
    included (the `leave_console` exemption is deliberately not shutdown's).

    What must hold is not a particular branch but CONSISTENCY between the
    two durable layers, because that is what exactly-once rests on:

    * stamped ledger  => the notice row exists and the pending entry is
      gone (the supervisor was woken; a restart must not re-announce it);
    * unstamped ledger => the pending entry AND the ◈ mark survive (a
      restart owes it and will claim it).

    A "stamped but nothing landed" or "unstamped but dropped from pending"
    outcome is a lost or duplicated completion, and neither is acceptable.

    **Measured branch** (probe run, recorded so nobody has to re-derive
    it): `stamped=True, notices=1, pending=False, marked=True`. The turn
    is ACCEPTED before it streams, so quitting mid-stream truncates the
    reply but does not un-deliver the wake -- the same semantics a mounted
    Console has when the user presses Stop on a wake turn. The ◈ mark
    survives (off-view), so the user is still pointed at it next launch.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        runtime = _runtime_for(rig)
        view = _mounted_view()
        runtime.attach_view(view)
        assert await runtime.leave_console(view) is True

        gateway.stream_gate = asyncio.Event()  # park the turn mid-stream
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        app.conversation_local_marks_service.set_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        )
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: gateway.payloads), (
            "harness precondition: the wake must be IN FLIGHT before quit"
        )

        await runtime.dispose()
        gateway.stream_gate.set()
        await _settle(lambda: wake.delivering_conversation_id() is None)

        stamped = bool((runs_db.get_run(run_id) or {}).get("wake_delivered_at"))
        notices = _notice_rows(store, session.id)
        if stamped:
            assert notices, (
                "the ledger says this completion was delivered, but no notice "
                "row ever landed -- a restart will never re-announce it"
            )
            assert not wake.has_pending(session.id), (
                "a stamped completion is still pending; the next claim would "
                "announce it twice"
            )
        else:
            assert wake.has_pending(session.id), (
                "an unstamped completion was dropped from the pending "
                "registry -- nothing will ever wake the supervisor for it"
            )
            assert _marked(app, session.id), (
                "an unstamped completion lost its ◈ mark, so no restart can "
                "claim it either"
            )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_the_kill_switch_is_read_fresh_at_the_headless_fire_point(
    tmp_path, monkeypatch
):
    """`autowake_enabled` OFF silences the headless fire point and loses nothing.

    Read fresh means: flipping the key ON delivers what OFF recorded, with
    no restart and no new controller -- the same live coordinator that
    just refused.
    """
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        await _leave_console(rig)
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        app.conversation_local_marks_service.set_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        )
        wake = controller.fleet_wake

        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "autowake_enabled=false must silence the HEADLESS fire point too"
        )
        # Nothing durable is lost while OFF.
        assert wake.has_pending(session.id), "OFF still records the completion"
        assert _marked(app, session.id), "OFF keeps the ◈ indicator working"
        assert not (runs_db.get_run(run_id) or {}).get("wake_delivered_at")
        assert wake.seed_from_marks() == 0, "OFF seeds nothing at the mount claim"

        monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
        wake.retry_soon()
        assert await _settle(lambda: gateway.payloads), (
            "the kill switch is not read fresh at the headless fire point: "
            "flipping it ON did not deliver what OFF recorded"
        )
        assert await _settle(
            lambda: bool((runs_db.get_run(run_id) or {}).get("wake_delivered_at"))
        )
    finally:
        chacha.close()


# ---------------------------------------------------------------------------
# 3. Caps: a headless wake is a normal turn under every one.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_global_cap_defers_a_headless_wake_like_any_other_send(tmp_path):
    """`max_parallel_runs` applies with no Console mounted."""
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        await _leave_console(rig)
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        busy_sessions = [store.create_session(title=f"busy {i}") for i in range(3)]
        for busy in busy_sessions:
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STREAMING, "busy"),
                session_id=busy.id,
            )
        assert controller.send_refusal_copy(session.id) is not None, (
            "harness precondition: the cap must actually be saturated"
        )
        wake = controller.fleet_wake

        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "a headless wake must wait for a cap slot like any other send"
        )
        assert wake.has_pending(session.id), "the deferred wake was lost"

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"),
            session_id=busy_sessions[0].id,
        )
        assert await _settle(lambda: gateway.payloads), (
            "freeing a cap slot never retried the deferred headless wake"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_busy_session_defers_a_headless_wake_until_it_goes_terminal(
    tmp_path,
):
    """Per-session busy refusal applies with no Console mounted.

    The refusal is asserted at the COORDINATOR's own gate, not merely by
    the absence of a provider payload. Measured (mutation M7): bypassing
    `send_refusal_copy` in `_attempt` left this test green, because
    `submit_draft` refuses a busy session on its own -- the read site is
    double-guarded, so "nothing streamed" cannot distinguish which guard
    did it. Counting the dispatches is what owns "the wake defers before
    it ever tries to send".
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        await _leave_console(rig)
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "a survivor's turn"),
            session_id=session.id,
        )
        wake = controller.fleet_wake
        dispatches: list[str] = []
        real_submit = controller.submit_draft

        async def _counting_submit(*args, **kwargs):
            dispatches.append(str(kwargs.get("session_id") or ""))
            return await real_submit(*args, **kwargs)

        controller.submit_draft = _counting_submit

        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "a headless wake must never fire into a session whose run is in "
            "flight"
        )
        assert dispatches == [], (
            "the wake tried to SEND into a busy session and relied on "
            f"`submit_draft` to bounce it: {dispatches}"
        )
        assert wake.has_pending(session.id)

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"),
            session_id=session.id,
        )
        assert await _settle(lambda: gateway.payloads), (
            "the terminal transition never retried the deferred headless wake"
        )
    finally:
        chacha.close()


class _SelectionRecordingGateway(_RecordingWakeGateway):
    """Records the `ConsoleProviderSelection` every send resolves with.

    That object carries the per-turn token ceiling (`max_tokens`) and every
    other generation parameter, so comparing a headless wake's against a
    manual send's is a direct test of "the wake rides the unchanged shared
    dispatch".
    """

    def __init__(self, reply: str = "wake reply") -> None:
        super().__init__(reply)
        self.selections: list = []

    async def resolve_for_send(self, selection):
        self.selections.append(selection)
        return await super().resolve_for_send(selection)


@pytest.mark.asyncio
async def test_a_headless_wake_resolves_with_the_same_selection_as_a_manual_send(
    tmp_path,
):
    """The token ceiling (and every other generation input) is unchanged.

    A manual send while Console is mounted and a headless wake into the
    same session must resolve with an IDENTICAL `ConsoleProviderSelection`
    -- `max_tokens` included -- because both go through
    `_provider_selection_for_session`. A wake that quietly built its own
    selection could raise or drop the ceiling.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, bridge, controller = rig
    try:
        gateway = _SelectionRecordingGateway()
        controller.provider_gateway = gateway
        controller.max_tokens = 512
        controller.temperature = 0.11
        controller.model = "test-model"

        runtime = _runtime_for(rig)
        view = _mounted_view()
        runtime.attach_view(view)
        manual = await controller.submit_draft("a manual send", session_id=session.id)
        assert manual.accepted, "harness precondition: the manual send must run"
        assert len(gateway.selections) == 1
        manual_selection = gateway.selections[-1]
        assert manual_selection.max_tokens == 512, (
            "harness precondition: the ceiling must actually be carried, or "
            "the comparison below is vacuous"
        )

        assert await runtime.leave_console(view) is True
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        controller.fleet_wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: len(gateway.selections) > 1), (
            "the headless wake never reached the provider"
        )

        wake_selection = gateway.selections[-1]
        assert wake_selection == manual_selection, (
            "a headless wake resolved with a DIFFERENT provider selection than "
            f"a manual send: {wake_selection!r} vs {manual_selection!r}"
        )
    finally:
        chacha.close()


class _RecordingRunReplyBridge(_FakeWakeBridge):
    """A fake bridge that records every `run_reply` dispatch."""

    def __init__(self, runs_db) -> None:
        super().__init__(runs_db)
        self.calls: list[dict] = []

    def run_reply(self, **kwargs):
        self.calls.append(kwargs)
        return "run-test", RunOutcome(
            status=RUN_DONE, steps=[], final_text="agent answer."
        )


@pytest.mark.asyncio
async def test_a_headless_wake_takes_the_same_agent_dispatch_and_budget(tmp_path):
    """Wall clocks and token ceilings: the wake cannot vary them.

    On the agent path both a manual send and a headless wake go through the
    single `ConsoleAgentBridge.run_reply` dispatch site, and that method has
    no budget parameter at all -- `CONSOLE_RUN_BUDGET` (`max_wall_seconds`,
    `max_total_tokens`) is applied inside it. So "same entry point, same
    inputs, no budget knob" is what makes the two turns identically bounded,
    and each half is asserted rather than assumed.
    """
    chacha, app, runs_db, store, session, _gateway, _bridge, _controller = (
        _controller_rig(tmp_path)
    )
    try:
        gateway = _RecordingWakeGateway()
        bridge = _RecordingRunReplyBridge(runs_db)
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            agent_bridge=bridge,
            agent_runtime_enabled=True,
        )
        controller.app = SimpleNamespace(
            call_from_thread=lambda fn, *a, **kw: fn(*a, **kw)
        )
        controller.fleet_wake.wire(app=app)

        runtime = ConsoleRuntime(app=app)
        runtime.set_chat_store(store)
        runtime.set_chat_controller(controller)
        view = _mounted_view()
        runtime.attach_view(view)

        manual = await controller.submit_draft("a manual send", session_id=session.id)
        assert manual.accepted, "harness precondition: the manual send must run"
        assert len(bridge.calls) == 1, (
            "harness precondition: the manual send must take the AGENT path"
        )

        assert await runtime.leave_console(view) is True
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        controller.fleet_wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: len(bridge.calls) > 1), (
            "the headless wake never reached the agent dispatch"
        )

        manual_call, wake_call = bridge.calls[0], bridge.calls[-1]
        assert wake_call["session_id"] == manual_call["session_id"]
        assert wake_call["model"] == manual_call["model"]
        assert wake_call["conversation_id"] == manual_call["conversation_id"]
        assert callable(wake_call["should_cancel"]), (
            "a headless wake must still be cancellable at the same checkpoints"
        )
        # The wake's only structural difference is its trailing machine
        # notice; the payload before it is the same conversation.
        assert WAKE_NOTICE_HEADER in str(wake_call["agent_messages"][-1]["content"])

        # No caller -- wake or manual -- can vary the wall clock or the token
        # ceiling: `run_reply` has no parameter for either.
        parameters = set(inspect.signature(ConsoleAgentBridge.run_reply).parameters)
        assert not {
            name
            for name in parameters
            if "budget" in name or "wall" in name or "max_tokens" in name
        }, (
            "`run_reply` grew a budget parameter -- a wake turn could now be "
            f"bounded differently from a manual send: {sorted(parameters)}"
        )
        assert set(wake_call) == set(manual_call), (
            "the wake dispatch passes a different set of arguments than a "
            f"manual send: {sorted(set(wake_call) ^ set(manual_call))}"
        )
        assert CONSOLE_RUN_BUDGET.max_wall_seconds > 0
        assert CONSOLE_RUN_BUDGET.max_total_tokens > 0
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_headless_wake_writes_no_user_row_in_the_app_owned_store(tmp_path):
    """The wake invariant, at the seam where it used to be broken.

    Asserted on the surviving app-owned store: the turn adds exactly one
    machine-origin SYSTEM row and one ASSISTANT row, and not a single USER
    row -- a wake notice is never user input, mounted or headless.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        await _leave_console(rig)
        before = list(store.messages_for_session(session.id))
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        controller.fleet_wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: gateway.payloads)
        assert await _settle(
            lambda: len(store.messages_for_session(session.id)) == len(before) + 2
        ), (
            "the headless wake did not add exactly a notice and a reply: "
            f"{[(m.role.value, m.content[:24]) for m in store.messages_for_session(session.id)]}"
        )
        added = store.messages_for_session(session.id)[len(before) :]
        assert [message.role for message in added] == [
            ConsoleMessageRole.SYSTEM,
            ConsoleMessageRole.ASSISTANT,
        ], [message.role for message in added]
        assert added[0].content.startswith(WAKE_NOTICE_HEADER)
        assert not any(
            message.role is ConsoleMessageRole.USER
            for message in store.messages_for_session(session.id)
        ), "the headless wake wrote a USER transcript row"
    finally:
        chacha.close()
