# Tests/Agents/test_fleet_stop_semantics.py
"""Fleet PR 3b Task 5: Stop semantics decoupled from child cancellation.

Spec `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md`
section 8 (Stop semantics move to PR 3b) via the plan's Task 5
(`Docs/superpowers/plans/2026-08-17-fleet-pr3b-steering.md`).

The contract under test, both directions of the existing
`[agents] subagents_outlive_turn` key:

* **Outlive ON (the shipped default)** -- a user Stop cancels the
  SUPERVISOR's turn only. A child still working keeps working: its own
  cancel Event is never set, `_surviving_handles` keeps it at settle, and
  `wait_agents`' cancel branch stops waiting WITHOUT cancelling. The
  user's kill switches for the children themselves are the panel's
  per-row Cancel and the new "Cancel all agents" (bridge
  `cancel_all_subagents`), plus the `subagents_outlive_turn = false`
  config kill switch.
* **Outlive OFF (the kill switch)** -- byte-identical to the phase-2
  rule: Stop kills the whole run tree, THROUGH THE CANCEL-EVENT PATH
  (`_cancel_fleet_handles` -> per-child Event + approval revocation),
  exactly as it always has.

MERGE-BASE PROBES (C1 style). The `_probe_a_*` tests were measured RED at
the untouched merge-base `98a189015` -- at that base a user Stop cancels
a child that should survive, via `child_should_cancel`'s `should_cancel()`
term and `_surviving_handles`' user-cancel branch (both of whose comments
literally cite "spec Sec 10 keeps Stop-semantics changes in PR 3b").
The `_probe_b_*` tests were measured GREEN at that same base and must
stay green UNTOUCHED through the change -- they are the byte-identical
guarantee for the kill-switch path.

Scripting note (Task 2's lesson, re-owned here): `_settle_fleet` sets
every SETTLING child's Event unconditionally at end of turn, so "the
Event is unset" is only meaningful for a SURVIVING child -- which is
exactly what these tests assert; a settling child's Event being set is
asserted through the terminal end-state as well, never the raw Event
alone.
"""

import threading
import time

import pytest

from Tests.Agents.conftest import pin_agent_settings, pin_turn_scoped_children
from Tests.Agents.test_agent_service import fence
from Tests.Agents.test_fleet_runtime import (
    FLEET_CFG,
    _after,
    _child_row,
    _gated_child,
    _tool_results,
    _wait_until,
    db,  # noqa: F401  -- pytest fixture, resolved via this import
    make_fleet_service,
)
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import (
    RUN_CANCELLED,
    RUN_DONE,
    RUN_RUNNING,
    SEND_TO_AGENT_TOOL_NAME,
    SPAWN_TOOL_NAME,
    STEERING_SOURCE_USER,
    WAIT_AGENTS_TOOL_NAME,
    format_steering_message,
)

_JOIN_TIMEOUT = 5.0


def _pin_outlive_on(monkeypatch):
    """Pin the shipped default EXPLICITLY -- these tests are about it.

    The default already is True; pinning keeps each probe's subject
    independent of any future default flip, the same way
    `pin_turn_scoped_children` pins the opposite pole.
    """
    pin_agent_settings(
        monkeypatch, **{agent_service.SUBAGENTS_OUTLIVE_TURN_KEY: True}
    )


def _two_turn_child(entered, release, timeout=10.0):
    """A gated child that needs ONE MORE loop boundary after release.

    First (gated) provider call returns a calculator fence; the result is
    appended and the child crosses its loop-top cancellation check before
    its second model call, which answers. That extra boundary is the
    whole point: a child that answers straight off its gated call would
    never poll cancellation again, and the mutation this suite must kill
    -- re-adding the parent-poll term to `child_should_cancel` -- would
    go unnoticed.
    """

    def gated_fence():
        entered.set()
        if not release.wait(timeout):
            raise AssertionError("child was never released by the test")
        return fence("calculator", {"expression": "1+1"})

    return [gated_fence, "late answer"]


# -- probe (a): outlive ON -- Stop spares the children ---------------------


def test_probe_a_stop_mid_turn_leaves_the_child_running(db, monkeypatch):
    """Outlive ON: a user Stop kills the TURN; the child survives it.

    RED at the untouched merge-base (the child came back `cancelled`):
    `_surviving_handles` settled everything on a user cancel, and
    `child_should_cancel` polled the parent's own probe. GREEN after the
    change: the turn returns `cancelled` promptly, the child's own Event
    is untouched, and the child later finishes DONE with its real result
    -- which is also what kills the re-added parent-poll mutant (the
    parent's cancel probe stays True forever, so a re-coupled child dies
    at its first post-release loop boundary instead of answering).
    """
    _pin_outlive_on(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()

    def spawn_then_cancel():
        # Stop once the child is provably live, so the turn ends with a
        # running child AND a user cancellation -- the combination whose
        # fate this task changes.
        if not entered.wait(_JOIN_TIMEOUT):
            raise AssertionError("the child never reached its model call")
        cancelled.set()
        return "parent stopped"

    service, _chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "slow task"}), spawn_then_cancel],
        {"slow task": _two_turn_child(entered, release)},
        allow_unconsumed=True,  # red-at-base strands the child's turns
    )
    try:
        started_at = time.monotonic()
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
            should_cancel=cancelled.is_set,
        )
        elapsed = time.monotonic() - started_at
        assert entered.is_set(), "precondition: the child reached its model call"
        # The user's Stop still stops the SUPERVISOR...
        assert outcome.status == RUN_CANCELLED
        # ... promptly: no settle-wait, no join, no abandonment grace.
        assert elapsed < 3.0, f"the stopped turn was held open for {elapsed:.2f}s"
        # ... and ONLY the supervisor: the child was not cancelled, not
        # abandoned, not forced terminal in the DB.
        handle = coordinator.snapshot()[0]
        assert handle.status == RUN_RUNNING, handle
        assert not service._fleet_cancels[handle.handle_id].is_set(), (
            "a user Stop set the surviving child's own cancel Event"
        )
        assert _child_row(db)["status"] == RUN_RUNNING
    finally:
        release.set()
    _wait_until(coordinator.all_finished, "the released child never finished")
    survivor = coordinator.snapshot()[0]
    assert survivor.status == RUN_DONE, survivor
    assert survivor.result == "late answer"
    assert _child_row(db)["status"] == RUN_DONE


def test_probe_a_wait_agents_cancel_stops_waiting_without_cancelling(
    db, monkeypatch
):
    """Outlive ON: Stop during `wait_agents` releases the WAIT, not the kids.

    RED at the untouched merge-base (`wait_agents`' cancel branch called
    `_cancel_fleet_handles` on everything pending). GREEN after: the wait
    breaks immediately, the note says the sub-agents continue in the
    background, no Event is set, and the child finishes DONE afterwards.

    The cancel is triggered from INSIDE the wait loop deterministically:
    `should_cancel` flips on its SECOND call after the wait fence was
    returned -- call one is the loop's own pre-dispatch check
    (`agent_runtime.py`'s per-call cancellation gate), call two is
    `wait_agents`' first poll. Flipping any earlier would kill the parent
    before `wait_agents` ever ran (which is exactly what the pre-existing
    `test_wait_agents_cancellation_*` script does, making it a settle
    test in disguise); if a future edit adds another pre-dispatch check
    this fails LOUDLY (no wait_agents result recorded), never silently.
    """
    _pin_outlive_on(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    wait_dispatched = threading.Event()
    polls_after_dispatch = {"count": 0}

    def wait_after_child_entered():
        if not entered.wait(_JOIN_TIMEOUT):
            raise AssertionError("the child never reached its model call")
        wait_dispatched.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    def should_cancel():
        if not wait_dispatched.is_set():
            return False
        polls_after_dispatch["count"] += 1
        return polls_after_dispatch["count"] >= 2

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            wait_after_child_entered,
        ],
        {"slow task": _two_turn_child(entered, release)},
        allow_unconsumed=True,  # red-at-base strands the child's turns
    )
    try:
        started_at = time.monotonic()
        run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
            should_cancel=should_cancel,
        )
        elapsed = time.monotonic() - started_at
        assert outcome.status == RUN_CANCELLED
        # The wait released without the drain grace or the settle wait.
        assert elapsed < 3.0, f"the stopped turn was held open for {elapsed:.2f}s"
        waits = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
        assert waits, "wait_agents never ran -- the cancel landed too early"
        assert "sub-agents continue in the background" in waits[0], waits[0]
        assert "sub-agents were stopped" not in waits[0], waits[0]
        # Not cancelled: no Event, still running, row still running.
        handle = coordinator.snapshot()[0]
        assert handle.status == RUN_RUNNING, handle
        assert not service._fleet_cancels[handle.handle_id].is_set()
        assert _child_row(db)["status"] == RUN_RUNNING
    finally:
        release.set()
    _wait_until(coordinator.all_finished, "the released child never finished")
    assert coordinator.snapshot()[0].status == RUN_DONE
    assert coordinator.snapshot()[0].result == "late answer"


def test_a_stopped_parents_survivor_still_drains_steering(db, monkeypatch):
    """The steering interaction pin: Stop does not disconnect the mailbox.

    A survivor of a stopped turn is still steerable (the mailbox lives on
    the conversation-lifetime coordinator, not the dead turn) AND still
    drains: an entry posted AFTER the Stop is delivered at the child's
    next model boundary, exactly as for any live child. RED at the
    merge-base by construction (the child there is already cancelled, so
    `post_steering` refuses it).
    """
    _pin_outlive_on(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()

    def spawn_then_cancel():
        if not entered.wait(_JOIN_TIMEOUT):
            raise AssertionError("the child never reached its model call")
        cancelled.set()
        return "parent stopped"

    service, chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "slow task"}), spawn_then_cancel],
        {"slow task": _two_turn_child(entered, release)},
        allow_unconsumed=True,
    )
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
            should_cancel=cancelled.is_set,
        )
        assert outcome.status == RUN_CANCELLED
        handle = coordinator.snapshot()[0]
        assert handle.status == RUN_RUNNING
        # Steer the survivor AFTER its parent turn was stopped -- the
        # panel path's post (USER source), which must still say yes...
        assert coordinator.post_steering(
            handle.handle_id, STEERING_SOURCE_USER, "check the appendix"
        ) is True
    finally:
        release.set()
    _wait_until(coordinator.all_finished, "the released child never finished")
    assert coordinator.snapshot()[0].status == RUN_DONE
    # ... and must still DELIVER: the child's second (post-release) model
    # call carries the labeled message, after its tool result.
    second_payload = chat.child_calls["slow task"][1]["messages_payload"]
    steer_text = format_steering_message(STEERING_SOURCE_USER, "check the appendix")
    assert any(
        message.get("role") == "user"
        and message.get("content") == steer_text
        for message in second_payload
    ), second_payload


# -- probe (b): outlive OFF -- byte-identical kill-switch path -------------


def test_probe_b_stop_kills_everything_through_the_cancel_event_path(
    db, monkeypatch
):
    """Outlive OFF: Stop still takes the whole run tree, via the Events.

    GREEN at the untouched merge-base and required to stay green through
    the change UNTOUCHED -- this is the byte-identical guarantee for the
    kill switch. The child's own Event being SET is the path assertion:
    the kill goes through `_cancel_fleet_handles` (Event + approval
    revocation), not through some new mechanism.
    """
    pin_turn_scoped_children(monkeypatch)
    monkeypatch.setattr(agent_service, "FLEET_JOIN_TIMEOUT_SECONDS", 0.2)
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()

    def spawn_then_cancel():
        if not entered.wait(_JOIN_TIMEOUT):
            raise AssertionError("the child never reached its model call")
        cancelled.set()
        return "parent stopped"

    service, _chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "slow task"}), spawn_then_cancel],
        {"slow task": [_gated_child(entered, release)]},
        allow_unconsumed=True,
    )
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
            should_cancel=cancelled.is_set,
        )
        assert entered.is_set()
        assert outcome.status == RUN_CANCELLED
        # The whole tree died, and died through the Event path.
        handle = coordinator.snapshot()[0]
        assert handle.status == RUN_CANCELLED, handle
        assert service._fleet_cancels[handle.handle_id].is_set(), (
            "the kill-switch stop must go through the cancel-Event path"
        )
        assert coordinator.all_finished()
        assert _child_row(db)["status"] == RUN_CANCELLED
    finally:
        release.set()


def test_probe_b_wait_agents_cancel_still_stops_children(db, monkeypatch):
    """Outlive OFF: `wait_agents`' cancel branch cancels, byte-identically.

    GREEN at the untouched merge-base: the same in-wait cancel trigger as
    probe (a2), under the kill switch -- children are cancelled through
    their Events and the note says so, in the exact pre-change copy.
    """
    pin_turn_scoped_children(monkeypatch)
    monkeypatch.setattr(agent_service, "FLEET_JOIN_TIMEOUT_SECONDS", 0.2)
    entered = threading.Event()
    release = threading.Event()
    wait_dispatched = threading.Event()
    polls_after_dispatch = {"count": 0}

    def wait_after_child_entered():
        if not entered.wait(_JOIN_TIMEOUT):
            raise AssertionError("the child never reached its model call")
        wait_dispatched.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    def should_cancel():
        if not wait_dispatched.is_set():
            return False
        polls_after_dispatch["count"] += 1
        return polls_after_dispatch["count"] >= 2

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            wait_after_child_entered,
        ],
        {"slow task": [_gated_child(entered, release)]},
        allow_unconsumed=True,  # the cancelled child strands its turn
    )
    try:
        run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
            should_cancel=should_cancel,
        )
        assert outcome.status == RUN_CANCELLED
        waits = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
        assert waits, "wait_agents never ran -- the cancel landed too early"
        assert "(The run was cancelled; sub-agents were stopped.)" in waits[0]
        handle = coordinator.snapshot()[0]
        assert service._fleet_cancels[handle.handle_id].is_set(), (
            "the kill-switch wait-cancel must go through the cancel-Event path"
        )
        assert handle.status == RUN_CANCELLED, handle
        assert coordinator.all_finished()
        assert _child_row(db)["status"] == RUN_CANCELLED
    finally:
        release.set()


def test_probe_b_a_mid_loop_child_notices_stop_before_settle_under_the_kill_switch(
    db, monkeypatch
):
    """Outlive OFF: the child-side parent poll itself, isolated.

    Every other kill-switch test kills its child through the settle or
    `wait_agents` (the Event path), which makes the OFF closure's
    `should_cancel()` term outcome-redundant there -- a mutant dropping
    it would survive them. Here the parent is held INSIDE its own model
    call after Stop, so no settle has run and no Event is set when the
    released child crosses its next loop boundary: the ONLY thing that
    can kill it there is the parent poll. With the term, the child dies
    `cancelled` after exactly one model call; without it, it would
    finish `done` off a second call. GREEN at the untouched merge-base;
    byte-identical through the change; the owner for the
    OFF-branch-altered mutation.
    """
    pin_turn_scoped_children(monkeypatch)
    entered = threading.Event()
    go = threading.Event()
    cancelled = threading.Event()

    def gated_fence():
        entered.set()
        if not go.wait(10.0):
            raise AssertionError("the child was never released by the test")
        return fence("calculator", {"expression": "1+1"})

    def stop_then_watch():
        if not entered.wait(_JOIN_TIMEOUT):
            raise AssertionError("the child never reached its model call")
        cancelled.set()
        # Release the child INTO its loop while the parent is still held
        # inside THIS model call -- pre-settle, pre-Event.
        go.set()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            handle = coordinator.snapshot()[0]
            if handle.status != RUN_RUNNING:
                return "parent stopped"
            time.sleep(0.01)
        raise AssertionError(
            "the child never went terminal while the parent was held -- "
            "the OFF parent poll did not fire"
        )

    service, chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "slow task"}), stop_then_watch],
        {"slow task": [gated_fence, "late answer"]},
        allow_unconsumed=True,  # the poll-killed child strands its 2nd turn
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
        should_cancel=cancelled.is_set,
    )
    assert outcome.status == RUN_CANCELLED
    assert coordinator.snapshot()[0].status == RUN_CANCELLED
    # Killed at the boundary BEFORE any second provider call: the poll,
    # not a stray completion.
    assert len(chat.child_calls["slow task"]) == 1
    assert _child_row(db)["status"] == RUN_CANCELLED


# -- the teardown / app-exit audit (plan Task 5's audit bullet) ------------
#
# Audit result, measured at the merge-base and recorded honestly: NO
# conversation-deletion or ephemeral-close path sets per-child cancel
# Events today. The only Event-setting paths are `_cancel_fleet_handles`'
# callers -- the end-of-turn settle (non-survivors), `wait_agents`' cancel
# and budget branches, and the per-row `cancel_subagent` (which "Cancel
# all agents" reuses per handle). Session close and both controller
# teardowns (`shutdown`, `leave_console`) reach an in-flight fleet only
# through `_signal_stop` -> the PARENT's cancel probe -- so with the
# outlive default ON their children now survive those teardowns exactly
# as `leave_console`'s own docstring already promised for earlier turns'
# survivors. What must therefore stay true, pinned below: the EVENT path
# remains fully effective on a decoupled child (that is the panel's and
# Cancel-all's whole mechanism), and app exit still takes everything
# because fleet children are daemon threads that die with the process.


def test_a_survivor_of_a_stopped_turn_dies_on_its_own_cancel_event(
    db, monkeypatch
):
    """The Event path survives the decoupling -- Cancel still works.

    The parent-poll term is gone from a decoupled child's poll, so its
    OWN Event is now the only cooperative stop there is. This is the
    audit's load-bearing half: per-row Cancel, "Cancel all agents", the
    settle under the kill switch, and any future teardown all speak this
    path. A mutant that drops the Event term from the outlive-ON closure
    (instead of just the parent term) dies here.
    """
    _pin_outlive_on(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()

    def spawn_then_cancel():
        if not entered.wait(_JOIN_TIMEOUT):
            raise AssertionError("the child never reached its model call")
        cancelled.set()
        return "parent stopped"

    service, _chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "slow task"}), spawn_then_cancel],
        {"slow task": _two_turn_child(entered, release)},
        allow_unconsumed=True,  # the cancelled child strands its 2nd turn
    )
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
            should_cancel=cancelled.is_set,
        )
        assert outcome.status == RUN_CANCELLED
        handle = coordinator.snapshot()[0]
        assert handle.status == RUN_RUNNING
        # The user cancels the SURVIVOR itself -- the per-row path, the
        # same seam `cancel_all_subagents` walks per handle.
        assert service.cancel_subagent(handle.handle_id) is True
        assert service._fleet_cancels[handle.handle_id].is_set()
    finally:
        release.set()
    # Released from its provider call, the child notices ITS OWN Event at
    # the next loop boundary and dies for real -- Event path intact.
    _wait_until(coordinator.all_finished, "the cancelled child never stopped")
    assert coordinator.snapshot()[0].status == RUN_CANCELLED
    assert _child_row(db)["status"] == RUN_CANCELLED


def test_a_cancel_alled_child_draws_the_not_retained_refusal_not_unknown(
    db, monkeypatch
):
    """Task 4 interaction: Cancel-all never manufactures a resumable child.

    "Cancel all agents" cancels each child through the per-handle
    `cancel_subagent` seam (pinned by the delegation spy in
    `test_console_agent_bridge_cancel_all`); a child cancelled that way
    finishes `cancelled`, which is NEVER a retained status -- so a later
    `send_to_agent` must draw the honest finished-but-not-retained
    refusal, never the unknown-id copy (the child is real; pretending
    the id is unknown would send the supervisor hunting for a typo).
    """
    _pin_outlive_on(monkeypatch)
    entered = threading.Event()
    release = threading.Event()

    def cancel_all_then_send():
        # The user presses Cancel-all while the child works: the panel's
        # walk lands on this service's per-handle seam.
        if not entered.wait(_JOIN_TIMEOUT):
            raise AssertionError("the child never reached its model call")
        handle_id = coordinator.snapshot()[0].handle_id
        assert service.cancel_subagent(handle_id) is True
        release.set()
        # Wait for the child to fully unwind so `finish(CANCELLED)` has
        # run and retention has (correctly) refused it.
        service._fleet_threads[handle_id].join(_JOIN_TIMEOUT)
        return fence(
            SEND_TO_AGENT_TOOL_NAME, {"id": handle_id, "message": "keep going"}
        )

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            cancel_all_then_send,
            "parent done",
        ],
        {"slow task": [_gated_child(entered, release)]},
        allow_unconsumed=True,  # the cancelled child strands its reply
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    sends = _tool_results(db.get_run(run_id), SEND_TO_AGENT_TOOL_NAME)
    assert sends, "send_to_agent never ran"
    assert "has finished (cancelled)" in sends[0], sends[0]
    assert "no retained transcript" in sends[0], sends[0]
    assert "no sub-agent matches id" not in sends[0], (
        "a real cancel-all'd child must never draw the unknown-id copy"
    )


def test_app_exit_takes_everything_fleet_children_are_daemon_threads(
    db, monkeypatch
):
    """The app-exit half of the audit: children die with the process.

    App exit (`ConsoleRuntime.dispose` -> `controller.shutdown()`) cancels
    in-flight TURNS through the parent probe; a decoupled child never
    hears that. What guarantees "exit takes everything" is that every
    fleet child runs on a ``daemon=True`` thread -- the interpreter does
    not wait for it -- which was already the only guarantee earlier
    turns' survivors had at exit BEFORE this task. Pinned here so a
    future thread-pool refactor cannot silently turn a stopped turn's
    survivor into a process-outliving non-daemon thread.
    """
    _pin_outlive_on(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "parent answered early"),
        ],
        {"slow task": [_gated_child(entered, release)]},
    )
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome.status == RUN_DONE
        threads = list(service._fleet_threads.values())
        assert threads, "precondition: the child's thread is registered"
        assert all(thread.daemon for thread in threads), (
            "a fleet child on a non-daemon thread would outlive app exit"
        )
        assert any(thread.is_alive() for thread in threads), (
            "precondition: the survivor is genuinely still running"
        )
    finally:
        release.set()
    _wait_until(coordinator.all_finished, "the released child never finished")
