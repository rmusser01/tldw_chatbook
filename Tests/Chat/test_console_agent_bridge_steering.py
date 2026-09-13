"""Fleet PR 3b Task 3: `ConsoleAgentBridge.steer_subagent` — the panel's
(USER's) path into Task 1's per-child steering mailbox.

Spec `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md`
section 6 (two paths, one mechanism; source labels), section 3 invariant 4
(steering never cancels). Plan Task 3's boundary rules, all pinned here:

* Validation happens at THIS producer boundary (non-empty after strip,
  ``MAX_STEERING_CHARS`` cap) — ``post_steering`` deliberately does not
  validate (Task 1's pinned decision, re-pinned by Task 2's report).
* Resolution reuses Task 2's SHAPE: live handles only, handle id FIRST,
  then a live handle's run id, over the whole coordinator — a
  pathological collision lands on the handle-id owner.
* The entry is labeled ``STEERING_SOURCE_USER`` — the exact literal
  ``"user"`` is asserted, because a mislabeling to the supervisor source
  would silently misattribute the user's words at the drain point.
* No service hop: the method touches only ``_fleet_coordinators``, so it
  is safe from the UI thread (the coordinator's own brief lock is the
  only lock taken).
"""

from __future__ import annotations

import time

from tldw_chatbook.Agents.agent_models import (
    MAX_STEERING_CHARS,
    STEERING_SOURCE_USER,
)
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

CONV = "conv-1"


def _bridge_with_fleet(tmp_path):
    """A REAL bridge with a REAL conversation coordinator injected.

    ``steer_subagent`` must depend on nothing but ``_fleet_coordinators``
    (the no-service-hop rule), so injecting the coordinator directly is
    not a shortcut around the production path — it IS the production
    path's whole state surface.
    """
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    coordinator = FleetCoordinator(max_live=4, clock=time.monotonic)
    bridge._fleet_coordinators[CONV] = coordinator
    return bridge, coordinator


def _live_child(coordinator, run_id="run-1", task="find pricing"):
    handle = coordinator.reserve(task, "researcher")
    coordinator.attach_run(handle.handle_id, run_id)
    return handle.handle_id


# -- resolution + the exact USER label ---------------------------------


def test_steer_by_handle_id_queues_an_exact_user_labeled_entry(tmp_path):
    """The mailbox entry is ``("user", text)`` — the literal label, not
    just "some entry landed". A swap to the supervisor source must fail
    HERE, not at a rendering three seams later."""
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    handle_id = _live_child(coordinator)

    assert bridge.steer_subagent(CONV, handle_id, "focus on pricing") is True
    assert coordinator.drain_steering(handle_id) == [("user", "focus on pricing")]
    # The constant this file (and the bridge) name IS that literal.
    assert STEERING_SOURCE_USER == "user"


def test_steer_by_run_id_reaches_the_same_mailbox(tmp_path):
    """A drill-in target is a RUN id (`_console_agent_drilldown_run_id`),
    so the run-id vocabulary must reach the same mailbox the handle id
    keys."""
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    handle_id = _live_child(coordinator, run_id="run-77")

    assert bridge.steer_subagent(CONV, "run-77", "check the appendix") is True
    assert coordinator.drain_steering(handle_id) == [("user", "check the appendix")]


def test_a_live_handle_id_beats_a_colliding_run_id(tmp_path):
    """Task 2's resolution-order pin, re-owned at this boundary: child
    A's run id is forged to equal child B's handle id; steering by that
    string must land on B — the handle-id owner."""
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    handle_a = _live_child(coordinator, run_id="placeholder")
    handle_b = _live_child(coordinator, run_id="run-b")
    # Forge the collision: A's run id becomes B's handle id.
    coordinator.attach_run(handle_a, handle_b)

    assert bridge.steer_subagent(CONV, handle_b, "for B") is True
    assert coordinator.drain_steering(handle_a) == []
    assert coordinator.drain_steering(handle_b) == [("user", "for B")]


def test_text_is_posted_stripped(tmp_path):
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    handle_id = _live_child(coordinator)

    assert bridge.steer_subagent(CONV, handle_id, "  hey  \n") is True
    assert coordinator.drain_steering(handle_id) == [("user", "hey")]


# -- validation at THIS boundary ----------------------------------------


def test_empty_and_whitespace_text_refused_without_posting(tmp_path):
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    handle_id = _live_child(coordinator)

    assert bridge.steer_subagent(CONV, handle_id, "") is False
    assert bridge.steer_subagent(CONV, handle_id, "   \n\t") is False
    assert coordinator.drain_steering(handle_id) == []


def test_oversize_refused_and_at_cap_accepted(tmp_path):
    """Boundary-exact: one char over the cap is refused, exactly at the
    cap is queued (kills a `>=` slip)."""
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    handle_id = _live_child(coordinator)

    assert (
        bridge.steer_subagent(CONV, handle_id, "x" * (MAX_STEERING_CHARS + 1)) is False
    )
    assert coordinator.drain_steering(handle_id) == []
    assert bridge.steer_subagent(CONV, handle_id, "x" * MAX_STEERING_CHARS) is True
    entries = coordinator.drain_steering(handle_id)
    assert len(entries) == 1
    assert entries[0] == ("user", "x" * MAX_STEERING_CHARS)


def test_an_empty_row_id_is_refused_without_posting(tmp_path):
    """Task 2's report bound this task: an empty target must never draw
    an unknown-id path naming `''` — here that means a plain False with
    nothing queued anywhere."""
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    handle_id = _live_child(coordinator)

    assert bridge.steer_subagent(CONV, "", "text") is False
    assert coordinator.drain_steering(handle_id) == []


# -- refusal shapes -----------------------------------------------------


def test_terminal_target_refused_by_both_vocabularies(tmp_path):
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    handle_id = _live_child(coordinator, run_id="run-done")
    coordinator.finish(handle_id, "done", result="42")

    assert bridge.steer_subagent(CONV, handle_id, "too late") is False
    assert bridge.steer_subagent(CONV, "run-done", "too late") is False
    assert coordinator.drain_steering(handle_id) == []


def test_unknown_conversation_and_unknown_id_refused(tmp_path):
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    handle_id = _live_child(coordinator)

    assert bridge.steer_subagent("conv-never-seen", handle_id, "hello") is False
    assert bridge.steer_subagent(CONV, "nope", "hello") is False
    assert coordinator.drain_steering(handle_id) == []


# -- steering never cancels (spec section 3 invariant 4) -----------------


def test_steering_never_cancels_or_mutates_the_handle(tmp_path):
    """The post touches the mailbox and nothing else: status stays
    running, no terminal fields appear, and the queued count is computed
    onto the next snapshot copy."""
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    handle_id = _live_child(coordinator)

    assert bridge.steer_subagent(CONV, handle_id, "keep going") is True

    [handle] = [h for h in coordinator.snapshot() if h.handle_id == handle_id]
    assert handle.status == "running"
    assert handle.finished_at is None
    assert handle.result == "" and handle.error == ""
    assert handle.queued_steering == 1
