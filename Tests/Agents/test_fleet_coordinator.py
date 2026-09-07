"""FleetCoordinator: pure handle/state machine for concurrent children."""

import threading

import pytest

from tldw_chatbook.Agents.agent_models import RUN_DONE, RUN_ERROR
from tldw_chatbook.Agents.fleet_coordinator import (
    FLEET_FINISHED,
    FLEET_STARTED,
    FleetCoordinator,
)


def _coord(max_live=3):
    ticks = iter(range(1000))
    return FleetCoordinator(max_live=max_live, clock=lambda: float(next(ticks)))


def test_reserve_returns_handle_and_emits_started():
    c = _coord()
    h = c.reserve(task="do x", agent="researcher")
    assert h is not None and h.task == "do x" and h.agent == "researcher"
    assert h.status == "running" and h.finished_at is None
    events = c.drain_events()
    assert [e.kind for e in events] == [FLEET_STARTED]
    assert events[0].handle_id == h.handle_id
    assert c.drain_events() == []  # drain is destructive


def test_reserve_refuses_past_live_cap():
    c = _coord(max_live=2)
    assert c.reserve(task="a", agent=None) is not None
    assert c.reserve(task="b", agent=None) is not None
    assert c.reserve(task="c", agent=None) is None
    assert c.live_count() == 2


def test_finish_frees_a_slot_and_emits_finished():
    c = _coord(max_live=1)
    h = c.reserve(task="a", agent=None)
    assert c.reserve(task="b", agent=None) is None
    c.finish(h.handle_id, RUN_DONE, result="answer")
    assert c.live_count() == 0
    assert c.reserve(task="b", agent=None) is not None
    kinds = [e.kind for e in c.drain_events()]
    assert kinds == [FLEET_STARTED, FLEET_FINISHED, FLEET_STARTED]
    done = c.get(h.handle_id)
    assert done.status == RUN_DONE and done.result == "answer"
    assert done.finished_at is not None


def test_finish_is_idempotent_first_writer_wins():
    # A child abandoned after a join timeout can finish LATE; the
    # coordinator must not let it overwrite a terminal status.
    c = _coord()
    h = c.reserve(task="a", agent=None)
    c.finish(h.handle_id, "cancelled")
    c.finish(h.handle_id, RUN_DONE, result="late answer")
    assert c.get(h.handle_id).status == "cancelled"
    assert c.get(h.handle_id).result == ""


def test_attach_run_records_run_id():
    c = _coord()
    h = c.reserve(task="a", agent=None)
    c.attach_run(h.handle_id, "run-123")
    assert c.get(h.handle_id).run_id == "run-123"
    c.finish(h.handle_id, RUN_ERROR, error="boom")
    assert c.drain_events()[-1].run_id == "run-123"


def test_snapshot_returns_copies_not_internals():
    c = _coord()
    h = c.reserve(task="a", agent=None)
    snap = c.snapshot()
    snap[0].status = "tampered"
    assert c.get(h.handle_id).status == "running"


def test_all_finished_reflects_live_state():
    c = _coord()
    assert c.all_finished() is True
    h = c.reserve(task="a", agent=None)
    assert c.all_finished() is False
    c.finish(h.handle_id, RUN_DONE)
    assert c.all_finished() is True


def test_concurrent_reserve_never_exceeds_cap():
    c = _coord(max_live=5)
    got = []
    lock = threading.Lock()

    def worker():
        h = c.reserve(task="t", agent=None)
        with lock:
            got.append(h)

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for h in got if h is not None) == 5
    assert c.live_count() == 5


def test_finish_records_total_tokens():
    """PR2b Task 5 (cost rollup): a handle's measured token spend is 0
    until `finish()` records it -- a running child's spend is not final."""
    c = _coord()
    h = c.reserve(task="a", agent=None)
    assert h.total_tokens == 0
    c.finish(h.handle_id, RUN_DONE, result="answer", total_tokens=250)
    assert c.get(h.handle_id).total_tokens == 250


def test_finish_without_total_tokens_defaults_to_zero():
    """Every pre-Task-5 caller of `finish()` omits `total_tokens` --
    byte-identical behavior, not a required migration."""
    c = _coord()
    h = c.reserve(task="a", agent=None)
    c.finish(h.handle_id, RUN_DONE, result="answer")
    assert c.get(h.handle_id).total_tokens == 0


def test_late_finish_does_not_overwrite_total_tokens():
    """First-writer-wins covers `total_tokens` too -- a late/abandoned
    finish must not clobber the real recorded spend with a fabricated or
    stale later figure, mirroring `test_finish_is_idempotent_first_writer_
    wins`'s own `result` assertion."""
    c = _coord()
    h = c.reserve(task="a", agent=None)
    c.finish(h.handle_id, "cancelled", total_tokens=10)
    c.finish(h.handle_id, RUN_DONE, result="late answer", total_tokens=999)
    assert c.get(h.handle_id).total_tokens == 10


def test_finish_guard_survives_a_status_outside_the_terminal_vocabulary():
    # The guard must not rely on status vocabulary membership. A handle that
    # finishes with "timeout" (not in TERMINAL_RUN_STATUSES) should reject a
    # later finish with RUN_DONE. This tests the idempotency guard's use of
    # liveness, not status membership.
    c = _coord()
    h = c.reserve(task="a", agent=None)
    c.finish(h.handle_id, "timeout")
    c.finish(h.handle_id, RUN_DONE, result="late answer")
    assert c.get(h.handle_id).status == "timeout"
    assert c.get(h.handle_id).result == ""
    assert c.live_count() == 0


# -- PR3a-1 Task 6a: what a CROSS-TURN owner needs ------------------------


def test_prune_terminal_forgets_finished_handles_and_keeps_live_ones():
    """A per-conversation coordinator lives for the whole process, so
    "never forget a handle" (fine for a one-turn object) would grow
    `_handles` without bound and hand the fleet panel every child the
    conversation ever ran. Pruning drops only the terminal ones."""
    c = _coord()
    done = c.reserve(task="finished", agent=None)
    live = c.reserve(task="still going", agent=None)
    c.finish(done.handle_id, RUN_DONE, result="answer")

    assert c.prune_terminal() == 1
    assert [h.handle_id for h in c.snapshot()] == [live.handle_id]
    assert c.get(done.handle_id) is None
    assert c.live_count() == 1
    # Idempotent: a second prune with nothing terminal left is a no-op.
    assert c.prune_terminal() == 0


def test_prune_terminal_frees_no_slots_because_terminal_handles_held_none():
    """Pruning must not be mistaken for a cap release: a terminal handle
    was already out of `_live_ids`, so the cap is unchanged either way."""
    c = _coord(max_live=2)
    first = c.reserve(task="a", agent=None)
    c.reserve(task="b", agent=None)
    assert c.reserve(task="c", agent=None) is None  # at cap
    c.finish(first.handle_id, RUN_DONE)
    c.prune_terminal()
    assert c.live_count() == 1
    assert c.reserve(task="c", agent=None) is not None


def test_set_max_live_resizes_in_place_without_dropping_live_handles():
    """`[agents] max_live_subagents` can change mid-conversation. Replacing
    the coordinator would drop every live handle from the only surface that
    can see or stop it -- a silent loss of exactly the survivors PR3a-1
    exists to keep -- so the owner re-sizes instead."""
    c = _coord(max_live=1)
    live = c.reserve(task="a", agent=None)
    assert c.max_live == 1
    assert c.reserve(task="b", agent=None) is None

    c.set_max_live(3)

    assert c.max_live == 3
    assert [h.handle_id for h in c.snapshot()] == [live.handle_id]
    assert c.reserve(task="b", agent=None) is not None


def test_lowering_max_live_below_the_live_count_refuses_rather_than_kills():
    """Back-pressure, not a cull: shrinking the cap while children are
    running must never terminate one -- it just refuses the next
    reservation until enough of them finish."""
    c = _coord(max_live=3)
    a = c.reserve(task="a", agent=None)
    b = c.reserve(task="b", agent=None)

    c.set_max_live(1)

    assert c.live_count() == 2
    assert {h.status for h in c.snapshot()} == {"running"}
    assert c.reserve(task="c", agent=None) is None
    c.finish(a.handle_id, RUN_DONE)
    assert c.reserve(task="c", agent=None) is None  # still 1 live, cap 1
    c.finish(b.handle_id, RUN_DONE)
    assert c.reserve(task="c", agent=None) is not None
