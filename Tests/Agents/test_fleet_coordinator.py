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
