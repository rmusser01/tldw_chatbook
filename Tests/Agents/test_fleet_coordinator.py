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


def test_successful_reserve_notifies_before_returning_the_handle():
    observations = []

    def on_reserve() -> None:
        observations.append("admitted")

    coordinator = FleetCoordinator(
        max_live=1,
        clock=lambda: 1.0,
        on_reserve=on_reserve,
    )

    assert coordinator.reserve(task="admitted", agent=None) is not None
    assert coordinator.reserve(task="refused", agent=None) is None

    assert observations == ["admitted"]


def test_terminal_fence_refuses_every_late_reservation():
    c = _coord(max_live=2)

    c.fence()

    assert c.reserve(task="late child", agent=None) is None
    assert c.live_count() == 0


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


def test_progress_binding_requires_attached_live_handle_and_captures_once():
    from tldw_chatbook.Agents.fleet_messages import MessageError, MessageStore

    inbox = MessageStore().open_inbox("conversation")
    coord = FleetCoordinator(2, lambda: 0.0, message_inbox=inbox)
    handle = coord.reserve("work", "researcher")
    bind = lambda: coord.bind_progress_sender(
        handle.handle_id, parent_run_id="parent", chain_id="chain"
    )
    assert bind() is None
    assert (
        coord.bind_progress_sender("missing", parent_run_id="parent", chain_id="chain")
        is None
    )
    coord.attach_run(handle.handle_id, "child")
    sender = bind()
    assert sender is not None
    assert bind() is sender
    assert (
        coord.bind_progress_sender(
            handle.handle_id, parent_run_id="other", chain_id="chain"
        )
        is None
    )
    coord.attach_run(handle.handle_id, "replacement")
    sender.send("progress")
    identity = inbox.snapshot()[0].identity
    assert (
        identity.handle_id,
        identity.run_id,
        identity.parent_run_id,
        identity.chain_id,
        identity.agent,
    ) == (handle.handle_id, "child", "parent", "chain", "researcher")
    assert coord.get(handle.handle_id).run_id == "child"
    coord.finish(handle.handle_id, RUN_DONE)
    assert bind() is None
    with pytest.raises(MessageError, match="unavailable"):
        sender.send("late")
    assert coord.prune_terminal() == 1
    assert inbox.snapshot()[0].body == "progress"
    reader = inbox.reader("later", chain_id="chain", automatic=True)
    assert reader.collect().collected_count == 1


def test_progress_finish_and_post_race_has_one_admission_boundary():
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.Agents.fleet_messages import MessageError, MessageStore

    for _ in range(20):
        inbox = MessageStore().open_inbox("conversation")
        coord = FleetCoordinator(1, lambda: 0.0, message_inbox=inbox)
        handle = coord.reserve("work", None)
        coord.attach_run(handle.handle_id, "child")
        sender = coord.bind_progress_sender(
            handle.handle_id, parent_run_id="p", chain_id=None
        )
        sender.send("before terminalization")
        barrier = threading.Barrier(2)

        def post(barrier=barrier, sender=sender):
            barrier.wait(timeout=5)
            try:
                return sender.send("racing report")
            except MessageError as exc:
                assert exc.code == "unavailable"
                return None

        def finish(barrier=barrier, coord=coord, handle=handle):
            barrier.wait(timeout=5)
            coord.finish(handle.handle_id, RUN_DONE)

        with ThreadPoolExecutor(max_workers=2) as pool:
            posting = pool.submit(post)
            finishing = pool.submit(finish)
            accepted = posting.result(timeout=5)
            finishing.result(timeout=5)
        with pytest.raises(MessageError, match="unavailable"):
            sender.send("late")
        assert coord.prune_terminal() == 1
        assert len(inbox.snapshot()) == (2 if accepted else 1)
        assert inbox.snapshot()[0].body == "before terminalization"
        assert coord.live_count() == 0


def test_progress_finish_after_inbox_disposal_still_releases_handle():
    from tldw_chatbook.Agents.fleet_messages import MessageStore

    store = MessageStore()
    coord = FleetCoordinator(
        1, lambda: 0.0, message_inbox=store.open_inbox("conversation")
    )
    handle = coord.reserve("work", None)
    coord.attach_run(handle.handle_id, "child")
    assert (
        coord.bind_progress_sender(handle.handle_id, parent_run_id="p", chain_id=None)
        is not None
    )
    store.close()
    coord.finish(handle.handle_id, RUN_DONE)
    assert coord.live_count() == 0
    assert coord.prune_terminal() == 1


def test_progress_disabled_coordinator_returns_no_sender():
    coord = _coord()
    handle = coord.reserve("work", None)
    coord.attach_run(handle.handle_id, "child")
    assert (
        coord.bind_progress_sender(handle.handle_id, parent_run_id="p", chain_id=None)
        is None
    )


@pytest.mark.parametrize("first_operation", ["post", "finish"])
def test_progress_finish_and_send_serialize_under_the_actual_owner_lock(
    first_operation,
):
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.Agents.fleet_messages import MessageError, MessageStore

    store = MessageStore()
    inbox = store.open_inbox("conversation")
    coord = FleetCoordinator(1, lambda: 0.0, message_inbox=inbox)
    handle = coord.reserve("work", None)
    coord.attach_run(handle.handle_id, "child")
    sender = coord.bind_progress_sender(
        handle.handle_id, parent_run_id="p", chain_id=None
    )
    acquired = threading.Event()
    release = threading.Event()
    second_attempted = threading.Event()

    class GateLock:
        """Hold the first entrant at its real locked admission boundary."""

        def __init__(self):
            self.lock = threading.Lock()
            self.first = True

        def __enter__(self):
            if acquired.is_set():
                second_attempted.set()
            self.lock.acquire()
            if self.first:
                self.first = False
                acquired.set()
                assert release.wait(timeout=5)
            return self

        def __exit__(self, *args):
            self.lock.release()

    store._lock = GateLock()

    def post():
        try:
            return sender.send("boundary report")
        except MessageError as exc:
            assert exc.code == "unavailable"
            return None

    def finish():
        coord.finish(handle.handle_id, RUN_DONE)

    operations = {"post": post, "finish": finish}
    second_operation = "finish" if first_operation == "post" else "post"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(operations[first_operation])
        try:
            assert acquired.wait(timeout=5)
            second = pool.submit(operations[second_operation])
            assert second_attempted.wait(timeout=5)
        finally:
            release.set()
        outcomes = {
            first_operation: first.result(timeout=5),
            second_operation: second.result(timeout=5),
        }
    assert (outcomes["post"] is not None) == (first_operation == "post")
    assert len(inbox.snapshot()) == (1 if first_operation == "post" else 0)
    assert coord.live_count() == 0
    coord.prune_terminal()
    assert len(inbox.snapshot()) == (1 if first_operation == "post" else 0)


@pytest.mark.parametrize("close_owner", [False, True])
def test_progress_rebinding_does_not_return_a_closed_capability(close_owner):
    from tldw_chatbook.Agents.fleet_messages import MessageStore

    store = MessageStore()
    coord = FleetCoordinator(
        1, lambda: 0.0, message_inbox=store.open_inbox("conversation")
    )
    handle = coord.reserve("work", None)
    coord.attach_run(handle.handle_id, "child")
    sender = coord.bind_progress_sender(
        handle.handle_id, parent_run_id="p", chain_id=None
    )
    if close_owner:
        store.close()
    else:
        sender.close()
    assert (
        coord.bind_progress_sender(handle.handle_id, parent_run_id="p", chain_id=None)
        is None
    )
