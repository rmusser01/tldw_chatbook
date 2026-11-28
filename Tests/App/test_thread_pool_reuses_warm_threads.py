"""TASK-40608: the app's thread pool hands the next job to its most recently idle thread.

``concurrent.futures.ThreadPoolExecutor`` queues work on a ``queue.SimpleQueue``,
which wakes whichever idle thread its lock picks: work done one piece after
another moves from thread to thread, and each thread then builds what it keeps
for itself (a SQLite connection to each store, above all).
``FreshContextExecutor`` queues on ``_MostRecentIdleFirst`` instead.

These tests fail if a CPython upgrade moves the private seam the pool uses
(``ThreadPoolExecutor._work_queue``): the pool then keeps the stock queue and
``test_the_pool_queues_on_most_recent_idle_first`` says so.

The tests' own sleeping threads are daemons, so a queue that loses a wake-up
fails its test instead of holding the test process at exit.
"""

from __future__ import annotations

import collections
import gc
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait

import pytest

from tldw_chatbook.Utils.text_selection_crash_guard import (
    FreshContextExecutor,
    _MostRecentIdleFirst,
)

_LIMIT = 30.0


def _until(condition, what: str) -> None:
    deadline = time.monotonic() + _LIMIT
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting until {what}"
        time.sleep(0.001)


def _started_pool(threads: int) -> FreshContextExecutor:
    """A pool whose ``threads`` threads all exist and are all asleep."""
    pool = FreshContextExecutor(max_workers=threads, thread_name_prefix="warm")
    gate = threading.Barrier(threads)
    wait([pool.submit(gate.wait, _LIMIT) for _ in range(threads)], timeout=_LIMIT)
    _all_idle(pool, threads)
    return pool


def _all_idle(pool: FreshContextExecutor, threads: int) -> None:
    _until(lambda: len(pool._work_queue._idle) == threads, f"{threads} threads sleep")


def test_the_pool_queues_on_most_recent_idle_first():
    """The seam this rests on: the pool's private queue, replaced before any thread starts."""
    assert type(ThreadPoolExecutor(max_workers=1)._work_queue) is queue.SimpleQueue
    pool = FreshContextExecutor(max_workers=2, thread_name_prefix="warm")
    try:
        assert type(pool._work_queue) is _MostRecentIdleFirst
        assert pool.submit(lambda: threading.current_thread().name).result(_LIMIT).startswith("warm")
    finally:
        pool.shutdown(wait=True)


def test_work_done_one_piece_after_another_stays_on_one_thread():
    pool = _started_pool(6)
    try:
        used = []
        for _ in range(25):
            used.append(pool.submit(threading.get_ident).result(_LIMIT))
            _all_idle(pool, 6)
        assert len(set(used)) == 1, f"sequential work visited {len(set(used))} threads"
    finally:
        pool.shutdown(wait=True)


def test_the_thread_that_went_idle_last_takes_the_next_job():
    pool = _started_pool(4)
    try:
        release = [threading.Event() for _ in range(4)]
        ready = threading.Barrier(5)
        idents = []

        def hold(event):
            idents.append(threading.get_ident())
            ready.wait(_LIMIT)
            event.wait(_LIMIT)
            return threading.get_ident()

        jobs = [pool.submit(hold, event) for event in release]
        ready.wait(_LIMIT)
        # Let them go idle one at a time, in a known order.
        order = []
        for index in (2, 0, 3, 1):
            release[index].set()
            order.append(jobs[index].result(_LIMIT))
            _all_idle(pool, len(order))
        assert pool.submit(threading.get_ident).result(_LIMIT) == order[-1]
    finally:
        pool.shutdown(wait=True)


def test_a_burst_uses_as_many_threads_as_it_needs():
    pool = _started_pool(5)
    try:
        together = threading.Barrier(5)
        used = [job.result(_LIMIT) for job in [
            pool.submit(lambda: (together.wait(_LIMIT), threading.get_ident())[1]) for _ in range(5)
        ]]
        assert len(set(used)) == 5
    finally:
        pool.shutdown(wait=True)


def test_queued_jobs_run_in_the_order_they_were_submitted():
    pool = FreshContextExecutor(max_workers=1, thread_name_prefix="warm")
    try:
        started, release, ran = threading.Event(), threading.Event(), []
        first = pool.submit(lambda: (started.set(), release.wait(_LIMIT)))
        assert started.wait(_LIMIT)
        later = [pool.submit(ran.append, index) for index in range(8)]
        assert pool._work_queue.qsize() == 8 and not pool._work_queue.empty()
        release.set()
        wait([first, *later], timeout=_LIMIT)
        assert ran == list(range(8))
    finally:
        pool.shutdown(wait=True)


def test_shutdown_wakes_every_sleeping_thread():
    pool = _started_pool(6)
    threads = list(pool._threads)
    pool.shutdown(wait=True)
    assert len(threads) == 6 and not any(thread.is_alive() for thread in threads)


def test_a_dropped_pool_wakes_its_sleeping_threads():
    """The pool's weak-reference callback puts the wake-up itself."""
    pool = _started_pool(4)
    threads = list(pool._threads)
    del pool
    gc.collect()
    for thread in threads:
        thread.join(_LIMIT)
    assert not any(thread.is_alive() for thread in threads)


def test_cancelling_at_shutdown_drains_what_was_queued():
    pool = FreshContextExecutor(max_workers=1, thread_name_prefix="warm")
    started, release = threading.Event(), threading.Event()
    first = pool.submit(lambda: (started.set(), release.wait(_LIMIT)))
    assert started.wait(_LIMIT)
    queued = [pool.submit(time.sleep, 0) for _ in range(5)]
    pool.shutdown(wait=False, cancel_futures=True)
    assert all(job.cancelled() for job in queued)
    release.set()
    first.result(_LIMIT)
    pool.shutdown(wait=True)


def test_jobs_from_many_threads_all_run_exactly_once():
    pool = FreshContextExecutor(max_workers=8, thread_name_prefix="warm")
    try:
        ran = collections.Counter()
        guard = threading.Lock()

        def job(number):
            with guard:
                ran[number] += 1

        def submit(start):
            return [pool.submit(job, start + index) for index in range(500)]

        with ThreadPoolExecutor(max_workers=4) as submitters:
            batches = list(submitters.map(submit, range(0, 2000, 500)))
        done, pending = wait([job for batch in batches for job in batch], timeout=_LIMIT)
        assert not pending and len(done) == 2000
        assert set(ran) == set(range(2000)) and set(ran.values()) == {1}
    finally:
        pool.shutdown(wait=True)


def test_the_queue_serves_oldest_first_and_reports_what_is_left():
    items = _MostRecentIdleFirst()
    assert items.empty() and items.qsize() == 0
    with pytest.raises(queue.Empty):
        items.get_nowait()
    for value in "abc":
        items.put(value)
    assert items.qsize() == 3 and not items.empty()
    assert [items.get(), items.get_nowait(), items.get(block=False)] == ["a", "b", "c"]
    assert items.empty()


def test_an_item_put_while_a_thread_is_going_idle_is_not_missed():
    """The put finds no sleeper to wake, so the thread must look again once it is one."""
    items = _MostRecentIdleFirst()

    class PutsAfterTheFirstLook(collections.deque):
        looked = False

        def popleft(self):
            if not PutsAfterTheFirstLook.looked:
                PutsAfterTheFirstLook.looked = True
                items.put("late")  # nobody sleeps yet: no wake-up is sent
                raise IndexError
            return super().popleft()

    items._items = PutsAfterTheFirstLook()
    taken = []
    worker = threading.Thread(target=lambda: taken.append(items.get()), daemon=True)
    worker.start()
    worker.join(_LIMIT)
    assert taken == ["late"] and not worker.is_alive()
    assert items._idle == [], "the thread did not stay listed as sleeping"


def test_a_wake_up_sent_to_a_thread_that_already_has_an_item_is_passed_on():
    """Otherwise that item waits beside a sleeping thread until another put's wake-up arrives."""
    items = _MostRecentIdleFirst()
    sleeper_took = []
    sleeper = threading.Thread(target=lambda: sleeper_took.append(items.get()), daemon=True)
    sleeper.start()
    _until(lambda: len(items._idle) == 1, "the first thread sleeps")

    class ChosenWhileTaking(list):
        """The second thread's registration, with two puts landing right behind it."""

        def append(self, gate):
            super().append(gate)
            if len(self) == 2:
                items._items.append("first")  # taken by the registering thread itself
                items._items.append("second")
                super().pop().release()  # this put's wake-up reaches that same thread

    items._idle = ChosenWhileTaking(items._idle)
    taker_took = []
    taker = threading.Thread(target=lambda: taker_took.append(items.get()), daemon=True)
    taker.start()
    taker.join(_LIMIT)
    sleeper.join(_LIMIT)
    assert taker_took == ["first"]
    assert sleeper_took == ["second"], "the sleeping thread was never woken for the second item"


def test_a_woken_thread_that_finds_nothing_goes_back_to_sleep():
    items = _MostRecentIdleFirst()
    taken = []
    sleeper = threading.Thread(target=lambda: taken.append(items.get()), daemon=True)
    sleeper.start()
    _until(lambda: len(items._idle) == 1, "the thread sleeps")
    gate = items._idle.pop()
    gate.release()  # a wake-up whose item another thread already took
    _until(lambda: len(items._idle) == 1, "the thread sleeps again")
    assert taken == [] and sleeper.is_alive()
    items.put("real")
    sleeper.join(_LIMIT)
    assert taken == ["real"]


def test_put_is_safe_inside_another_put_or_get_on_the_same_thread():
    """A weak-reference callback may call ``put`` anywhere; nothing here may hold a lock."""
    items = _MostRecentIdleFirst()

    class PutsWhileAppending(collections.deque):
        nested = False

        def append(self, item):
            super().append(item)
            if not PutsWhileAppending.nested:
                PutsWhileAppending.nested = True
                items.put("from the callback")

    items._items = PutsWhileAppending()
    worker = threading.Thread(target=items.put, args=("outer",), daemon=True)
    worker.start()
    worker.join(_LIMIT)
    assert not worker.is_alive(), "put waited on itself"
    assert [items.get_nowait(), items.get_nowait()] == ["outer", "from the callback"]
