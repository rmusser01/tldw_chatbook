"""task-16838: the per-(subscription, url) in-flight guard.

From the TASK-15764 review (PR #1679, finding 1): no serialization mechanism
existed for concurrent checks of the same source. The scheduler is an async
worker on the app's event loop, and a UI "Check Now" runs `launch_run` /
`execute_run` as a coroutine worker on the same loop -- so a scheduled check
and a manual check of source X could interleave across `check_url`'s awaits
(the network fetch plus the off-loop sqlite/CPU hops). Both read the same
baseline before either wrote, and the review's forced interleave got
`dispositions=['changed','changed']`: one page change double-reported, two
snapshots written.

The guard (`LocalWatchlistsService._check_url_guarded`) is a module-level
in-flight set keyed `(id(db), subscription_id, url)`, claimed before the
check and released in `finally`. The second entrant SKIPS with an honest
`skipped` disposition; it does not queue or wait.

The headline test below was born red at HEAD `1af8c0414` (pre-guard): the
manual entrant reached the network while the scheduled fetch was still
gated, and the run pair double-reported exactly as the review demonstrated.

Threading note (same as `test_url_monitor_off_loop.py`): the file-backed DBs
under `tmp_path` are load-bearing -- `SubscriptionsDB` keeps thread-local
connections, so the check path's `run_db_off_loop` hops need a database
whose file another thread can open.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.local_watchlists_service import (
    LocalWatchlistsService,
)

pytestmark = pytest.mark.unit


def _in_flight() -> set:
    """The module-level claim registry, looked up lazily.

    `getattr` with a default rather than a top-level import, so this file
    still COLLECTS on a pre-guard tree and reddens on the double-report
    behavior itself (the born-red evidence), not on an ImportError. The
    release guarantees are also pinned behaviorally (the recovered checks
    below), so a renamed registry cannot silently hollow out AC#3.
    """
    from tldw_chatbook.Subscriptions import local_watchlists_service as svc

    return getattr(svc, "_IN_FLIGHT_URL_CHECKS", set())


_PAGE_ONE = (
    "<html><body><p>alpha bravo charlie.</p>\n<p>delta echo foxtrot.</p></body></html>"
)
_PAGE_TWO = (
    "<html><body><p>alpha bravo charlie.</p>\n<p>golf hotel india.</p></body></html>"
)

#: All the counters a url-family run zero-fills (`_disposition_counts`).
_ZERO_COUNTS = {
    "changed": 0,
    "unchanged": 0,
    "withheld": 0,
    "baseline": 0,
    "rebaselined": 0,
    "error": 0,
    "skipped": 0,
}


def _response(text: str, url: str):
    """Stand in for the `httpx.Response` `guarded_fetch_httpx_async` returns."""
    return SimpleNamespace(
        status_code=200,
        headers={"content-type": "text/html"},
        text=text,
        final_url=url,
        raise_for_status=lambda: None,
    )


def _serve(monkeypatch, body_for) -> list[str]:
    """Serve HTML from the real fetch seam. Returns the fetch log."""
    fetched: list[str] = []

    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        fetched.append(url)
        return _response(body_for(url, len(fetched)), url)

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        fake_guarded,
    )
    return fetched


async def _run_check(service: LocalWatchlistsService, source_id: int) -> dict:
    """One full check the way every real entrant does it: launch + execute."""
    launched = await service.launch_run(source_id=source_id)
    return await service.execute_run(launched["run_id"])


def _snapshot_count(db: SubscriptionsDB, source_id: int, url: str) -> int:
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM url_snapshots "
            "WHERE subscription_id = ? AND url = ?",
            (source_id, url),
        ).fetchone()
    return int(row["n"])


def _item_count(db: SubscriptionsDB, source_id: int) -> int:
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM subscription_items WHERE subscription_id = ?",
            (source_id,),
        ).fetchone()
    return int(row["n"])


# --- AC#1: the interleave the 15764 review demonstrated ----------------------


@pytest.mark.asyncio
async def test_scheduled_and_manual_check_of_same_source_cannot_double_report(
    tmp_path, monkeypatch
):
    """Born red at HEAD `1af8c0414`: one page change, reported once.

    Deterministic interleave: the fetch is gated on an `asyncio.Event`. The
    "scheduled" entrant starts and is held mid-fetch; the "manual" entrant
    then starts for the same source. Pre-guard, the manual entrant also went
    to the network, both read the same baseline, and the pair reported
    `changed` twice with two new snapshots. With the guard, the manual run
    completes as `skipped` WITHOUT touching the network, and only after that
    is the scheduled fetch released.

    Two service instances over the one shared db mirror production exactly:
    `app.py` wires the UI's `local_watchlists_service` and the
    `WatchlistCheckHandler`'s own default-constructed service over ONE
    `SubscriptionsDB` (task-15463) -- which is why instance-level guard state
    would never have seen this interleave.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    url = "https://example.com/page"
    source_id = db.add_subscription(name="Watched page", type="url", source=url)
    scheduler_service = LocalWatchlistsService(db_factory=lambda: db)
    ui_service = LocalWatchlistsService(db_factory=lambda: db)

    # Seed the baseline with an ordinary, ungated check.
    _serve(monkeypatch, lambda u, n: _PAGE_ONE)
    seeded = await _run_check(scheduler_service, source_id)
    assert seeded["stats"]["dispositions"]["baseline"] == 1
    assert _snapshot_count(db, source_id, url) == 1

    # Now the page changes, and the fetch is gated so the interleave window
    # is held open deterministically.
    gate = asyncio.Event()
    fetch_started = asyncio.Event()
    second_fetch = asyncio.Event()
    overlap_fetches: list[str] = []

    async def gated_fetch(fetch_url, *, client, max_bytes, **kwargs):
        overlap_fetches.append(fetch_url)
        if len(overlap_fetches) == 1:
            fetch_started.set()
        else:
            second_fetch.set()
        await gate.wait()
        return _response(_PAGE_TWO, fetch_url)

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        gated_fetch,
    )

    scheduled = asyncio.create_task(_run_check(scheduler_service, source_id))
    await asyncio.wait_for(fetch_started.wait(), timeout=10)

    manual = asyncio.create_task(_run_check(ui_service, source_id))
    # Either the manual run finishes without the network (the guard), or it
    # reaches the network while the scheduled fetch is still in flight (the
    # pre-guard interleave). Wait for whichever happens so the test FAILS
    # rather than hangs on the pre-guard shape.
    second_waiter = asyncio.create_task(second_fetch.wait())
    done, _ = await asyncio.wait(
        {manual, second_waiter}, timeout=15, return_when=asyncio.FIRST_COMPLETED
    )
    assert done, (
        "the manual check neither finished nor reached the network -- "
        "no entrant made progress"
    )
    interleaved = second_fetch.is_set()

    gate.set()
    scheduled_run = await asyncio.wait_for(scheduled, timeout=10)
    manual_run = await asyncio.wait_for(manual, timeout=10)
    second_waiter.cancel()

    # THE core claim, red pre-guard: while the scheduled check was mid-fetch,
    # the manual entrant must not also have fetched the page.
    assert not interleaved and len(overlap_fetches) == 1, (
        "a manual Check Now overlapping a scheduled check of the same source "
        "went to the network too -- the 15764 review's double-check interleave"
    )

    # The winner reports the change exactly once...
    assert scheduled_run["status"] == "completed"
    assert scheduled_run["stats"]["dispositions"] == {**_ZERO_COUNTS, "changed": 1}
    # ...the loser says, honestly, that it checked nothing...
    assert manual_run["status"] == "completed"
    assert manual_run["stats"]["dispositions"] == {**_ZERO_COUNTS, "skipped": 1}
    assert manual_run["found_count"] == 0

    # ...and the durable record carries ONE report and ONE new snapshot,
    # not two of each (pre-guard: dispositions ['changed','changed'], three
    # snapshot rows, and the same change persisted from both runs).
    assert _item_count(db, source_id) == 1
    assert _snapshot_count(db, source_id, url) == 2

    assert not _in_flight(), "no claim may outlive its check"


# --- AC#2: distinct sources stay concurrent ----------------------------------


@pytest.mark.asyncio
async def test_distinct_sources_still_check_concurrently(tmp_path, monkeypatch):
    """The guard is per-(subscription, url), not a global serializer.

    Both sources' fetches are gated; both must be IN FLIGHT AT ONCE (both
    gated-fetch calls observed while neither run has completed) before the
    gate opens. A global lock would hold the second source's fetch back
    until the first run finished, and the wait below would time out.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    url_a = "https://a.example/one"
    url_b = "https://b.example/two"
    source_a = db.add_subscription(name="Source A", type="url", source=url_a)
    source_b = db.add_subscription(name="Source B", type="url", source=url_b)
    service = LocalWatchlistsService(db_factory=lambda: db)

    gate = asyncio.Event()
    started: dict[str, asyncio.Event] = {
        url_a: asyncio.Event(),
        url_b: asyncio.Event(),
    }

    async def gated_fetch(fetch_url, *, client, max_bytes, **kwargs):
        started[fetch_url].set()
        await gate.wait()
        return _response(_PAGE_ONE, fetch_url)

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        gated_fetch,
    )

    run_a = asyncio.create_task(_run_check(service, source_a))
    run_b = asyncio.create_task(_run_check(service, source_b))
    await asyncio.wait_for(
        asyncio.gather(started[url_a].wait(), started[url_b].wait()), timeout=10
    )
    assert not run_a.done() and not run_b.done()
    gate.set()

    result_a = await asyncio.wait_for(run_a, timeout=10)
    result_b = await asyncio.wait_for(run_b, timeout=10)
    for result in (result_a, result_b):
        assert result["status"] == "completed"
        assert result["stats"]["dispositions"] == {**_ZERO_COUNTS, "baseline": 1}
    assert not _in_flight()


# --- AC#3: nothing can strand a source as "in flight" ------------------------


@pytest.mark.asyncio
async def test_a_failed_check_releases_the_guard(tmp_path, monkeypatch):
    """A fetch that raises must not leave the pair claimed forever."""
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    url = "https://example.com/page"
    source_id = db.add_subscription(name="Watched page", type="url", source=url)
    service = LocalWatchlistsService(db_factory=lambda: db)

    async def dead_host(fetch_url, *, client, max_bytes, **kwargs):
        raise ConnectionError("host unreachable")

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        dead_host,
    )
    failed = await _run_check(service, source_id)
    assert failed["status"] == "failed"
    assert not _in_flight(), (
        "the failed check left its claim behind -- the source is stranded"
    )

    # The next check must run for real, not skip against a ghost claim.
    _serve(monkeypatch, lambda u, n: _PAGE_ONE)
    recovered = await _run_check(service, source_id)
    assert recovered["status"] == "completed"
    assert recovered["stats"]["dispositions"] == {**_ZERO_COUNTS, "baseline": 1}


@pytest.mark.asyncio
async def test_a_cancelled_check_releases_the_guard(tmp_path, monkeypatch):
    """Cancellation mid-fetch (the user navigating away) releases the claim.

    `execute_run`'s own `except asyncio.CancelledError` records the run as
    failed and re-raises; the guard's `finally` must have discarded the
    claim on that same unwind.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    url = "https://example.com/page"
    source_id = db.add_subscription(name="Watched page", type="url", source=url)
    service = LocalWatchlistsService(db_factory=lambda: db)

    gate = asyncio.Event()
    fetch_started = asyncio.Event()

    async def gated_fetch(fetch_url, *, client, max_bytes, **kwargs):
        fetch_started.set()
        await gate.wait()
        return _response(_PAGE_ONE, fetch_url)

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        gated_fetch,
    )
    task = asyncio.create_task(_run_check(service, source_id))
    await asyncio.wait_for(fetch_started.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not _in_flight(), (
        "the cancelled check left its claim behind -- the source is stranded"
    )

    _serve(monkeypatch, lambda u, n: _PAGE_ONE)
    recovered = await _run_check(service, source_id)
    assert recovered["status"] == "completed"
    assert recovered["stats"]["dispositions"] == {**_ZERO_COUNTS, "baseline": 1}


# --- Same-run re-entry: sequential duplicates neither deadlock nor self-skip --


@pytest.mark.asyncio
async def test_duplicate_urls_within_one_run_neither_deadlock_nor_self_skip(
    tmp_path, monkeypatch
):
    """A url_list that lists the same URL twice keeps its pre-guard behavior.

    The 15764 review established the url_list loop is sequential: each check
    is awaited to completion before the next starts, so by the time the loop
    reaches the duplicate, the guard's `finally` has already released the
    claim. The duplicate is checked (fetched) again exactly as before --
    first `baseline`, then `unchanged` -- with no skip and no deadlock.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    url = "https://a.example/one"
    source_id = db.add_subscription(
        name="Twice-listed page", type="url_list", source=f"{url}\n{url}"
    )
    service = LocalWatchlistsService(db_factory=lambda: db)
    fetched = _serve(monkeypatch, lambda u, n: _PAGE_ONE)

    run = await asyncio.wait_for(_run_check(service, source_id), timeout=30)

    assert run["status"] == "completed"
    assert run["stats"]["dispositions"] == {
        **_ZERO_COUNTS,
        "baseline": 1,
        "unchanged": 1,
    }
    assert fetched == [url, url], "both listed occurrences must really check"
    assert not _in_flight()
