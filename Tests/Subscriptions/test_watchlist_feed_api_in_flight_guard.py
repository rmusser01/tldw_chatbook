"""task-19562 part A: the feed and API arms must share the in-flight guard.

`_check_url_guarded` (task-16838) serializes concurrent checks of the same
source -- but only for the url-family arms (`url`, `url_list`, `sitemap`).
The **feed** arm (`FeedMonitor().check_feed`) and the **API** arm never
registered a claim at all, and feeds are the commonest source type. A
scheduler tick overlapping a manual "Check Now" therefore ran the check
TWICE: the alert notification fired twice and statistics double-counted.

The tests below force the overlap for real -- the check is gated on an
`asyncio.Event`, the second entrant starts while the first is still inside
its await -- rather than asserting that a guard function was called.

The skip must also carry a `DISPOSITION_SKIPPED_IN_FLIGHT` disposition, not
just return zero items. That is load-bearing: `_entirely_skipped_
dispositions` -> `execute_run` reads it to SKIP source-health accounting, so
a turned-away check cannot take `record_check_result`'s SUCCESS branch and
reset the auto-pause breaker, clear `last_error` and stamp
`last_successful_check` for a run that never contacted the source. Zero items
with `None` dispositions would have looked exactly like a clean "nothing new"
check.
"""

from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.local_watchlists_service import (
    LocalWatchlistsService,
)

pytestmark = pytest.mark.unit


async def _run_check(service: LocalWatchlistsService, source_id: int) -> dict:
    """One full check the way every real entrant does it: launch + execute."""
    launched = await service.launch_run(source_id=source_id)
    return await service.execute_run(launched["run_id"])


def _gate_feed_checks(monkeypatch, gate: asyncio.Event, started: asyncio.Event):
    """Make FeedMonitor.check_feed block on `gate`, counting entries."""
    from tldw_chatbook.Subscriptions import monitoring_engine

    calls: list[int] = []

    async def gated_check_feed(self, config, *args, **kwargs):
        calls.append(1)
        started.set()
        await gate.wait()
        return [
            {
                "title": "Item",
                "url": "https://example.com/feed/1",
                "content": "body",
            }
        ]

    monkeypatch.setattr(
        monitoring_engine.FeedMonitor, "check_feed", gated_check_feed
    )
    return calls


@pytest.mark.asyncio
async def test_overlapping_feed_checks_run_the_check_once(tmp_path, monkeypatch):
    """The headline: a scheduler tick and a Check Now must not both fetch."""
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    source_id = db.add_subscription(
        name="A feed", type="rss", source="https://example.com/feed.xml"
    )
    scheduler_service = LocalWatchlistsService(db_factory=lambda: db)
    ui_service = LocalWatchlistsService(db_factory=lambda: db)

    gate = asyncio.Event()
    started = asyncio.Event()
    calls = _gate_feed_checks(monkeypatch, gate, started)

    scheduled = asyncio.create_task(_run_check(scheduler_service, source_id))
    await asyncio.wait_for(started.wait(), timeout=10)

    # The manual entrant arrives while the scheduled fetch is still gated.
    manual = asyncio.create_task(_run_check(ui_service, source_id))
    # The scheduled entrant is blocked on `gate` and CANNOT progress, so this
    # wait is deterministic, not a race: it only gives the manual entrant loop
    # time to get through launch_run's DB work and reach the guard.
    await asyncio.sleep(0.25)
    gate.set()

    scheduled_result = await asyncio.wait_for(scheduled, timeout=10)
    manual_result = await asyncio.wait_for(manual, timeout=10)

    assert len(calls) == 1, (
        f"the feed was fetched {len(calls)} times for one overlap; the "
        "second entrant was not turned away"
    )

    dispositions = [
        (result["stats"] or {}).get("dispositions") for result in
        (scheduled_result, manual_result)
    ]
    skipped_counts = [
        int((d or {}).get("skipped", 0) or 0) for d in dispositions
    ]
    assert sum(skipped_counts) == 1, (
        f"expected exactly one run to record a skip, got {dispositions}"
    )


@pytest.mark.asyncio
async def test_skipped_feed_check_does_not_count_as_a_successful_check(
    tmp_path, monkeypatch
):
    """The skip must not reset source health for a check that never ran."""
    from tldw_chatbook.Subscriptions.local_watchlists_service import (
        _entirely_skipped_dispositions,
    )

    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    source_id = db.add_subscription(
        name="A feed", type="rss", source="https://example.com/feed.xml"
    )
    scheduler_service = LocalWatchlistsService(db_factory=lambda: db)
    ui_service = LocalWatchlistsService(db_factory=lambda: db)

    gate = asyncio.Event()
    started = asyncio.Event()
    _gate_feed_checks(monkeypatch, gate, started)

    scheduled = asyncio.create_task(_run_check(scheduler_service, source_id))
    await asyncio.wait_for(started.wait(), timeout=10)
    manual = asyncio.create_task(_run_check(ui_service, source_id))
    await asyncio.sleep(0.25)  # deterministic: the first is gate-blocked
    gate.set()
    results = [
        await asyncio.wait_for(scheduled, timeout=10),
        await asyncio.wait_for(manual, timeout=10),
    ]

    skipped_runs = [
        r for r in results
        if _entirely_skipped_dispositions((r["stats"] or {}).get("dispositions"))
    ]
    assert len(skipped_runs) == 1, (
        "the turned-away run must match _entirely_skipped_dispositions, or "
        "execute_run will run the SUCCESS health path for a check that never "
        "contacted the source"
    )


@pytest.mark.asyncio
async def test_distinct_feed_sources_still_check_concurrently(
    tmp_path, monkeypatch
):
    """The guard is per-source: two different feeds must not block each other."""
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    first = db.add_subscription(
        name="Feed one", type="rss", source="https://example.com/one.xml"
    )
    second = db.add_subscription(
        name="Feed two", type="rss", source="https://example.com/two.xml"
    )
    service_a = LocalWatchlistsService(db_factory=lambda: db)
    service_b = LocalWatchlistsService(db_factory=lambda: db)

    gate = asyncio.Event()
    started = asyncio.Event()
    calls = _gate_feed_checks(monkeypatch, gate, started)

    task_a = asyncio.create_task(_run_check(service_a, first))
    await asyncio.wait_for(started.wait(), timeout=10)
    task_b = asyncio.create_task(_run_check(service_b, second))
    await asyncio.sleep(0.25)  # deterministic: task_a is gate-blocked
    gate.set()
    await asyncio.wait_for(task_a, timeout=10)
    await asyncio.wait_for(task_b, timeout=10)

    assert len(calls) == 2, (
        "two DIFFERENT feed sources must both check; the guard is keyed per "
        f"source, but only {len(calls)} check(s) ran"
    )


@pytest.mark.asyncio
async def test_a_failed_feed_check_releases_the_guard(tmp_path, monkeypatch):
    """A raise must not strand the source as permanently in flight."""
    from tldw_chatbook.Subscriptions import monitoring_engine

    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    source_id = db.add_subscription(
        name="A feed", type="rss", source="https://example.com/feed.xml"
    )
    service = LocalWatchlistsService(db_factory=lambda: db)

    async def boom(self, config, *args, **kwargs):
        raise RuntimeError("feed exploded")

    monkeypatch.setattr(monitoring_engine.FeedMonitor, "check_feed", boom)
    await _run_check(service, source_id)

    calls: list[int] = []

    async def ok(self, config, *args, **kwargs):
        calls.append(1)
        return []

    monkeypatch.setattr(monitoring_engine.FeedMonitor, "check_feed", ok)
    await _run_check(service, source_id)

    assert calls == [1], (
        "the next check never ran: the failed check stranded the in-flight "
        "claim"
    )
