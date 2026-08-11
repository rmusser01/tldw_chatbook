"""task-15463: one SubscriptionsDB per wiring, and due checks off the loop.

Two defects, one path, measured in `Docs/Design/2026-08-11-input-latency-audit.md`:

1. `LocalWatchlistsService._db()` called `self.db_factory()` on EVERY service
   method, and the production factory (`app.py`'s
   `_wire_watchlists_and_notifications_services`) constructed a brand-new
   `SubscriptionsDB` each time -- a ~52-statement `executescript` plus
   migration probes, measured at 3.4 ms against 0.04 ms for the same query on
   a held instance (~85x; 35 ms for the first construction). A single
   Watchlists refresh fires five or more of those.

2. A scheduled check ran its sqlite bookkeeping and its feed parse inline on
   the event loop, so an unattended check -- enabled by default, firing on
   whatever tab the user is looking at -- froze the UI for the duration.

The tests below are the evidence for both, and they are deliberately
*mechanical*: a factory that counts, a sqlite trace callback bound to the
event loop's own connection, and thread identities. Nothing here asserts a
duration, so nothing here is timing-flaky.

Threading note: the file-backed DBs under `tmp_path` are load-bearing, not
habit. `SubscriptionsDB` keeps thread-local connections and builds its schema
on the constructing thread, so an in-memory instance handed across a thread
hop would land on a private, EMPTY database (see the class's own
`_initialize_schema` docstring). That is exactly why the offload helper
refuses to hop for an in-memory DB, which `test_in_memory_db_work_stays_on_
the_calling_thread` pins.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler import (
    WatchlistCheckHandler,
)
from tldw_chatbook.Subscriptions.db_offload import run_db_off_loop
from tldw_chatbook.Subscriptions.local_watchlists_service import (
    LocalWatchlistsService,
)
from tldw_chatbook.Subscriptions.monitoring_engine import FeedMonitor

pytestmark = pytest.mark.unit


_RSS_BODY = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>First</title><link>https://example.com/1</link>
        <description>one</description></item>
  <item><title>Second</title><link>https://example.com/2</link>
        <description>two</description></item>
</channel></rss>"""

_JSON_BODY = """{"items": [
  {"title": "First", "url": "https://example.com/1", "content_text": "one"}
]}"""


def _response(text: str, url: str, *, content_type: str):
    """Stand in for the `httpx.Response` `guarded_fetch_httpx_async` returns."""
    return SimpleNamespace(
        status_code=200,
        headers={"content-type": content_type},
        text=text,
        final_url=url,
        raise_for_status=lambda: None,
    )


def _serve(monkeypatch, body: str, *, content_type: str) -> list[str]:
    """Serve one body from the real fetch seam. Returns the fetch log."""
    fetched: list[str] = []

    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        fetched.append(url)
        return _response(body, url, content_type=content_type)

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        fake_guarded,
    )
    return fetched


def _add_due_source(db: SubscriptionsDB, **kwargs) -> int:
    """Add an active source whose cadence has already elapsed."""
    kwargs.setdefault("check_frequency", 3600)
    subscription_id = db.add_subscription(**kwargs)
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET last_checked = ? WHERE id = ?",
            (stale, subscription_id),
        )
    return subscription_id


def _task(subscription_id: int) -> dict:
    """The projected task shape the scheduler dispatches."""
    return {
        "id": f"watchlist:{subscription_id}",
        "type": "watchlist_job",
        "title": "Scheduled source",
        "owner_id": "local",
    }


# --- AC#1: one instance -----------------------------------------------------


@pytest.mark.asyncio
async def test_the_service_resolves_its_db_factory_exactly_once(tmp_path):
    """Every service method used to reconstruct the database.

    Mutation: drop the memoization in `_db()` and this reddens at 8+ calls,
    which is precisely the ~85x-per-call cost the audit measured.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    calls: list[int] = []

    def counting_factory() -> SubscriptionsDB:
        calls.append(1)
        return db

    service = LocalWatchlistsService(db_factory=counting_factory)
    source_id = db.add_subscription(
        name="Feed", type="rss", source="https://example.com/feed"
    )

    await service.list_sources()
    await service.get_source(source_id)
    await service.list_items()
    await service.list_runs()
    await service.list_watchlists()
    launched = await service.launch_run(source_id=source_id)
    await service.get_run(launched["run_id"])
    await service.list_alert_rules(source_id=source_id)

    assert len(calls) == 1, (
        "LocalWatchlistsService must construct its SubscriptionsDB once per "
        f"wiring, not once per operation (factory called {len(calls)} times)"
    )


@pytest.mark.asyncio
async def test_reassigning_db_factory_repoints_the_service(tmp_path):
    """The injectable factory seam survives the caching.

    `Tests/UI/test_watchlists_inspector.py` repoints
    `app.local_watchlists_service.db_factory` at a spied database mid-test, so
    a cache that outlived a factory reassignment would silently keep serving
    the old instance and that test would assert against a database nothing
    writes to.
    """
    first = SubscriptionsDB(tmp_path / "first.db", "test")
    second = SubscriptionsDB(tmp_path / "second.db", "test")
    first.add_subscription(name="In first", type="rss", source="https://a.example")
    second.add_subscription(name="In second", type="rss", source="https://b.example")

    service = LocalWatchlistsService(db_factory=lambda: first)
    assert [row["title"] for row in await service.list_sources()] == ["In first"]

    service.db_factory = lambda: second

    assert [row["title"] for row in await service.list_sources()] == ["In second"], (
        "assigning a new db_factory must take effect on the next call -- the "
        "cached instance has to be dropped with the factory that produced it"
    )


# --- AC#2: nothing synchronous on the loop ----------------------------------


@pytest.mark.parametrize(
    ("source_type", "body", "content_type"),
    [
        ("rss", _RSS_BODY, "application/rss+xml"),
        ("json_feed", _JSON_BODY, "application/json"),
        ("url", "<html><body><article>hello</article></body></html>", "text/html"),
    ],
)
@pytest.mark.asyncio
async def test_a_scheduled_check_runs_no_sqlite_on_the_event_loop(
    tmp_path, monkeypatch, source_type, body, content_type
):
    """A due check must not execute one single statement on the loop thread.

    The probe is `sqlite3.Connection.set_trace_callback` installed on the
    connection belonging to THIS thread -- the thread the event loop runs on.
    `SubscriptionsDB` connections are thread-local, so any statement this
    callback sees is by construction a statement that ran on the event loop:
    no timing, no sampling, no flake. Statements executed on
    `asyncio.to_thread`'s workers use those workers' own connections and are
    invisible to it, which is the whole point.

    Mutation: revert any one of the `run_db_off_loop` hops on the check path
    (the handler's `get_subscription`, `launch_run`, `_mark_run_started`, the
    filter/alert loads, the item upsert, `record_check_result`,
    `record_run_result`, or `URLMonitor`'s baseline read / snapshot write) and
    this reddens with that statement quoted in the failure message.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    subscription_id = _add_due_source(
        db, name="Scheduled", type=source_type, source="https://example.com/feed"
    )
    _serve(monkeypatch, body, content_type=content_type)
    handler = WatchlistCheckHandler(subscriptions_db=db)

    loop_statements: list[str] = []
    db.conn.set_trace_callback(loop_statements.append)
    try:
        await handler.handle(_task(subscription_id))
    finally:
        db.conn.set_trace_callback(None)

    assert not loop_statements, (
        "a scheduled check ran SQL on the event-loop thread: "
        f"{loop_statements[:3]}"
    )

    # Never vacuous: the check must really have done its work.
    with db.transaction() as conn:
        runs = conn.execute(
            "SELECT status FROM local_watchlist_runs WHERE source_id = ?",
            (subscription_id,),
        ).fetchall()
    assert [row["status"] for row in runs] == ["completed"], (
        "the check has to have produced its usual completed run receipt, or "
        "this test proves nothing about where the work ran"
    )


@pytest.mark.parametrize(
    ("source_type", "body", "content_type", "parser"),
    [
        ("rss", _RSS_BODY, "application/rss+xml", "_parse_xml_feed"),
        ("json_feed", _JSON_BODY, "application/json", "_parse_json_feed"),
    ],
)
@pytest.mark.asyncio
async def test_the_feed_parse_runs_off_the_event_loop_thread(
    monkeypatch, source_type, body, content_type, parser
):
    """`ET.fromstring` over a whole feed body is not loop work.

    The fetch either side of it is already async httpx and stays there; only
    the parse moves.
    """
    monitor = FeedMonitor()
    _serve(monkeypatch, body, content_type=content_type)
    real_parser = getattr(monitor, parser)
    parse_threads: list[int] = []

    def spy(*args, **kwargs):
        parse_threads.append(threading.get_ident())
        return real_parser(*args, **kwargs)

    setattr(monitor, parser, spy)

    items = await monitor.check_feed(
        {"id": 1, "name": "Feed", "type": source_type, "source": "https://example.com/feed"}
    )

    assert items, "the parse must have produced items, or the spy proves nothing"
    assert parse_threads, "the parser must have been called at all"
    assert parse_threads[0] != threading.get_ident(), (
        "the feed body parse must run under asyncio.to_thread, not inline on "
        "the event loop"
    )


@pytest.mark.asyncio
async def test_in_memory_db_work_stays_on_the_calling_thread(tmp_path):
    """The offload helper must refuse to hop for an in-memory database.

    `SubscriptionsDB` builds its schema on the constructing thread and keeps
    connections thread-local, so a hop would hand the work a private, empty
    `:memory:` database -- writes would vanish and reads would raise
    `no such table`. Two live callers depend on this: `WatchlistPreviewService`
    (a throwaway `:memory:` DB so previews persist nothing) and the in-memory
    service tests in `test_watchlist_noise_not_volume.py`.
    """
    memory_db = SubscriptionsDB(":memory:", "test")
    file_db = SubscriptionsDB(tmp_path / "subs.db", "test")

    def where_am_i() -> int:
        return threading.get_ident()

    assert await run_db_off_loop(memory_db, where_am_i) == threading.get_ident()
    assert await run_db_off_loop(file_db, where_am_i) != threading.get_ident()


@pytest.mark.asyncio
async def test_offload_preserves_call_order_and_propagates_errors(tmp_path):
    """Hops are awaited one at a time, so ordering and error paths are intact."""
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    seen: list[str] = []

    for label in ("first", "second", "third"):
        await run_db_off_loop(db, seen.append, label)
    assert seen == ["first", "second", "third"]

    def boom() -> None:
        raise sqlite3.OperationalError("no such table: nope")

    with pytest.raises(sqlite3.OperationalError):
        await run_db_off_loop(db, boom)


@pytest.mark.asyncio
async def test_every_db_call_a_scheduled_check_makes_runs_off_the_loop_thread(
    tmp_path, monkeypatch
):
    """The thread-identity half of AC#2, complementing the trace probe.

    The trace callback above proves nothing ran on the loop's connection;
    this proves the calls happened at all, and names the thread each ran on.
    Deliberately identity-based rather than a wall-clock "did the loop stay
    responsive" measurement -- a duration floor would be machine-dependent
    and would go flaky on a fast host, which is the one thing a regression
    guard must not do.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    subscription_id = _add_due_source(
        db, name="Scheduled", type="rss", source="https://example.com/feed"
    )
    _serve(monkeypatch, _RSS_BODY, content_type="application/rss+xml")

    threads: dict[str, int] = {}
    for name in ("get_subscription", "record_check_result"):
        real = getattr(db, name)

        def spy(*args, _name=name, _real=real, **kwargs):
            threads[_name] = threading.get_ident()
            return _real(*args, **kwargs)

        setattr(db, name, spy)

    handler = WatchlistCheckHandler(subscriptions_db=db)
    await handler.handle(_task(subscription_id))

    assert set(threads) == {"get_subscription", "record_check_result"}, (
        "both calls must have happened -- a check that skipped them proves "
        f"nothing about where they ran (saw {sorted(threads)})"
    )
    loop_thread = threading.get_ident()
    for name, thread_id in threads.items():
        assert thread_id != loop_thread, (
            f"SubscriptionsDB.{name} ran on the event-loop thread during a "
            "scheduled check"
        )
