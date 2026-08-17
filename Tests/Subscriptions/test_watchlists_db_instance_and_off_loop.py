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

import asyncio
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler import (
    WatchlistCheckHandler,
)
from tldw_chatbook.Subscriptions import monitoring_engine
from tldw_chatbook.Subscriptions.db_offload import run_db_off_loop
from tldw_chatbook.Subscriptions.local_watchlists_service import (
    LocalWatchlistsService,
)
from tldw_chatbook.Subscriptions.monitoring_engine import (
    ContentExtractor,
    FeedMonitor,
    URLMonitor,
)

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


def _serve_in_order(monkeypatch, bodies: list[str]) -> None:
    """Serve one HTML body per request from the real fetch seam."""
    remaining = list(bodies)

    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        return _response(remaining.pop(0), url, content_type="text/html")

    monkeypatch.setattr(monitoring_engine, "guarded_fetch_httpx_async", fake_guarded)


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


def test_concurrent_threads_still_construct_the_database_exactly_once(tmp_path):
    """`_db()` is called from more than one thread, and not always the loop's.

    Review round 1, Important. `list_home_run_snapshot` is synchronous and
    calls `_db()` itself, and Home runs it under `asyncio.to_thread`
    (`Home/active_work_adapter.py::_compute_active_work_fields`) -- so a
    worker thread can reach an unprimed cache at the same moment the event
    loop does. Unlocked, that double-constructs, which for a CONSTRUCTING
    factory means a second `_initialize_schema` running while other threads
    hold connections: the exact schema-cache poisoning this task removed.

    The factory below sleeps inside the construction to widen the window --
    without the lock this fails with 8 constructions, not a rare flake.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    constructions: list[int] = []
    lock = threading.Lock()

    def slow_counting_factory() -> SubscriptionsDB:
        with lock:
            constructions.append(threading.get_ident())
        time.sleep(0.05)
        return db

    service = LocalWatchlistsService(db_factory=slow_counting_factory)
    resolved: list[SubscriptionsDB] = []
    start = threading.Barrier(8)

    def hammer() -> None:
        start.wait()
        instance = service._db()
        with lock:
            resolved.append(instance)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(constructions) == 1, (
        "eight threads racing an unprimed cache must produce ONE "
        f"SubscriptionsDB, not {len(constructions)}"
    )
    assert len(resolved) == 8 and all(instance is db for instance in resolved), (
        "every waiting thread must be handed the one constructed instance"
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


@pytest.mark.asyncio
async def test_a_source_deleted_mid_launch_still_raises_keyerror(tmp_path):
    """The `launch_run` contract survives its own widened window.

    Review round 1, Minor 3. The existence check and the INSERT are now two
    awaits apart, so a source can be deleted between them -- and
    `local_watchlist_runs.source_id` is a foreign key with
    `PRAGMA foreign_keys = ON`, so the INSERT raises `IntegrityError` where
    every caller was written against `KeyError`. Simulated here by deleting
    the row from inside the existence-check hop, which is exactly the state
    the loser of that race observes.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    source_id = db.add_subscription(
        name="Doomed", type="rss", source="https://example.com/feed"
    )
    service = LocalWatchlistsService(db_factory=lambda: db)
    real_get = db.get_subscription

    def delete_after_read(subscription_id):
        row = real_get(subscription_id)
        db.delete_subscription(int(subscription_id))
        return row

    db.get_subscription = delete_after_read

    with pytest.raises(KeyError):
        await service.launch_run(source_id=source_id)


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
async def test_url_html_extraction_runs_off_the_event_loop(monkeypatch):
    """URL HTML parsing must not block the event-loop thread."""
    body = "<html><body><article>Hello <b>watchlist</b></article></body></html>"
    _serve(monkeypatch, body, content_type="text/html")
    real_extract = ContentExtractor.extract_text_from_html
    extraction_threads: list[int] = []

    def recording_extract(html, ignore_selectors=None):
        extraction_threads.append(threading.get_ident())
        return real_extract(html, ignore_selectors)

    monkeypatch.setattr(ContentExtractor, "extract_text_from_html", recording_extract)
    loop_thread = threading.get_ident()

    content = await URLMonitor(SubscriptionsDB(":memory:", "test"))._fetch_url_content(
        {
            "source": "https://example.com/article",
            "extraction_method": "auto",
        }
    )

    assert content["text"] == "Hello watchlist"
    assert extraction_threads, "the real HTML extractor must have run"
    assert all(thread_id != loop_thread for thread_id in extraction_threads), (
        "URL HTML extraction must run under asyncio.to_thread, not inline on "
        "the event loop"
    )


@pytest.mark.asyncio
async def test_changed_item_cpu_work_runs_off_the_event_loop_without_changing_semantics(
    tmp_path, monkeypatch
):
    """Percentage and significant-change details are worker-thread CPU work."""
    before_html = """<html><body>
<p>Alpha sentence.</p>
<p>Shared context.</p>
</body></html>"""
    after_html = """<html><body>
<p>Beta sentence.</p>
<p>Shared context.</p>
<p>Extra details.</p>
</body></html>"""
    previous_text = "Alpha sentence. Shared context."
    current_text = "Beta sentence. Shared context. Extra details."

    real_percentage = ContentExtractor.calculate_change_percentage
    real_segment = monitoring_engine._segment_for_diff
    real_build = monitoring_engine.build_change_diff
    real_added_removed = monitoring_engine.added_and_removed_text
    real_classify = monitoring_engine.classify_change_type
    expected_percentage = real_percentage(previous_text, current_text)
    old_segments = real_segment(previous_text)
    new_segments = real_segment(current_text)
    expected_diff, expected_summary = real_build(
        previous_text,
        current_text,
        old_segments=old_segments,
        new_segments=new_segments,
    )
    expected_added, expected_removed = real_added_removed(
        previous_text,
        current_text,
        old_segments=old_segments,
        new_segments=new_segments,
    )
    expected_type = real_classify(previous_text, current_text)

    calls: dict[str, list[int]] = {
        "percentage": [],
        "segment": [],
        "build": [],
        "added_removed": [],
        "classify": [],
    }

    def record(name, function):
        def wrapper(*args, **kwargs):
            calls[name].append(threading.get_ident())
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        ContentExtractor,
        "calculate_change_percentage",
        staticmethod(record("percentage", real_percentage)),
    )
    monkeypatch.setattr(
        monitoring_engine, "_segment_for_diff", record("segment", real_segment)
    )
    monkeypatch.setattr(
        monitoring_engine, "build_change_diff", record("build", real_build)
    )
    monkeypatch.setattr(
        monitoring_engine,
        "added_and_removed_text",
        record("added_removed", real_added_removed),
    )
    monkeypatch.setattr(
        monitoring_engine, "classify_change_type", record("classify", real_classify)
    )

    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    source_id = db.add_subscription(
        name="Changed page",
        type="url",
        source="https://example.com/page",
        change_threshold=0.0,
    )
    subscription = db.get_subscription(source_id)
    _serve_in_order(monkeypatch, [before_html, after_html])
    monitor = URLMonitor(db)
    loop_thread = threading.get_ident()

    first_item, first_disposition = await monitor.check_url(subscription)
    item, disposition = await monitor.check_url(subscription)

    assert first_item is None
    assert first_disposition["kind"] == "baseline_stored"
    assert item is not None, "the changed page must produce a real item"
    assert disposition["kind"] == "changed"
    assert item["type"] == "url_change"
    assert item["content_kind"] == "change"
    assert item["content_format"] == "diff"
    assert item["change_type"] == expected_type == "content"
    assert item["change_percentage"] == pytest.approx(expected_percentage * 100.0)
    assert item["content"] == expected_diff
    assert item["diff_summary"] == expected_summary
    assert item[monitoring_engine.RULE_MATCH_TEXT_KEY] == current_text
    assert item[monitoring_engine.RULE_MATCH_ADDED_TEXT_KEY] == expected_added
    assert item[monitoring_engine.RULE_MATCH_REMOVED_TEXT_KEY] == expected_removed

    snapshots = db.conn.execute(
        "SELECT content_hash, extracted_content FROM url_snapshots "
        "WHERE subscription_id = ? ORDER BY id",
        (source_id,),
    ).fetchall()
    assert [(row["extracted_content"], row["content_hash"]) for row in snapshots] == [
        (previous_text, ContentExtractor.calculate_content_hash(previous_text)),
        (current_text, ContentExtractor.calculate_content_hash(current_text)),
    ]

    assert len(calls["percentage"]) == 1
    assert calls["percentage"][0] != loop_thread
    # 4 = 2 inside the percentage hop (TASK-16839: the percentage is computed
    # on the same `_segment_for_diff` basis as the diff) + 2 in the details
    # hop, which still segments once per side and shares (segment-once rule).
    assert len(calls["segment"]) == 4
    assert len(calls["build"]) == 1
    assert len(calls["added_removed"]) == 1
    assert len(calls["classify"]) == 1
    for name in ("segment", "build", "added_removed", "classify"):
        assert all(thread_id != loop_thread for thread_id in calls[name]), (
            f"{name} must run inside the grouped worker-thread comparison"
        )


@pytest.mark.asyncio
async def test_below_threshold_cpu_work_stops_before_significant_change_details(
    tmp_path, monkeypatch
):
    """A withheld change offloads its ratio but never builds diff evidence."""
    before_html = """<html><body>
<p>Alpha sentence.</p>
<p>Shared context.</p>
</body></html>"""
    after_html = """<html><body>
<p>Beta sentence.</p>
<p>Shared context.</p>
</body></html>"""
    previous_text = "Alpha sentence. Shared context."
    current_text = "Beta sentence. Shared context."
    real_percentage = ContentExtractor.calculate_change_percentage
    actual_ratio = real_percentage(previous_text, current_text)
    threshold = actual_ratio + 0.1
    assert threshold < 1.0

    percentage_threads: list[int] = []

    def recording_percentage(old_content, new_content):
        percentage_threads.append(threading.get_ident())
        return real_percentage(old_content, new_content)

    grouped_calls: list[int] = []

    def significant_details_must_not_run(*args, **kwargs):
        grouped_calls.append(threading.get_ident())
        pytest.fail("below-threshold changes must not build significant details")

    monkeypatch.setattr(
        ContentExtractor,
        "calculate_change_percentage",
        staticmethod(recording_percentage),
    )
    monkeypatch.setattr(
        monitoring_engine,
        "_build_significant_change_details",
        significant_details_must_not_run,
        raising=False,
    )

    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    source_id = db.add_subscription(
        name="Withheld page",
        type="url",
        source="https://example.com/page",
        change_threshold=threshold,
    )
    subscription = db.get_subscription(source_id)
    _serve_in_order(monkeypatch, [before_html, after_html])
    monitor = URLMonitor(db)
    loop_thread = threading.get_ident()

    first_item, first_disposition = await monitor.check_url(subscription)
    item, disposition = await monitor.check_url(subscription)

    assert first_item is None
    assert first_disposition["kind"] == "baseline_stored"
    assert item is None
    assert disposition["kind"] == "withheld_below_threshold"
    assert disposition["reason"] == "below_change_threshold"
    assert disposition["withheld_percentage"] == pytest.approx(actual_ratio * 100.0)
    assert len(percentage_threads) == 1
    assert percentage_threads[0] != loop_thread
    assert grouped_calls == []
    snapshots = db.conn.execute(
        "SELECT extracted_content FROM url_snapshots "
        "WHERE subscription_id = ? ORDER BY id",
        (source_id,),
    ).fetchall()
    assert [row["extracted_content"] for row in snapshots] == [previous_text]


@pytest.mark.asyncio
async def test_cancelled_change_worker_does_not_resume_check_or_mutate_state(
    tmp_path, monkeypatch
):
    """Cancelling the await abandons a late worker result without failure."""
    before_html = """<html><body>
<p>Alpha sentence.</p>
<p>Shared context.</p>
</body></html>"""
    after_html = """<html><body>
<p>Beta sentence.</p>
<p>Shared context.</p>
</body></html>"""
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    source_id = db.add_subscription(
        name="Cancelled page",
        type="url",
        source="https://example.com/page",
        change_threshold=0.0,
    )
    subscription = db.get_subscription(source_id)
    _serve_in_order(monkeypatch, [before_html, after_html])
    monitor = URLMonitor(db)

    first_item, first_disposition = await monitor.check_url(subscription)
    assert first_item is None
    assert first_disposition["kind"] == "baseline_stored"

    breaker = monitor.circuit_breakers[source_id]
    success_calls: list[int] = []
    real_record_success = breaker.record_success

    def recording_success() -> None:
        success_calls.append(threading.get_ident())
        real_record_success()

    monkeypatch.setattr(breaker, "record_success", recording_success)
    real_details = monitoring_engine._build_significant_change_details
    worker_entered = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()
    worker_threads: list[int] = []

    def blocked_details(previous_text, current_text):
        worker_threads.append(threading.get_ident())
        worker_entered.set()
        try:
            release_worker.wait()
            return real_details(previous_text, current_text)
        finally:
            worker_finished.set()

    monkeypatch.setattr(
        monitoring_engine, "_build_significant_change_details", blocked_details
    )
    loop_thread = threading.get_ident()
    check_task = asyncio.create_task(monitor.check_url(subscription))

    try:
        assert await asyncio.to_thread(worker_entered.wait, 5.0)
        check_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await check_task
    finally:
        release_worker.set()
        assert await asyncio.to_thread(worker_finished.wait, 5.0)

    snapshots = db.conn.execute(
        "SELECT extracted_content FROM url_snapshots "
        "WHERE subscription_id = ? ORDER BY id",
        (source_id,),
    ).fetchall()
    assert len(snapshots) == 1
    assert snapshots[0]["extracted_content"] == "Alpha sentence. Shared context."
    assert breaker.failure_count == 0
    assert success_calls == []
    assert check_task.cancelled()
    assert len(worker_threads) == 1
    assert worker_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_failing_change_worker_propagates_and_records_breaker_failure(
    tmp_path, monkeypatch
):
    """A grouped-worker exception propagates through the existing failure path."""
    before_html = """<html><body>
<p>Alpha sentence.</p>
<p>Shared context.</p>
</body></html>"""
    after_html = """<html><body>
<p>Beta sentence.</p>
<p>Shared context.</p>
</body></html>"""
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    source_id = db.add_subscription(
        name="Failing page",
        type="url",
        source="https://example.com/page",
        change_threshold=0.0,
    )
    subscription = db.get_subscription(source_id)
    _serve_in_order(monkeypatch, [before_html, after_html])
    monitor = URLMonitor(db)

    first_item, first_disposition = await monitor.check_url(subscription)
    assert first_item is None
    assert first_disposition["kind"] == "baseline_stored"

    class SignificantDetailsError(RuntimeError):
        pass

    worker_threads: list[int] = []

    def failing_details(_previous_text, _current_text):
        worker_threads.append(threading.get_ident())
        raise SignificantDetailsError("significant details failed")

    monkeypatch.setattr(
        monitoring_engine, "_build_significant_change_details", failing_details
    )
    loop_thread = threading.get_ident()

    with pytest.raises(SignificantDetailsError, match="significant details failed"):
        await monitor.check_url(subscription)

    snapshots = db.conn.execute(
        "SELECT extracted_content FROM url_snapshots "
        "WHERE subscription_id = ? ORDER BY id",
        (source_id,),
    ).fetchall()
    assert len(snapshots) == 1
    assert snapshots[0]["extracted_content"] == "Alpha sentence. Shared context."
    assert monitor.circuit_breakers[source_id].failure_count == 1
    assert len(worker_threads) == 1
    assert worker_threads[0] != loop_thread


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
