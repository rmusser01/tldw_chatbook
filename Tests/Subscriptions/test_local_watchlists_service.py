import asyncio
import json
import logging
import threading
from inspect import isawaitable
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Notifications import (
    ClientNotificationsDB,
    NotificationDispatchService,
)
from tldw_chatbook.Subscriptions import LocalWatchlistsService, WatchlistScopeService
from tldw_chatbook.Subscriptions import local_watchlists_service, monitoring_engine
from tldw_chatbook.Subscriptions.monitoring_engine import ContentExtractor
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.Subscriptions.watchlist_item_page import (
    WatchlistItemCursor,
    WatchlistItemPage,
)


@pytest.mark.asyncio
async def test_list_reader_items_page_normalizes_rows_and_runs_db_off_loop():
    cursor = WatchlistItemCursor("2026-08-25 12:00:00", 21)
    raw_page = WatchlistItemPage(
        items=(
            {
                "id": 21,
                "subscription_id": 7,
                "title": "Reader post",
                "effective_date": "2026-08-25 12:00:00",
            },
        ),
        has_more=True,
        snapshot_max_item_id=42,
        snapshot_count=6,
        next_cursor=cursor,
    )
    db = Mock()
    db.get_reader_items_page.return_value = raw_page
    service = LocalWatchlistsService(db_factory=lambda: db)

    async def execute(_db, func, *args, **kwargs):
        return func(*args, **kwargs)

    offload = AsyncMock(side_effect=execute)
    with patch(
        "tldw_chatbook.Subscriptions.local_watchlists_service.run_db_off_loop",
        offload,
    ):
        page = await service.list_reader_items_page(
            source_id="7",
            statuses=["new", "reviewed", "ingested"],
            snapshot_max_item_id=42,
            after=cursor,
        )

    assert offload.await_count == 1
    db.get_reader_items_page.assert_called_once_with(
        subscription_id=7,
        status=None,
        limit=50,
        run_id=None,
        watchlist_id=None,
        unassigned_only=False,
        statuses=["new", "reviewed", "ingested"],
        is_flagged=None,
        search=None,
        since=None,
        snapshot_max_item_id=42,
        after=cursor,
    )
    assert page.items[0]["id"] == "local:watchlist_item:21"
    assert page.items[0]["effective_date"] == "2026-08-25 12:00:00"
    assert page.has_more is True
    assert page.snapshot_max_item_id == 42
    assert page.snapshot_count == 6
    assert page.next_cursor is cursor


@pytest.mark.asyncio
async def test_count_reader_item_arrivals_forwards_scope_off_loop():
    db = Mock()
    db.count_reader_item_arrivals.return_value = 3
    service = LocalWatchlistsService(db_factory=lambda: db)

    async def execute(_db, func, *args, **kwargs):
        return func(*args, **kwargs)

    offload = AsyncMock(side_effect=execute)
    with patch(
        "tldw_chatbook.Subscriptions.local_watchlists_service.run_db_off_loop",
        offload,
    ):
        count = await service.count_reader_item_arrivals(
            snapshot_max_item_id=42,
            source_id="7",
            status="new",
            run_id="8",
            watchlist_id="9",
            unassigned_only=True,
            is_flagged=True,
            search="reader",
            since="2026-08-25 00:00:00",
        )

    assert offload.await_count == 1
    db.count_reader_item_arrivals.assert_called_once_with(
        snapshot_max_item_id=42,
        subscription_id=7,
        status="new",
        run_id=8,
        watchlist_id=9,
        unassigned_only=True,
        statuses=None,
        is_flagged=True,
        search="reader",
        since="2026-08-25 00:00:00",
    )
    assert count == 3


def test_local_watchlists_service_publishes_create_form_source_types():
    assert LocalWatchlistsService.CREATE_FORM_SOURCE_TYPES == ("rss", "atom", "url")


@pytest.mark.asyncio
async def test_local_watchlists_service_rejects_invalid_type_before_opening_db():
    db_factory = Mock()
    service = LocalWatchlistsService(db_factory=db_factory)

    with pytest.raises(
        ValueError, match="Unsupported local watchlist source type: playlist"
    ):
        await service.create_source(
            {
                "name": "Playlist",
                "url": "https://example.com/playlist",
                "source_type": "playlist",
            }
        )

    db_factory.assert_not_called()


@pytest.mark.asyncio
async def test_local_watchlists_service_persists_run_queue_state(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
        }
    )

    launched = await service.launch_run(source_id=source["source_id"])
    listed = await service.list_runs()
    fetched = await service.get_run(launched["run_id"])
    detail = await service.get_run_detail(launched["run_id"])
    cancelled = await service.cancel_run(launched["run_id"])

    assert launched["id"].startswith("local:watchlist_run:")
    assert launched["status"] == "queued"
    assert listed[0]["run_id"] == launched["run_id"]
    assert fetched["source_id"] == source["source_id"]
    assert detail["stats"]["source_id"] == source["source_id"]
    assert cancelled["status"] == "cancelled"


@pytest.mark.asyncio
async def test_run_lifecycle_uses_database_owned_claim_transitions(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
        }
    )
    accepted: list[int] = []
    terminal: list[tuple[int, str]] = []
    original_accept = db.accept_watchlist_run
    original_transition = db.transition_watchlist_run

    def accept(source_id: int, *, created_at: str):
        accepted.append(source_id)
        return original_accept(source_id, created_at=created_at)

    def transition(run_id: int, *, status: str, **kwargs):
        terminal.append((run_id, status))
        return original_transition(run_id, status=status, **kwargs)

    db.accept_watchlist_run = accept
    db.transition_watchlist_run = transition

    launched = await service.launch_run(source_id=source["source_id"])
    cancelled = await service.cancel_run(launched["run_id"])

    assert accepted == [source["source_id"]]
    assert terminal == [(launched["run_id"], "cancelled")]
    assert cancelled["status"] == "cancelled"


@pytest.mark.asyncio
async def test_two_scope_services_execute_one_durable_source_claim(tmp_path):
    path = tmp_path / "subscriptions.db"
    first_db = SubscriptionsDB(path, "first")
    second_db = SubscriptionsDB(path, "second")
    executor_started = asyncio.Event()
    release_executor = asyncio.Event()
    loser_accepted = threading.Event()
    executor_calls = 0

    async def executor(_subscription):
        nonlocal executor_calls
        executor_calls += 1
        executor_started.set()
        await release_executor.wait()
        return {"items": []}

    first_service = LocalWatchlistsService(
        db_factory=lambda: first_db, run_executor=executor
    )
    second_service = LocalWatchlistsService(
        db_factory=lambda: second_db, run_executor=executor
    )
    source = await first_service.create_source(
        {
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
        }
    )
    original_accept = second_db.accept_watchlist_run

    def accept_loser(source_id: int, *, created_at: str):
        receipt = original_accept(source_id, created_at=created_at)
        if receipt["_claim_acquired"] is False:
            loser_accepted.set()
        return receipt

    second_db.accept_watchlist_run = accept_loser
    first_scope = WatchlistScopeService(
        local_service=first_service, server_service=None
    )
    second_scope = WatchlistScopeService(
        local_service=second_service, server_service=None
    )

    first = asyncio.create_task(
        first_scope.launch_run(runtime_backend="local", source_id=source["source_id"])
    )
    await asyncio.wait_for(executor_started.wait(), timeout=2)
    second = asyncio.create_task(
        second_scope.launch_run(runtime_backend="local", source_id=source["source_id"])
    )
    assert await asyncio.to_thread(loser_accepted.wait, 2)
    release_executor.set()
    receipts = await asyncio.gather(first, second)

    assert executor_calls == 1
    assert receipts[0]["run_id"] == receipts[1]["run_id"]
    assert [receipt["status"] for receipt in receipts] == ["completed", "completed"]
    durable = await first_service.get_run(receipts[0]["run_id"])
    assert durable["status"] == "completed"


@pytest.mark.asyncio
async def test_wait_for_terminal_run_has_bounded_backoff(monkeypatch):
    service = LocalWatchlistsService(db_factory=Mock())
    service.get_run = AsyncMock(return_value={"run_id": 7, "status": "running"})
    ticks = iter((0.0, 0.0, 0.01, 0.03))
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        local_watchlists_service, "_RUN_CLAIM_WAIT_TIMEOUT_SECONDS", 0.025
    )
    monkeypatch.setattr(
        local_watchlists_service,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    monkeypatch.setattr(
        local_watchlists_service, "asyncio", SimpleNamespace(sleep=sleep)
    )

    with pytest.raises(TimeoutError, match="Timed out waiting for watchlist run 7"):
        await service.wait_for_terminal_run(7)

    assert service.get_run.await_count == 3
    assert sleeps == pytest.approx([0.01, 0.015])


@pytest.mark.asyncio
async def test_wait_for_terminal_run_is_cancellable() -> None:
    service = LocalWatchlistsService(db_factory=Mock())
    polled = asyncio.Event()
    queries = 0

    async def get_run(_run_id: int) -> dict[str, object]:
        nonlocal queries
        queries += 1
        polled.set()
        return {"run_id": 7, "status": "running"}

    service.get_run = get_run
    waiting = asyncio.create_task(service.wait_for_terminal_run(7))
    await asyncio.wait_for(polled.wait(), timeout=1)
    waiting.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert queries == 1


@pytest.mark.asyncio
async def test_local_watchlists_service_exposes_sync_home_run_snapshot(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    queued_source = await service.create_source(
        {
            "name": "Queued Feed",
            "url": "https://example.com/queued.xml",
            "source_type": "rss",
        }
    )
    failed_source = await service.create_source(
        {
            "name": "Failed Feed",
            "url": "https://example.com/failed.xml",
            "source_type": "rss",
        }
    )

    queued = await service.launch_run(source_id=queued_source["source_id"])
    failed = await service.launch_run(source_id=failed_source["source_id"])
    await service.record_run_result(
        failed["run_id"],
        status="failed",
        error_msg="boom",
        dispatch_notifications=False,
    )

    snapshot = service.list_home_run_snapshot(limit=5)

    assert not isawaitable(snapshot)
    assert [run["run_id"] for run in snapshot[:2]] == [
        failed["run_id"],
        queued["run_id"],
    ]
    assert snapshot[0]["id"] == f"local:watchlist_run:{failed['run_id']}"
    assert snapshot[0]["status"] == "failed"
    assert snapshot[0]["source_title"] == "Failed Feed"
    assert snapshot[0]["source_id"] == failed_source["source_id"]
    assert snapshot[1]["status"] == "queued"
    assert snapshot[1]["source_title"] == "Queued Feed"
    # `title` is the key Home's active-work rail reads FIRST (see
    # `HomeActiveWorkAdapter._local_watchlist_run_items`). Only the
    # Home-specific normalizer set it before TASK-2305 folded the three run
    # reads into one, so it is pinned here rather than left to the rail's
    # `source_title` fallback to cover for it.
    assert snapshot[0]["title"] == "Failed Feed"


@pytest.mark.asyncio
async def test_local_watchlists_service_executes_run_and_records_subscription_items(
    tmp_path,
):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")

    async def fake_run_executor(subscription):
        return {
            "items": [
                {
                    "url": "https://example.com/post-1",
                    "title": "Post 1",
                    "content_hash": "hash-1",
                    "published_date": "2026-04-25T00:00:00+00:00",
                }
            ],
            "stats": {"bytes_transferred": 512},
            "log_text": "fetched 1 item",
        }

    service = LocalWatchlistsService(
        db_factory=lambda: db, run_executor=fake_run_executor
    )
    source = await service.create_source(
        {
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    assert completed["status"] == "completed"
    assert completed["started_at"] is not None
    assert completed["finished_at"] is not None
    assert completed["stats"]["items_found"] == 1
    assert completed["stats"]["items_ingested"] == 1
    assert completed["stats"]["bytes_transferred"] == 512
    assert completed["log_text"] == "fetched 1 item"
    assert db.get_subscription(source["source_id"])["last_successful_check"] is not None
    stored_items = db.conn.execute(
        "SELECT url, title, content_hash FROM subscription_items WHERE subscription_id = ?",
        (source["source_id"],),
    ).fetchall()
    assert [dict(row) for row in stored_items] == [
        {
            "url": "https://example.com/post-1",
            "title": "Post 1",
            "content_hash": "hash-1",
        }
    ]


@pytest.mark.asyncio
async def test_local_watchlists_service_persists_alert_rule_crud(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
        }
    )

    created = await service.create_alert_rule(
        name="No items",
        condition_type="no_items",
        condition_value={"threshold": 0},
        job_id=source["source_id"],
        severity="warning",
    )
    listed = await service.list_alert_rules(job_id=source["source_id"])
    fetched = await service.get_alert_rule(created["rule_id"])
    updated = await service.update_alert_rule(
        created["rule_id"], enabled=False, severity="critical"
    )
    deleted = await service.delete_alert_rule(created["rule_id"])

    assert created["id"].startswith("local:watchlist_alert_rule:")
    assert created["condition_value"] == {"threshold": 0}
    assert listed[0]["rule_id"] == created["rule_id"]
    assert fetched["job_id"] == source["source_id"]
    assert updated["enabled"] is False
    assert updated["severity"] == "critical"
    assert deleted["deleted"] is True
    with pytest.raises(KeyError):
        await service.get_alert_rule(created["rule_id"])


@pytest.mark.asyncio
async def test_local_watchlists_service_filters_sources_by_query(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    await service.create_source(
        {
            "name": "Python Weekly",
            "url": "https://example.com/python.xml",
            "source_type": "rss",
        }
    )
    await service.create_source(
        {
            "name": "Cooking Notes",
            "url": "https://example.com/cooking.xml",
            "source_type": "rss",
        }
    )

    results = await service.list_sources(q="python", limit=10, offset=0)

    assert [item["title"] for item in results] == ["Python Weekly"]


@pytest.mark.asyncio
async def test_local_watchlists_service_persists_source_execution_settings(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)

    source = await service.create_source(
        {
            "name": "Docs",
            "source_type": "url_list",
            "extraction_rules": {
                "urls": ["https://example.com/a", "https://example.com/b"]
            },
            "processing_options": {"max_urls": 2},
            "extraction_method": "full",
            "check_frequency": 300,
        }
    )
    updated = await service.update_source(
        source["source_id"],
        {
            "processing_options": {"max_urls": 1},
            "extraction_rules": {"urls": ["https://example.com/c"]},
        },
    )

    assert source["source_type"] == "url_list"
    assert source["url"] == "https://example.com/a"
    assert source["settings"]["extraction_rules"] == {
        "urls": ["https://example.com/a", "https://example.com/b"]
    }
    assert source["settings"]["processing_options"] == {"max_urls": 2}
    assert source["settings"]["extraction_method"] == "full"
    assert source["settings"]["check_frequency"] == 300
    assert updated["url"] == "https://example.com/c"
    assert updated["settings"]["processing_options"] == {"max_urls": 1}
    assert updated["settings"]["extraction_rules"] == {
        "urls": ["https://example.com/c"]
    }


@pytest.mark.asyncio
async def test_url_list_offloads_cpu_work_for_every_url_in_order(tmp_path, monkeypatch):
    urls = ["https://example.com/a", "https://example.com/b"]
    bodies = {
        "a:baseline": "<html><body><p>URL A baseline body.</p></body></html>",
        "b:baseline": "<html><body><p>URL B baseline body.</p></body></html>",
        "a:changed": "<html><body><p>URL A changed body.</p></body></html>",
        "b:changed": "<html><body><p>URL B changed body.</p></body></html>",
    }
    text_markers = {
        "URL A baseline body.": "a:baseline",
        "URL B baseline body.": "b:baseline",
        "URL A changed body.": "a:changed",
        "URL B changed body.": "b:changed",
    }
    phase = "baseline"
    calls: list[tuple[str, str, int]] = []

    async def serve(url, **_kwargs):
        marker = f"{url.rsplit('/', 1)[-1]}:{phase}"
        calls.append(("fetch", marker, threading.get_ident()))
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "text/html"},
            text=bodies[marker],
            final_url=url,
            raise_for_status=lambda: None,
        )

    real_extract = ContentExtractor.extract_text_from_html
    real_percentage = ContentExtractor.calculate_change_percentage
    real_details = monitoring_engine._build_significant_change_details

    def recording_extract(html, ignore_selectors=None):
        marker = next(key for key, body in bodies.items() if body == html)
        calls.append(("extract", marker, threading.get_ident()))
        return real_extract(html, ignore_selectors)

    # Both spies forward the pre-built segment lists `check_url` now passes
    # (TASK-16839 fix round: one segmentation per side, shared across hops).
    def recording_percentage(old_content, new_content, **kwargs):
        marker = f"{text_markers[old_content]}->{text_markers[new_content]}"
        calls.append(("percentage", marker, threading.get_ident()))
        return real_percentage(old_content, new_content, **kwargs)

    def recording_details(previous_text, current_text, **kwargs):
        marker = f"{text_markers[previous_text]}->{text_markers[current_text]}"
        calls.append(("details", marker, threading.get_ident()))
        return real_details(previous_text, current_text, **kwargs)

    monkeypatch.setattr(monitoring_engine, "guarded_fetch_httpx_async", serve)
    monkeypatch.setattr(
        ContentExtractor,
        "extract_text_from_html",
        staticmethod(recording_extract),
    )
    monkeypatch.setattr(
        ContentExtractor,
        "calculate_change_percentage",
        staticmethod(recording_percentage),
    )
    monkeypatch.setattr(
        monitoring_engine,
        "_build_significant_change_details",
        recording_details,
    )

    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    output_orders: list[list[str]] = []
    real_apply_filters = service._apply_filters_and_alerts

    def recording_apply_filters(items, filters, content_alert_rules, run_id):
        output_orders.append([item["url"] for item in items])
        return real_apply_filters(items, filters, content_alert_rules, run_id)

    monkeypatch.setattr(service, "_apply_filters_and_alerts", recording_apply_filters)
    source = await service.create_source(
        {
            "name": "Docs",
            "source_type": "url_list",
            "extraction_rules": {"urls": urls},
            "change_threshold": 0.0,
        }
    )
    loop_thread = threading.get_ident()

    baseline_run = await service.launch_run(source_id=source["source_id"])
    baseline = await service.execute_run(baseline_run["run_id"])
    phase = "changed"
    changed_run = await service.launch_run(source_id=source["source_id"])
    changed = await service.execute_run(changed_run["run_id"])

    expected_call_order = [
        ("fetch", "a:baseline"),
        ("extract", "a:baseline"),
        ("fetch", "b:baseline"),
        ("extract", "b:baseline"),
        ("fetch", "a:changed"),
        ("extract", "a:changed"),
        ("percentage", "a:baseline->a:changed"),
        ("details", "a:baseline->a:changed"),
        ("fetch", "b:changed"),
        ("extract", "b:changed"),
        ("percentage", "b:baseline->b:changed"),
        ("details", "b:baseline->b:changed"),
    ]
    assert [(kind, marker) for kind, marker, _thread in calls] == expected_call_order
    assert all(
        thread == loop_thread for kind, _marker, thread in calls if kind == "fetch"
    )
    assert all(
        thread != loop_thread for kind, _marker, thread in calls if kind != "fetch"
    )

    assert baseline["stats"]["items_found"] == 0
    assert baseline["stats"]["dispositions"]["baseline"] == 2
    assert changed["status"] == "completed"
    assert changed["stats"]["items_found"] == 2
    assert changed["stats"]["items_ingested"] == 2
    assert changed["stats"]["dispositions"] == {
        "changed": 2,
        "unchanged": 0,
        "withheld": 0,
        "baseline": 0,
        "rebaselined": 0,
        "error": 0,
        # task-16838: no URL was skipped by the in-flight guard.
        "skipped": 0,
    }
    assert output_orders == [[], urls]

    snapshots = db.conn.execute(
        "SELECT url, extracted_content FROM url_snapshots "
        "WHERE subscription_id = ? ORDER BY id ASC",
        (source["source_id"],),
    ).fetchall()
    assert [(row["url"], row["extracted_content"]) for row in snapshots] == [
        (urls[0], "URL A baseline body."),
        (urls[1], "URL B baseline body."),
        (urls[0], "URL A changed body."),
        (urls[1], "URL B changed body."),
    ]
    stored_items = db.conn.execute(
        "SELECT url FROM subscription_items WHERE subscription_id = ? ORDER BY id ASC",
        (source["source_id"],),
    ).fetchall()
    assert [row["url"] for row in stored_items] == urls


@pytest.mark.asyncio
async def test_local_watchlists_service_executes_url_list_sources_with_default_url_monitor(
    tmp_path, monkeypatch
):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    seen_urls = []

    class FakeURLMonitor:
        def __init__(self, db):
            self.db = db

        async def check_url(self, subscription):
            # TASK-1362: the real `check_url` returns `(item, disposition)`.
            seen_urls.append(subscription["source"])
            return (
                {
                    "url": subscription["source"],
                    "title": f"Changed {len(seen_urls)}",
                    "content_hash": f"hash-{len(seen_urls)}",
                    "published_date": "2026-04-25T00:00:00+00:00",
                },
                {"kind": "changed", "reason": None, "withheld_percentage": None},
            )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.URLMonitor",
        FakeURLMonitor,
    )
    source = await service.create_source(
        {
            "name": "Docs",
            "source_type": "url_list",
            "extraction_rules": {
                "urls": ["https://example.com/a", "https://example.com/b"],
            },
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    stored_items = db.conn.execute(
        "SELECT url, title, content_hash FROM subscription_items WHERE subscription_id = ? ORDER BY id ASC",
        (source["source_id"],),
    ).fetchall()
    assert completed["status"] == "completed"
    assert completed["stats"]["items_found"] == 2
    assert completed["stats"]["items_ingested"] == 2
    assert completed["stats"]["dispositions"] == {
        "changed": 2,
        "unchanged": 0,
        "withheld": 0,
        "baseline": 0,
        # Split from `baseline` by the whole-branch review's Critical 1: a
        # first check discarded nothing, a settings-change re-baseline threw
        # away a real diff window.
        "rebaselined": 0,
        # task-1394: no URL raised in this run.
        "error": 0,
        # task-16838: no URL was skipped by the in-flight guard.
        "skipped": 0,
    }, "the url_list arm must aggregate one disposition per URL checked"
    assert seen_urls == ["https://example.com/a", "https://example.com/b"]
    assert [dict(row) for row in stored_items] == [
        {
            "url": "https://example.com/a",
            "title": "Changed 1",
            "content_hash": "hash-1",
        },
        {
            "url": "https://example.com/b",
            "title": "Changed 2",
            "content_hash": "hash-2",
        },
    ]


@pytest.mark.asyncio
async def test_local_watchlists_service_executes_sitemap_sources_with_default_url_monitor(
    tmp_path, monkeypatch
):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    fetched_sitemaps = []
    seen_urls = []

    SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page-a</loc></url>
            <url><loc>https://example.com/page-b</loc></url>
        </urlset>
        """

    async def fake_guarded(url, *, client, max_bytes, trusted_origins=frozenset(), headers=None, params=None, auth=None):
        fetched_sitemaps.append(url)
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/xml"},
            text=SITEMAP_XML,
            final_url=url,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.local_watchlists_service.guarded_fetch_httpx_async",
        fake_guarded,
    )

    class FakeURLMonitor:
        def __init__(self, db):
            self.db = db

        async def check_url(self, subscription):
            # TASK-1362: the real `check_url` returns `(item, disposition)`.
            seen_urls.append(subscription["source"])
            return (
                {
                    "url": subscription["source"],
                    "title": f"Sitemap page {len(seen_urls)}",
                    "content_hash": f"sitemap-hash-{len(seen_urls)}",
                    "published_date": "2026-04-25T00:00:00+00:00",
                },
                {"kind": "changed", "reason": None, "withheld_percentage": None},
            )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.URLMonitor",
        FakeURLMonitor,
    )
    source = await service.create_source(
        {
            "name": "Docs sitemap",
            "url": "https://example.com/sitemap.xml",
            "source_type": "sitemap",
            "processing_options": {"max_urls": 2},
        }
    )
    assert source["source_type"] == "sitemap"
    assert "sitemap" not in LocalWatchlistsService.CREATE_FORM_SOURCE_TYPES
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    stored_items = db.conn.execute(
        "SELECT url, title, content_hash FROM subscription_items WHERE subscription_id = ? ORDER BY id ASC",
        (source["source_id"],),
    ).fetchall()
    assert fetched_sitemaps == ["https://example.com/sitemap.xml"]
    assert seen_urls == ["https://example.com/page-a", "https://example.com/page-b"]
    assert completed["status"] == "completed"
    assert completed["stats"]["items_found"] == 2
    assert completed["stats"]["dispositions"] == {
        "changed": 2,
        "unchanged": 0,
        "withheld": 0,
        "baseline": 0,
        "rebaselined": 0,
        # task-1394: no URL raised in this run.
        "error": 0,
        # task-16838: no URL was skipped by the in-flight guard.
        "skipped": 0,
    }, "the sitemap arm must aggregate one disposition per URL checked"
    assert [dict(row) for row in stored_items] == [
        {
            "url": "https://example.com/page-a",
            "title": "Sitemap page 1",
            "content_hash": "sitemap-hash-1",
        },
        {
            "url": "https://example.com/page-b",
            "title": "Sitemap page 2",
            "content_hash": "sitemap-hash-2",
        },
    ]


@pytest.mark.asyncio
async def test_local_watchlists_service_url_list_isolates_one_failing_url(
    tmp_path, monkeypatch
):
    """task-1394 AC#1/#3: one bad URL must not sink the whole `url_list` run.

    Before per-URL isolation, `check_url` raising for ANY url in the loop
    propagated straight out of `_default_run_executor` uncaught; `execute_run`
    then caught it at the top level and called `record_run_failure`, which
    discards the items the OTHER, successful URLs in this same run already
    collected. A 50-URL source with one dead link used to yield nothing at
    all. This is the discriminator test: it reds under that old
    all-or-nothing behaviour (confirmed by reverting the
    `_check_url_isolated` try/except and re-running -- see the task's
    Implementation Notes) and passes with the per-URL isolation in place.

    Args:
        tmp_path: pytest tmp dir for the on-disk `SubscriptionsDB`.
        monkeypatch: patches the URL monitor so one URL raises.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    seen_urls = []

    class FakeURLMonitor:
        def __init__(self, db):
            self.db = db

        async def check_url(self, subscription):
            url = subscription["source"]
            seen_urls.append(url)
            if url == "https://example.com/b":
                # Deliberately NOT a subclass of any of the others -- proves
                # the isolation catches an arbitrary exception, not just one
                # anticipated type.
                raise TimeoutError("connect timed out")
            return (
                {
                    "url": url,
                    "title": f"Changed {len(seen_urls)}",
                    "content_hash": f"hash-{len(seen_urls)}",
                    "published_date": "2026-04-25T00:00:00+00:00",
                },
                {"kind": "changed", "reason": None, "withheld_percentage": None},
            )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.URLMonitor",
        FakeURLMonitor,
    )
    source = await service.create_source(
        {
            "name": "Docs",
            "source_type": "url_list",
            "extraction_rules": {
                "urls": [
                    "https://example.com/a",
                    "https://example.com/b",
                    "https://example.com/c",
                ],
            },
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    stored_items = db.conn.execute(
        "SELECT url, title, content_hash FROM subscription_items WHERE subscription_id = ? ORDER BY id ASC",
        (source["source_id"],),
    ).fetchall()
    # The run completed -- it did NOT fail via `record_run_failure` merely
    # because one of its three URLs raised.
    assert completed["status"] == "completed"
    # And the loop kept going past the poisoned URL to check the one after it.
    assert seen_urls == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]
    # The two URLs that succeeded persisted their items...
    assert [dict(row) for row in stored_items] == [
        {
            "url": "https://example.com/a",
            "title": "Changed 1",
            "content_hash": "hash-1",
        },
        {
            "url": "https://example.com/c",
            "title": "Changed 3",
            "content_hash": "hash-3",
        },
    ]
    # ...and the run's dispositions say exactly one URL errored, rather than
    # reporting a clean run that simply found less than it should have.
    assert completed["stats"]["dispositions"] == {
        "changed": 2,
        "unchanged": 0,
        "withheld": 0,
        "baseline": 0,
        "rebaselined": 0,
        "error": 1,
        # task-16838: no URL was skipped by the in-flight guard.
        "skipped": 0,
    }


@pytest.mark.asyncio
async def test_local_watchlists_service_sitemap_isolates_one_failing_url(
    tmp_path, monkeypatch
):
    """task-1394: the sitemap arm gets the same per-URL isolation as url_list.

    The sitemap FETCH that produces this URL list (`_urls_for_sitemap`) is a
    separate concern that this task deliberately leaves alone: it runs once,
    before this loop starts, so a failure fetching the sitemap itself still
    fails the whole run. This test is only about the per-URL loop that walks
    the URLs the sitemap already produced.

    Args:
        tmp_path: pytest tmp dir for the on-disk `SubscriptionsDB`.
        monkeypatch: patches the URL monitor so one URL raises.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    seen_urls = []

    SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page-a</loc></url>
            <url><loc>https://example.com/page-b</loc></url>
            <url><loc>https://example.com/page-c</loc></url>
        </urlset>
        """

    async def fake_guarded(url, *, client, max_bytes, trusted_origins=frozenset(), headers=None, params=None, auth=None):
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/xml"},
            text=SITEMAP_XML,
            final_url=url,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.local_watchlists_service.guarded_fetch_httpx_async",
        fake_guarded,
    )

    class FakeURLMonitor:
        def __init__(self, db):
            self.db = db

        async def check_url(self, subscription):
            url = subscription["source"]
            seen_urls.append(url)
            if url == "https://example.com/page-b":
                raise ConnectionError("connection refused")
            return (
                {
                    "url": url,
                    "title": f"Sitemap page {len(seen_urls)}",
                    "content_hash": f"sitemap-hash-{len(seen_urls)}",
                    "published_date": "2026-04-25T00:00:00+00:00",
                },
                {"kind": "changed", "reason": None, "withheld_percentage": None},
            )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.URLMonitor",
        FakeURLMonitor,
    )
    source = await service.create_source(
        {
            "name": "Docs sitemap",
            "url": "https://example.com/sitemap.xml",
            "source_type": "sitemap",
            "processing_options": {"max_urls": 3},
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    stored_items = db.conn.execute(
        "SELECT url, title, content_hash FROM subscription_items WHERE subscription_id = ? ORDER BY id ASC",
        (source["source_id"],),
    ).fetchall()
    assert completed["status"] == "completed"
    assert seen_urls == [
        "https://example.com/page-a",
        "https://example.com/page-b",
        "https://example.com/page-c",
    ]
    assert [dict(row) for row in stored_items] == [
        {
            "url": "https://example.com/page-a",
            "title": "Sitemap page 1",
            "content_hash": "sitemap-hash-1",
        },
        {
            "url": "https://example.com/page-c",
            "title": "Sitemap page 3",
            "content_hash": "sitemap-hash-3",
        },
    ]
    assert completed["stats"]["dispositions"] == {
        "changed": 2,
        "unchanged": 0,
        "withheld": 0,
        "baseline": 0,
        "rebaselined": 0,
        "error": 1,
        # task-16838: no URL was skipped by the in-flight guard.
        "skipped": 0,
    }


@pytest.mark.asyncio
async def test_local_watchlists_service_url_list_all_error_advances_breaker_and_pauses(
    tmp_path, monkeypatch
):
    """Fix wave, task-1394 whole-branch review Finding #1 (MAJOR).

    The per-URL isolation above (`_check_url_isolated`) correctly turns one
    dead URL among many into a single `"error"` disposition instead of failing
    the whole run -- but `execute_run`'s success path used to call
    `db.record_check_result(source_id, items=None, stats=stats)` with
    `error=None` UNCONDITIONALLY, even when every single URL in the run
    errored. That call's success branch
    (`DB/Subscriptions_DB.py:1504-1517`) resets `consecutive_failures` and
    `error_count` to 0 on a run that found nothing and whose every check
    failed -- so a permanently dead `url_list` source could never reach
    `auto_pause_threshold` and auto-pause; its failure streak was wiped every
    single run instead of accumulating.

    Discriminator: this test REDs if `_all_error_check_message` is reverted to
    always return `None` (the pre-fix-wave behaviour) -- the breaker resets to
    0 instead of advancing to 2, and the source never pauses.

    Args:
        tmp_path: pytest tmp dir for the on-disk `SubscriptionsDB`.
        monkeypatch: patches the URL monitor so one URL raises.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)

    class FakeURLMonitor:
        def __init__(self, db):
            self.db = db

        async def check_url(self, subscription):
            raise TimeoutError("connect timed out")

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.URLMonitor",
        FakeURLMonitor,
    )
    source = await service.create_source(
        {
            "name": "Docs",
            "source_type": "url_list",
            "auto_pause_threshold": 2,
            "extraction_rules": {
                "urls": [
                    "https://example.com/a",
                    "https://example.com/b",
                ],
            },
        }
    )
    source_id = source["source_id"]
    # One short of the (lowered, for this test) threshold -- the state a
    # permanently-broken source would already be in from prior all-dead runs.
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET consecutive_failures = ?, error_count = ? WHERE id = ?",
            (1, 1, source_id),
        )
    launched = await service.launch_run(source_id=source_id)

    completed = await service.execute_run(launched["run_id"])

    # Honest run status: every URL failed, so the run itself failed -- not a
    # clean "completed" with zero items.
    assert completed["status"] == "failed"
    # AC#2's error-count visibility survives the fix: the dispositions are
    # still recorded even though the breaker also advanced.
    assert completed["stats"]["dispositions"] == {
        "changed": 0,
        "unchanged": 0,
        "withheld": 0,
        "baseline": 0,
        "rebaselined": 0,
        "error": 2,
        # task-16838: no URL was skipped by the in-flight guard.
        "skipped": 0,
    }
    row = db.get_subscription(source_id)
    assert row["consecutive_failures"] == 2, (
        "the breaker must ADVANCE on an all-error run, not reset"
    )
    assert row["error_count"] == 2
    assert row["is_paused"] == 1, "threshold reached -- the source must auto-pause"
    assert row["last_error"], "last_error must be set on an all-error run, not cleared"


@pytest.mark.asyncio
async def test_local_watchlists_service_url_list_partial_error_still_resets_breaker(
    tmp_path, monkeypatch
):
    """Fix wave, task-1394 review Finding #1: a PARTIAL run must not over-correct.

    Companion to the all-error test above. One URL succeeds, one errors --
    a working, reachable source should still be treated as healthy: the
    breaker resets to 0 (exactly as a clean run would) and the successful
    URL's item persists. This pins that the all-error fix does not regress
    into treating ANY per-URL error as a subscription-level failure.

    Args:
        tmp_path: pytest tmp dir for the on-disk `SubscriptionsDB`.
        monkeypatch: patches the URL monitor so one URL raises.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)

    class FakeURLMonitor:
        def __init__(self, db):
            self.db = db

        async def check_url(self, subscription):
            url = subscription["source"]
            if url == "https://example.com/b":
                raise TimeoutError("connect timed out")
            return (
                {
                    "url": url,
                    "title": "Changed",
                    "content_hash": "hash-a",
                    "published_date": "2026-04-25T00:00:00+00:00",
                },
                {"kind": "changed", "reason": None, "withheld_percentage": None},
            )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.URLMonitor",
        FakeURLMonitor,
    )
    source = await service.create_source(
        {
            "name": "Docs",
            "source_type": "url_list",
            "extraction_rules": {
                "urls": [
                    "https://example.com/a",
                    "https://example.com/b",
                ],
            },
        }
    )
    source_id = source["source_id"]
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET consecutive_failures = ?, error_count = ? WHERE id = ?",
            (5, 5, source_id),
        )
    launched = await service.launch_run(source_id=source_id)

    completed = await service.execute_run(launched["run_id"])

    assert completed["status"] == "completed", (
        "at least one URL succeeded -- the run stays completed, not failed"
    )
    assert completed["stats"]["dispositions"]["error"] == 1
    assert completed["stats"]["dispositions"]["changed"] == 1
    stored_items = db.conn.execute(
        "SELECT url FROM subscription_items WHERE subscription_id = ?",
        (source_id,),
    ).fetchall()
    assert [row["url"] for row in stored_items] == ["https://example.com/a"]

    row = db.get_subscription(source_id)
    assert row["consecutive_failures"] == 0, (
        "a working source is healthy -- the breaker resets on ANY successful check"
    )
    assert row["error_count"] == 0
    assert row["last_error"] is None
    assert row["is_paused"] == 0


@pytest.fixture
def _loguru_to_caplog():
    """Bridge loguru output into pytest's ``caplog`` for the tests below.

    loguru does not propagate to stdlib ``logging`` (and therefore not to
    ``caplog``) without an explicit bridge -- the same pattern used in
    ``Tests/Model_Artifacts/test_credentials_and_boundaries.py``. Scoped to
    an explicit, non-autouse fixture so it only applies to the tests that
    request it.
    """
    from loguru import logger as loguru_logger

    class PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    handler_id = loguru_logger.add(PropagateHandler(), format="{message}")
    yield
    loguru_logger.remove(handler_id)


@pytest.mark.asyncio
async def test_local_watchlists_service_record_run_failure_auto_pauses_at_threshold(
    tmp_path, caplog, _loguru_to_caplog
):
    """task-1410 AC#2.

    The MAIN failure path -- ``LocalWatchlistsService.record_run_failure``
    calling ``SubscriptionsDB.record_check_error`` -- previously bumped
    ``consecutive_failures`` but never consulted ``auto_pause_threshold`` at
    all, so a source that failed forever would never auto-pause. This drives
    the fix through the REAL path (``execute_run`` raising ->
    ``record_run_failure`` -> ``record_check_error``), not by calling
    ``SubscriptionsDB.record_check_error`` directly -- the AC explicitly
    forbids that shortcut, since bypassing the service is exactly how this
    path stayed unreachable in the first place.

    Reds if the threshold check inside the shared
    ``_advance_failure_and_maybe_pause`` helper is removed or bypassed:
    ``is_paused`` stays 0 after 3 failures and no WARNING is logged.

    Args:
        tmp_path: pytest temp dir for the on-disk `SubscriptionsDB`.
        caplog: pytest log capture, asserts the auto-pause WARNING.
        _loguru_to_caplog: routes loguru into `caplog` for the assertion.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")

    async def always_fails(subscription):
        raise TimeoutError("connect timed out")

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=always_fails)
    source = await service.create_source(
        {
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
            "auto_pause_threshold": 3,
        }
    )
    source_id = source["source_id"]

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            launched = await service.launch_run(source_id=source_id)
            completed = await service.execute_run(launched["run_id"])
            assert completed["status"] == "failed"

    row = db.get_subscription(source_id)
    assert row["consecutive_failures"] == 3
    assert row["is_paused"] == 1, "threshold reached via the real failure path -- must auto-pause"

    auto_pause_warnings = [
        record for record in caplog.records if "Auto-paused subscription" in record.message
    ]
    assert len(auto_pause_warnings) == 1, (
        "exactly one auto-pause WARNING must fire (on the 3rd failure only), got "
        f"{[r.message for r in auto_pause_warnings]}"
    )


@pytest.mark.asyncio
async def test_local_watchlists_service_both_failure_paths_pause_at_the_same_threshold(
    tmp_path,
):
    """task-1410 consistency guard.

    ``record_check_result``'s error branch (reached for an all-error
    ``url_list``/``sitemap`` run, task-1394) and ``record_check_error`` (the
    main failure path, reached via ``record_run_failure``) now share
    ``_advance_failure_and_maybe_pause``. This proves they cannot diverge:
    an all-error ``url_list`` run AND a plain single-URL failure, each with
    the same ``auto_pause_threshold``, both end paused after the same
    number of consecutive failures.

    Args:
        tmp_path: pytest temp dir for the on-disk `SubscriptionsDB`.
    """
    threshold = 2
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")

    class FakeURLMonitor:
        def __init__(self, db):
            self.db = db

        async def check_url(self, subscription):
            raise TimeoutError("connect timed out")

    async def always_fails(subscription):
        raise TimeoutError("connect timed out")

    # Path 1: record_check_result's error branch, via an all-error url_list run.
    service_a = LocalWatchlistsService(db_factory=lambda: db)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "tldw_chatbook.Subscriptions.monitoring_engine.URLMonitor",
            FakeURLMonitor,
        )
        source_a = await service_a.create_source(
            {
                "name": "Docs",
                "source_type": "url_list",
                "auto_pause_threshold": threshold,
                "extraction_rules": {"urls": ["https://example.com/a"]},
            }
        )
        source_a_id = source_a["source_id"]
        for _ in range(threshold):
            launched = await service_a.launch_run(source_id=source_a_id)
            await service_a.execute_run(launched["run_id"])

    # Path 2: record_check_error, via a plain source whose executor always raises.
    service_b = LocalWatchlistsService(db_factory=lambda: db, run_executor=always_fails)
    source_b = await service_b.create_source(
        {
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
            "auto_pause_threshold": threshold,
        }
    )
    source_b_id = source_b["source_id"]
    for _ in range(threshold):
        launched = await service_b.launch_run(source_id=source_b_id)
        await service_b.execute_run(launched["run_id"])

    row_a = db.get_subscription(source_a_id)
    row_b = db.get_subscription(source_b_id)
    assert row_a["consecutive_failures"] == threshold
    assert row_b["consecutive_failures"] == threshold
    assert row_a["is_paused"] == 1, "record_check_result's error branch must pause at threshold"
    assert row_b["is_paused"] == 1, "record_check_error must pause at threshold"


@pytest.mark.asyncio
async def test_local_watchlists_service_successful_manual_recheck_resumes_a_paused_source(
    tmp_path,
):
    """Fix wave for the task-1410 review, Finding #1 (the important one).

    Once a source auto-pauses, the scheduler never re-checks it:
    `get_pending_checks` excludes `is_paused = 1` rows and
    `WatchlistCheckHandler` skips them too. But `launch_run`/`execute_run`
    have no paused guard at all -- a MANUAL re-check of a paused source
    still runs. That is meant to be the source's only recourse, but until
    this fix wave nothing on the success side ever cleared `is_paused`
    (the auto-pause helper only ever writes `is_paused = 1`;
    `reset_subscription_errors` has zero callers app-wide) -- so even a
    manual re-check that fully succeeded left the source stranded, paused
    forever.

    Drives the real path end to end: an already-paused source, a manual
    `launch_run` + `execute_run` whose executor succeeds, and confirms
    `is_paused` clears with the failure counters reset -- exactly as an
    ordinary successful check resets them.

    Reds if `record_check_result`'s success branch drops its new
    `is_paused = 0` write.

    Args:
        tmp_path: pytest temp dir for the on-disk `SubscriptionsDB`.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")

    async def always_succeeds(subscription):
        return []

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=always_succeeds)
    source = await service.create_source(
        {
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
            "auto_pause_threshold": 3,
        }
    )
    source_id = source["source_id"]
    with db.transaction() as conn:
        conn.execute(
            """
            UPDATE subscriptions
            SET is_paused = 1, error_count = 3, consecutive_failures = 3,
                last_error = 'connection refused'
            WHERE id = ?
            """,
            (source_id,),
        )

    launched = await service.launch_run(source_id=source_id)
    completed = await service.execute_run(launched["run_id"])

    assert completed["status"] == "completed"
    row = db.get_subscription(source_id)
    assert row["is_paused"] == 0, "a successful manual re-check must resume a paused source"
    assert row["consecutive_failures"] == 0
    assert row["error_count"] == 0
    assert row["last_error"] is None


@pytest.mark.asyncio
async def test_local_watchlists_service_executes_api_sources_with_json_field_mapping(
    tmp_path, monkeypatch
):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    requests = []

    API_PAYLOAD = {
        "payload": {
            "entries": [
                {
                    "headline": "Alpha update",
                    "link": "https://api.example.com/a",
                    "summary": "First item",
                    "published": "2026-04-25T00:00:00+00:00",
                },
                {
                    "headline": "Beta update",
                    "link": "https://api.example.com/b",
                    "summary": "Second item",
                    "published": "2026-04-25T01:00:00+00:00",
                },
            ]
        }
    }

    async def fake_guarded(url, *, client, max_bytes, trusted_origins=frozenset(), headers=None, params=None, auth=None):
        requests.append({"url": url, "headers": headers, "params": params})
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            final_url=url,
            raise_for_status=lambda: None,
            json=lambda: API_PAYLOAD,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.local_watchlists_service.guarded_fetch_httpx_async",
        fake_guarded,
    )
    source = await service.create_source(
        {
            "name": "API changelog",
            "url": "https://api.example.com/changes",
            "source_type": "api",
            "custom_headers": {"X-API-Key": "secret"},
            "extraction_rules": {
                "items_path": "payload.entries",
                "field_map": {
                    "title": "headline",
                    "url": "link",
                    "content": "summary",
                    "published_date": "published",
                },
            },
            "processing_options": {"max_items": 1},
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    stored_items = db.conn.execute(
        "SELECT url, title, content_hash FROM subscription_items WHERE subscription_id = ? ORDER BY id ASC",
        (source["source_id"],),
    ).fetchall()
    assert requests == [
        {
            "url": "https://api.example.com/changes",
            "headers": {
                "Accept": "application/json",
                "User-Agent": "tldw-chatbook/1.0 (+https://github.com/tldw/chatbook)",
                "X-API-Key": "secret",
            },
            "params": None,
        }
    ]
    assert completed["status"] == "completed"
    assert completed["stats"]["items_found"] == 1
    assert [dict(row) for row in stored_items] == [
        {
            "url": "https://api.example.com/a",
            "title": "Alpha update",
            "content_hash": "0592ea3b5b28611c52b3b7cbb5382cfbe977f978f3239984bb5f5a6425c55794",
        }
    ]


@pytest.mark.asyncio
async def test_local_watchlists_service_evaluates_completed_run_alerts_into_notifications(
    tmp_path,
):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    notification_store = ClientNotificationsDB(tmp_path / "notifications.db")
    dispatcher = NotificationDispatchService(store=notification_store)
    service = LocalWatchlistsService(
        db_factory=lambda: db, notification_dispatcher=dispatcher
    )
    source = await service.create_source(
        {
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])
    failed_rule = await service.create_alert_rule(
        name="Run failed",
        condition_type="run_failed",
        job_id=source["source_id"],
        severity="critical",
    )
    await service.create_alert_rule(
        name="Bad threshold",
        condition_type="items_above",
        condition_value={"threshold": "abc"},
        job_id=source["source_id"],
    )

    completed = await service.record_run_result(
        launched["run_id"],
        status="failed",
        stats={"items_found": 4, "items_ingested": 1},
        error_msg="boom",
    )

    notifications = notification_store.list_notifications(limit=10)
    assert completed["status"] == "failed"
    assert completed["error_msg"] == "boom"
    assert len(completed["triggered_alerts"]) == 1
    assert completed["triggered_alerts"][0]["rule_id"] == failed_rule["rule_id"]
    assert len(notifications) == 1
    assert notifications[0]["category"] == "watchlists"
    assert notifications[0]["severity"] == "critical"
    assert notifications[0]["source_backend"] == "local"
    assert notifications[0]["source_entity_kind"] == "watchlist_run"
    assert notifications[0]["source_entity_id"] == str(launched["run_id"])
    assert notifications[0]["payload"]["dedupe_key"] == (
        f"watchlist-alert:{failed_rule['rule_id']}:{launched['run_id']}"
    )


@pytest.mark.asyncio
async def test_create_source_honors_inactive(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    result = await service.create_source(
        {"name": "Inactive", "source_type": "rss", "url": "http://example.com/feed", "active": False}
    )
    assert result["active"] is False


@pytest.mark.asyncio
async def test_create_source_persists_check_frequency(tmp_path):
    """TASK-1210: the cadence chosen on the create form has to reach the column.

    ``WatchlistProjection`` computes ``next_run_at`` from ``check_frequency``, so a
    source that does not carry one is never queued and never checked.

    Args:
        tmp_path: pytest temp dir for the on-disk `SubscriptionsDB`.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    result = await service.create_source(
        {
            "name": "Daily",
            "source_type": "rss",
            "url": "http://example.com/feed",
            "check_frequency": 86_400,
        }
    )
    stored = db.get_subscription(int(str(result["id"]).rsplit(":", 1)[-1]))
    assert stored["check_frequency"] == 86_400


@pytest.mark.asyncio
async def test_execute_run_persists_items_and_evaluates_filters(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")

    async def fake_run_executor(subscription):
        return {
            "items": [
                {"url": "https://example.com/ai-post", "title": "AI news", "content_hash": "hash-ai"},
                {"url": "https://example.com/cooking-post", "title": "Cooking tips", "content_hash": "hash-cooking"},
            ],
            "stats": {},
        }

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=fake_run_executor)
    source = await service.create_source(
        {"name": "Feed", "url": "https://example.com/feed.xml", "source_type": "rss"}
    )
    # Add an exclude filter for "AI".
    db.add_filter(
        name="exclude ai",
        conditions={"type": "keyword", "pattern": "AI"},
        action="exclude",
        subscription_id=source["source_id"],
    )

    launched = await service.launch_run(source_id=source["source_id"])
    completed = await service.execute_run(launched["run_id"])

    assert completed["status"] == "completed"
    assert completed["stats"]["items_found"] == 2
    assert completed["stats"]["items_ingested"] == 1
    stored = db.conn.execute(
        "SELECT url FROM subscription_items WHERE subscription_id = ?",
        (source["source_id"],),
    ).fetchall()
    assert [row["url"] for row in stored] == ["https://example.com/cooking-post"]


@pytest.mark.asyncio
async def test_execute_run_stores_content_alert_matches(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")

    async def fake_run_executor(subscription):
        return {
            "items": [
                {"url": "https://example.com/ai-post", "title": "AI news", "content_hash": "hash-ai"},
            ],
            "stats": {},
        }

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=fake_run_executor)
    source = await service.create_source(
        {"name": "Feed", "url": "https://example.com/feed.xml", "source_type": "rss"}
    )
    db.add_filter(
        name="AI alert",
        conditions={"type": "keyword", "pattern": "AI"},
        action="notify",
        action_params={"severity": "warning"},
        subscription_id=source["source_id"],
    )

    launched = await service.launch_run(source_id=source["source_id"])
    completed = await service.execute_run(launched["run_id"])

    assert completed["stats"]["items_ingested"] == 1
    row = db.conn.execute(
        "SELECT alert_matches FROM subscription_items WHERE subscription_id = ?",
        (source["source_id"],),
    ).fetchone()
    assert row["alert_matches"] is not None
    matches = json.loads(row["alert_matches"])
    assert len(matches) == 1
    assert matches[0]["rule_name"] == "AI alert"


@pytest.mark.asyncio
async def test_get_item_status_reads_one_row_and_refuses_a_missing_one(tmp_path):
    """PR #1091 review, F1: the new authoritative single-item status read.

    The reader's `Mark unread` guard used to infer an item's status from a
    paged `list_items` call per candidate status, so an item past the page
    depth looked exactly like an item that did not hold the status at all,
    and the guard let a destructive write through. This method exists so the
    guard reads the item's own row instead.

    A missing row raises rather than returning a falsy status: the guard's
    caller treats an exception as a refusal, and "the item is gone" is an
    unanswered question, not permission to overwrite.

    Args:
        tmp_path: pytest temp dir for the on-disk `SubscriptionsDB`.
    """
    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item

    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source_id = db.add_subscription(
        name="Feed", type="rss", source="https://example.com/feed.xml"
    )
    with db.transaction() as conn:
        item_id = persist_subscription_item(
            conn,
            source_id,
            {
                "url": "https://example.com/one/",
                "title": "One",
                "content_hash": "hash-status-read",
            },
            run_id=None,
            now="2026-07-29T09:00:00+00:00",
        )

    assert await service.get_item_status(item_id) == "new"
    await service.update_item(item_id=item_id, status="ingested")
    assert await service.get_item_status(item_id) == "ingested"
    # Namespaced ids are the screen's currency; the scope service strips the
    # namespace, so the service itself takes the bare id -- and rejects a
    # value it cannot read as one rather than guessing.
    with pytest.raises(ValueError):
        await service.get_item_status("local:watchlist_item:1")
    with pytest.raises(KeyError):
        await service.get_item_status(item_id + 10_000)


@pytest.mark.asyncio
async def test_list_items_can_be_scoped_to_one_run_with_alert_counts(tmp_path):
    """TASK-2306: the Runs tab's Items sub-region asks for ONE run's items.

    `subscription_items.run_id` (and its index) have existed since the column
    was added, and nothing had ever queried them -- so the only item read the
    product offered was "every item of this source", which cannot answer "what
    did this run find". The `alert_count` in the same result is the Alerts
    column's only possible source; without it that column rendered `0` over
    every item however many content-alert rules had fired.
    """
    executed: list[str] = []

    async def fake_run_executor(subscription):
        # One new item per run: run 2 must not be able to inherit run 1's.
        index = len(executed)
        executed.append(f"run-{index}")
        return {
            "items": [
                {
                    "url": f"https://example.com/ai-post-{index}",
                    "title": f"AI news {index}",
                    "content_hash": f"hash-{index}",
                }
            ],
            "stats": {},
        }

    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(
        db_factory=lambda: db, run_executor=fake_run_executor
    )
    source = await service.create_source(
        {"name": "Feed", "url": "https://example.com/feed.xml", "source_type": "rss"}
    )
    db.add_filter(
        name="AI alert",
        conditions={"type": "keyword", "pattern": "AI"},
        action="notify",
        action_params={"severity": "warning"},
        subscription_id=source["source_id"],
    )

    first = await service.launch_run(source_id=source["source_id"])
    await service.execute_run(first["run_id"])
    second = await service.launch_run(source_id=source["source_id"])
    await service.execute_run(second["run_id"])

    first_items = await service.list_items(run_id=first["run_id"], status=None)
    second_items = await service.list_items(run_id=second["run_id"], status=None)
    every_item = await service.list_items(status=None)

    assert [item["title"] for item in first_items] == ["AI news 0"]
    assert [item["title"] for item in second_items] == ["AI news 1"]
    assert len(every_item) == 2, "an unfiltered read must still see both runs"
    assert first_items[0]["run_id"] == first["run_id"]
    assert first_items[0]["alert_count"] == 1, (
        "the item matched one content-alert rule, so the Alerts column has a 1 "
        "to show"
    )


@pytest.mark.asyncio
async def test_list_items_reports_zero_alerts_for_an_unmatched_item(tmp_path):
    """The discriminating half of `alert_count`: no rules matched means 0."""

    async def fake_run_executor(subscription):
        return {
            "items": [
                {
                    "url": "https://example.com/quiet",
                    "title": "Quiet post",
                    "content_hash": "hash-quiet",
                }
            ],
            "stats": {},
        }

    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(
        db_factory=lambda: db, run_executor=fake_run_executor
    )
    source = await service.create_source(
        {"name": "Feed", "url": "https://example.com/feed.xml", "source_type": "rss"}
    )
    launched = await service.launch_run(source_id=source["source_id"])
    await service.execute_run(launched["run_id"])

    items = await service.list_items(run_id=launched["run_id"], status=None)

    assert len(items) == 1
    assert items[0]["alert_count"] == 0


@pytest.mark.asyncio
async def test_a_real_check_produces_a_run_that_names_its_source_and_counts(
    tmp_path,
):
    """TASK-2305 AC#1/AC#2/AC#3, against a stub feed and the real pipeline.

    UAT: a check that demonstrably harvested ~30 items produced a Runs row
    reading `Untitled · completed · Found 0 · Processed 0 · Filtered 0 ·
    Errors 0 · Duration -`. Both halves are asserted here on the record the
    Runs pane actually reads -- `list_runs`' output -- not on the nested
    `stats` blob that was never the problem.
    """

    async def fake_run_executor(subscription):
        return {
            "items": [
                {
                    "url": f"https://example.com/post-{index}",
                    "title": f"Post {index}",
                    "content_hash": f"hash-{index}",
                }
                for index in range(30)
            ],
            "stats": {},
        }

    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(
        db_factory=lambda: db, run_executor=fake_run_executor
    )
    source = await service.create_source(
        {
            "name": "Hacker News",
            "url": "https://hnrss.org/frontpage",
            "source_type": "rss",
        }
    )
    # One filter excluding a single post, so Processed and Filtered are
    # distinguishable from Found and from each other.
    db.add_filter(
        name="exclude one",
        conditions={"type": "keyword", "pattern": "Post 7"},
        action="exclude",
        subscription_id=source["source_id"],
    )
    bundles = WatchlistBundleService(db)
    watchlist = bundles.create("Morning read")
    bundles.add_source(watchlist["id"], source["source_id"])

    launched = await service.launch_run(source_id=source["source_id"])
    await service.execute_run(launched["run_id"])

    listed = await service.list_runs()
    run = listed[0]

    assert run["source_title"] == "Hacker News", (
        "F32: a run row must name its source, not read 'Untitled'"
    )
    assert run["watchlist_names"] == ["Morning read"]
    assert run["found_count"] == 30, (
        "F33: the ~30-item check must show ~30 found"
    )
    assert run["processed_count"] == 29
    assert run["filtered_count"] == 1
    assert run["error_count"] == 0
    assert run["duration"] is not None and run["duration"] != "-", (
        "a finished run knows how long it took"
    )
    # The same record, read one at a time, must agree with the list.
    fetched = await service.get_run(launched["run_id"])
    assert fetched["source_title"] == "Hacker News"
    assert fetched["found_count"] == 30


@pytest.mark.asyncio
async def test_a_failed_run_reports_one_error_and_no_found_items(tmp_path):
    """The discriminating half: zeros must be REAL zeros, not missing keys."""

    async def exploding_executor(subscription):
        raise RuntimeError("feed unreachable")

    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(
        db_factory=lambda: db, run_executor=exploding_executor
    )
    source = await service.create_source(
        {"name": "Broken", "url": "https://example.com/x.xml", "source_type": "rss"}
    )
    launched = await service.launch_run(source_id=source["source_id"])
    await service.execute_run(launched["run_id"])

    run = (await service.list_runs())[0]

    assert run["status"] == "failed"
    assert run["source_title"] == "Broken"
    assert run["watchlist_names"] == []
    assert run["found_count"] == 0
    assert run["processed_count"] == 0
    assert run["filtered_count"] == 0
    assert run["error_count"] == 1, (
        "a failed run reporting zero errors is the flattering answer"
    )


@pytest.mark.asyncio
async def test_a_queued_run_has_no_duration_yet(tmp_path):
    """A run that has not finished reports no duration rather than a fake one."""
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {"name": "Feed", "url": "https://example.com/feed.xml", "source_type": "rss"}
    )

    launched = await service.launch_run(source_id=source["source_id"])

    assert launched["status"] == "queued"
    assert launched["duration"] is None
    assert launched["source_title"] == "Feed"


@pytest.mark.asyncio
async def test_list_items_filters_by_is_flagged(tmp_path):
    """task-3072: the Starred feed's item page.

    Falsey (`None`, the default) means NO flag filter -- the same convention
    the TASK-2513 scope kwargs established; `True` narrows to starred rows,
    and the normalized dicts carry the flag as a real bool.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {"name": "Feed", "url": "https://example.com/feed.xml", "source_type": "rss"}
    )
    with db.transaction() as conn:
        for index in range(3):
            conn.execute(
                "INSERT INTO subscription_items (subscription_id, url, title) "
                "VALUES (?, ?, ?)",
                (source["source_id"], f"https://example.com/{index}", f"Item {index}"),
            )
    starred_id = db.conn.execute(
        "SELECT id FROM subscription_items WHERE url = ?", ("https://example.com/1",)
    ).fetchone()[0]
    db.set_item_flagged(starred_id, True)

    starred = await service.list_items(status=None, is_flagged=True)
    everything = await service.list_items(status=None)

    assert [item["item_id"] for item in starred] == [starred_id]
    assert starred[0]["is_flagged"] is True
    assert len(everything) == 3, "falsey is_flagged must not filter at all"


@pytest.mark.asyncio
async def test_resolve_watchlist_name_is_not_limited_to_the_first_10000_rows(
    tmp_path,
):
    """ADR-043 reuse remains correct beyond the old list-scan cap.

    Replacing the direct lookup with ``list_watchlists(limit=10000)`` makes
    this create ``Target (2)`` instead of returning the existing final row.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO watchlists (name) VALUES (?)",
            [(f"Filler {index:05d}",) for index in range(10001)]
            + [("Target",)],
        )
        target_id = conn.execute(
            "SELECT id FROM watchlists WHERE name = ?", ("Target",)
        ).fetchone()[0]
    service = LocalWatchlistsService(db_factory=lambda: db)

    resolved, created = await service.resolve_or_create_watchlist("  TARGET  ")

    assert created is False
    assert resolved["id"] == target_id
    assert resolved["name"] == "Target"
    assert db.conn.execute(
        "SELECT COUNT(*) FROM watchlists WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))",
        ("Target",),
    ).fetchone()[0] == 1


@pytest.mark.asyncio
async def test_resolve_watchlist_name_keeps_unicode_case_insensitive_reuse(tmp_path):
    """The direct SQL lookup preserves the former Python ``lower`` rule.

    SQLite's built-in ``LOWER`` is ASCII-only, so using it directly misses
    this row and creates a visually duplicate watchlist.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    existing = WatchlistBundleService(db).create("ÄI")
    service = LocalWatchlistsService(db_factory=lambda: db)

    resolved, created = await service.resolve_or_create_watchlist("äi")

    assert created is False
    assert resolved["id"] == existing["id"]
    assert [row["name"] for row in WatchlistBundleService(db).list_watchlists()] == [
        "ÄI"
    ]
