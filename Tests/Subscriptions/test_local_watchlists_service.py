from inspect import isawaitable

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Notifications import ClientNotificationsDB, NotificationDispatchService
from tldw_chatbook.Subscriptions import LocalWatchlistsService


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
    assert [run["run_id"] for run in snapshot[:2]] == [failed["run_id"], queued["run_id"]]
    assert snapshot[0]["id"] == f"local:watchlist_run:{failed['run_id']}"
    assert snapshot[0]["status"] == "failed"
    assert snapshot[0]["source_title"] == "Failed Feed"
    assert snapshot[0]["source_id"] == failed_source["source_id"]
    assert snapshot[1]["status"] == "queued"
    assert snapshot[1]["source_title"] == "Queued Feed"


@pytest.mark.asyncio
async def test_local_watchlists_service_executes_run_and_records_subscription_items(tmp_path):
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

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=fake_run_executor)
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
    updated = await service.update_alert_rule(created["rule_id"], enabled=False, severity="critical")
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
            "extraction_rules": {"urls": ["https://example.com/a", "https://example.com/b"]},
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
    assert updated["settings"]["extraction_rules"] == {"urls": ["https://example.com/c"]}


@pytest.mark.asyncio
async def test_local_watchlists_service_executes_url_list_sources_with_default_url_monitor(tmp_path, monkeypatch):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    seen_urls = []

    class FakeURLMonitor:
        def __init__(self, db):
            self.db = db

        async def check_url(self, subscription):
            seen_urls.append(subscription["source"])
            return {
                "url": subscription["source"],
                "title": f"Changed {len(seen_urls)}",
                "content_hash": f"hash-{len(seen_urls)}",
                "published_date": "2026-04-25T00:00:00+00:00",
            }

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
async def test_local_watchlists_service_executes_sitemap_sources_with_default_url_monitor(tmp_path, monkeypatch):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    fetched_sitemaps = []
    seen_urls = []

    class FakeResponse:
        text = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page-a</loc></url>
            <url><loc>https://example.com/page-b</loc></url>
        </urlset>
        """

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def get(self, url):
            fetched_sitemaps.append(url)
            return FakeResponse()

    class FakeURLMonitor:
        def __init__(self, db):
            self.db = db

        async def check_url(self, subscription):
            seen_urls.append(subscription["source"])
            return {
                "url": subscription["source"],
                "title": f"Sitemap page {len(seen_urls)}",
                "content_hash": f"sitemap-hash-{len(seen_urls)}",
                "published_date": "2026-04-25T00:00:00+00:00",
            }

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)
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
async def test_local_watchlists_service_executes_api_sources_with_json_field_mapping(tmp_path, monkeypatch):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    requests = []

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            return None

        def json(self):
            return {
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

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def get(self, url, **kwargs):
            requests.append({"url": url, **kwargs})
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)
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
async def test_local_watchlists_service_evaluates_completed_run_alerts_into_notifications(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    notification_store = ClientNotificationsDB(tmp_path / "notifications.db")
    dispatcher = NotificationDispatchService(store=notification_store)
    service = LocalWatchlistsService(db_factory=lambda: db, notification_dispatcher=dispatcher)
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
