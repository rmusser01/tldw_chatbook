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
