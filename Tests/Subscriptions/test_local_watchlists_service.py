import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
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
