from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ClaimNotificationsAckRequest,
    ClaimNotificationsDigestResponse,
    ClaimNotificationResponse,
    ClaimsAlertConfigCreate,
    ClaimsAlertConfigResponse,
    ClaimsAlertConfigUpdate,
    TLDWAPIClient,
)


def _notification_payload(**overrides) -> dict:
    payload = {
        "id": 31,
        "user_id": "7",
        "kind": "watchlist_cluster",
        "target_user_id": "7",
        "target_review_group": None,
        "resource_type": "claim_cluster",
        "resource_id": "cluster-9",
        "payload": {"summary": "New matching claim cluster"},
        "created_at": "2026-04-25T12:00:00Z",
        "delivered_at": None,
    }
    payload.update(overrides)
    return payload


def _alert_payload(**overrides) -> dict:
    payload = {
        "id": 41,
        "user_id": "7",
        "name": "Unsupported ratio spike",
        "alert_type": "unsupported_ratio",
        "threshold_ratio": 0.25,
        "baseline_ratio": 0.1,
        "channels": {"notification": True, "email": False},
        "slack_webhook_url": None,
        "webhook_url": None,
        "email_recipients": [],
        "enabled": True,
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_claims_client_routes_notifications_and_alerts(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            [_notification_payload()],
            {
                "total": 1,
                "counts_by_kind": {"watchlist_cluster": 1},
                "counts_by_target_user": {"7": 1},
                "counts_by_review_group": {},
                "notifications": [_notification_payload()],
            },
            {"updated": 1},
            {"evaluated": 1, "created": 1},
            [_alert_payload()],
            _alert_payload(),
            _alert_payload(enabled=False),
            {"success": True},
            {"evaluated": 1, "triggered": 1},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    notifications = await client.list_claim_notifications(kind="watchlist_cluster", delivered=False, limit=25, offset=5)
    digest = await client.get_claim_notifications_digest(include_items=True, ack=True, limit=25)
    acked = await client.ack_claim_notifications(ClaimNotificationsAckRequest(ids=[31]))
    watchlist_eval = await client.evaluate_claim_watchlist_notifications()
    alerts = await client.list_claim_alerts()
    created = await client.create_claim_alert(
        ClaimsAlertConfigCreate(
            name="Unsupported ratio spike",
            alert_type="unsupported_ratio",
            threshold_ratio=0.25,
            baseline_ratio=0.1,
            channels={"notification": True, "email": False},
        )
    )
    updated = await client.update_claim_alert(41, ClaimsAlertConfigUpdate(enabled=False))
    deleted = await client.delete_claim_alert(41)
    evaluated = await client.evaluate_claim_alerts(window_sec=1800, baseline_sec=7200)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/claims/notifications")
    assert mocked.await_args_list[0].kwargs["params"] == {
        "kind": "watchlist_cluster",
        "delivered": False,
        "limit": 25,
        "offset": 5,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/claims/notifications/digest")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "include_items": True,
        "ack": True,
        "limit": 25,
        "offset": 0,
    }
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/claims/notifications/ack")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"ids": [31]}
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/claims/notifications/watchlists/evaluate")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/claims/alerts")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/claims/alerts")
    assert mocked.await_args_list[5].kwargs["json_data"] == {
        "name": "Unsupported ratio spike",
        "alert_type": "unsupported_ratio",
        "threshold_ratio": 0.25,
        "baseline_ratio": 0.1,
        "channels": {"notification": True, "email": False},
    }
    assert mocked.await_args_list[6].args[:2] == ("PATCH", "/api/v1/claims/alerts/41")
    assert mocked.await_args_list[6].kwargs["json_data"] == {"enabled": False}
    assert mocked.await_args_list[7].args[:2] == ("DELETE", "/api/v1/claims/alerts/41")
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/claims/alerts/evaluate")
    assert mocked.await_args_list[8].kwargs["params"] == {"window_sec": 1800, "baseline_sec": 7200}

    assert isinstance(notifications[0], ClaimNotificationResponse)
    assert isinstance(digest, ClaimNotificationsDigestResponse)
    assert acked["updated"] == 1
    assert watchlist_eval["created"] == 1
    assert isinstance(alerts[0], ClaimsAlertConfigResponse)
    assert isinstance(created, ClaimsAlertConfigResponse)
    assert updated.enabled is False
    assert deleted["success"] is True
    assert evaluated["triggered"] == 1
