"""Tests for server jobs namespace gateway access."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.asyncio
async def test_jobs_namespace_gateway_routes_status_events_and_attachment_surfaces(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    await client.call_server_jobs_endpoint("GET", "queue/status")
    await client.call_server_jobs_endpoint("GET", "/api/v1/jobs/events", params={"limit": 50})
    await client.call_server_jobs_endpoint(
        "POST",
        "job-1/attachments",
        payload={"name": "trace.json", "content_type": "application/json"},
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/jobs/queue/status")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/jobs/events")
    assert mocked.await_args_list[1].kwargs["params"] == {"limit": 50}
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/jobs/job-1/attachments")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "name": "trace.json",
        "content_type": "application/json",
    }


@pytest.mark.asyncio
async def test_jobs_namespace_gateway_rejects_cross_namespace_and_unsafe_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    unsafe_calls = [
        client.call_server_jobs_endpoint("GET", "/api/v1/admin/jobs"),
        client.call_server_jobs_endpoint("GET", "/api/v1/notifications/stream"),
        client.call_server_jobs_endpoint("GET", "../admin/jobs"),
        client.call_server_jobs_endpoint("OPTIONS", "stats"),
    ]
    for call in unsafe_calls:
        with pytest.raises(ValueError):
            await call

    mocked.assert_not_awaited()
