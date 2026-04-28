"""Tests for server setup, kanban, moderation, and monitoring namespace gateways."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.asyncio
async def test_control_surface_gateways_route_namespace_scoped_server_surfaces(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    await client.call_server_setup_endpoint("GET", "status")
    await client.call_server_kanban_endpoint(
        "POST",
        "/api/v1/kanban/cards/search",
        payload={"query": "sync"},
    )
    await client.call_server_moderation_endpoint(
        "PUT",
        "settings",
        payload={"enabled": True},
    )
    await client.call_server_monitoring_endpoint("GET", "notifications/recent", params={"limit": 10})

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/setup/status")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/kanban/cards/search")
    assert mocked.await_args_list[1].kwargs["json_data"] == {"query": "sync"}
    assert mocked.await_args_list[2].args[:2] == ("PUT", "/api/v1/moderation/settings")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"enabled": True}
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/monitoring/notifications/recent")
    assert mocked.await_args_list[3].kwargs["params"] == {"limit": 10}


@pytest.mark.asyncio
async def test_control_surface_gateways_reject_cross_namespace_and_unsafe_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    unsafe_calls = [
        client.call_server_setup_endpoint("GET", "/api/v1/admin/users"),
        client.call_server_kanban_endpoint("GET", "/api/v1/setup/status"),
        client.call_server_moderation_endpoint("GET", "../admin/users"),
        client.call_server_monitoring_endpoint("OPTIONS", "alerts"),
    ]
    for call in unsafe_calls:
        with pytest.raises(ValueError):
            await call

    mocked.assert_not_awaited()
