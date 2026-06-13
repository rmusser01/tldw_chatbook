"""Tests for server ACP namespace access on the shared API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.asyncio
async def test_acp_namespace_gateway_routes_agent_session_and_governance_surfaces(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    await client.call_server_acp_endpoint("GET", "agents/health")
    await client.call_server_acp_endpoint(
        "post",
        "/api/v1/acp/sessions/prompt-async",
        payload={"session_id": "session-1", "prompt": "Continue"},
        headers={"Idempotency-Key": "idem-1"},
    )
    await client.call_server_acp_endpoint(
        "DELETE",
        "permissions/decisions/decision-1",
        params={"reason": "revoked"},
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/acp/agents/health")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/acp/sessions/prompt-async")
    assert mocked.await_args_list[1].kwargs["json_data"] == {"session_id": "session-1", "prompt": "Continue"}
    assert mocked.await_args_list[1].kwargs["headers"] == {"Idempotency-Key": "idem-1"}
    assert mocked.await_args_list[2].args[:2] == ("DELETE", "/api/v1/acp/permissions/decisions/decision-1")
    assert mocked.await_args_list[2].kwargs["params"] == {"reason": "revoked"}


@pytest.mark.asyncio
async def test_acp_namespace_gateway_rejects_admin_cross_namespace_and_unsafe_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    unsafe_calls = [
        client.call_server_acp_endpoint("GET", "/api/v1/admin/acp/agents"),
        client.call_server_acp_endpoint("GET", "/api/v1/mcp/hub/acp-profiles"),
        client.call_server_acp_endpoint("GET", "../admin/acp/agents"),
        client.call_server_acp_endpoint("OPTIONS", "health"),
    ]
    for call in unsafe_calls:
        with pytest.raises(ValueError):
            await call

    mocked.assert_not_awaited()
