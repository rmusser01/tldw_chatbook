from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.mcp_unified_client import MCPUnifiedClient
from tldw_chatbook.tldw_api.mcp_unified_schemas import CatalogConnectionTestRequest


def _assert_request_call(call_args, expected_method, expected_endpoint, expected_kwargs):
    args, kwargs = call_args
    assert args[:2] == (expected_method, expected_endpoint)
    for key, value in expected_kwargs.items():
        assert kwargs[key] == value


class _DummyRootClient:
    def __init__(self):
        self._request = AsyncMock()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name, call_kwargs, expected_method, expected_endpoint, expected_kwargs, response",
    [
        (
            "get_status",
            {},
            "GET",
            "/api/v1/mcp/status",
            {"params": None},
            {"status": "ok"},
        ),
        (
            "get_health",
            {},
            "GET",
            "/api/v1/mcp/health",
            {"params": None},
            {"healthy": True},
        ),
        (
            "get_metrics",
            {},
            "GET",
            "/api/v1/mcp/metrics",
            {"params": None},
            {"connections": {}, "modules": {}},
        ),
        (
            "list_modules",
            {},
            "GET",
            "/api/v1/mcp/modules",
            {"params": None},
            {"modules": []},
        ),
        (
            "get_module_health",
            {},
            "GET",
            "/api/v1/mcp/modules/health",
            {"params": None},
            {"modules": []},
        ),
        (
            "list_tools",
            {"catalog": "demo", "module": ["media", "chat"], "catalog_strict": True},
            "GET",
            "/api/v1/mcp/tools",
            {"params": {"catalog": "demo", "module": ["media", "chat"], "catalog_strict": "1"}},
            {"tools": []},
        ),
        (
            "list_resources",
            {},
            "GET",
            "/api/v1/mcp/resources",
            {"params": None},
            {"resources": []},
        ),
        (
            "list_prompts",
            {},
            "GET",
            "/api/v1/mcp/prompts",
            {"params": None},
            {"prompts": []},
        ),
        (
            "list_visible_tool_catalogs",
            {},
            "GET",
            "/api/v1/mcp/tool_catalogs",
            {"params": None},
            {"catalogs": []},
        ),
    ],
)
async def test_unified_client_routes_expected_endpoints(
    method_name,
    call_kwargs,
    expected_method,
    expected_endpoint,
    expected_kwargs,
    response,
):
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value=response)
    root._request = mocked

    method = getattr(client, method_name)
    await method(**call_kwargs)

    mocked.assert_awaited_once()
    _assert_request_call(mocked.await_args, expected_method, expected_endpoint, expected_kwargs)


@pytest.mark.asyncio
async def test_execute_tool_posts_to_execute_endpoint(monkeypatch):
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"result": {"ok": True}})
    root._request = mocked

    await client.execute_tool(tool_name="mcp.tools.list", arguments={"scope": "all"})

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "POST",
        "/api/v1/mcp/tools/execute",
        {"json_data": {"tool_name": "mcp.tools.list", "arguments": {"scope": "all"}}},
    )


@pytest.mark.asyncio
async def test_test_catalog_connection_posts_to_test_connection_endpoint(monkeypatch):
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"success": True})
    root._request = mocked

    await client.test_catalog_connection(
        CatalogConnectionTestRequest(
            catalog_name="demo",
            connection={"base_url": "https://catalog.example.com"},
        )
    )

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "POST",
        "/api/v1/mcp/catalog/test-connection",
        {"json_data": {"catalog_name": "demo", "connection": {"base_url": "https://catalog.example.com"}}},
    )


def test_access_context_bootstrap_helper_normalizes_scope_options():
    client = MCPUnifiedClient(_DummyRootClient())

    personal = client.normalize_access_context()
    team = client.normalize_access_context(team_id=17)
    org = client.normalize_access_context(org_id="org-9")
    explicit = client.normalize_access_context(scope_kind="system_admin")

    assert personal.scope_kind == "personal"
    assert personal.scope_ref is None
    assert team.scope_kind == "team"
    assert team.scope_ref == "17"
    assert org.scope_kind == "org"
    assert org.scope_ref == "org-9"
    assert explicit.scope_kind == "system_admin"
    assert explicit.scope_ref is None
    assert client.build_access_context_params(team) == {"scope_kind": "team", "scope_ref": "17", "team_id": "17"}
