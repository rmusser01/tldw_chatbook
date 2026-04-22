from __future__ import annotations

import importlib
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
            url="https://catalog.example.com",
            auth_type="api_key",
            secret="top-secret",
            auth_key_name="X-Test-Key",
        )
    )

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "POST",
        "/api/v1/mcp/catalog/test-connection",
        {
            "json_data": {
                "url": "https://catalog.example.com",
                "auth_type": "api_key",
                "secret": "top-secret",
                "auth_key_name": "X-Test-Key",
            }
        },
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


def test_package_root_keeps_legacy_and_mcp_exports_lazy():
    api = importlib.import_module("tldw_chatbook.tldw_api")

    exported_names = set(getattr(api, "__all__", ()))

    assert {"TLDWAPIClient", "SourceResponse", "MCPUnifiedClient"} <= exported_names
    assert api.SourceResponse.__module__.endswith("watchlists_schemas")
    assert api.MCPUnifiedClient is MCPUnifiedClient


@pytest.mark.asyncio
async def test_get_current_user_profile_requests_user_and_memberships_sections():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(
        return_value={
            "user": {
                "id": 42,
                "username": "casey",
                "email": "casey@example.com",
                "role": "member",
                "is_active": True,
                "is_verified": True,
                "created_at": "2026-04-20T10:00:00Z",
            },
            "memberships": {"orgs": [], "teams": []},
        }
    )
    root._request = mocked

    profile = await client.get_current_user_profile()

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "GET",
        "/api/v1/users/me/profile",
        {"params": {"sections": "user,memberships"}},
    )
    assert profile.user is not None
    assert profile.user.username == "casey"


@pytest.mark.asyncio
async def test_bootstrap_access_context_derives_manageable_scopes_and_admin_signal():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    root._request = AsyncMock(
        return_value={
            "profile_version": "2026-04-22T01:02:03Z",
            "catalog_version": "v1",
            "user": {
                "id": 7,
                "username": "operator",
                "email": "operator@example.com",
                "role": "admin",
                "is_active": True,
                "is_verified": True,
                "created_at": "2026-01-01T00:00:00Z",
            },
            "memberships": {
                "orgs": [
                    {"org_id": 11, "role": "owner"},
                    {"org_id": 12, "role": "member"},
                    {"org_id": 13, "role": "lead"},
                ],
                "teams": [
                    {"team_id": 21, "org_id": 11, "role": "admin"},
                    {"team_id": 22, "org_id": 11, "role": "viewer"},
                    {"team_id": 23, "org_id": 13, "role": "lead"},
                ],
            },
        }
    )

    bootstrap = await client.bootstrap_access_context()

    assert bootstrap.principal is not None
    assert bootstrap.principal.user_id == 7
    assert bootstrap.principal.username == "operator"
    assert bootstrap.principal.role == "admin"
    assert bootstrap.principal.is_admin is True
    assert bootstrap.manageable_org_ids == [11, 13]
    assert bootstrap.manageable_team_ids == [21, 23]
    assert bootstrap.can_use_system_admin_scope is True
