from __future__ import annotations

import importlib
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.mcp_unified_client import MCPUnifiedClient
from tldw_chatbook.tldw_api.mcp_governance_schemas import MCPGovernanceEvent
from tldw_chatbook.tldw_api.mcp_unified_schemas import (
    ApprovalPolicyUpdateRequest,
    CatalogConnectionTestRequest,
    ExternalSecretSetRequest,
    ExternalServerCreateRequest,
    PermissionProfileCreateRequest,
    ScopedToolCatalogCreateRequest,
)


def _assert_request_call(call_args, expected_method, expected_endpoint, expected_kwargs):
    args, kwargs = call_args
    assert args[:2] == (expected_method, expected_endpoint)
    for key, value in expected_kwargs.items():
        assert kwargs[key] == value


class _DummyRootClient:
    def __init__(self):
        self._request = AsyncMock()
        self._request_bytes = AsyncMock()


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


@pytest.mark.asyncio
async def test_unified_client_routes_remaining_mcp_runtime_contract_edges():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"ok": True})
    root._request = mocked

    await client.create_auth_token(username="admin", api_key="demo-secret")
    await client.refresh_auth_token(refresh_token="refresh-token", token_id="token-id")
    await client.list_catalog(archetype_key="research_assistant")
    await client.send_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, client_id="chatbook")
    await client.send_request_batch(
        [{"jsonrpc": "2.0", "id": 2, "method": "resources/list"}],
        client_id="chatbook",
        mcp_session_id="session-1",
        config="encoded-config",
    )
    await client.get_profile_slot_credential_binding_status(
        profile_id=7,
        server_id="docs",
        slot_name="token_readonly",
    )
    await client.get_assignment_slot_credential_binding_status(
        assignment_id=11,
        server_id="docs",
        slot_name="token_write",
    )

    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("POST", "/api/v1/mcp/auth/token"),
        ("POST", "/api/v1/mcp/auth/refresh"),
        ("GET", "/api/v1/mcp/catalog"),
        ("POST", "/api/v1/mcp/request"),
        ("POST", "/api/v1/mcp/request/batch"),
        ("GET", "/api/v1/mcp/hub/permission-profiles/7/credential-bindings/docs/token_readonly/status"),
        ("GET", "/api/v1/mcp/hub/policy-assignments/11/credential-bindings/docs/token_write/status"),
    ]
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "username": "admin",
        "api_key": "demo-secret",
    }
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "refresh_token": "refresh-token",
        "token_id": "token-id",
    }
    assert mocked.await_args_list[2].kwargs["params"] == {"archetype_key": "research_assistant"}
    assert mocked.await_args_list[3].kwargs["params"] == {"client_id": "chatbook"}
    assert mocked.await_args_list[4].kwargs["params"] == {"client_id": "chatbook", "config": "encoded-config"}
    assert mocked.await_args_list[4].kwargs["headers"] == {"mcp-session-id": "session-1"}


@pytest.mark.asyncio
async def test_unified_client_streams_mcp_governance_events():
    root = _DummyRootClient()
    calls = []

    async def fake_stream(endpoint, *, params=None, event_model=None, headers=None):
        calls.append((endpoint, params, event_model, headers))
        yield event_model.model_validate(
            {
                "event_id": "evt-2",
                "event_type": "mcp_hub.approval_policy.created",
                "resource_type": "mcp_approval_policy",
                "resource_id": "12",
            }
        )

    root._stream_sse_request = fake_stream
    client = MCPUnifiedClient(root)

    events = [
        event
        async for event in client.stream_governance_events(
            after_event_id="evt-1",
            event_types=["mcp_hub.approval_policy.created"],
            replay=True,
        )
    ]

    assert calls == [
        (
            "/api/v1/mcp/hub/events/stream",
            {
                "after_event_id": "evt-1",
                "event_type": ["mcp_hub.approval_policy.created"],
                "replay": "true",
            },
            MCPGovernanceEvent,
            None,
        )
    ]
    assert events[0].event_id == "evt-2"
    assert events[0].resource_type == "mcp_approval_policy"


@pytest.mark.asyncio
async def test_unified_client_fetches_prometheus_metrics_as_text():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    root._request_bytes = AsyncMock(return_value=b"mcp_requests_total 1\n")

    metrics = await client.get_prometheus_metrics()

    assert metrics == "mcp_requests_total 1\n"
    root._request_bytes.assert_awaited_once_with("GET", "/api/v1/mcp/metrics/prometheus")


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


@pytest.mark.asyncio
async def test_list_scoped_team_tool_catalogs_routes_to_team_catalog_endpoint():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value=[{"id": 9, "name": "Team Catalog"}])
    root._request = mocked

    await client.list_scoped_tool_catalogs(scope_kind="team", scope_ref="21")

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "GET",
        "/api/v1/teams/21/mcp/tool_catalogs",
        {"params": None},
    )


@pytest.mark.asyncio
async def test_create_scoped_org_tool_catalog_routes_to_org_catalog_endpoint():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"id": 11, "name": "Org Catalog"})
    root._request = mocked

    await client.create_scoped_tool_catalog(
        scope_kind="org",
        scope_ref="44",
        request=ScopedToolCatalogCreateRequest(name="Org Catalog", description="Scoped", is_active=True),
    )

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "POST",
        "/api/v1/orgs/44/mcp/tool_catalogs",
        {
            "json_data": {
                "name": "Org Catalog",
                "description": "Scoped",
                "is_active": True,
            }
        },
    )


@pytest.mark.asyncio
async def test_list_external_servers_routes_to_mcp_hub_endpoint_with_scope_filters():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value=[{"id": "docs", "name": "Docs"}])
    root._request = mocked

    await client.list_external_servers(owner_scope_type="team", owner_scope_id=21)

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "GET",
        "/api/v1/mcp/hub/external-servers",
        {"params": {"owner_scope_type": "team", "owner_scope_id": 21}},
    )


@pytest.mark.asyncio
async def test_create_external_server_routes_to_mcp_hub_create_endpoint():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"id": "docs", "name": "Docs"})
    root._request = mocked

    await client.create_external_server(
        ExternalServerCreateRequest(
            server_id="docs",
            name="Docs",
            transport="http",
            config={"url": "https://docs.example/mcp"},
            owner_scope_type="team",
            owner_scope_id=21,
            enabled=True,
        )
    )

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "POST",
        "/api/v1/mcp/hub/external-servers",
        {
            "json_data": {
                "server_id": "docs",
                "name": "Docs",
                "transport": "http",
                "config": {"url": "https://docs.example/mcp"},
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "enabled": True,
            }
        },
    )


@pytest.mark.asyncio
async def test_list_permission_profiles_routes_to_mcp_hub_endpoint_with_scope_filters():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value=[{"id": 1, "name": "Default"}])
    root._request = mocked

    await client.list_permission_profiles(owner_scope_type="team", owner_scope_id=21)

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "GET",
        "/api/v1/mcp/hub/permission-profiles",
        {"params": {"owner_scope_type": "team", "owner_scope_id": 21}},
    )


@pytest.mark.asyncio
async def test_create_permission_profile_routes_to_mcp_hub_create_endpoint():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"id": 1, "name": "Default"})
    root._request = mocked

    await client.create_permission_profile(
        PermissionProfileCreateRequest(
            name="Default",
            description="Default profile",
            owner_scope_type="team",
            owner_scope_id=21,
            mode="custom",
            policy_document={"allowed_tools": ["mcp.tool"]},
            is_active=True,
        )
    )

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "POST",
        "/api/v1/mcp/hub/permission-profiles",
        {
            "json_data": {
                "name": "Default",
                "description": "Default profile",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "mode": "custom",
                "policy_document": {"allowed_tools": ["mcp.tool"]},
                "is_active": True,
            }
        },
    )


@pytest.mark.asyncio
async def test_update_approval_policy_routes_to_mcp_hub_update_endpoint():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"id": 7, "name": "Updated"})
    root._request = mocked

    await client.update_approval_policy(
        approval_policy_id=7,
        request=ApprovalPolicyUpdateRequest(name="Updated"),
    )

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "PUT",
        "/api/v1/mcp/hub/approval-policies/7",
        {"json_data": {"name": "Updated"}},
    )


@pytest.mark.asyncio
async def test_get_assignment_external_access_routes_to_mcp_hub_endpoint():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"servers": []})
    root._request = mocked

    await client.get_assignment_external_access(assignment_id=14)

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "GET",
        "/api/v1/mcp/hub/policy-assignments/14/external-access",
        {"params": None},
    )


@pytest.mark.asyncio
async def test_get_tool_registry_summary_routes_to_summary_endpoint():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"entries": [], "modules": []})
    root._request = mocked

    await client.get_tool_registry_summary()

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "GET",
        "/api/v1/mcp/hub/tool-registry/summary",
        {"params": None},
    )


@pytest.mark.asyncio
async def test_tool_registry_entries_and_modules_route_to_full_registry_endpoints():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)

    root._request = AsyncMock(return_value=[{"tool_name": "docs.search"}])
    await client.list_tool_registry_entries()
    _assert_request_call(
        root._request.await_args,
        "GET",
        "/api/v1/mcp/hub/tool-registry",
        {"params": None},
    )

    root._request = AsyncMock(return_value=[{"module": "search"}])
    await client.list_tool_registry_modules()
    _assert_request_call(
        root._request.await_args,
        "GET",
        "/api/v1/mcp/hub/tool-registry/modules",
        {"params": None},
    )


@pytest.mark.asyncio
async def test_list_governance_packs_routes_to_endpoint_with_scope_filters():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value=[{"id": 1, "name": "Baseline"}])
    root._request = mocked

    await client.list_governance_packs(owner_scope_type="team", owner_scope_id=21)

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "GET",
        "/api/v1/mcp/hub/governance-packs",
        {"params": {"owner_scope_type": "team", "owner_scope_id": 21}},
    )


@pytest.mark.asyncio
async def test_update_governance_pack_trust_policy_routes_to_endpoint():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"mode": "allowlist"})
    root._request = mocked

    await client.update_governance_pack_trust_policy(
        {"mode": "allowlist", "allowed_sources": ["git@example.com:trusted/repo.git"]}
    )

    mocked.assert_awaited_once()
    _assert_request_call(
        mocked.await_args,
        "PUT",
        "/api/v1/mcp/hub/governance-packs/trust-policy",
        {
            "json_data": {
                "mode": "allowlist",
                "allowed_sources": ["git@example.com:trusted/repo.git"],
            }
        },
    )


@pytest.mark.asyncio
async def test_governance_pack_source_and_upgrade_routes_cover_remaining_endpoints():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)

    payload = {
        "owner_scope_type": "team",
        "owner_scope_id": 21,
        "pack": {"manifest": {"pack_id": "baseline", "version": "1.0.0"}},
    }
    source_payload = {
        "candidate_id": "cand-1",
        "source": {"kind": "git", "url": "git@example.com:trusted/repo.git", "ref": "main"},
        "owner_scope_type": "team",
        "owner_scope_id": 21,
    }
    upgrade_payload = {
        "source_governance_pack_id": 81,
        "owner_scope_type": "team",
        "owner_scope_id": 21,
        "pack": {"manifest": {"pack_id": "baseline", "version": "1.1.0"}},
        "planner_inputs_fingerprint": "planner-1",
        "adapter_state_fingerprint": "adapter-1",
    }
    source_upgrade_payload = {
        "candidate_id": "cand-1",
        "source_governance_pack_id": 81,
        "owner_scope_type": "team",
        "owner_scope_id": 21,
        "planner_inputs_fingerprint": "planner-1",
        "adapter_state_fingerprint": "adapter-1",
    }

    root._request = AsyncMock(return_value={"report": {}})
    await client.dry_run_governance_pack(payload)
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/governance-packs/dry-run",
        {"json_data": payload},
    )

    root._request = AsyncMock(return_value={"candidate_id": "cand-1"})
    await client.prepare_governance_pack_source({"source": source_payload["source"]})
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/governance-packs/source/prepare",
        {"json_data": {"source": source_payload["source"]}},
    )

    root._request = AsyncMock(return_value={"report": {}})
    await client.dry_run_governance_pack_source(
        {
            "candidate_id": "cand-1",
            "owner_scope_type": "team",
            "owner_scope_id": 21,
        }
    )
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/governance-packs/source/dry-run",
        {
            "json_data": {
                "candidate_id": "cand-1",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
            }
        },
    )

    root._request = AsyncMock(return_value={"has_update": True})
    await client.check_governance_pack_updates(governance_pack_id=81)
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/governance-packs/81/check-updates",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"candidate_id": "cand-2"})
    await client.prepare_governance_pack_upgrade_candidate(governance_pack_id=81)
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/governance-packs/81/prepare-upgrade-candidate",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"plan": {}})
    await client.dry_run_governance_pack_upgrade(upgrade_payload)
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/governance-packs/dry-run-upgrade",
        {"json_data": upgrade_payload},
    )

    root._request = AsyncMock(return_value={"plan": {}})
    await client.dry_run_governance_pack_source_upgrade(source_upgrade_payload)
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/governance-packs/source/dry-run-upgrade",
        {"json_data": source_upgrade_payload},
    )

    root._request = AsyncMock(return_value={"governance_pack_id": 81})
    await client.import_governance_pack(payload)
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/governance-packs/import",
        {"json_data": payload},
    )

    root._request = AsyncMock(return_value={"governance_pack_id": 82})
    await client.import_governance_pack_source(
        {
            "candidate_id": "cand-1",
            "owner_scope_type": "team",
            "owner_scope_id": 21,
        }
    )
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/governance-packs/source/import",
        {
            "json_data": {
                "candidate_id": "cand-1",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
            }
        },
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.execute_governance_pack_source_upgrade(source_upgrade_payload)
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/governance-packs/source/execute-upgrade",
        {"json_data": source_upgrade_payload},
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.execute_governance_pack_upgrade(upgrade_payload)
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/governance-packs/execute-upgrade",
        {"json_data": upgrade_payload},
    )

    root._request = AsyncMock(return_value=[{"from_version": "1.0.0", "to_version": "1.1.0"}])
    await client.list_governance_pack_upgrade_history(governance_pack_id=81)
    _assert_request_call(
        root._request.await_args,
        "GET",
        "/api/v1/mcp/hub/governance-packs/81/upgrade-history",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"id": 81, "pack_id": "baseline"})
    await client.get_governance_pack_detail(governance_pack_id=81)
    _assert_request_call(
        root._request.await_args,
        "GET",
        "/api/v1/mcp/hub/governance-packs/81",
        {"params": None},
    )


@pytest.mark.asyncio
async def test_capability_mapping_routes_cover_preview_and_crud():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)

    root._request = AsyncMock(return_value={"normalized_mapping": {}})
    await client.preview_capability_mapping(
        {
            "mapping_id": "filesystem-write",
            "owner_scope_type": "team",
            "owner_scope_id": 21,
            "capability_name": "filesystem.write",
        }
    )
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/capability-mappings/preview",
        {
            "json_data": {
                "mapping_id": "filesystem-write",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "capability_name": "filesystem.write",
            }
        },
    )

    root._request = AsyncMock(return_value={"id": 11})
    await client.create_capability_mapping(
        {
            "mapping_id": "filesystem-write",
            "owner_scope_type": "team",
            "owner_scope_id": 21,
            "capability_name": "filesystem.write",
        }
    )
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/capability-mappings",
        {
            "json_data": {
                "mapping_id": "filesystem-write",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "capability_name": "filesystem.write",
            }
        },
    )

    root._request = AsyncMock(return_value={"id": 11})
    await client.update_capability_mapping(
        capability_adapter_mapping_id=11,
        payload={"title": "Filesystem Write"},
    )
    _assert_request_call(
        root._request.await_args,
        "PUT",
        "/api/v1/mcp/hub/capability-mappings/11",
        {"json_data": {"title": "Filesystem Write"}},
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.delete_capability_mapping(capability_adapter_mapping_id=11)
    _assert_request_call(
        root._request.await_args,
        "DELETE",
        "/api/v1/mcp/hub/capability-mappings/11",
        {"params": None},
    )


@pytest.mark.asyncio
async def test_path_scope_workspace_set_and_shared_workspace_routes_cover_admin_crud():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)

    root._request = AsyncMock(return_value={"id": 5})
    await client.update_path_scope_object(
        path_scope_object_id=5,
        payload={"name": "Workspace Root", "is_active": False},
    )
    _assert_request_call(
        root._request.await_args,
        "PUT",
        "/api/v1/mcp/hub/path-scope-objects/5",
        {"json_data": {"name": "Workspace Root", "is_active": False}},
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.delete_path_scope_object(path_scope_object_id=5)
    _assert_request_call(
        root._request.await_args,
        "DELETE",
        "/api/v1/mcp/hub/path-scope-objects/5",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"id": 6})
    await client.create_workspace_set_object(
        {
            "name": "Research Set",
            "owner_scope_type": "team",
            "owner_scope_id": 21,
        }
    )
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/workspace-set-objects",
        {
            "json_data": {
                "name": "Research Set",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
            }
        },
    )

    root._request = AsyncMock(return_value={"id": 6})
    await client.update_workspace_set_object(
        workspace_set_object_id=6,
        payload={"description": "Updated"},
    )
    _assert_request_call(
        root._request.await_args,
        "PUT",
        "/api/v1/mcp/hub/workspace-set-objects/6",
        {"json_data": {"description": "Updated"}},
    )

    root._request = AsyncMock(return_value=[{"workspace_id": "ws-1"}])
    await client.list_workspace_set_members(workspace_set_object_id=6)
    _assert_request_call(
        root._request.await_args,
        "GET",
        "/api/v1/mcp/hub/workspace-set-objects/6/members",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"workspace_id": "ws-1"})
    await client.add_workspace_set_member(
        workspace_set_object_id=6,
        payload={"workspace_id": "ws-1"},
    )
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/workspace-set-objects/6/members",
        {"json_data": {"workspace_id": "ws-1"}},
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.delete_workspace_set_member(workspace_set_object_id=6, workspace_id="ws-1")
    _assert_request_call(
        root._request.await_args,
        "DELETE",
        "/api/v1/mcp/hub/workspace-set-objects/6/members/ws-1",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.delete_workspace_set_object(workspace_set_object_id=6)
    _assert_request_call(
        root._request.await_args,
        "DELETE",
        "/api/v1/mcp/hub/workspace-set-objects/6",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"id": 7})
    await client.create_shared_workspace(
        {
            "workspace_id": "shared-ws",
            "display_name": "Shared Workspace",
            "absolute_root": "/srv/shared",
            "owner_scope_type": "team",
            "owner_scope_id": 21,
        }
    )
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/shared-workspaces",
        {
            "json_data": {
                "workspace_id": "shared-ws",
                "display_name": "Shared Workspace",
                "absolute_root": "/srv/shared",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
            }
        },
    )

    root._request = AsyncMock(return_value={"id": 7})
    await client.update_shared_workspace(
        shared_workspace_id=7,
        payload={"display_name": "Shared Workspace Updated"},
    )
    _assert_request_call(
        root._request.await_args,
        "PUT",
        "/api/v1/mcp/hub/shared-workspaces/7",
        {"json_data": {"display_name": "Shared Workspace Updated"}},
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.delete_shared_workspace(shared_workspace_id=7)
    _assert_request_call(
        root._request.await_args,
        "DELETE",
        "/api/v1/mcp/hub/shared-workspaces/7",
        {"params": None},
    )


@pytest.mark.asyncio
async def test_remaining_governance_workspace_binding_and_secret_routes_cover_admin_tail():
    root = _DummyRootClient()
    client = MCPUnifiedClient(root)

    root._request = AsyncMock(return_value=[{"workspace_id": "ws-1"}])
    await client.list_policy_assignment_workspaces(assignment_id=2)
    _assert_request_call(
        root._request.await_args,
        "GET",
        "/api/v1/mcp/hub/policy-assignments/2/workspaces",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"workspace_id": "ws-1"})
    await client.add_policy_assignment_workspace(
        assignment_id=2,
        payload={"workspace_id": "ws-1"},
    )
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/policy-assignments/2/workspaces",
        {"json_data": {"workspace_id": "ws-1"}},
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.delete_policy_assignment_workspace(assignment_id=2, workspace_id="ws-1")
    _assert_request_call(
        root._request.await_args,
        "DELETE",
        "/api/v1/mcp/hub/policy-assignments/2/workspaces/ws-1",
        {"params": None},
    )

    root._request = AsyncMock(return_value=[{"external_server_id": "docs"}])
    await client.list_profile_credential_bindings(profile_id=1)
    _assert_request_call(
        root._request.await_args,
        "GET",
        "/api/v1/mcp/hub/permission-profiles/1/credential-bindings",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"external_server_id": "docs"})
    await client.upsert_profile_credential_binding(
        profile_id=1,
        server_id="docs",
        payload={"managed_secret_ref_id": "secret-1"},
    )
    _assert_request_call(
        root._request.await_args,
        "PUT",
        "/api/v1/mcp/hub/permission-profiles/1/credential-bindings/docs",
        {"json_data": {"managed_secret_ref_id": "secret-1"}},
    )

    root._request = AsyncMock(return_value={"external_server_id": "docs", "slot_name": "token_readonly"})
    await client.upsert_profile_slot_credential_binding(
        profile_id=1,
        server_id="docs",
        slot_name="token_readonly",
        payload={"managed_secret_ref_id": "secret-1"},
    )
    _assert_request_call(
        root._request.await_args,
        "PUT",
        "/api/v1/mcp/hub/permission-profiles/1/credential-bindings/docs/token_readonly",
        {"json_data": {"managed_secret_ref_id": "secret-1"}},
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.delete_profile_credential_binding(profile_id=1, server_id="docs")
    _assert_request_call(
        root._request.await_args,
        "DELETE",
        "/api/v1/mcp/hub/permission-profiles/1/credential-bindings/docs",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.delete_profile_slot_credential_binding(
        profile_id=1,
        server_id="docs",
        slot_name="token_readonly",
    )
    _assert_request_call(
        root._request.await_args,
        "DELETE",
        "/api/v1/mcp/hub/permission-profiles/1/credential-bindings/docs/token_readonly",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"status": "configured"})
    await client.get_profile_slot_credential_status(
        profile_id=1,
        server_id="docs",
        slot_name="token_readonly",
    )
    _assert_request_call(
        root._request.await_args,
        "GET",
        "/api/v1/mcp/hub/permission-profiles/1/credential-bindings/status/docs/token_readonly",
        {"params": None},
    )

    root._request = AsyncMock(return_value=[{"external_server_id": "docs"}])
    await client.list_assignment_credential_bindings(assignment_id=2)
    _assert_request_call(
        root._request.await_args,
        "GET",
        "/api/v1/mcp/hub/policy-assignments/2/credential-bindings",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"external_server_id": "docs"})
    await client.upsert_assignment_credential_binding(
        assignment_id=2,
        server_id="docs",
        payload={"binding_mode": "grant", "managed_secret_ref_id": "secret-1"},
    )
    _assert_request_call(
        root._request.await_args,
        "PUT",
        "/api/v1/mcp/hub/policy-assignments/2/credential-bindings/docs",
        {"json_data": {"binding_mode": "grant", "managed_secret_ref_id": "secret-1"}},
    )

    root._request = AsyncMock(return_value={"external_server_id": "docs", "slot_name": "token_readonly"})
    await client.upsert_assignment_slot_credential_binding(
        assignment_id=2,
        server_id="docs",
        slot_name="token_readonly",
        payload={"binding_mode": "grant", "managed_secret_ref_id": "secret-1"},
    )
    _assert_request_call(
        root._request.await_args,
        "PUT",
        "/api/v1/mcp/hub/policy-assignments/2/credential-bindings/docs/token_readonly",
        {"json_data": {"binding_mode": "grant", "managed_secret_ref_id": "secret-1"}},
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.delete_assignment_credential_binding(assignment_id=2, server_id="docs")
    _assert_request_call(
        root._request.await_args,
        "DELETE",
        "/api/v1/mcp/hub/policy-assignments/2/credential-bindings/docs",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"ok": True})
    await client.delete_assignment_slot_credential_binding(
        assignment_id=2,
        server_id="docs",
        slot_name="token_readonly",
    )
    _assert_request_call(
        root._request.await_args,
        "DELETE",
        "/api/v1/mcp/hub/policy-assignments/2/credential-bindings/docs/token_readonly",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"status": "configured"})
    await client.get_assignment_slot_credential_status(
        assignment_id=2,
        server_id="docs",
        slot_name="token_readonly",
    )
    _assert_request_call(
        root._request.await_args,
        "GET",
        "/api/v1/mcp/hub/policy-assignments/2/credential-bindings/status/docs/token_readonly",
        {"params": None},
    )

    root._request = AsyncMock(return_value={"secret_ref_id": "secret-1"})
    await client.set_external_server_secret(
        server_id="docs",
        request=ExternalSecretSetRequest(secret="replace-me"),
    )
    _assert_request_call(
        root._request.await_args,
        "POST",
        "/api/v1/mcp/hub/external-servers/docs/secret",
        {"json_data": {"secret": "replace-me"}},
    )
