"""Isolated service decisions complement the real stdio refresh journeys."""

import asyncio
from typing import Any

import pytest

from Tests.MCP.test_local_control_service import FakeLocalStore, FakeMCPClient
from Tests.MCP.test_profile_refresh import RefreshPolicy
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile


@pytest.fixture
def service_parts() -> tuple[
    LocalMCPControlService, FakeMCPClient, FakeLocalStore, RefreshPolicy
]:
    """Keep the service real while isolating transport, persistence and policy."""
    store = FakeLocalStore()
    store.profiles["profile-a"] = LocalExternalMCPProfile(
        profile_id="profile-a", command="python", args=("-m", "demo.server")
    )
    store.save_discovery_snapshot("profile-a", {"tools": [{"name": "saved_tool"}]})
    client = FakeMCPClient()
    policy = RefreshPolicy()
    service = LocalMCPControlService(
        store=store, client=client, policy_enforcer=policy, manifest_provider=dict
    )
    return service, client, store, policy


@pytest.mark.asyncio
@pytest.mark.parametrize("connected", [True, False])
async def test_refresh_unit_persists_new_catalog_and_restores_state(
    service_parts: tuple, connected: bool
) -> None:
    """A refresh must save discovery and retain only an originally live session."""
    service, client, store, _policy = service_parts
    if connected:
        await service.connect_profile("profile-a")
    original = client.sessions.get("profile-a")
    snapshot = await service.refresh_external_profile("profile-a")
    assert snapshot["tools"] == [{"name": "remote_tool"}]
    assert store.get_discovery_snapshot("profile-a")["tools"] == [
        {"name": "remote_tool"}
    ]
    assert ("profile-a" in client.sessions) is connected
    if connected:
        assert client.sessions["profile-a"] is not original


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["observe", "launch"])
async def test_refresh_unit_permission_denial_preserves_session_and_catalog(
    service_parts: tuple, action: str
) -> None:
    """Either gate must refuse before replacing transport or saved discovery."""
    service, client, store, policy = service_parts
    await service.connect_profile("profile-a")
    original = client.sessions["profile-a"]
    before = store.get_discovery_snapshot("profile-a")
    policy.denied = f"mcp.external_profiles.{action}.local"
    with pytest.raises(PermissionError, match=policy.denied):
        await service.refresh_external_profile("profile-a")
    assert client.sessions["profile-a"] is original
    assert store.get_discovery_snapshot("profile-a") == before


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["discovery", "empty", "save"])
@pytest.mark.parametrize("replacement", [True, False])
async def test_refresh_unit_failure_cleans_only_the_established_session(
    service_parts: tuple,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    replacement: bool,
) -> None:
    """Failure preserves prior discovery and never closes a replacement owner."""
    service, client, store, _policy = service_parts
    before = store.get_discovery_snapshot("profile-a")
    other = object()

    def replace_if_requested() -> None:
        if replacement:
            client.sessions["profile-a"] = other

    async def fail_discovery(profile_id: str) -> dict[str, Any]:
        replace_if_requested()
        if failure == "empty":
            return {"tools": [], "resources": [], "prompts": []}
        raise RuntimeError("discovery unavailable")

    def fail_save(profile_id: str, snapshot: dict[str, Any]) -> None:
        replace_if_requested()
        raise OSError("disk unavailable")

    if failure == "save":
        monkeypatch.setattr(store, "save_discovery_snapshot", fail_save)
    else:
        monkeypatch.setattr(client, "describe_server", fail_discovery)
    with pytest.raises(OSError if failure == "save" else RuntimeError):
        await service.refresh_external_profile("profile-a")
    assert store.get_discovery_snapshot("profile-a") == before
    if replacement:
        assert client.sessions["profile-a"] is other
    else:
        assert "profile-a" not in client.sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["false", "exception", "cancelled"])
async def test_refresh_unit_does_not_report_success_with_retained_temporary_session(
    service_parts: tuple, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Failed temporary cleanup is visible; cancellation keeps its precedence."""
    service, client, store, _policy = service_parts

    async def fail_disconnect(profile_id: str) -> bool:
        if failure == "exception":
            raise OSError("process cleanup unavailable")
        if failure == "cancelled":
            raise asyncio.CancelledError
        return False

    monkeypatch.setattr(client, "disconnect_from_server", fail_disconnect)
    expected = asyncio.CancelledError if failure == "cancelled" else RuntimeError
    with pytest.raises(expected):
        await service.refresh_external_profile("profile-a")
    assert "profile-a" in client.sessions
    assert store.get_discovery_snapshot("profile-a")["tools"] == [
        {"name": "remote_tool"}
    ]


@pytest.mark.asyncio
async def test_refresh_unit_cleanup_postcondition_allows_a_replacement_owner(
    service_parts: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newer owner must not be mistaken for failed temporary-session cleanup."""
    service, client, _store, _policy = service_parts
    replacement = object()

    async def replace_during_disconnect(profile_id: str) -> bool:
        client.sessions[profile_id] = replacement
        return False

    monkeypatch.setattr(client, "disconnect_from_server", replace_during_disconnect)
    assert (await service.refresh_external_profile("profile-a"))["tools"] == [
        {"name": "remote_tool"}
    ]
    assert client.sessions["profile-a"] is replacement
