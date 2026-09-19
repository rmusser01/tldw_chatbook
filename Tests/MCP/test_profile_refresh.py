"""Refresh must cross the real stdio discovery boundary, not reuse its cache."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.MCP.client import MCPClient
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore


class RefreshPolicy:
    def __init__(self):
        self.denied = None
        self.actions = []

    def require_allowed(self, *, action_id, runtime_state_override):
        self.actions.append(action_id)
        if action_id == self.denied:
            raise PermissionError(action_id)


@asynccontextmanager
async def profile(tmp_path):
    state = tmp_path / "state.json"
    trace = tmp_path / "trace.jsonl"
    state.write_text(json.dumps({"version": "original"}))
    store = LocalMCPStore(tmp_path / "store.json")
    store.save_profile(
        LocalExternalMCPProfile(
            profile_id="catalog",
            command=sys.executable,
            args=(
                str(Path(__file__).parent / "fixtures/stdio_catalog_server.py"),
                str(state),
                str(trace),
            ),
        )
    )
    client = MCPClient()
    policy = RefreshPolicy()
    service = LocalMCPControlService(
        store=store,
        client=client,
        policy_enforcer=policy,
        manifest_provider=dict,
    )
    try:
        yield service, client, store, policy, state, trace
    finally:
        await client.disconnect_from_server("catalog")


def assert_catalog(snapshot, version):
    assert [tool["name"] for tool in snapshot["tools"]] == [f"{version}_tool"]
    assert [resource["uri"] for resource in snapshot["resources"]] == [
        f"fixture://{version}"
    ]
    assert [prompt["name"] for prompt in snapshot["prompts"]] == [f"{version}_prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("connected", [True, False])
@private_profile_test
async def test_refresh_discovers_changes_and_preserves_connection_state(
    request, tmp_path, connected
):
    async with profile(tmp_path) as (service, client, store, policy, state, trace):
        await service.connect_profile("catalog")
        original_process = client.sessions["catalog"].process
        if not connected:
            await service.disconnect_profile("catalog")
        state.write_text(json.dumps({"version": "updated"}))
        snapshot = await service.refresh_external_profile("catalog")
        assert_catalog(snapshot, "updated")
        assert_catalog(store.get_discovery_snapshot("catalog"), "updated")
        assert ("catalog" in client.sessions) is connected
        assert original_process.returncode is not None
        requests = [json.loads(line) for line in trace.read_text().splitlines()]
        for method in ("initialize", "tools/list", "resources/list", "prompts/list"):
            assert sum(row["method"] == method for row in requests) == 2
        assert policy.actions[-2:] == [
            "mcp.external_profiles.observe.local",
            "mcp.external_profiles.launch.local",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["observe", "launch"])
@private_profile_test
async def test_refresh_denial_preserves_live_session_and_catalog(
    request, tmp_path, action
):
    async with profile(tmp_path) as (service, client, store, policy, state, trace):
        before = await service.connect_profile("catalog")
        session = client.sessions["catalog"]
        wire_before = trace.read_text()
        state.write_text(json.dumps({"version": "updated"}))
        policy.denied = f"mcp.external_profiles.{action}.local"
        with pytest.raises(PermissionError, match=policy.denied):
            await service.refresh_external_profile("catalog")
        assert client.sessions["catalog"] is session
        assert session.process.returncode is None
        assert store.get_discovery_snapshot("catalog") == before
        assert trace.read_text() == wire_before


@pytest.mark.asyncio
@pytest.mark.parametrize("connected", [True, False])
@private_profile_test
async def test_failed_refresh_preserves_saved_catalog_and_can_retry(
    request, tmp_path, connected
):
    async with profile(tmp_path) as (service, client, store, _policy, state, _trace):
        before = await service.connect_profile("catalog")
        original_process = client.sessions["catalog"].process
        if not connected:
            await service.disconnect_profile("catalog")
        state.write_text(json.dumps({"version": "updated", "fail": True}))
        with pytest.raises(RuntimeError, match="Failed to connect profile"):
            await service.refresh_external_profile("catalog")
        assert store.get_discovery_snapshot("catalog") == before
        assert original_process.returncode is not None
        assert not client.sessions
        assert not client._pending_connections
        assert not client._connect_reservations
        state.write_text(json.dumps({"version": "recovered"}))
        assert_catalog(await service.refresh_external_profile("catalog"), "recovered")
        assert_catalog(store.get_discovery_snapshot("catalog"), "recovered")
        assert not client.sessions


@pytest.mark.asyncio
@private_profile_test
async def test_disconnected_refresh_cleans_up_if_saving_catalog_fails(
    request, tmp_path, monkeypatch
):
    async with profile(tmp_path) as (service, client, store, _policy, _state, _trace):

        def fail_save(*args):
            raise OSError("disk unavailable")

        monkeypatch.setattr(store, "save_discovery_snapshot", fail_save)
        with pytest.raises(OSError, match="disk unavailable"):
            await service.refresh_external_profile("catalog")
        assert not client.sessions
        assert not client._pending_connections
        assert not client._connect_reservations


@pytest.mark.asyncio
@pytest.mark.parametrize("denied", [False, True])
@private_profile_test
async def test_rejected_refresh_does_not_stop_another_pending_connection(
    request, tmp_path, denied
):
    async with profile(tmp_path) as (service, client, _store, policy, state, trace):
        state.write_text(json.dumps({"version": "original", "hold": True}))
        connecting = asyncio.create_task(service.connect_profile("catalog"))
        try:
            async with asyncio.timeout(5):
                while not (trace.exists() and '"initialize"' in trace.read_text()):
                    await asyncio.sleep(0.01)
            pending = client._pending_connections["catalog"]
            assert not client.sessions
            if denied:
                policy.denied = "mcp.external_profiles.launch.local"
            with pytest.raises(PermissionError if denied else RuntimeError):
                await service.refresh_external_profile("catalog")
            assert client._pending_connections.get("catalog") is pending
            assert pending.process.returncode is None
        finally:
            # The fixture is polling this file; publish a complete JSON value.
            replacement = state.with_suffix(".next")
            replacement.write_text(json.dumps({"version": "original"}))
            replacement.replace(state)
            policy.denied = None
            outcome = await asyncio.gather(connecting, return_exceptions=True)
        assert_catalog(outcome[0], "original")
        assert "catalog" in client.sessions
