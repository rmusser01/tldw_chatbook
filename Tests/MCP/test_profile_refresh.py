"""Refresh must cross the real stdio discovery boundary, not reuse its cache."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import copy_context
from pathlib import Path
from typing import Any

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.MCP.client import MCPClient
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore


class RefreshPolicy:
    """Record real service gate checks and deny only the selected action."""

    def __init__(self) -> None:
        self.denied: str | None = None
        self.actions: list[str] = []

    def require_allowed(self, *, action_id: str, runtime_state_override: Any) -> None:
        """Apply the fixture policy at the real service boundary."""
        self.actions.append(action_id)
        if action_id == self.denied:
            raise PermissionError(action_id)


@asynccontextmanager
async def profile(
    tmp_path: Path,
) -> AsyncIterator[
    tuple[LocalMCPControlService, MCPClient, LocalMCPStore, RefreshPolicy, Path, Path]
]:
    """Yield a real private store/client and reap its owned stdio session."""
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
                str(tmp_path.resolve()),
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


def assert_catalog(snapshot: dict[str, Any], version: str) -> None:
    """Check each catalog section against the fixture's independent wire values."""
    assert [tool["name"] for tool in snapshot["tools"]] == [f"{version}_tool"]
    assert [resource["uri"] for resource in snapshot["resources"]] == [
        f"fixture://{version}"
    ]
    assert [prompt["name"] for prompt in snapshot["prompts"]] == [f"{version}_prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("connected", [True, False])
@private_profile_test
async def test_refresh_discovers_changes_and_preserves_connection_state(
    request: pytest.FixtureRequest, tmp_path: Path, connected: bool
) -> None:
    """Fresh discovery crosses stdio and preserves the original connection state."""
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
    request: pytest.FixtureRequest, tmp_path: Path, action: str
) -> None:
    """A denied refresh leaves both the live transport and saved catalog untouched."""
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
    request: pytest.FixtureRequest, tmp_path: Path, connected: bool
) -> None:
    """A failed reconnect preserves saved discovery and admits a clean retry."""
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
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed persistence must not leak the temporary discovered session."""
    async with profile(tmp_path) as (service, client, store, _policy, _state, _trace):

        def fail_save(*args: Any) -> None:
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
    request: pytest.FixtureRequest, tmp_path: Path, denied: bool
) -> None:
    """A rejected refresh cannot close another caller's in-progress connection."""
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


@pytest.mark.asyncio
@private_profile_test
async def test_refresh_cleanup_preserves_a_concurrent_replacement(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection queued during save survives the refresh's owned teardown."""
    async with profile(tmp_path) as (service, client, store, _policy, _state, _trace):
        replacement: asyncio.Task | None = None
        temporary = None
        save = store.save_discovery_snapshot
        # A separate caller must acquire its own activation, not inherit the
        # in-flight refresh's task-bound storage lease through create_task.
        caller_context = copy_context()

        def save_and_queue_connection(profile_id: str, snapshot: dict[str, Any]) -> Any:
            nonlocal replacement, temporary
            result = save(profile_id, snapshot)
            if replacement is None:
                temporary = client.sessions[profile_id]
                replacement = asyncio.create_task(
                    service.connect_profile(profile_id), context=caller_context
                )
            return result

        monkeypatch.setattr(store, "save_discovery_snapshot", save_and_queue_connection)
        try:
            assert_catalog(
                await service.refresh_external_profile("catalog"), "original"
            )
        finally:
            # Join the second real connection even if refresh itself fails.
            if replacement is not None:
                await asyncio.wait_for(replacement, timeout=10)
        assert temporary is not None
        assert temporary.process.returncode is not None
        current = client.sessions["catalog"]
        assert current is not temporary
        assert current.process.returncode is None
        assert_catalog(await client.describe_server("catalog"), "original")
