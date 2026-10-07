from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)


class FakeLocalService:
    def __init__(
        self,
        store: LocalMCPStore,
        *,
        connect_delay: float = 0.0,
        connect_error: Exception | None = None,
    ) -> None:
        self.store = store
        self.connect_delay = connect_delay
        self.connect_error = connect_error
        self.calls: list[tuple[str, str]] = []

    def get_external_servers(self):
        return [
            {
                "profile_id": "docs",
                "command": "python",
                "args": [],
                "env_placeholders": {},
                "discovery_snapshot": None,
                "is_connected": False,
            }
        ]

    async def run_action(self, name, payload):  # matches control-plane delegation
        raise AssertionError("typed methods must not route through run_action in tests")

    def save_external_profile(self, payload):
        self.calls.append(("save", str(payload.get("profile_id"))))
        return dict(payload)

    def delete_external_profile(self, profile_id):
        self.calls.append(("delete", profile_id))
        return True

    async def connect_profile(self, profile_id):
        self.calls.append(("connect", profile_id))
        if self.connect_delay:
            await asyncio.sleep(self.connect_delay)
        if self.connect_error:
            raise self.connect_error
        return {
            "server_id": profile_id,
            "tools": [{"name": "a"}],
            "resources": [],
            "prompts": [],
        }

    async def disconnect_profile(self, profile_id):
        self.calls.append(("disconnect", profile_id))
        return True

    async def test_external_profile(self, profile_id):
        self.calls.append(("test", profile_id))
        return {
            "ok": True,
            "profile_id": profile_id,
            "tools": 1,
            "resources": 0,
            "prompts": 0,
        }

    async def refresh_external_profile(self, profile_id):
        self.calls.append(("refresh", profile_id))
        return {"server_id": profile_id, "tools": [], "resources": [], "prompts": []}


def _service(
    tmp_path: Path, **fake_kwargs
) -> tuple[UnifiedMCPControlPlaneService, FakeLocalService, LocalMCPStore]:
    store = LocalMCPStore(tmp_path / "store.json")
    fake = FakeLocalService(store, **fake_kwargs)
    service = UnifiedMCPControlPlaneService(
        local_service=fake, server_service=None, target_store=None, context_store=None
    )
    return service, fake, store


@pytest.mark.asyncio
async def test_connect_success_records_ok(tmp_path, monkeypatch):
    service, fake, store = _service(tmp_path)
    result = await service.connect_local_profile("docs")
    assert result["server_id"] == "docs"
    record = store.get_profile_runtime_state("docs")
    assert record["ok"] is True and record["last_error"] is None
    assert record["last_action"] == "connect" and record["last_ok_at"]


@pytest.mark.asyncio
async def test_connect_failure_records_error_and_reraises(tmp_path):
    service, fake, store = _service(
        tmp_path, connect_error=RuntimeError("spawn failed")
    )
    with pytest.raises(RuntimeError, match="spawn failed"):
        await service.connect_local_profile("docs")
    record = store.get_profile_runtime_state("docs")
    assert record["ok"] is False and "spawn failed" in record["last_error"]


@pytest.mark.asyncio
async def test_connect_timeout_records_and_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.MCP.unified_control_plane_service.get_cli_setting",
        lambda section, key, default=None: 0.05,
    )
    service, fake, store = _service(tmp_path, connect_delay=1.0)
    with pytest.raises(RuntimeError, match="Timed out"):
        await service.connect_local_profile("docs")
    record = store.get_profile_runtime_state("docs")
    assert record["ok"] is False and "Timed out" in record["last_error"]


@pytest.mark.asyncio
async def test_local_external_catalog_merges_runtime_state(tmp_path):
    service, fake, store = _service(tmp_path)
    store.save_profile_runtime_state("docs", {"ok": False, "last_error": "boom"})
    catalog = await service.local_external_catalog()
    assert catalog[0]["profile_id"] == "docs"
    assert catalog[0]["runtime_state"]["last_error"] == "boom"


@pytest.mark.asyncio
async def test_save_and_delete_delegate(tmp_path):
    service, fake, store = _service(tmp_path)
    saved = await service.save_local_profile({"profile_id": "x", "command": "y"})
    assert saved["profile_id"] == "x"
    assert await service.delete_local_profile("x") is True
    assert ("save", "x") in fake.calls and ("delete", "x") in fake.calls


class _RaisingStore(LocalMCPStore):
    def save_profile_runtime_state(self, profile_id, record):
        raise OSError("disk full")


def _service_with_raising_store(
    tmp_path: Path, **fake_kwargs
) -> tuple[UnifiedMCPControlPlaneService, FakeLocalService, LocalMCPStore]:
    store = _RaisingStore(tmp_path / "store.json")
    fake = FakeLocalService(store, **fake_kwargs)
    service = UnifiedMCPControlPlaneService(
        local_service=fake, server_service=None, target_store=None, context_store=None
    )
    return service, fake, store


@pytest.mark.asyncio
async def test_record_failure_does_not_mask_success_result(tmp_path):
    service, fake, store = _service_with_raising_store(tmp_path)
    result = await service.connect_local_profile("docs")
    assert result["server_id"] == "docs"
    assert ("connect", "docs") in fake.calls


@pytest.mark.asyncio
async def test_record_failure_does_not_mask_original_error(tmp_path):
    service, fake, store = _service_with_raising_store(
        tmp_path, connect_error=RuntimeError("spawn failed")
    )
    with pytest.raises(RuntimeError, match="spawn failed"):
        await service.connect_local_profile("docs")


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "rpc_error", "hang"])
async def test_actual_http_profile_uses_existing_service_audit_owner(
    tmp_path, monkeypatch, fault
):
    from Tests.MCP.test_streamable_http import peer
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile

    async with peer(monkeypatch, fault=fault) as (url, _messages, calls):
        store = LocalMCPStore(tmp_path / "http-store.json")
        store.save_profile(
            LocalExternalMCPProfile(
                profile_id="owned",
                transport="streamable_http",
                protocol_version="2026-07-28",
                url=url,
                development_loopback=True,
            )
        )
        client = MCPClient()
        local = LocalMCPControlService(
            store=store, client=client, manifest_provider=dict
        )
        service = UnifiedMCPControlPlaneService(
            local_service=local,
            server_service=None,
            target_store=None,
            context_store=None,
        )
        try:
            snapshot = await local.connect_profile("owned")
            assert snapshot["tools"][0]["name"] == "echo"
            result = await service.execute_hub_tool_result(
                "local:owned",
                "echo",
                {},
                timeout_seconds=0.05 if fault == "hang" else 2,
            )
            assert calls["count"] == 1
            assert result.dispatch_state == (
                "uncertain" if fault == "hang" else "settled"
            )
            if fault is None:
                assert result.structured_content == {"pass": True}
                assert result.duplicate_keys_checked
            else:
                assert result.transport_error
                assert "private sentinel" not in str(result)
            rows = service.execution_log.read_recent()
            assert len(rows) == 1
            assert rows[0]["status"] == (
                "success"
                if fault is None
                else "timeout" if fault == "hang" else "error"
            )
        finally:
            await client.disconnect_all()


@pytest.mark.asyncio
async def test_actual_http_auth_challenge_has_explicit_service_diagnostic(
    tmp_path, monkeypatch
):
    from Tests.MCP.test_streamable_http import peer
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile

    async with peer(monkeypatch, fault="auth") as (url, _messages, calls):
        store = LocalMCPStore(tmp_path / "store.json")
        store.save_profile(
            LocalExternalMCPProfile(
                profile_id="owned",
                transport="streamable_http",
                url=url,
                protocol_version="2026-07-28",
                development_loopback=True,
            )
        )
        client = MCPClient()
        service = LocalMCPControlService(
            store=store, client=client, manifest_provider=dict
        )
        with pytest.raises(RuntimeError, match="^mcp_authentication_unsupported$"):
            await service.connect_profile("owned")
        assert not client.sessions and not client._pending_connections
        assert calls["count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"env_placeholders": {"API_KEY": "$M2_MISSING_REVIEW_KEY"}},
        {"env_literals": {"LOG_LEVEL": "debug"}},
        {"legacy_env_literals": {"MODE": "private"}},
        {"env": {"API_KEY": "$M2_MISSING_REVIEW_KEY"}},
        {"env_placeholders": {"": ""}},
        {"env_literals": ["discarded"]},
        {"legacy_env_literals": "discarded"},
        {"env": ["discarded"]},
        {"headers": {"Authorization": "private sentinel"}},
        {"auth": {}},
        {"credentials": None},
        {"command": "   "},
        {"args": ["   "]},
        {"args": "discarded"},
    ],
)
async def test_reopened_http_unsupported_fields_never_dispatch(
    tmp_path, monkeypatch, fields
):
    import json

    from Tests.MCP.test_streamable_http import peer
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import (
        LocalExternalMCPProfile,
        LocalMCPStoreLoadError,
    )

    async with peer(monkeypatch) as (url, messages, calls):
        path = tmp_path / "reopened-http.json"
        store = LocalMCPStore(path)
        store.save_profile(
            LocalExternalMCPProfile(
                profile_id="owned",
                transport="streamable_http",
                url=url,
                protocol_version="2026-07-28",
                development_loopback=True,
            )
        )
        valid_bytes = path.read_bytes()
        client = MCPClient()
        try:
            valid = LocalMCPControlService(
                store=LocalMCPStore(path), client=client, manifest_provider=dict
            )
            assert (await valid.connect_profile("owned"))["tools"][0]["name"] == "echo"
            await client.disconnect_all()
            payload = json.loads(valid_bytes)
            payload["profiles"][0].update(fields)
            invalid_bytes = json.dumps(payload).encode()
            path.write_bytes(invalid_bytes)
            reopened = LocalMCPControlService(
                store=LocalMCPStore(path), client=client, manifest_provider=dict
            )
            before = len(messages)
            with pytest.raises(LocalMCPStoreLoadError, match="mcp_store_invalid"):
                await reopened.connect_profile("owned")
            assert len(messages) == before and calls["count"] == 0
            assert not client.sessions and not client._pending_connections
            assert path.read_bytes() == invalid_bytes
            path.write_bytes(valid_bytes)
            restored = LocalMCPControlService(
                store=LocalMCPStore(path), client=client, manifest_provider=dict
            )
            assert (await restored.connect_profile("owned"))["tools"][0][
                "name"
            ] == "echo"
        finally:
            await client.disconnect_all()
