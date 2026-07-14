from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService


class FakeLocalService:
    def __init__(self, store: LocalMCPStore, *, connect_delay: float = 0.0,
                 connect_error: Exception | None = None) -> None:
        self.store = store
        self.connect_delay = connect_delay
        self.connect_error = connect_error
        self.calls: list[tuple[str, str]] = []

    def get_external_servers(self):
        return [
            {"profile_id": "docs", "command": "python", "args": [],
             "env_placeholders": {}, "discovery_snapshot": None, "is_connected": False}
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
        return {"server_id": profile_id, "tools": [{"name": "a"}], "resources": [], "prompts": []}

    async def disconnect_profile(self, profile_id):
        self.calls.append(("disconnect", profile_id))
        return True

    async def test_external_profile(self, profile_id):
        self.calls.append(("test", profile_id))
        return {"ok": True, "profile_id": profile_id, "tools": 1, "resources": 0, "prompts": 0}

    async def refresh_external_profile(self, profile_id):
        self.calls.append(("refresh", profile_id))
        return {"server_id": profile_id, "tools": [], "resources": [], "prompts": []}


def _service(tmp_path: Path, **fake_kwargs) -> tuple[UnifiedMCPControlPlaneService, FakeLocalService, LocalMCPStore]:
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
    service, fake, store = _service(tmp_path, connect_error=RuntimeError("spawn failed"))
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
