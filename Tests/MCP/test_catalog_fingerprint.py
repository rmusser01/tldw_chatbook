"""MCP stores identify what a catalog composition would read (TASK-33620.15.1).

The Console reuses a run's composed MCP catalog only while each store's
``catalog_fingerprint`` -- read through the same admission scope as
``load()`` -- is unchanged. These pin the identity itself on real stores:
equal for unchanged content, different for any change, and a distinct value
when ``load()`` would read no file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore
from tldw_chatbook.MCP.permission_store import MCPPermissionStore


def test_permission_store_fingerprint_follows_its_content(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    assert store.catalog_fingerprint() == ("fresh",)
    store.set_kill_switch(True)
    on = store.catalog_fingerprint()
    assert on[0] == "file" and len(on[1]) == 64
    assert store.catalog_fingerprint()[:2] == on[:2]
    store.set_kill_switch(False)
    assert store.catalog_fingerprint()[:2] != on[:2]


def test_local_store_fingerprint_follows_its_content(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    assert store.catalog_fingerprint() == ("missing",)
    store.save_profile(
        LocalExternalMCPProfile.from_input_dict(
            {"profile_id": "one", "command": "echo", "args": ["hi"]}
        )
    )
    saved = store.catalog_fingerprint()
    assert saved[0] == "file"
    assert store.catalog_fingerprint()[:2] == saved[:2]
    store.save_discovery_snapshot("one", {"tools": [{"name": "ping"}]})
    assert store.catalog_fingerprint()[:2] != saved[:2]


def test_service_fingerprint_reports_connections_and_governance(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    service = LocalMCPControlService(store=store, manifest_provider=lambda: {"tools": []})
    profiles = (("one", False), ("two", False))
    service.client = SimpleNamespace(sessions={"one": SimpleNamespace(_closed=False)})
    _store, manifest, connected = service.catalog_fingerprint(profiles)
    assert manifest == ("manifest", {"tools": []})
    assert connected == (True, False)
    service.client.sessions["one"]._closed = True
    assert service.catalog_fingerprint(profiles)[2] == (False, False)

    class Refusing:
        def require_allowed(self, **_kwargs):
            raise PermissionError("governance refused")

    service.policy_enforcer = Refusing()
    with pytest.raises(PermissionError):
        service.catalog_fingerprint(profiles)
