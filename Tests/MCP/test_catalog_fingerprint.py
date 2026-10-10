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


#: The governance checks a composition's two reads make, one per read.
GOVERNANCE_CHECKS = ("mcp.external_profiles.list.local", "mcp.inventory.list.local")


class RefusingOne:
    """A runtime policy that refuses exactly one action."""

    def __init__(self, refused: str | None = None) -> None:
        self.refused = refused

    def require_allowed(self, *, action_id, **_kwargs):
        if action_id == self.refused:
            raise PermissionError(f"governance refused {action_id}")


@pytest.mark.parametrize("refused", GOVERNANCE_CHECKS)
def test_service_fingerprint_makes_each_governance_check(tmp_path, refused):
    """Either list refused at runtime fails the fingerprint on its own.

    Governance decisions come from the runtime policy context, not from the
    hashed stores, so these checks are all that stops a reuse after a deny.
    """
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    policy = RefusingOne()
    service = LocalMCPControlService(
        store=store, manifest_provider=lambda: {"tools": []}, policy_enforcer=policy
    )
    service.catalog_fingerprint(())  # Allowed: nothing refused yet.
    policy.refused = refused
    with pytest.raises(PermissionError, match=refused):
        service.catalog_fingerprint(())
