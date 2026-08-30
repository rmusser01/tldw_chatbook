"""Provisioner: convenience auto-create is reference-backed and non-fatal."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.Workspaces.agent_provisioning import (
    WorkspaceAgentProvisioner,
    run_workspace_agent_backfill,
)
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


class StubPersonaService:
    def __init__(self):
        self.created = []

    def create_persona_profile(self, payload):
        payload = dict(payload)
        payload.setdefault("id", f"local-persona-{len(self.created) + 1}")
        self.created.append(payload)
        return payload


def build(tmp_path: Path, personas=None):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    provisioner = (
        WorkspaceAgentProvisioner(personas, store) if personas is not None else None
    )
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="c1"),
        agent_provisioner=provisioner.provision if provisioner is not None else None,
    )
    return registry, store


def test_provision_creates_persona_profile_and_reference(tmp_path):
    personas = StubPersonaService()
    registry, store = build(tmp_path, personas)
    record = registry.create_workspace(workspace_id="w-2", name="Research")
    assert personas.created and personas.created[0]["name"] == "Research Agent"
    assert record.assistant_defaults is not None
    assert record.assistant_defaults.tool_policy_profile_id == "ws-w-2"
    assert "ws-w-2" in store.list_profiles()


def test_provision_failure_is_non_fatal(tmp_path):
    class Broken:
        def create_persona_profile(self, payload):
            raise RuntimeError("boom")

    registry, _store = build(tmp_path, Broken())
    record = registry.create_workspace(workspace_id="w-3", name="W3")
    assert record.assistant_defaults is None


def test_backfill_skips_archived_and_default_and_is_idempotent(tmp_path):
    registry, store = build(tmp_path)
    personas = StubPersonaService()
    provisioner = WorkspaceAgentProvisioner(personas, store)
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="w-4", name="Keep")
    registry.create_workspace(workspace_id="w-5", name="Skip")
    registry.archive_workspace("w-5")
    first = run_workspace_agent_backfill(registry=registry, provisioner=provisioner)
    assert first == 1
    assert registry.get_workspace("w-4").assistant_defaults is not None
    assert registry.get_workspace("w-5").assistant_defaults is None
    assert run_workspace_agent_backfill(registry=registry, provisioner=provisioner) == 0


def test_backfill_retries_when_persist_fails(tmp_path):
    """A failed persist must not mark the backfill complete (Qodo #1).

    Transient persistence failures previously still set the completion
    flag, permanently leaving that workspace without defaults. The flag
    must stay unset so the next startup retries; the retry (with the
    stub fixed) then completes.
    """
    from tldw_chatbook.Workspaces.registry_service import (
        WorkspaceRegistryServiceError,
    )

    class FlakyPersistRegistry(LocalWorkspaceRegistryService):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fail_next = False

        def set_assistant_defaults(self, *args, **kwargs):
            if self.fail_next:
                self.fail_next = False
                raise WorkspaceRegistryServiceError("persist boom")
            return super().set_assistant_defaults(*args, **kwargs)

    personas = StubPersonaService()
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    provisioner = WorkspaceAgentProvisioner(personas, store)
    registry = FlakyPersistRegistry(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="c1"),
    )
    registry.create_workspace(workspace_id="w-8", name="Retry Me")

    registry.fail_next = True
    assert run_workspace_agent_backfill(registry=registry, provisioner=provisioner) == 0
    assert registry.db.is_agent_backfill_complete() is False

    second = run_workspace_agent_backfill(registry=registry, provisioner=provisioner)
    assert second == 1
    assert registry.get_workspace("w-8").assistant_defaults is not None
    assert registry.db.is_agent_backfill_complete() is True


def test_default_workspace_is_not_provisioned(tmp_path):
    """The ctor hook must not seed an agent onto the built-in Default workspace.

    ``ensure_default_workspace`` reaches ``create_workspace`` for the Default
    workspace whenever its row is missing, and the backfill deliberately skips
    Default -- the create-time hook must agree (task-8).
    """
    personas = StubPersonaService()
    registry, store = build(tmp_path, personas)
    registry.ensure_default_workspace()
    default_record = registry.get_workspace(DEFAULT_WORKSPACE_ID)
    assert default_record is not None
    assert default_record.assistant_defaults is None
    assert personas.created == []


def test_set_agent_provisioner_wires_post_construction(tmp_path):
    """app.py wires the registry before persona services exist (task-8).

    The hook therefore has to be attachable after construction; creations
    after the attach point are provisioned, before are not.
    """
    personas = StubPersonaService()
    registry, store = build(tmp_path)
    registry.create_workspace(workspace_id="w-6", name="Before Wire")
    provisioner = WorkspaceAgentProvisioner(personas, store)
    registry.set_agent_provisioner(provisioner.provision)
    wired = registry.create_workspace(workspace_id="w-7", name="After Wire")
    assert wired.assistant_defaults is not None
    assert wired.assistant_defaults.tool_policy_profile_id == "ws-w-7"
    assert [payload["name"] for payload in personas.created] == ["After Wire Agent"]
