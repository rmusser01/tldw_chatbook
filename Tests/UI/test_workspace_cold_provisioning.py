"""Cold workspace provisioning crosses the real deferred profile guard."""

import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest
from loguru import logger
from textual.widgets import Button, Input, Select

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.Tool_Packs.catalog_snapshot import PermissionInventoryRegistry
from tldw_chatbook.Tool_Packs.service import ToolPackService
from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateModal
from tldw_chatbook.Widgets.workspace_persona_default import WorkspacePersonaChoice
from tldw_chatbook.Workspaces.registry_service import (
    DeferredWorkspaceToolProfileGuard,
    LocalWorkspaceRegistryService,
)


class Host(ConsolidatedCSSApp):
    CSS_PATH: ClassVar = [BUNDLED_STYLESHEET]

    def __init__(self, root):
        super().__init__()
        from tldw_chatbook.app import TldwCli

        self.root = root
        self.loguru_logger = logger
        self._tool_pack_guard_bootstrap = DeferredWorkspaceToolProfileGuard()
        self.workspace_registry_service = LocalWorkspaceRegistryService(
            WorkspaceDB(root / "workspace.sqlite", client_id="cold-test"),
        )
        self.workspace_registry_service.attach_tool_profile_guard(
            self._tool_pack_guard_bootstrap
        )
        self.local_character_persona_service = LocalCharacterPersonaService(
            None, persona_store_path=root / "personas.json"
        )
        self.unified_mcp_service = SimpleNamespace(
            permission_store=MCPPermissionStore(root / "permissions.json")
        )
        self.tool_pack_service = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.compositions = 0
        self.fail_composition = False
        for name in (
            "_wire_workspace_agent_provisioning",
            "_deferred_wire_workspace_agent_provisioning",
            "ensure_workspace_agent_provisioning",
        ):
            method = getattr(TldwCli, name, None)
            if method is not None:
                setattr(self, name, method.__get__(self))

    def _deferred_wire_tool_pack_service(self):
        if self.tool_pack_service is not None:
            return None
        if getattr(self, "_composition", None) is None:
            self._composition = self.run_worker(self._compose(), exit_on_error=False)
        return self._composition

    async def _compose(self):
        self.compositions += 1
        self.started.set()
        await self.release.wait()
        if self.fail_composition:
            return
        self.tool_pack_service = ToolPackService.compose(
            permission_store=self.unified_mcp_service.permission_store,
            inventory=PermissionInventoryRegistry(current_permission_namespaces=set),
            workspace_registry=self.workspace_registry_service,
            receipt_root=self.root / "receipts",
        )
        self._tool_pack_guard_bootstrap.activate(self.tool_pack_service.binding_guard)

    def personas(self):
        return self.local_character_persona_service.list_persona_profiles()


async def settled(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
@private_profile_test
async def test_cold_wiring_creates_no_orphans_before_authority(request, tmp_path):
    host = Host(tmp_path)
    registry = host.workspace_registry_service
    try:
        registry.create_workspace(workspace_id="old", name="Old")
        host._wire_workspace_agent_provisioning()
        assert host.personas() == []
        assert not registry.db.is_agent_backfill_complete()
        assert registry.get_workspace("old").assistant_defaults is None
    finally:
        registry.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("eligible", [False, True])
@private_profile_test
async def test_startup_initializes_only_for_eligible_backfill(
    request, tmp_path, eligible
):
    host = Host(tmp_path)
    registry = host.workspace_registry_service
    try:
        registry.ensure_default_workspace()
        registry.create_workspace(
            workspace_id="none", name="None", assistant_defaults=None
        )
        if eligible:
            registry.create_workspace(workspace_id="old", name="Old")
        async with host.run_test() as pilot:
            host._deferred_wire_workspace_agent_provisioning()
            await pilot.pause()
            assert host.compositions == int(eligible)
            assert host.personas() == []
            host.release.set()
            if eligible:
                await settled(lambda: registry.get_workspace("old").assistant_defaults)
                assert len(host.personas()) == 1
                assert registry.db.is_agent_backfill_complete()
            assert registry.get_workspace("none").assistant_defaults is None
    finally:
        registry.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["save", "cancel", "unavailable"])
@private_profile_test
async def test_cold_create_waits_without_duplicate_or_late_commits(
    request, tmp_path, outcome
):
    host = Host(tmp_path)
    registry = host.workspace_registry_service
    host.fail_composition = outcome == "unavailable"
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            # This is the same post-ready wiring that runs before a cold UI visit.
            host._wire_workspace_agent_provisioning()
            await host.push_screen(
                WorkspaceCreateModal(
                    registry_service=registry,
                    persona_service=host.local_character_persona_service,
                )
            )
            modal = host.screen
            modal.query_one("#workspace-create-name", Input).value = "Cold"
            modal._create()
            modal._create()
            await pilot.pause()
            assert registry.list_workspaces() == ()
            assert host.personas() == []
            assert host.compositions == 1
            if outcome == "cancel":
                await pilot.press("escape")
                await pilot.pause()
                assert host.screen is not modal
            host.release.set()
            if outcome != "cancel":
                await settled(lambda: len(registry.list_workspaces()) == 1)
                saved = registry.list_workspaces()[0]
                if outcome == "save":
                    assert saved.assistant_defaults is not None
                    assert (
                        saved.assistant_defaults.assistant_id
                        == host.personas()[0]["id"]
                    )
                    assert (
                        saved.assistant_defaults.tool_policy_profile_id
                        in host.unified_mcp_service.permission_store.list_profiles()
                    )
                    reopened = WorkspaceDB(
                        tmp_path / "workspace.sqlite", client_id="reopen"
                    )
                    try:
                        assert (
                            LocalWorkspaceRegistryService(reopened)
                            .get_workspace(saved.workspace_id)
                            .assistant_defaults
                            == saved.assistant_defaults
                        )
                    finally:
                        reopened.close()
                else:
                    assert saved.assistant_defaults is None
                    assert host.personas() == []
            else:
                await settled(lambda: host.tool_pack_service is not None)
                assert registry.list_workspaces() == ()
                assert host.personas() == []
    finally:
        registry.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [WorkspacePersonaChoice.NONE, "saved"])
@private_profile_test
async def test_explicit_choice_does_not_initialize_automatic_provisioning(
    request, tmp_path, choice
):
    host = Host(tmp_path)
    registry = host.workspace_registry_service
    host.local_character_persona_service.create_persona_profile(
        {"id": "saved", "name": "Saved", "system_prompt": "Help."}
    )
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            await host.push_screen(
                WorkspaceCreateModal(
                    registry_service=registry,
                    persona_service=host.local_character_persona_service,
                )
            )
            host.screen.query_one("#workspace-default-persona", Select).value = choice
            host.screen.query_one("#workspace-create-confirm", Button).press()
            await pilot.pause()
            assert host.compositions == 0
            saved = registry.list_workspaces()[0]
            if choice is WorkspacePersonaChoice.NONE:
                assert saved.assistant_defaults is None
                assert saved.assistant_defaults_explicit_none
            else:
                assert saved.assistant_defaults.assistant_id == "saved"
            assert len(host.personas()) == 1
    finally:
        registry.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["nested_screen", "explicit_none"])
@private_profile_test
async def test_pending_creation_revalidates_ownership_and_form(
    request, tmp_path, change
):
    from textual.screen import Screen

    host = Host(tmp_path)
    registry = host.workspace_registry_service
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            await host.push_screen(
                WorkspaceCreateModal(
                    registry_service=registry,
                    persona_service=host.local_character_persona_service,
                )
            )
            modal = host.screen
            modal._create()
            await settled(host.started.is_set)
            if change == "nested_screen":
                await host.push_screen(Screen())
            else:
                modal.query_one("#workspace-create-name", Input).value = "Changed"
                modal.query_one(
                    "#workspace-default-persona", Select
                ).value = WorkspacePersonaChoice.NONE
                await pilot.pause()
            host.release.set()
            if change == "nested_screen":
                await settled(lambda: not modal._committed)
                assert registry.list_workspaces() == ()
                await host.pop_screen()
                modal._create()
            await settled(lambda: len(registry.list_workspaces()) == 1)
            saved = registry.list_workspaces()[0]
            if change == "explicit_none":
                assert saved.name == "Changed"
                assert saved.assistant_defaults is None
                assert saved.assistant_defaults_explicit_none
                assert host.personas() == []
            else:
                assert saved.assistant_defaults is not None
                assert len(host.personas()) == 1
            assert host.compositions == 1
    finally:
        registry.db.close()
