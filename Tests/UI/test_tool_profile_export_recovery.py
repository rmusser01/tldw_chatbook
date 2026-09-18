"""Tool Profile export destination recovery with real archive publication."""

import asyncio
import json
import zipfile
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_configuration_hub import (
    StyledSettingsDestinationHarness,
    _build_test_app,
    _open_settings_category,
)
from Tests.UI.test_settings_tool_profiles import _profile, _WorkflowService
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.Tool_Packs.catalog_snapshot import PermissionInventoryRegistry
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Tool_Packs.publication import ToolPackPublicationResult
from tldw_chatbook.Tool_Packs.service import ToolPackService, ToolProfileListing
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave
from tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review import (
    ToolPackExportReviewModal,
)
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


async def _wait(pilot, predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await pilot.pause(0.03)
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", ["existing", "suffix", "appeared"])
@pytest.mark.parametrize("cancel", [False, True])
@private_profile_test
async def test_export_corrects_destination_without_repeating_review(
    request, tmp_path, monkeypatch, conflict, cancel
):
    app = _build_test_app()
    store = MCPPermissionStore(tmp_path / "permissions.json")
    store.ensure_profile("research")
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="export-test")
    )
    request.addfinalizer(registry.db.close)
    service = ToolPackService.compose(
        permission_store=store,
        inventory=PermissionInventoryRegistry.v1(
            SimpleNamespace(
                get_inventory=lambda: {"tools": []}, get_external_servers=list
            ),
            fallback_root=tmp_path,
        ),
        workspace_registry=registry,
        receipt_root=tmp_path / "receipts",
    )
    captures = []
    publications = []
    capture = service.capture_export
    publish = service.publish_export

    def capture_once(*args, **kwargs):
        review = capture(*args, **kwargs)
        captures.append(review)
        return review

    def publish_real(review, destination, **kwargs):
        assert review is captures[0]
        publications.append(destination.path)
        if conflict == "appeared" and len(publications) == 1:
            destination.path.write_bytes(b"incumbent")
        return publish(review, destination, **kwargs)

    monkeypatch.setattr(service, "capture_export", capture_once)
    monkeypatch.setattr(service, "publish_export", publish_real)
    app.tool_pack_service = service
    assert service.list_profiles().unavailable_category is None
    host = StyledSettingsDestinationHarness(app, "settings")
    selected = tmp_path / (
        "invalid.txt" if conflict == "suffix" else "taken.tldw-tool-pack"
    )
    if conflict == "existing":
        selected.write_bytes(b"incumbent")
    corrected = tmp_path / "new.tldw-tool-pack"

    async with host.run_test(size=(120, 35)) as pilot:
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await host.workers.wait_for_complete()
        settings = host.screen
        profile_ids = settings._tool_profiles_listing.profiles
        index = next(
            i for i, row in enumerate(profile_ids) if row.profile_id == "research"
        )
        button = settings.query_one(f"#tool-profile-export-{index}", Button)
        button.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait(pilot, lambda: isinstance(host.screen, ToolPackExportReviewModal))
        await pilot.press("enter")
        await _wait(pilot, lambda: isinstance(host.screen, EnhancedFileSave))
        first_picker = host.screen
        filename = first_picker.query_one("#filename-input", Input)
        filename.value = str(selected)
        filename.focus()
        await pilot.press("enter")
        await _wait(
            pilot,
            lambda: (
                isinstance(host.screen, EnhancedFileSave)
                and host.screen is not first_picker
            ),
        )
        retry_picker = host.screen
        error = str(retry_picker.query_one("#error-line", Static).renderable)
        assert "filename" in error.casefold()
        assert "platform" not in error.casefold()
        assert retry_picker.query_one("#filename-input", Input).value == selected.name
        assert len(captures) == 1
        if conflict != "suffix":
            assert selected.read_bytes() == b"incumbent"

        if cancel:
            await pilot.press("escape")
            await _wait(
                pilot, lambda: settings._tool_profiles_result == "Export cancelled"
            )
            assert not corrected.exists()
        else:
            filename = retry_picker.query_one("#filename-input", Input)
            filename.value = corrected.name
            filename.focus()
            await pilot.press("enter")
            await _wait(
                pilot,
                lambda: settings._tool_profiles_result.startswith("Exported Tool Pack"),
            )
            with zipfile.ZipFile(corrected) as archive:
                manifest = json.loads(archive.read("tool-pack.json"))
            assert manifest["profile"]["suggested_id"] == "research"
        assert len(captures) == 1
        if conflict != "suffix":
            assert selected.read_bytes() == b"incumbent"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["publication_unsupported", "publication_failed", "durability_uncertain"]
)
@private_profile_test
async def test_terminal_publication_outcomes_do_not_offer_another_destination(
    request, tmp_path, monkeypatch, outcome
):
    service = _WorkflowService(ToolProfileListing(profiles=(_profile("research"),)))
    app = _build_test_app()
    app.tool_pack_service = service
    host = StyledSettingsDestinationHarness(app, "settings")
    pickers = []
    original_push = host.push_screen_wait

    async def push(screen):
        if isinstance(screen, EnhancedFileSave):
            pickers.append(screen)
        return await original_push(screen)

    def publish(*args, **kwargs):
        if outcome == "durability_uncertain":
            return ToolPackPublicationResult("9" * 64, True, True)
        raise ToolPackError("export", outcome)

    monkeypatch.setattr(host, "push_screen_wait", push)
    monkeypatch.setattr(service, "publish_export", publish)
    async with host.run_test(size=(120, 35)) as pilot:
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await host.workers.wait_for_complete()
        settings = host.screen
        settings.query_one("#tool-profile-export-0", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait(pilot, lambda: isinstance(host.screen, ToolPackExportReviewModal))
        await pilot.press("enter")
        await _wait(pilot, lambda: isinstance(host.screen, EnhancedFileSave))
        filename = host.screen.query_one("#filename-input", Input)
        filename.value = str(tmp_path / "research.tldw-tool-pack")
        filename.focus()
        await pilot.press("enter")
        await _wait(pilot, lambda: outcome in settings._tool_profiles_result)
        await host.workers.wait_for_complete()
        assert len(pickers) == 1
        assert host.screen is settings
        if outcome == "durability_uncertain":
            assert settings._tool_profiles_result.startswith(
                "Export may have completed"
            )
