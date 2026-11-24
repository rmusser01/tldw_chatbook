"""Mounted Copy actions require a frozen confirmation and a completed receipt."""

from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Static

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from tldw_chatbook.Research_Workspace.contracts import QualifiedWorkspaceRef
from tldw_chatbook.Research_Workspace.copy_models import (
    CopyOwnerRef,
    ResearchCopyItem,
    ResearchCopyManifest,
    ResearchCopyStatus,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("completed", (True, False))
async def test_copy_and_share_opens_sharing_only_after_completed_copy(
    tmp_path, monkeypatch, completed
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    planned = []
    executed = []
    shares = []
    target = QualifiedWorkspaceRef(
        "server", "server-copy-test", "profile-1", "principal-1"
    )

    async def preflight(source, destination, *, target_name):
        assert destination == target
        item = ResearchCopyItem(
            CopyOwnerRef(source, "artifact", "42", version=1),
            "copy_content",
            estimated_bytes=500,
        )
        manifest = ResearchCopyManifest(
            source, target, (item,), "2026-11-24T19:00:00Z", target_name
        )
        planned.append(manifest)
        return manifest

    async def execute(manifest, confirmation, **_kwargs):
        confirmation.validate(manifest)
        executed.append(manifest)
        status = (
            ResearchCopyStatus.COMPLETED
            if completed
            else ResearchCopyStatus.PARTIALLY_COMPLETED
        )
        return SimpleNamespace(
            manifest=manifest,
            items=(),
            status=status,
            server_migration_finalized=completed,
        )

    async def recover(_ref):
        return ()

    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(60, 20)) as pilot:
            screen, _ = await open_chat(app, pilot)
            current = screen.controller.selected_ref
            screen.copy_workspace = SimpleNamespace(
                new_server_target=lambda: target,
                preflight_copy=preflight,
                list_recoverable=recover,
                service=SimpleNamespace(execute_copy=execute),
            )

            async def share(ref, *, draft=None):
                shares.append((ref, draft))

            monkeypatch.setattr(screen, "_open_research_sharing", share)
            screen.query_one("#research-workspace-actions", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#research-action-copy-server-share")),
                "Local workspace actions",
            )
            assert not app.screen.query("#research-action-share")
            app.screen.query_one("#research-action-copy-server-share", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#research-copy-name")),
                "Server target form",
            )
            app.screen.query_one(
                "#research-copy-name", Input
            ).value = "Copy of reading group"
            app.screen.query_one("#research-copy-share-settings", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#research-share-target")),
                "optional share settings",
            )
            app.screen.query_one("#research-share-target", Input).value = "17"
            app.screen.query_one("#research-share-create", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#research-copy-preview")),
                "returned to Copy",
            )
            app.screen.query_one("#research-copy-preview", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#research-copy-confirm")),
                "frozen Copy preview",
            )
            await _until(
                pilot,
                lambda: any(
                    "500 bytes" in str(widget.render())
                    for widget in app.screen.query(Static)
                ),
                "Copy size disclosure",
            )
            preview = " ".join(
                str(widget.render()) for widget in app.screen.query(Static)
            )
            assert (
                "500" in preview
                and "Device-only" in preview
                and "Copy content" in preview
            )
            assert not executed and not shares
            app.screen.query_one("#research-copy-confirm", Button).press()
            if completed:
                await _until(
                    pilot, lambda: bool(shares), "completed destination sharing"
                )
                assert shares[0][0] == target and shares[0][1].scope_id == 17
            else:
                await _until(
                    pilot,
                    lambda: bool(app.screen.query("#research-copy-receipt-status")),
                    "partial receipt",
                )
                assert not shares
            assert executed == planned and screen.controller.selected_ref == current


async def test_workspace_actions_export_the_reviewed_local_bundle(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    from zipfile import ZipFile

    from Tests.Research_Workspace.test_workspace_export import _LocalOwner

    app, _, _ = build_research_app(tmp_path, monkeypatch)
    target = tmp_path / "workspace-bundle.zip"
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(100, 30)) as pilot:
            screen, _ = await open_chat(app, pilot)
            fixture = _LocalOwner()
            fixture.ref = screen.controller.selected_ref
            fixture.notes = tuple(
                replace(note, ref=fixture.ref) for note in fixture.notes
            )
            port = screen.controller.port_for_data_source(fixture.ref.data_source)
            monkeypatch.setattr(port, "list_sources", fixture.list_sources)
            monkeypatch.setattr(port, "list_notes", fixture.list_notes)
            monkeypatch.setattr(port, "list_outputs", fixture.list_outputs)
            monkeypatch.setattr(port, "get_note", fixture.get_note)
            monkeypatch.setattr(
                app.workspace_registry_service,
                "get_item_memberships",
                fixture.get_item_memberships,
            )

            screen.query_one("#research-workspace-actions", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#research-action-export")),
                "Export action",
            )
            app.screen.query_one("#research-action-export", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#confirm-button")),
                "Export review",
            )
            assert not target.exists()
            app.screen.query_one("#confirm-button", Button).press()
            await _until(
                pilot, lambda: bool(app.screen.query("#filename-input")), "File picker"
            )
            app.screen.query_one("#filename-input", Input).value = str(target)
            app.screen.query_one("#select", Button).press()
            await _until(pilot, target.exists, "Export written")
            with ZipFile(target) as bundle:
                assert b"Reviewed notes" in bundle.read("items/1-note.md")
                manifest = bundle.read("manifest.md")
                assert (
                    b"Metadata only" in manifest and b"Device preferences" in manifest
                )
                assert b"Device-only body" not in manifest
