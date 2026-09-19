"""Restore choices remain keyboard reachable and require a reviewed profile name."""

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """The recovery-only harness never starts full-app catalog refresh."""


@pytest.mark.asyncio
async def test_keyboard_profile_destination_review_requires_name(tmp_path):
    from textual.widgets import Button, Input, Static

    from Tests.Backup_Recovery.test_restore_destinations import profile_archive
    from Tests.UI.consolidated_css import ConsolidatedCSSApp as App
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    (tmp_path / "restore-locations").mkdir(mode=0o700)
    service = RecoveryService(tmp_path / "control")
    op = service.start_inspection(profile_archive(tmp_path), password=None)
    assert service.wait(op)["state"] == "succeeded"

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    async def press(pilot, identifier):
        pilot.app.screen.query_one(identifier, Button).focus()
        await pilot.press("enter")
        await pilot.pause()

    try:
        async with Harness().run_test(size=(110, 40)) as pilot:
            screen = pilot.app.screen
            await press(pilot, "#backup-open-inspect")
            async with asyncio.timeout(10):
                while screen._inspection_summary is None:
                    await asyncio.sleep(0.02)
            fields = list(screen.query("#backup-destination-slots Input"))
            assert len(fields) == 2
            screen.query_one("#backup-root-0", Input).value = str(
                tmp_path / "restore-locations" / "new"
            )
            await press(pilot, "#backup-review-restore")
            assert "display name" in str(
                screen.query_one("#backup-message", Static).render()
            )
            assert screen.query_one("#backup-start-restore", Button).disabled
            screen.query_one("#backup-profile-name-0", Input).focus()
            await pilot.press("N", "e", "w")
            await press(pilot, "#backup-review-restore")
            async with asyncio.timeout(10):
                while screen._restore_plan is None:
                    await asyncio.sleep(0.02)
            assert dict(screen._restore_plan.profile_names) == {"source": "New"}
            assert not screen.query_one("#backup-start-restore", Button).disabled
            assert "Local profile name: New" in str(
                screen.query_one("#backup-restore-preview", Static).render()
            )
    finally:
        service.close()


@pytest.mark.asyncio
async def test_acknowledged_partial_review_does_not_ask_for_acknowledgement_again(
    tmp_path,
):
    from textual.widgets import Static

    from Tests.UI.consolidated_css import ConsolidatedCSSApp as App
    from tldw_chatbook.Backup_Recovery.models import Inventory
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    service = RecoveryService(tmp_path / "control")

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    try:
        async with Harness().run_test() as pilot:
            screen = pilot.app.screen
            inventory = Inventory((), False, "test", ())
            details = {
                "inventory": inventory,
                "complete": False,
                "maintenance": "Writers resume after copying.",
                "availability": (True, ""),
                "capacity": (),
                "credential_mode": "exclude",
            }
            screen._show_preview(
                screen._revision,
                details,
                ((), tmp_path / "backup.zip", {"allow_partial": True}),
            )
            text = str(screen.query_one("#backup-coverage", Static).render())
            assert "Partial — acknowledged" in text
            assert "acknowledgement required" not in text
    finally:
        service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["status", "source", "start"])
async def test_operation_failures_show_corrective_guidance(tmp_path, surface):
    from textual.widgets import Static

    from Tests.UI.consolidated_css import ConsolidatedCSSApp as App
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    service = RecoveryService(tmp_path / "control")
    operation = service.start_inspection(tmp_path / "missing.zip", password=None)
    assert service.wait(operation)["state"] == "failed"

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    try:
        async with Harness().run_test() as pilot:
            screen = pilot.app.screen
            if surface == "status":
                screen._refresh_status()
                text = str(screen.query_one("#backup-status", Static).render())
            elif surface == "source":
                screen._show_preview(screen._revision, None, "destination_exists")
                text = str(screen.query_one("#backup-message", Static).render())
            else:
                screen._operation_start_ready(
                    screen._revision, "restore", None, "destination_exists"
                )
                text = str(screen.query_one("#backup-message", Static).render())
            assert any(word in text for word in ("Choose", "Review", "Enter", "Check"))
            assert "missing.zip" not in text
    finally:
        service.close()
