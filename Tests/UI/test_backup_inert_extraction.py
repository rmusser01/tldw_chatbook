"""The recovery view extracts explicitly reviewed groups through the real service."""

import asyncio
import json

import pytest


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """This minimal screen host does not construct the production application."""


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["extract", "destination_changed", "source_changed"])
async def test_real_inert_extraction_requires_current_visible_review(tmp_path, change):
    from textual.app import App
    from textual.widgets import Button, Checkbox, Input, Static

    from Tests.Backup_Recovery.test_inert_extraction import _archive
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    archive, payloads = _archive(tmp_path)
    service = RecoveryService(tmp_path / "control")

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    async def visible_click(screen, pilot, selector):
        screen.query_one(selector).scroll_visible(animate=False)
        await pilot.pause()
        assert await pilot.click(selector)

    try:
        async with Harness().run_test(size=(90, 30)) as pilot:
            screen = pilot.app.screen
            await pilot.click("#backup-open-inspect")
            screen.query_one("#backup-source", Input).value = str(archive.path)
            await pilot.click("#backup-inspect")
            async with asyncio.timeout(10):
                while screen._inspection_id is None:
                    await asyncio.sleep(0.03)
            screen.query_one("#backup-inert-group-0", Checkbox).value = True
            destination = tmp_path / "manual"
            screen.query_one("#backup-inert-destination", Input).value = str(
                destination
            )
            await visible_click(screen, pilot, "#backup-review-extraction")
            async with asyncio.timeout(10):
                while screen.query_one("#backup-start-extraction", Button).disabled:
                    await asyncio.sleep(0.03)
            assert not destination.exists()
            assert "unsupported" in str(
                screen.query_one("#backup-inert-preview", Static).render()
            )
            if change == "extract":
                await visible_click(screen, pilot, "#backup-start-extraction")
                async with asyncio.timeout(10):
                    while "Inert files extracted" not in str(
                        screen.query_one("#backup-status", Static).render()
                    ):
                        await asyncio.sleep(0.03)
                report = json.loads((destination / "inert-mapping.json").read_text())
                assert {row["logical_id"] for row in report["files"]} == {
                    "db",
                    "config",
                }
                for row in report["files"]:
                    assert (destination / row["output"]).read_bytes() == payloads[
                        row["logical_id"]
                    ]
                assert not service.current()["result"].get("restoration_validated")
            else:
                field = (
                    "backup-source"
                    if change == "source_changed"
                    else "backup-inert-destination"
                )
                screen.query_one("#" + field, Input).value = str(tmp_path / "changed")
                await pilot.pause()
                assert screen.query_one("#backup-start-extraction", Button).disabled
                assert screen._extraction_plan is None
                assert not destination.exists()
    finally:
        service.close()
