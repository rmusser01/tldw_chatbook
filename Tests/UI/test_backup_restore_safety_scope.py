"""Only explicit current local selections enter the extra rollback-copy scope."""

import asyncio

import pytest

from Tests.Backup_Recovery.conftest import (
    helper_resource_root as helper_resource_root,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_builtin_later_snapshot import (
    complete_builtin_case as complete_builtin_case,  # noqa: PLC0414
)


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """The recovery-only harness has no full-app startup/catalog refresh."""


@pytest.mark.asyncio
@pytest.mark.parametrize("change_target", [False, True])
async def test_preserved_builtin_safety_scope_is_explicit_and_target_bound(
    complete_builtin_case, tmp_path, monkeypatch, change_target
):
    from textual.app import App
    from textual.widgets import Button, Input, Select
    
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    _, plan_for, members, selected = complete_builtin_case
    original = plan_for(())
    service = RecoveryService(tmp_path / "control")
    # Native fixture owns the exact selected inventory and installed asset root.
    monkeypatch.setattr(service, "preview_backup", lambda *a, **kw: original.target)

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    async def click(pilot, selector):
        pilot.app.screen.query_one(selector, Button).focus()
        await pilot.press("enter")
        await pilot.pause()

    async def reviewed(pilot):
        await click(pilot, "#backup-review-restore")
        async with asyncio.timeout(10):
            while pilot.app.screen._restore_plan is None:
                await asyncio.sleep(.02)

    app = Harness()
    before = selected.read_bytes(), selected.stat().st_ino
    try:
        async with app.run_test(size=(110, 40)) as pilot:
            screen = app.screen
            await click(pilot, "#backup-open-inspect")
            screen.query_one("#backup-source", Input).value = str(tmp_path / "replacement.zip")
            await click(pilot, "#backup-inspect")
            async with asyncio.timeout(10):
                while screen._inspection_summary is None:
                    await asyncio.sleep(.02)
            mapping = dict((*original.destinations, *original.selectors))
            for index, slot in enumerate(screen._inspection_summary["destination_slots"]):
                screen.query_one(f"#backup-root-{index}", Input).value = str(mapping[slot["logical_id"]])
            selector = next(i.path for i in original.target.items if i.owner == "config")
            screen.query_one("#backup-target-config", Input).value = str(selector)
            screen.query_one("#backup-profile-name-0", Input).value = "Local"
            screen.query_one("#backup-restore-mode", Select).value = "replace"
            await pilot.pause()
            await reviewed(pilot)
            expected = {i.logical_id for i in members}
            boxes = {b.name: b for b in screen.query(".backup-safety-member")}
            assert expected.issubset(boxes)
            assert not any(b.value for b in boxes.values())
            assert not screen._restore_plan.safety_scope
            for key in expected:
                boxes[key].value = True
            await pilot.pause()
            assert screen._restore_plan is None
            if change_target:
                screen.query_one("#backup-target-config", Input).value += "-changed"
                await pilot.pause()
                assert not any(b.value for b in boxes.values())
                assert all(b.disabled for b in boxes.values())
            else:
                await reviewed(pilot)
                plan = screen._restore_plan
                assert set(plan.safety_scope) == expected
                assert expected.issubset(dict(plan.preserve))
                assert not expected.intersection(dict(plan.restore))
            assert (selected.read_bytes(), selected.stat().st_ino) == before
    finally:
        await asyncio.to_thread(service.close)
