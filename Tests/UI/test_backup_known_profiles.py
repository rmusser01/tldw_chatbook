"""Canonical backup includes known profiles and retains reviewed source scope."""

import asyncio

import pytest

from Tests.Backup_Recovery.conftest import (
    helper_resource_root as helper_resource_root,  # noqa: PLC0414 - component helper fixture
)


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """The focused screen host does not compose runtime model services."""


@pytest.mark.asyncio
async def test_default_profile_review_keeps_known_profiles_and_added_selection(
    tmp_path, monkeypatch, helper_resource_root
):
    from textual.app import App
    from textual.widgets import Button, Checkbox, Input, Static

    from tldw_chatbook.Backup_Recovery import bootstrap, crypto
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    authority = admission_authority(root)

    def profile(name, *, known=True):
        parent = tmp_path / name
        parent.mkdir(mode=0o700)
        selector = parent / "config.toml"
        selector.write_text(
            f'[general]\nusers_name="{name}"\n[paths]\ndata_dir="{parent / "data"}"\n'
        )
        selector.chmod(0o600)
        if known:
            authority.register(name, (parent,))
            bind_profile(root, selector, (name,), root / "admission")
        return selector

    current, known = profile("current"), profile("other")
    added = profile("added", known=False)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(current))
    service = RecoveryService(tmp_path / "control")

    class Harness(App):
        def on_mount(self):
            self.push_screen(
                BackupRestoreScreen(
                    service, config_paths=(current,), include_known_profiles=True
                )
            )

    app = Harness()
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            await pilot.click("#backup-open-create")
            screen = app.screen
            screen._profile_chosen(added)
            destination = tmp_path / "reviewed.tldw-backup.zip"
            screen.query_one("#backup-destination", Input).value = str(destination)
            screen.query_one("#backup-partial", Checkbox).value = True
            await pilot.pause()
            await pilot.click("#backup-review")
            async with asyncio.timeout(15):
                while screen._preview is None:
                    await asyncio.sleep(0.03)
                    message = str(screen.query_one("#backup-message", Static).render())
                    assert "failed" not in message.lower(), message
            shown = str(screen.query_one("#backup-profiles", Static).render())
            assert all(str(path) in shown for path in (current, known, added))
            assert not screen.query_one("#backup-create", Button).disabled
            profile("newly-known")
            await pilot.click("#backup-create")
            async with asyncio.timeout(15):
                while (
                    service.current() is None or service.current()["state"] == "running"
                ):
                    await asyncio.sleep(0.03)
            assert service.current()["state"] == "failed", dict(service.current())
            assert service.current()["issues"] == ("review_required",)
            assert "scope_changed" in service.current()["review_issues"]
            assert not destination.exists()
    finally:
        await asyncio.to_thread(service.close)
