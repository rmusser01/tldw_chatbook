"""Concurrent display refreshes use rows with no native authority claims."""

import asyncio

import pytest


def _rows(name):
    current = {
        "status": "requirements_unavailable",
        "requirements_checked": False,
        "config": f"/display/{name}.toml",
        "generation": None,
    }
    entry = {
        "profile_id": name,
        "config": f"/display/{name}/restored.toml",
        "data": f"/display/{name}/data",
        "status": "restoration_validated",
        "requirements_checked": True,
        "needs_setup": True,
        "pending_owners": ("config",),
    }
    return (entry,), current


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["same_revision", "navigation", "new_revision"])
async def test_list_delivery_keeps_only_current_widgets(tmp_path, monkeypatch, change):
    from textual.app import App

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    service = RecoveryService(tmp_path / "control")

    class Host(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    app = Host()
    try:
        async with app.run_test(size=(100, 36)):
            screen = app.screen
            screen._show_mode("profiles")
            revision = screen._revision
            listing = screen.query_one("#backup-list")
            entered, release = asyncio.Event(), asyncio.Event()
            original_remove = listing.remove_children
            calls = 0

            async def held_first_remove():
                nonlocal calls
                calls += 1
                first = calls == 1
                await original_remove()
                if first:
                    entered.set()
                    await release.wait()

            monkeypatch.setattr(listing, "remove_children", held_first_remove)

            async def deliver(name, selected_revision):
                entries, current = _rows(name)
                await screen._list_ready(
                    selected_revision, "profiles", entries, (), None, current
                )

            older = asyncio.create_task(deliver("older", revision))
            try:
                async with asyncio.timeout(3):
                    await entered.wait()
                    if change == "navigation":
                        screen._show_mode("home")
                    else:
                        if change == "new_revision":
                            screen._show_mode("profiles")
                        await deliver("newer", screen._revision)
                    release.set()
                    await older
            finally:
                release.set()
                if not older.done():
                    older.cancel()
                    await asyncio.gather(older, return_exceptions=True)

            if change == "navigation":
                assert not listing.children
                assert screen.query_one("#backup-home").display
            else:
                assert len(screen.query("#backup-current-requirements")) == 1
                assert len(screen.query("#backup-open-media")) == 1
                buttons = list(screen.query(".backup-open-profile"))
                assert len(buttons) == 1 and buttons[0].name == "newer"
                assert "/display/newer.toml" in str(
                    screen.query_one("#backup-current-requirements").render()
                )
    finally:
        await asyncio.to_thread(service.close)
