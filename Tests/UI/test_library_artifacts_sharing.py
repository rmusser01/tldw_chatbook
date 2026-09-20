"""Library sharing uses real dialogs and staging, with only child transport faked."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from textual import on
from textual.screen import Screen
from textual.widgets import Button, SelectionList, Static

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.UI.Screens.artifact_share_dialog import ArtifactShareDialog
from tldw_chatbook.Web_Server.artifact_share import ArtifactShareController

pytestmark = pytest.mark.ui


class LibraryHostScreen(Screen):
    def __init__(self, owner):
        super().__init__()
        from tldw_chatbook.UI.Library_Modules.library_artifacts_share_controller import (
            LibraryArtifactsShareController,
        )

        self.app_instance = owner
        self.sharing = LibraryArtifactsShareController(self)

    def compose(self):
        yield self.sharing.build_strip()
        yield Static("Chatbooks", id="route")
        yield Button("Share", id="open-share")

    @on(Button.Pressed, "#open-share")
    def open_share(self):
        self.sharing.open_dialog()

    def on_mount(self):
        self.sharing.refresh_status()

    def on_screen_suspend(self):
        self.sharing.suspend()

    def on_screen_resume(self):
        self.sharing.resume()

    def on_unmount(self):
        self.sharing.dispose()


class ShareHost(ConsolidatedCSSApp):
    CSS_PATH: ClassVar = list(APP_STYLESHEETS)

    def __init__(self, service, controller):
        super().__init__()
        self.local_chatbook_service = service
        self.artifact_share_controller = controller
        self.app_config = {}
        self.notes_user_id = "sharing-test"
        self.notices = []
        self.library = LibraryHostScreen(self)
        self.ensures = 0

    def _get_artifact_share_controller(self):
        self.ensures += 1
        if self.artifact_share_controller is None:
            self.artifact_share_controller = ArtifactShareController()
        return self.artifact_share_controller

    def on_mount(self):
        self.push_screen(self.library)

    def notify(self, message, **kwargs):
        self.notices.append(str(message))


async def wait_until(pilot, predicate):
    for _ in range(120):
        if predicate():
            return
        await pilot.pause(0.025)
    raise AssertionError("Sharing operation did not settle")


@pytest.fixture
def staged_controller(tmp_path, monkeypatch):
    """Keep real validation, staging, status, and cleanup; fake only the server."""
    import tldw_chatbook.Web_Server.artifact_share as module

    root = tmp_path / "shares"
    monkeypatch.setattr(module, "share_root_dir", lambda: root)

    original_popen = module.subprocess.Popen

    def launch(args, **kwargs):
        if len(args) < 3 or args[2] != "tldw_chatbook.Web_Server.artifact_share_server":
            return original_popen(args, **kwargs)
        manifest = Path(args[3])
        manifest.with_name("status.json").write_text(
            json.dumps({"url": "http://127.0.0.1:8123"})
        )
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(
        ArtifactShareController, "_terminate_child", lambda self, child: None
    )
    controller = ArtifactShareController()
    try:
        yield controller
    finally:
        controller.stop_share()


async def exported_registry(tmp_path):
    """Export real note-containing packs, then register through the owner API."""
    from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
    from tldw_chatbook.Chatbooks.chatbook_models import ContentType
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    source = tmp_path / "source.db"
    db = CharactersRAGDB(source, client_id="sharing-test")
    try:
        note = db.add_note(title="Packing list", content="Map, water, boots")
    finally:
        db.close_connection()
    creator = ChatbookCreator({"ChaChaNotes": str(source)})
    service = LocalChatbookService(registry_path=tmp_path / "registry.json")
    records = []
    for index in (1, 2):
        output = tmp_path / f"pack-{index}.zip"
        success, message, _ = creator.create_chatbook(
            name=f"Pack {index}",
            description="A real exported note",
            content_selections={ContentType.NOTE: [str(note)]},
            output_path=output,
        )
        assert success, message
        records.append(
            await service.create_chatbook(name=f"Pack {index}", file_path=output)
        )
    return service, records


async def confirm_two(pilot, host):
    host.library.sharing.open_dialog()
    await wait_until(
        pilot,
        lambda: isinstance(host.screen, ArtifactShareDialog) and host.screen.is_mounted,
    )
    dialog = host.screen
    options = dialog.query_one("#share-artifact-list", SelectionList)
    options.select_all()
    dialog.query_one("#share-start", Button).press()
    await pilot.pause()
    return dialog


@pytest.mark.asyncio
@private_profile_test
async def test_own_modal_suspend_accepts_once_and_strip_stops_real_staged_packs(
    request,
    tmp_path,
    staged_controller,
):
    service, records = await exported_registry(tmp_path)
    host = ShareHost(service, staged_controller)
    async with host.run_test(size=(140, 55)) as pilot:
        await pilot.pause()
        assert not host.library.query_one("#library-artifacts-share-strip").display
        dialog = await confirm_two(pilot, host)
        await wait_until(pilot, lambda: staged_controller.status is not None)
        status = staged_controller.status
        assert status.artifact_count == 2
        manifest = json.loads((status.share_dir / "manifest.json").read_text())
        assert {item["display_name"] for item in manifest["artifacts"]} == {
            "Pack 1",
            "Pack 2",
        }
        assert all(
            (status.share_dir / item["staged_name"]).is_file()
            for item in manifest["artifacts"]
        )
        await wait_until(
            pilot,
            lambda: host.library.query_one("#library-artifacts-share-strip").display,
        )
        assert "Sharing 2 Chatbooks" in str(
            host.library.query_one("#library-artifacts-share-status", Static).renderable
        )
        # Simulate a duplicate delivery of this exact dialog's callback.
        host.library.sharing._accept_dialog(
            dialog,
            host.library.sharing._profile(),
            staged_controller,
            {
                "selected_records": records,
                "share_name": "duplicate",
                "bind": "127.0.0.1",
                "port": 0,
            },
        )
        await pilot.pause()
        assert staged_controller.status.share_dir == status.share_dir
        for route in ("Reports", "Notes"):
            host.library.query_one("#route", Static).update(route)
            host.library.sharing.invalidate_pending_presentation()
            assert host.library.query_one("#library-artifacts-share-stop").display
        host.library.query_one("#library-artifacts-share-manage", Button).press()
        await wait_until(
            pilot,
            lambda: (
                isinstance(host.screen, ArtifactShareDialog) and host.screen.is_mounted
            ),
        )
        assert host.screen.query_one("#share-active-note")
        await pilot.press("escape")
        await pilot.pause()
        assert staged_controller.status.share_dir == status.share_dir
        host.library.query_one("#library-artifacts-share-stop", Button).press()
        await wait_until(pilot, lambda: staged_controller.status is None)
        assert not status.share_dir.exists()
        await wait_until(
            pilot,
            lambda: (
                not host.library.query_one("#library-artifacts-share-strip").display
            ),
        )


@pytest.mark.asyncio
@private_profile_test
async def test_cancel_and_empty_strip_never_start_or_construct_sharing(
    request, tmp_path
):
    service = LocalChatbookService(registry_path=tmp_path / "registry.json")
    host = ShareHost(service, None)
    async with host.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        assert host.artifact_share_controller is None
        host.library.sharing.open_dialog()
        await wait_until(
            pilot,
            lambda: (
                isinstance(host.screen, ArtifactShareDialog) and host.screen.is_mounted
            ),
        )
        await pilot.press("escape")
        await pilot.pause()
        assert host.artifact_share_controller.status is None


class GatedRegistry:
    def __init__(self, service):
        self.service = service
        self.started = threading.Event()
        self.release = threading.Event()

    def artifact_read_snapshot(self):
        self.started.set()
        assert self.release.wait(5), "test listing gate timed out"
        return self.service.artifact_read_snapshot()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transition", ["canvas", "leave", "leave-return", "unmount", "profile"]
)
@private_profile_test
async def test_late_listing_cannot_present_after_library_visit_changes(
    request,
    tmp_path,
    staged_controller,
    transition,
):
    service = LocalChatbookService(registry_path=tmp_path / "registry.json")
    gated = GatedRegistry(service)
    host = ShareHost(gated, staged_controller)
    async with host.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        host.library.sharing.open_dialog()
        assert await asyncio.to_thread(gated.started.wait, 3)
        if transition == "canvas":
            host.library.sharing.invalidate_pending_presentation()
            host.library.query_one("#route", Static).update("Notes")
        elif transition == "profile":
            host.app_config = {}
        elif transition == "unmount":
            await host.pop_screen()
        else:
            await host.push_screen(Screen())
            if transition == "leave-return":
                await host.pop_screen()
        await pilot.pause()
        screen_after_change = host.screen
        notices_after_change = list(host.notices)
        gated.release.set()
        await pilot.pause(0.2)
        assert host.screen is screen_after_change
        assert host.notices == notices_after_change
        assert not any(
            isinstance(screen, ArtifactShareDialog) for screen in host.screen_stack
        )
        if transition in ("canvas", "leave-return", "profile"):
            host.library.sharing.open_dialog()
            await wait_until(
                pilot,
                lambda: (
                    isinstance(host.screen, ArtifactShareDialog)
                    and host.screen.is_mounted
                ),
            )
            await pilot.press("escape")


@pytest.mark.asyncio
@private_profile_test
async def test_approved_staging_survives_parent_unmount_and_new_library_manages_it(
    request,
    tmp_path,
    monkeypatch,
    staged_controller,
):
    service, _ = await exported_registry(tmp_path)
    started, release = threading.Event(), threading.Event()
    original_start = staged_controller.start_share

    def delayed_start(**options):
        started.set()
        assert release.wait(5), "test staging gate timed out"
        return original_start(**options)

    monkeypatch.setattr(staged_controller, "start_share", delayed_start)
    host = ShareHost(service, staged_controller)
    async with host.run_test(size=(140, 55)) as pilot:
        await pilot.pause()
        await confirm_two(pilot, host)
        assert await asyncio.to_thread(started.wait, 3)
        await host.pop_screen()
        await pilot.pause()
        release.set()
        await wait_until(pilot, lambda: staged_controller.status is not None)
        status = staged_controller.status
        assert status.artifact_count == 2
        replacement = LibraryHostScreen(host)
        await host.push_screen(replacement)
        await pilot.pause()
        assert replacement.query_one("#library-artifacts-share-strip").display
        replacement.query_one("#library-artifacts-share-stop", Button).press()
        await wait_until(pilot, lambda: staged_controller.status is None)
        assert not status.share_dir.exists()


@pytest.mark.asyncio
@private_profile_test
async def test_share_dialog_lists_past_old_thousand_limit_and_preselects_exact_key(
    request,
    tmp_path,
    staged_controller,
):
    from tldw_chatbook.Library.library_artifacts_state import ArtifactKey

    service, records = await exported_registry(tmp_path)
    # Valid records originate from the public registry API, including the tail.
    for index in range(1001):
        await service.create_chatbook(
            name=f"Registered {index}", file_path=records[0]["file_path"]
        )
    tail = await service.create_chatbook(
        name="Tail export", file_path=records[1]["file_path"]
    )
    await service.create_chatbook(name="No export")
    await service.create_chatbook(
        name="Missing export", file_path=tmp_path / "gone.zip"
    )
    unusable = tmp_path / "invalid.zip"
    unusable.write_text("not a ZIP")
    await service.create_chatbook(name="Invalid export", file_path=unusable)
    host = ShareHost(service, staged_controller)
    async with host.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        host.library.sharing.open_dialog(ArtifactKey("chatbook", tail["chatbook_id"]))
        await wait_until(
            pilot,
            lambda: (
                isinstance(host.screen, ArtifactShareDialog) and host.screen.is_mounted
            ),
        )
        options = host.screen.query_one("#share-artifact-list", SelectionList)
        assert options.option_count == 1004
        await wait_until(pilot, lambda: options.selected == [str(tail["id"])])
        await pilot.press("escape")


@pytest.mark.asyncio
@private_profile_test
async def test_dialog_result_from_replaced_profile_cannot_start_share(
    request,
    tmp_path,
    staged_controller,
):
    service, _ = await exported_registry(tmp_path)
    host = ShareHost(service, staged_controller)
    async with host.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        host.library.sharing.open_dialog()
        await wait_until(
            pilot,
            lambda: (
                isinstance(host.screen, ArtifactShareDialog) and host.screen.is_mounted
            ),
        )
        dialog = host.screen
        dialog.query_one("#share-artifact-list", SelectionList).select_all()
        host.app_config = {}
        dialog.query_one("#share-start", Button).press()
        await pilot.pause()
        assert staged_controller.status is None


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["refresh", "dispose"])
@private_profile_test
async def test_status_lifecycle_does_not_wait_for_share_staging_lock(
    request,
    tmp_path,
    staged_controller,
    action,
):
    """A staging worker must not freeze UI status reads or screen teardown."""
    from time import perf_counter

    service = LocalChatbookService(registry_path=tmp_path / "registry.json")
    host = ShareHost(service, staged_controller)
    entered, release = threading.Event(), threading.Event()

    def staging_lock():
        with staged_controller._lock:
            entered.set()
            release.wait(0.4)

    async with host.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        task = asyncio.create_task(asyncio.to_thread(staging_lock))
        assert await asyncio.to_thread(entered.wait, 2)
        try:
            start = perf_counter()
            if action == "refresh":
                host.library.sharing.refresh_status()
            else:
                host.library.sharing.dispose()
            elapsed = perf_counter() - start
        finally:
            release.set()
            await task
        assert elapsed < 0.1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "urls",
    [
        ("http://127.0.0.1:8123",),
        ("http://127.0.0.1:8123", "http://192.168.1.25:8123"),
    ],
)
@private_profile_test
async def test_share_strip_exposes_every_active_address_without_inventing_lan_url(
    request, tmp_path, staged_controller, urls
):
    from tldw_chatbook.Web_Server.artifact_share import ShareStatus

    service = LocalChatbookService(registry_path=tmp_path / "registry.json")
    host = ShareHost(service, staged_controller)
    async with host.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        status = ShareStatus("Trip", urls, 2, tmp_path / "staged")
        host.library.sharing._render_status(status)
        rendered = str(
            host.library.query_one("#library-artifacts-share-status", Static).renderable
        )
        assert rendered == f"Sharing 2 Chatbooks · {' · '.join(urls)}"
        host.library.sharing._render_status(None)
        assert not host.library.query_one("#library-artifacts-share-strip").display


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["profile", "owner", "shutdown"])
@private_profile_test
async def test_accepted_but_unstarted_share_rechecks_application_authority(
    request, tmp_path, monkeypatch, staged_controller, transition
):
    service, _ = await exported_registry(tmp_path)
    host = ShareHost(service, staged_controller)
    async with host.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        run_worker = host.run_worker
        pending = []

        def defer_action(work, **kwargs):
            if kwargs.get("group") == "library-artifacts-share-action":
                pending.append(work)
                return None
            return run_worker(work, **kwargs)

        monkeypatch.setattr(host, "run_worker", defer_action)
        await confirm_two(pilot, host)
        assert len(pending) == 1
        if transition == "profile":
            host.app_config = {}
        elif transition == "owner":
            host.artifact_share_controller = None
        else:
            host._shutting_down = True
        await pending.pop()
        assert staged_controller.status is None
