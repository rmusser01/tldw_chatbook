"""Real Models route, service/store/HTTP boundary, and terminal-frame coverage."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import httpx
import pytest
from textual.widgets import Button, Collapsible, DataTable, Input

from Tests.LLM_Management.test_snapshot_admission import (
    _explicit_command,
    _launch_files,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
)
from tldw_chatbook.LLM_Management import snapshot_service
from tldw_chatbook.LLM_Management.snapshot_admission import prepare_launch
from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient
from tldw_chatbook.LLM_Management.snapshot_store import SnapshotStore
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen


@pytest.fixture
def snapshot_ui(tmp_path, monkeypatch):
    runtime, model = _launch_files(tmp_path)
    store = SnapshotStore(tmp_path / "snapshot-store")
    claim = ServerLaunchClaim("llamacpp")
    descriptor = prepare_launch(
        _explicit_command(runtime, model), {}, claim, "ui-launch"
    )
    state = SimpleNamespace(calls=[], busy=False, unknown=False, hold=asyncio.Event())
    state.hold.set()

    async def transport(request):
        state.calls.append((request.method, request.url.path, request.url.query))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            return httpx.Response(
                200, json={"build_info": "427291b", "model_path": str(model)}
            )
        if request.url.path == "/slots":
            return httpx.Response(
                200, json=[{"id": 0, "is_processing": state.busy, "n_ctx": 4096}]
            )
        await state.hold.wait()
        if state.unknown:
            raise httpx.ReadTimeout("private detail")
        filename = json.loads(request.content)["filename"]
        if request.url.params["action"] == "save":
            (store.prepare_launch_directory("ui-launch") / filename).write_bytes(
                b"cache"
            )
            return httpx.Response(
                200,
                json={"id_slot": 0, "filename": filename, "n_saved": 7, "n_written": 5},
            )
        return httpx.Response(
            200, json={"id_slot": 0, "filename": filename, "n_restored": 7, "n_read": 5}
        )

    monkeypatch.setattr(
        snapshot_service,
        "SnapshotClient",
        lambda value: SnapshotClient(value, transport=httpx.MockTransport(transport)),
    )
    state.service = snapshot_service.LlamaCppSnapshotService(
        store, lambda value: value is claim and not claim.cancel_event.is_set()
    )
    state.service.attach(descriptor)
    state.store, state.claim = store, claim
    app = _build_test_app()
    app.llamacpp_snapshot_service = state.service
    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda self: asyncio.sleep(0, result=False),
    )
    # The app factory has production CSS; disable only the competing splash timer.
    from tldw_chatbook.config import get_cli_setting

    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if (section, key) == ("splash_screen", "enabled")
            else get_cli_setting(section, key, default)
        ),
    )
    state.app = app
    return state


def frame(app, name):
    svg = app.export_screenshot()
    directory = os.environ.get("SNAPSHOT_CAPTURE_DIR")
    if directory:
        import cairosvg

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{name}.svg").write_text(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(target / f"{name}.png"))
    return " ".join(
        " ".join(ElementTree.fromstring(svg).itertext()).replace("│", " ").split()
    )


def painted_text(app, widget):
    region = widget.region
    strips = app.screen._compositor.render_strips()
    return " ".join(
        " ".join(
            strip.crop(region.x, region.right).text
            for strip in strips[max(0, region.y) : region.bottom]
        ).split()
    )


async def settle_ui(pilot, service):
    async with asyncio.timeout(5):
        while (
            service.view().operation_id is not None
            and service.view().status != "outcome_unknown"
        ):
            await pilot.pause(0.01)
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (140, 45)])
async def test_manager_paints_actions_and_saves_without_modal(snapshot_ui, size):
    from tldw_chatbook.Widgets.llamacpp_snapshot_manager import LlamaCppSnapshotManager

    h = snapshot_ui
    async with h.app.run_test(size=size) as pilot:
        await h.app.push_screen(LLMScreen(h.app))
        await pilot.pause()
        manager = h.app.screen.query_one(LlamaCppSnapshotManager)
        async with asyncio.timeout(5):
            while manager.query_one("#snapshot-save", Button).disabled:
                await pilot.pause(0.01)
        manager.query_one("#snapshot-save").focus()
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        frame(h.app, f"manager-{size[0]}-empty")
        assert "Keeps the newest 10 across all models" == painted_text(
            h.app, manager.query_one("#snapshot-retention")
        )
        for action in ("save", "restore", "delete", "refresh"):
            button = manager.query_one(f"#snapshot-{action}", Button)
            assert button in h.app.screen._compositor.visible_widgets
            assert h.app.screen.content_region.contains_region(button.region)
            assert action.capitalize() in painted_text(h.app, button)
        await pilot.press("enter")
        await settle_ui(pilot, h.service)
        assert len(h.store.list_records().records) == 1
        assert h.app.screen.query_one(LlamaCppSnapshotManager) is manager
        manager.query_one("#snapshot-records", DataTable).focus()
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        frame(h.app, f"manager-{size[0]}-saved")
        assert "Matching configuration" in painted_text(
            h.app, manager.query_one("#snapshot-records")
        )


@pytest.mark.asyncio
async def test_restore_confirmation_revalidates_launch_and_escape_keeps_source(
    snapshot_ui,
):
    from tldw_chatbook.Widgets.llamacpp_snapshot_manager import LlamaCppSnapshotManager

    h = snapshot_ui
    await h.service.refresh()
    h.service.start_save(0)
    while h.service.view().operation_id:
        await asyncio.sleep(0.01)
    async with h.app.run_test(size=(80, 24)) as pilot:
        await h.app.push_screen(LLMScreen(h.app))
        await pilot.pause()
        manager = h.app.screen.query_one(LlamaCppSnapshotManager)
        manager.query_one("#snapshot-restore").focus()
        await pilot.press("enter")
        assert "failure" in frame(h.app, "manager-80-confirmation").lower()
        await pilot.press("escape")
        assert len(h.store.list_records().records) == 1
        manager.query_one("#snapshot-restore").focus()
        await pilot.press("enter")
        h.claim.cancel_event.set()
        h.app.screen.query_one("#confirm-button", Button).press()
        await pilot.pause()
        assert not [call for call in h.calls if call[2] == b"action=restore"]


@pytest.mark.asyncio
async def test_manager_shortcuts_do_not_consume_launcher_input(snapshot_ui):
    h = snapshot_ui
    async with h.app.run_test(size=(140, 45)) as pilot:
        await h.app.push_screen(LLMScreen(h.app))
        await pilot.pause()
        field = h.app.screen.query_one("#llamacpp-additional-args", Input)
        field.focus()
        await pilot.press("s", "r", "d")
        assert field.value == "srd"
        assert not [call for call in h.calls if call[0] == "POST"]


@pytest.mark.asyncio
async def test_busy_unknown_counts_and_unknown_operation_recovery(snapshot_ui):
    from tldw_chatbook.Widgets.llamacpp_snapshot_manager import LlamaCppSnapshotManager

    h = snapshot_ui
    h.busy = True
    async with h.app.run_test(size=(140, 45)) as pilot:
        await h.app.push_screen(LLMScreen(h.app))
        await pilot.pause()
        manager = h.app.screen.query_one(LlamaCppSnapshotManager)
        await h.service.refresh()
        assert manager.query_one("#snapshot-save", Button).disabled
        assert manager._slot_id is None  # No automatic busy-slot destination.
        assert "Select an idle slot" in str(
            manager.query_one("#snapshot-disabled-reason").render()
        )
        manager.query_one("#snapshot-slots").scroll_visible(animate=False)
        await pilot.pause()
        assert "Unknown" in frame(h.app, "manager-140-busy")
        h.busy = False
        h.unknown = True
        await h.service.refresh()
        manager.query_one("#snapshot-save", Button).press()
        await settle_ui(pilot, h.service)
        assert h.service.view().status == "outcome_unknown"
        assert manager.query_one("#snapshot-save", Button).disabled
        assert "Stop the server" in str(
            manager.query_one("#snapshot-operation-status").render()
        )
        manager.query_one("#snapshot-operation-status").scroll_visible(animate=False)
        await pilot.pause()
        frame(h.app, "manager-140-unknown")


@pytest.mark.asyncio
async def test_detach_does_not_cancel_submitted_save(snapshot_ui):
    from tldw_chatbook.Widgets.llamacpp_snapshot_manager import LlamaCppSnapshotManager

    h = snapshot_ui
    async with h.app.run_test(size=(80, 24)) as pilot:
        await h.app.push_screen(LLMScreen(h.app))
        await pilot.pause()
        manager = h.app.screen.query_one(LlamaCppSnapshotManager)
        await h.service.refresh()
        h.hold.clear()
        manager.query_one("#snapshot-save", Button).press()
        async with asyncio.timeout(5):
            while not [call for call in h.calls if call[0] == "POST"]:
                await pilot.pause(0.01)
        manager.query_one("#snapshot-operation-status").scroll_visible(animate=False)
        await pilot.pause()
        frame(h.app, "manager-80-pending")
        await h.app.pop_screen()
        h.hold.set()
        await settle_ui(pilot, h.service)
        assert len(h.store.list_records().records) == 1


@pytest.mark.asyncio
async def test_delete_confirmation_removes_only_selected_record(snapshot_ui):
    from tldw_chatbook.Widgets.llamacpp_snapshot_manager import LlamaCppSnapshotManager

    h = snapshot_ui
    await h.service.refresh()
    h.service.start_save(0)
    while h.service.view().operation_id:
        await asyncio.sleep(0.01)
    async with h.app.run_test(size=(140, 45)) as pilot:
        await h.app.push_screen(LLMScreen(h.app))
        await pilot.pause()
        manager = h.app.screen.query_one(LlamaCppSnapshotManager)
        manager.query_one("#snapshot-delete", Button).press()
        await pilot.pause()
        text = frame(h.app, "manager-140-delete")
        assert "Permanently" in text and "5 bytes" in text
        h.app.screen.query_one("#confirm-button", Button).press()
        async with asyncio.timeout(5):
            while h.store.list_records().records:
                await pilot.pause(0.01)
        assert not [call for call in h.calls if call[2] == b"action=restore"]


@pytest.mark.asyncio
async def test_launcher_refresh_does_not_advance_dirty_preference_baseline(snapshot_ui):
    from tldw_chatbook.LLM_Management import snapshot_settings as preferences
    from tldw_chatbook.Widgets.llamacpp_snapshot_manager import LlamaCppSnapshotManager

    h = snapshot_ui
    async with h.app.run_test(size=(140, 45)) as pilot:
        await h.app.push_screen(LLMScreen(h.app))
        await pilot.pause()
        manager = h.app.screen.query_one(LlamaCppSnapshotManager)
        manager.query_one(Collapsible).collapsed = False
        manager.query_one("#snapshot-keep", Input).value = "20"
        newer = preferences.SnapshotPreferences(enabled=True, keep_count=12)
        await asyncio.to_thread(preferences.save_snapshot_preferences, newer)
        manager.request_refresh()
        await pilot.pause(0.2)
        manager.query_one("#snapshot-apply", Button).press()
        await pilot.pause(0.2)
        assert preferences.load_snapshot_preferences() == newer
        assert "Reload" in str(
            manager.query_one("#snapshot-preferences-result").render()
        )


@pytest.mark.asyncio
async def test_storage_location_is_readonly_and_materialized_only_in_details(
    snapshot_ui,
):
    from tldw_chatbook.Widgets.llamacpp_snapshot_manager import LlamaCppSnapshotManager

    h = snapshot_ui
    async with h.app.run_test(size=(140, 45)) as pilot:
        await h.app.push_screen(LLMScreen(h.app))
        await pilot.pause()
        manager = h.app.screen.query_one(LlamaCppSnapshotManager)
        assert str(h.store.root) not in str(
            manager.query_one("#snapshot-details").render()
        )
        manager.query_one(Collapsible).collapsed = False
        await pilot.pause()
        assert str(h.store.root) in str(manager.query_one("#snapshot-details").render())
        assert not list(manager.query("#snapshot-details Input"))
        manager.query_one("#snapshot-details").scroll_visible(animate=False)
        await pilot.pause()
        frame(h.app, "manager-140-details")
        manager.query_one(Collapsible).collapsed = True
        await pilot.pause()
        assert str(h.store.root) not in str(
            manager.query_one("#snapshot-details").render()
        )


@pytest.mark.asyncio
async def test_known_empty_slot_disables_save_with_reason(snapshot_ui, monkeypatch):
    from dataclasses import replace

    from tldw_chatbook.Widgets.llamacpp_snapshot_manager import LlamaCppSnapshotManager

    h = snapshot_ui
    original = h.service.view

    def known_empty():
        view = original()
        return replace(
            view,
            slots=tuple(slot.model_copy(update={"tokens": 0}) for slot in view.slots),
        )

    monkeypatch.setattr(h.service, "view", known_empty)
    async with h.app.run_test(size=(140, 45)) as pilot:
        await h.app.push_screen(LLMScreen(h.app))
        await pilot.pause()
        await h.service.refresh()
        manager = h.app.screen.query_one(LlamaCppSnapshotManager)
        assert manager.query_one("#snapshot-save", Button).disabled
        assert "empty" in str(manager.query_one("#snapshot-disabled-reason").render())


@pytest.mark.asyncio
async def test_catalog_paging_uses_no_http_and_screen_reentry_refreshes(snapshot_ui):
    from textual.screen import Screen

    from Tests.LLM_Management.snapshot_fixtures import test_evidence as evidence
    from tldw_chatbook.LLM_Management.snapshot_models import SlotReceipt
    from tldw_chatbook.Widgets.llamacpp_snapshot_manager import LlamaCppSnapshotManager

    h = snapshot_ui
    for index in range(51):
        working = h.store.reserve_save("catalog-launch", 0)
        working.path.write_bytes(b"cache")
        h.store.commit_save(
            working,
            SlotReceipt(slot_id=0, filename=working.path.name, tokens=7, bytes=5),
            evidence(),
            f"Model {index}",
            1000,
        )
    async with h.app.run_test(size=(140, 45)) as pilot:
        await h.app.push_screen(LLMScreen(h.app))
        await pilot.pause(0.2)
        manager = h.app.screen.query_one(LlamaCppSnapshotManager)
        await h.service.refresh()
        await pilot.pause()
        before = len(h.calls)
        manager.query_one("#snapshot-next", Button).press()
        await pilot.pause(0.2)
        assert len(h.service.view().catalog.records) == 1
        assert len(h.calls) == before
        manager.query_one("#snapshot-previous", Button).press()
        await pilot.pause(0.2)
        assert len(h.service.view().catalog.records) == 50
        assert len(h.calls) == before
        await h.app.push_screen(Screen())
        await h.app.pop_screen()
        await pilot.pause(0.2)
        assert len(h.calls) > before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", [{"keep_count": 0}, {"keep_count": "10"}, {"enabled": "yes"}]
)
async def test_malformed_preferences_keep_models_catalog_usable_and_reload_recovers(
    snapshot_ui, bad
):
    from tldw_chatbook import config
    from tldw_chatbook.LLM_Management import snapshot_settings as preferences
    from tldw_chatbook.Widgets.llamacpp_snapshot_manager import LlamaCppSnapshotManager

    h = snapshot_ui
    await h.service.refresh()
    h.service.start_save(0)
    await settle_ui_without_pilot(h.service)
    assert config.apply_settings_mutation_to_cli_config(
        {"llamacpp_snapshots": bad}
    ).fully_applied
    async with h.app.run_test(size=(80, 24)) as pilot:
        await h.app.push_screen(LLMScreen(h.app))
        await pilot.pause()
        manager = h.app.screen.query_one(LlamaCppSnapshotManager)
        assert manager.query_one("#snapshot-save", Button).disabled
        assert manager.query_one("#snapshot-restore", Button).disabled
        assert manager.query_one("#snapshot-apply", Button).disabled
        assert manager.query_one("#snapshot-keep", Input).disabled
        assert manager.query_one("#snapshot-keep", Input).value == ""
        assert "Advanced Config" in str(
            manager.query_one("#snapshot-disabled-reason").render()
        )
        assert not manager.query_one("#snapshot-delete", Button).disabled
        before = len(h.calls)
        await h.service.browse_catalog()
        assert len(h.calls) == before
        manager.query_one("#snapshot-delete", Button).press()
        await pilot.pause()
        h.app.screen.query_one("#confirm-button", Button).press()
        await pilot.pause()
        assert not h.store.list_records().records
        await manager._save_preferences(True)  # Still malformed: no defaults accepted.
        assert manager.query_one("#snapshot-apply", Button).disabled
        assert preferences.save_snapshot_preferences(
            preferences.SnapshotPreferences(enabled=True, keep_count=23)
        )
        await manager._save_preferences(True)
        assert manager.query_one("#snapshot-keep", Input).value == "23"
        assert not manager.query_one("#snapshot-save", Button).disabled
        assert config.apply_settings_mutation_to_cli_config(
            {"llamacpp_snapshots": {"keep_count": True}}
        ).fully_applied
        await manager._refresh()
        assert manager.query_one("#snapshot-save", Button).disabled
        assert manager.query_one("#snapshot-apply", Button).disabled


async def settle_ui_without_pilot(service):
    async with asyncio.timeout(5):
        while service.view().operation_id is not None:
            await asyncio.sleep(0.01)
