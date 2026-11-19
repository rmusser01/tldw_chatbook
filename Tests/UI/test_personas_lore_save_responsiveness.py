"""TASK-36932: a Lore settings save releases the owning screen message queue."""

import asyncio
import threading
from contextlib import asynccontextmanager

import pytest
from textual.widgets import Button, Input, Static, TabbedContent
from textual.worker import WorkerCancelled

from Tests.private_profile import private_profile_test
from Tests.UI.roleplay_frame_harness import (
    PersonasTestApp,
    select_roleplay_item,
    visible_item_ids,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.CCP_Modules import ccp_character_handler
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    PersonasLibraryPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_lore_detail import (
    PersonasLoreDetailWidget,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(120)]


async def until(predicate, seconds=20):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.01)


class SaveHold:
    """Hold only the actual manager write boundary, then delegate to SQLite."""

    def __init__(self, monkeypatch, *, held=True, fail=False):
        self.entered = threading.Event()
        self.finished = threading.Event()
        self.release = threading.Event()
        self.calls = []
        if not held:
            self.release.set()
        original = WorldBookManager.update_world_book

        def update(manager, book_id, **kwargs):
            self.calls.append((book_id, dict(kwargs), threading.get_ident()))
            self.entered.set()
            try:
                assert self.release.wait(90), "controlled Lore service hold expired"
                if fail:
                    raise OSError("controlled Lore write failure")
                return original(manager, book_id, **kwargs)
            finally:
                self.finished.set()

        monkeypatch.setattr(WorldBookManager, "update_world_book", update)


@asynccontextmanager
async def lore_app(mock_app_instance, tmp_path, monkeypatch, **hold_options):
    db = CharactersRAGDB(tmp_path / "lore.db", "responsive-lore-settings")
    monkeypatch.setattr(ccp_character_handler, "_default_character_db", lambda: db)
    manager = WorldBookManager(db)
    first = manager.create_world_book("First Lore", description="original")
    second = manager.create_world_book("Second Lore")
    mock_app_instance.chachanotes_db = db
    mock_app_instance.character_persona_scope_service = None
    app = PersonasTestApp(mock_app_instance)
    hold = SaveHold(monkeypatch, **hold_options)
    try:
        async with app.run_test(size=(220, 60)) as pilot:
            screen = app.screen
            await until(lambda: bool(screen.query("#personas-mode-lore")))
            screen.query_one("#personas-mode-lore", Button).press()
            library = screen.query_one(PersonasLibraryPane)
            await until(lambda: f"lore:{first}" in visible_item_ids(library))
            await select_roleplay_item(pilot, library, f"lore:{first}")
            await until(
                lambda: screen.state.selected_entity_id == str(first)
                and bool(screen.query(PersonasLoreDetailWidget))
            )
            detail = screen.query_one(PersonasLoreDetailWidget)
            detail.query_one(TabbedContent).active = "personas-lore-tab-settings"
            await pilot.pause()
            notices = []
            notify = app.notify

            def observed_notify(message, **kwargs):
                notices.append(str(message))
                return notify(message, **kwargs)

            monkeypatch.setattr(app, "notify", observed_notify)
            try:
                yield pilot, screen, detail, manager, first, second, hold, notices
            finally:
                # Release before Textual teardown, also when the assertion is RED.
                hold.release.set()
                if hold.entered.is_set():
                    await until(hold.finished.is_set)
                await pilot.pause()
                for worker in list(app.workers):
                    try:
                        await worker.wait()
                    except WorkerCancelled:
                        pass
                await pilot.pause()
    finally:
        with db.quiesce_connections(timeout_seconds=5):
            pass


async def submit(pilot, detail):
    field = detail.query_one("#personas-lore-name", Input)
    field.focus()
    field.action_select_all()
    await pilot.press(*"Submitted Lore")
    await pilot.pause()
    assert field.value == "Submitted Lore"
    detail.query_one("#personas-lore-settings-save", Button).press()


async def settle_save(pilot, screen, hold):
    await until(hold.finished.is_set)
    # Manager return precedes handler/worker reconciliation and list refresh.
    queued = asyncio.Event()
    screen.call_later(queued.set)
    await queued.wait()
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


@private_profile_test
async def test_held_save_releases_screen_controls_and_ignores_repeat(
    request, mock_app_instance, tmp_path, monkeypatch
):
    async with lore_app(mock_app_instance, tmp_path, monkeypatch) as context:
        pilot, screen, detail, manager, first, _, hold, _ = context
        await submit(pilot, detail)
        await until(hold.entered.is_set)
        worker = next(w for w in pilot.app.workers if w.group == "personas-lore-save")
        assert (
            "Submitted Lore" not in worker.description
        ), "Lore text leaked into worker diagnostics"
        assert worker.name == "personas-lore-settings-save"
        app_callback, screen_callback = asyncio.Event(), asyncio.Event()
        pilot.app.call_later(app_callback.set)
        screen.call_later(screen_callback.set)
        rail = screen.query_one("#personas-library-rail-collapse", Button)
        assert not rail.disabled
        rail.press()
        detail.query_one("#personas-lore-settings-save", Button).press()
        await asyncio.wait_for(app_callback.wait(), 5)
        try:
            await asyncio.wait_for(screen_callback.wait(), 3)
        except TimeoutError:
            pass
        assert screen_callback.is_set(), "Lore Save retained the screen message queue"
        await until(lambda: screen._library_rail_collapsed)
        assert not hold.finished.is_set()
        hold.release.set()
        await settle_save(pilot, screen, hold)
        saved = manager.get_world_book(first)
        assert saved["name"] == "Submitted Lore" and saved["version"] == 2
        assert len(hold.calls) == 1
        assert hold.calls[0][2] != threading.get_ident()
        assert hold.calls[0][1]["expected_version"] == 1


@pytest.mark.parametrize("conflict", [False, True])
@private_profile_test
async def test_ordinary_save_reports_real_sqlite_result(
    request, mock_app_instance, tmp_path, monkeypatch, conflict
):
    async with lore_app(
        mock_app_instance, tmp_path, monkeypatch, held=False
    ) as context:
        pilot, screen, detail, manager, first, _, hold, _ = context
        if conflict:
            manager.update_world_book(first, name="External Lore", expected_version=1)
        await submit(pilot, detail)
        await settle_save(pilot, screen, hold)
        saved = manager.get_world_book(first)
        status = str(detail.query_one("#personas-lore-status", Static).render())
        assert saved["name"] == ("External Lore" if conflict else "Submitted Lore")
        assert saved["version"] == 2
        assert ("Save failed" in status) is conflict
        assert detail.query_one("#personas-lore-name", Input).value == "Submitted Lore"


@private_profile_test
async def test_newer_raw_settings_survive_and_next_save_uses_saved_version(
    request, mock_app_instance, tmp_path, monkeypatch
):
    async with lore_app(mock_app_instance, tmp_path, monkeypatch) as context:
        pilot, screen, detail, manager, first, _, hold, _ = context
        await submit(pilot, detail)
        await until(hold.entered.is_set)
        name = detail.query_one("#personas-lore-name", Input)
        budget = detail.query_one("#personas-lore-token-budget", Input)
        name.value = "Submitted Lore  "
        budget.value = "invalid"
        hold.release.set()
        await settle_save(pilot, screen, hold)
        assert name.value == "Submitted Lore  " and budget.value == "invalid"
        assert manager.get_world_book(first)["token_budget"] == 500
        status = str(detail.query_one("#personas-lore-status", Static).render())
        assert "newer" in status.lower() and "not saved" in status.lower()
        assert screen._selected_lore_book_version == 2
        budget.value = "750"
        name.value = "Later Lore"
        await pilot.pause()
        detail.query_one("#personas-lore-settings-save", Button).press()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        saved = manager.get_world_book(first)
        assert saved["name"] == "Later Lore" and saved["token_budget"] == 750
        assert saved["version"] == 3


@pytest.mark.parametrize("retirement", ["selection", "same-book", "panel-failure"])
@private_profile_test
async def test_retired_save_reports_captured_book_without_replacing_current_panel(
    request, mock_app_instance, tmp_path, monkeypatch, retirement
):
    async with lore_app(
        mock_app_instance,
        tmp_path,
        monkeypatch,
        fail=retirement == "panel-failure",
    ) as context:
        pilot, screen, detail, manager, first, second, hold, notices = context
        list_entered, list_release = asyncio.Event(), asyncio.Event()
        if retirement == "selection":
            render = screen._render_lore_rows

            async def held_render(*args, **kwargs):
                # Run the actual refresh, then hold its completion before the
                # save's continuation can publish its loaded-row marker.
                await render(*args, **kwargs)
                if not list_entered.is_set():
                    list_entered.set()
                    await list_release.wait()

            monkeypatch.setattr(screen, "_render_lore_rows", held_render)
        try:
            await submit(pilot, detail)
            await until(hold.entered.is_set)
            if retirement == "selection":
                hold.release.set()
                await until(list_entered.is_set)
            if retirement == "panel-failure":
                await detail.remove()
                # Force an actual panel replacement, including its existing
                # screen-owned demand-mount readiness cache.
                screen._ready_center_views.pop("lore-detail")
                await screen._select_lore_entry(str(first), "First Lore")
                replacement = screen.query_one(PersonasLoreDetailWidget)
                assert replacement is not detail
            else:
                await screen._select_lore_entry(
                    str(second if retirement == "selection" else first),
                    "Second Lore" if retirement == "selection" else "First Lore",
                )
                replacement = detail
            replacement.query_one("#personas-lore-name", Input).value = "Current draft"
            replacement.set_status("Current panel status")
            current_id = screen.state.selected_entity_id
            current_record = screen._selected_lore_book
            current_version = screen._selected_lore_book_version
            hold.release.set()
            list_release.set()
            await settle_save(pilot, screen, hold)
        finally:
            list_release.set()
        assert screen.state.selected_entity_id == current_id
        assert (
            screen.query_one(PersonasLibraryPane).item_list.active_id
            == f"lore:{current_id}"
        )
        assert screen._selected_lore_book is current_record
        assert screen._selected_lore_book_version == current_version
        assert (
            replacement.query_one("#personas-lore-name", Input).value == "Current draft"
        )
        assert "Current panel status" in str(
            replacement.query_one("#personas-lore-status", Static).render()
        )
        saved = manager.get_world_book(first)
        assert saved["name"] == (
            "First Lore" if retirement == "panel-failure" else "Submitted Lore"
        )
        assert manager.get_world_book(second)["name"] == "Second Lore"
        assert any(
            "First Lore" in text
            and ("failed" if retirement == "panel-failure" else "saved") in text.lower()
            for text in notices
        )


@private_profile_test
async def test_cancelling_save_keeps_admission_until_native_write_finishes(
    request, mock_app_instance, tmp_path, monkeypatch
):
    async with lore_app(mock_app_instance, tmp_path, monkeypatch) as context:
        pilot, screen, detail, manager, first, _, hold, _ = context
        await submit(pilot, detail)
        await until(hold.entered.is_set)
        worker = next(w for w in pilot.app.workers if w.group == "personas-lore-save")
        worker.cancel()
        detail.query_one("#personas-lore-settings-save", Button).press()
        callback = asyncio.Event()
        screen.call_later(callback.set)
        await asyncio.wait_for(callback.wait(), 5)
        assert not hold.finished.is_set()
        hold.release.set()
        await until(hold.finished.is_set)
        with pytest.raises(WorkerCancelled):
            await worker.wait()
        await pilot.pause()
        saved = manager.get_world_book(first)
        assert saved["name"] == "Submitted Lore" and saved["version"] == 2
        assert len(hold.calls) == 1


@private_profile_test
async def test_cancelled_before_entry_save_allows_a_later_explicit_retry(
    request, mock_app_instance, tmp_path, monkeypatch
):
    async with lore_app(mock_app_instance, tmp_path, monkeypatch) as context:
        pilot, screen, detail, manager, first, _, hold, _ = context
        run_worker = screen.run_worker
        cancelled = []

        def cancel_first_save(work, **kwargs):
            worker = run_worker(work, **kwargs)
            if kwargs.get("group") == "personas-lore-save" and not cancelled:
                worker.cancel()
                cancelled.append(worker)
            return worker

        monkeypatch.setattr(screen, "run_worker", cancel_first_save)
        await submit(pilot, detail)
        await pilot.pause()
        assert cancelled and not hold.entered.is_set()
        assert cancelled[0].name == "personas-lore-settings-save"
        assert "Submitted Lore" not in cancelled[0].description
        detail.query_one("#personas-lore-settings-save", Button).press()
        await until(hold.entered.is_set)
        hold.release.set()
        await settle_save(pilot, screen, hold)
        saved = manager.get_world_book(first)
        assert saved["name"] == "Submitted Lore" and saved["version"] == 2
        assert len(hold.calls) == 1
