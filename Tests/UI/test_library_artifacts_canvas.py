"""Reports are a Library reader with real storage and production styles."""

from contextlib import asynccontextmanager

import pytest
from textual.widgets import Input, OptionList

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _CssTrueDestinationHarness
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.ui


@asynccontextmanager
async def artifact_library(tmp_path, *, size=(160, 50), theme="textual-dark"):
    app = _build_test_app(configured_default="library")
    db = CharactersRAGDB(tmp_path / "library.sqlite", client_id="artifact-ui")
    app.chachanotes_db = db
    watch = int(
        WatchlistBundleService(app.subscriptions_db).create("Weekly reading")["id"]
    )
    for index in range(3):
        record = app.subscriptions_db.insert_briefing(watch)
        app.subscriptions_db.update_briefing(
            record,
            status="complete",
            body_markdown=f"# Report {index}\n\nA readable report.",
        )
    host = _CssTrueDestinationHarness(app, "library")
    host.theme = theme
    try:
        async with host.run_test(size=size) as pilot:
            screen = host.screen
            await screen._select_library_rail_row("artifacts-reports")
            for _ in range(100):
                await pilot.pause(0.03)
                controller = getattr(screen, "_artifacts_controller", None)
                if controller and controller.detail is not None:
                    break
            yield screen, pilot
    finally:
        db.close_connection()


@pytest.mark.asyncio
@private_profile_test
async def test_enter_reader_back_preserves_row_then_arrows_work(tmp_path, request):
    async with artifact_library(tmp_path) as (screen, pilot):
        rows = screen.query_one("#library-artifacts-list", OptionList)
        assert rows.option_count == 3
        assert rows.region.height > 0
        rows.focus()
        await pilot.press("down")
        selected = screen._artifacts_controller.selected
        await pilot.press("enter")
        assert screen.focused.id == "library-artifacts-body"
        await pilot.press("down")
        assert screen._artifacts_controller.selected == selected
        await pilot.press("escape")
        assert screen.focused is rows
        await pilot.press("down")
        assert screen._artifacts_controller.selected != selected


@pytest.mark.asyncio
@private_profile_test
async def test_escape_clears_search_and_applied_query(tmp_path, request):
    async with artifact_library(tmp_path) as (screen, pilot):
        field = screen.query_one("#library-artifacts-search", Input)
        field.focus()
        field.value = "zzzz-not-found"
        for _ in range(100):
            await pilot.pause(0.02)
            if (
                screen._artifacts_controller.page
                and screen._artifacts_controller.page.scope.query == field.value
            ):
                break
        assert screen._artifacts_controller.page.total == 0
        await pilot.press("escape")
        for _ in range(100):
            await pilot.pause(0.02)
            if (
                screen._artifacts_controller.page
                and not screen._artifacts_controller.page.scope.query
            ):
                break
        assert field.value == ""
        assert screen._artifacts_controller.page.total == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,theme",
    [
        ((160, 50), "textual-dark"),
        ((100, 40), "textual-light"),
        ((64, 30), "textual-dark"),
        ((50, 25), "textual-light"),
    ],
)
@private_profile_test
async def test_reader_layout_uses_visible_panes_and_returns_to_items(
    tmp_path, request, size, theme
):
    async with artifact_library(tmp_path, size=size, theme=theme) as (screen, pilot):
        c = screen._artifacts_controller
        c.focus_items()
        await pilot.pause()
        rows = screen.query_one("#library-artifacts-list", OptionList)
        assert rows.region.width >= 30
        assert rows.region.height >= 3
        assert rows.region.right <= screen.size.width
        await pilot.press("enter")
        await pilot.pause()
        body = screen.query_one("#library-artifacts-body")
        assert body.region.width > 20
        assert body.region.height >= 3
        assert body.region.right <= screen.size.width
        assert screen.focused is body
        await pilot.press("escape")
        assert screen.focused is rows


@pytest.mark.asyncio
@private_profile_test
async def test_keep_delete_watchlist_read_and_export_durable_copy(tmp_path, request):
    from tldw_chatbook.Third_Party.textual_fspicker import FileSave

    async with artifact_library(tmp_path) as (screen, pilot):
        c = screen._artifacts_controller
        body = c.detail.body
        live_id = c.selected.native_id
        watch = screen.app_instance.subscriptions_db.get_briefing(live_id)[
            "watchlist_id"
        ]
        c.action("keep")
        for _ in range(100):
            await pilot.pause(0.03)
            if c.detail and c.selected.source == "kept_report" and not c.busy:
                break
        assert c.selected.source == "kept_report"
        WatchlistBundleService(screen.app_instance.subscriptions_db).delete(watch)
        c.action("kept")
        for _ in range(100):
            await pilot.pause(0.03)
            if c.page and c.page.scope.kept_only and c.detail:
                break
        assert c.detail.body == body
        assert c.page.total == 1
        c.action("export")
        for _ in range(100):
            await pilot.pause(0.03)
            if isinstance(screen.app.screen, FileSave):
                break
        assert isinstance(screen.app.screen, FileSave)
        await pilot.pause(0.15)
        destination = tmp_path / "kept.md"
        screen.app.screen.dismiss(destination)
        for _ in range(100):
            await pilot.pause(0.03)
            if destination.exists():
                break
        assert body in destination.read_text()


@pytest.mark.asyncio
@private_profile_test
async def test_late_detail_cannot_replace_current_selection(
    tmp_path, request, monkeypatch
):
    import threading

    from tldw_chatbook.Library.library_artifacts_catalog import LibraryArtifactsCatalog

    async with artifact_library(tmp_path) as (screen, pilot):
        c = screen._artifacts_controller
        old_key, new_key = (row.key for row in c.page.items[:2])
        entered, release = threading.Event(), threading.Event()
        original = LibraryArtifactsCatalog.read_detail

        def delayed(catalog, key):
            if key == old_key:
                entered.set()
                assert release.wait(10)
            return original(catalog, key)

        monkeypatch.setattr(LibraryArtifactsCatalog, "read_detail", delayed)
        try:
            c.select(old_key, refresh=True)
            for _ in range(100):
                await pilot.pause(0.01)
                if entered.is_set():
                    break
            assert entered.is_set()
            c.select(new_key)
            for _ in range(100):
                await pilot.pause(0.01)
                if c.detail and c.detail.key == new_key:
                    break
            release.set()
            await pilot.pause(0.1)
            assert c.detail.key == c.selected == new_key
        finally:
            release.set()


@pytest.mark.asyncio
@private_profile_test
async def test_resume_preserves_artifact_query_independently_of_rail(tmp_path, request):
    from textual.screen import Screen

    async with artifact_library(tmp_path) as (screen, pilot):
        c = screen._artifacts_controller
        field = screen.query_one("#library-artifacts-search", Input)
        field.value = "Weekly"
        for _ in range(100):
            await pilot.pause(0.02)
            if c.page and c.page.scope.query == "Weekly" and not c.loading:
                break
        await screen.app.push_screen(Screen())
        await pilot.pause()
        screen.app.pop_screen()
        await pilot.pause(0.2)
        assert c.scope.query == field.value == "Weekly"
        assert c.page.total == 3


@pytest.mark.asyncio
@private_profile_test
async def test_failed_detail_exposes_retry_and_recovers_same_identity(
    tmp_path, request, monkeypatch
):
    from textual.widgets import Button

    from tldw_chatbook.Library.library_artifacts_catalog import LibraryArtifactsCatalog

    async with artifact_library(tmp_path) as (screen, pilot):
        c = screen._artifacts_controller
        key = c.selected
        original = LibraryArtifactsCatalog.read_detail
        monkeypatch.setattr(LibraryArtifactsCatalog, "read_detail", lambda *_: None)
        c.select(key, refresh=True)
        for _ in range(100):
            await pilot.pause(0.02)
            if c.detail_error:
                break
        retry = screen.query_one("#library-artifacts-retry", Button)
        assert retry.display
        monkeypatch.setattr(LibraryArtifactsCatalog, "read_detail", original)
        retry.press()
        for _ in range(100):
            await pilot.pause(0.02)
            if c.detail:
                break
        assert c.detail.key == key
        assert not c.detail_error


@pytest.mark.asyncio
@private_profile_test
async def test_rekeep_after_rename_clears_excluding_query_for_exact_saved_copy(
    tmp_path, request
):
    from tldw_chatbook.Subscriptions.briefing_keep import keep_briefing

    async with artifact_library(tmp_path) as (screen, pilot):
        c = screen._artifacts_controller
        db = screen.app_instance.subscriptions_db
        live = c.selected.native_id
        saved = keep_briefing(
            db, screen.app_instance.chachanotes_db, live, origin="manual"
        )
        watch = db.get_briefing(live)["watchlist_id"]
        WatchlistBundleService(db).rename(watch, "Renamed source")
        screen.query_one("#library-artifacts-search", Input).value = "Renamed"
        for _ in range(100):
            await pilot.pause(0.02)
            if (
                c.detail
                and c.page
                and c.page.scope.query == "Renamed"
                and not c.loading
            ):
                break
        c.action("keep")
        for _ in range(100):
            await pilot.pause(0.02)
            if c.detail and c.detail.key.source == "kept_report" and not c.loading:
                break
        assert c.selected.native_id == saved["kept_id"]
        assert c.selected.source == "kept_report"
        assert c.scope.query == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["changed", "deleted"])
@private_profile_test
async def test_action_revalidation_clears_stale_detail_and_exposes_retry(
    tmp_path, request, change
):
    from textual.widgets import Button, Markdown

    async with artifact_library(tmp_path) as (screen, pilot):
        c = screen._artifacts_controller
        key = c.selected
        db = screen.app_instance.subscriptions_db
        watch = db.get_briefing(key.native_id)["watchlist_id"]
        manager = WatchlistBundleService(db)
        if change == "changed":
            manager.rename(watch, "Updated source")
        else:
            manager.delete(watch)
        c.action("keep")
        for _ in range(100):
            await pilot.pause(0.02)
            if not c.busy:
                break
        assert c.detail is None
        assert c.detail_error and "Retry" in c.detail_error
        assert not screen.query_one("#library-artifacts-markdown", Markdown).display
        retry = screen.query_one("#library-artifacts-retry", Button)
        assert retry.display and not retry.disabled
        retry.press()
        for _ in range(100):
            await pilot.pause(0.02)
            if not c.loading and not c.detail_loading:
                break
        if change == "changed":
            assert c.detail and c.detail.key == key
            assert not c.detail_error
            assert c.page.items[0].title == "Updated source"
        else:
            assert c.detail is None
            assert c.error and "missing" in c.error


@pytest.mark.asyncio
@private_profile_test
async def test_returning_to_reports_preserves_details_mode(tmp_path, request):
    from textual.widgets import Markdown, Static

    async with artifact_library(tmp_path) as (screen, pilot):
        c = screen._artifacts_controller
        key = c.selected
        c.action("details")
        await pilot.pause()
        await screen._select_library_rail_row("artifacts-chatbooks")
        await pilot.pause()
        await screen._select_library_rail_row("artifacts-reports")
        for _ in range(100):
            await pilot.pause(0.02)
            if c.detail and c.detail.key == key and c.mode == "details":
                break
        assert c.mode == "details"
        assert not screen.query_one("#library-artifacts-markdown", Markdown).display
        content = screen.query_one("#library-artifacts-content", Static)
        assert content.display and "Created:" in str(content.renderable)
