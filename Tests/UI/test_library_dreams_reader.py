"""Dreams uses the actual production Library route and existing action owners."""

import asyncio
import time

import pytest
from textual.widgets import Button, OptionList, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _CssTrueDestinationHarness
from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route
from tldw_chatbook.UI.Screens.artifacts_dreams_goals_modal import DreamsGoalsModal
from tldw_chatbook.UI.Screens.artifacts_dreams_modal import DreamsStoryModal
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = pytest.mark.ui


class ProductionArtifactsHarness(_CssTrueDestinationHarness):
    async def on_mount(self):
        route = resolve_screen_route("artifacts")
        screen_class = route.load_screen_class()
        assert screen_class is LibraryScreen
        await self.push_screen(screen_class(self.app_instance))


async def loaded(pilot, controller, selector="#library-artifacts-dreams-actions"):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        await pilot.pause(0.03)
        button = controller.screen.query_one(selector, Button)
        # Data publication can precede layout. Pilot caches the target origin
        # before its own pause, so wait for the actual hit before clicking once.
        if (
            controller.detail is not None
            and not (controller.loading or controller.detail_loading)
            and button.is_mounted
            and button.display
            and button.visible
            and not button.disabled
            and button.region.width > 0
            and button.region.height > 0
            and button.region.offset in pilot.app.screen.size.region
            and pilot.app.get_widget_at(*button.region.offset)[0] is button
        ):
            return
    pytest.fail(f"Dreams did not load: {controller.error} / {controller.detail_error}")


async def mounted_modal(pilot, modal_type, selector):
    """Wait within the route test's existing modal readiness budget."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        modal = pilot.app.screen
        if isinstance(modal, modal_type) and modal.is_mounted and modal.query(selector):
            return modal
        await pilot.pause(0.03)
    pytest.fail(f"{modal_type.__name__} did not mount {selector}")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["story", "goals"])
@private_profile_test
async def test_modal_readiness_waits_for_real_composition(
    tmp_path, request, monkeypatch, kind
):
    from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp

    db = DreamsDB(tmp_path / "dreams.sqlite")
    if kind == "story":
        modal = DreamsStoryModal(
            {"id": 1, "title": "Story", "body": "Body", "url": "https://example.com"},
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: None,
        )
        selector = "#dsm-keep-button"
    else:
        modal = DreamsGoalsModal(dreams_db_getter=lambda: db, on_changed=lambda: None)
        selector = "#dgm-goals"
    entered, release = asyncio.Event(), asyncio.Event()
    compose = modal._compose

    async def held_compose():
        entered.set()
        await release.wait()
        await compose()

    monkeypatch.setattr(modal, "_compose", held_compose)
    app = ConsolidatedCSSApp(css_path=list(APP_STYLESHEETS))
    try:
        async with app.run_test(size=(160, 50)) as pilot:
            mount = app.push_screen(modal)
            waiter = None
            try:
                await asyncio.wait_for(entered.wait(), 5)
                assert app.screen is modal  # Stack identity precedes child mount.
                assert not modal.is_mounted
                assert not modal.query(selector)
                waiter = asyncio.create_task(
                    mounted_modal(pilot, type(modal), selector)
                )
                await asyncio.sleep(
                    0
                )  # Let the readiness check reach the held boundary.
                assert not waiter.done()
                release.set()
                assert await asyncio.wait_for(waiter, 5) is modal
                assert modal.is_mounted
                assert modal.query_one(selector).is_mounted
            finally:
                release.set()
                await asyncio.wait_for(mount, 5)
                if waiter is not None:
                    await asyncio.gather(waiter, return_exceptions=True)
    finally:
        db.close_all()


@pytest.mark.asyncio
@private_profile_test
async def test_production_library_dreams_story_actions_goals_and_literal_notice(
    tmp_path, request, monkeypatch
):
    from tldw_chatbook.Dreams import settings

    original = settings.dreams_setting
    monkeypatch.setattr(
        settings,
        "dreams_setting",
        lambda key, *args: True if key == "enabled" else original(key, *args),
    )
    app = _build_test_app(configured_default="library")
    db = DreamsDB(tmp_path / "dreams.sqlite", "library-ui")
    app.dreams_db = db
    cid = db.create_collection("2026-10-12", "manual", "d")
    sid = db.insert_story(
        cid,
        title="[broken]Literal title[/other]",
        url="https://example.com/story",
        snippet="snippet",
        body="A science story",
        status="complete",
        source="web",
        kind="content",
        event_date=None,
        location=None,
        matched_topics=["science"],
        query="science",
    )
    notice = "[link=https://evil.invalid]Fallback queries[/link] [/unmatched]"
    with db.transaction() as conn:
        conn.execute(
            "UPDATE dreams_collections SET status='partial', degradation_notes=?",
            (notice,),
        )
    host = ProductionArtifactsHarness(app, "library")
    try:
        async with host.run_test(size=(160, 50)) as pilot:
            screen = host.screen
            await screen._select_library_rail_row("artifacts-dreams")
            c = screen._artifacts_controller
            await loaded(pilot, c)
            assert (
                screen.query_one("#library-artifacts-list", OptionList).option_count
                == 1
            )
            assert (
                screen.query_one("#library-artifacts-title", Static).renderable
                == "[broken]Literal title[/other]"
            )
            rendered = screen.query_one(
                "#library-artifacts-degradation", Static
            ).renderable
            assert (
                rendered.plain == "Collection notice: " + notice and not rendered.spans
            )
            assert await pilot.click("#library-artifacts-dreams-actions")
            modal = await mounted_modal(pilot, DreamsStoryModal, "#dsm-keep-button")
            assert isinstance(modal, DreamsStoryModal)
            assert modal._story["id"] == sid
            for name in (
                "keep",
                "dive",
                "export",
                "ingest",
                "track",
                "watch",
                "more",
                "less",
            ):
                assert modal.query_one(f"#dsm-{name}-button", Button).display
            await pilot.press("k")
            assert db.get_story(sid)["kept"] == 1
            await pilot.press("q")
            await loaded(pilot, c, "#library-artifacts-dreams-goals")
            assert c.detail.dream_story["kept"] == 1
            assert await pilot.click("#library-artifacts-dreams-goals")
            await mounted_modal(pilot, DreamsGoalsModal, "#dgm-goals")
            assert isinstance(host.screen, DreamsGoalsModal)
            await pilot.press("q")
    finally:
        db.close_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
@private_profile_test
async def test_library_dreams_lazy_owner_native_close_before_worker_exit(
    tmp_path, request, monkeypatch, borrowed
):
    import asyncio
    import sqlite3
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from tldw_chatbook import app_service_wiring
    from tldw_chatbook.DB.base_db import BaseDB
    from tldw_chatbook.Dreams import settings
    from tldw_chatbook.Library.library_artifacts_catalog import LibraryArtifactsCatalog
    from tldw_chatbook.Library.library_artifacts_state import ArtifactScope

    path = tmp_path / "lazy-dreams.sqlite"
    seed = DreamsDB(path)
    seed.create_collection("2026-10-12", "manual", "d")
    seed.close_all()
    monkeypatch.setattr(app_service_wiring, "get_dreams_db_path", lambda: path)
    original_setting = settings.dreams_setting
    monkeypatch.setattr(
        settings,
        "dreams_setting",
        lambda key, *args: True if key == "enabled" else original_setting(key, *args),
    )
    captured = []
    acquire = BaseDB._get_connection

    def capture(self, *args, **kwargs):
        connection = acquire(self, *args, **kwargs)
        if type(self) is DreamsDB:
            captured.append(connection)
        return connection

    monkeypatch.setattr(BaseDB, "_get_connection", capture)
    app = SimpleNamespace(dreams_db=None, _dreams_db_lock=threading.Lock())
    getter = lambda: app_service_wiring.ServiceWiringMixin.get_dreams_db(app)
    catalog = LibraryArtifactsCatalog(
        subscriptions_db=None, chachanotes_db=None, dreams_db_getter=getter
    )
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1)
    previous = loop._default_executor
    loop.set_default_executor(pool)
    try:
        if borrowed:
            await loop.run_in_executor(pool, getter)
        page = await LibraryScreen._run_library_service_call(
            catalog.read_page, ArtifactScope(view="dreams")
        )
        assert page.total == 1

        def verify():
            assert len(captured) == 1
            if borrowed:
                assert captured[0].execute("SELECT 1").fetchone()[0] == 1
            else:
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    captured[0].execute("SELECT 1")
            app.dreams_db.close_all()

        await loop.run_in_executor(pool, verify)
        assert (
            await loop.run_in_executor(pool, lambda: "worker alive") == "worker alive"
        )
    finally:
        await loop.run_in_executor(
            pool, lambda: app.dreams_db.close_all() if app.dreams_db else None
        )
        loop._default_executor = previous
        pool.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["story", "goals", "guardian"])
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
@private_profile_test
async def test_feature_modals_keep_controls_inside_production_viewport(
    tmp_path, request, kind, size
):
    from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
    from tldw_chatbook.DB.Guardian_DB import GuardianDB
    from tldw_chatbook.UI.Screens.settings_screen import GuardianRuleEditModal

    db = (
        GuardianDB(tmp_path / "guardian.sqlite")
        if kind == "guardian"
        else DreamsDB(tmp_path / "dreams.sqlite")
    )
    if kind == "guardian":
        modal = GuardianRuleEditModal(db_getter=lambda: db, on_changed=lambda: None)
        selector = "#grm-dialog"
    elif kind == "goals":
        modal = DreamsGoalsModal(dreams_db_getter=lambda: db, on_changed=lambda: None)
        selector = "#dgm-dialog"
    else:
        modal = DreamsStoryModal(
            {
                "id": 1,
                "title": "Readable story",
                "body": "A short story.",
                "url": "https://example.com/story",
                "query": "science",
            },
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: None,
        )
        selector = "#dsm-dialog"
    app = ConsolidatedCSSApp(css_path=list(APP_STYLESHEETS))
    try:
        async with app.run_test(size=size) as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            app.save_screenshot(f"{kind}-{size[0]}x{size[1]}.svg", path=str(tmp_path))
            panel = modal.query_one(selector)
            assert panel.region.width <= size[0] and panel.region.height <= size[1]
            for button in modal.query(Button):
                if not button.display:
                    continue
                button.scroll_visible(animate=False)
                await pilot.pause()
                assert panel.content_region.x <= button.region.x
                assert button.region.right <= panel.content_region.right, (
                    button.id,
                    button.region,
                    panel.content_region,
                )
                assert (
                    button.region.intersection(panel.content_region).height
                    == button.region.height
                )
            await pilot.press("escape")
            assert app.screen is not modal
    finally:
        db.close_all() if kind != "guardian" else db.close()
