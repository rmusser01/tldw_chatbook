"""Dreams rows on the canonical Library route (ADR172/196)."""

import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from textual.widgets import OptionList, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_dreams_reader import ProductionArtifactsHarness
from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = pytest.mark.ui


def _enable_dreams(monkeypatch):
    """Flip ``[dreams] enabled`` the way ``test_dreams_settings`` does.

    ``dreams_setting`` resolves through ``get_cli_setting``'s own config
    cache, which the app factory's snapshot overrides do not reach, so the
    monkeypatch (not ``config_overrides``) is the deterministic seam.
    """
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default: True if key == "enabled" else default,
    )


def _seed_story_db(tmp_path) -> DreamsDB:
    db = DreamsDB(tmp_path / "dreams.sqlite", "artifacts-dreams")
    collection = db.create_collection("2026-09-22", "scheduled", "digest")
    db.insert_story(
        collection,
        title="Cheap flights to Japan",
        url="https://example.com/flights",
        snippet="Fares from $89",
        body="A story about fares.",
        status="complete",
        source="web",
        kind="deal",
        event_date=None,
        location="Japan",
        matched_topics=["visit japan"],
        query="cheap flights japan",
    )
    return db


def _seed_failed_cycle_db(tmp_path) -> DreamsDB:
    db = DreamsDB(tmp_path / "dreams.sqlite", "artifacts-dreams")
    collection = db.create_collection("2026-09-22", "scheduled", "digest")
    db.set_collection_status(collection, "failed")
    return db


def _seed_tracked_item_db(tmp_path) -> DreamsDB:
    """One story plus one active tracked question whose latest run changed."""
    db = _seed_story_db(tmp_path)
    item_id = db.create_tracked_item(
        mechanism="question",
        intent="deal",
        query_template="cheap flights japan",
        event_date="2026-10-01",
        cadence_seconds=3600,
    )
    db.insert_track_run(item_id, status="changed", digest_hash="digest")
    db.touch_tracked_checked(item_id, "2026-09-22T00:00:00+00:00")
    return db


async def _settle_artifacts_refreshes(screen, pilot, *, timeout=10.0):
    """Wait on the real Library data/presentation owner, with the old deadline."""
    c = screen._artifacts_controller
    deadline = time.monotonic() + timeout
    while c.loading or c.detail_loading or c.busy:
        assert time.monotonic() < deadline, "Library artifact work never settled"
        await pilot.pause(0.02)
    await pilot.pause()
    await pilot.pause()


async def _wait_for_dreams(screen, pilot, selector, *, attempts=50):
    if (
        screen._artifacts_controller is None
        or screen._artifacts_controller.scope.view != "dreams"
    ):
        await screen._select_library_rail_row("artifacts-dreams")
    c = screen._artifacts_controller
    source = (
        "dream_track"
        if "-track-row-" in selector
        else "dream_collection"
        if "-row-cycle-" in selector
        else "dream_story"
    )
    for _ in range(attempts):
        await pilot.pause(0.05)
        rows = c.page.items if c.page else ()
        selected = next(
            (
                row
                for row in rows
                if row.key.source == source
                and (
                    source == "dream_collection"
                    and selector.rsplit("cycle-", 1)[1] in row.title
                    or source != "dream_collection"
                    and row.key.native_id == int(selector.rsplit("-", 1)[1])
                )
            ),
            None,
        )
        if not c.loading and selected:
            c.select(selected.key)
            await _settle_artifacts_refreshes(screen, pilot)
            assert c.detail is not None
            return
    raise AssertionError(f"production Library never displayed {selector}")


@asynccontextmanager
async def _mounted_dreams(app):
    host = ProductionArtifactsHarness(app, "library")
    try:
        async with host.run_test(size=(160, 50)) as pilot:
            assert isinstance(host.screen, LibraryScreen)
            yield host, host.screen, pilot
    finally:
        db = getattr(app, "dreams_db", None)
        if db is not None:
            db.close_all()


async def _open_dreams_view(screen, pilot):
    await screen._select_library_rail_row("artifacts-dreams")
    await _settle_artifacts_refreshes(screen, pilot)
    return screen._artifacts_controller


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_seeded_dream_rows_paint_and_mode_label_names_dreams(
    tmp_path, monkeypatch
):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="library")
    app.dreams_db = _seed_story_db(tmp_path)
    async with _mounted_dreams(app) as (_host, screen, pilot):
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-row-1")
        rows = screen.query_one("#library-artifacts-list", OptionList)
        assert rows.region.height >= 1 and rows.option_count == 1
        assert "Cheap flights to Japan" in str(rows.get_option_at_index(0).prompt)
        assert "Dreams" in str(
            screen.query_one("#library-artifacts-retention", Static).renderable
        )
        assert "No Dreams" not in str(
            screen.query_one("#library-artifacts-count", Static).renderable
        )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_failed_cycle_synthetic_row_paints(tmp_path, monkeypatch):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="library")
    app.dreams_db = _seed_failed_cycle_db(tmp_path)
    async with _mounted_dreams(app) as (_host, screen, pilot):
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-row-cycle-2026-09-22")
        rows = screen.query_one("#library-artifacts-list", OptionList)
        assert rows.region.height >= 1 and rows.option_count == 1
        assert "Cycle 2026-09-22" in str(rows.get_option_at_index(0).prompt)
        assert "failed" in str(rows.get_option_at_index(0).prompt)


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_disabled_dreams_renders_disabled_even_with_rows(tmp_path):
    app = _build_test_app(configured_default="library")
    app.dreams_db = _seed_story_db(tmp_path)
    async with _mounted_dreams(app) as (_host, screen, pilot):
        c = await _open_dreams_view(screen, pilot)
        assert c.page.items == ()
        assert screen.query_one("#library-artifacts-list", OptionList).option_count == 0
        assert "Enable Dreams in F4 Settings" in str(
            screen.query_one("#library-artifacts-count", Static).renderable
        )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_enabled_dreams_with_no_rows_renders_none_yet(tmp_path, monkeypatch):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="library")
    app.dreams_db = DreamsDB(tmp_path / "empty-dreams.sqlite", "artifacts-dreams")
    async with _mounted_dreams(app) as (_host, screen, pilot):
        c = await _open_dreams_view(screen, pilot)
        assert c.page.items == ()
        assert "No Dreams yet" in str(
            screen.query_one("#library-artifacts-count", Static).renderable
        )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_missing_dreams_db_degrades_to_unavailable_without_crashing(monkeypatch):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="library")
    app.get_dreams_db = lambda: None
    async with _mounted_dreams(app) as (_host, screen, pilot):
        c = await _open_dreams_view(screen, pilot)
        assert c.page is None and "unavailable" in c.error
        assert "unavailable" in str(
            screen.query_one("#library-artifacts-count", Static).renderable
        )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_unmount_invalidates_late_dreams_apply(tmp_path, monkeypatch):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="library")
    app.dreams_db = _seed_story_db(tmp_path)
    async with _mounted_dreams(app) as (host, screen, pilot):
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-row-1")
        c = screen._artifacts_controller
        generation, profile, scope, page = c.generation, c.profile(), c.scope, c.page
        await host.pop_screen()
        await pilot.pause()
        assert c.disposed and c.generation != generation
        empty = SimpleNamespace(items=(), total=0)
        await c._load_page(
            SimpleNamespace(read_page=lambda *a, **kw: empty),
            generation,
            profile,
            scope,
            None,
            "after",
            None,
        )
        assert c.page is page, "late result cannot clear the retired reader"


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_tracked_rows_render_before_older_dream_rows(tmp_path, monkeypatch):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="library")
    app.dreams_db = _seed_tracked_item_db(tmp_path)
    with app.dreams_db.transaction() as connection:
        connection.execute(
            "UPDATE dreams_collections SET created_at='2000-01-01T00:00:00+00:00'"
        )
    async with _mounted_dreams(app) as (_host, screen, pilot):
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-track-row-1")
        rows = screen.query_one("#library-artifacts-list", OptionList)
        assert rows.option_count == 2 and rows.can_focus
        assert rows.get_option_at_index(0).id == "dream_track:1"
        assert rows.get_option_at_index(1).id == "dream_story:1"
        text = str(rows.get_option_at_index(0).prompt)
        assert "cheap flights japan" in text and "changed" in text


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_tracked_empty_state_has_no_phantom_rows(tmp_path, monkeypatch):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="library")
    app.dreams_db = _seed_story_db(tmp_path)
    async with _mounted_dreams(app) as (_host, screen, pilot):
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-row-1")
        c = screen._artifacts_controller
        assert [row.key.source for row in c.page.items] == ["dream_story"]
        assert screen.query_one("#library-artifacts-list", OptionList).option_count == 1
