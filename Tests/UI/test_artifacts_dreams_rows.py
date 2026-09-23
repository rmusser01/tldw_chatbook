"""Artifacts screen Dreams slot: rows, mode label, empty states, teardown.

Harness copied from ``test_artifacts_screen_reports.py``: a factory-built
TldwCli (never mounted, so the app's post-``_ui_ready`` Dreams wiring never
runs and ``app.dreams_db`` stays whatever the test assigns) wrapped in
``DestinationHarness``, with a real ``DreamsDB`` on ``tmp_path`` for seeding.
"""

import pytest
from textual.widgets import Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen

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


async def _wait_for_dreams(screen, pilot, selector: str, *, attempts: int = 50):
    """Wait until the worker has landed rows AND composed them.

    Waiting on the widget (not just ``screen._dreams``) is deliberate:
    ``_apply_dreams`` sets state, then recomposes asynchronously, so a
    query made the moment the state flips can race the repaint.
    """
    for _ in range(attempts):
        await pilot.pause(0.05)
        if screen._dreams and screen.query(selector):
            return
    raise AssertionError(f"dreams refresh never landed {selector!r}")


def _dream_row_widgets(screen):
    return [
        widget
        for widget in screen.query(Static)
        if (widget.id or "").startswith("artifacts-dream-row-")
    ]


@pytest.mark.asyncio
async def test_seeded_dream_rows_paint_and_mode_label_names_dreams(
    tmp_path, monkeypatch
):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_story_db(tmp_path)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)

        await _wait_for_dreams(screen, pilot, "#artifacts-dream-row-1")
        row = screen.query_one("#artifacts-dream-row-1", Static)
        assert row.region.height >= 1, "dream row must paint, not just mount"
        mode_label = screen.query_one("#artifacts-mode-label", Static)
        assert "Dreams" in str(mode_label.renderable)
        assert not screen.query("#artifacts-list-dreams"), (
            "rows on screen mean the empty state must be gone"
        )


@pytest.mark.asyncio
async def test_failed_cycle_synthetic_row_paints(tmp_path, monkeypatch):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_failed_cycle_db(tmp_path)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        await _wait_for_dreams(
            screen, pilot, "#artifacts-dream-row-cycle-2026-09-22"
        )
        row = screen.query_one("#artifacts-dream-row-cycle-2026-09-22", Static)
        assert row.region.height >= 1, "the synthetic failed-cycle row must paint"


@pytest.mark.asyncio
async def test_disabled_dreams_renders_disabled_even_with_rows(tmp_path):
    # No [dreams] section in the sandbox config: dreams_setting defaults off.
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_story_db(tmp_path)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        empty = screen.query_one("#artifacts-list-dreams", Static)
        assert str(empty.renderable) == "> Dreams: disabled"
        assert not _dream_row_widgets(screen), (
            "a disabled feature must not paint story rows"
        )


@pytest.mark.asyncio
async def test_enabled_dreams_with_no_rows_renders_none_yet(tmp_path, monkeypatch):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = DreamsDB(tmp_path / "empty-dreams.sqlite", "artifacts-dreams")
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        empty = screen.query_one("#artifacts-list-dreams", Static)
        assert str(empty.renderable) == "> Dreams: none yet"


@pytest.mark.asyncio
async def test_missing_dreams_db_degrades_to_none_yet_without_crashing(monkeypatch):
    """Wiring order tolerance: no dreams_db attribute yet means no rows, not
    an exception out of the refresh worker."""
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")  # no dreams_db
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)

        empty = screen.query_one("#artifacts-list-dreams", Static)
        assert str(empty.renderable) == "> Dreams: none yet"
        assert screen._dreams == []


@pytest.mark.asyncio
async def test_unmount_invalidates_late_dreams_apply(tmp_path, monkeypatch):
    """Same teardown contract as the daily-reports trio (Qodo #15): unmount
    bumps the generation, so a late ``call_from_thread`` apply is a no-op
    instead of recomposing an unmounted screen."""
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_story_db(tmp_path)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-row-1")
        stale_generation = screen._dreams_generation

        await pilot.app.pop_screen()  # real unmount
        await pilot.pause()

        assert screen._dreams_generation != stale_generation
        screen._apply_dreams(stale_generation, [])
        assert screen._dreams, "stale apply must not clear the landed rows"
