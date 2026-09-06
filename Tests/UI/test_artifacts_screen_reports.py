"""Artifacts screen Reports slot: empty-state CTA and seeded report rows.

Real app via `_build_test_app`, real `SubscriptionsDB`, real write paths for
seeding; no LLM/fetch involved (no briefing generation here, just rows).
TASK-21514: row actions -- kept badge, View preview, Open deep-link, Keep,
Export.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.briefing_keep import keep_briefing
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.Third_Party.textual_fspicker import FileSave
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen

pytestmark = pytest.mark.ui

_SEED_BODY = "## Daily Brief\n\nOne story [item 1]."


def _seed_report(app, *, status: str = "complete") -> int:
    db: SubscriptionsDB = app.subscriptions_db
    watchlist_id = int(WatchlistBundleService(db).create("Daily Brief")["id"])
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(
        briefing_id, status=status,
        body_markdown=_SEED_BODY, item_count=1,
    )
    return briefing_id


def _attach_file_chacha(app, tmp_path) -> CharactersRAGDB:
    """A file-backed ChaChaNotes DB on `app.chachanotes_db`.

    Deliberately NOT `attach_chachanotes_db` (`:memory:`): CharactersRAGDB
    hands each calling thread its own connection, and every thread gets a
    separate empty `:memory:` database -- the keep worker and the badge
    refresh both read from worker threads, which must see what the test
    thread wrote. `Tests/Subscriptions/test_briefing_keep.py` records the
    same rule ("never `:memory:` for either").
    """
    db = CharactersRAGDB(Path(tmp_path) / "chacha.sqlite", client_id="artifacts-row")
    app.chachanotes_db = db
    return db


@asynccontextmanager
async def _open_artifacts(app, *, size=(160, 50), seen_contexts=None):
    host = DestinationHarness(app, "artifacts", seen_contexts=seen_contexts)
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)
        yield screen, pilot


async def _wait_for_rows(screen, pilot, *, attempts: int = 50):
    for _ in range(attempts):
        await pilot.pause(0.05)
        if screen._daily_reports:
            return
    raise AssertionError("refresh worker never landed report rows")


async def _wait_until(pilot, predicate, *, attempts: int = 100, what: str = "state"):
    for _ in range(attempts):
        await pilot.pause(0.05)
        if predicate():
            return
    raise AssertionError(f"timed out waiting for {what}")


def _row_label(screen, briefing_id: int) -> str:
    row = screen.query_one(f"#artifacts-report-row-{briefing_id}", Static)
    return getattr(row.renderable, "plain", str(row.renderable))


async def _preview_report(screen, pilot, briefing_id: int) -> None:
    screen.query_one(f"#artifacts-report-view-{briefing_id}", Button).press()
    await _wait_until(
        pilot,
        lambda: screen._previewed_report is not None,
        what="report preview",
    )
    await pilot.pause()


@pytest.mark.asyncio
async def test_empty_state_offers_demo_cta():
    app = _build_test_app(configured_default="artifacts")
    async with _open_artifacts(app) as (screen, pilot):
        cta = screen.query_one("#artifacts-daily-report-demo", Button)
        assert cta.region.height >= 1, "CTA must paint, not just mount"
        assert screen.query_one("#artifacts-list-reports", Static)


@pytest.mark.asyncio
async def test_seeded_reports_list_rows_with_open_button():
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    async with _open_artifacts(app) as (screen, pilot):
        screen._start_daily_reports_refresh()
        for _ in range(50):
            await pilot.pause(0.05)
            if screen._daily_reports:
                break
        assert screen._daily_reports, "refresh worker must land rows"
        row = screen.query_one(f"#artifacts-report-row-{briefing_id}", Static)
        assert row.region.height >= 1, "report row must paint"
        assert screen.query_one("#artifacts-open-watchlists", Button)
        # CTA belongs to the empty state only
        assert not screen.query("#artifacts-daily-report-demo")


@pytest.mark.asyncio
async def test_audio_row_shows_play_button():
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    db: SubscriptionsDB = app.subscriptions_db
    script_id = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Daily Brief",
        roster_snapshot_json="[]",
    )
    db.update_briefing_script(script_id, status="complete", turns_json="[]")
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    # A lexically-safe path: under the real briefing_audio_dir the guard passes
    # without touching disk (the play handler itself checks existence).
    from tldw_chatbook.Subscriptions.briefing_audio import briefing_audio_dir
    db.update_briefing_audio(
        audio_id, status="complete",
        file_path=str(briefing_audio_dir() / f"script-{script_id}-audio-{audio_id}.wav"),
        duration_seconds=1.0, turn_count=1,
    )
    async with _open_artifacts(app) as (screen, pilot):
        screen._start_daily_reports_refresh()
        for _ in range(50):
            await pilot.pause(0.05)
            if screen._daily_reports:
                break
        play = screen.query_one(f"#artifacts-report-play-{briefing_id}", Button)
        assert play.region.height >= 1


@pytest.mark.asyncio
async def test_demo_cta_starts_the_detached_demo_task():
    """Qodo #10: the CTA starts the service-owned task; it never runs the
    demo inside a screen worker (unmount would cancel it mid-seed)."""
    app = _build_test_app(configured_default="artifacts")

    class _StubDemo:
        def __init__(self):
            self.started = 0

        def run_demo_detached(self):
            self.started += 1
            return object()  # truthy task stand-in

    stub = _StubDemo()
    app.daily_report_demo_service = stub
    async with _open_artifacts(app) as (screen, pilot):
        screen.query_one("#artifacts-daily-report-demo").press()
        for _ in range(50):
            await pilot.pause(0.05)
            if stub.started:
                break
        assert stub.started == 1


# --- TASK-31801: a failed brief must keep a demo retry affordance -----------


@pytest.mark.asyncio
async def test_failed_report_keeps_demo_retry_cta():
    """A failed brief leaves a report row, so the empty-state branch is gone;
    the demo CTA must still be reachable (the failure toast says "run the demo
    again"). It renders and starts the same detached demo task."""
    app = _build_test_app(configured_default="artifacts")
    _seed_report(app, status="failed")
    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        cta = screen.query_one("#artifacts-daily-report-demo", Button)
        assert cta.region.height >= 1, "retry CTA must paint, not just mount"


@pytest.mark.asyncio
async def test_completed_report_hides_demo_cta():
    """A completed report means the user succeeded -- no retry CTA is shown
    (the empty-state CTA belongs to the no-reports and no-success states)."""
    app = _build_test_app(configured_default="artifacts")
    _seed_report(app, status="complete")
    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        assert not screen.query("#artifacts-daily-report-demo")


# --- TASK-31802: don't advertise an import path the user cannot take --------


@pytest.mark.asyncio
async def test_import_precondition_is_explained_and_empty_copy_is_honest():
    """The Import button is permanently disabled, so the empty-state copy must
    not tell users to "import an artifact", and the disabled control must
    carry a visible inline explanation of its precondition."""
    app = _build_test_app(configured_default="artifacts")
    async with _open_artifacts(app) as (screen, pilot):
        import_btn = screen.query_one("#artifacts-import-artifact", Button)
        assert import_btn.disabled
        note = screen.query_one("#artifacts-import-note", Static)
        assert note.region.height >= 1, "import precondition note must paint"
        empty = screen.query_one("#artifacts-detail-empty", Static)
        assert "import an artifact" not in str(empty.renderable).lower()


# --- TASK-21514: row actions (badge, preview, deep-link, keep, export) ------


@pytest.mark.asyncio
async def test_kept_report_row_shows_kept_badge(tmp_path):
    app = _build_test_app(configured_default="artifacts")
    chacha = _attach_file_chacha(app, tmp_path)
    briefing_id = _seed_report(app)
    keep_briefing(app.subscriptions_db, chacha, briefing_id, origin="manual")
    async with _open_artifacts(app) as (screen, pilot):
        screen._start_daily_reports_refresh()
        await _wait_for_rows(screen, pilot)
        assert "· kept" in _row_label(screen, briefing_id)


@pytest.mark.asyncio
async def test_unkept_report_row_has_no_kept_badge():
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    async with _open_artifacts(app) as (screen, pilot):
        screen._start_daily_reports_refresh()
        await _wait_for_rows(screen, pilot)
        assert "kept" not in _row_label(screen, briefing_id)


@pytest.mark.asyncio
async def test_view_button_previews_report_body_in_detail_pane():
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        await _preview_report(screen, pilot, briefing_id)
        previewed = screen._previewed_report
        assert previewed is not None
        assert previewed["id"] == briefing_id
        assert previewed["body_markdown"] == _SEED_BODY
        preview = screen.query_one("#artifacts-report-preview", Static)
        assert preview.region.height >= 1, "preview body must paint"
        clear = screen.query_one("#artifacts-report-preview-clear", Button)
        assert clear.region.height >= 1


@pytest.mark.asyncio
async def test_clear_preview_restores_previous_pane():
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        await _preview_report(screen, pilot, briefing_id)
        screen.query_one("#artifacts-report-preview-clear", Button).press()
        await pilot.pause()
        assert screen._previewed_report is None
        assert not screen.query("#artifacts-report-preview")


@pytest.mark.asyncio
async def test_open_button_deep_links_to_watchlists_artifacts_pane():
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    contexts: list = []
    async with _open_artifacts(app, seen_contexts=contexts) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        screen.query_one(f"#artifacts-report-open-{briefing_id}", Button).press()
        await pilot.pause()
        assert contexts == [
            {
                "section": "artifacts",
                "backend": "local",
                "briefing_id": f"local:briefing:{briefing_id}",
            }
        ]


@pytest.mark.asyncio
async def test_keep_button_keeps_previewed_report_and_flips_badge(tmp_path):
    app = _build_test_app(configured_default="artifacts")
    chacha = _attach_file_chacha(app, tmp_path)
    briefing_id = _seed_report(app)
    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        await _preview_report(screen, pilot, briefing_id)
        keep_button = screen.query_one("#artifacts-report-keep", Button)
        assert not keep_button.disabled
        keep_button.press()
        await _wait_until(
            pilot,
            lambda: chacha.get_kept_briefing_by_source(briefing_id) is not None,
            what="kept_briefings row",
        )
        await _wait_until(
            pilot,
            lambda: "· kept" in _row_label(screen, briefing_id),
            what="kept badge flip",
        )


@pytest.mark.asyncio
async def test_keep_button_disabled_without_chacha_handle():
    app = _build_test_app(configured_default="artifacts")  # no chachanotes_db
    briefing_id = _seed_report(app)
    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        await _preview_report(screen, pilot, briefing_id)
        assert screen.query_one("#artifacts-report-keep", Button).disabled


@pytest.mark.asyncio
async def test_keep_button_disabled_for_failed_report_preview(tmp_path):
    app = _build_test_app(configured_default="artifacts")
    _attach_file_chacha(app, tmp_path)
    briefing_id = _seed_report(app, status="failed")
    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        await _preview_report(screen, pilot, briefing_id)
        # A failed preview still paints a status body, but Keep refuses it.
        assert screen.query_one("#artifacts-report-preview", Static).region.height >= 1
        assert screen.query_one("#artifacts-report-keep", Button).disabled
        assert screen.query_one("#artifacts-report-export", Button).disabled


@pytest.mark.asyncio
async def test_export_button_pushes_file_save_dialog():
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        await _preview_report(screen, pilot, briefing_id)
        export_button = screen.query_one("#artifacts-report-export", Button)
        assert not export_button.disabled
        export_button.press()
        await _wait_until(
            pilot,
            lambda: isinstance(pilot.app.screen, FileSave),
            what="FileSave dialog",
        )


@pytest.mark.asyncio
async def test_write_report_export_file_writes_markdown_document(tmp_path):
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    target = Path(tmp_path) / "daily-brief.md"
    async with _open_artifacts(app) as (screen, pilot):
        row = app.subscriptions_db.get_briefing(briefing_id)
        row["watchlist_name"] = "Daily Brief"
        await screen._write_report_export_file(target, row)
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "watchlist: Daily Brief" in text
    assert "## Daily Brief" in text


# --- Qodo fix wave -----------------------------------------------------------


def _seed_audio_report(app, *, file_exists: bool) -> tuple[int, Path]:
    """Seed one complete report with a complete audio row.

    The stored path always sits under `briefing_audio_dir()` (so the view
    layer's lexical guard passes and `has_audio` is True); `file_exists`
    decides whether the file is actually on disk.
    """
    from tldw_chatbook.Subscriptions.briefing_audio import briefing_audio_dir

    briefing_id = _seed_report(app)
    db: SubscriptionsDB = app.subscriptions_db
    script_id = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Daily Brief",
        roster_snapshot_json="[]",
    )
    db.update_briefing_script(script_id, status="complete", turns_json="[]")
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    audio_path = (
        briefing_audio_dir() / f"script-{script_id}-audio-{audio_id}.wav"
    )
    db.update_briefing_audio(
        audio_id, status="complete",
        file_path=str(audio_path),
        duration_seconds=1.0, turn_count=1,
    )
    if file_exists:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"RIFF....WAVEfmt ")
    return briefing_id, audio_path


@pytest.mark.asyncio
async def test_play_button_validates_path_and_plays_normalized_result(
    tmp_path, monkeypatch
):
    """Qodo #1: playback goes through the centralized path validator, and
    the player receives the validator's normalized result."""
    from unittest.mock import Mock

    from tldw_chatbook.UI.Screens import artifacts_screen

    app = _build_test_app(configured_default="artifacts")
    app.notify = Mock()
    briefing_id, _audio_path = _seed_audio_report(app, file_exists=True)
    played: list[Path] = []
    monkeypatch.setattr(artifacts_screen, "play_audio_file", played.append)

    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        screen.query_one(f"#artifacts-report-play-{briefing_id}", Button).press()
        await _wait_until(pilot, lambda: len(played) == 1, what="playback call")
    assert len(played) == 1
    assert played[0].name.startswith("script-")
    app.notify.assert_not_called()


@pytest.mark.asyncio
async def test_play_button_missing_file_warns_through_the_validator(monkeypatch):
    """Qodo #1: a missing file is the validator's rejection, surfaced as the
    existing "no longer exists" warning -- and nothing is handed to the
    player."""
    from unittest.mock import Mock

    from tldw_chatbook.UI.Screens import artifacts_screen

    app = _build_test_app(configured_default="artifacts")
    app.notify = Mock()
    briefing_id, _audio_path = _seed_audio_report(app, file_exists=False)
    played: list[Path] = []
    monkeypatch.setattr(artifacts_screen, "play_audio_file", played.append)

    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        screen.query_one(f"#artifacts-report-play-{briefing_id}", Button).press()
        await _wait_until(
            pilot, lambda: app.notify.call_count == 1, what="missing-file warning"
        )
    assert played == [], "a rejected path must never reach the player"
    warning = str(app.notify.call_args.args[0])
    assert "no longer exists" in warning


@pytest.mark.asyncio
async def test_kept_badge_is_exact_per_row_not_page_bound(tmp_path):
    """Qodo #14: kept-ness is resolved per displayed row id, so a keep that
    is NOT among the newest `list_kept_briefings` page still earns its
    badge. The report shown here is old (a newer briefing exists above it)
    but is the one that was kept."""
    app = _build_test_app(configured_default="artifacts")
    chacha = _attach_file_chacha(app, tmp_path)
    kept_id = _seed_report(app)  # the OLD, kept one (complete: Keep refuses others)
    for _ in range(3):  # newer briefings push it down the newest-first list
        _seed_report(app)
    keep_briefing(app.subscriptions_db, chacha, kept_id, origin="manual")
    async with _open_artifacts(app) as (screen, pilot):
        screen._start_daily_reports_refresh()
        await _wait_for_rows(screen, pilot)
        assert "· kept" in _row_label(screen, kept_id), (
            "the kept row must show the badge however many newer "
            "briefings sit above it"
        )


@pytest.mark.asyncio
async def test_unmount_tears_down_daily_reports_and_preview_workers():
    """Qodo #15: unmount cancels BOTH new worker paths and invalidates
    their in-flight apply callbacks (generation bump), so a late
    `call_from_thread` apply is a no-op instead of a write into an
    unmounted screen."""
    from unittest.mock import Mock

    app = _build_test_app(configured_default="artifacts")
    app.notify = Mock()
    briefing_id = _seed_report(app)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)
        for _ in range(50):
            await pilot.pause(0.05)
            if screen._daily_reports:
                break
        assert screen._daily_reports, "refresh worker must land rows first"
        stale_reports = screen._daily_reports_generation
        stale_preview = screen._report_preview_generation

        await pilot.app.pop_screen()  # real unmount
        await pilot.pause()

        assert screen._daily_reports_generation != stale_reports
        assert screen._report_preview_generation != stale_preview
        # Stale applies are no-ops: neither clears the landed rows, nor
        # installs a preview, nor fires the missing-row warning notify.
        screen._apply_daily_reports(stale_reports, [])
        screen._apply_report_preview(stale_preview, briefing_id, None)
        assert screen._daily_reports, "stale apply must not clear the rows"
        assert screen._previewed_report is None
        app.notify.assert_not_called()
