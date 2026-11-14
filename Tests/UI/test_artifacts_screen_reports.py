"""Artifacts screen Reports slot: empty-state CTA and seeded report rows.

Real app via `_build_test_app`, real `SubscriptionsDB`, real write paths for
seeding; no LLM/fetch involved (no briefing generation here, just rows).
TASK-21514: row actions -- kept badge, View preview, Open deep-link, Keep,
Export.
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
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
        briefing_id,
        status=status,
        body_markdown=_SEED_BODY,
        item_count=1,
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
        briefing_id,
        preset_id=None,
        preset_name="Daily Brief",
        roster_snapshot_json="[]",
    )
    db.update_briefing_script(script_id, status="complete", turns_json="[]")
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    # A lexically-safe path: under the real briefing_audio_dir the guard passes
    # without touching disk (the play handler itself checks existence).
    from tldw_chatbook.Subscriptions.briefing_audio import briefing_audio_dir

    db.update_briefing_audio(
        audio_id,
        status="complete",
        file_path=str(
            briefing_audio_dir() / f"script-{script_id}-audio-{audio_id}.wav"
        ),
        duration_seconds=1.0,
        turn_count=1,
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

            # Match the real contract: run_demo_detached returns an
            # asyncio.Task (or None), which the screen attaches a completion
            # callback to (Qodo #5). A bare object() would not have one.
            async def _noop():
                return {"status": "ok"}

            return asyncio.create_task(_noop())

    stub = _StubDemo()
    app.daily_report_demo_service = stub
    async with _open_artifacts(app) as (screen, pilot):
        screen.query_one("#artifacts-daily-report-demo").press()
        for _ in range(50):
            await pilot.pause(0.05)
            if stub.started:
                break
        assert stub.started == 1


@pytest.mark.asyncio
async def test_demo_cta_tooltip_is_the_shared_consequence_copy():
    """TASK-34000.23 (AC#4): the Artifacts CTA's tooltip is the same
    consequence copy the Library confirm shows, from one shared source --
    every seeded feed, the hourly checks and the 24 h cadence, in one line."""
    from tldw_chatbook.Subscriptions.daily_report_demo import (
        DEMO_SOURCES,
        daily_report_demo_tooltip,
    )

    app = _build_test_app(configured_default="artifacts")
    async with _open_artifacts(app) as (screen, pilot):
        cta = screen.query_one("#artifacts-daily-report-demo", Button)
        tooltip = str(cta.tooltip)
        assert tooltip == daily_report_demo_tooltip()
        for source in DEMO_SOURCES:
            assert source["name"] in tooltip
        assert "hourly" in tooltip and "every 24 h" in tooltip
        assert "\n" not in tooltip


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


@pytest.mark.asyncio
async def test_older_success_then_new_failure_keeps_retry_cta():
    """Qodo #4 (PR #2460): a historical success must not suppress the retry CTA
    when the NEWEST run failed. The gate tracks the latest report's status, not
    "any report ever completed"."""
    app = _build_test_app(configured_default="artifacts")
    db: SubscriptionsDB = app.subscriptions_db
    # Older successful brief, then a newer failed brief (distinct watchlists to
    # avoid a duplicate-name collision). Rows sort created_at DESC, id DESC, so
    # the later-inserted failure is newest.
    ok_wl = int(WatchlistBundleService(db).create("Older OK")["id"])
    ok_id = db.insert_briefing(ok_wl)
    db.update_briefing(ok_id, status="complete", body_markdown=_SEED_BODY, item_count=1)
    fail_wl = int(WatchlistBundleService(db).create("Newer Fail")["id"])
    fail_id = db.insert_briefing(fail_wl)
    db.update_briefing(fail_id, status="failed")
    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        assert screen._daily_reports[0]["id"] == fail_id, "newest row is the failure"
        cta = screen.query_one("#artifacts-daily-report-demo", Button)
        assert cta.region.height >= 1, "retry CTA must survive an older success"


@pytest.mark.asyncio
async def test_successful_retry_refreshes_rows_and_drops_cta():
    """Qodo #5 (PR #2460): a retry that completes while Artifacts stays open must
    refresh the rows on task completion -- the failed row + retry CTA give way to
    the completed report without waiting for a screen resume."""
    app = _build_test_app(configured_default="artifacts")
    db: SubscriptionsDB = app.subscriptions_db
    wl = int(WatchlistBundleService(db).create("Daily Brief")["id"])
    fail_id = db.insert_briefing(wl)
    db.update_briefing(fail_id, status="failed")

    class _RetryDemo:
        """Detached-task stub: flips the failed briefing to complete, mirroring
        a real retry that succeeds, then resolves."""

        def run_demo_detached(self):
            async def _run():
                db.update_briefing(
                    fail_id,
                    status="complete",
                    body_markdown=_SEED_BODY,
                    item_count=1,
                )
                return {"status": "ok"}

            return asyncio.create_task(_run())

    app.daily_report_demo_service = _RetryDemo()
    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        screen.query_one("#artifacts-daily-report-demo", Button).press()
        await _wait_until(
            pilot,
            lambda: not screen.query("#artifacts-daily-report-demo"),
            what="retry CTA to clear after a successful retry",
        )
        assert screen._daily_reports[0]["status"] == "complete"


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


# --- TASK-31803: the PREVIEW header formats timestamps too -------------------


@pytest.mark.asyncio
async def test_report_preview_header_uses_formatted_timestamp():
    """Qodo #3 (PR #2460): the preview header renderable must format the
    timestamp (local, minute precision) -- never render raw microsecond ISO."""
    from rich.console import Console

    from tldw_chatbook.Subscriptions.daily_reports_view import (
        format_report_timestamp,
    )

    app = _build_test_app(configured_default="artifacts")
    async with _open_artifacts(app) as (screen, pilot):
        raw = "2026-09-05T23:10:20.123456+00:00"
        screen._previewed_report = {
            "id": 1,
            "watchlist_name": "Daily Brief",
            "watchlist_id": 1,
            "status": "complete",
            "created_at": raw,
            "item_count": 1,
            "body_markdown": "## Brief\n\nbody",
        }
        console = Console(width=200)
        with console.capture() as capture:
            console.print(screen._report_preview_renderable())
        rendered = capture.get()
        assert format_report_timestamp(raw) in rendered
        assert ".123456" not in rendered
        assert raw not in rendered


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


def _seed_audio_report(
    app, *, file_exists: bool, stored_name: str | None = None
) -> tuple[int, Path]:
    """Seed one complete report with a complete audio row.

    The stored path always sits under `briefing_audio_dir()` (so the view
    layer's lexical guard passes and `has_audio` is True); `file_exists`
    decides whether the file is actually on disk. `stored_name` stores that
    bare relative name in the row instead of the absolute path (TASK-39040
    fix round 1: the listing resolves it against the audio folder).
    """
    from tldw_chatbook.Subscriptions.briefing_audio import briefing_audio_dir

    briefing_id = _seed_report(app)
    db: SubscriptionsDB = app.subscriptions_db
    script_id = db.insert_briefing_script(
        briefing_id,
        preset_id=None,
        preset_name="Daily Brief",
        roster_snapshot_json="[]",
    )
    db.update_briefing_script(script_id, status="complete", turns_json="[]")
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    name = stored_name or f"script-{script_id}-audio-{audio_id}.wav"
    audio_path = briefing_audio_dir() / name
    db.update_briefing_audio(
        audio_id,
        status="complete",
        file_path=stored_name or str(audio_path),
        duration_seconds=1.0,
        turn_count=1,
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


# TASK-39040: on dev the Play handler trusted the row the Reports refresh
# cached, checked only by `validate_path_simple`. The Library and Watchlists
# players confine the same kind of row to `briefing_audio_dir()` at the press;
# this PR makes the Artifacts handler do the same, so a row or file that
# changed after the listing cannot hand an outside file to the player.

_OUTSIDE_REFUSAL = (
    "This report's audio is not in the app's audio folder, so it was not played."
)


async def _press_play_and_read_refusal(app, screen, pilot, briefing_id, played):
    screen.query_one(f"#artifacts-report-play-{briefing_id}", Button).press()
    await _wait_until(
        pilot, lambda: bool(played) or app.notify.call_count == 1, what="the press"
    )
    assert played == [], "an outside path must never reach the player"
    call = app.notify.call_args
    return str(call.args[0]), call.kwargs


@pytest.mark.asyncio
async def test_play_button_refuses_a_row_outside_the_audio_dir(tmp_path, monkeypatch):
    """A cached row whose path now points outside `briefing_audio_dir()` is
    refused with a literal toast that never carries the path."""
    from unittest.mock import Mock

    from tldw_chatbook.UI.Screens import artifacts_screen

    app = _build_test_app(configured_default="artifacts")
    app.notify = Mock()
    briefing_id, _audio_path = _seed_audio_report(app, file_exists=True)
    outside = tmp_path / "outside&calc.wav"
    outside.write_bytes(b"RIFF....WAVEfmt ")
    played: list[Path] = []
    monkeypatch.setattr(artifacts_screen, "play_audio_file", played.append)

    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        row = next(r for r in screen._daily_reports if r["id"] == briefing_id)
        row["audio_file_path"] = str(outside)
        with _captured_logs() as logged:
            message, kwargs = await _press_play_and_read_refusal(
                app, screen, pilot, briefing_id, played
            )

    assert message == _OUTSIDE_REFUSAL
    assert kwargs.get("markup") is False
    assert outside.name not in message and str(tmp_path) not in message
    leaked = [line for line in logged if outside.name in line]
    assert leaked == [], "the refusal must not log the row's path"


@contextmanager
def _captured_logs():
    """Every loguru line emitted inside the block, as text."""
    from loguru import logger

    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    try:
        yield lines
    finally:
        logger.remove(sink_id)


# Fix round 1 (TASK-39040 review, Important 1). The first commit checked the
# row with `audio_file_path_is_safe`, which resolves a RELATIVE row against
# `briefing_audio_dir()`, then played `validate_path_simple`'s unresolved
# `Path(raw)`, which the player opens against the working directory. Dev has
# the same play-the-unresolved-path step. The press now plays the resolved,
# confined result of `validate_path(raw, briefing_audio_dir())`.


@pytest.mark.asyncio
async def test_play_button_plays_a_relative_row_from_inside_the_audio_dir(
    tmp_path, monkeypatch
):
    """A bare relative name in the row means the file in the audio folder,
    and the player receives that file's absolute path."""
    from unittest.mock import Mock

    from tldw_chatbook.Subscriptions.briefing_audio import briefing_audio_dir
    from tldw_chatbook.UI.Screens import artifacts_screen

    monkeypatch.chdir(tmp_path)
    app = _build_test_app(configured_default="artifacts")
    app.notify = Mock()
    briefing_id, audio_path = _seed_audio_report(
        app, file_exists=True, stored_name="relative-clip.wav"
    )
    played: list[Path] = []
    monkeypatch.setattr(artifacts_screen, "play_audio_file", played.append)

    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        screen.query_one(f"#artifacts-report-play-{briefing_id}", Button).press()
        await _wait_until(
            pilot, lambda: bool(played) or app.notify.call_count, what="the press"
        )

    assert played == [audio_path.resolve()]
    assert played[0].is_absolute()
    assert played[0].parent == briefing_audio_dir().resolve()
    app.notify.assert_not_called()


@pytest.mark.asyncio
async def test_play_button_never_plays_a_relative_row_from_the_working_directory(
    tmp_path, monkeypatch
):
    """The same relative name with the file only in the working directory is
    not in the audio folder, so nothing is played."""
    from unittest.mock import Mock

    from tldw_chatbook.UI.Screens import artifacts_screen

    monkeypatch.chdir(tmp_path)
    (tmp_path / "cwd-only-clip.wav").write_bytes(b"RIFF....WAVEfmt ")
    app = _build_test_app(configured_default="artifacts")
    app.notify = Mock()
    briefing_id, audio_path = _seed_audio_report(
        app, file_exists=False, stored_name="cwd-only-clip.wav"
    )
    audio_path.unlink(missing_ok=True)
    played: list[Path] = []
    monkeypatch.setattr(artifacts_screen, "play_audio_file", played.append)

    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        screen.query_one(f"#artifacts-report-play-{briefing_id}", Button).press()
        await _wait_until(
            pilot, lambda: bool(played) or app.notify.call_count, what="the press"
        )

    assert played == [], "a file outside the audio folder must never be played"
    assert "no longer exists" in str(app.notify.call_args.args[0])


@pytest.mark.asyncio
async def test_play_button_never_hands_a_dash_leading_name_to_the_player_as_an_option(
    tmp_path, monkeypatch
):
    """A row named `-clip.wav` reaches the REAL player's argv as an absolute
    path, never as a bare `-clip.wav` that the player would parse as an
    option. Only `audio_player`'s own `subprocess` reference is faked, so the
    app's other processes are untouched."""
    import subprocess
    import types
    from unittest.mock import MagicMock, Mock

    from tldw_chatbook.TTS import audio_player

    monkeypatch.chdir(tmp_path)
    (tmp_path / "-clip.wav").write_bytes(b"RIFF....WAVEfmt ")
    app = _build_test_app(configured_default="artifacts")
    app.notify = Mock()
    briefing_id, _audio_path = _seed_audio_report(
        app, file_exists=True, stored_name="-clip.wav"
    )
    spawned: list[list[str]] = []

    def popen(argv, **kwargs):
        spawned.append([str(part) for part in argv])
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = lambda *a, **k: 0
        return process

    monkeypatch.setattr(
        audio_player,
        "subprocess",
        types.SimpleNamespace(
            Popen=popen,
            DEVNULL=subprocess.DEVNULL,
            PIPE=subprocess.PIPE,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    player = audio_player.SimpleAudioPlayer()
    player._system = "Darwin"  # afplay: deterministic on any host, no PATH probe
    monkeypatch.setattr(audio_player, "_audio_player_instance", player)

    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        screen.query_one(f"#artifacts-report-play-{briefing_id}", Button).press()
        await _wait_until(
            pilot, lambda: bool(spawned) or app.notify.call_count, what="the press"
        )
    player.stop()

    assert len(spawned) == 1, (spawned, app.notify.call_args_list)
    argv = spawned[0]
    assert "-clip.wav" not in argv
    assert Path(argv[-1]).is_absolute() and Path(argv[-1]).name == "-clip.wav"


@pytest.mark.asyncio
async def test_play_button_refuses_a_symlink_that_escapes_the_audio_dir(
    tmp_path, monkeypatch
):
    """The in-dir file is swapped for a symlink to an outside file after the
    listing: the press resolves the link and refuses it."""
    from unittest.mock import Mock

    from tldw_chatbook.UI.Screens import artifacts_screen

    app = _build_test_app(configured_default="artifacts")
    app.notify = Mock()
    briefing_id, audio_path = _seed_audio_report(app, file_exists=False)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF....WAVEfmt ")
    played: list[Path] = []
    monkeypatch.setattr(artifacts_screen, "play_audio_file", played.append)

    async with _open_artifacts(app) as (screen, pilot):
        await _wait_for_rows(screen, pilot)
        # The session shares one user data dir, so an earlier test may have
        # left a regular file at this name; replace it, and leave none behind.
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.unlink(missing_ok=True)
        audio_path.symlink_to(outside)
        try:
            message, kwargs = await _press_play_and_read_refusal(
                app, screen, pilot, briefing_id, played
            )
        finally:
            audio_path.unlink(missing_ok=True)

    assert message == _OUTSIDE_REFUSAL
    assert kwargs.get("markup") is False


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
            "the kept row must show the badge however many newer briefings sit above it"
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
