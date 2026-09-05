"""task-28007 AC#1/AC#2 (Task 3): "Analyze N skipped" run-summary action.

An import run with "Analyze after import" on and no provider configured
marks every row analysis-skipped, with no batch remediation short of a
full re-import or one Reader visit per item. This action runs Task 2's
bulk-Analyze worker over exactly the ids the Import queue currently shows
as analysis-skipped, gated on Task 1's provider-reason resolver, and
reports per item back onto each row's own progress line.

The action id is FIXED (``library-ingest-analyze-skipped``, never job- or
batch-suffixed): it is one canvas-wide control over every skipped id
currently visible in the queue, not one per batch group -- offering it
per batch could mount the same id twice and crash.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button

from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from Tests.UI.test_library_ingest_retry_last import _ingest_screen, _pilot_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _wait_for_condition,
    _wait_for_selector,
)


def _skipped_job(**overrides) -> LibraryIngestJob:
    defaults = dict(
        job_id="ingest-job-1",
        source_path="/tmp/notes.txt",
        state=IngestJobState.DONE,
        media_id=7,
        submitted_at=100.0,
        finished_at=101.0,
        progress={
            "message": (
                "Imported notes.txt — analysis skipped: no analysis "
                "provider is configured"
            ),
            "analysis_skipped": "no analysis provider is configured",
        },
    )
    defaults.update(overrides)
    return LibraryIngestJob(**defaults)


def _ready_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        library_screen_module, "analysis_unavailable_reason", lambda *_a, **_k: ""
    )


def _unready_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        library_screen_module,
        "analysis_unavailable_reason",
        lambda *_a, **_k: "No analysis provider is configured.",
    )


# --- visibility gate (×2) -----------------------------------------------


@pytest.mark.asyncio
async def test_analyze_skipped_action_absent_without_a_ready_provider(monkeypatch):
    """A skipped row alone is not enough -- Task 1's reason must be empty."""
    app = _pilot_app()
    app.library_ingest_jobs.restore([_skipped_job()], next_id=2)
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _unready_provider(monkeypatch)
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        assert not screen.query("#library-ingest-analyze-skipped")


@pytest.mark.asyncio
async def test_analyze_skipped_action_appears_with_skipped_items_and_ready_provider(
    monkeypatch,
):
    app = _pilot_app()
    app.library_ingest_jobs.restore([_skipped_job()], next_id=2)
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        button = await _wait_for_selector(
            screen, pilot, "#library-ingest-analyze-skipped"
        )
        assert str(button.label) == "Analyze 1 skipped"
        assert button.disabled is False


# --- ids passed ------------------------------------------------------------


@pytest.mark.asyncio
async def test_pressing_analyze_skipped_runs_the_worker_over_exactly_those_ids(
    monkeypatch,
):
    app = _pilot_app()
    app.library_ingest_jobs.restore(
        [
            _skipped_job(job_id="ingest-job-1", media_id=7, source_path="/tmp/a.txt"),
            _skipped_job(job_id="ingest-job-2", media_id=9, source_path="/tmp/b.txt"),
            _job_done_and_analyzed(),
        ],
        next_id=4,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

        calls = []
        screen._start_library_media_analyze = (
            lambda media_ids, **kwargs: calls.append((media_ids, kwargs))
        )
        button = await _wait_for_selector(
            screen, pilot, "#library-ingest-analyze-skipped"
        )
        button.press()
        await pilot.pause()

        assert len(calls) == 1
        media_ids, kwargs = calls[0]
        assert set(media_ids) == {"7", "9"}
        assert kwargs.get("overwrite") is False
        assert callable(kwargs.get("on_item_done"))


def _job_done_and_analyzed(**overrides) -> LibraryIngestJob:
    defaults = dict(
        job_id="ingest-job-3",
        source_path="/tmp/c.txt",
        state=IngestJobState.DONE,
        media_id=11,
        submitted_at=100.0,
        finished_at=101.0,
        progress={"message": "Imported c.txt"},
    )
    defaults.update(overrides)
    return LibraryIngestJob(**defaults)


# --- outcome rendering -------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_skipped_run_paints_per_item_outcomes_on_their_own_rows(
    monkeypatch,
):
    """AC#2: rows are individually addressable -- a completed run's outcome
    replaces the stale "analysis skipped: ..." note on that SAME row."""
    app = _pilot_app()
    app.library_ingest_jobs.restore(
        [
            _skipped_job(job_id="ingest-job-1", media_id=7, source_path="/tmp/ok.txt"),
            _skipped_job(
                job_id="ingest-job-2", media_id=9, source_path="/tmp/bad.txt"
            ),
        ],
        next_id=3,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

        async def _unanalyzed(media_ids):
            return media_ids

        async def _one(media_id, *, resolution):
            return media_id == "7"

        screen._library_media_unanalyzed_ids = _unanalyzed
        screen._analyze_one_library_media_item = _one

        button = await _wait_for_selector(
            screen, pilot, "#library-ingest-analyze-skipped"
        )
        button.press()
        await pilot.pause()

        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_analyze_running is False,
            message="the run never settled",
        )
        await pilot.pause()

        outcomes = screen._library_ingest_analyze_outcomes
        assert outcomes["7"] == (True, "")
        assert outcomes["9"][0] is False

        canvas_text = "\n".join(
            str(row.renderable)
            for row in screen.query("#library-ingest-queue-panel Static")
        )
        assert "✓ analyzed · ok.txt" in canvas_text
        assert "✗ analysis failed · bad.txt" in canvas_text
        # The action re-renders too: both ids are now accounted for (one
        # fixed, one failed-but-still-unanalysed) -- it must not vanish
        # while a genuinely unanalysed item remains, and it must not still
        # claim "2 skipped" once one of them is fixed.
        remaining = await _wait_for_selector(
            screen, pilot, "#library-ingest-analyze-skipped"
        )
        assert str(remaining.label) == "Analyze 1 skipped"
        assert remaining.disabled is False


# --- rulings: second press, select-mode no-op --------------------------------


@pytest.mark.asyncio
async def test_second_analyze_skipped_press_while_running_gets_the_existing_notice(
    monkeypatch,
):
    app = _pilot_app()
    app.library_ingest_jobs.restore([_skipped_job()], next_id=2)
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)
        screen._library_media_analyze_running = True
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

        notices = []
        screen.app_instance.notify = lambda message, **kwargs: notices.append(
            (message, kwargs)
        )
        button = await _wait_for_selector(
            screen, pilot, "#library-ingest-analyze-skipped"
        )
        assert button.disabled is True, "the action must disable while running"


@pytest.mark.asyncio
async def test_pressing_analyze_skipped_does_not_toggle_media_select_mode(
    monkeypatch,
):
    """``_start_library_media_analyze`` exits select mode as a side effect;
    from the Import canvas, with select mode never entered, that must be
    a no-op (no crash, no stray canvas swap)."""
    app = _pilot_app()
    app.library_ingest_jobs.restore([_skipped_job()], next_id=2)
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)
        assert screen._library_media_select_mode is False
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

        async def _one(media_id, *, resolution):
            return True

        screen._analyze_one_library_media_item = _one
        button = await _wait_for_selector(
            screen, pilot, "#library-ingest-analyze-skipped"
        )
        button.press()
        await pilot.pause()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_analyze_running is False,
            message="the run never settled",
        )
        assert screen._library_media_select_mode is False
        # Still on the Import canvas -- the Media canvas's own select-mode
        # exit path was never exercised into an unrelated screen state.
        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA
