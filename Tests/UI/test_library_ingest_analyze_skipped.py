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
from textual.worker import WorkerState

from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from tldw_chatbook.UI.Library_Modules import library_media_analysis_controller as media_analysis_module
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from Tests.Library.test_library_ingest_state import _skipped_job
from Tests.UI.test_library_ingest_retry_last import _ingest_screen, _pilot_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _wait_for_condition,
    _wait_for_selector,
)


def _ready_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        library_screen_module, "analysis_unavailable_reason", lambda *_a, **_k: ""
    )
    monkeypatch.setattr(
        media_analysis_module, "analysis_unavailable_reason", lambda *_a, **_k: ""
    )


def _unready_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        library_screen_module,
        "analysis_unavailable_reason",
        lambda *_a, **_k: "No analysis provider is configured.",
    )
    monkeypatch.setattr(
        media_analysis_module,
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
        screen._media_analysis_controller._start_library_media_analyze = (
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

        screen._media_analysis_controller._library_media_unanalyzed_ids = _unanalyzed
        screen._media_analysis_controller._analyze_one_library_media_item = _one

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
        # (fix round 1, M-4) AC#2's actual promise is that the note is
        # REPLACED, not merely that a new one also appears somewhere.
        assert "analysis skipped" not in canvas_text
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
    """(fix round 1, I-4) The original version of this test built a
    ``notices`` list, never pressed a second time, and never asserted it --
    the only thing it actually checked was ``button.disabled``. This
    presses a REAL first time (starting a real, blocked run), then calls
    the handler a second time while that run is still active, and asserts
    the seam's own "Analysis already running" notice fires."""
    app = _pilot_app()
    app.library_ingest_jobs.restore([_skipped_job()], next_id=2)
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def _unanalyzed(media_ids):
            return media_ids

        async def _one(media_id, *, resolution):
            entered.set()
            await release.wait()
            return True

        screen._media_analysis_controller._library_media_unanalyzed_ids = _unanalyzed
        screen._media_analysis_controller._analyze_one_library_media_item = _one
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

        button = await _wait_for_selector(
            screen, pilot, "#library-ingest-analyze-skipped"
        )
        button.press()
        await pilot.pause()
        await _wait_for_condition(
            pilot, entered.is_set, message="the first run never started"
        )
        assert screen._library_media_analyze_running is True
        button_after_first_press = screen.query_one(
            "#library-ingest-analyze-skipped", Button
        )
        assert button_after_first_press.disabled is True, (
            "the action must disable while running"
        )

        notices = []
        screen.app_instance.notify = lambda message, **kwargs: notices.append(
            (message, kwargs)
        )
        # A second physical click reaches a disabled Textual Button as a
        # no-op, so this calls the handler directly -- exactly what the
        # ruling names as the belt-and-braces path a stray keyboard route
        # or a race could still reach.
        screen.handle_library_ingest_analyze_skipped(
            Button.Pressed(button_after_first_press)
        )
        assert notices == [("Analysis already running", {"severity": "warning"})]

        release.set()
        await pilot.pause()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_analyze_running is False,
            message="the first run never settled",
        )


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

        screen._media_analysis_controller._analyze_one_library_media_item = _one
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


# --- fix round 1: C-1 (mixed already-analysed set), I-1 (real reasons),
# I-3 (origin-aware unmount notice) --------------------------------------


@pytest.mark.asyncio
async def test_press_over_a_mixed_set_auto_skips_already_analysed_and_notifies(
    monkeypatch,
):
    """C-1: an Import-started run has no Skip/Overwrite card to arm. One of
    the two skipped ids already has an analysis (e.g. it was fixed through
    the Reader since the import completed) -- the press must run the
    OTHER id, tell the user what it skipped, and never arm the Media
    canvas's own choice anywhere."""
    app = _pilot_app()
    app.library_ingest_jobs.restore(
        [
            _skipped_job(job_id="ingest-job-1", media_id=7, source_path="/tmp/a.txt"),
            _skipped_job(job_id="ingest-job-2", media_id=9, source_path="/tmp/b.txt"),
        ],
        next_id=3,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)
        notices = []
        screen.app_instance.notify = lambda message, **kwargs: notices.append(
            (message, kwargs)
        )
        analyzed: list[str] = []

        async def _unanalyzed(media_ids):
            # "7" already has an analysis (fixed via the Reader); "9" does
            # not.
            return tuple(mid for mid in media_ids if mid != "7")

        async def _one(media_id, *, resolution):
            analyzed.append(media_id)
            return True

        screen._media_analysis_controller._library_media_unanalyzed_ids = _unanalyzed
        screen._media_analysis_controller._analyze_one_library_media_item = _one
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

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

        assert analyzed == ["9"], "must run exactly the still-unanalysed id"
        assert screen._library_media_analyze_choice is None, (
            "the Media canvas's Skip/Overwrite choice must never arm here"
        )
        assert not screen.query("#library-media-analyze-skip")
        assert any(
            message == "1 already analyzed · skipped" for message, _ in notices
        ), notices


@pytest.mark.asyncio
async def test_press_over_an_entirely_already_analysed_set_notifies_and_runs_nothing(
    monkeypatch,
):
    """C-1's other leg: every id in the set already has an analysis --
    still no silent no-op, and no worker runs."""
    app = _pilot_app()
    app.library_ingest_jobs.restore(
        [_skipped_job(job_id="ingest-job-1", media_id=7, source_path="/tmp/a.txt")],
        next_id=2,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)
        notices = []
        screen.app_instance.notify = lambda message, **kwargs: notices.append(
            (message, kwargs)
        )
        analyzed: list[str] = []

        async def _unanalyzed(media_ids):
            return ()

        async def _one(media_id, *, resolution):
            analyzed.append(media_id)
            return True

        screen._media_analysis_controller._library_media_unanalyzed_ids = _unanalyzed
        screen._media_analysis_controller._analyze_one_library_media_item = _one
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

        button = await _wait_for_selector(
            screen, pilot, "#library-ingest-analyze-skipped"
        )
        button.press()
        await pilot.pause()

        assert analyzed == []
        assert screen._library_media_analyze_choice is None
        assert not screen.query("#library-media-analyze-skip")
        assert any(message == "Nothing left to analyze" for message, _ in notices), (
            notices
        )


@pytest.mark.asyncio
async def test_analyze_outcome_reports_a_raised_exceptions_own_message(monkeypatch):
    """I-1: a raised exception's own text is the reason, not the generic
    catch-all -- matching an import row's own "analysis failed: <reason>"
    honesty."""
    app = _pilot_app()
    app.library_ingest_jobs.restore(
        [
            _skipped_job(
                job_id="ingest-job-1", media_id=7, source_path="/tmp/flaky.txt"
            )
        ],
        next_id=2,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)

        async def _unanalyzed(media_ids):
            return media_ids

        async def _one(media_id, *, resolution):
            raise RuntimeError("provider timeout")

        screen._media_analysis_controller._library_media_unanalyzed_ids = _unanalyzed
        screen._media_analysis_controller._analyze_one_library_media_item = _one
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

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

        canvas_text = "\n".join(
            str(row.renderable)
            for row in screen.query("#library-ingest-queue-panel Static")
        )
        assert "✗ analysis failed · flaky.txt · provider timeout" in canvas_text


@pytest.mark.asyncio
async def test_import_run_that_dies_with_the_screen_names_the_import_run(
    monkeypatch,
):
    """I-3: an Import-started run interrupted by leaving Library must point
    the user back at the Import queue's own action, not Select mode's."""
    app = _pilot_app()
    app.library_ingest_jobs.restore(
        [
            _skipped_job(job_id="ingest-job-1", media_id=7, source_path="/tmp/a.txt"),
            _skipped_job(job_id="ingest-job-2", media_id=9, source_path="/tmp/b.txt"),
            _skipped_job(
                job_id="ingest-job-3", media_id=11, source_path="/tmp/c.txt"
            ),
        ],
        next_id=4,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)
        notices = []
        screen.app_instance.notify = lambda message, **kwargs: notices.append(
            (message, kwargs)
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def _unanalyzed(media_ids):
            return media_ids

        async def _one(media_id, *, resolution):
            if media_id != "9":
                return True
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                raise
            return True

        screen._media_analysis_controller._library_media_unanalyzed_ids = _unanalyzed
        screen._media_analysis_controller._analyze_one_library_media_item = _one
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

        button = await _wait_for_selector(
            screen, pilot, "#library-ingest-analyze-skipped"
        )
        button.press()
        await pilot.pause()
        worker = next(
            candidate
            for candidate in host.workers
            if candidate.group
            == library_screen_module._ANALYZE_SELECTED_WORKER_GROUP
        )
        await _wait_for_condition(
            pilot, entered.is_set, message="the run never reached item 2"
        )
        await host.pop_screen()
        await pilot.pause()
        await pilot.pause()

        assert worker.state is WorkerState.CANCELLED
        assert notices, "a cancelled Import run must still say where it stopped"
        assert notices[0][0] == (
            "Analysis stopped at 1 of 3 · reopen the import run and press "
            "Analyze N skipped to continue"
        ), notices
        assert notices[0][1].get("severity") == "warning"


@pytest.mark.asyncio
async def test_a_structural_change_during_a_no_fallback_repaint_still_repaints(
    monkeypatch,
):
    """(Task 3 re-review, N-1) The run's ``finally`` repaint passes
    ``allow_screen_fallback=False`` so an unmount mid-run cannot schedule a
    whole-screen recompose on a dying screen. When the state ALSO changed
    structurally in that same window (a pre-flight result landing, the
    media DB going away), the barred branch returned having only stored
    ``canvas.state`` -- a plain attribute, not a reactive -- so nothing
    repainted at all and "Analyze N skipped" stayed disabled until some
    unrelated later tick. The fallback being barred must degrade to the
    TARGETED repaint, not to no repaint."""
    app = _pilot_app()
    app.library_ingest_jobs.restore([_skipped_job()], next_id=2)
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)
        screen._library_media_analyze_running = True
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        button = await _wait_for_selector(
            screen, pilot, "#library-ingest-analyze-skipped"
        )
        assert button.disabled is True

        # Exactly what the run's ``finally`` does -- except a structural
        # change (``unavailable_line``) landed in the same window.
        screen._library_media_analyze_running = False
        app.media_db = None
        screen._update_library_ingest_dynamic_regions(allow_screen_fallback=False)
        await pilot.pause()
        await pilot.pause()
        assert (
            screen.query_one("#library-ingest-analyze-skipped", Button).disabled
            is False
        ), "the action must re-enable within the run's own settling sync"


# --- Qodo review round (PR #2400) ---------------------------------------


def test_analyze_origin_values_are_named_constants():
    """Qodo #1: the "media"/"import" origin literals are repeated across
    init, assignment, receipt rendering, and unmount handling -- named
    once, next to ``_ANALYZE_SELECTED_WORKER_GROUP``, so they cannot drift
    apart from one another."""
    assert library_screen_module._ANALYZE_ORIGIN_MEDIA == "media"
    assert library_screen_module._ANALYZE_ORIGIN_IMPORT == "import"


@pytest.mark.asyncio
async def test_clear_finished_prunes_the_stale_outcome_for_a_reused_media_id(
    monkeypatch,
):
    """Qodo #2: ``_library_ingest_analyze_outcomes`` is keyed only by media
    id for the screen's whole lifetime. Without pruning it when its job is
    cleared, a later job reusing that same id inherits the stale success --
    hiding the remediation action and painting the new (unanalyzed) row as
    already analyzed. After Clear finished, a new job with the same media
    id and analysis_skipped must be offered the action again, and its row
    must not be painted as analyzed (AC (a)); the count must reflect only
    the current job's row, not a stale memory of the old one (AC (b))."""
    app = _pilot_app()
    app.library_ingest_jobs.restore(
        [_skipped_job(job_id="ingest-job-1", media_id=7, source_path="/tmp/a.txt")],
        next_id=2,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)
        # A prior press of the action already fixed media id 7.
        screen._library_ingest_analyze_outcomes["7"] = (True, "")
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        assert not screen.query("#library-ingest-analyze-skipped"), (
            "already-fixed id must not offer the action"
        )

        # Clear finished: arm, then confirm past the double-click dead zone.
        screen.query_one("#library-ingest-clear-finished", Button).press()
        await pilot.pause()
        # (wave-5 merge) The armed-at stamp is a `LibraryIngestState` field
        # now, not a flat screen attribute -- the screen's generated shim
        # block was deleted in the ingest cleanup PR.
        screen._ingest_state.clear_finished_armed_at -= 1.0
        screen.query_one("#library-ingest-clear-finished", Button).press()
        await pilot.pause()

        # A NEW job reuses the same media id and is skipped again (e.g. a
        # newer, unanalyzed version/re-import).
        app.library_ingest_jobs.restore(
            [_skipped_job(job_id="ingest-job-2", media_id=7, source_path="/tmp/a.txt")],
            next_id=3,
        )
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

        button = await _wait_for_selector(
            screen, pilot, "#library-ingest-analyze-skipped"
        )
        assert str(button.label) == "Analyze 1 skipped", (
            "N must reflect only the current job's row, not the stale outcome"
        )
        canvas_text = "\n".join(
            str(row.renderable)
            for row in screen.query("#library-ingest-queue-panel Static")
        )
        assert "analyzed" not in canvas_text, (
            "the new job's row must not be painted as already analyzed"
        )


@pytest.mark.asyncio
async def test_auto_skipped_ids_are_resolved_not_left_actionable_forever(
    monkeypatch,
):
    """Qodo #3: the AC#3 partition pass auto-skips ids that already carry
    an analysis without recording an outcome for them -- they stay counted
    by "Analyze N skipped" forever, and every later press reports nothing
    left to run. After a mixed run (one auto-skipped, one generated), N
    must drop to 0, the action must disappear, and the auto-skipped row
    must show a resolved receipt."""
    app = _pilot_app()
    app.library_ingest_jobs.restore(
        [
            _skipped_job(job_id="ingest-job-1", media_id=7, source_path="/tmp/a.txt"),
            _skipped_job(job_id="ingest-job-2", media_id=9, source_path="/tmp/b.txt"),
        ],
        next_id=3,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        _ready_provider(monkeypatch)

        async def _unanalyzed(media_ids):
            # "7" already has an analysis (auto-skipped by the partition
            # pass); "9" does not.
            return tuple(mid for mid in media_ids if mid != "7")

        async def _one(media_id, *, resolution):
            return True

        screen._media_analysis_controller._library_media_unanalyzed_ids = _unanalyzed
        screen._media_analysis_controller._analyze_one_library_media_item = _one
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

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
        assert outcomes.get("7", (None,))[0] is True, (
            "the auto-skipped id must get a recorded outcome too"
        )
        assert outcomes.get("9", (None,))[0] is True

        assert not screen.query("#library-ingest-analyze-skipped"), (
            "N must drop to 0 and the action must disappear"
        )
        canvas_text = "\n".join(
            str(row.renderable)
            for row in screen.query("#library-ingest-queue-panel Static")
        )
        assert "analysis skipped" not in canvas_text
        assert "analyzed · a.txt" in canvas_text, (
            "the auto-skipped row must show a resolved receipt"
        )
        assert "analyzed · b.txt" in canvas_text
