"""Library critique #9, Import queue (tasks 32216, 32231).

Two live findings from the 2026-09-10 dual-agent walk at dev 02374bf66a:

- 32216 (register row 13): pressing ``Show details`` on a failed row left
  focus on ``#library-ingest-keywords`` -- the ``Keywords (optional)`` input
  ~25 rows up the form -- because the toggle recomposes the queue panel and
  the pressed Button is pruned with it. A keyboard user reaching for Retry
  next was typing into the metadata field for the NEXT import.
- 32231 (register row 30): a folder import with one cause painted four
  byte-identical ``✗ failed`` rows x three buttons and no way to clear them
  in one gesture.

The host's POSIX semaphores are exhausted, so a real local import cannot
start its parse pool -- these pilots use the shell's
``_LibraryIngestCanvasHarness`` (real registry + writer, fake in-process
parse pool) and missing source files, which fail deterministically before
any pool work.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from tldw_chatbook.Library.library_ingest_jobs import IngestJobState
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    _INGEST_POLL_ATTEMPTS,
    _INGEST_POLL_INTERVAL,
    _LibraryIngestCanvasHarness,
    _open_library_ingest_canvas,
    _painted_text,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
    LibraryIngestQueuePanel,
)


async def _seed_failed_jobs(harness, pilot, paths):
    """Submit one job per missing path and wait for every one to fail."""
    jobs = [
        harness.submit_library_ingest_job(source_path=str(path), batch_id="batch-1")
        for path in paths
    ]
    for _ in range(_INGEST_POLL_ATTEMPTS):
        states = {job.job_id: job.state for job in harness.library_ingest_jobs.jobs()}
        if all(states.get(job.job_id) == IngestJobState.FAILED for job in jobs):
            break
        await pilot.pause(_INGEST_POLL_INTERVAL)
    else:  # pragma: no cover - a stuck queue is a harness failure
        raise AssertionError("seeded ingest jobs never failed")
    return jobs


@pytest.mark.asyncio
async def test_show_details_leaves_focus_on_the_button_it_toggled(tmp_path):
    """task-32216 AC#1/AC#2: the toggle owns its own focus target.

    Both directions: opening the details AND closing them again must leave
    focus on the control the user actually pressed, so Tab from there walks
    the queue rather than the form 25 rows up.
    """
    db = MediaDatabase(tmp_path / "crit9-import.db", client_id="crit9-focus")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        (job,) = await _seed_failed_jobs(harness, pilot, [tmp_path / "gone.md"])
        await _open_library_ingest_canvas(screen, pilot)

        button_id = f"library-ingest-details-{job.job_id}"
        await _wait_for_selector(screen, pilot, f"#{button_id}")
        screen.query_one(f"#{button_id}", Button).press()
        await _wait_for_selector(
            screen, pilot, f"#library-ingest-detail-{job.job_id}-0"
        )
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None) == button_id,
            message=lambda: (
                "Show details threw focus off the button it toggled: "
                f"{screen.focused!r}"
            ),
        )

        screen.query_one(f"#{button_id}", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen.query(f"#library-ingest-detail-{job.job_id}-0"),
            message="details never collapsed",
        )
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None) == button_id,
            message=lambda: (
                "Hide details threw focus off the button it toggled: "
                f"{screen.focused!r}"
            ),
        )
        keywords = screen.query_one("#library-ingest-keywords", Input)
        assert not keywords.has_focus, (
            "focus landed in the metadata form for the NEXT import"
        )


@pytest.mark.asyncio
async def test_show_details_refocus_rides_the_queue_panels_own_pump(tmp_path):
    """task-32216: the ordering the pilot above cannot see.

    ``Button.press()`` settles the message pump on its way out, so the test
    above passes whether the follow-up is queued on the SCREEN's hook or the
    queue panel's. Live it does not: the panel recomposes on its OWN pump, a
    ``call_after_refresh`` follow-up fires against the pre-recompose children,
    and the prune then drops focus into "Keywords (optional)" -- measured by
    clicking "Hide details" and typing, which put the characters in that
    field. Pin the seam, not the settled outcome.
    """
    db = MediaDatabase(tmp_path / "crit9-import.db", client_id="crit9-pump")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        (job,) = await _seed_failed_jobs(harness, pilot, [tmp_path / "gone.md"])
        await _open_library_ingest_canvas(screen, pilot)

        button_id = f"library-ingest-details-{job.job_id}"
        await _wait_for_selector(screen, pilot, f"#{button_id}")
        panel = screen.query_one(LibraryIngestQueuePanel)
        assert not panel.has_pending_recompose_callback

        # Run the handler to completion, then look BEFORE the pump gets a
        # turn -- the recompose it asked for has not happened yet.
        screen._on_ingest_job_details(
            Button.Pressed(screen.query_one(f"#{button_id}", Button))
        )
        assert panel._recompose_required, "the repaint never reached the panel"
        assert panel.has_pending_recompose_callback, (
            "the refocus was queued somewhere other than the panel's own "
            "post-recompose hook -- it will fire against the old children"
        )
        # ...and focus is PARKED for the duration, not left on a widget the
        # prune is about to drop: with a live target still set, Textual
        # re-picks one itself and a keystroke in that window lands in the
        # metadata form (measured live -- "kk" typed 100ms after the click
        # went into "Keywords (optional)").
        assert screen.focused is None, screen.focused

        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None) == button_id,
            message=lambda: f"refocus never landed: {screen.focused!r}",
        )


#: The exact failure a folder import takes on a host whose POSIX semaphores
#: are exhausted -- the shape that produced this finding. It names no file,
#: so all four rows carry the identical reason.
_POOL_START_ERROR = (
    "Ingest worker pool could not start: [Errno 28] No space left on device"
)


async def _seed_four_identical_failures(harness, pilot, tmp_path):
    """Four files in one batch failing for one file-independent cause."""
    screen = harness.screen_stack[-1]
    await _wait_for_library_shell(screen, pilot)
    jobs = await _seed_failed_jobs(
        harness, pilot, [tmp_path / f"note{n}.md" for n in range(4)]
    )
    # The missing-file failure names the path, so it is legitimately
    # per-file. Re-stamp the run with the one cause a whole folder shares.
    for job in jobs:
        harness.library_ingest_jobs.mark_failed(
            job.job_id, error=_POOL_START_ERROR
        )
    await _open_library_ingest_canvas(screen, pilot)
    rows = screen._build_library_ingest_state().queue_rows
    assert len({row.reason for row in rows}) == 1, [row.line for row in rows]
    # Newest-first: the group's key is whichever row leads the run.
    return screen, rows[0].job_id, jobs


@pytest.mark.asyncio
async def test_four_identical_failures_paint_one_row_with_three_actions(tmp_path):
    """task-32231 AC#1: one grouped row, three actions, members on demand."""
    db = MediaDatabase(tmp_path / "crit9-import.db", client_id="crit9-group")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, leader, jobs = await _seed_four_identical_failures(
            harness, pilot, tmp_path
        )
        await _wait_for_selector(screen, pilot, f"#library-ingest-group-{leader}")

        assert len(screen.query(".library-ingest-row")) == 1, [
            str(node.renderable) for node in screen.query(".library-ingest-row")
        ]
        # The queue sits below the fold on this canvas; assert the painted
        # glyphs, not a renderable, so a rule that paints OVER the row would
        # still be caught.
        queue = screen.query_one("#library-ingest-queue-panel")
        queue.scroll_visible(animate=False)
        await pilot.pause()
        painted = _painted_text(harness, queue.region)
        assert "✗ failed · 4 files · " in painted, painted
        for label in ("Show the 4 files", "Dismiss all"):
            assert label in painted, painted
        # Every per-file row and its buttons are gone until asked for.
        for job in jobs:
            assert not screen.query(f"#library-ingest-dismiss-{job.job_id}")


@pytest.mark.asyncio
async def test_show_the_files_expands_the_members_and_collapses_again(tmp_path):
    """task-32231 AC#1: the disclosure reveals the members' own rows."""
    db = MediaDatabase(tmp_path / "crit9-import.db", client_id="crit9-expand")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, leader, jobs = await _seed_four_identical_failures(
            harness, pilot, tmp_path
        )
        expand_id = f"library-ingest-group-expand-{leader}"
        await _wait_for_selector(screen, pilot, f"#{expand_id}")

        screen.query_one(f"#{expand_id}", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: len(screen.query(".library-ingest-row")) == 5,
            message=lambda: (
                "expanding never revealed the members: "
                f"{len(screen.query('.library-ingest-row'))} rows"
            ),
        )
        for job in jobs:
            assert screen.query_one(f"#library-ingest-dismiss-{job.job_id}", Button)
        assert "Hide the 4 files" in str(
            screen.query_one(f"#{expand_id}", Button).label
        )
        # Live-caught: the batch header belongs above the group and only
        # once -- the leader's own row used to re-emit it on expansion.
        headers = [
            str(node.renderable)
            for node in screen.query(".library-ingest-batch-header")
        ]
        assert len(headers) == len(set(headers)) == 1, headers
        # task-32216's discipline applies to the group toggle too.
        assert getattr(screen.focused, "id", None) == expand_id, screen.focused

        screen.query_one(f"#{expand_id}", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: len(screen.query(".library-ingest-row")) == 1,
            message="collapsing never hid the members again",
        )


@pytest.mark.asyncio
async def test_dismiss_all_clears_the_whole_group(tmp_path):
    """task-32231 AC#1: one gesture clears every member of the group."""
    db = MediaDatabase(tmp_path / "crit9-import.db", client_id="crit9-dismiss")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, leader, jobs = await _seed_four_identical_failures(
            harness, pilot, tmp_path
        )
        dismiss_id = f"library-ingest-group-dismiss-{leader}"
        await _wait_for_selector(screen, pilot, f"#{dismiss_id}")

        screen.query_one(f"#{dismiss_id}", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen.query(".library-ingest-row"),
            message=lambda: (
                "Dismiss all left rows behind: "
                f"{len(screen.query('.library-ingest-row'))}"
            ),
        )
        visible = {job.job_id for job in harness.library_ingest_jobs.jobs()}
        assert not visible & {job.job_id for job in jobs}


@pytest.mark.asyncio
async def test_retry_all_requeues_every_member(tmp_path):
    """task-32231 AC#1: Retry all reuses the per-row retry, once per member."""
    db = MediaDatabase(tmp_path / "crit9-import.db", client_id="crit9-retry")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, leader, jobs = await _seed_four_identical_failures(
            harness, pilot, tmp_path
        )
        retried: list[str] = []
        harness.retry_library_ingest_job = retried.append
        await _wait_for_selector(screen, pilot, f"#library-ingest-group-retry-{leader}")

        screen.query_one(f"#library-ingest-group-retry-{leader}", Button).press()
        await pilot.pause()
        # Newest-first render order, one call per member, no duplicates.
        assert sorted(retried) == sorted(job.job_id for job in jobs), retried
