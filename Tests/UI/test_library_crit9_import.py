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
