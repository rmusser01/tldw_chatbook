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

from dataclasses import replace

import pytest
from textual.widgets import Button, Input

from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJob,
)
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestFormState,
    build_library_ingest_state,
    group_ingest_queue_rows,
)
from Tests.UI.test_library_ingest_canvas import _QueuePanelHost
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
    # The group's own key -- derived from the outcome, not from whichever
    # row happens to lead the run (Qodo 5). Newest-first, so ``rows[0]`` is
    # the leading MEMBER, which per-row actions still address by job id.
    return screen, group_ingest_queue_rows(rows)[0].key, rows[0].job_id, jobs


@pytest.mark.asyncio
async def test_four_identical_failures_paint_one_row_with_three_actions(tmp_path):
    """task-32231 AC#1: one grouped row, three actions, members on demand."""
    db = MediaDatabase(tmp_path / "crit9-import.db", client_id="crit9-group")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, group_key, leader, jobs = await _seed_four_identical_failures(
            harness, pilot, tmp_path
        )
        await _wait_for_selector(screen, pilot, f"#library-ingest-group-{group_key}")

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
        for label in ("Show the 4 files", "Retry all", "Dismiss all"):
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
        screen, group_key, leader, jobs = await _seed_four_identical_failures(
            harness, pilot, tmp_path
        )
        expand_id = f"library-ingest-group-expand-{group_key}"
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
        screen, group_key, leader, jobs = await _seed_four_identical_failures(
            harness, pilot, tmp_path
        )
        dismiss_id = f"library-ingest-group-dismiss-{group_key}"
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
        # ...and every member left the task-2140 durable record behind, the
        # one reason `_dismiss_library_ingest_job` was extracted at all.
        ledger = {job.job_id for job in screen._ingest_state.recent_ledger}
        assert ledger >= {job.job_id for job in jobs}, ledger
        # (review finding 2) The pressed button is gone with the group, so
        # the toggle's "keep focus on yourself" rule cannot apply -- but
        # focus must still LAND somewhere. Parked at None, the user's next
        # Tab restarts from the top of the screen.
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None)
            == "library-ingest-path",
            message=lambda: (
                "Dismiss all stranded focus instead of landing it on the "
                f"import form: {screen.focused!r}"
            ),
        )


@pytest.mark.asyncio
async def test_retry_all_requeues_every_member(tmp_path):
    """task-32231 AC#1: Retry all reuses the per-row retry, once per member."""
    db = MediaDatabase(tmp_path / "crit9-import.db", client_id="crit9-retry")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, group_key, leader, jobs = await _seed_four_identical_failures(
            harness, pilot, tmp_path
        )
        retried: list[str] = []
        harness.retry_library_ingest_job = retried.append
        await _wait_for_selector(screen, pilot, f"#library-ingest-group-retry-{group_key}")

        screen.query_one(f"#library-ingest-group-retry-{group_key}", Button).press()
        # Wait for the calls rather than one pump turn: a single `pause`
        # lost the race once under load and reported an empty list.
        await _wait_for_condition(
            pilot,
            lambda: len(retried) == len(jobs),
            message=lambda: f"Retry all did not reach every member: {retried}",
        )
        # Newest-first render order, one call per member, no duplicates.
        assert sorted(retried) == sorted(job.job_id for job in jobs), retried
        # (review finding 2) The group action keeps focus the way every
        # other queue toggle now does: on itself while it survives, on the
        # import form once the group it belonged to is gone. Never nowhere.
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None)
            in (f"library-ingest-group-retry-{group_key}", "library-ingest-path"),
            message=lambda: (
                f"Retry all stranded focus: {screen.focused!r}"
            ),
        )


# --- review finding 1 (Medium): STT failures never offer a bare "Retry all" ---


def _stt_failed_job(n: int):
    """One failed job whose recovery is a chosen model, not a bare requeue."""
    return LibraryIngestJob(
        job_id=f"ingest-job-{n}",
        source_path=f"/tmp/audio/talk{n}.mp3",
        state=IngestJobState.FAILED,
        submitted_at=100.0,
        finished_at=120.0,
        error="Transcription failed: the model could not be loaded.",
        error_detail={
            "category": "stt_failure",
            "actions": ["retry_faster_whisper", "choose_another_gguf"],
        },
    )


@pytest.mark.asyncio
async def test_a_group_of_stt_failures_offers_no_bare_retry_all():
    """The collapsed row must not offer what the expanded rows withhold.

    A row whose ``error_detail`` is an ``stt_failure`` gets "Choose another
    GGUF…" / "Retry with faster-whisper" INSTEAD of plain "Retry" -- a bare
    requeue fails the same way against the same broken provider. A folder of
    audio files failing on one missing model is a contiguous run of
    identical failures, so it collapses; the group must inherit that gate,
    or the collapsed and expanded views of the same rows disagree about
    what is allowed.
    """
    jobs = tuple(_stt_failed_job(n) for n in range(3))
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    rows = state.queue_rows
    assert len(group_ingest_queue_rows(rows)) == 1, [row.line for row in rows]
    # Guard the fixture: without per-row retryability the gate below would
    # pass for the wrong reason.
    assert all(row.can_retry for row in rows), [row.can_retry for row in rows]

    host = _QueuePanelHost(state)
    async with host.run_test(size=LIBRARY_TEST_SIZE):
        group_key = group_ingest_queue_rows(rows)[0].key
        assert host.query_one(f"#library-ingest-group-expand-{group_key}", Button)
        assert host.query_one(f"#library-ingest-group-dismiss-{group_key}", Button)
        assert not host.query(f"#library-ingest-group-retry-{group_key}"), (
            "the grouped row offers a bare Retry all for failures whose own "
            "rows deliberately withhold Retry"
        )


@pytest.mark.asyncio
async def test_a_group_of_research_owned_stt_failures_still_offers_retry_all():
    """(re-review finding A) The mirror image of the gate above.

    A Research-Workspace-owned job never gets the GGUF picker: its row shows
    a plain "Retry Research source" whatever its ``error_detail`` says. So a
    run of research-owned STT failures must still offer "Retry all" -- the
    group has to match the row's gate for that ownership too, not just for
    the category.
    """
    jobs = tuple(
        replace(_stt_failed_job(n), research_source_operation_id="op-1")
        for n in range(3)
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    rows = state.queue_rows
    assert len(group_ingest_queue_rows(rows)) == 1, [row.line for row in rows]
    assert all(row.research_owned and row.can_retry for row in rows)

    host = _QueuePanelHost(state)
    async with host.run_test(size=LIBRARY_TEST_SIZE):
        group_key = group_ingest_queue_rows(rows)[0].key
        assert host.query_one(f"#library-ingest-group-retry-{group_key}", Button), (
            "the group withheld Retry all from rows that each offer a plain "
            "retry of their own"
        )


@pytest.mark.asyncio
async def test_a_toggle_never_steals_focus_back_from_the_user(tmp_path):
    """(re-review finding B) The chained fallback honours the stand-down.

    ``preserve_same_id_focus_after_recompose`` refuses to restore the
    captured id when the user has already moved focus to a different,
    still-attached widget -- but it then calls whatever was chained behind
    it unconditionally. The chained fallback therefore has to make the same
    decision itself, or a toggle re-steals focus out from under a user who
    Tabbed away in the window between the press and the repaint.
    """
    db = MediaDatabase(tmp_path / "crit9-import.db", client_id="crit9-steal")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        (job,) = await _seed_failed_jobs(harness, pilot, [tmp_path / "gone.md"])
        await _open_library_ingest_canvas(screen, pilot)

        button_id = f"library-ingest-details-{job.job_id}"
        await _wait_for_selector(screen, pilot, f"#{button_id}")

        # Run the handler to completion -- focus is now parked and the
        # restore is queued -- then move focus the way a user would before
        # the repaint has landed.
        screen._on_ingest_job_details(
            Button.Pressed(screen.query_one(f"#{button_id}", Button))
        )
        keywords = screen.query_one("#library-ingest-keywords", Input)
        screen.set_focus(keywords)
        assert screen.focused is keywords

        await _wait_for_selector(
            screen, pilot, f"#library-ingest-detail-{job.job_id}-0"
        )
        await pilot.pause()
        assert getattr(screen.focused, "id", None) == "library-ingest-keywords", (
            "the toggle stole focus back from the field the user moved to: "
            f"{screen.focused!r}"
        )


@pytest.mark.asyncio
async def test_dismissing_the_leading_member_keeps_the_group_expanded(tmp_path):
    """(Qodo 5) An open group survives losing the row it was keyed by.

    Keyed by the leading job id, the per-row Dismiss on the first visible
    member re-keyed the whole run, and the panel's expansion set -- which
    holds keys -- no longer matched, so the rows the user had just opened
    collapsed under them mid-task.
    """
    db = MediaDatabase(tmp_path / "crit9-import.db", client_id="crit9-rekey")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, group_key, leader, jobs = await _seed_four_identical_failures(
            harness, pilot, tmp_path
        )
        expand_id = f"library-ingest-group-expand-{group_key}"
        await _wait_for_selector(screen, pilot, f"#{expand_id}")

        screen.query_one(f"#{expand_id}", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: len(screen.query(".library-ingest-row")) == 5,
            message="expanding never revealed the members",
        )

        # The user dismisses the first visible member from its own row.
        screen.query_one(f"#library-ingest-dismiss-{leader}", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen.query(f"#library-ingest-dismiss-{leader}"),
            message="the leading member was never dismissed",
        )
        await pilot.pause()

        # 1 group row + the 3 surviving members, still open.
        assert len(screen.query(".library-ingest-row")) == 4, (
            "the group collapsed when it lost the member it was keyed by"
        )


@pytest.mark.asyncio
async def test_dismiss_all_really_clears_a_skipped_group(tmp_path):
    """(Qodo 1, declined) Skipped rows ARE dismissible -- proof, not prose.

    The bot read ``LibraryIngestJobRegistry.dismiss``'s docstring ("Hide a
    FAILED or CANCELLED job"), which is stale: ``_DISMISSIBLE_STATES``
    (library_ingest_jobs.py) has included ``SKIPPED`` since task-2220. This
    pins the behaviour end to end so the claim cannot be re-raised from the
    same stale sentence.
    """
    db = MediaDatabase(tmp_path / "crit9-import.db", client_id="crit9-skip")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        jobs = await _seed_failed_jobs(
            harness, pilot, [tmp_path / f"skip{n}.md" for n in range(3)]
        )
        for job in jobs:
            harness.library_ingest_jobs.mark_skipped(
                job.job_id, reason="Unsupported file type: .md."
            )
        await _open_library_ingest_canvas(screen, pilot)

        rows = screen._build_library_ingest_state().queue_rows
        assert all(row.state is IngestJobState.SKIPPED for row in rows)
        group = group_ingest_queue_rows(rows)[0]
        assert len(group.members) == 3, [row.line for row in rows]

        dismiss_id = f"library-ingest-group-dismiss-{group.key}"
        await _wait_for_selector(screen, pilot, f"#{dismiss_id}")
        screen.query_one(f"#{dismiss_id}", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen.query(".library-ingest-row"),
            message=lambda: (
                "Dismiss all left skipped rows behind: "
                f"{len(screen.query('.library-ingest-row'))}"
            ),
        )
        visible = {job.job_id for job in harness.library_ingest_jobs.jobs()}
        assert not visible & {job.job_id for job in jobs}
