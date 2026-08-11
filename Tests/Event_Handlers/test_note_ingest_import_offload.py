# Tests/Event_Handlers/test_note_ingest_import_offload.py
"""TASK-15468 evidence: the "Import Selected Notes Now" loop must run off
the event loop.

Before this task, `import_worker_notes()` (the per-file parse / per-note
`notes_service.add_note` / template JSON I/O loop inside
`handle_ingest_notes_import_now_button_pressed`) was declared `async def`
but contained no internal `await`, so calling it with a bare `await` ran
the *entire* O(files x notes) loop inline, in one uninterruptible step, on
the main event loop -- freezing all UI input for the duration of a large
import (see Docs/Design/2026-08-11-input-latency-audit.md). The fix wraps
the now-plain-sync `import_worker_notes` in `asyncio.to_thread(...)`.

This file adds three tests on top of (not replacing) the pre-existing
`test_note_ingest_events.py` suite, which is left green and unmodified:

1. An "evidence-seam" test: the per-file parse and per-note `add_note`
   calls must happen on a different thread than the one that invoked the
   button handler.
2. A mechanism A/B test: a concurrent `await asyncio.sleep(0)` loop must
   get many scheduling turns while a (Mock-timed) large import runs, and
   almost none if the dispatch is reverted to the pre-fix on-loop shape.
3. An honest before/after probe against a *real* `CharactersRAGDB` /
   `NotesInteropService` and a few hundred real generated notes, reporting
   genuine wall-clock and event-loop-tick numbers for both shapes. This
   runs under pytest (not a bare script) specifically so it inherits the
   repo's config/data isolation from `Tests/conftest.py` -- see
   `backlog/docs/lessons-live-verification.md`, "Importing the test
   harness outside pytest is NOT config-isolated", for why a standalone
   script poking real `tldw_chatbook` DB/config code is unsafe here.
"""

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock

import pytest
from textual.css.query import QueryError
from textual.widgets import Button, Collapsible, RadioButton, TextArea

from tldw_chatbook.Event_Handlers import note_ingest_events
from tldw_chatbook.Event_Handlers.note_ingest_events import (
    handle_ingest_notes_import_now_button_pressed,
)


def _write_note_files(tmp_path: Path, note_count: int, files: int) -> List[Path]:
    """Write `note_count` notes spread round-robin across `files` JSON files.

    Each file is a JSON array of note objects -- `JSONImporter.parse_file`
    (tldw_chatbook/Utils/note_importers.py) turns each array entry into its
    own note, so this exercises both the per-file parse loop and the
    per-note `add_note` loop the audit flagged.
    """
    buckets: List[List[dict]] = [[] for _ in range(files)]
    for i in range(note_count):
        buckets[i % files].append(
            {"title": f"Note {i}", "content": f"Body for generated note {i}. " * 5}
        )
    paths = []
    for idx, bucket in enumerate(buckets):
        path = tmp_path / f"notes_{idx}.json"
        path.write_text(json.dumps(bucket), encoding="utf-8")
        paths.append(path)
    return paths


def _make_mock_app(note_files: List[Path]) -> Mock:
    app = Mock()
    app.selected_note_files_for_import = list(note_files)
    app.notes_user_id = "user-1"
    app.notes_service = Mock()
    app.notes_service.add_note = Mock(return_value="note-id-1")
    app.notify = Mock()
    app.call_later = Mock()
    app.screen = Mock()

    widgets = {
        "#import-as-templates-radio": Mock(spec=RadioButton, value=False),
        "#ingest-notes-import-status-area": Mock(spec=TextArea, text=""),
        "#chat-notes-collapsible": Mock(spec=Collapsible),
    }

    def query_one_side_effect(selector, widget_type=None):
        try:
            return widgets[selector]
        except KeyError:
            raise QueryError(f"{selector} not found")

    app.query_one = Mock(side_effect=query_one_side_effect)

    captured_worker = {}

    def run_worker_side_effect(worker_callable, **kwargs):
        captured_worker["callable"] = worker_callable
        return Mock()

    app.run_worker = Mock(side_effect=run_worker_side_effect)
    app._captured_worker = captured_worker
    return app


async def _dispatch_and_get_worker(app: Mock):
    await handle_ingest_notes_import_now_button_pressed(
        app, Button.Pressed(Mock(spec=Button))
    )
    return app._captured_worker["callable"]


# --- 1. Evidence-seam: parse + add_note run off the main/event-loop thread ---


@pytest.mark.asyncio
async def test_import_worker_runs_parse_and_add_note_off_the_main_thread(
    tmp_path, monkeypatch
):
    note_files = _write_note_files(tmp_path, note_count=6, files=2)
    app = _make_mock_app(note_files)

    main_thread_ident = threading.get_ident()
    add_note_threads: List[int] = []
    parse_threads: List[int] = []

    def add_note_side_effect(**kwargs):
        add_note_threads.append(threading.get_ident())
        return "note-id"

    app.notes_service.add_note = Mock(side_effect=add_note_side_effect)

    original_parse = note_ingest_events._parse_single_note_file_for_preview

    def spy_parse(*args, **kwargs):
        parse_threads.append(threading.get_ident())
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(
        note_ingest_events, "_parse_single_note_file_for_preview", spy_parse
    )

    worker_callable = await _dispatch_and_get_worker(app)
    await worker_callable()

    assert add_note_threads, "add_note was never called"
    assert parse_threads, "the per-file parse was never called"
    assert main_thread_ident not in add_note_threads, (
        f"add_note ran on the event-loop thread ({main_thread_ident}) -- "
        "the import loop is still blocking the UI (task-15468 regression)"
    )
    assert main_thread_ident not in parse_threads, (
        f"note parsing ran on the event-loop thread ({main_thread_ident}) -- "
        "the import loop is still blocking the UI (task-15468 regression)"
    )
    # The whole batch is handed to the executor in ONE `asyncio.to_thread`
    # call (not one hop per note), so every call lands on the same thread.
    assert len(set(add_note_threads) | set(parse_threads)) == 1


# --- 2. Mechanism A/B: the event loop keeps ticking while the import runs ---


@pytest.mark.asyncio
async def test_import_worker_keeps_event_loop_responsive_during_large_import(
    tmp_path, monkeypatch
):
    """TASK-15468 AC1. A/B against the SAME production dispatch function
    (`_run_note_import_worker_and_dispatch`, captured via the `run_worker`
    call), with only the thread-dispatch primitive swapped for an inline
    stand-in that reproduces the pre-fix on-loop shape for the "before" arm.
    """
    note_count = 40
    per_note_cost = 0.004  # seconds; stands in for a real synchronous DB commit
    note_files = _write_note_files(tmp_path, note_count=note_count, files=4)

    def make_app() -> Mock:
        app = _make_mock_app(note_files)

        def slow_add_note(**kwargs):
            time.sleep(per_note_cost)
            return "note-id"

        app.notes_service.add_note = Mock(side_effect=slow_add_note)
        return app

    async def run_and_count_ticks(app: Mock) -> int:
        worker_callable = await _dispatch_and_get_worker(app)
        import_task = asyncio.ensure_future(worker_callable())
        ticks = 0
        while not import_task.done():
            await asyncio.sleep(0)
            ticks += 1
        await import_task  # propagate any exception; avoid "never retrieved"
        return ticks

    # "after" -- real production path (asyncio.to_thread offload).
    after_ticks = await run_and_count_ticks(make_app())
    assert after_ticks >= 20, (
        f"expected the event loop to get many scheduling turns while a "
        f"{note_count}-note import ran off-thread; only got {after_ticks} -- "
        "asyncio.to_thread offload does not appear to be working"
    )

    # "before" -- same production code, `asyncio.to_thread` swapped for an
    # inline stand-in that reproduces the pre-TASK-15468 on-loop shape.
    async def _inline_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(note_ingest_events.asyncio, "to_thread", _inline_to_thread)
    before_ticks = await run_and_count_ticks(make_app())
    assert before_ticks <= 2, (
        f"expected the pre-fix (on-loop) shape to starve the event loop of "
        f"scheduling turns for the whole import; got {before_ticks} ticks -- "
        "the A/B baseline is not reproducing the original bug"
    )

    assert after_ticks > before_ticks * 10


# --- 3. Honest before/after probe: a real DB, a few hundred real notes ---


@pytest.mark.integration
@pytest.mark.asyncio
async def test_note_import_probe_hundreds_of_real_notes_before_after(
    tmp_path, monkeypatch
):
    """TASK-15468 AC1/AC3 honest probe -- real `CharactersRAGDB` /
    `NotesInteropService`, no artificial per-note delay. Reports genuine
    wall-clock and event-loop-tick numbers for both dispatch shapes; run
    with `-s` to see the printed line.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService

    note_count = 300
    file_count = 30
    note_files = _write_note_files(tmp_path, note_count=note_count, files=file_count)

    db_path = tmp_path / "chachanotes_probe.db"
    real_db = CharactersRAGDB(db_path, "probe_client")
    notes_service = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="probe_client",
        global_db_to_use=real_db,
    )

    def make_app() -> Mock:
        app = _make_mock_app(note_files)
        app.notes_service = notes_service
        app.notes_user_id = "probe-user"
        return app

    async def run_and_measure(app: Mock) -> Tuple[float, int]:
        worker_callable = await _dispatch_and_get_worker(app)
        start = time.monotonic()
        import_task = asyncio.ensure_future(worker_callable())
        ticks = 0
        while not import_task.done():
            await asyncio.sleep(0)
            ticks += 1
        results = await import_task
        elapsed = time.monotonic() - start
        successes = sum(1 for r in results if r.get("status") == "success")
        assert successes == note_count, (
            f"expected all {note_count} generated notes to import "
            f"successfully, got {successes}"
        )
        return elapsed, ticks

    # "after" -- real production path (asyncio.to_thread offload).
    after_elapsed, after_ticks = await run_and_measure(make_app())

    # "before" -- same production code, `asyncio.to_thread` swapped for an
    # inline stand-in that reproduces the pre-TASK-15468 on-loop shape.
    async def _inline_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(note_ingest_events.asyncio, "to_thread", _inline_to_thread)
    before_elapsed, before_ticks = await run_and_measure(make_app())

    print(
        f"\n[TASK-15468 probe] {note_count} real notes / {file_count} files, "
        f"real ChaChaNotes DB (WAL, synchronous=NORMAL):\n"
        f"  before (on-loop):  wall={before_elapsed * 1000:.1f}ms  "
        f"loop_ticks_during_import={before_ticks}\n"
        f"  after  (threaded): wall={after_elapsed * 1000:.1f}ms  "
        f"loop_ticks_during_import={after_ticks}\n"
    )

    # AC1: the event loop must get meaningfully more scheduling turns once
    # the import is off-thread, proving real input would be processed
    # during a real several-hundred-note import.
    assert before_ticks <= 2, (
        f"before-arm baseline did not reproduce a blocked loop: "
        f"{before_ticks} ticks (expected <= 2)"
    )
    assert after_ticks > before_ticks, (
        f"threaded import did not free the event loop more than the "
        f"on-loop shape: before={before_ticks} ticks, after={after_ticks} ticks"
    )

    # AC3: wall time must not be materially regressed by the thread hop --
    # a single whole-batch `asyncio.to_thread` dispatch adds microseconds of
    # overhead; this is a generous ceiling against a gross regression, not a
    # tight benchmark assertion.
    assert after_elapsed < max(before_elapsed * 3, before_elapsed + 0.25), (
        f"import wall time regressed materially: before={before_elapsed:.3f}s, "
        f"after={after_elapsed:.3f}s"
    )
