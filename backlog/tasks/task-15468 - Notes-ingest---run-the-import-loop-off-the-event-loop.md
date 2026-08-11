---
id: TASK-15468
title: Notes ingest: run the import loop off the event loop
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: `Event_Handlers/note_ingest_events.py:306` ("Import Selected Notes Now") dispatches a coroutine worker whose own comment (`:638-641`) states it runs on the main event loop: per file a sync parse, per note a sync `notes_service.add_note(...)` transaction (INSERT + FTS triggers + commit/fsync), plus sync template JSON I/O — O(files x notes), serially. Importing dozens/hundreds of notes is a guaranteed multi-second full-app freeze, exactly matching the reported symptom on this surface.

Fix direction: `thread=True` worker with `call_from_thread` for the UI updates (the callbacks already marshal results), or `to_thread` the per-file/per-note body. Preserve per-note error accounting and the preview flow. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The UI processes input while a large import runs (evidence: interaction during an N-hundred-note import)
- [x] #2 Import results, preview, template handling, and per-note failure accounting unchanged (tests)
- [x] #3 Import wall-time not materially regressed (before/after)
<!-- AC:END -->

## Implementation Plan

1. Read `Event_Handlers/note_ingest_events.py:306-656` end to end; confirm
   `import_worker_notes()` (the per-file/per-note loop, `:344-518`) contains
   no internal `await`s despite being declared `async def` -- it's pure sync
   work (file parse, `notes_service.add_note`, template JSON I/O) executed
   inline on the caller's coroutine, which today is the main event loop
   (per the `_run_note_import_worker_and_dispatch` comment at `:630-641`).
2. Confirm `NotesInteropService._get_db`/`CharactersRAGDB` are safe to call
   from a non-main thread (`threading.Lock`-guarded instance cache +
   per-thread `threading.local` SQLite connections) so moving the loop to a
   worker thread doesn't need new locking.
3. Minimal-diff fix: drop `async` from `import_worker_notes` (make it a
   plain sync function -- it has no awaits to lose), and in
   `_run_note_import_worker_and_dispatch` replace
   `results = await import_worker_notes()` with
   `results = await asyncio.to_thread(import_worker_notes)`. This keeps
   `_run_note_import_worker_and_dispatch` itself `async def` and its
   `on_import_success_notes`/`on_import_failure_notes` calls unchanged and
   still on the main thread (asyncio.to_thread resumes the awaiting
   coroutine on the same event-loop thread once the executor thread
   finishes) -- no `call_from_thread` marshaling needed, and the existing
   `run_worker(..., group="file_operations")` call and its cancellation
   semantics are untouched. `asyncio.to_thread` is an established pattern
   elsewhere in this codebase (e.g. `app.py`, `ccp_character_handler.py`).
4. Add an "evidence-seam" test: patch `add_note`/the parse call to record
   `threading.current_thread()` and assert it differs from the thread that
   invoked the button handler.
5. Add a responsiveness test: run the import (with an artificially slowed
   sync `add_note`, e.g. `time.sleep`) as a background asyncio task and
   assert a concurrent lightweight loop keeps ticking (`await
   asyncio.sleep(0)`) with low latency while the import is in flight --
   demonstrating the event loop is not blocked (AC1).
6. Write an isolated before/after probe script (scratchpad, not committed)
   that calls `import_worker_notes()` directly (old, on-loop shape) vs
   `await asyncio.to_thread(import_worker_notes)` (new shape) against a
   few hundred generated note files + a real `NotesInteropService`/
   `CharactersRAGDB`, measuring max event-loop tick gap and total wall
   time for both arms -- honest before/after numbers for AC1/AC3.
7. Run the existing note-ingest + notes-service suites (baseline already
   green at 86 passed pre-change) plus the new tests; confirm nothing
   regresses.

## Implementation Notes

**Approach.** Two-line functional change in
`Event_Handlers/note_ingest_events.py`: `import_worker_notes` (the nested
per-file/per-note loop inside
`handle_ingest_notes_import_now_button_pressed`) lost its `async` keyword
(it had no internal `await`s, so this changes nothing about its body), and
`_run_note_import_worker_and_dispatch` now calls it via `results = await
asyncio.to_thread(import_worker_notes)` instead of `results = await
import_worker_notes()`. `asyncio.to_thread` hands the whole batch to the
default executor as a single call (not one hop per note) and, critically,
resumes the awaiting coroutine back on the *same* event-loop thread once
the executor thread finishes -- so `on_import_success_notes`/
`on_import_failure_notes` (both UI-touching: `query_one`, `notify`,
`call_later`) stay exactly where they were, called directly, no
`call_from_thread` marshaling needed. `run_worker(..., group="file_operations")`
is untouched, so cancellation/group semantics are unchanged. This is an
established pattern elsewhere in the codebase (`app.py`,
`ccp_character_handler.py`, `stts_profile_library.py`, etc).

**Why this shape over `thread=True` + `call_from_thread`:** it's the
smaller diff, and it keeps `_run_note_import_worker_and_dispatch` itself
`async def`, which is exactly the shape the pre-existing
`test_note_ingest_events.py` suite's `await worker_callable()` pattern
already assumes -- so that suite needed **zero** changes and stayed green
throughout (confirmed: 86 passed before AND after, byte-identical file).

**Thread-safety check (not just assumed):** `NotesInteropService._get_db`
caches one `CharactersRAGDB` per user_id behind a `threading.Lock`, and
`CharactersRAGDB` itself opens one SQLite connection per thread via
`threading.local` (`check_same_thread=False`) -- so calling
`notes_service.add_note` from a worker thread needs no new locking.
`note_importer_registry.parse_file` and the template JSON read/write are
plain sync file I/O, also thread-safe.

**Evidence (new file, does not touch the existing test file):**
`Tests/Event_Handlers/test_note_ingest_import_offload.py`, three tests:
1. `test_import_worker_runs_parse_and_add_note_off_the_main_thread` --
   evidence-seam: patches `add_note` and `_parse_single_note_file_for_preview`
   to record `threading.get_ident()`, asserts neither call lands on the
   thread that invoked the button handler.
2. `test_import_worker_keeps_event_loop_responsive_during_large_import` --
   A/B against the *production* dispatch function, with `asyncio.to_thread`
   swapped for an inline stand-in (`async def _inline: return fn(*a,**kw)`)
   to reproduce the pre-fix on-loop shape for the "before" arm. Counts how
   many `await asyncio.sleep(0)` scheduling turns a concurrent loop gets
   while a Mock-timed 40-note import runs: after >= 20 ticks, before <= 2.
3. `test_note_import_probe_hundreds_of_real_notes_before_after` -- the
   task's "isolated probe importing a few hundred generated notes": a
   *real* `CharactersRAGDB`/`NotesInteropService` (no per-note delay
   injected), 300 real generated notes across 30 files, same A/B swap.
   Written as a pytest test (not a bare script) specifically so it
   inherits `Tests/conftest.py`'s config/data isolation --
   `backlog/docs/lessons-live-verification.md` documents a prior incident
   where a bare-script probe touched the real user config/data dir; this
   task's own investigation nearly repeated it (a `python -c "import
   note_ingest_events"` sanity check, run without isolation, was later
   found to have only *read* `~/.config/tldw_cli/config.toml`, confirmed
   via mtime -- no write occurred, but it was a needless risk that a
   pytest-based probe avoids by construction). Honest measured numbers
   from one local run (fast M-series Mac, real WAL+`synchronous=NORMAL`
   DB per task-15465's fix, no artificial delay):
   `before (on-loop): wall=91.4ms ticks=1` vs.
   `after (threaded): wall=72.4ms ticks=4181` -- i.e. the on-loop shape
   gives a concurrent scheduling loop exactly one turn for the whole
   import regardless of duration, while the threaded shape gives it
   thousands, and wall time is not regressed (here, noise-level faster).

**Test run:** `Tests/Event_Handlers/test_note_ingest_events.py` +
`test_note_ingest_import_offload.py` + `Tests/Notes/test_notes_library_unit.py`
+ `test_notes_adapter.py` + `test_notes_integration.py` +
`test_notes_scope_service_library_canvas.py` + `test_sync_engine.py` +
`Tests/Tools/test_note_tool_user_id.py`: **114 passed**. Full
`Tests/Event_Handlers/`: **50 passed, 1 skipped** (pre-existing, unrelated
tilde-path environment skip). `ruff check` on both changed/added files:
clean.

**Files changed:** `tldw_chatbook/Event_Handlers/note_ingest_events.py`
(2-line functional change + comments); added
`Tests/Event_Handlers/test_note_ingest_import_offload.py`.
