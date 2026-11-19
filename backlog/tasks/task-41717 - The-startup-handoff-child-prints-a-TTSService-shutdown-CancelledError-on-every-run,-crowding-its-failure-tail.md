---
id: TASK-41717
title: "The startup-handoff child prints a TTSService shutdown CancelledError on every run, crowding its failure tail"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 06:57
updated_date: 2026-11-19 15:17
labels:
- follow-up
- tests
- backup
priority: low
references:
- Tests/Backup_Recovery/test_runtime_startup_handoff.py
- Tests/Backup_Recovery/test_home_citation_retirement.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41532 (round 16). The startup-handoff child prints `unhandled exception during asyncio.run() shutdown` for `TTSService._complete_shutdown` (CancelledError) on every run, passing or failing. `_run` (`test_home_citation_retirement.py`) keeps only the last 6000 characters of a failure (7000 on the restart path), so this noise pushes the real assertion out of view.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The child's shutdown prints no CancelledError traceback, by fixing TTS shutdown or the child's teardown
- [x] #2 A failing child's assertion falls inside the kept tail, shown with a forced failure
<!-- AC:END -->

## Implementation Plan

1. Cause, from a passing child's captured log: the three startup-handoff scripts call `app.tts_service.close()` but never `wait_closed()`, so `TTSService._complete_shutdown` is still pending when `asyncio.run` ends; its cancellation is reported as `RuntimeError: TTS shutdown cleanup failed (CancelledError)`. The product path (`_close_bound_service`) does join it.
2. Join `wait_closed()` after `close()` in the three scripts, as the other Backup_Recovery child scripts already do.
3. `_run` keeps `stderr[-6000:] + stdout[-1000:]`, but stderr is merged into the log and is always empty, so the kept tail is 1000 characters. Keep `stdout[-7000:]`, as the restart path does.
4. Evidence: a passing child's log without the traceback, and a forced child assertion whose message sits in the kept tail.

## Implementation Notes

- Cause: the three scripts in `test_runtime_startup_handoff.py` awaited `app.tts_service.close()` but not `wait_closed()`. `close()` joins only the registry close; `TTSService._complete_shutdown` was still pending when `asyncio.run` ended, so its cancellation became `RuntimeError: TTS shutdown cleanup failed (CancelledError)` and asyncio printed "unhandled exception during asyncio.run() shutdown". The product path (`_close_bound_service`) joins `wait_closed()`, as the other Backup_Recovery child scripts do, so the fix is in the child teardown, not in TTS.
- Each script now joins `await app.tts_service.wait_closed()` after `close()`.
- `_run` (`test_home_citation_retirement.py`) kept `stderr[-6000:] + stdout[-1000:]`, but the child's stderr is merged into its log file and `stderr` was always `""`, so the kept tail was 1000 characters, not 6000. It now keeps `stdout[-7000:]`, the restart path's size. Every importer of `_run` gets the longer tail; only failure text changes.
- Evidence (load 38-58, `-n 4`, child logs copied out by a temporary line in `_run`, since removed): before the fix 8 of 8 passing children printed the traceback (`unhandled` 1, `TTS shutdown cleanup failed` 2 per log); after it 0 of 8, and all 8 nodes pass. A forced `assert False, 'FORCED-41717 child assertion'` after `settle_producers` (temporary): before the fix the 1000-character tail opened with the TTS traceback and kept the assertion only because the asyncio frames were short; after it the tail's last lines are the asyncio frames and `AssertionError: FORCED-41717 child assertion`, with no TTS traceback.
- Files: `Tests/Backup_Recovery/test_runtime_startup_handoff.py`, `Tests/Backup_Recovery/test_home_citation_retirement.py`.
- Runs: `Tests/Backup_Recovery/test_runtime_startup_handoff.py` 8 passed; with `Tests/test_logging_pipeline_single_pass.py` and `test_home_citation_retirement.py`, 41 passed.

