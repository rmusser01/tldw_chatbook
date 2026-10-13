---
id: TASK-34928
title: Stop-replaying-the-whole-log-buffer-synchronously-at-Logs-mount
status: Done
assignee: []
created_date: 2026-10-13 14:52
labels:
- performance
- ui-perf
- wave-3-screen-switch
dependencies: []
priority: medium
updated_date: 2026-10-13 17:51
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
logs_screen.on_mount calls logs_window.load_from_app() synchronously (logs_screen.py:73), replaying the entire in-memory log buffer into widgets on every Logs visit. The buffer grows for the whole session uptime, so the cost of opening Logs grows the longer the app has been running.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Logs mount is O(viewport) not O(buffer) — replay chunked or deferred to a worker
- [x] Latest log lines render within one refresh of mount (test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read LogsWindow.load_from_app and the buffer source
2. Chunk/defer the replay so mount is not O(buffer)
3. Test latest-lines-first rendering
4. Targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `load_from_app` (Logs_Window.py) now seeds only the newest `_MOUNT_SEED_RECORDS` (1000) records synchronously — snapshot + construct + count + render — so a Logs visit paints the latest lines within one refresh regardless of session uptime. The older backlog is reconstructed by a `logs-mount-backfill` worker in `_BACKFILL_CHUNK_RECORDS` (2000) chunks with `sleep(0)` yields between them, then spliced in FRONT of the seeded/live records in one event-loop-atomic section (no awaits inside), counts rebuilt, one final render.
- Ordering safety: the snapshot tuple is immutable, `_seeded_through` is still taken at snapshot time (live appends dedupe as before), live records arriving during the backfill land after the tail slice and stay last after the splice; deque-maxlen eviction during the splice removes from the spliced backlog's front, i.e. the oldest data by construction.
- Investigation note: `_render_view` was already capped (`MAX_RENDERED_LINES`, task-15476), so the mount cost was record construction + counting + the visible-scan over up to 10,000 records — bounded to 1,000 on the mount path now.
- Tests: new `Tests/UI/test_logs_mount_backfill.py` (3 tests: synchronous newest-slice seed with correct `_seeded_through`, full-buffer restore in order after the worker, live-record ordering preserved when arriving mid-backfill) — 3/3 pass. Regression: `Tests/test_logging_pipeline_single_pass.py` 17/17; the two Logs UI suites are at their pre-existing baseline (1 passed / 14 errors with the change stashed — same mount breakage as other UI suites).
- ADR required: no — load-path scheduling inside one widget.
