---
id: TASK-26038
title: 'Config: pick up external edits without a manual reload'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:47'
updated_date: '2026-09-01 22:45'
labels:
  - ops
  - config
  - ux
dependencies:
  - TASK-26036
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Editing config.toml outside the app has no effect until the user finds the reload button. Verified on origin/dev: the config cache is keyed on path only (config.py:5101-5107) and requires force_reload=True to re-read; the only user-facing path is a Reload config action in Settings (UI/Screens/settings_screen.py:9037), and a named grep for watchdog, Observer( and hot reload returns no config hits. A user who edits the file in their editor sees stale behavior and reasonably concludes their edit did not work. Hermes keys its cache on file mtime and size so the next read picks up an external edit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An external edit to the config file is picked up on the next read without a manual reload action
- [x] #2 Detection is by cheap file metadata comparison, not a filesystem watcher or a polling thread
- [x] #3 A file being written concurrently does not cause a partial parse - a failed parse falls back per the last-known-good behavior
- [x] #4 The existing in-process config generation protocol still holds: readers that validated their view are not silently invalidated mid-operation
- [x] #5 The manual reload action continues to work
- [x] #6 No measurable overhead is added to the config read hot path - measured and recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: external edit picked up next read (throttle=0); unchanged file returns SAME object; throttle window suppresses re-stat; manual reload still works\n2. Throttled inline (mtime_ns,size) stamp check: _current_config_file_stamp + _external_edit_detected (monotonic throttle, no thread)\n3. Lock-free fast path stats only when it would otherwise HIT; an edit forces the locked re-read; stamp recorded on successful load\n4. Measure per-call overhead
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TENSION RESOLVED: AC#1 (pick up an external edit on the NEXT read) + AC#2 (cheap metadata, NO watcher/thread) + AC#6 (no measurable overhead) collide for a lock-free zero-I/O read path (TASK-21124). A naive per-read os.stat measured +1.22us/call = +488us per 400-call render — a real regression. Resolution: a THROTTLED inline (mtime_ns, size) check — _external_edit_detected stats at most once per _CONFIG_STAT_THROTTLE_SECONDS (default 1.0s), so a burst of ~400 get_cli_setting calls in one render statts ONCE, not 400×. Measured added per-call cost within the window = 0.054us (one time.monotonic + compare) = +21.6us/render, plus one ~1.3us stat per second — negligible (AC#6). The check runs ONLY when the fast path would otherwise HIT (a miss already re-reads), turns that hit into a forced locked re-read (never a false hit, so TASK-21124's lock-free soundness is intact — AC#4), and the changed file re-reads through the last-known-good path from 26036 so a concurrent partial write falls back safely (AC#3). Stamp recorded on every successful load; force_reload / manual Settings reload untouched (AC#5). Throttle exposed as a module global for deterministic tests (0.0 = always stat). 4 new tests. Two test_config_read_fastpath_task21124 failures remain PRE-EXISTING (re-bisected: fail with config.py reverted). Trade-off documented: an edit is seen on the next read AFTER the ≤1s throttle window, not literally the next microsecond — the honest reading of AC#1 'without a manual reload'.
<!-- SECTION:NOTES:END -->
