---
id: TASK-32920
title: Stall records name the code the UI loop was stuck in
status: Done
assignee:
- '@claude'
created_date: 2026-09-23 15:43
labels:
- performance
- diagnostics
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Fedora user reported random multi-second UI lag. The UIResponsivenessMonitor (TASK-18908) records that the loop stalled and for how long, but not what code stalled it -- the heartbeat only notices a stall after the loop recovers, when the culprit is already off the stack. This machine's own log holds 42 stall records up to 9.6 s, none attributable. Without attribution every lag report is guesswork.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An event_loop_stall record names the innermost frame and the two deepest tldw_chatbook frames (module, function, line) the loop was executing during the stall
- [x] #2 Attribution carries only identifiers and line numbers through the existing persistent-diagnostics allowlist -- never locals, values or paths
- [x] #3 A stall observed without a live sample (synthetic delta) persists exactly as before, with no attribution fields
- [x] #4 The sampling thread is on the boot thread allowlist and exits when the monitor closes
- [x] #5 Verified in the real app: an induced loop block is recorded with its function name
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Watchdog daemon thread started by the first heartbeat (records the loop thread id)
2. While the heartbeat is overdue by the stall threshold, sample sys._current_frames() for the loop thread once per stall
3. Attach the sample to the edge-triggered stall record; clear it on every heartbeat
4. Allowlist leaf_*/caller_* fields next to TASK-32533's site_* fields
5. Unit test with real threads + real sink; boot census row; live headless-app probe
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Watchdog daemon thread (`ui-stall-watchdog`) in `Utils/ui_responsiveness.py`, started by the first heartbeat, polls cheaply and -- only while the heartbeat is overdue by the stall threshold -- samples the loop thread via `sys._current_frames()` once per stall. The sample rides on the existing edge-triggered `event_loop_stall` record: `leaf_module/leaf_function` (innermost frame, often library code such as a keyring backend), `site_*` (deepest tldw_chatbook frame, reusing TASK-32533's fields) and new `caller_*`. Identifier/line-only, through the persistent-diagnostics allowlist. A sample that races a recovering heartbeat is dropped.

Verified live: a headless real-app boot (the boot-census harness, isolated profile) with an induced 1.6 s loop block persisted `lag_ms=1069 leaf_function=load_marker_like_block`; an earlier run also caught a genuine 460 ms boot stall attributed to `Backup_Recovery.native_files.pinned_directory` via `qualification.qualified_for:259` (one sample marks where the loop was at the threshold crossing -- precise for one long call, representative for a run of short ones; not yet investigated).

Files: `Utils/ui_responsiveness.py`, `Utils/persistent_diagnostics.py`, `Tests/Utils/test_ui_responsiveness_stall_persist.py`, `Tests/Performance/test_boot_worker_census.py` (allowlist row).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
