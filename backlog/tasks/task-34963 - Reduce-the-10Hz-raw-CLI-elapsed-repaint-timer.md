---
id: TASK-34963
title: Reduce-the-10Hz-raw-CLI-elapsed-repaint-timer
status: Done
assignee: []
created_date: 2026-10-13 14:50
labels:
- performance
- ui-perf
- wave-2-streaming
dependencies: []
priority: low
updated_date: 2026-10-13 16:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConsoleTranscript arms set_interval(0.1, _tick_raw_cli_elapsed) per message widget while a raw-CLI run is running/stopping (console_transcript.py:1546-1552) — 10Hz repaints of elapsed-text Statics for the whole run duration, when a 1Hz display (or coalesced tick) is the convention elsewhere in the app.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Elapsed repaint cadence reduced or coalesced (e.g. 1Hz) while run state changes land immediately
- [x] Test covers elapsed display during a raw-CLI run
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the raw-CLI elapsed timer + tick and its display granularity
2. Reduce cadence to 1Hz while terminal-state changes land immediately
3. Targeted test run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `_sync_raw_cli_timer` (console_transcript.py) now arms the elapsed ticker at 0.5s instead of 0.1s — 5× fewer repaints per mounted raw-CLI row for the whole run duration. Terminal-state changes were never carried by this timer (they land via sync_header), so immediacy is unaffected.
- Cadence choice: the label renders `{elapsed:.1f}s` (tenths), pinned as a contract by `test_raw_cli_status_copy_distinguishes_every_required_terminal_state` and shared with non-timer surfaces — so whole-second/1Hz would have made the tenths digit look stuck. At 2Hz the tenths advance in consistent half-second steps; the copy is always increasing.
- Test: `test_mounted_raw_cli_row_is_literal_focusable_and_lifecycle_complete` waited a fixed 0.2s assuming 10Hz ticks; it now polls up to 1.2s for the copy to advance (asserts the row owns a firing timer, not a cadence). Suite: `Tests/UI/test_console_raw_cli_transcript.py` 17/17 passed.
- ADR required: no — timer cadence on one widget.
