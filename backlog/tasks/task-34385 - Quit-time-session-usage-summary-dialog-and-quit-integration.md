---
id: TASK-34385
title: Quit-time session usage summary dialog and quit integration
status: To Do
assignee: []
created_date: '2026-10-07 03:34'
labels: []
dependencies:
  - TASK-34384
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-facing half of issue #365: SessionSummaryDialog modal (auto-dismiss + any-key skip) shown by the quit worker after approved-quit cleanup and before App.exit() with a hard cap so exit always proceeds; [session_summary] config section (default off; duration clamped 1-30); settings-screen instant-apply group; User Guide documentation. Spec: Docs/superpowers/specs/2026-09-22-quit-time-session-summary-design.md. Plan: Docs/superpowers/plans/2026-09-22-quit-time-session-summary.md (Tasks 4-7). Depends on TASK-34384 (ledger + tap points).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Config [session_summary] defaults exist with clamp tests
- [ ] #2 Enabled quit shows the dialog after cleanup then exits; disabled quit is byte-identical to today
- [ ] #3 Dialog renders token total and elapsed time with estimate marker and no-data fallback
- [ ] #4 Any key skips; auto-dismiss fires; stuck dialog hard cap still exits
- [ ] #5 Settings group persists toggle and duration; User Guide documents default and semantics
<!-- AC:END -->
