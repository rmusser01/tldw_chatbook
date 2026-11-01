---
id: TASK-34385
title: Quit-time session usage summary dialog and quit integration
status: Done
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
- [x] #1 Config [session_summary] defaults exist with clamp tests
- [x] #2 Enabled quit shows the dialog after cleanup then exits; disabled quit is byte-identical to today
- [x] #3 Dialog renders token total and elapsed time with estimate marker and no-data fallback
- [x] #4 Any key skips; auto-dismiss fires; stuck dialog hard cap still exits
- [x] #5 Settings group persists toggle and duration; User Guide documents default and semantics
<!-- AC:END -->

## Implementation Notes

Executed via the same plan, Tasks 4-7.

- Approach: `[session_summary]` config (default off, duration clamped 1-30 with OverflowError guard); `SessionSummaryDialog` modal (splash-style idempotent timed close + any-key skip, no BINDINGS per ADR-031, fixed-width card per ConfirmationDialog pattern); quit integration in app_lifecycle.LifecycleMixin after approved-quit persistence and before `App.exit()` — fail-closed enabled read, `asyncio.wait_for` hard cap, CancelledError propagates, persistence failure skips the summary; settings-screen instant-apply group + User Guide documentation (default-off AC).
- Findings fixed along the way: bare config read broke pre-existing quit-guard harnesses (→ fail-closed read, strictly better); `width:auto` card rendered blank (Textual trap, lesson recorded); SVG `&#160;` broke multi-word asserts; final review's Critical (gateway nested-usage) fixed under task 34384's wave.
- Verification: 60 targeted tests green (ledger, taps, config, dialog pilot with render evidence, quit harness incl. stuck-dialog hard cap, settings helper, plus pre-existing quit-guard regressions); live tmux run with sandboxed config+DBs: quit via Ctrl+Q shows the card ("Session summary / No usage recorded this session / 0m session / press any key to exit"), auto-dismisses, process exits 0.
- Files: config.py; app_lifecycle.py; Widgets/session_summary_dialog.py; UI/Screens/settings_screen.py; Docs/User_Guide/settings.md; Tests/{test_config_session_summary_defaults,UI/test_session_summary_dialog,UI/test_session_summary_quit,UI/test_settings_session_summary}.py. Commits 0028a32662, 14f789d8cd, 8e278f79ee, 7cdcd1e5a8.
- ADR: none required (see spec).
