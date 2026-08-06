---
id: TASK-2721
title: >-
  Footer-status timer crashes with NoMatches on splash and first-run wizard
  screens
status: Done
assignee: []
created_date: '2026-08-06 17:00'
labels:
  - app-shell
  - bug
  - uat
  - first-run
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Full-app UAT on `origin/dev` `b0185749c`: the in-app Logs screen showed "4 errors in buffer" on a fresh first-run session before any user-triggered feature was touched. All four are the same crash, hit twice (chained traceback pairs):

`app.py:8158` in `_schedule_footer_status_updates` does `self._db_size_status_widget = self.query_one(AppFooterStatus)`, which raised `textual.css.query.NoMatches: No nodes match 'AppFooterStatus' on Screen(id='_default')` once during splash and `... on FirstRunSetupWizard()` once while the wizard was up.

The footer-status callback assumes the active screen always mounts an `AppFooterStatus`; the splash screen and the first-run wizard don't. The errors are swallowed into the log buffer (nothing visible breaks), but every fresh install currently starts its session log with two tracebacks, which poisons the "Errors" signal on the Logs screen the very first time a user opens it — and any future consumer of that error count inherits the noise.

Evidence: Logs screen error filter captures, 2026-08-06 UAT session (fresh scratch profile, wizard path).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: extend `Tests/UI/test_ui_responsiveness.py` fake-app pattern — `query_one` raises `QueryError`; assert the timers are still scheduled, the widget cache is `None`, and no error is logged.
2. GREEN: in `_schedule_footer_status_updates`, catch `QueryError` around only the `AppFooterStatus` acquisition (cache → `None`, debug log) and continue to start the timers; keep the outer guard for genuine timer-setup failures. Per-tick updates already re-resolve the active screen's footer via `_active_footer_status`, so a `None` cache self-heals.
3. Note: the current behavior is worse than log noise — the aborted setup means the DB-size/token timers never start at all in any session where the deferred timer fires while splash/wizard is active.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] `_schedule_footer_status_updates` tolerates an active screen without `AppFooterStatus` (skips quietly, retries on a later tick or next screen).
- [x] A fresh first-run session (splash → wizard → main app) reaches the main app with zero errors in the Logs buffer.
- [x] A regression test drives the footer-status callback while a screen without the footer widget is active and asserts no error is recorded.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The `AppFooterStatus` acquisition in `_schedule_footer_status_updates` is now tolerant: a `QueryError` (splash/wizard active) sets the cache to `None`, logs at debug, and setup continues. This also fixes the worse latent defect the investigation surfaced: the old `except` aborted the whole method, so the DB-size and token timers never started in any session where the deferred timer fired while a footer-less screen was up. Per-tick updates already resolve the active screen's footer via `_active_footer_status`, so a `None` cache self-heals. Test: `test_footer_status_scheduling_tolerates_screen_without_footer` (Tests/UI/test_ui_responsiveness.py), watched RED (error logged, no timers) then GREEN. Files: tldw_chatbook/app.py, Tests/UI/test_ui_responsiveness.py.
<!-- SECTION:NOTES:END -->
