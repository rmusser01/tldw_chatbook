---
id: TASK-2721
title: >-
  Footer-status timer crashes with NoMatches on splash and first-run wizard
  screens
status: To Do
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

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] `_schedule_footer_status_updates` tolerates an active screen without `AppFooterStatus` (skips quietly, retries on a later tick or next screen).
- [ ] A fresh first-run session (splash → wizard → main app) reaches the main app with zero errors in the Logs buffer.
- [ ] A regression test drives the footer-status callback while a screen without the footer widget is active and asserts no error is recorded.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
