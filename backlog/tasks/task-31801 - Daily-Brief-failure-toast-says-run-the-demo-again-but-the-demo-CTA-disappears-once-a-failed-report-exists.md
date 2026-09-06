---
id: TASK-31801
title: >-
  Daily Brief failure toast says 'run the demo again' but the demo CTA
  disappears once a failed report exists
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-05 19:15'
updated_date: '2026-09-06 14:50'
labels:
  - bug
  - ux
  - artifacts
  - watchlists
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). With no API key, Artifacts > 'Create Your First Daily Report' seeds the watchlist, fetches RSS, fails at the LLM stage, and the toast instructs fixing the key then 'run the demo again' - but once the failed report row exists, the CTA button is no longer rendered anywhere on the Artifacts screen, so the advertised retry path is gone (user must discover Watchlists to regenerate).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a failed demo brief, a retry affordance for the demo remains reachable from the Artifacts screen (or the toast copy points at a path that exists).
- [x] #2 Test covering the failed-demo retry affordance.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: seed a failed briefing row, open Artifacts; confirm the demo CTA is absent (it lives only in the no-reports else-branch).\n2. Add a _has_complete_report property; in the reports-exist branch keep the demo CTA (relabelled 'Run the Daily Report demo again', same #artifacts-daily-report-demo id + handler) until at least one report has completed.\n3. RED test: failed report keeps the retry CTA; control: a completed report hides it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced via a RED UI test (failed-status briefing seeded; #artifacts-daily-report-demo absent). Root cause: the demo CTA was rendered only in compose_content's no-reports else-branch, so any report row (including a failed one) removed the retry path the failure toast advertises ('...then run the demo again').

Fix: added the _has_complete_report property and, in the reports-exist branch, kept the demo control (relabelled 'Run the Daily Report demo again') reachable until at least one report reaches 'complete'. Reused the same #artifacts-daily-report-demo id so the existing @on handler and detached-task wiring are unchanged. Extracted the shared tooltip to DAILY_REPORT_DEMO_TOOLTIP.

Tests (Tests/UI/test_artifacts_screen_reports.py): test_failed_report_keeps_demo_retry_cta (RED->GREEN) and control test_completed_report_hides_demo_cta. The existing test_seeded_reports_list_rows_with_open_button (complete report -> no CTA) still passes.

Files: tldw_chatbook/UI/Screens/artifacts_screen.py, Tests/UI/test_artifacts_screen_reports.py, Docs/User_Guide/artifacts.md.
<!-- SECTION:NOTES:END -->
