---
id: TASK-40728
title: Diagnose snapshot settings save deadlines on local CI
status: Done
created_date: 2026-11-13 22:09
assignee:
- '@codex'
updated_date: 2026-11-13 23:24
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two snapshot settings regressions time out waiting for asynchronous save completion on the local worker pool. Identify the delayed phase and preserve the persistence and newer-edit contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The failing snapshot-save phase is identified using timing and thread evidence on a representative runner.
- [x] #2 A focused correction preserves persistence, concurrent-edit assertions, isolation and meaningful bounded waits.
- [x] #3 Affected tests pass focused verification and the normal PR gate.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Diagnose and repair an existing regression within its current boundaries; reassess if the fix changes a runtime contract.
1. Read both CI failures and existing SaveTimeline/incident guidance; reproduce unchanged exact tests locally and on Linux as needed.
2. Profiled19277 bounded-section retries during a save; geometry probe shows ChatScreen.is_current remains true while SettingsScreen is on top, because Textual includes background screens. Change the existing covered-screen retry gate to identify the actual top screen, retaining its layout-signal wake and all save deadlines. Add a deterministic stale-background-layout regression, then replay exact Linux snapshot cases.
3. Run targeted regression and negative-control checks, independent review, and loaded PR qualification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The bounded Console section now identifies the actual top screen with `self.screen is self.app.screen` before immediately retrying geometry. Textual's `is_current` includes background screens, which allowed ChatScreen under SettingsScreen to spin against stale viewport dimensions. Covered sections retain their existing owner-layout signal wake, virtual callback and unmount cleanup. Snapshot-save code, deadlines, assertions and isolation are unchanged. No ADR is required for this existing scheduling-boundary bug fix; design-token/pattern rules are unaffected. Updated the Console architecture note and recorded the incident in lessons-testing-evidence.

Evidence: gate43510161110 UI jobs132180869664/132180869681 failed5-second save waits. Both unchanged exact tests passed on macOS19.00s, but an isolated Linux2vCPU/6GiB replay reproduced the newer-edit failure with4.96s event-loop CPU during5.01s elapsed. cProfile found19277 bounded-section retries; telemetry confirmed ChatScreen background/current while SettingsScreen was top, with11-row viewport against13-row target. The deterministic mounted stale-background-layout regression failed before the change (1566callbacks/0.1s), then passed including geometry resumption after uncovering. All53 focused bounded-section/shutdown/snapshot tests passed locally28.09s. Four Linux trials each passed the new regression and both original snapshot tests (3passed in23.10/22.37/23.23/22.71s); diagnostic service is inactive. These are focused Linux results, not full-gate timing. External evidence /tmp/tldw-ci-snapshot-{probe,profile,geometry,fixed}.log; production image/controller unchanged. Independent review found no blockers. Ruff, changed-test formatting, whitespace and backlog validators pass; unrelated pre-existing production-file formatting debt retained. Normal loaded PR qualification passed.
Live loaded normal PR qualification on head473734ddcb9b957e41916e3685a9d8b7460f8f12: run43519057842 UI1 job132211763086 passed106 tests with1 existing skip in696.06s; UI2 job132211763023 passed136 tests with1 existing skip in642.46s. Both original snapshot-save tests ran and passed with unchanged deadlines. Jobs completed2026-11-13T23:20:55Z and23:06:00Z on ghq-test-1046a690 andghq-test-57be63b6. Evidence: /tmp/tldw-ci-snapshot-ui{1,2}-live.log. Full gate remains in progress; this is lane qualification, not successful full-run timing. Bookkeeping stays local to avoid restarting healthy CI.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
