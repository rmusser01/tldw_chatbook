---
id: TASK-40004
title: Back off refused Console full refreshes after readiness warm-up
status: In Progress
created_date: 2026-11-09 07:33
labels:
- ci
- console
- performance
priority: high
references:
- Tests/UI/test_console_projection_reads_off_loop.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- tldw_chatbook/UI/Screens/chat_screen.py
updated_date: 2026-11-09 08:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI3 observed six whole Console refreshes in the idle cold-config window. Mounted traces show that checked presentation can refuse after a successful warm-up, taking the fixed whole-refresh replay rather than the existing refusal backoff.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An actual checked display refusal after successful warm-up uses the existing bounded whole-refresh retry schedule and recovers when custody clears.
- [x] #2 The unchanged idle unparseable-config case and related native presentation cases pass without weakening source, freshness, lock, or attach checks or raising runtime limits.
- [ ] #3 Independent exact review and the published repair’s own hosted gates verify the correction.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a retry ownership correction inside the existing checked display and full-refresh contracts. It does not change native storage admission or the timing policy.

1. Retain the actual UI3 failure and trace full refresh, core, checked display, rail and retry paths on the original private-profile test.
2. Reproduce a fixed-delay replay after real downstream checked-display refusal while retaining the original source and native guards.
3. Route unfinished full presentation through the existing bounded retry schedule, preserving direct controls, source switches, teardown, maintenance resume and immediate explicit recovery.
4. Use the sibling test’s existing real execution-owner fixture for the historical cadence case that failed at store construction. Preserve all historical-read assertions and the explicit store close.
5. Verify the focused native control, original idle case, connected presentation tests, fresh original UAT, and exact independent review before publication. Require the successor’s own hosted gates before closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
UI3 logged six whole refreshes in the unchanged cold-config assertion window. Local traces observed expired checked displays and coordinator-lock refusal after successful warm-up, each scheduling a whole refresh. The hosted trigger was unobserved. Real lock controls reproduced three fixed 0.2-second timers at both the checked projection and rail phases. The ordinary full render now owns the existing bounded refusal retry through both phases. The same attempted owner is required to continue a streak. Native checks and the active-poll, direct-control and forced-settle paths retain their behavior.

The retry scope accepts only the live issuing task and thread. A nested refusal remains owed if the full render raises. The original exception stays primary. Connected qualification initially found an existing cadence fixture opening its writable AgentRuns store without a lease. The same real lease helper as its sibling test now supplies the owner, and all six original assertions and the explicit close remain.

The original idle assertion, focused controls, corrected cadence fixture, 34 boot checks and 24 compiled-source controls passed. The initial focused projection case inherited a prior mount retry and showed the advancing backoff. Both corrected controls then passed after waiting for an actual completed full render, with production unchanged. Fresh browser UAT passed all 15 cases, and mounted UAT passed its parent and three children with ordinary reopen, teardown and empty guards. The hosted gates and exact committed review are still due. No ADR is needed for the existing retry contract.

Independent review found that a long same-owner refusal streak overflowed before applying the existing five-second cap. The delay now saturates before multiplication. The focused four cases passed, and an executable comparison verified identical prior delays for counts through 1024 in both maintenance states. The fresh original UAT results carry through that equivalence. Hosted qualification remains pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
