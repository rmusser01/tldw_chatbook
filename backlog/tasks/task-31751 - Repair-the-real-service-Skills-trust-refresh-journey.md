---
id: TASK-31751
title: Repair the real-service Skills trust refresh journey
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 20:07'
updated_date: '2026-09-05 20:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The complete Skills library flow fails after bootstrap and an external skill edit: reselecting the row leaves the trusted v1 detail visible instead of the updated quarantined detail. The same failure reproduces before the current repairs. Determine whether the remaining issue is a stale fixture transition or runtime refresh defect and preserve the full real-service trust journey.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The first-run bootstrap, external skill update, review and approval journey observes real trust-state transitions and completes
- [x] #2 The fix preserves dirty-editor vetoes and all existing trust enforcement without bypassing service calls
- [x] #3 Complete Skills state and library-flow tests and scoped static checks pass
- [x] #4 Retry and pagination refresh the retained Skills Items pane while the open Work pane, editor widgets, draft text and cursor survive; selecting another item remains vetoed while dirty.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the pre-assembly and current real-service failure; inspect obsolete list-only admission/settlement gates and retained Items versus Work ownership under ADR-086. Remove the diagnostic Back workaround.
2. Characterize committed source refresh while a real Skills editor remains open; add real Retry/pagination failure-recovery coverage proving retained Items/Work/TextArea identity, draft text/cursor and dirty selection veto. Keep real service calls and all trust-state assertions.
3. Subject to parent approval, remove the three obsolete list-only browse gates while retaining route/generation fences. Follow the existing Prompt source-only sync boundary for Skills browse settlement so Work state and callbacks remain independently owned. Do not alter trust posture, import or detail policy gates.
4. Verify complete Skills state and real-service flow files, focused Skills canvas/wiring tests, scoped static checks, parent review and an independent commit.
ADR required: no
ADR path: backlog/decisions/009-local-skill-trust-boundary.md; backlog/decisions/086-library-adaptive-reader-shell.md
Reason: Restore existing trust enforcement and retained source/work independence; no new trust policy, service contract or ownership decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause confirmed: three list-only gates suppressed refresh admission, mutation refresh and accepted source-page projection while the permanent Skills Work editor was open. Removed only those obsolete gates, retaining route/generation fences. Browse settlement now sets sync_skill_work=False at the existing canvas-sync boundary (default True elsewhere), matching Prompt source-only projection and keeping browse callbacks on Items. Trust/import/detail gates are unchanged.
RED evidence: real bootstrap never refreshed retained Items; new real-service dirty-editor request never reached its source failure. After the fix both pass. Forcing only the previous Work-sync behavior via a test-local wrapper makes the new journey fail real TextArea identity after filtering, proving source/work separation is required. New journey exercises real source failure, Retry, next page, vetoed dirty row selection and submitted filtering while retaining Items/Work/TextArea identity, draft text and cursor. Removed root diagnostic Back workaround; original trust journey and v2 quarantine assertion remain.
Verification so far: complete Skills state29+flow24=53 passed. Combined architecture118 selection:117 passed37.99s, only concurrent Console ratchet slack505vs559 failed; Library checks green. Focused canvas16:15 passed, the unchanged previously paired baseline clean SimpleNamespace missing focused failure remains in test_action_library_skill_back_honors_dirty_guard. No new Ruff findings: screen40, Skills controller1 preexisting F401, canvas_sync0; targeted test Ruff, changed-range formatter and diffcheck pass. Logger statements unchanged. ADR009 and ADR086 preserve existing policy/ownership.

Final browse-controller file:7 passed1.42s, preserving exact generation and source-fence behavior. Parent reviewed full production/new-test diff and approved scoped commit. Final owned pins tightened to Library41301/1301 and Skills controller3139; all37 Library ratchet checks pass, pin Ruff and diffcheck pass. Full-state/flow evidence remains53 green; unrelated baseline fake and concurrent Console pin failures are explicitly qualified above rather than claimed green.
<!-- SECTION:NOTES:END -->
