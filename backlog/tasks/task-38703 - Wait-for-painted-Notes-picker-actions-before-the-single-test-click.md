---
id: TASK-38703
title: Wait for painted Notes picker actions before the single test click
status: In Progress
created_date: 2026-11-05 01:00
labels:
- ci
- testing
- library-notes
references:
- https://github.com/rmusser01/tldw_chatbook/actions/runs/41940008359/job/126963140839
documentation:
- backlog/docs/lessons-testing-evidence.md
updated_date: 2026-11-05 01:12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the real production-CSS Notes folder-picker journey reliable when its toolbar has a pending layout frame. The existing enabled/nonzero-region predicate admits stale Pilot click coordinates; current-head UI Fast Lane17 timed out opening the first Move picker, and a native pending-layout diagnostic reproduces that helper hazard without proving the hosted pointer phase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The original unperturbed folder-picker journey and a controlled pending-layout variant open the picker with one click and preserve every existing folder, filter, title and database assertion.
- [ ] #2 Readiness includes a completed Textual frame and a current live hit target within the existing ten-second budget; modal waits, mutation counts and other limits are unchanged.
- [ ] #3 Affected UI checks, source-continuity evidence and an independent exact-head review pass without modifying production UI behavior or weakening checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only synchronization correction preserving existing runtime, UI, storage and security boundaries.

1. Retain the hosted UI17 failure and unchanged local control, then prove the helper's pending-layout hazard with one native offset change and actual event-recipient observations (diagnostic399).
2. Add a permanent pending-layout parameter alongside the original real production-CSS journey and verify its original modal assertion fails before correcting the helper.
3. Await painted readiness and reacquire/hit-test the live button within the existing ten-second readiness deadline; issue exactly one original click and preserve every dialog/filter/database assertion.
4. Run the affected file and UI17 cohort, scoped static checks and unchanged-product evidence. Obtain independent review, publish to the actual PR ref with a fresh exact lease while auto is off, and require the new head's genuine hosted gates before final reconciliation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The shared test picker helper now waits for a Textual frame, reacquires the live visible/enabled button and checks its effective compositor target within the existing ten-second readiness budget. It then issues one original Pilot click; every modal/folder/filter/title/database assertion and all production behavior remain unchanged. The original journey stays beside a native pending-layout parameter; styles are restored in finally. Independent review caught and corrected a setup lookup before button composition.

Diagnostic399 captured the same button moving while all three mouse events hit its old point. Permanent final arm403 fails the original modal assertion with the old helper; selector-setup run400 is retained but excluded from evidence. Exact UI17 cohort404 passed all96 cases in303.29s with the unchanged180s per-case limit. After mechanical import sorting, final file405 passed all3 in24.37s; Ruff passes. Both runs physically retired/source_changes[]. AST continuity preserves the original journey and all sibling behavior. All3488 production blobs equal Windows-qualified e0422a; original browser/mounted UAT inputs are unchanged.

No ADR required: test-only synchronization within existing boundaries. Modified paths are the picker test, scoped QA note, incident lesson and this task. Local verification and source review are complete; exact committed-head review, publication and the new head's genuine hosted gates remain required. Keep In Progress until final reconciliation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
