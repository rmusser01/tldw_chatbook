---
id: TASK-38901
title: Wait for painted Notes picker actions before the single test click
status: Done
created_date: 2026-11-05 01:00
labels:
- ci
- testing
- library-notes
references:
- https://github.com/rmusser01/tldw_chatbook/actions/runs/41940008359/job/126963140839
documentation:
- backlog/docs/lessons-testing-evidence.md
updated_date: 2026-11-08 19:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the real production-CSS Notes folder-picker journey reliable when its toolbar has a pending layout frame. The existing enabled/nonzero-region predicate admits stale Pilot click coordinates; current-head UI Fast Lane17 timed out opening the first Move picker, and a native pending-layout diagnostic reproduces that helper hazard without proving the hosted pointer phase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The original unperturbed folder-picker journey and a controlled pending-layout variant open the picker with one click and preserve every existing folder, filter, title and database assertion.
- [x] #2 Readiness includes a completed Textual frame and a current live hit target within the existing ten-second budget; modal waits, mutation counts and other limits are unchanged.
- [x] #3 Affected UI checks, source-continuity evidence and an independent exact-head review pass without modifying production UI behavior or weakening checks.
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
Renumbered from this branch's former TASK-38703 before dev integration because the older upstream GGUF-source task owns that numeric ID. The installed Backlog renumber command preserved this picker task's body, creation metadata, ACs and status; prior source/history and RED/GREEN evidence remain retained. The scoped QA reference now uses this ID. New-head dev integration and CI/UAT qualification remain pending.

### Final software qualification on the accepted source (2026-11-08 UTC)

Source d6f494767bd38fe7462a4200f93ec39658035d81 is independently accepted with zero findings and has passed its OWN required aggregate and all43 Derived software jobs, including all32UI shards. The current census places this complete original picker file exactly once at zero-based index1231 in UI16, not its historical UI17 position. Own UI16 job128803538943/run42502880088 passed122cases without skips in659.38s; all three original picker cases, including settled and pending-layout variants, passed. The recorded11.20/11.10s are WHOLE TEST durations, not the helper's ten-second readiness elapsed time. Own UI17 and UI18 separately passed132/204cases. Actual roster, raw job log and source-bound ownership proof are retained in G/picker-current-census-ownership.json and G/own-d6f-Derived-job-logs/current-original-picker-UI16-proof.json.

The helper still requires a completed Textual frame and a current live compositor hit target within the ORIGINAL ten-second deadline, then issues exactly one original click. All modal waits, folder/filter/title/database assertions, parameters and production UI behavior remain unchanged. Meaningful403 RED and final404/405 GREEN remain historical source-bound receipts; current hosted results are a separate genuine qualification, not an unchanged-head retry. All current admission, P0, architecture, Dreams, Windows Eval, platform, Perf, Voice, lint and recovery gates passed with no actionable failure or unresolved thread. Original795browser15passes and796mounted remain retained through measured runtime/package/UAT-module/helper identity, not a later UAT or whole old Test-map claim.

AC1–3 and software DoD are satisfied by those actual results, the preserved source-continuity chain and independent exact-source review. ADR required:no; test-only synchronization preserves existing boundaries. The three actual task records and QA README are the only completion paths. All product/Test/workflow/script/packaging trees must remain identical; the metadata successor needs its own independent exact-head review and OWN full CI before conditional auto-merge. Every original source, failure, capture, ref, history and122 artifacts remains preserved.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The original one-click Notes picker and pending-layout variants pass on accepted source d6f494767bd38fe7462a4200f93ec39658035d81 in their current UI16 lane. Original ten-second painted-readiness deadline, all assertions and production UI behavior remain unchanged. Exact source review and all genuine source gates pass. Completion metadata requires its own review and CI before auto-merge.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
