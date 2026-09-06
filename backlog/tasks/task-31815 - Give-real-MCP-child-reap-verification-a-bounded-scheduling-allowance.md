---
id: TASK-31815
title: Give real MCP child reap verification a bounded scheduling allowance
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 05:19'
updated_date: '2026-09-06 05:34'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove a test-only 10ms reap race while retaining proof that a failed kill preserves ownership and a later disconnect reaps the same real child.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A real child remains registered after the first denied kill and is reaped before registry removal on retry, including delayed wait delivery.
- [x] #2 The delayed real-child case fails with the old 10ms reap budget and passes with a bounded test-only allowance; close timeout and production deadlines remain unchanged.
- [x] #3 Complete MCP pagination tests, scoped static checks and independent review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test-only scheduling allowance with unchanged process ownership semantics. 1. Preserve deterministic 20ms delayed-reap diagnosis. 2. Parameterize the real-child kill-permission retry test with immediate and delayed wait delivery; prove delayed case RED under the original 10ms deadline. 3. Widen only this test reap allowance to 250ms while retaining the 10ms close deadline, real child and all ownership/privacy assertions. 4. Verify complete file and repeated exact cases, review and document evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added immediate and20ms wait-result-delivery variants to the real-child denied-kill retry test. Delayed case failed at second disconnect under old10ms reap allowance; test-only250ms allowance passes while close remains10ms. Actual child.wait and both ownership/privacy/reap assertions preserved; production deadlines unchanged.20/20 repeated variant runs pass; independent wholefile145passed3.57s,2 dependency warnings. Ruff, changed-region format, diff checks and independent review pass; unrelated existing fullfile formatting left untouched. Checkpoint and testing lesson updated. ADR required:no, test-only deadline isolation.
<!-- SECTION:NOTES:END -->
