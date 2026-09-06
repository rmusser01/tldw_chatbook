---
id: TASK-31733
title: Include the Subagents inspector section in staged-context ordering coverage
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:28'
updated_date: '2026-09-06 03:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve the inspector ordering contract after upstream moved Subagents into the right rail before staged context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The inspector body explicitly orders Environment, Tasks, Subagents, then staged context
- [x] #2 Staged context remains above run and source-readiness content with mounted geometry assertions unchanged
- [x] #3 The full Console session settings file and static checks pass
- [x] #4 The workbench contract also retains the exact Subagents-before-staged-context ordering; complete workbench and right-rail files pass with independent review.
- [x] #5 A deterministic late geometry invalidation is drained before both initial and swapped-state assertions, retaining exact geometry, widget identity, owner-pass counts and bounded waits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only reconciliation with the established upstream right-rail topology; no new UI or ownership decision.
1. Reproduce the full-file ordering failure and read right_rail.py composition.
2. Add the exact Subagents child expectation and preserve relative and painted geometry assertions.
3. Run the targeted topology test and full settings file with static checks.
4. Follow-up: the complete workbench file reproduces the same missed ordering expectation (1 failed/71 passed). Add its exact Subagents slot before staged context, preserving left-rail exclusion, pinned-authority ownership and staged-before-readiness assertions. Verify both complete workbench and right-rail files and obtain independent review. Existing session-settings evidence remains historical, not a new run.
5. Timing follow-up: add deterministic initial/swapped late-geometry variants using the real geometry-reconcile entry point at a controlled Pilot pause boundary. Observe their failure before changing wait ordering. Move the optional paint pause before the final predicate wait, so assertions immediately consume qualified state without another yield. Keep the shared bounded helper, all limits and every existing behavioral assertion unchanged; rerun both complete files and review the regression controls.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the inspector fixture to assert the upstream right rail exact sequence: Environment, Tasks, Subagents, staged context. Relative staged-before-run/readiness assertions and all mounted geometry remain unchanged. Original full-file RED: staged tray expected at child2 but the intentional Subagents section occupied that slot. Targeted GREEN passed; complete session-settings file passed 416 tests in 282.54s with RuntimeWarning escalated. Ruff lint/changed-region format passed; root reviewed the scope. Test-only current-topology repair, no ADR required; self-review complete.
<!-- SECTION:NOTES:END -->

### Workbench follow-up

The same obsolete third-child expectation remained in the workbench contract:
its complete baseline returned 71 passed/1 failed
(`/private/tmp/tldw-workbench-contract-baseline.xml`). The assertion now includes
the existing Subagents section before staged context, matching production
`right_rail.py` and the previously repaired session-settings test. Left-rail
exclusion, pinned-authority sequence/parentage and staged-before-live-work checks
remain unchanged. Independent review found no actionable issues. No production,
CSS, geometry or ownership changes were needed. The earlier 416-test
session-settings result remains historical evidence, not a fresh run.

Verification: complete workbench + right-rail selection returned 107 passed/1
failed in 199.23 seconds (`/private/tmp/tldw-workbench-ordering-final.xml`). All
72 workbench tests pass. The unchanged right-rail file failed its readiness-to-
pending initial geometry recheck: expected demand 15, viewport 15 and hidden
hint were present, but `_outer_reconcile_scheduled` became true after the extra
`pilot.pause()` following a successful predicate wait. A separate complete
right-rail rerun passed all 36 tests in 95.38 seconds
(`/private/tmp/tldw-right-rail-isolated-baseline.xml`). Three existing dependency
warnings remain. Lint, changed-region formatting and diff checks pass.

The intermittent right-rail result is not waived by the isolated pass. This
task remains In Progress with follow-up qualification open; diagnose the late
layout invalidation and reproduce it deterministically before changing the
readiness helper or runtime. No geometry, reconciliation-count or timing bound
assertion was relaxed.

### Right-rail timing closeout

The late-invalidation regression is now deterministic: initial and swapped phases
in both directions request real geometry reconciliation at the paint-pause return
boundary. With the old wait ordering, all four injected cases failed and both
ordinary cases passed (`/private/tmp/tldw-rail-late-geometry-red.xml`). The one-shot
injector restores the original pause and verifies exactly one injection occurred.

Moved each paint pause before its final bounded predicate wait, leaving no yield
between qualified state and assertions. All physical row geometry, stable widget
identity, local-before-outer ordering and exact logical owner-pass counts remain
unchanged. No production code, scheduler flags, shared helper or timing bounds
were changed. This test-only repair requires no new ADR.

Fresh complete right-rail + workbench verification: **112 passed in 241.61 seconds**
with three existing dependency warnings
(`/private/tmp/tldw-rail-late-geometry-final.xml`). Full-file Ruff lint,
changed-region formatting and diff checks pass. Independent review found no
actionable issues. Added the observed asynchronous-readiness trap to
`backlog/docs/lessons-testing-evidence.md`. This supersedes the open timing result
above; the original session-settings evidence remains historical. Broader review
failures and full-suite qualification remain outside this completed task.
