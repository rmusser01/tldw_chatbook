---
id: TASK-40818
title: Wait for Notes delete publication before disabled-control mouse checks
status: In Progress
created_date: 2026-11-14 01:09
assignee:
- '@codex'
updated_date: 2026-11-14 01:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The required Notes deletion test can deliver a real mouse click while a later delete-owned canvas publication is replacing its children. Keep the mouse and deletion assertions while observing the finished view before clicking.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The deletion journey verifies both disabled placement buttons with real mouse input after the delete and its follow-up Trash publication have finished rendering.
- [x] #2 The readiness boundary uses the existing bounded wait and fails if the required publication never settles; selection, painted labels, no mutation and direct action refusal assertions remain.
- [ ] #3 Focused affected tests and independent review pass; the normal self-hosted UI9 lane qualifies without test removal, isolation changes or larger timeouts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Existing test fixture readiness correction; no production or CI boundary change.
1. Inspect the exact detached-widget failure, trace finite delete-owned producers, and stage a delayed real Trash publication to prove the original click races it.
2. Reuse settled-canvas readiness and observe only delete-owned work before requiring the actual current mouse target; retain real clicks and all assertions.
3. Run affected tests, delayed-publication and stuck-publication controls, static checks and independent review; qualify normal UI9 before marking Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The existing deletion journey now observes mutation completion and the deleted note in the real Trash snapshot before exercising the disabled controls. Each actual mouse click waits for the existing settled-canvas predicate and a current attached/displayed positive-region button at the target coordinates; only transient NoWidget during readiness is handled. Click errors still propagate, and the click return is asserted (initial target only). All original selection, painted-label, disabled-state, no-dialog, no-mutation and direct refusal assertions remain. No production code, fixture isolation, selectors or timeout values changed. The formatter normalized existing lambda layout in the touched test module. Updated the testing lesson; ADR-359 applies, with no new ADR needed.

Evidence: normal gate43530844245 UI9 job132249289557 failed the deletion test with a detached-widget parent error in Textual mouse routing;150 other tests passed (/tmp/tldw-ci-eighth-ui9-failure.log). The exact test passed locally6.88s; delayed-removal and queued-refresh probes also passed, so the exact framework exception remains unreproduced locally. A controlled1s delay in the real delete-owned Trash read, with an observation-only assertion at the existing mouse call, proved the original Add click raced that pending publication (/tmp/tldw-ci-notes-delete-trash-red.log). The corrected exact case passed7.30s and all3 cases in the affected module passed17.98s under the same delayed read (/tmp/tldw-ci-notes-delete-{trash-green,module-green}.log). A30s delayed read fails the unchanged10s readiness wait (/tmp/tldw-ci-notes-delete-negative.log). Independent review found no actionable blockers and verified source ownership, real input/assertion retention, failure propagation and unchanged cleanup. Ruff, formatting, whitespace and backlog ID/readability checks pass; AST comparison retains every original assertion and test name. Loaded UI9 qualification remains pending. Task stays In Progress until live qualification.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
