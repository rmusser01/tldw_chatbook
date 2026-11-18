---
id: TASK-40818
title: Wait for Notes delete publication before disabled-control mouse checks
status: Done
created_date: 2026-11-14 01:09
assignee:
- '@codex'
updated_date: 2026-11-14 02:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The required Notes deletion test can deliver a real mouse click while a later delete-owned canvas publication is replacing its children. Keep the mouse and deletion assertions while observing the finished view before clicking.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The deletion journey verifies both disabled placement buttons with real mouse input after the delete and its follow-up Trash publication have finished rendering.
- [x] #2 The readiness boundary uses the existing bounded wait and fails if the required publication never settles; selection, painted labels, no mutation and direct action refusal assertions remain.
- [x] #3 Focused affected tests and independent review pass; the normal self-hosted UI9 lane qualifies without test removal, isolation changes or larger timeouts.
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

Evidence: normal gate43530844245 UI9 job132249289557 failed the deletion test with a detached-widget parent error in Textual mouse routing;150 other tests passed (/tmp/tldw-ci-eighth-ui9-failure.log). The exact test passed locally6.88s; delayed-removal and queued-refresh probes also passed, so the exact framework exception remains unreproduced locally. A controlled1s delay in the real delete-owned Trash read, with an observation-only assertion at the existing mouse call, proved the original Add click raced that pending publication (/tmp/tldw-ci-notes-delete-trash-red.log). The corrected exact case passed7.30s and all3 cases in the affected module passed17.98s under the same delayed read (/tmp/tldw-ci-notes-delete-{trash-green,module-green}.log). A30s delayed read fails the unchanged10s readiness wait (/tmp/tldw-ci-notes-delete-negative.log). Independent review found no actionable blockers and verified source ownership, real input/assertion retention, failure propagation and unchanged cleanup. Ruff, formatting, whitespace and backlog ID/readability checks pass; AST comparison retains every original assertion and test name. Live qualification passed on 2026-11-14: normal pull_request run43539534151 at head75d5127ab71496349fc0dcf3e7b94e1b839b85ca, UI9 job132278677953 on ghq-test-129f6d4c, passed all151 tests in685.26s with two existing warnings. The previously failing deletion journey ran in8.15s. The job ran02:07:31–02:20:25Z (12m54s). Retained log: /tmp/tldw-ci-notes-delete-live.log. This completes loaded UI9 qualification; successful full-gate timing remains pending. Completion evidence is deliberately local until a later necessary push, avoiding a healthy CI restart.
Integration with dev531acb8c retains its existing _pointer_target_ready helper instead of a duplicate local mouse-target helper. The deletion journey still waits for its actual Trash publication, checks the current button is displayed, and asserts both real mouse clicks. The installed Textual compositor handles transient missing coordinate hits; click exceptions still propagate. All three Notes cases passed in the combined twelve-case rebase integration run (/tmp/tldw-ci-dev-rebase-ui.log,86.34s total); independent review found no remaining integration blockers. Earlier loaded qualification above refers to its recorded head; the new combined head will receive normal PR CI.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
