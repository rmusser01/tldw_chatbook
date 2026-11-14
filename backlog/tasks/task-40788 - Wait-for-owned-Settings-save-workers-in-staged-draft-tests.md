---
id: TASK-40788
title: Wait for owned Settings save workers in staged draft tests
status: Done
created_date: 2026-11-14 00:33
assignee:
- '@codex'
updated_date: 2026-11-14 02:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The required UI lane can fail after successful save assertions because an app-wide wait includes a legitimately superseded prompt-summary read. Observe the actual save operation while retaining the staged draft, persistence and lifecycle checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The staged draft matrix waits for each actual save worker and still verifies later edits, queued events, failures, repeated saves and Revert.
- [x] #2 Legitimate background prompt-summary replacement does not fail the save test, while a failed or cancelled save worker still fails it.
- [x] #3 Focused checks and independent review pass with unchanged test selection, private-profile isolation and deadlines; the affected normal PR lane qualifies.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Test observation ownership correction within existing CI/runtime boundaries.
1. Trace the failed worker identity and reproduce a real prompt-summary replacement with a delayed real read.
2. Retain each save's actual Textual worker from the existing held-writer fixture and await it rather than unrelated app-wide workers; keep all assertions and deadlines.
3. Run the affected staged matrix, delayed-read and failing-save controls, lint and independent review; qualify the normal self-hosted lane before marking Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The existing held save_sections fixture now retains each actual Textual worker via get_current_worker(). Both saves await their own worker, including the UI acknowledgement, and the second uses the existing 15-second writer-admission helper. This removes the accidental dependency on replaceable prompt-summary reads. Every existing draft/persistence/focus/selection/Revert/failure assertion, test parameter, private-profile boundary and timeout remains intact. No production code or CI topology changed. Updated the testing lesson with the observed incident; ADR-359 applies and no new ADR is required.

Evidence: required run43530844245 UI18 job132249289569 failed at the second app-wide wait with the prompt-summary revision2 worker cancelled; 123 other tests passed (log /tmp/tldw-ci-eighth-ui18-failure.log). The exact test passed unmodified locally in11.52s; delaying the real summary read by1s reproduced the identical cancellation at the original line230 (/tmp/tldw-ci-settings-save-controlled-red.log). With the corrected wait, that controlled case passed in11.99s (/tmp/tldw-ci-settings-save-controlled-green.log). A negative control calling real Worker.cancel on the owned save fails at the corrected wait with WorkerCancelled (/tmp/tldw-ci-settings-save-negative.log). Independent review verified thread context, admission ordering, full worker/acknowledgement lifetime, error propagation and shutdown; no blockers. Ruff and format checks, whitespace and backlog ID/readability validators pass. The complete affected module passed all19 cases in419.01s with the delayed real read, including all18 staged-save parameters and the normalized-console control (/tmp/tldw-ci-settings-save-matrix.log). This is targeted local evidence, not full-CI timing. Live qualification passed on 2026-11-14: normal pull_request run43539534151 at head75d5127ab71496349fc0dcf3e7b94e1b839b85ca, UI18 job132278677919 on ghq-test-97d1dbe0, passed all124 tests in615.84s. The previously failing later-console-behavior case ran in13.52s. The whole job ran01:40:29–01:52:16Z (11m47s). Retained log: /tmp/tldw-ci-settings-save-live.log. This completes the affected-lane qualification; successful full-gate timing remains pending. Completion evidence is deliberately local until a later necessary push, to avoid restarting healthy CI.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
