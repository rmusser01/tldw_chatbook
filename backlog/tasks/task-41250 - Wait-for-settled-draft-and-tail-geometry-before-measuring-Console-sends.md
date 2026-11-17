---
id: TASK-41250
title: Wait for settled draft and tail geometry before measuring Console sends
status: Done
created_date: 2026-11-17 09:33
assignee:
- '@codex'
labels:
- ci
- testing
updated_date: 2026-11-17 14:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3537 UI shard 9 recorded an extra whole-transcript movement at the start of the send measurement. The mounted harness starts measuring after a fixed pause, although loading the measured draft can shrink the composer and leave a following Textual tail adjustment outstanding. Establish the intended idle geometry before measuring the send, preserving all send-time limits and assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recording begins only after the measured draft is one row tall and the final transcript rule reaches the current scrolling viewport bottom.
- [x] #2 A controlled setup shrink demonstrates the old measurement can count setup layout; the repaired harness waits for that layout and retains every reflow after Enter.
- [x] #3 The stock and acknowledged mounted tests and adjacent composer controls pass with unchanged assertion and timeout budgets.
- [x] #4 The repaired current-head UI shard passes on the self-hosted CI pool.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test-harness precondition repair, no production behavior or runtime boundary change.
1. Trace the setup shrink and subsequent Textual tail adjustment with an observation-only diagnostic; retain a controlled failing case.
2. Wait on actual draft and tail geometry before recording, reusing the existing bounded wait helper and preserving all measured-send assertions and deadlines.
3. Verify the controlled interleaving, run the focused file and request independent review.
4. Push only the reviewed repair, qualify the normal PR UI shard and record live evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The shared mounted-send driver now waits for the measured draft to be one row tall and the transcript end rule to reach the current scrolling viewport bottom before recording. This is an 11-line setup precondition; production code, all post-Enter observations/assertions, the two-reflow limit, same-frame echo/placeholder check, provider deadline and CI timeout are unchanged. No new ADR is required: test-harness repair only.

Evidence: PR #3537 run 43946198034 job 133582861408 failed with whole-transcript moves [(0,9),(2,8),(9,8)] and 130 other cases passed. Eight unmodified focused runs passed locally (including 3 with CI pytest9.2.0/pytest-asyncio1.5.0/Rich15.0.0; Textual8.2.8 already matched). A temporary diagnostic resumes the existing third-send driver exactly after its real composer 2-to-1 setup reflow: before fix it reproduced [(0,9),(2,8),(7,8)] with all measured composer heights 1; after fix the same control passed in 31.92 s. Logs: /tmp/tldw-ci-3537-ui9-shrink-edge-{before,after}.log; observation/control plugin /tmp/tldw_ci_reflow_probe.py.

The full targeted file passed 8 tests in 71.97 s (/tmp/tldw-ci-3537-reflow-focused.log). Negative control disabling only the production draft-height clamp still failed with composer 1-to-2-to-1 and 4 whole-transcript moves (/tmp/tldw-ci-3537-ui9-clamp-negative-control.log). Independent requesting-code-review found no blockers. Ruff differential verified the same 10 pre-existing findings and the same 2 formatter hunks on base and branch, with none added; whitespace, backlog ID/file and ADR checks pass. The lesson records why scroll_y==max_scroll_y is insufficient during a resize.

Live qualification: PR #3537 head e17154459080659ef827fbcf4ede4fa2222907e3, run 44057405523, UI shard 9 job 134050863390 passed all 131 cases in 668.32 s; job window 2026-11-17 11:53:10–12:05:44Z (12m34s), runner ghq-test-8f38fcac. Log /tmp/tldw-ci-3537-ui9-e171-live.log. The complete gate later failed only in unrelated UI shard 15; this records the repaired lane qualification, not a successful full baseline.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
