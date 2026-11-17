---
id: TASK-41251
title: Wait through the custom endpoint status remount in Settings acceptance
status: In Progress
created_date: 2026-11-17 14:28
assignee:
- '@codex'
labels:
- ci
- testing
updated_date: 2026-11-17 14:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3537 UI shard 15 failed because the acceptance poll queried the Custom endpoints status after Rename removed the old section children and before Textual remounted them. Preserve the visible success checks while allowing that temporary missing-widget observation within the existing deadline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The click completion poll survives a temporarily absent status widget and still requires the original visible success text.
- [x] #2 A controlled original recomposition reproduces the failure before the repair and passes afterward; a permanently missing status still fails.
- [x] #3 The affected Settings acceptance file passes without changing production behavior, retry counts, deadlines or secret-redaction checks.
- [ ] #4 The repaired UI shard passes on the self-hosted CI pool.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: a test polling precondition repair, with no production or runtime boundary change.
1. Inspect every caller and trace the actual Textual remove/remount boundary; retain controlled failure evidence.
2. Treat only NoMatches from the completion predicate as not ready inside the existing bounded poll.
3. Run the controlled gap, missing-status negative control and affected file; obtain independent review.
4. Push the reviewed repair, qualify the normal PR shard and record live evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The shared Custom endpoints click-completion poll now treats only NoMatches from its predicate as not ready during Textual recomposition. All four callers retain their original visible form/status predicates, real clicks, three attempts, four-second observation windows, final strict assertion and secret-redaction checks. Production code and isolation are unchanged; no ADR required.

Evidence: PR #3537 run 44057405523 job 134050868128 failed on the missing status immediately after Rename, with 117 other shard cases passing. The original case passed locally in 18.65 s. A temporary plugin held the original Textual mount_all after actual child removal: the original poll reproduced the exact NoMatches (15.39 s overall); the repaired poll passed (14.59 s). Withholding only the status during original remount still failed at the unchanged final done() check (17.09 s). Logs: /tmp/tldw-ci-3537-ui15-gap-{before,after}.log and /tmp/tldw-ci-3537-ui15-missing-status-negative.log; plugin /tmp/tldw_ci_ui15_recompose_probe.py. This demonstrates the legitimate remount gap, not an instrumented trace of the historical CI scheduling.

All 57 cases in the affected file passed in 153.52 s (/tmp/tldw-ci-3537-ui15-focused.log). Ruff has zero findings; formatter differences are identical to the six pre-existing hunks. Whitespace, backlog file/ID and ADR validators pass. Independent requesting-code-review found no blockers and separately verified temporary absence recovers, persistent absence/false predicates fail, and WrongType, malformed queries, other exceptions and click-time NoMatches propagate. Added this recurrence to the existing raising-predicate testing lesson. Live current-head shard qualification remains pending, so the task remains In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
