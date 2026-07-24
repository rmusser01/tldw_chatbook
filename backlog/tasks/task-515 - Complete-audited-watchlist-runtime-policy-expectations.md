---
id: TASK-515
title: Complete audited watchlist runtime-policy expectations
status: Done
assignee: []
created_date: '2026-07-24 18:27'
updated_date: '2026-07-24 18:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the full audited runtime-policy registry guard aligned with intentionally added watchlist preview, import/export, item-read, and run-cancel actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The audited watchlist action-id expectation contains every registry action and no extras
- [x] #2 Both local and server expectations include preview, import, export, item list/detail, and run cancel actions
- [x] #3 Runtime policy registry production definitions remain unchanged
- [x] #4 The focused guard and full RuntimePolicy tests pass
- [x] #5 Task documentation records the merge-base failure, originating registry changes, ADR decision, and verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the exact mismatch on both feature branch and merge base and compute the 12-action watchlist difference.
2. Verify each extra action comes from intentional watchlist registry resources, then add the exact local/server IDs to the audited expectation.
3. Run the focused guard and full RuntimePolicy suite.
4. Run Ruff format/check and git diff --check; independently review before completion.

ADR required: no
ADR path: N/A
Reason: This updates a stale test oracle to match already-defined runtime-policy rows; it does not change policy ownership, capabilities, enforcement, or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the exact test-oracle update for intentionally registered watchlist actions. Added the 12 missing local/server IDs for preview, import, export, item list/detail, and run cancel; the post-change watchlist set has no actual-minus-expected or expected-minus-actual IDs. Production runtime-policy definitions were not changed.

Root cause and history: the exact guard was RED on both the feature branch and the supplied merge-base comparison because its watchlist expectation lagged intentional registry additions. `85dcabc46` introduced preview/import/export and run-cancel policy rows; `08a159bf1` introduced item list/detail rows.

ADR required: no. ADR path: N/A. This is an exact test-oracle correction for existing policy definitions and changes no enforcement or architecture.

Verification: exact audited-registry guard 1 passed; full `Tests/RuntimePolicy` 248 passed with one existing Requests dependency warning. Ruff check passed, Ruff format check reported the test already formatted, and `git diff --check` passed.
<!-- SECTION:NOTES:END -->
