---
id: TASK-15703
title: Make Library conversations list scrollable and paginated
status: Done
assignee: []
created_date: '2026-08-12'
updated_date: '2026-08-13 16:22'
labels:
  - library
  - conversations
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users browse every saved conversation from Library without the result set
being clipped by terminal height or silently capped at the first fetched page.
Filtering must search the complete saved-conversation collection so older
conversations remain discoverable.

Design: `Docs/superpowers/specs/2026-08-12-library-conversations-pagination-design.md`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Library conversation rows render in a vertically scrollable viewport, and every row on the current page is reachable by mouse and keyboard regardless of terminal height
- [x] #2 The view shows at most 20 conversations per page, exposes Previous and Next controls plus the current page and result range, and allows every saved conversation to be reached
- [x] #3 Submitting a conversation filter searches the complete saved-conversation collection, resets to page 1, and reports the filtered total
- [x] #4 Paging and filtering keep the last successful page visible while loading, reject stale responses, and present a recoverable error without misreporting an empty library
- [x] #5 Automated state, service-call, and Textual Pilot tests cover scrolling, first/middle/last pages, full-dataset filtering, empty results, and failures
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A

Reason: the change consumes the existing paginated conversation-service contract and only changes bounded Library view state and presentation.

1. Extend the pure conversation canvas state with page/range/navigation/loading/error fields and remove its client-side cap/filter behavior.
2. Render the page rows inside Textual's native VerticalScroll and add fixed Previous/Next controls with focused Pilot coverage.
3. Add screen-owned page records and service-backed page/filter workers using query, limit=20, offset, and a stale-response generation guard.
4. Preserve the last successful page during loading/errors and isolate it from broad Library source snapshot refreshes.
5. Update the user guide, run focused and broader Library verification plus isolated live TUI checks, then record evidence before closing the task.

Detailed plan: `Docs/superpowers/plans/2026-08-12-library-conversations-pagination.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented a 20-row, service-backed conversation page model with query/offset
  filtering, stale-response rejection, preserved last-successful content during
  loading or recoverable errors, a native vertically scrolling row viewport, and
  fixed Previous/Next controls with range and page status.
- Updated state, Library screen/canvas presentation, CSS source/generated bundle,
  focused state and Textual Pilot coverage, plus the feature design, plan, and task
  documentation. The clean `dev` integration intentionally preserves deletion of
  the retired `Docs/User_Guide` tree rather than resurrecting its old Library page.
  No ADR was required because the existing conversation-service pagination contract
  and module boundaries were retained.
- Focused verification passed: Ruff; CSS regeneration with no semantic bundle diff;
  30 state/visibility tests; and 21 focused Library conversation UI tests. Isolated
  live TUI verification passed with 45 scratch-only conversations at 100x30 and
  170x48, covering row 20 scrolling, first/middle/final pages, oldest-row filtering,
  and clearing back to page 1. Evidence:
  `/tmp/tldw-task15703-live.ZWehQn/evidence/live-verification.txt`.
- Replayed the feature onto current `dev` without the old-base command/footer guard
  maintenance, which is not present on this branch. Removed five existing lint
  artifacts exposed by the clean-base check and hardened seven async UI
  assertions/polls to wait for mounted widgets instead of racing Textual
  recomposition.
- Fresh clean-base verification passed: 28 focused state/visibility tests, 20
  focused conversation UI tests, all 557 `Tests/Library` tests, all 268
  `Tests/UI/test_library_shell.py` tests, Ruff, and `git diff --check`. Final review
  found no Critical, Important, or Minor feature issues. No generalized lesson entry
  was added because the only test trap encountered is already documented in
  `lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
