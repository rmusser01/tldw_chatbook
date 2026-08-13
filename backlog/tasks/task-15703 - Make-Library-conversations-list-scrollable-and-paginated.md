---
id: TASK-15703
title: Make Library conversations list scrollable and paginated
status: In Progress
assignee: []
created_date: '2026-08-12'
labels:
  - library
  - conversations
  - ux
priority: medium
dependencies: []
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
- [ ] #1 The Library conversation rows render in a vertically scrollable viewport, and every row on the current page is reachable by mouse and keyboard regardless of terminal height
- [ ] #2 The view shows at most 20 conversations per page, exposes Previous and Next controls plus the current page and result range, and allows every saved conversation to be reached
- [ ] #3 Submitting a conversation filter searches the complete saved-conversation collection, resets to page 1, and reports the filtered total
- [ ] #4 Paging and filtering keep the last successful page visible while loading, reject stale responses, and present a recoverable error without misreporting an empty library
- [ ] #5 Automated state, service-call, and Textual Pilot tests cover scrolling, first/middle/last pages, full-dataset filtering, empty results, and failures
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

Draft — implementation and live verification are complete, but task closure remains
blocked by two pre-existing repository-wide verification failures, so the acceptance
criteria and In Progress status intentionally remain unchanged.

- Implemented a 20-row, service-backed conversation page model with query/offset
  filtering, stale-response rejection, preserved last-successful content during
  loading or recoverable errors, a native vertically scrolling row viewport, and
  fixed Previous/Next controls with range and page status.
- Updated state, Library screen/canvas presentation, CSS source/generated bundle,
  focused state and Textual Pilot coverage, and
  `Docs/User_Guide/library/media-and-conversations.md`. No ADR was required because
  the existing conversation-service pagination contract and module boundaries were
  retained.
- Focused verification passed: Ruff; CSS regeneration with no semantic bundle diff;
  30 state/visibility tests; and 21 focused Library conversation UI tests. Isolated
  live TUI verification passed with 45 scratch-only conversations at 100x30 and
  170x48, covering row 20 scrolling, first/middle/final pages, oldest-row filtering,
  and clearing back to page 1. Evidence:
  `/tmp/tldw-task15703-live.ZWehQn/evidence/live-verification.txt`.
- Closure blockers: the broader Library gate reported 1 failure and 1118 passes due
  to the command-name drift guard (`generate-video`, `stream-video`), and the fresh
  corrected UI checks reported 1 failure and 1 pass due to the landing-footer copy
  guard. Both failures reproduce unchanged on pre-feature base `438739778`; project
  policy nevertheless requires green gates before marking the task Done.
