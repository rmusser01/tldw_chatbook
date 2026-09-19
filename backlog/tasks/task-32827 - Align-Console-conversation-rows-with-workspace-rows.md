---
id: TASK-32827
title: Align Console conversation rows with workspace rows
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 01:05'
updated_date: '2026-09-19 03:56'
labels: []
dependencies:
  - TASK-32826
documentation:
  - >-
    Docs/superpowers/specs/2026-09-18-console-conversation-review-and-attention-design.md
  - backlog/decisions/171-console-conversation-review-and-attention.md
  - Docs/superpowers/plans/2026-09-18-console-conversation-row-parity.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Console Conversations list and workspace conversation rows consistent, compact, readable and easy to operate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversations and workspace chat rows share compact density, alignment, selection, focus and right-edge action placement using design tokens.
- [x] #2 Titles truncate by terminal cell width while full titles remain available through keyboard and pointer interaction; meaningful activity has concise text.
- [x] #3 Identity, ordering, paging, draft state, subagent progress and workspace ownership survive refreshes and row interactions.
- [x] #4 Both row surfaces use the attention action contract from TASK-32826 and provide equivalent keyboard menu access.
- [ ] #5 Targeted tests and mounted checks cover narrow layouts, long titles, ASCII mode, scroll boundaries and pointer target stability; the Inspector sidebar remains unchanged.
- [x] #6 Collapsed workspace and capped-list indicators retain attention using the same semantic precedence as their conversation rows; focus explains state and menu action.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-09-18-console-conversation-row-parity.md.
ADR required: yes
ADR path: backlog/decisions/171-console-conversation-review-and-attention.md
Reason: direct implementation of the approved durable-read and long-lived Console UX contracts; no new ADR required.
Preserve the Inspector sidebar and existing ownership/privacy boundaries; targeted tests only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented compact title/action rows and matching workspace children with terminal-cell truncation, saved appearance, representative attention, ASCII fallback, meaningful hidden attention and keyboard menu access. Existing ownership, ordering, paging and subagent structure are preserved.

ADR: backlog/decisions/171-console-conversation-review-and-attention.md. Updated the Console user guides and production-style tests; rebuilt CSS from token-backed source modules. Fresh review findings were reproduced and fixed (stuck receipts, cost-entry estimates, narrow reader focus).

Verification: final focused qualification 131 passed; broad feature regression 698 passed with nine independently reproduced baseline failures and three separately proven baseline exclusions. Full suite not run. See Docs/superpowers/reports/2026-09-18-console-conversation-ux-verification.md for scope and limitations.

Remains In Progress: required native terminal qualification is unavailable because computer-use access to iTerm was denied and Windows Terminal is not available. Headless production-style checks are not claimed as native evidence.

Final flat-list owner/recompose qualification: 58 passed. Final focused qualification: 131 passed. New files pass Ruff; modified legacy code has no changed-line diagnostics. See the linked verification report for the exact baseline exclusions and native-check limitation.
<!-- SECTION:NOTES:END -->
