---
id: TASK-32828
title: Redesign the Conversation Inspector modal for context and costs
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 01:05'
updated_date: '2026-09-19 03:56'
labels: []
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-09-18-console-conversation-review-and-attention-design.md
  - backlog/decisions/171-console-conversation-review-and-attention.md
  - Docs/superpowers/plans/2026-09-18-conversation-inspector-modal.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make current and next-send context and conversation usage understandable and inspectable in the existing Conversation Inspector modal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The modal has Context, Usage & cost and Exchange history views; cost and context entry points open their relevant view and identify the conversation.
- [x] #2 Context offers a readable section-list/detail flow without nested Next Send tabs; current content and next-send preview, freshness and token estimates are explicit.
- [x] #3 Usage & cost shows totals and scannable per-turn usage with per-call detail; estimated, reported, partial and unavailable amounts remain distinguishable.
- [x] #4 Copy, Save, Refresh, capture and disclosure controls are contextual and preserve existing privacy, persistence, redaction and explicit Next Send body-view boundaries.
- [x] #5 The modal supports keyboard navigation, visible Close and narrow list-to-detail navigation with Back; large histories load detail lazily.
- [ ] #6 Targeted modal and integration tests and rendered checks verify supported sizes; no Inspector sidebar layout or behavior is redesigned.
- [x] #7 Opening Usage & cost does not prepare hidden Next Send content; target or disclosure changes invalidate all relevant content and exports across views.
- [x] #8 Historical captured project instructions remain inspectable under ADR-069/097 while live automatic bodies stay confined to explicit preview; Safe/Full controls remain reachable from usage call detail.
- [x] #9 Totals label retained-message estimates and their actual coverage, preserve unavailable totals for unpriced rows, and never double-count captured calls already included in turn usage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-09-18-conversation-inspector-modal.md.
ADR required: yes
ADR path: backlog/decisions/171-console-conversation-review-and-attention.md
Reason: direct implementation of the approved durable-read and long-lived Console UX contracts; no new ADR required.
Preserve the Inspector sidebar and existing ownership/privacy boundaries; targeted tests only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced nested Inspector modal expansion with Context, Usage & cost and Exchange history section/detail readers, lazy loading, narrow Back/focus handling, captured target authority and cross-view disclosure/export invalidation. Both entry routes estimate the prepared request using the captured session. The Inspector sidebar was not redesigned.

ADR: backlog/decisions/171-console-conversation-review-and-attention.md. Updated the Console user guides and production-style tests; rebuilt CSS from token-backed source modules. Fresh review findings were reproduced and fixed (stuck receipts, cost-entry estimates, narrow reader focus).

Verification: final focused qualification 131 passed; broad feature regression 698 passed with nine independently reproduced baseline failures and three separately proven baseline exclusions. Full suite not run. See Docs/superpowers/reports/2026-09-18-console-conversation-ux-verification.md for scope and limitations.

Remains In Progress: required native terminal qualification is unavailable because computer-use access to iTerm was denied and Windows Terminal is not available. Headless production-style checks are not claimed as native evidence.

Final flat-list owner/recompose qualification: 58 passed. Final focused qualification: 131 passed. New files pass Ruff; modified legacy code has no changed-line diagnostics. See the linked verification report for the exact baseline exclusions and native-check limitation.
<!-- SECTION:NOTES:END -->
