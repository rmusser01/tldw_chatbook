---
id: TASK-32828
title: Redesign the Conversation Inspector modal for context and costs
status: To Do
assignee: []
created_date: '2026-09-19 01:05'
labels: []
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-09-18-console-conversation-review-and-attention-design.md
  - backlog/decisions/171-console-conversation-review-and-attention.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make current and next-send context and conversation usage understandable and inspectable in the existing Conversation Inspector modal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The modal has Context, Usage & cost and Exchange history views; cost and context entry points open their relevant view and identify the conversation.
- [ ] #2 Context offers a readable section-list/detail flow without nested Next Send tabs; current content and next-send preview, freshness and token estimates are explicit.
- [ ] #3 Usage & cost shows totals and scannable per-turn usage with per-call detail; estimated, reported, partial and unavailable amounts remain distinguishable.
- [ ] #4 Copy, Save, Refresh, capture and disclosure controls are contextual and preserve existing privacy, persistence, redaction and explicit Next Send body-view boundaries.
- [ ] #5 The modal supports keyboard navigation, visible Close and narrow list-to-detail navigation with Back; large histories load detail lazily.
- [ ] #6 Targeted modal and integration tests and rendered checks verify supported sizes; no Inspector sidebar layout or behavior is redesigned.
<!-- AC:END -->
