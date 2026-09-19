---
id: TASK-32827
title: Align Console conversation rows with workspace rows
status: To Do
assignee: []
created_date: '2026-09-19 01:05'
updated_date: '2026-09-19 01:18'
labels: []
dependencies:
  - TASK-32826
documentation:
  - >-
    Docs/superpowers/specs/2026-09-18-console-conversation-review-and-attention-design.md
  - backlog/decisions/171-console-conversation-review-and-attention.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Console Conversations list and workspace conversation rows consistent, compact, readable and easy to operate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Conversations and workspace chat rows share compact density, alignment, selection, focus and right-edge action placement using design tokens.
- [ ] #2 Titles truncate by terminal cell width while full titles remain available through keyboard and pointer interaction; meaningful activity has concise text.
- [ ] #3 Identity, ordering, paging, draft state, subagent progress and workspace ownership survive refreshes and row interactions.
- [ ] #4 Both row surfaces use the attention action contract from TASK-32826 and provide equivalent keyboard menu access.
- [ ] #5 Targeted tests and mounted checks cover narrow layouts, long titles, ASCII mode, scroll boundaries and pointer target stability; the Inspector sidebar remains unchanged.
- [ ] #6 Collapsed workspace and capped-list indicators retain attention using the same semantic precedence as their conversation rows; focus explains state and menu action.
<!-- AC:END -->
