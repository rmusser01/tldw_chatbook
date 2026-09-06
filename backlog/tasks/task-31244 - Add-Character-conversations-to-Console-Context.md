---
id: TASK-31244
title: Add Character conversations to Console Context
status: To Do
assignee: []
created_date: '2026-09-04 02:08'
labels:
  - console
  - context
  - characters
  - ux
dependencies:
  - TASK-31243
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Renumbering provenance

Renumbered from TASK-31236 on 2026-09-04. The final pre-commit worktree sweep
found the older `Review set Dismiss gets an Undo receipt` task created at 01:50;
it keeps TASK-31236 under the older-arrival rule. This unshipped task moves with
all plan and dependency references.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make recent character conversations discoverable in the Console Context rail through a bounded, date-sorted, local-only Character section that preserves first-use comprehension and expert state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Character is always composed directly after Conversations and before Model, independent of avatar-image visibility.
- [ ] #2 At most four character headers render; the current character is included even with zero chats and the Unavailable group consumes one header slot when present.
- [ ] #3 Only one group is expanded; first use opens current or most recent, while explicit saved disclosure preference wins thereafter without responsive-state persistence.
- [ ] #4 Each ordinary nonempty group shows at most five recent chats and ends with the exact View all N in Roleplay action.
- [ ] #5 Global Keyword search returns at most eight local character-chat results and clearing or escaping restores browse disclosure, focus, and scroll.
- [ ] #6 Unavailable rows offer only valid Library recovery; empty state offers Open Roleplay; exact chat Enter uses the shared typed activation contract.
- [ ] #7 This PR renders no Continue search in Character chats control and makes no narrow-terminal claim before the Ctrl+K fallback exists.
- [ ] #8 Production CSS and tests cover 52x20, standard widths, keyboard, pointer, truncation, empty, failure, preference migration, and exact activation.
<!-- AC:END -->
