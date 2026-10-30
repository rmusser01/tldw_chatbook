---
id: TASK-37872
title: >-
  Console: a pending expand question owns both token edges while the caret rule
  counts only the left edge
status: To Do
assignee: []
created_date: '2026-10-30 01:37'
labels:
  - console
  - composer
  - ux-review-2026-09-29
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-37800 made only a collapsed token's left edge count as the caret being on it, but a pending expand question still owns Enter at either edge, because a click leaves the caret right after the token and the next Enter must answer it. The two rules differ, so the User Guide has to explain the exception, and a keyboard move to the right edge keeps a question the user may think they left. If a click put the caret on the token's left edge, a pending question could own only that edge too, and one block-cursor rule would hold throughout. Optional cleanup raised by the TASK-37800 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner or a reviewer has decided whether a click should land the caret on the clicked token's left edge, recorded in this task
- [ ] #2 If adopted: a pending expand question owns Enter only at its token's left edge, the Enter after a click still answers it, and moving the caret to the right edge answers no, each pinned by a Pilot test on the real ChatScreen
- [ ] #3 If adopted: Docs/User_Guide/console/chat-basics.md 'Large pastes' describes one caret rule with no edge exception
<!-- AC:END -->
