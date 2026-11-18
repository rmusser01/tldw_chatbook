---
id: TASK-37802
title: >-
  Console: a collapsed chip or expand confirm runs straight into the draft text
  that follows
status: To Do
assignee: []
created_date: '2026-10-29 17:14'
labels:
  - console
  - composer
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33622.1 renamed the composer's collapsed tokens ('Pasted text | N characters | Expand', '<file label> | Expand') and their confirm ('Expand pasted text (N characters)? Enter: yes · Esc: no'). With text typed straight after a token, the display runs the two together, e.g. 'Expand pasted text (51 characters)? Enter: yes · Esc: noquestion: ', which reads as one word. The draft text itself is right; only the display needs a visual boundary (a styled gap, a separator cell or the token's own background) that never enters the sent text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A collapsed chip or confirm followed or preceded directly by draft text renders with a visible boundary between token and text
- [ ] #2 The sent draft and the caret's canonical offsets are unchanged
- [ ] #3 Click-to-token mapping still lands on the token and on the neighbouring text as before
<!-- AC:END -->
