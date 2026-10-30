---
id: TASK-37871
title: 'Console: a question typed in front of a paste still takes three Enters to send'
status: To Do
assignee: []
created_date: '2026-10-30 01:36'
labels:
  - console
  - composer
  - ux-review-2026-09-29
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under the TASK-37800 ruling (owner, 2026-10-29) a caret right before a collapsed token counts as on it. A user who pastes first, presses Home and types a question in front leaves the caret right before the token, so Enter asks 'Expand pasted text (N characters)?' instead of sending, and the draft takes three Enters (ask, expand, send). A review probe on the TASK-33622.1 branch head showed 0 sends and 'Q: Expand pasted text (199 characters)? Enter: yes · Esc: no'. This follows the ruling's words, but it is the original P1 symptom with the question typed in front; every flow the ruling names ends with the caret after the token. No test pins this flow either way. Needs an owner ruling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner has ruled whether Enter sends or asks to expand when the caret sits right before a token because text was just typed in front of it, and the ruling is recorded in this task
- [ ] #2 A Pilot test on the real ChatScreen pins the ruled outcome of paste, Home, type a question, Enter (one send with the full text, or the named expand question)
- [ ] #3 If the ruling changes behaviour, Docs/User_Guide/console/chat-basics.md 'Large pastes' states it
<!-- AC:END -->
