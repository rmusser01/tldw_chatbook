---
id: TASK-37800
title: >-
  Console: Enter right after a collapsed token that ends the draft arms its
  expand instead of sending
status: To Do
assignee: []
created_date: '2026-10-29 17:13'
labels:
  - console
  - composer
  - ux-review-2026-09-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33622.1 implemented the owner decision of 2026-10-27 (Enter expands a token only when the caret is on it; otherwise Enter sends) as: the caret at EITHER edge of a collapsed token is on it. The caret never sits inside a token, so some reading of 'on' was needed; counting the right edge was the implementer's reading, not the owner's words. Its consequence: whenever the draft ends in a token, Enter still does not send. Paste then Enter, question then paste then Enter, a /prompt insert then Enter, and a file insert then Enter each still take three Enters (confirm, splice open, send), which is the original P1 symptom in token-last order. Review of the TASK-33622.1 branch reproduced it: 'Summarize this log: ' + paste + Enter showed the expand confirm, and a second Enter spliced the paste open. One alternative is block-cursor semantics: only a token's left edge counts as on it, so paste-then-Enter sends while Left/Home still reach a token to expand it. This needs an owner ruling before any change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner has ruled which caret positions count as on a collapsed token, and the ruling is recorded in this task
- [ ] #2 If the ruling changes the behaviour: with the caret right after a collapsed token that ends the draft, a single Enter dispatches exactly one send carrying the full text, pinned by a Pilot test on the real ChatScreen that fails on the pre-change code
- [ ] #3 Docs/User_Guide/console/chat-basics.md 'Large pastes' states the ruled caret rule
<!-- AC:END -->
