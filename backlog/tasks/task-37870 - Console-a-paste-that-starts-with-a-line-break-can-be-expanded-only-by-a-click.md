---
id: TASK-37870
title: 'Console: a paste that starts with a line break can be expanded only by a click'
status: To Do
assignee: []
created_date: '2026-10-30 01:35'
labels:
  - console
  - composer
  - ux-review-2026-09-29
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-37800 (owner ruling 2026-10-29, paste then Enter sends) made Enter expand a collapsed token only with the caret right before it. A paste whose text starts with a line break draws that break outside its chip, so the caret can never sit right before the chip, and the only way to expand such a paste is a mouse click. Before TASK-37800, Enter could reach it (dev stepped the first token; the first TASK-33622.1 reading also counted the right edge). In a keyboard-first TUI a token only the mouse can expand is a gap. The User Guide states the limit today. Options for the owner: let the left edge count across a leading break (the caret before the break), or add a dedicated expand key.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner has chosen how a keyboard user expands a paste that starts with a line break, and the choice is recorded in this task
- [ ] #2 With the composer focused and no mouse, a paste that starts with a line break can be expanded to its full text, pinned by a Pilot test on the real ChatScreen that fails on the pre-change code
- [ ] #3 Paste then Enter still sends on one Enter, and the TASK-37800 tests stay green
- [ ] #4 Docs/User_Guide/console/chat-basics.md 'Large pastes' no longer says such a paste must be clicked
<!-- AC:END -->
