---
id: TASK-15671
title: 'The .gitignore force-add carve-out does not extend to a survivor window'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - console
  - change-review
  - security
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Change tracking force-adds paths an agent touched even when .gitignore would hide them, so an agent writing to an ignored path (`.env` is the canonical case) still shows up for review. PR 3a-1 Task 6c's survivor window passes no `touched_paths` when it closes, so that carve-out does not apply to changes a sub-agent makes AFTER its turn: an ignored path written by a survivor surfaces inside its own turn's window and not after it. Closing this means tracking which child runs a window covers so their persisted steps can be read; the gap is named in the code at `_close_post_turn_change_window` rather than half-built.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A survivor writing to a .gitignore'd path has that write surfaced in its own survivor window
- [ ] #2 The turn window's existing force-add behaviour is unchanged
- [ ] #3 A test writes an ignored path from a post-turn child and fails when the carve-out is absent
- [ ] #4 The gap comment at _close_post_turn_change_window is removed rather than reworded
<!-- AC:END -->
