---
id: TASK-31968
title: >-
  Library media: the Reader's reading-position restore does not land on a mode
  change
status: To Do
assignee: []
created_date: '2026-09-07 20:27'
labels:
  - library
  - media
  - reader
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by PR L (task-31954). On Read → Analysis → Read the reader progress restore is scheduled once now, but the position is not visibly restored — reproduced on the merge-base 801216bb0 too, so this predates the rider work. `_restore_library_media_loaded_progress` calls `scroll_to` while the rendered Markdown body is still parsing, so the scroll lands on a shorter document and is lost when the body grows. Two ineffective schedules became one ineffective schedule; the user still loses their place on every mode change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Switching Read → Analysis → Read returns the Reader to the same scroll offset (painted pin, not a state probe)
- [ ] #2 The restore waits for the rendered body's layout (or re-applies once after the parse settles) rather than racing it
- [ ] #3 No second scheduler is introduced; the one owner from task-31954 stays the only one
<!-- AC:END -->
