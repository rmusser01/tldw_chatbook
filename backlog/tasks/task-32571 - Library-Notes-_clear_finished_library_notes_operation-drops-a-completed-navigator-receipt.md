---
id: TASK-32571
title: >-
  Library Notes: _clear_finished_library_notes_operation drops a completed
  navigator receipt
status: To Do
assignee: []
created_date: '2026-09-14 22:44'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by group 4 (task-32536) and confirmed in its review round. When a notes operation finishes, _clear_finished_library_notes_operation clears the operation state and takes the navigator's completed receipt with it, so the outcome a user was meant to read is gone before they can read it. The receipt is the only place some operations report what they did.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A completed operation's navigator receipt survives the operation state being cleared
- [ ] #2 The receipt still goes away on the routes that are supposed to dismiss it (a newer operation, leaving the list)
- [ ] #3 Pinned on the real route with a test that fails when the receipt is cleared early
<!-- AC:END -->
