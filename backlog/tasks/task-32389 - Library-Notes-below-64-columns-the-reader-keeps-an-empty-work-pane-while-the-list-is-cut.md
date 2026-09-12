---
id: TASK-32389
title: 'Library Notes: below 64 columns the reader keeps an empty work pane while the list is cut'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - notes
  - layout
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32217 settled that an empty work pane hands its columns to the list. `_sync_library_notes_reader_layout_from_shell` (`library_screen.py:6248-6258`) passes `priority="items"` unconditionally, so at 60 columns the notes reader lays out 32/18 with nothing in the work pane -- 18 columns held by an empty stage while the list is cut to 32. The reviewer of task-32360 ruled this a separate defect from that task's narrow-stage return, and it bypasses an already-shipped rule rather than raising an open design question. See also task-32304 (the same empty-work-pane symptom below 64 columns on Collections, Conversations and Skills) and task-32065, which introduced the `list_first_when_empty` rule both want -- whoever takes one should take the other rather than re-deriving it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 60 columns with nothing open, the notes list takes the columns the empty work pane was holding, as task-32217's rule requires
- [ ] #2 Opening a note restores the reading split at that width
- [ ] #3 The 60-column case is pinned in the notes layout test file
<!-- AC:END -->
