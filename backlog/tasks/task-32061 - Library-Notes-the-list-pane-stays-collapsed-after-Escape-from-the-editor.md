---
id: TASK-32061
title: 'Library Notes: the list pane stays collapsed after Escape from the editor'
status: Done
assignee: []
created_date: '2026-09-08 18:24'
updated_date: '2026-09-08 20:03'
labels:
  - library
  - notes
  - ux
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Opening the first note in a wide session auto-collapses both the Library rail and the Notes list; Escape returns to 'Select a note to edit it here.' with no list until the '--->' grip is clicked. The guide says only Library navigation auto-closes. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 12.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Escape or Back from the editor restores the Notes list pane to the visibility it had before the editor opened
- [x] #2 The guide and the behaviour agree on which panes auto-close
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The reported collapse did not reproduce, on the harness or live at 235x52: only Library navigation auto-closes for a wide Notes work session (_library_notes_work_first_preferences sets library_open=False and nothing touches items_open), which is exactly what Docs/User_Guide/library/notes.md documents -- so AC#2 already held. Both halves of AC#1 are now pinned instead: Escape from the editor leaves the list visible, and Escape does NOT reopen a list the reader had collapsed. No production change. Files: Tests/UI/test_library_crit8_polish_shell.py, Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
