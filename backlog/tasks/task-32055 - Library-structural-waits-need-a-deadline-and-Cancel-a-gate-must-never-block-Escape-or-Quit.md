---
id: TASK-32055
title: >-
  Library structural waits need a deadline and Cancel; a gate must never block
  Escape or Quit
status: To Do
assignee: []
created_date: '2026-09-08 18:23'
labels:
  - library
  - file-notes
  - notes
  - ux
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The File Notes folder change sat on 'Folder Files · No folder selected · Changing folder…' indefinitely and Escape, the '‹ Library / Notes' cue, a palette deep link and Ctrl+Q were all swallowed (observed once, in a session already wedged by the note-load hang; the second assessor linked the same folder fine). Note loads, skill imports and exports share the same shape: a wait with no progress, no timeout and no exit. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Any structural wait longer than about 3 s shows a 'Still working…' line with a Cancel action
- [ ] #2 Escape, the back cue and Ctrl+Q keep working while a wait is in progress; the gate vetoes the write, not the exit
- [ ] #3 A folder change that fails or times out reports why and leaves the previously linked folder intact
- [ ] #4 Covered by tests that simulate a never-resolving service call for folder change and note load
<!-- AC:END -->
