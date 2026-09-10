---
id: TASK-32229
title: >-
  Library file dialogs have no path field; Ctrl+A in the file-name box is
  move-to-start
status: To Do
assignee: []
created_date: '2026-09-10 14:56'
labels:
  - library
  - export
  - import
  - ux
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The export destination picker (and Import Browse…, Folder files) makes a terminal user click through a tree from $HOME; the only way to type a path is the File name box. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 28.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A focused path Input above the tree accepts a pasted absolute path (with ~) and jumps the tree to it
<!-- AC:END -->
