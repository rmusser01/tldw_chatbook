---
id: TASK-32217
title: >-
  Library canvases: no density rule — landing, Study staging, Prompts, Skills
  and File Notes are ~90% empty while the reader starves its controls
status: To Do
assignee: []
created_date: '2026-09-10 14:54'
labels:
  - library
  - layout
  - design
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The landing hub is ten lines in a 190x44 canvas; Study staging six lines; Prompts and Skills give a 48-cell list a 145-cell 'select something' pane; File Notes two lines on the full screen. Meanwhile the Media reader's Analysis actions sit ~28 blank rows below the text they act on and the note editor's Body gets an 11-row box in a 45-row pane. Media already proved 'a pane with nothing open gives its columns to its sibling'; no sibling canvas got that rule. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 14.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The empty-pane widening rule is applied to Prompts, Skills, Collections and Conversations
- [ ] #2 The note editor Body and the reader content box grow to fill their pane; Analysis actions sit under the content
- [ ] #3 The landing either earns its space (recent items, last import, pending review sets) or is narrowed to a readable measure
<!-- AC:END -->
