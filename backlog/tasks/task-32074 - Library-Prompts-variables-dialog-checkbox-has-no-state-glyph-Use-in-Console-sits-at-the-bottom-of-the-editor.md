---
id: TASK-32074
title: >-
  Library Prompts: variables-dialog checkbox has no state glyph; 'Use in
  Console' sits at the bottom of the editor
status: To Do
assignee: []
created_date: '2026-09-08 18:26'
labels:
  - library
  - prompts
  - ux
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The prompt-variables dialog renders its checkbox as an empty box with no checked/unchecked glyph, and 'Use in Console' is at row 49 of 52 in the editor while Media places the equivalent action in the reader header. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 25.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The variables checkbox shows a text-labelled or glyph state
- [ ] #2 'Use in Console' is placed consistently with the Media reader's 'Use in Console'
<!-- AC:END -->
