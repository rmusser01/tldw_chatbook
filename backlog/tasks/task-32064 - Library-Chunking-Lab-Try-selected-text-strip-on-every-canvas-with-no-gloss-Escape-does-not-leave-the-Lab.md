---
id: TASK-32064
title: >-
  Library: 'Chunking Lab / Try selected text' strip on every canvas with no
  gloss; Escape does not leave the Lab
status: Done
assignee: []
created_date: '2026-09-08 18:25'
updated_date: '2026-09-08 20:03'
labels:
  - library
  - chunking-lab
  - ux
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The strip is the first interactive row under the header on every Library canvas; a first-time user clicked it and landed in a full-screen A/B tool ('Run B / Run both / Pin A / Save A / Save B') with no explanation, and Escape does not return (only its own Back). Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 15.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The strip is either moved under Details ▸ Actions (or the Media reader's More strip) or carries a one-line gloss
- [x] #2 Escape from the Chunking Lab returns to the Library canvas it was opened from
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests: the strip sits above every canvas; the Lab has no escape binding.
2. Move the pair into the existing Details > Actions group with a one-line gloss.
3. Add an escape binding to ChunkingLabScreen routed at its own lab-back action, text fields excluded.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The 'Chunking Lab / Try selected text' pair was composed unconditionally right under the header, making it the first interactive row on EVERY Library canvas. It moved into the Details rail's EXISTING Actions group (_workspaces_detail_rows already builds one -- a second group would have collided on its id) above the line 'Chunking Lab - compare how text is split for search'. ChunkingLabScreen gained an escape binding routed at its own lab-back action; a focused Input/TextArea keeps Escape for itself, so this does not touch the text-box Escape semantics another branch owns. Verified live: the strip is gone from the canvas (caps/10), the pair and its gloss render under Details > Actions (caps/11), and Escape returns to the Library canvas that opened the Lab (caps/12). Files: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/UI/Screens/chunking_lab_screen.py, Tests/UI/test_library_crit8_polish_shell.py, Docs/User_Guide/library.md.
<!-- SECTION:NOTES:END -->
