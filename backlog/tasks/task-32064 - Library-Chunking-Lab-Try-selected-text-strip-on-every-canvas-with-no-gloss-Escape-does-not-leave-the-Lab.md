---
id: TASK-32064
title: >-
  Library: 'Chunking Lab / Try selected text' strip on every canvas with no
  gloss; Escape does not leave the Lab
status: To Do
assignee: []
created_date: '2026-09-08 18:25'
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
- [ ] #1 The strip is either moved under Details ▸ Actions (or the Media reader's More strip) or carries a one-line gloss
- [ ] #2 Escape from the Chunking Lab returns to the Library canvas it was opened from
<!-- AC:END -->
