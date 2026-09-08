---
id: TASK-32070
title: >-
  Library footer: duplicate footer after a search at 235 wide; select hints
  appear only after a round trip; wrong hint while typing in the query box
status: To Do
assignee: []
created_date: '2026-09-08 18:26'
labels:
  - library
  - footer
  - ux
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a rail search at 235x52 a wrapped 'vidence | F6 …' line painted above the real footer; Space/s hints in Media select mode appeared only after an Escape round trip; while typing in the Search/RAG query box the footer reads 'enter select evidence' although Enter runs the search. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 21.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The footer never paints twice
- [ ] #2 Select-mode hints are present as soon as select mode is entered
- [ ] #3 The footer names the focused control's Enter action in the Search/RAG query box
<!-- AC:END -->
