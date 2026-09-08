---
id: TASK-32053
title: >-
  Library Search/RAG: evidence cards unreachable by keyboard; `o`/`u` inert; Tab
  lands on the only source toggle
status: To Do
assignee: []
created_date: '2026-09-08 18:22'
labels:
  - library
  - search-rag
  - ux
  - keyboard
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a search, 8–14 Tabs never focus an evidence card, `o` (open) and `u` (use in Console) do nothing while the footer advertises them, and Tab from the query box lands on the sole enabled source toggle where Enter empties the results while the footer still says 'enter select evidence'. The Run button shows no focus either. The documented keyboard flow for evidence does not exist. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each evidence card can be reached by keyboard and shows a visible cursor (shape, not colour alone)
- [ ] #2 Enter selects the focused card, `o` opens it and `u` stages it, matching the footer
- [ ] #3 The footer names the focused control's Enter action (for example 'enter toggle Notes' on a source toggle, 'enter run search' in the query box)
- [ ] #4 The Run button has a visible focus state
<!-- AC:END -->
