---
id: TASK-32219
title: >-
  library.md still documents the Chunking Lab strip under every canvas header;
  Details ▸ Actions sits below the rail fold
status: To Do
assignee: []
created_date: '2026-09-10 14:54'
labels:
  - library
  - docs
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Layout tour says the strip sits 'directly under the header, on every Library canvas'; task-32064 moved it to Details ▸ Actions and the same page's control table says so, so the page contradicts itself. At 52 rows the Actions group is below the rail's fold (six wheel notches to reach it) with no scroll cue. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 16.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 library.md's Layout tour matches the shipped placement
- [ ] #2 The rail shows a scroll cue when Details ▸ Actions is below the fold, or Actions moves above the fold
<!-- AC:END -->
