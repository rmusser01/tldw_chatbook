---
id: TASK-28002
title: Library media viewer - Escape cannot leave the viewer after in-item search
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live-reproduced in the 2026-09-01 dual-agent Library media UX critique: open a media item, run an in-item content search (query + Enter pins the Match-N-of-M header), then Escape three times - nothing happens, while the footer still advertises esc back to list. From a freshly opened viewer without search interaction, one Escape returns to the list correctly. Hypothesis only (trace before fixing, and grep for pinning tests on the viewer Escape chain first): the search sub-state gating or input focus consumes Escape without stepping back.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After using in-item search, Escape steps back per the advertised chain and subsequently returns to the media list
- [ ] #2 Footer key hints match actual Escape behavior in every viewer sub-state
- [ ] #3 A regression test pins the search-then-Escape path
<!-- AC:END -->
