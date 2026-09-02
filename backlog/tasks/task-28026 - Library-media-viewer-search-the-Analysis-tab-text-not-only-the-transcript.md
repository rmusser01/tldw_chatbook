---
id: TASK-28026
title: 'Library media viewer - search the Analysis tab text, not only the transcript'
status: To Do
assignee: []
created_date: '2026-09-02 06:46'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the 2026-09-01 Library media UX critique (task-28011 split): in-item content search (find_content_matches over viewer.content) searches the transcript ONLY. When the user is reading an analysis in the Reader's Analysis tab, the search box does not find text in the analysis. For the sequential-review flow (reading analyses), searching within the analysis is as useful as searching the transcript. Extend the search corpus to include the active tab's text (analysis when the Analysis tab is showing), with matching/highlighting/scroll targeting that tab. task-28011 delivered the keyboard match-navigation (Enter advances, wrapping, footer-advertised) for the content corpus; this task is the remaining corpus-breadth half.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Searching while the Analysis tab is active finds and highlights matches in the analysis text
- [ ] #2 Match count, prev/next, and Enter-advance operate over whichever tab's text is being searched
- [ ] #3 Switching tabs re-scopes the search corpus without stale highlights
<!-- AC:END -->
