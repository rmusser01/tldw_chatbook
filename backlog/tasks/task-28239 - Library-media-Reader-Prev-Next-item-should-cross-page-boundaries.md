---
id: TASK-28239
title: Library media Reader - Prev/Next item should cross page boundaries
status: To Do
assignee: []
created_date: '2026-09-02 14:11'
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
Split from task-28005 / Qodo review of PR #2307. The ] / [ Reader traversal derives neighbours from the mounted rows only (_library_media_adjacent_row), so at the first or last row of a page it stops even when the browse result has another page. Media pages are 20 items, so a conference of 40 talks (the flagship sequential-review scenario) stops at item 20. 28005 deliberately scoped to within-page and the footer honestly drops the key at the edge; this task makes traversal seamless across the whole filtered browse result. Approach (from Qodo): at a row boundary, if the applied scope has an adjacent page, request that neighbouring page and, after its guarded result applies, select its last (prev) or first (next) row while preserving Reader mode and stale-request protection. Needs the async page-load-then-select machinery + a gated-service test (the reader-flow ControlledDetailMediaService pattern).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 From the last row of a non-final page, ] loads the next page and opens its first item in the Reader
- [ ] #2 From the first row of a non-first page, [ loads the previous page and opens its last item
- [ ] #3 Traversal preserves the active Reader mode and respects stale-request protection
- [ ] #4 At the true first/last item of the whole result the key is still gated off (footer drops it)
<!-- AC:END -->
