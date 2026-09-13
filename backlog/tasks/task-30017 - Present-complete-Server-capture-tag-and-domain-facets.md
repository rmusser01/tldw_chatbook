---
id: TASK-30017
title: Present complete Server capture tag and domain facets
status: To Do
assignee: []
created_date: '2026-09-03 02:58'
updated_date: '2026-09-03 02:59'
labels:
  - library
  - collections
  - reading-list
  - facets
  - server-parity
dependencies:
  - TASK-18919
references:
  - TASK-18919
  - backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md
  - Docs/superpowers/specs/2026-09-01-collections-followup-backlog-design.md
  - 'tldw_server:TASK-13154'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reference S2 as `tldw_server:TASK-13154`. Until exact `hasReadingAggregateFacetsV1=true` exists, retain typed filters and never label suggestions from returned rows as complete facets. When supported, every facet browse and value search calls the Server endpoint with bounded paging; Chatbook never filters a loaded prefix to claim a complete result. Expose exact counts, deterministic paging, complete-scope filtering, generation fencing, and explicit loading/empty/error/retry states. Source changes discard prior-authority facet state, and narrow layouts preserve the adaptive reader.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Without `hasReadingAggregateFacetsV1=true`, typed filters remain available and current-page suggestions are never labelled complete facets.
- [ ] #2 With the capability, tag/domain browse and `facet_q` search use bounded Server pages with exact totals; the client never filters a loaded prefix to claim completeness.
- [ ] #3 Counts and values reflect the active capture scope and documented self-excluding semantics, with deterministic navigation across deep pages.
- [ ] #4 Authority/scope changes and unmounts fence late results; loading, empty, stale, failure, and Retry states remain explicit.
- [ ] #5 Service/state/mounted and 160×50/120×35/100×30/80×24 regressions preserve the adaptive reader and Local typed-filter behavior.
<!-- AC:END -->
