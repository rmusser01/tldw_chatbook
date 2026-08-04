---
id: TASK-2301
title: Items list honors its status filter and triaged items stay reachable
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - bug
  - uat-2026-08-04
dependencies:
  - task-2300
priority: high
---

## Description (the why)

UAT: the Items list behaves as "new items only" while its filter label reads
"All statuses". Items flip to review/ingested and then silently VANISH from
the list on the next reload; with the filter broken (TASK-2300) they are
unreachable anywhere in the tab. Acting on an item reading as data loss is
the single worst moment in the current flow. Ingest also gives no feedback,
so the disappearance is the only signal (F31).

UAT findings F29 (high), F31.

## Acceptance Criteria (the what)

- [ ] With the filter on "All statuses", items of every status (new, review,
      ingested, ignored) appear in the list, visibly distinguishable.
- [ ] Acting on an item (view, Ingest, Ignore) never removes it from the
      current view unless the active filter genuinely excludes it.
- [ ] Ingest/Ignore give immediate visible feedback beyond row removal.
- [ ] A regression test covers "triaged item remains findable via the Items
      tab".
