---
id: TASK-2301
title: Items list honors its status filter and triaged items stay reachable
status: In Progress
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

## Implementation Plan

1. Confirm the mechanism at the data layer: `SubscriptionsDB.get_new_items`
   always applies `WHERE status = ?`, and `LocalWatchlistsService.list_items`
   collapses `status=None` to `"new"`, so `_load_items(status=None)` can only
   ever hand the pane NEW items -- the pane's "All statuses" filter has
   nothing else to filter.
2. Give the DB an honest all-statuses path (`status=None` drops the
   predicate), keeping `"new"` as the default so existing callers are
   unchanged.
3. Stop the service collapsing `None` to `"new"`; pass the caller's intent
   through.
4. Confirm the screen's loader and the pane's filter now agree, and that the
   Status column makes the statuses distinguishable.
5. Give Ingest/Ignore immediate in-place feedback: repaint the row's Status
   cell on success (not only via the async reload) and keep the toast, with
   `markup=False`.
6. Regression tests: all statuses reach the pane under "All statuses"; a
   triaged item is still findable; picking a status filters to it; Ingest
   repaints the live cell rather than removing the row.
