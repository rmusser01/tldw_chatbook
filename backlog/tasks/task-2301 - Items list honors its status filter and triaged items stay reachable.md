---
id: TASK-2301
title: Items list honors its status filter and triaged items stay reachable
status: Done
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

- [x] With the filter on "All statuses", items of every status (new, review,
      ingested, ignored) appear in the list, visibly distinguishable.
- [x] Acting on an item (view, Ingest, Ignore) never removes it from the
      current view unless the active filter genuinely excludes it.
- [x] Ingest/Ignore give immediate visible feedback beyond row removal.
- [x] A regression test covers "triaged item remains findable via the Items
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

## Implementation Notes

The pane was right all along; it was being handed a new-only list. Two data
layers made "every status" inexpressible:

* `SubscriptionsDB.get_new_items` applied `WHERE i.status = ?` unconditionally
  -- there was no code path through it that did not filter by exactly one
  status.
* `LocalWatchlistsService.list_items` collapsed `status=None` to `"new"`.

`_load_items` asks with `status=None`, so `ItemsPane.items` could only ever
contain `new` rows, and the pane's client-side "All statuses" option had
nothing else to filter. A triaged item was not stale in that list -- it was
absent, which is why it read as deletion. (The screen already knew this: it is
stated in `_blocking_status_for`'s docstring as the reason that cache cannot
be used as a system of record. That comment is now corrected.)

`status=None` now means every status at both layers. The DB default stays
`"new"`, so `briefing_selection`, the subscriptions smoke suite and the
read-status tests are untouched -- only a caller that deliberately passes
`None` sees the change. The two predicates (subscription, status) are also now
composed independently rather than as hand-written variants of one SELECT,
which is how the all-statuses case came to be missing; values remain bound
parameters.

**AC#3.** Ingest/Ignore carry no `patch_item`, so their only visible result
used to arrive whenever the async `_load_items` reload landed -- and that
reload was what deleted the row. `_update_item_status` now repaints the
acted-on row's Status cell the moment the write succeeds, on the same
single-cell `update_item_status_cell` path mark-read-on-open already uses, and
the toast moves to `_notify_watchlists(..., markup=False)`.

**Two existing tests asserted the old behaviour as a premise** and were
corrected rather than deleted:

* `test_a_cancelled_mark_read_still_leaves_the_cached_dict_coherent` asserted
  the item was ABSENT from `_loaded_items`, because absence was the only form
  "coherent" could take. It can now name the status the cache must hold
  (`ignored`) -- a strictly stronger assertion.
* `test_mark_unread_refuses_an_ingest_that_sits_beyond_a_lookup_page` seeded
  520 ingested fillers up front and relied on the target being the only `new`
  item, hence the only row on screen. With every status listed newest-first
  the target (dated 2020) would be buried 520 rows deep and unreachable by any
  real gesture, so the fillers are now seeded AFTER the target's own Ingest.
  That is also the truer reproduction: the page fills up *between* the ingest
  and the `Mark unread` press, which is exactly the situation the guard is for.

### Verification

* New file `Tests/UI/test_watchlists_items_status_filter.py` (9 tests), driven
  bottom-up: DB, service, pane, then the user gesture. The Status column is
  read off the **mounted `DataTable`**, not off `pane.items` -- the two
  disagreeing is this task's whole subject. The Ingest test stubs `_load_items`
  to a no-op on purpose: with the reload left in, an assertion made after it
  lands cannot distinguish the in-place repaint from the rebuild, and would
  stay green with the repaint deleted.
* Mutation-verified: 4 mutations (DB predicate, service collapse, the repaint
  branch, `markup=False`), each reverted individually -> RED -> restored
  byte-exact (md5).
* Gates: `Tests/UI/test_watchlists_read_status.py` +
  `test_watchlists_item_actions.py` + `test_watchlists_inspector.py` +
  `Tests/Subscriptions/` + `Tests/DB/test_subscriptions_db_watchlists.py`
  **666 passed, 2 skipped**; `test_watchlists_content_pane.py` +
  this file **57 passed**; poisoned-order pass
  (`test_watchlists_content_pane.py` + the create-form e2e in one invocation)
  **50 passed**.

### Live verification (235x52, fresh profile, 20 real HN items)

```
=== after opening two items (filter: All statuses) ===
  Stephen Wolfram's Wife Has Died          HN Front Page  new
  Launch HN: EdotEnv (YC S26) ...          HN Front Page  review
  Waymo – Dallas Open to All               HN Front Page  review     <- stayed
=== after Ingest ===
  Waymo – Dallas Open to All               HN Front Page  ingested   <- stayed
  toast: "Item marked ingested."
=== filter = Ingested ===
  Waymo – Dallas Open to All               HN Front Page  ingested
=== filter = Read ===
  Launch HN: EdotEnv (YC S26) ...          HN Front Page  reviewed
```

(The Ingested row also appears under `Read` because `_filtered_items` pins the
currently-SELECTED item into every view by design -- pre-existing, documented
behaviour that exists so the open item never drops out of its own list.)

### Files

* `tldw_chatbook/DB/Subscriptions_DB.py`,
  `tldw_chatbook/Subscriptions/local_watchlists_service.py`,
  `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`.
* `Tests/UI/test_watchlists_items_status_filter.py` (new),
  `Tests/UI/test_watchlists_read_status.py`,
  `Tests/UI/test_watchlists_content_pane.py`.
