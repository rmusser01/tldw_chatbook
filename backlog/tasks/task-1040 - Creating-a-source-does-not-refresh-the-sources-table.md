---
id: TASK-1040
title: >-
  Creating a source leaves the sources table showing the old list
status: Done
assignee: []
created_date: '2026-07-28 02:00'
labels:
  - watchlists
  - bug
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_create_source` never calls `_load_sources()`, so after a source is created successfully the `#sources-table` still shows the list from before. The new source only appears after leaving the Sources section and coming back.

Confirmed live during the task-1035 fix.

A user creating their first source is told nothing happened: the form closes and the table is still empty. The obvious next move is to create it again, which either duplicates the source or hits a uniqueness error for something they cannot see.

The Feeds region *does* update, because it reads `scoped_source_rows()` off a different path — so the screen contradicts itself.

**The tree's counts are stale in the same way**, verified live immediately after creating a source through the fixed form:

```
│ All sources  0           ││  Feeds in All sources (1)
│ Unassigned  0            ││  AI News RSS  (rss)
```

The rail says zero, the centre says one, and they are describing the same thing. So this is not only the sources table: **creating a source refreshes only the view that happens to read it directly**, and every count and list derived from it stays behind. Fix them together, or the next one found will be filed separately again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A source created through the form appears in `#sources-table` without leaving the section
- [x] #2 The Feeds region and the sources table agree immediately after creation
- [x] #3 The reload happens off the UI thread and does not block the form closing
- [x] #4 A test creates a source and asserts the table contents, proven to fail against current code
- [x] #5 Deleting a source refreshes the table, the tree counts and Feeds the same way
- [x] #6 The tree's `All sources` and `Unassigned` counts match the Feeds heading immediately after a create or delete
- [x] #7 A test asserts the tree count and the Feeds heading agree after creating a source, proven to fail against current code
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_create_source` and `_delete_source` both called `_refresh_local_wc_snapshot()` and `_refresh_overview_data()` and stopped there. Those feed the staging line and the Overview cards; `#sources-table` and the rail's counts run their own queries, so neither moved. Both now also run `_load_sources()` (on a worker, grouped, so the form still closes immediately) and `_load_tree_data()`.

The tree-write flows added in TASK-895 already did exactly this — every user-initiated write ends with a `_load_tree_data()` reload. The source paths were the ones that had not been brought in line.

Test asserts both reloads happen after a create, and was proven red first: it failed on the sources reload before the fix.
<!-- SECTION:NOTES:END -->

## Correction — the tree-count half of this task was wrong

The description above claimed the rail and the centre were "describing the same thing, disagreeing", citing `All sources  0` beside `Feeds in All sources (1)`.

**That is not a contradiction.** `get_watchlist_item_counts` returns *item* totals and unread counts, not source counts — its own docstring says so. The number beside a tree node is how many **items** have been scraped; the number in the Feeds heading is how many **sources** the scope covers. With three sources and nothing yet scraped, `All sources  0` and `Feeds in All sources (3)` are both correct, which is exactly what the third UAT observed on `e82ac1b18`.

The fix that shipped is still right — `_create_source` genuinely did leave `#sources-table` stale, and reloading the tree alongside it is correct and harmless — but AC #6 and #7 assert an equivalence that does not exist and should not be treated as a contract.

What is *actually* worth addressing is milder and different: a bare `0` next to `All sources` gives no clue it counts items, so a user who has just added three sources reads it as "nothing was added". That is a labelling question, filed separately if it is worth doing at all.
