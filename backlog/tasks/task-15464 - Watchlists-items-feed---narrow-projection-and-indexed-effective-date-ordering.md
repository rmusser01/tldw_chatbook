---
id: TASK-15464
title: Watchlists items feed: narrow projection and indexed effective-date ordering
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - watchlists
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: the items list queries (`DB/Subscriptions_DB.py:1833-1844/:1862-1872` and `get_new_items`' main path) select `i.*` — pulling `content` (full scraped article text) for up to 100 list rows — and order the whole table by `COALESCE(datetime(published_date), datetime(created_at)) DESC`: per-row datetime parsing, unindexable, LIMIT applied post-sort. O(table) work per Items-pane refresh, multiplied by the fresh-construction cost until task-15463 lands.

Fix direction: a list projection without `content` (fetch on select), plus a stored normalized effective-date column with an index. The schema change goes through `_ensure_watchlists_schema` as a proper migration with legacy-row backfill — identical sort semantics for rows with and without `published_date`. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Items list queries no longer select the content column for list rows (evidence); item detail still loads full content
- [ ] #2 Ordering uses an indexed column with sort order identical to today for both legacy and new rows (migration + tests)
- [ ] #3 Items-pane refresh latency before/after on a large corpus recorded
<!-- AC:END -->
