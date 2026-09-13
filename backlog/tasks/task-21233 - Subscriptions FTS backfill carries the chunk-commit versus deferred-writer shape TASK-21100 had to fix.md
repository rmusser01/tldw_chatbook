---
id: TASK-21233
title: >-
  Subscriptions FTS backfill carries the chunk-commit versus deferred-writer
  shape TASK-21100 had to fix
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - performance
  - database
  - reliability
  - subscriptions
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; sweep candidate
raised by the TASK-21100 review.

TASK-21100's fix round found that a chunked FTS backfill which commits between chunks collides
with a DEFERRED writer that upgrades to a write lock mid-transaction: on the ChaChaNotes DB
this produced an **instant** `database is locked` on `add_message` during the first-boot
backfill, and SQLite's busy timeout did not help because the lock-upgrade path bypasses it.
Throttling the backfill was tried and proven not to be a fix; the fix was `immediate=True`
scoped to the hot message writers (12 hot writers plus 5 outer wrappers, merged as
`41a240ccd`).

`DB/Subscriptions_DB.py:1439` `backfill_items_fts(chunk_size=500)` has the same structural
shape — a resumable chunked backfill committing per chunk against a table other code writes.
It was not fixed alongside TASK-21100 because the subscriptions DB is idle in the observed
first-boot scenario, so the collision was never provoked. It is therefore a latent instance of
a defect class this burn-down has already paid to learn, not a measured failure.

## Acceptance Criteria

- [ ] The subscriptions writers that can run concurrently with `backfill_items_fts` are enumerated, and each is shown either to take an immediate transaction or to be provably unable to overlap the backfill
- [ ] A test drives a real write against `subscription_items` while a chunked backfill is in progress and fails if that writer receives `database is locked`
- [ ] Throttling or retry is not used as the fix
- [ ] The backfill remains resumable by rowid and its per-chunk cost is unchanged
