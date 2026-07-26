---
id: TASK-688
title: >-
  Wire backfill_items_fts so pre-existing items become searchable
status: To Do
assignee: []
created_date: '2026-07-25 22:05'
labels:
  - watchlists
  - followup
  - blocks-phase-b
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase A (PR #917) added an FTS5 index over subscription_items plus SubscriptionsDB.backfill_items_fts(chunk_size), a chunked and resumable backfill that returns the number of rows indexed and 0 when complete. Nothing calls it — verified: the only reference in tldw_chatbook/ is its own definition.

The index is created empty over a table that may already hold rows, and only the insert/update triggers populate it going forward. So every item a user scraped before upgrading is absent from the index permanently, and Phase C's search returns nothing for their entire back catalogue while appearing to work.

Wiring was deliberately deferred at the final Phase A review so that phase could stay data-layer only (no app.py or UI changes). This is the task that closes it, and it blocks Phase B from claiming search works.

Related: the same missing-index condition was also behind the Phase A Critical where FTS delete triggers rejected mutations of un-indexed rows. That is fixed independently (the delete legs are membership-guarded), so un-indexed rows are now merely unsearchable rather than fatal — but they stay unsearchable until this runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Items that existed before the FTS index was created become searchable without any user action
- [ ] #2 The backfill runs off the UI thread and never blocks app startup or screen mount, on a database with a large subscription_items table
- [ ] #3 An interrupted backfill resumes where it left off rather than restarting
- [ ] #4 The backfill is idempotent — running it again after completion indexes nothing and does not corrupt the index (FTS integrity-check stays clean)
- [ ] #5 Progress is observable to the user, or at minimum logged, rather than being silently in-flight
- [ ] #6 A test covers the upgrade path end to end: a database with pre-existing un-indexed items becomes fully searchable after the wired path runs
<!-- AC:END -->
