---
id: TASK-21593
title: >-
  The media database has no statistics   audit its query plans without sqlite stat1
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - performance
  - database
  - media
priority: medium
---
## Description

TASK-21126 established that the media database has **no `sqlite_stat1`** — there is no `ANALYZE`
anywhere in `Client_Media_DB_v2.py` — and that without stats SQLite's planner makes materially
different index choices. That task proved it for one query, where the textbook covering index was
never chosen and bought 1%. Other media-DB queries may be sitting on similarly bad plans for the
same reason, and nobody has looked.

## Acceptance Criteria

- [ ] The hot media-DB queries are enumerated and each has its `EXPLAIN QUERY PLAN` captured **with `sqlite_stat1` absent**, which is the production state
- [ ] Any query on a scan or temp B-tree plan where an index would be chosen is either fixed with a stats-independent index or recorded with the reason it was left alone
- [ ] A decision is recorded on whether the media DB should run `ANALYZE` or `PRAGMA optimize` at all, with its cost — deliberately not taken in 21126 because it would re-plan every query in the database at once
- [ ] Any index added states its write-side cost (file size, per-batch ingest time, one-off build time), as 21126 did
- [ ] A guard or documented convention requires future "add an index" work in this repo to check the plan with stats absent

## Evidence

From TASK-21126, measured on the real production schema: the prescribed
`(chunk_engine_version, media_id) WHERE deleted = 0` index measured 118.8 ms without it versus
**120.2 ms with it** — 5 MB of disk for 1% — because with no stats the planner stays on the
`deleted` index. Four index shapes were measured; only ones leading with `deleted` are ever
chosen. The shipped index had to lead with a redundant column to be picked at all.

The general rule this produced: **any "add an index" finding in this repo needs a query-plan
check with `sqlite_stat1` absent, not just present.** A probe that runs `ANALYZE` to make its
corpus realistic will confirm an index that production never uses.
