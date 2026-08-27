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
- [x] Any index added states its write-side cost (file size, per-batch ingest time, one-off build time), as 21126 did
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

## Implementation Notes

**Shipped**: media schema **v8 → v9** adding four partial indexes on `Media` —
`idx_media_active_recent`, `_active_type`, `_active_ingested`, `_active_title` — plus
`scripts/check_index_plan_pins.py`, a census of all 215 declared indexes under `DB/`, wired into
`preflight.sh` and the derived-artifacts CI job.

**The design point, which is the whole reason this task existed.** Every plan assertion first
asserts `sqlite_stat1` is **absent**, so a future fixture that runs `ANALYZE` cannot quietly restore
a flattering plan. There is a **negative control** test proving the point directly: the textbook
shape for these queries, `(last_modified DESC, id DESC) WHERE deleted = 0 AND is_trash = 0`, is
**never chosen** without stats — the planner keeps its one-column `idx_media_deleted` search plus
the temp B-tree. That is what makes "the index exists" a worthless assertion.

**Write-side cost, measured** (5,000 rows, same corpus, indexes dropped vs present):

| | without v9 | with v9 |
|---|---|---|
| bulk insert of 5,000 rows | 106.8 ms | 135.9 ms (**+27.3%**) |
| per 50-row ingest batch | — | **+0.291 ms** |
| database file size | 4.17 MB | 4.84 MB (**+16.1%**) |
| one-off index build, 5,000 rows | — | ~5.3 ms per index |

Four indexes cost noticeably more than TASK-21126's single one (+9% size, +0.06 ms/batch), as
expected. The trade is judged worth it for list/browse surfaces a user hits constantly, but the
number is recorded rather than assumed so a future reader can revisit it.

## Completion provenance

The implementing agent hit a session limit **before committing or writing notes**, leaving the work
uncommitted. Everything above was verified independently before commit:

- media schema **v9 confirmed free** across all remote refs (dev is at v8) — this programme has had
  four version collisions;
- `Tests/DB/test_media_db_schema_v9.py` — **48 passed**, including the 9 plan tests and the negative
  control;
- `check_index_plan_pins.py` — 215 indexes censused, 5 plan-pinned, OK;
- write-side cost measured here, because that AC was unticked and unquantified.

**A caution worth recording.** My own first verification probe reported the new indexes were *not*
chosen — because I **paraphrased** the queries and omitted `is_trash = 0`, so they could not match a
partial index predicate. That is exactly the trap TASK-21126 recorded ("a reduced-column EXPLAIN
reports a covering index the real statement cannot use — do not paraphrase the query"), and I walked
straight into it while checking someone else's compliance with it. The pinned tests use the real
production statements; those pass.

**Not done, and left open deliberately**: the `ANALYZE` / `PRAGMA optimize` recommendation (AC 5).
The agent did not reach it and I am not inventing one — deciding whether this database should ever
gather statistics deserves its own measurement, not a guess appended to someone else's work.

## Read-side gain (measured after the fact)

The original work proved the planner *chooses* these indexes and measured what they *cost*, but
never measured what they *bought* — the agent hit a session limit first. Filling that in, with the
real `list_library_media_page` statement, median of 40 runs per arm:

| library size | without v9 | with v9 | |
|---|---|---|---|
| 1,000 | 0.35 ms | 0.07 ms | 5.3x |
| 10,000 | 3.94 ms | 0.07 ms | 58.6x |
| 50,000 | 19.58 ms | **0.07 ms** | **296x** |

The shape matters more than the multiplier: **flat at 0.07 ms regardless of library size**, against
a cost that otherwise grows linearly with every item ingested. Against +0.291 ms per 50-row batch
and +16.1% file size, the trade is clearly worth taking.

**Process note**: a schema migration adding four indexes, justified only by query plans and write
cost, is not evaluable. Plan output tells you the index will be *used*; it does not tell you the
use is *worth* anything. Both halves are required.
