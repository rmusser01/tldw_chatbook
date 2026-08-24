---
id: TASK-21126
title: >-
  Library Search/RAG panel runs an unindexed full-table GROUP BY on the event loop per panel mount
status: Done
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - library
  - rag
  - database
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21126).

`RAG_Admin/local_rag_admin_service.py:592-596` runs `SELECT chunk_engine_version,
COUNT(DISTINCT media_id) ... GROUP BY chunk_engine_version` over `UnvectorizedMediaChunks`
(rows-per-chunk; no index on chunk_engine_version; temp B-tree for the DISTINCT). The
`_maybe_await` seam (rag_admin_scope_service.py:81-84) evaluates this sync call ON the loop,
and the panel remounts per destination switch (its own docstring,
library_search_rag_panel.py:95-98). Full scan + sort per Library navigation click at realistic
chunk counts (1e5-1e6 rows).

## Acceptance Criteria

- [x] The census query runs off the event loop
- [x] An index or maintained count removes the full-scan; EXPLAIN QUERY PLAN before/after recorded in the task
- [x] Panel content unchanged

### Amended during implementation (2026-08-23)

**AC 1 originally read "…and its result is cached per session (invalidated on
ingest/re-chunk)". The cache half was dropped, deliberately.** Once the query
is off the loop and indexed it costs 23 ms at 200k live chunk rows and 123 ms
at 1M — on a worker thread, once per panel show. A session cache would save a
measured zero of user-visible time and would have to be invalidated by ingest,
re-chunk, media hard-delete, media soft-delete/undelete, sync-in, and
`process_unvectorized_chunks`; missing any one leaves the panel claiming
"Chunked by an older engine: N items" with a Re-chunk button that either lies
or hides real work. Re-reading per show IS the freshness protocol and it has
nothing to get wrong. Recorded per the "prescribed fix is a hypothesis" rule.

## Implementation Plan

1. Build the differential harness BEFORE the fix: a real `MediaDatabase` corpus
   at 200k and 1M chunk rows, timing the census and capturing EXPLAIN QUERY
   PLAN, plus a write-side A/B for any index considered.
2. Confirm (or refute) each half of the filing: is the query really unindexed,
   does it really run on the loop, and is a cache warranted once it is fixed?
3. Pick the index shape by measurement, not by shape-of-query reasoning.
4. Move the census off the loop at the narrowest safe seam, keeping every other
   backend's behaviour byte-identical.
5. Walk unmount / quit / error / empty explicitly; test the interleaving the
   offload creates.
6. Mutation-check every assertion.

## Implementation Notes

Two changes: a media-DB index (schema v7 -> v8) and an off-loop seam in
`RAGAdminScopeService`. The panel's rendering, its state machine and every
string it can show are untouched.

**The filing was right about the disease and wrong about the cure, twice.**

1. *"no index on chunk_engine_version … full scan"* — the scan is real, but
   the fix the wording implies is a dead index. An index on
   `(chunk_engine_version, media_id) WHERE deleted = 0` is a perfect covering
   index for this query and **SQLite never chooses it**: no media DB has ever
   run `ANALYZE` (there is no `ANALYZE` anywhere in `Client_Media_DB_v2.py`),
   and with no `sqlite_stat1` the planner keeps using the existing
   `idx_unvectorizedmediachunks_deleted`. Measured at 200k rows: 118.8 ms
   without the index, 120.2 ms with it — 5 MB of disk for nothing. What works
   is leading with the redundant `deleted` column so the index answers the same
   equality search the planner already likes, while also covering the GROUP BY
   and the DISTINCT: `(deleted, chunk_engine_version, media_id) WHERE
   deleted = 0`, chosen with no stats, 23.4 ms.
2. *"cache per session"* — see the amended AC above.

`RAGAdminScopeService._maybe_await(service.get_template_diagnostics())`
evaluated its argument before the first suspension point, so the *synchronous*
local backend ran to completion on the event loop even though the panel already
scheduled an async worker. `_call_off_loop` moves it to `asyncio.to_thread`,
but only for a backend that opts in via `diagnostics_are_thread_safe()`. That
gate is not ceremony: `MediaDatabase` connections are thread-local, so a
`:memory:` database handed to a worker thread opens a *different, empty*
database and the census would have started reporting zero legacy items in
silence. Async backends and unknown/test doubles keep the exact pre-existing
inline path.

### Measurements (real production schema, no ANALYZE — the production state)

| corpus | before | after | plan before | plan after |
|---|---|---|---|---|
| 200k chunk rows / 4k media / 64.3 MB | **118.8 ms** | **23.4 ms** (5.1x) | `SEARCH … USING INDEX idx_unvectorizedmediachunks_deleted (deleted=?)` + `USE TEMP B-TREE FOR GROUP BY` + `USE TEMP B-TREE FOR count(DISTINCT)` | `SEARCH … USING COVERING INDEX idx_unvectorizedmediachunks_engine_census (deleted=?)`, no temp B-trees |
| 1M chunk rows / 20k media / 325.2 MB | **700.9 ms** | **122.8 ms** (5.7x) | same | same |

Off the loop as well as faster, so the UI cost of both columns is now zero: a
300 ms census leaves 28 heartbeat ticks in a 10 ms-tick probe; the same probe
against the pre-change inline call records 2.

Costs, stated: +9% media-DB file size (64.3 -> 70.2 MB at 200k, 325.2 -> 354.8
at 1M; ~30 bytes per LIVE chunk row, soft-deleted rows excluded by the partial
predicate); +0.06 ms on a 50-chunk ingest batch (0.660 -> 0.720 ms median); a
one-off index build at the first open after upgrade (167 ms at 200k, 2.05 s at
1M). Rejected shapes are recorded in the code comment beside the DDL, with
their numbers.

### Lifecycle walk

* **Panel unmount mid-census** — Textual cancels the widget's workers, the
  `await` raises `CancelledError` and nothing after it runs; the SELECT
  finishes on its executor thread and is discarded. `_apply_legacy_chunk_report`
  already returns on `NoMatches`. Strictly better than before, where the same
  unmount left the loop blocked for the full query. Covered by
  `test_unmounting_mid_census_neither_raises_nor_paints`.
* **App quit mid-census** — nothing awaits the thread on the way out; measured
  exit well inside the harness timeout. Covered by
  `test_quitting_mid_census_exits_cleanly`.
* **Error paths** — `to_thread` re-raises into the awaiting coroutine, so
  `get_template_diagnostics`'s existing `except Exception` guard around the
  report line still fires (an unmigrated media DB missing the column is the
  real case) and a genuine backend failure still propagates. Both pinned.
  A failed v7->v8 migration rolls back and leaves the DB at v7, working, on the
  old plan.
* **Empty / first run** — no media at all, and a fully stamped library, both
  render nothing (no line, no Re-chunk button, never a zero). Pinned at the
  service and at the panel.
* **Interleaving** — twelve concurrent censuses against a live DB all return
  the same count (WAL readers take a consistent snapshot).

### Files

* `tldw_chatbook/DB/Client_Media_DB_v2.py` — `_CURRENT_SCHEMA_VERSION` 7 -> 8,
  `_MIGRATIONS[7]`, `_CHUNK_ENGINE_CENSUS_INDEX_MIGRATION_SQL`,
  `_apply_migration_v7_to_v8`.
* `tldw_chatbook/RAG_Admin/rag_admin_scope_service.py` — `_call_off_loop`,
  used by `get_template_diagnostics`.
* `tldw_chatbook/RAG_Admin/local_rag_admin_service.py` —
  `diagnostics_are_thread_safe()`, census docstring.
* `tldw_chatbook/Widgets/Library/library_search_rag_panel.py` — docstring only.
* `Tests/DB/test_media_db_schema_v8.py`,
  `Tests/RAG/test_rag_admin_diagnostics_off_loop.py`,
  `Tests/UI/test_library_rag_legacy_chunk_report_real_backend.py` (new).
