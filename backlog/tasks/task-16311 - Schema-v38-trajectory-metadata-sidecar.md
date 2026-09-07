---
id: TASK-16311
title: Schema v38 trajectory metadata sidecar
status: Done
assignee: []
created_date: '2026-08-15 00:10'
updated_date: '2026-08-15 05:48'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Local-only message_trajectory_metadata table (turn_id, seq, timing, tool payload_json) with migration, accessors, and tests. Per ADR-066 and Docs/superpowers/specs/2026-08-14-console-trajectory-view-design.md; plan task 1 in Docs/superpowers/plans/2026-08-14-console-trajectory-view.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Migration v37->v38 runs idempotently,Accessors roundtrip rows,Per-migration tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in commits 380ea19e4/d74d8851e/f53b656f4: schema v38 sidecar, accessors, BEGIN IMMEDIATE concurrency fix; 5 migration tests green; see Implementation Notes and ADR-066

- **Approach**: local-only `message_trajectory_metadata` sidecar per ADR-066
  (`backlog/decisions/066-console-trajectory-view-and-trace-metadata.md`) and the design
  spec (`Docs/superpowers/specs/2026-08-14-console-trajectory-view-design.md`). Stable
  synced `messages` core, evolvable local edge — future trajectory columns land here
  without touching sync triggers.
- **Key files**: `tldw_chatbook/DB/migrations/chachanotes_v37_to_v38_message_trajectory_metadata.sql`
  (new); `tldw_chatbook/DB/ChaChaNotes_DB.py` (dataclasses `TrajectoryRowWrite`/`TrajectoryRowRead`,
  `_CURRENT_SCHEMA_VERSION = 38`, `_migrate_from_v37_to_v38` + dispatch entry, accessors
  `upsert_trajectory_rows` / `get_trajectory_rows` / `get_next_trajectory_seq`);
  `tldw_chatbook/DB/sql_validation.py` (table allowlist entry);
  `Tests/DB/test_chachanotes_trajectory_metadata_migration.py` (5 tests, green).
- **Decisions**: tool records are **sidecar-only** — TOOL-role messages are deliberately
  never persisted to `messages` (the TOOL-marker invariant), so `tool_call`/`tool_result`
  rows live entirely in this table keyed to the parent assistant message, with
  `payload_json` carrying name/args/result. PK `(message_id, event_kind, seq)` allows
  multiple tool calls per assistant step. Review round made `conv_seq` a UNIQUE index
  (ledger-ordering guarantee), added the `conversations` FK, and an `message_id` index.
- **Concurrency**: `upsert_trajectory_rows` assigns `max(seq)+1` inside the write
  transaction, which now opens with `BEGIN IMMEDIATE` (`transaction(immediate=True)`,
  opt-in — all other callers keep DEFERRED). Under deferred BEGIN, two concurrent
  writers hit SQLite's non-retryable snapshot-upgrade deadlock; IMMEDIATE makes them
  queue on the busy timeout. Pinned by `test_concurrent_direct_db_upserts_produce_unique_seqs`
  (2 threads x 25 rows -> unique, gap-free 1..50). `get_next_trajectory_seq` opens its
  own transaction and must not be nested.
- **Deviations**: version bump required two out-of-brief fixes to hold the DB baseline —
  sql_validation allowlist entry and v37 pins at three reopen sites in
  `test_chachanotes_provider_continuation_migration.py`. Tests live at repo-root
  `Tests/DB/` (no `tldw_chatbook/Tests/` exists). 4 pre-existing `Tests/DB/` failures
  (stale v36 pins from the v37 merge) verified unchanged by this work.
<!-- SECTION:NOTES:END -->
