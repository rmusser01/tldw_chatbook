---
id: TASK-34427
title: Sargable SQL trio with timestamp normalization ADR-215
status: Done
created_date: 2026-10-07 02:42
dependencies:
- TASK-34419
updated_date: 2026-10-07 19:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 6 / F11: julianday keyset ordering datetime(next_review) filtering and json_extract visibility plus NOCASE sort all defeat existing indexes - fixing requires normalizing stored timestamp formats with a schema migration
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR written before code (ADR-224; the plan draft's "ADR-215" number was superseded — 224 is the next free slot),Format audit test documents the split,Migration normalizes timestamps and bumps schema version (v78→v79),Writers emit canonical format only,EXPLAIN QUERY PLAN shows index use for all three paths,Ordering identical on mixed-format fixture,Migration idempotent
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 15 (T15)
Reviewer-verified: migration idempotency via GLOB guard + %f three-digit glob match; monotonic normalization with golden-ordering tests incl. cross-format order flips and same-instant ties; trigger recreation byte-identical to v70_to_v71 origin; recovery wiring mirrors fleet-v78 precedent shape-for-shape; character-card partial expression index serves page+count with no post-sort. Deferred minors: messages-table normalization scoped out (ADR-documented); _DUE_FLASHCARDS drift risk; unconsumed fixture fact; broad pytest.raises in one test; EXPLAIN table in SDD report.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- **ADR**: `backlog/decisions/224-conversation-timestamp-normalization.md` (registered in the decisions README index). Canonical format is exactly ADR-173's `YYYY-MM-DDTHH:MM:SS.mmmZ` as emitted by `Utils/timestamps.py` / `_get_current_utc_timestamp_iso()` — the brief's second-precision sketch was corrected to match the shipped helper, per the plan's binding resolution.
- **Migration** `DB/migrations/chachanotes_v78_to_v79_sargable_timestamp_normalization.sql` (+ `_migrate_from_v78_to_v79` step, `_CURRENT_SCHEMA_VERSION = 79`): normalizes `conversations.last_modified` and `flashcards.next_review` via `strftime('%Y-%m-%dT%H:%M:%fZ', julianday(col))`, GLOB-guarded to non-canonical rows (idempotent) with `julianday(...) IS NOT NULL` so garbage is never NULLed; drops/recreates `conversations_sync_update` around the UPDATE so a format change does not enqueue spurious `sync_log` events (v70→v71 precedent); creates `idx_conv_char_lm (character_id, last_modified DESC, id DESC)` and partial expression index `idx_character_cards_visible_name (name COLLATE NOCASE) WHERE deleted = 0 AND json_extract(...) IS NULL`. Index census updated (2 new pins).
- **Writers**: every `conversations.last_modified` writer already stamped the canonical helper (audit table in the task-15 report); `update_flashcard_review` flipped from `strftime("%Y-%m-%d %H:%M:%S")` (TASK-32803.2 shape) to `to_utc_iso(...)`. The `conversations` table `DEFAULT CURRENT_TIMESTAMP` is the one consciously-left site (table rebuild required; no INSERT relies on it — documented in ADR-224 residual risks). Seek cursors are canonicalized in Python (`to_utc_iso(parse_utc(...))`); unparseable cursor text now raises `InputError`.
- **Query rewrites**: raw keyset + `ORDER BY last_modified DESC, id DESC`; due queries bind a canonical now and compare/order raw `next_review` (NULLs-first preserved); character browse page/count go through named statements served by the new partial index.
- **Tests**: new `Tests/ChaChaNotesDB/test_sargable_timestamps.py` (audit, migration correctness on genuinely-v78 mixed fixtures: canonicalization, idempotent replay, persistence across reopen, golden ordering via julianday/datetime oracles, no spurious sync events, garbage untouched, EXPLAIN QUERY PLAN assertions); `test_flashcard_due_boundary.py` rewritten to the canonical contract; two seek-pagination tests updated (mixed-format scenario moved to the migration suite).
- **Backup/Recovery wiring** (the v77→v78 fleet precedent, required because the restore validator matches whole frozen `sqlite_schema` catalogs): `recovery_core_schema.py` gains `CHACHANOTES_V79_SCHEMAS` (+ CORE head 79, dictionary variant rebased) and the v78→v79 stamp step; `recovery_core.py` accepts the full v78 lineage at version 78 and registers the step; `sqlite_validation.py` admits v78 at the pre-migration gate, executes the installed `.sql` under a new `sargable_migration` authorizer branch (normalization UPDATEs, both indexes + internal REINDEX, the recreated sync trigger, and the trigger-sourced FTS/search/sync writes the UPDATEs compile) plus its function allowlist, and adds V79 to the canvas frozen list; `recovery_operations.py` adds the shared-file Subscriptions hybrid tier (`_SUBSCRIPTIONS_SARGABLE_SCHEMAS`, stamp 79). New `Tests/DB/test_chachanotes_v79_sargable_migration.py` pins the restricted migration, foreign-DDL rollback, and stamp acceptance; the fleet migration test's head expectations moved 78→79.
- **Verification**: `Tests/ChaChaNotesDB/` = 484 passed / 10 failed, failure list byte-identical to the pre-change baseline (pre-existing census-drift/parity/cascade failures, incl. `test_no_unexpected_indexes` which was already red at 7f49cc4856 with the same 16 unpinned legacy indexes); family files (seek pagination, archive, cards paging, study interop, due boundary, study functionality, console chat store, actor pack, personas workbench) green; Backup_Recovery focused set (restore validation, core dependency discovery, restore plan, held rollback, canvas validation) failure list identical to baseline after the wiring; full Tests/DB + Tests/Backup_Recovery compared against a HEAD worktree (see the task-15 report).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
