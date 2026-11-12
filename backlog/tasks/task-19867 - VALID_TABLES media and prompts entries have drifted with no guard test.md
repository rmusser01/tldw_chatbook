---
id: TASK-19867
title: >-
  VALID_TABLES media and prompts entries have drifted with no guard test
status: Done
assignee:
  - '@t1-chatbook'
created_date: '2026-08-22'
labels:
  - testing
  - test-integrity
  - database
  - hygiene
priority: low
dependencies:
  - TASK-19568
---

## Description

Source: reported by **TASK-19568**'s implementer and upheld in both directions
by that task's reviewer. Re-measured at `3605bd52d` by initializing a real
`MediaDatabase` and `PromptsDatabase` and diffing `sqlite_master` against the
allowlist.

`DB/sql_validation.py`'s `VALID_TABLES` is a hand-maintained allowlist of table
names, keyed by database. TASK-19568 added a guard test for the `chachanotes`
entry, after that entry went stale. The `media` and `prompts` entries have
drifted just as far and have **no guard test at all**.

Measured drift (FTS shadow tables `*_config` / `*_data` / `*_docsize` /
`*_idx` and `sqlite_sequence` excluded):

**`media`** — 6 real tables are not allowlisted
(`ChunkingTemplates`, `MediaReadItLaterState`, `ReadingProgress`,
`keyword_fts`, `media_fts`, `schema_version`) and 6 allowlisted names no longer
exist (`IngestionTriggerTracking`, `Keywords_fts`, `MediaChunks_fts`,
`MediaModifications`, `MediaVersion`, `Media_fts`). The FTS tables were renamed
— `Keywords_fts` → `keyword_fts`, `Media_fts` → `media_fts` — and
`MediaChunks_fts` is simply gone.

**`prompts`** — only `Prompts` and `sync_log` still match reality. The
allowlist names `Keywords`, `Keywords_fts`, `PromptKeywords` and `Prompts_fts`,
none of which exist; the live schema has `PromptKeywordLinks`,
`PromptKeywordsTable`, `prompt_keywords_fts`, `prompts_fts` and
`schema_version`, none of which are allowlisted.

**This is hygiene, not a live defect, and it should not be re-rated. State the
verification plainly so nobody does:**

1. The two `_get_next_version` methods that could take an arbitrary table name
   (`Client_Media_DB_v2.py:1549`, `Prompts_DB.py:1040`) have **no callers
   anywhere** in `tldw_chatbook/` or `Tests/`. There is no reachable path that
   feeds a caller-chosen name into the validator for these two databases.
2. The residual drift fails **closed** in both directions: an allowlisted name
   that no longer exists validates and then fails at SQLite; a real table that
   is not allowlisted is rejected. Neither direction admits an unvalidated
   identifier.

The value of fixing it is that the next person who *does* add a caller inherits
an allowlist that describes reality, and that the guard makes the drift visible
the moment a table is added or renamed — which is the whole reason the
`chachanotes` entry got a guard.

## Acceptance Criteria

- [x] `VALID_TABLES['media']` matches the tables a freshly initialized media
      database actually contains
- [x] `VALID_TABLES['prompts']` matches the tables a freshly initialized
      prompts database actually contains
- [x] A guard test fails when either entry drifts from the live schema, in both
      directions (a real table missing from the allowlist, and an allowlisted
      name that no longer exists), mirroring the `chachanotes` guard TASK-19568
      added
- [x] The guard builds the live set from a real database rather than from a
      hand-written second list, so the guard cannot drift alongside what it
      guards
- [x] The treatment of FTS shadow tables and `sqlite_sequence` is explicit and
      documented, not incidental
- [x] The implementation notes state the currently-unexploitable finding
      explicitly — no callers on either `_get_next_version`, and the drift fails
      closed — so this is not later re-reported as a live injection defect

## Implementation Plan

1. Add `_live_media_table_names()` / `_live_prompts_table_names()` helpers to
   `Tests/DB/test_sql_validation.py` mirroring the chachanotes helper: a real
   `MediaDatabase(":memory:")` / `PromptsDatabase(":memory:")` (full migration
   chain), `sqlite_master` read directly, `sqlite_sequence` and `*_fts`
   (virtual + shadow) excluded with the exclusion documented.
2. Add `TestMediaValidTablesMatchesLiveSchema` / `TestPromptsValidTables
   MatchesLiveSchema` guard classes with the both-directions assertions and
   paste-ready failure messages, stating the unexploitable status in their
   docstrings.
3. Run the guards born-red to enumerate the current drift; fix both
   `VALID_TABLES` entries to the live sets (comment at each entry recording
   what was added/removed and why); update the legacy
   `test_valid_media_tables` / `test_valid_prompts_tables` lists to match.
4. Mutation-verify the guard (remove a live table from the allowlist -> red;
   restore -> green) and run the adjacent suites
   (test_schema_table_allowlist_guard, the static checker script, the CI
   derived-artifacts workflow test).

## Implementation Notes

- Hygiene fix as filed; NOT a live injection defect, verified at the current
  tree (a3c47c6c72, which has moved since the 3605bd52d measurement):
  * `MediaDatabase._get_next_version` (Client_Media_DB_v2.py) and
    `PromptsDatabase._get_next_version` (Prompts_DB.py) still have no callers
    anywhere in `tldw_chatbook/` or `Tests/` -- there is no reachable path
    that feeds a caller-chosen name into the validator for these databases.
  * The residual drift failed closed in both directions: an allowlisted name
    that no longer exists validated and then failed at SQLite; a real table
    not on the allowlist was rejected.
- Measured drift at this tree (born-red guard output): media was missing
  `MediaReadItLaterState`, `ReadingProgress`, `schema_version` and carried
  stale `IngestionTriggerTracking`, `MediaModifications`, `MediaVersion`,
  `Keywords_fts`, `Media_fts`, `MediaChunks_fts`; prompts was missing
  `LocalPromptDrafts`, `PromptKeywordLinks`, `PromptKeywordsTable`,
  `schema_version` and carried stale `Keywords`, `PromptKeywords`,
  `Prompts_fts`, `Keywords_fts`. (`ChunkingTemplates` had already been
  added since the audit's measurement, and `DocumentVersions`/`sync_log`/
  `Transcripts` matched.)
- FTS treatment is explicit and shared with the chachanotes helper: the live
  set excludes `sqlite_sequence` (SQLite AUTOINCREMENT bookkeeping, never a
  validation target) and every `*_fts` name (the FTS5 virtual table plus its
  `_data`/`_idx`/`_docsize`/`_config` shadows, written only by triggers) --
  documented in the helper docstrings, not incidental.
- Guard classes mirror TASK-864/19568's chachanotes runtime pin:
  `_live_*_table_names()` builds the expectation from a real database (never
  a hand-written second list), the classes assert both directions with
  paste-ready failure messages, and the legacy positive-list tests were
  re-synced so they cannot contradict the guards.
- Mutation-verified: removing `MediaReadItLaterState` from the allowlist
  turns `TestMediaValidTablesMatchesLiveSchema::test_no_missing_tables` red;
  restoring it is green. The stale direction was verified by the born-red
  run before the entries were fixed (both classes failed, naming every row).
- No regressions: `Tests/DB/test_sql_validation.py` (51) and
  `Tests/DB/test_schema_table_allowlist_guard.py` +
  `Tests/CI/test_derived_artifacts_workflow.py` all pass;
  `scripts/check_schema_table_allowlist.py` still reports chachanotes clean
  (the static authoring-time half covers chachanotes only, unchanged here).
- ADR required: no -- allowlist hygiene plus a test guard; no schema,
  boundary, or contract change.
- Modified files: `tldw_chatbook/DB/sql_validation.py`,
  `Tests/DB/test_sql_validation.py`.

## Notes

Filed low on purpose. Calling this a SQL-injection exposure would misrepresent
the evidence twice over: there is no reachable caller, and the failure mode is
closed. The finding is that a safety allowlist has quietly stopped describing
the thing it allows, and only one of its three entries has a test.
