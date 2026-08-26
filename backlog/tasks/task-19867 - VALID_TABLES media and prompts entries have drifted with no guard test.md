---
id: TASK-19867
title: >-
  VALID_TABLES media and prompts entries have drifted with no guard test
status: To Do
assignee: []
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

- [ ] `VALID_TABLES['media']` matches the tables a freshly initialized media
      database actually contains
- [ ] `VALID_TABLES['prompts']` matches the tables a freshly initialized
      prompts database actually contains
- [ ] A guard test fails when either entry drifts from the live schema, in both
      directions (a real table missing from the allowlist, and an allowlisted
      name that no longer exists), mirroring the `chachanotes` guard TASK-19568
      added
- [ ] The guard builds the live set from a real database rather than from a
      hand-written second list, so the guard cannot drift alongside what it
      guards
- [ ] The treatment of FTS shadow tables and `sqlite_sequence` is explicit and
      documented, not incidental
- [ ] The implementation notes state the currently-unexploitable finding
      explicitly — no callers on either `_get_next_version`, and the drift fails
      closed — so this is not later re-reported as a live injection defect

## Notes

Filed low on purpose. Calling this a SQL-injection exposure would misrepresent
the evidence twice over: there is no reachable caller, and the failure mode is
closed. The finding is that a safety allowlist has quietly stopped describing
the thing it allows, and only one of its three entries has a test.
