---
id: TASK-19565
title: >-
  Schema artifacts nobody verifies — 52 of 75 triggers unpinned, no trigger has
  a pinned body, and 12 migration .sql files are decorative
status: Done
assignee:
  - rmusser01
created_date: '2026-08-21 20:15'
labels:
  - db
  - testing
  - schema
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 3 (data layer & schema integrity) —
its **F2** and **F10**. Grouped: both are schema artifacts that look
authoritative and are checked by nothing. Re-verified at this branch base.

**A — the index census's method was never extended to triggers.** The lane
measured this with the index census's own method: **indexes 96, zero unnamed in
`Tests/`. Triggers 75, of which 52 (69%) are named nowhere in `Tests/` at
all.** Unpinned families include every `notes_sync_*`, `character_cards_sync_*`,
`keywords_sync_*`, `world_books_sync_*`, and `messages_ai`/`messages_ad`.

**Worse: even a *named* trigger has no pinned body.** That is not
hypothetical — `ChaChaNotes_DB.py:13165-13178` is a **runtime self-heal that
`DROP`s and re-`CREATE`s `notes_au` on every single database open**, because
the trigger shipped without its `deleted = 0` guards. A trigger whose body was
wrong shipped to every user, and the repair is a permanent startup fixup rather
than a migration. Nothing would have caught the body being wrong.

This directly enables TASK-19566: the FTS soft-delete guard lives in a trigger
body, and no test asserts any trigger body.

Column coverage is only half-covered, and for a specific reason worth carrying
forward: the parity sweep compares **chain-derived sides against each other**,
which is an identity comparison and cannot detect a shared error.

**B — 12 of 26 `DB/migrations/*.sql` files are decorative.** The migration step
executes an **embedded Python constant**; the on-disk `.sql` twin is never
opened. They are kept aligned only by a comment. Meanwhile the packaging test
**pins nine of them as shipped wheel content**, which makes them look
authoritative to anyone reading the repo. **No test compares any file to its
constant**, so they can silently diverge — and a future maintainer editing the
`.sql` file would change nothing at all.

## Acceptance Criteria

- [x] Trigger existence is pinned the way index existence already is: a census
      test fails when a trigger is added, removed or renamed without the test
      being updated
- [x] Trigger **bodies** are pinned, not just their names — the `notes_au`
      incident is the proof this is needed
- [x] The `notes_au` runtime self-heal is replaced by a real migration, so a
      correct trigger body ships rather than being re-patched on every database
      open
- [x] The column parity sweep no longer compares two chain-derived sides
      against each other; it compares against an independently-declared
      expectation
- [x] Each `DB/migrations/*.sql` file either becomes the actual source the
      migration executes, or is deleted — a file that is shipped in the wheel,
      pinned by the packaging test, and never opened is worse than no file
- [x] If any `.sql` files are kept alongside embedded constants, a test
      compares each file to its constant and fails on divergence

## Implementation Plan

1. **Trigger census** (mirrors `test_index_census.py` / task-19045):
   `Tests/ChaChaNotesDB/test_trigger_census.py` with a hand-maintained
   `EXPECTED_CHACHANOTES_TRIGGERS` literal — per trigger: table, timing,
   event, and a sha256 digest of the whitespace-normalized body — asserted
   in BOTH directions against a live fully-migrated DB (fresh bootstrap +
   chain-migrated-from-v4 fixture), plus full normalized-body literals for
   the load-bearing FTS soft-delete family (`notes_ai`/`notes_au`/`notes_ad`
   et al.) so the notes_au-class defect is reviewable in the test file.
2. **notes_au migration**: bump `_CURRENT_SCHEMA_VERSION` 73→74, add
   file-backed `_migrate_from_v73_to_v74`
   (`migrations/chachanotes_v73_to_v74_notes_fts_undelete_guard.sql`) that
   DROPs and re-CREATEs `notes_au` with the guarded body; remove the
   `_ensure_notes_fts_update_trigger_handles_undelete` runtime self-heal and
   both `_initialize_schema` call sites. Red-first test: a genuinely-v73 DB
   carrying the legacy unguarded `notes_au` body is repaired by the chain
   replay (red when the migration's trigger recreation is neutered).
3. **Column parity sweep**: add a hand-maintained
   `EXPECTED_TABLE_COLUMNS` literal (new
   `Tests/ChaChaNotesDB/expected_table_columns.py`) and assert it, both
   directions, inside the historical-bootstrap sweep, so the sweep no longer
   compares only two chain-derived sides.
4. **Migrations single source of truth** — direction: *a `DB/migrations/*.sql`
   must be the executed source of its step, or a pinned standalone-migration
   script actually executed by a test; nothing decorative*:
   - convert the 14 constant-backed ChaChaNotes steps (v16→v17 … v25→v26,
     v28→v31) to file-backed steps (verified: each `.sql` twin is an exact
     statement-sequence match for what the step executes today) and delete
     their constants + "Keep this runner SQL aligned" comments;
   - delete `chachanotes_v41_to_v42_console_project_context.sql` and
     `chachanotes_v42_to_v43_research_quick_note_proofs.sql` (steps carry
     Python recovery logic; constants are the source; never-opened files
     pinned as wheel content are worse than none) and drop their explicit
     packaging-test pins;
   - delete `add_sync_fields_to_notes.sql` (orphaned v4-era one-off, zero
     readers);
   - delete the 7 `workspaces_v*.sql` twins (Workspace_DB executes class
     constants; converting that module is out of scope) incl. the one
     byte-for-byte test that pins the file and the two alignment comments;
   - keep the `agent_runs_v*.sql` family: they ARE executed — by the
     standalone-migration tests — and extend the orchestration test's
     standalone path from v15→v18 to v15→v21 so every file in the family is
     exercised;
   - new guard test `Tests/DB/test_migration_file_sources.py`: every
     `migrations/*.sql` must be runtime-read by its module or executed by a
     test; every `chachanotes_*.sql` must be runtime-read by
     ChaChaNotes_DB.py specifically; no "Keep this runner SQL aligned with"
     twin-marker comments may remain.
5. Targeted test runs (census + bootstrap sweep + migration/interruption
   suites + packaging-derived migration expectations); task notes with the
   per-file dispositions and red→green evidence.

ADR required: no
ADR path: N/A
Reason: no new architectural decision — the migration-source rule follows the
repo's established file-backed convention (v26+ steps, packaging
RUNTIME_MIGRATION_PATHS) and AGENTS.md's standing migration instructions;
the per-file dispositions and tradeoffs are recorded in Implementation Notes
below, per the task-level exception the brief allows.

## Implementation Notes

**Lineage.** Implementation began in a quota-interrupted agent session (2026-10-01) and was completed and validated in the coordinator session the same day; both worked in `.worktrees/data-schema-integrity` on branch `fix/task-19565-19566-schema-artifacts` at base `ef831d9f38`.

**Direction chosen (with ADR).** The `.sql` files became the executed migration source: embedded constants no longer shadow them; six decorative twins that cannot express their Python-orchestrated steps were deleted; `Tests/DB/test_migration_file_sources.py` enforces that every file-backed migration opens its file. Recorded as ADR-208 (`backlog/decisions/208-migration-sql-files-are-the-executed-source.md`).

**AC-by-AC evidence.**
- *Trigger existence census*: `Tests/ChaChaNotesDB/test_trigger_census.py` — 19 passed. Census fails when a trigger is added/removed/renamed without the pin being updated (the index-census method extended to triggers).
- *Trigger bodies pinned*: the census pins a whitespace-normalized digest of each trigger's `sqlite_master.sql` (formatting-only edits pass; ANY body edit fails) — the `notes_au` incident class.
- *notes_au self-heal -> real migration*: `_CURRENT_SCHEMA_VERSION` 73 -> 74 with `DB/migrations/chachanotes_v73_to_v74_notes_fts_undelete_guard.sql`; the runtime DROP/reCREATE-on-open self-heal is removed from `ChaChaNotes_DB.py` (net -838 lines with the shadowed-constant removal); `Tests/ChaChaNotesDB/test_notes_fts_undelete_guard_migration.py` — 3 passed.
- *Parity sweep against a declared expectation*: `Tests/ChaChaNotesDB/expected_table_columns.py` is the independently-declared column set consumed by `test_historical_bootstrap.py`; the chain-derived-vs-chain-derived identity comparison is gone.
- *`.sql` real-or-deleted + comparator*: covered by the direction above and `test_migration_file_sources.py` (4 passed); no decorative twin survives.

**Validation.** New tests 26/26 green. The six touched existing suites: 156 passed on this tree and 156 passed on a clean `ef831d9f38` baseline worktree; the single failure (`test_chachanotes_citation_provenance_migration.py::test_schema_has_every_unique_and_partial_index`) is identical on the baseline — pre-existing on dev, out of scope. All `Tests/Packaging/test_installed_distribution.py` parametrizations ERROR identically on both trees at the `built_distributions` fixture (`python -m build` backend unavailable in local venvs — environmental); the only count delta, `test_release_checker_rejects_missing_database_migration` 82 -> 110, is the intended parametrization growth from the conversion (file-backed migrations 41 -> 55, x2 archive formats), not breakage.

ADR required: yes — ADR-208 (storage/migrations source-of-truth decision per AGENTS.md).
