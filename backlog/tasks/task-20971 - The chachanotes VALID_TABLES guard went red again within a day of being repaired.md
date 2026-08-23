---
id: TASK-20971
title: >-
  The chachanotes VALID_TABLES guard went red again within a day of being
  repaired
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-22'
labels:
  - bug
  - database
  - testing
  - test-integrity
  - process
priority: high
dependencies:
  - TASK-19568
  - TASK-19057
---

## Description

Source: found on a clean `origin/dev` worktree at `684c6aba4` while renumbering
TASK-19564's migration after the schema-version collision with TASK-19057.

`Tests/DB/test_sql_validation.py::TestChachanotesValidTablesMatchesLiveSchema::
test_no_missing_tables` is red on `dev`:

> `AssertionError: Live schema has tables not in VALID_TABLES['chachanotes']:
> ['actor_pack_persona_intents', 'actor_portable_identities']. Add them (or
> document a deliberate exclusion) in tldw_chatbook/DB/sql_validation.py.`

TASK-19057's v44→v45 migration created two tables and did not add them to the
hand-maintained literal in `DB/sql_validation.py`.

**The drift itself is minor. The recurrence interval is the finding.** This is
the same pin TASK-19568 repaired — that task existed *because* the entry had
gone stale and a red gate stops gating. Its repair merged at `aaec11812`,
2026-08-22 00:16 -0700. TASK-19057 merged at `2fe6ca20f`, 2026-08-22 14:51
-0700 and broke it again. **Fourteen and a half hours.** A one-shot literal
update demonstrably does not hold, and this programme has already paid for a
masked gate once: TASK-19191's per-row inventory review cleared 13 red gate
tests and surfaced three privacy findings that had been sitting behind them,
and TASK-19044's stale version pin had been masking a shipped can't-migrate-past-v39
packaging bug.

Scope note, measured: the **index** census is unaffected —
`idx_actor_pack_persona_intents_state` is already pinned
(`Tests/ChaChaNotesDB/test_index_census.py:93`), and
`Tests/ChaChaNotesDB/test_actor_pack_migration.py` covers the new tables'
migration. It is specifically the `VALID_TABLES` tables literal that drifted,
because nothing connects adding a table to updating it except a human
remembering.

The related TASK-19867 covers the `media` and `prompts` entries of the same
allowlist, which have drifted further and have no guard at all. This task is
about the entry that *does* have a guard and went red anyway.

## Acceptance Criteria

- [x] `Tests/DB/test_sql_validation.py::TestChachanotesValidTablesMatchesLiveSchema`
      is green on `dev`
- [x] `VALID_TABLES['chachanotes']` matches the tables a freshly initialized
      ChaChaNotes database actually contains
- [x] A mechanism exists that makes the *next* table-adding migration fail
      loudly at authoring time rather than after merge — the outcome required is
      that the literal cannot silently fall behind the schema again, not that it
      is correct today
- [x] The chosen mechanism is proven to bite: adding a table without updating
      the allowlist is shown to fail, and the proof does not depend on the same
      hand-maintained list it guards
- [x] The recurrence is recorded where a future migration author will encounter
      it, so the cost of this class of drift is not re-learned a third time

## Notes

Not filed as "update the literal". Twice-repaired-in-a-day is evidence that the
repair shape is wrong, and the acceptance criteria are written to ask for
whatever stops recurrence rather than for another correct-on-the-day list.

This is residue of the TASK-19057 merge, not of the retention work that found
it: `VALID_TABLES` is byte-identical to `dev`'s on TASK-19564's branch, and that
branch's migration contains zero `CREATE TABLE` statements.

## Implementation Plan

1. Reproduce the red pin and re-derive the drift.
2. Answer why the index census survived the same migration and this did not —
   the asymmetry is the design input, not a curiosity.
3. Build an authoring-time guard whose expectation comes from an artifact
   *other* than `VALID_TABLES`, and wire it into `scripts/preflight.sh` and the
   required `derived-artifacts` job.
4. Repair the literal; make the runtime pin's failure message paste-ready.
5. Mutation-prove the new guard on the real tree; Edit-restore.
6. Record the recurrence where a migration author will meet it.
7. Re-derive the `media`/`prompts` state rather than inheriting TASK-19867's.

## Implementation Notes

Two halves: the literal is repaired, and the reason it kept needing repairing
is closed.

**Why the index census survived and this did not.** Both are hand-maintained
literals guarding the same migration; neither is generated. The difference is
purely discovery path. Measured on the TASK-19057 branch
(`git diff b593f853d 09b768239`), the author updated
`Tests/ChaChaNotesDB/test_index_census.py` — the directory where they had just
written `test_actor_pack_migration.py`, so a directory run turned it red — and
bumped three schema-version pins under `Tests/DB/`, which a grep for the
version constant finds. `VALID_TABLES` is reachable by neither: it carries no
schema version, and nothing else in `Tests/DB/test_sql_validation.py` mentions
the feature. Its guard had been surviving by geography. That rules out "update
the literal" and also rules out "add another test".

**Mechanism: `scripts/check_schema_table_allowlist.py`**, run by
`scripts/preflight.sh` and as a step of the required `derived-artifacts` CI
job. Stdlib-only, no database, no install, 0.10 s.

*Independent source of truth: the schema-defining SQL text.* The expected set
is every `CREATE TABLE <name>` in `DB/migrations/chachanotes_*.sql` plus the
SQL string literals of `ChaChaNotes_DB.py`. `VALID_TABLES` is only ever read
(by `ast.literal_eval`, not import) and never contributes to the expectation —
TASK-19045's rule, which is also why auto-generating the literal was rejected.
The `.py` side is scanned through `ast` string constants: a raw-text scan
reports three phantom tables (`IF`, `column`, `as`) from prose inside `#`
comments, and a guard that reports phantoms gets muted.

*Proven to bite, three ways.* (1) Against the un-repaired tree the checker
named `actor_pack_persona_intents` and `actor_portable_identities` and their
migration file, from a static scan alone. (2) Adding
`CREATE TABLE mutation_probe_task_20971` to
`chachanotes_v45_to_v46_sync_log_retention.sql` with `VALID_TABLES` untouched
turned preflight red on exactly one of its five checks, and turned the runtime
pin red on the same name; Edit-restored, `git diff` on that file empty.
(3) `Tests/DB/test_schema_table_allowlist_guard.py` pins the properties
permanently — both drift directions, the vacuous-empty-scan case, the
comment-prose case, the stdlib-only contract, and a direct independence test
that holds the SQL fixed and moves only the allowlist (an identity check could
not change verdict). It also asserts the static scan equals a live
fully-migrated `CharactersRAGDB(":memory:")`: 69 substantive tables, symmetric
difference empty.

The runtime pin stays — two independent oracles — and its failure messages now
print the exact lines to paste and name the fast checker.

**Recurrence recorded** in `tldw_chatbook/DB/migrations/README.md` (new; the
directory a migration author is already in, with the three-date timeline and
the discovery-path explanation), in the `VALID_TABLES` comment block, in
`backlog/docs/lessons-testing-evidence.md`, and as one clause on CLAUDE.md
gotcha #1.

**`media` / `prompts`, re-derived, not inherited** (live `MediaDatabase` /
`PromptsDatabase`, FTS shadows + `sqlite_sequence` excluded): media has 5 live
tables unlisted and 6 listed names that do not exist; prompts has 5 and 4.
TASK-19867's write-up still holds except that `ChunkingTemplates` has since
been allowlisted (6 → 5 on the media side). Both `_get_next_version` methods
still have zero callers in `tldw_chatbook/` and `Tests/`, so the hygiene rating
stands. Left to TASK-19867 deliberately: enabling them here would make
preflight red for everyone on day one, and their sources
(`Client_Media_DB_v2.py`'s `ChunkingTemplates_v7` rebuild-and-rename step)
need a decision this task has no mandate for. The checker's `SCHEMAS` table is
the one-row-each extension point.

**Adjacent finding, not fixed here (different owner).**
`_migrate_from_v45_to_v46` reads
`migrations/chachanotes_v45_to_v46_sync_log_retention.sql` at runtime, and that
filename is absent from `MANIFEST.in`, `[tool.setuptools.package-data]`, and
`Packaging/check_manifest.py` — all three of which list migrations by name with
no wildcard (`include-package-data = false`). As shipped, an installed
distribution cannot migrate past v45. This is TASK-19564 residue; flagged for
filing.

**Files:** added `scripts/check_schema_table_allowlist.py`,
`Tests/DB/test_schema_table_allowlist_guard.py`,
`tldw_chatbook/DB/migrations/README.md`; modified
`tldw_chatbook/DB/sql_validation.py`, `Tests/DB/test_sql_validation.py`,
`scripts/preflight.sh`, `.github/workflows/derived-artifacts.yml`,
`backlog/docs/lessons-testing-evidence.md`, `CLAUDE.md`.

**Verification.** `Tests/DB/` + `Tests/ChaChaNotesDB/`: **1479 passed, 1
skipped**. Repo-wide `--collect-only -q`: **56,532 collected**, 5 collection
errors, all pre-existing dev reds not touched here (4 × `Tests/UI/
test_settings_*` = TASK-20970, `Tests/UI/test_library_file_notes_workspace.py`
= TASK-20972). `./scripts/preflight.sh` green on all five checks. Note the
`preflight.sh` default `python3` may be below the repo's 3.11 floor (it is 3.9
here, which breaks three of the other four checks); run it with
`PYTHON=.venv/bin/python`.
