---
id: TASK-19568
title: >-
  Schema pins merged red — two gates no longer gate, and an equality pin
  short-circuits the entire persona-visual migration test
status: Done
assignee: []
created_date: '2026-08-21 20:18'
labels:
  - testing
  - db
  - architecture-gate
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 5 (test-suite health & guard efficacy)
— its **(b) masked gates** and **B3 stale schema pins**; Lane 3 **F3**
independently found the same `sql_validation` red, and the review controller
**re-confirmed it by running the test**. All numbers below were re-measured at
this branch base.

**One merge caused all of it.** `452d09b12` (persona-visual, PR #1882) added
the v40→v41 migration and updated neither pinned literal. The gates then fired
exactly as designed — and were **merged red anyway**, which is live proof of
their efficacy and of the process gap that let them be ignored (see
TASK-19572: no required status check exists on `dev`, so a red guard cannot
block a merge).

**Currently red at this base:**

| pin | state |
|---|---|
| `Tests/DB/test_sql_validation.py::test_no_missing_tables` (line 387) | RED — `assert not {'persona_visual_assets','persona_visual_bindings','persona_visual_pack_versions','persona_visual_packs'}`; whole file **1 failed / 22 passed** |
| `Tests/ChaChaNotesDB/test_index_census.py::test_no_unexpected_indexes` | RED under **both** parametrizations (`fresh_bootstrap`, `chain_migrated_from_v4`) — `idx_persona_visual_assets_version_key`, `idx_persona_visual_bindings_persona_active` |
| `Tests/DB/test_chachanotes_default_assistant_enrichment_migration.py:459` | RED — `assert 42 == 40` |
| `Tests/ChaChaNotesDB/test_persona_visual_migration.py:157` | RED — `assert 42 == 41` |

**The worst one is the last, and it is a self-inflicted coverage hole on the
exact migration that caused the others.** In
`test_persona_visual_migration.py`, the function
`test_real_v40_upgrade_installs_separate_persona_visual_schema` is defined
across lines 154-156 and **line 157 is the first statement of its body**:
`assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 41`. Because that assertion
fails immediately, **nothing below it ever runs** — not the v40→v41 migration
call at line 164, not the table census, not the five `EXPECTED_COLUMNS` checks,
not the foreign-key matrix, and **not the two index assertions at lines
184-185, which pin the very indexes the index census reports as unpinned**.
The persona-visual migration is, in practice, untested.

**Correction to the review's own summary, from this filing's verification:**
there is only **one** `== 40` pin, not two. The three ChaChaNotes equality pins
are `== 40` (red), `== 41` (red), `== 42` (green). A fourth hit,
`Tests/DB/test_media_db_schema_v6.py:19` (`== 6`), is the Media database — a
different class, correctly green, not part of this.

**Not in scope here:** `test_screen_size_ratchet` is the third masked gate but
is already tracked as **TASK-3070** (In Progress) and **TASK-3070.11**. Its
*unbudgeted* siblings are filed separately with the architecture findings.

The equality-pin pattern is the root cause worth fixing, not just the four
values: a test that asserts `_CURRENT_SCHEMA_VERSION == <literal>` guarantees a
red on the next migration, and a red at the top of a function body silently
deletes the rest of the test.

## Acceptance Criteria

- [x] `sql_validation.VALID_TABLES` includes the four `persona_visual_*` tables
      and `test_no_missing_tables` is green
- [x] The index census pins the two `persona_visual_*` indexes and is green
      under both parametrizations
- [x] `test_real_v40_upgrade_installs_separate_persona_visual_schema` executes
      its body — the migration, the table census, the column checks, the index
      assertions and the foreign-key matrix all actually run and pass
- [x] Schema-version pins stop using bare equality against
      `_CURRENT_SCHEMA_VERSION`: a migration test asserts the behaviour of
      *its own* migration (v40→v41 produces this schema), not the global
      current version
- [x] A guard makes it hard to short-circuit a test body with a version
      assertion — e.g. version preconditions expressed as a skip/xfail marker
      or `>=`, so a bump cannot silently disable a whole test
- [ ] Adding a migration without updating the derived pins fails a check the
      author sees **before** merge (this is the dependency on TASK-19572) —
      **not done here**: this requires the CI required-status-check
      infrastructure TASK-19572 builds (still To Do at close of this task).
      The local guard already exists and reds correctly (proven by this
      task's mutation tests) — what is missing is CI *enforcing* it, which
      is explicitly out of this task's tooling scope.
- [x] The full ChaChaNotes DB suite is green at the end, with counts recorded

## Implementation Plan

1. Re-derive current state at this branch's base (dev had moved to schema v44
   via task-19554 since filing) — run the four named pins plus
   `Tests/DB/` + `Tests/ChaChaNotesDB/` to get an honest census, not the
   filing's numbers.
2. For each red pin, determine stale-literal vs wrong-schema by reading the
   live-schema derivation the test does (`sqlite_master`/`PRAGMA index_list`
   against a real migrated DB) versus the hand-maintained literal.
3. Fix the two stale literals (`VALID_TABLES['chachanotes']`, the index
   census `EXPECTED_CHACHANOTES_INDEXES` dict) by hand-adding entries —
   never derive them from the code under test (task-19045's rule).
4. Rewrite `test_real_v40_upgrade_installs_separate_persona_visual_schema` to
   stop gating its body on a bare `_CURRENT_SCHEMA_VERSION == 41` equality;
   use the existing `chachanotes_db_at_version` helper to exercise the
   v40->v41 step in isolation regardless of the current global version.
5. Verify the already-landed convention (task-19554's pin-lives-in-the-
   newest-migration's-file design) held through the v44 bump, and that any
   older sibling files use `>=`/dynamic comparisons, not stale equality.
6. Mutation-test every repaired pin: break the thing it protects, confirm
   red, restore via Edit, confirm `git diff` clean.
7. Measure media/prompts VALID_TABLES drift for the report; do not extend
   the guard to them (out of this task's ACs) — record findings only.
8. Run the full targeted suites plus a repo-wide `--collect-only` sweep,
   baseline remaining failures against this branch's own pre-edit state
   (== origin/dev, since no other changes exist on this branch).

## Implementation Notes

**Per-pin state at this branch's base** (`origin/dev` @ `3193816e7`, schema
v44 after task-19554's migration landed — the task's original `== 40`/`RED`
line no longer exists; that pin was independently fixed by commit
`4b08bd7ac9` on 2026-08-21 10:12, before this task started):

| pin | state at base | cause | fix |
|---|---|---|---|
| `test_sql_validation.py::TestChachanotesValidTablesMatchesLiveSchema::test_no_missing_tables` | RED | stale literal — `VALID_TABLES['chachanotes']` missing the 4 `persona_visual_*` tables added by the v40->v41 migration (`452d09b12`) | added the 4 table names |
| `test_index_census.py::test_no_unexpected_indexes` (both params) | RED | stale literal — `EXPECTED_CHACHANOTES_INDEXES` missing 3 indexes: the 2 named in the task PLUS `idx_message_exchanges_message` (v42->v43, task-18300), which joined after filing — dev moved | added all 3 pins (test file's own error message supplied ready-to-paste `IndexPin` literals); also added rationale bullets for the two new UNIQUE indexes in the module docstring, matching the existing per-index documentation convention |
| `test_persona_visual_migration.py:157` (`test_real_v40_upgrade_installs_separate_persona_visual_schema`) | RED, `assert 44 == 41` | **the worst one**: bare equality against the global `_CURRENT_SCHEMA_VERSION`, positioned as the function's first statement, so once dev advanced past v41 nothing below it ever ran — not the migration, not the 5 column checks, not the 2 index assertions, not the FK matrix | rewrote to use `chachanotes_db_at_version(path, 41, ...)` (the repo's own historical-bootstrap primitive) instead of a plain `CharactersRAGDB(path, ...)` open — this exercises the v40->v41 step in isolation at any future global version, so there is no version gate left to short-circuit |
| `test_chachanotes_default_assistant_enrichment_migration.py:459` (`test_current_schema_version_is_current`) | GREEN already | task said `RED, assert 42 == 40` at filing; already fixed by `4b08bd7ac9` before this task started (`>= 41`, non-blocking trailing check, not the function's first line) | none needed — verified still green and still positioned correctly |
| `test_chachanotes_sync_conflict_preservation_migration.py:55` (`test_schema_version_is_44`) | GREEN | this is task-19554's own migration file — the ONE exact `== 44` pin, correctly living with the newest migration per the established convention | none needed |
| `test_media_db_schema_v6.py:19` | GREEN | Media DB, different class, unrelated (confirmed by the task filing) | none needed |

**Convention check (AC: "verify it held through v44"):** confirmed. The
`>=`-and-newest-file-owns-the-exact-pin convention (established by
`4b08bd7ac9`, then reaffirmed by task-19554's own migration commit
`9d62eb07d`) held cleanly through the v42, v43, and v44 bumps — the only
straggler was the persona-visual file fixed here, which predates that
convention (it shipped in the same PR that caused the version drift, #1882,
before the convention existed).

**Mutation evidence (every repaired pin, Edit-based restore, `git diff`
clean after each):**
- `VALID_TABLES['chachanotes']`: removed `persona_visual_packs` -> census
  test reds with `Live schema has tables not in VALID_TABLES...
  ['persona_visual_packs']`; restored, diff clean.
- Index census literal: removed the `idx_message_exchanges_message` pin ->
  `test_no_unexpected_indexes` reds on both parametrizations; restored,
  diff clean.
- Index census against the real schema: dropped
  `CREATE UNIQUE INDEX idx_persona_visual_assets_version_key` from
  `chachanotes_v40_to_v41_persona_visual.sql` -> `test_no_missing_indexes`
  reds on both parametrizations (the exact MUT-INDEX escape class task-19045
  exists to catch); this same mutation also reds the rewritten migration
  test's own index assertion. Restored, diff clean.
- **Coverage-hole proof** (the AC's most important half): reverted the
  migration test to its ORIGINAL (unfixed) form via Edit, re-applied the
  same index-drop mutation, and reran — got the byte-identical failure
  (`assert 44 == 41`, line 157) as the unmutated baseline, proving the old
  test was completely blind to the mutation. Then restored the fixed test
  body and, to prove the fix actually runs the body at the *current* global
  version (44, not 41), mutated an `EXPECTED_COLUMNS` entry instead — this
  now fails on the real column-mismatch assertion, not a version gate.
  Restored via Edit; `git diff` on the migration `.sql` file and the test
  file both clean afterward.

**Test counts:**
- `Tests/DB/test_sql_validation.py` + `Tests/ChaChaNotesDB/test_index_census.py`
  + `Tests/ChaChaNotesDB/test_persona_visual_migration.py`: 35 passed (0
  failed) after the fix, vs 4 failed at base.
- `Tests/DB/` + `Tests/ChaChaNotesDB/` full run: 1364 passed / 6 failed / 1
  skipped after the fix, vs 1360 passed / 10 failed / 1 skipped at base — the
  4 target pins now pass; the remaining 6 failures
  (`Tests/DB/test_core_sqlite_owner_privacy.py`, media-namespace/PrivatePathError
  cases) are byte-identical to the base failure set (diffed the FAILED lines
  directly) — pre-existing on `origin/dev`, unrelated to this task, not
  touched.
- Repo-wide `pytest --collect-only -q`: 54162 tests collected, exit 0, no
  new collection/import errors from these edits.

**Other `VALID_TABLES` maps (AC #5, reported not fixed — out of this task's
ACs):** measured against a live, fully-migrated `MediaDatabase`/
`PromptsDatabase`, both `media` and `prompts` have drifted, and neither has a
derived-schema guard test (only `chachanotes` does):
- `media`: **missing** (real, live, unallowlisted) `ChunkingTemplates`,
  `MediaReadItLaterState`, `ReadingProgress`; **stale** (allowlisted, no
  longer live) `IngestionTriggerTracking`, `MediaModifications`,
  `MediaVersion`, plus the allowlisted FTS names
  (`Keywords_fts`/`Media_fts`/`MediaChunks_fts`) not matching live reality —
  the live FTS tables are named `keyword_fts`/`media_fts` (renamed, not just
  recased) and there is no `MediaChunks_fts` at all (fully absent live).
  Not currently exploitable: the two
  reachable `Client_Media_DB_v2.py` call sites (soft-delete/undelete cascade,
  lines 3047/3282) only ever pass the 4 literal cascade-child names
  (`Transcripts`, `MediaChunks`, `UnvectorizedMediaChunks`,
  `DocumentVersions`), all of which are correctly allowlisted and live; the
  one call site that could pass an arbitrary table (`_get_next_version`,
  line 1574) has no callers anywhere in the tree (dead code).
- `prompts`: **much larger drift** — the live schema uses
  `PromptKeywordsTable`/`PromptKeywordLinks`, not the allowlisted
  `Keywords`/`PromptKeywords`/`Keywords_fts`/`Prompts_fts` (none of which
  exist live); only `Prompts` and `sync_log` still match. The one reachable
  call site (`Prompts_DB.py:1044`, `_get_next_version`) also has no callers
  anywhere in the tree (dead code) — not currently exploitable either, but
  the allowlist itself is almost entirely wrong as an inventory.
- Recommendation: file a follow-up task to add
  `TestMediaValidTablesMatchesLiveSchema` /
  `TestPromptsValidTablesMatchesLiveSchema` guards mirroring the chachanotes
  one, and fix these two maps at the same time — deliberately not done here
  since it's outside this task's ACs and both maps need their own
  correctness pass (this is inventory-scale drift, not a one-line stale
  literal).

**Files modified:**
- `tldw_chatbook/DB/sql_validation.py` — added 4 `persona_visual_*` tables
  to `VALID_TABLES['chachanotes']`.
- `Tests/ChaChaNotesDB/test_index_census.py` — added 3 `IndexPin` entries
  (`idx_message_exchanges_message`, `idx_persona_visual_assets_version_key`,
  `idx_persona_visual_bindings_persona_active`) plus 2 UNIQUE-index rationale
  bullets in the module docstring.
- `Tests/ChaChaNotesDB/test_persona_visual_migration.py` — rewrote
  `test_real_v40_upgrade_installs_separate_persona_visual_schema` to use
  `chachanotes_db_at_version` instead of a version-gated plain DB open.

**Not done (explicitly out of scope, see AC list above):** the CI
required-status-check enforcement (TASK-19572, still To Do) and extending
the `VALID_TABLES` guard to `media`/`prompts` (reported above, recommend a
follow-up task).
