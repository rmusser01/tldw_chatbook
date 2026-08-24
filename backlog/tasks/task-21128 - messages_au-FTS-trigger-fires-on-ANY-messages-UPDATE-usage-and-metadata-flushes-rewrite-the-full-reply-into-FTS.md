---
id: TASK-21128
title: >-
  messages_au FTS trigger fires on ANY messages UPDATE - usage and metadata
  flushes rewrite the full reply into FTS
status: Done
assignee:
  - '@claude-fable-5'
created_date: '2026-08-22'
updated_date: '2026-08-24 00:33'
labels:
  - performance
  - database
  - fts
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21128).

The `messages_au` trigger (recreated, still unconditional, in the v46 migration SQL ~lines
396-405) has no `UPDATE OF content` column list, so usage-only and metadata-only flushes - now
3-4 UPDATEs per chat turn (content finalize, usage flush ChaChaNotes_DB.py:11030-11082,
metadata flush :11098+) - each re-tokenize and rewrite the full assistant reply into
`messages_fts`. WAL+NORMAL, so write amplification rather than fsync storm.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The `messages_au` trigger fires only when an FTS-relevant column is written (`AFTER UPDATE OF content, deleted ON messages`), preserving v47's `messages_fts_docsize` membership guard and the `old.deleted = 0` / `new.deleted = 0` guards unchanged
- [x] #2 A new migration with a version bump per repo policy (v48 -> v49, renumbered from v47 -> v48; no earlier step is edited in place) ships the trigger change, so already-upgraded databases converge on one trigger shape
- [x] #3 FTS search stays correct across the whole trigger matrix - content edit, usage-only flush, metadata-only flush, soft delete, undelete, hard delete, streaming finalize - for an indexed row AND for an un-backfilled row inside v47's backfill window, with FTS5 integrity-check(0) green after every cell
- [x] #4 No path can leave the index STALE: every column `messages_fts` indexes, plus the column that governs index membership, is covered by the trigger's UPDATE OF list, asserted against the live schema
- [x] #5 A write-count probe over one streamed turn shows the FTS rewrite count drop from 3-4 to 1
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-locate the defect on base fb0a9601e and re-measure. DONE before coding: a scratch probe over a
   simulated streamed turn measured 4 FTS index rewrites (messages_fts_data 55 -> 12,636 bytes for one
   400-token reply) under the shipped `AFTER UPDATE ON messages` trigger.
2. Determine the CORRECT column list. `AFTER UPDATE OF content` alone (the wording the filing proposed)
   is WRONG: soft delete is `UPDATE messages SET deleted = 1` and never names `content`, so the trigger
   would not fire and a soft-deleted message would STAY searchable. Measured on the scratch matrix
   (soft_delete_drops: False). The dependency set is {columns messages_fts indexes} + {the column that
   governs membership} = {content, deleted}. AC #1 amended accordingly before coding.
3. Sweep every remote ref + on-disk worktree for a concurrent claim on the next schema version.
4. Ship as a NEW migration chachanotes_v47_to_v48_messages_fts_update_scope.sql + `_migrate_from_v47_to_v48`
   + `_CURRENT_SCHEMA_VERSION` bump + migration-map registration. v47 is not edited in place. DROP + bare
   CREATE so the step is re-enterable (task-19553), version bump as a separate rowcount-guarded UPDATE.
5. Tests, red-first: a new Tests/DB/test_chachanotes_v48_messages_fts_update_scope.py carrying the exact
   current-version pin (v47's file relaxes to `>=`, the established convention), with (a) the full trigger
   matrix on an indexed row, (b) the same matrix on an un-backfilled row inside v47's backfill window,
   (c) integrity-check(0) after every cell, (d) a structural no-stale-index census that derives the
   required UPDATE OF columns from the live messages_fts schema, (e) the write-count probe asserting
   exactly one FTS rewrite per streamed turn.
6. Verify the .sql reaches the wheel AND the sdist (MANIFEST.in glob + pyproject package-data glob +
   Packaging/check_manifest.py + Tests/Packaging), rather than assuming the glob covers it.
7. Run the ChaChaNotes migration/schema suites, the FTS search tests, Tests/DB and Tests/ChaChaNotesDB,
   tee'd under test-logs/, plus a --collect-only sweep. A/B anything red against base fb0a9601e.
8. ./scripts/preflight.sh; read any drift rows before regenerating anything.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ships the trigger rescoping as a new migration, v48 -> v49 (authored as v47 -> v48; see the
renumber section). The production diff is four things: `_CURRENT_SCHEMA_VERSION = 49`, a
`_migrate_from_v48_to_v49` runner, its entry in the migration map, and
`migrations/chachanotes_v48_to_v49_messages_fts_update_scope.sql`. That
`.sql` differs from v47's trigger by exactly ONE line -- `AFTER UPDATE ON messages` becomes
`AFTER UPDATE OF content, deleted ON messages` -- with both v47 guards (`old.deleted = 0` plus
the `messages_fts_docsize` membership test on the delete half, `new.deleted = 0` on the insert
half) preserved byte-for-byte, verified by diffing the two trigger bodies.

## The AC's prescribed fix was wrong -- an index-layer leak of deleted text

The filing named `AFTER UPDATE OF content`. Soft delete is
`UPDATE messages SET deleted = 1 ...` and never names `content`, so under that shape the
trigger does not fire and the tombstoned row STAYS IN THE INDEX -- the task-19567 guarantee.
Measured on a scratch matrix before any code was written (a direct `messages_fts MATCH`
returned the tombstoned rowid), and re-proved by mutation afterwards: that shape turns 8 tests
in the new file and all three `messages` cases in
`Tests/DB/test_fts_soft_delete_index_witness.py` red.

Scope it precisely (review correction): all SIX production `messages_fts` consumers re-filter
on `m.deleted = 0` (`ChaChaNotes_DB.py:9131, 10318, 12496, 13935`;
`RAG_Search/simplified/rag_service.py:2371, 2402` -- verified one by one), so the rejected
shape would have been an INDEX-LAYER leak: the deleted message's tokens retained in
`messages_fts_data` and reachable by a direct index query, NOT a user-visible search leak. It
is still a real regression of the task-19567 guarantee, which is stated at the index precisely
because that consumer-side filtering is what kept the original trigger defect invisible. AC #1 was amended before coding; the
correct column set is mechanical -- every column the index STORES (`content`) plus the column
that decides MEMBERSHIP (`deleted`). `Docs/Design/2026-08-22-holistic-perf-review.md` carries
the correction in place (its "Corrections found during implementation" section, entry 3).

## Schema-version collision, resolved: renumbered v47->v48 to v48->v49

Flagged at authoring time: `origin/dev` was at 47 and one other UNMERGED branch,
`origin/docs/console-rag-ux-design`, also claimed 48 with
`chachanotes_v47_to_v48_console_library_policy.sql`. Swept every remote/local ref and every
on-disk worktree, twice. Schema versions must be CONTIGUOUS, so the task-id "leapfrog" rule
does not apply -- taking 49 pre-emptively would have left a hole no database could cross, and
the rule recorded was "whichever branch merges SECOND renumbers".

That branch merged first. This step is now **v48 -> v49**: the `.sql` filename, the runner
`_migrate_from_v48_to_v49`, its `_require_migration_entry_version(conn, 48, "V48→V49")`, the
five `V48→V49` label strings, the `SET version = 49 ... AND version = 48` bump, the
`final_version != 49` check, the map key `48:` (dev's `47:` entry untouched), the constant, the
test filename, the exact version pin, the two version-bearing test names, the
`chachanotes_db_at_version(..., 48)` fixtures and the poison label.

Dev's v48 was RE-READ rather than predicted before renumbering, because the "before" arm of the
write-count probe and the upgrade test's pre-fix assertion both depend on what a v48 database's
`messages_au` actually looks like. It adds `console_conversation_library_policy`,
`console_dispatch_checkpoints`, one index, the `messages.assistant_generation_state` column and
a rewrite of the four `messages_sync_*` triggers -- and it does NOT touch
`messages_au`/`_ai`/`_ad`, so the baseline is byte-identical to v47's and both assertions stand
(and are pinned, so a future step that does touch `messages_au` fails them loudly).

One finding from that read, in our favour: dev's v48 adds THREE new
`UPDATE messages SET assistant_generation_state = ...` dispatch writers
(`Chat/console_dispatch_repository.py:401, 668, 1017`) that name no indexed column. Under the
pre-fix trigger each would have been another full re-tokenization of the reply, so the
amplification this step removes grew while the task sat in review.

## Evidence

- Write-count probe, in-process A/B varying only the trigger definition
  (`test_a_streamed_turn_rewrites_the_index_once`): one streamed turn with a 400-token reply
  rewrites `messages_fts` **4 times before, 1 time after**; `messages_fts_data` grows
  **12,581 bytes before, 3,146 after**. The pre-fix arm is asserted too, so the probe fails
  loudly if it ever stops measuring the defect.
- Red-first at base `fb0a9601e` (pinned detached worktree, not a moving ref): the new file runs
  **10 failed / 9 passed** there, failing on the behavioural assertions rather than on import.
  The 9 that pass at base are regression guards for behaviour that was already correct.
- Trigger matrix, twice: once on an INDEXED row and once on an UN-BACKFILLED row inside
  task-21100's backfill window (content edit, usage-only flush, metadata-only flush, soft
  delete, undelete, hard delete, streaming finalize), plus three EDGE sequences a per-cell
  matrix cannot see (content edit AFTER auxiliary flushes; soft delete after; hard delete
  after). FTS5 `integrity-check(0)` after every cell. "Must not write" cells are witnessed
  against a sha256 of `messages_fts_data`, not against MATCH results, because a
  delete-then-reinsert of identical content is invisible to MATCH.
- No-stale-index census (`test_the_update_of_list_covers_every_fts_relevant_column`): derives
  the required column set from the LIVE schema (`PRAGMA table_info(messages_fts)` plus
  `deleted`) and parses the trigger's `UPDATE OF` list out of `sqlite_master`, so widening the
  fts5 table without widening the trigger fails at authoring time.
- Mutation-tested three ways: `OF content` alone (11 red), delete-half membership guard removed
  (4 red), `new.deleted = 0` removed (11 red).
- Packaging verified empirically, not assumed: with setuptools+pip installed,
  `test_built_artifact_ships_the_migrations_its_own_code_opens` and both new
  `test_release_checker_rejects_missing_database_migration[...v47_to_v48...]` params pass, and
  `test_installed_distribution_migrates_v35_database_to_current[source|sdist]` migrates a v35
  database to 48 inside an installed wheel. The four derived lists are all globs/regex since
  TASK-19860; nothing hand-maintained needed an entry.

## Test counts

- `Tests/DB` + `Tests/ChaChaNotesDB`: **1592 passed, 1 skipped, 0 failed**. Collection grew by
  20 against the base run: the 19 new tests, plus one new `test_historical_bootstrap` replay
  parameter (it ranges over `MINIMUM_BOOTSTRAP_VERSION .. _CURRENT_SCHEMA_VERSION`).
- Adjacent suites (Sync_Interop, Character_Chat, Prompts_DB, Media_DB, two RAG FTS files,
  db_upgrade_notice, conversation_local_marks, smoke, packaging-data-safety): 3 reds, all three
  reproduced identically at base `fb0a9601e` -- pre-existing dev reds, not touched.
- `Tests/Packaging` with a build-capable venv: 109 passed, 1 failed. That one
  (`test_installed_wheel_loaders_entry_points_and_assets_are_immutable`) asserts no `sys.path`
  entry sits under the checkout root, and this worktree keeps its `.venv` INSIDE the checkout,
  so `site-packages` trips it. Proven to be a venv-location artifact independent of the diff:
  the same path resolves outside the base worktree's root, which is the only reason it passes
  there.
- Full `--collect-only` sweep: **57,697 tests collected**, 4 collection errors, all four
  reproduced identically at base (missing optional `torch` / Confluence deps).
- `./scripts/preflight.sh`: all five checks pass. The diagnostic inventory reported +3 calls in
  `ChaChaNotes_DB.py`; the three statements were read individually
  (`--statements ... --since`) before regenerating -- the two `logger.info` migration lines and
  the `logger.opt(exception=True).error` handler, each identical in shape to the v46->v47
  step's, interpolating the schema name, the DB path and the exception, which is the existing
  reviewed pattern for all 44 migration steps.

## Files

Added `tldw_chatbook/DB/migrations/chachanotes_v47_to_v48_messages_fts_update_scope.sql` and
`Tests/DB/test_chachanotes_v48_messages_fts_update_scope.py`. Modified
`tldw_chatbook/DB/ChaChaNotes_DB.py`,
`Tests/DB/test_chachanotes_v47_messages_fts_backfill.py` (the exact version pin moves to the
newest step's file, and its four END-STATE version literals become
`_CURRENT_SCHEMA_VERSION`-relative -- a literal is only correct at a fixture's seeded starting
point), `Docs/Design/2026-08-22-holistic-perf-review.md`,
`Docs/security/production-diagnostic-inventory.json`, and
`backlog/docs/lessons-testing-evidence.md`.

## Rebase onto dev's v48 (renumber to v49)

Rebased onto `origin/dev` d20dd733b. Collapsed the branch's five commits into one FIRST: a
multi-commit replay would have produced intermediate commits carrying two
`_migrate_from_v47_to_v48` definitions and a duplicate migration-map key, which Python accepts
silently (last one wins). `rerere` was explicitly disabled for the rebase -- the rr-cache was
NOT empty, contrary to the hand-off note, and a bad cached resolution for this file had already
corrupted it once.

The one conflicted file (`ChaChaNotes_DB.py`) was resolved by discarding the merge entirely
(`git checkout --ours`, i.e. dev's file byte-for-byte) and re-applying three surgical edits:
the constant, the map entry, and the method extracted verbatim from the pre-rebase commit. Git
had interleaved the two migration methods line-by-line, which is the shape that produces silent
corruption. Final diff for that file vs dev: **96 insertions, 1 deletion** (the old constant
line) -- nothing of dev's is lost, and both runners exist exactly once.

Four repairs were needed to make the renumbered tests pass on dev's schema, three of them
caused by dev's v48 rather than by the renumber:

1. Three post-migration `_version(...) == 48` assertions in this task's test file were missed by
   the scripted renumber and had to become 49. (The seeded-fixture `48`s and the "a failing
   chain must not bump the stamp" `48` are correct and stay.)
2. `_seed_v45` now inserts rows with explicit SQL instead of `add_message`. Dev's v48 added
   `messages.assistant_generation_state` to `add_message`'s unconditional INSERT column list, so
   the current writer can no longer write ANY pre-v48 schema -- `historical_bootstrap` fixtures
   below 48 raise `table messages has no column named assistant_generation_state`.
3. The one fixture that crosses schema 47 now passes `console_library_migration_seed`. Dev's v48
   made that a hard precondition for upgrading any non-fresh database.
4. Dev's own v48 test file (`test_chachanotes_console_library_policy_migration.py`) held the
   exact current-version pin; per the repo convention the pin moves to the newest step's file, so
   its two end-state assertions are now `_CURRENT_SCHEMA_VERSION`-relative and its docstring
   records the handover. Same repair this task already applied to the v47 file.

### Dev reds, measured not assumed

`Tests/DB` + `Tests/ChaChaNotesDB` on pristine `origin/dev` d20dd733b: **15 failed, 1648
passed**. Same suites on this branch: **14 failed, 1669 passed**. Set-differenced
mechanically: **every red on this branch is reproduced on pristine dev**, and the branch FIXES
one of dev's (`test_schema_version_is_47`, via the `>=` relaxation). Fourteen of dev's reds are
the two v48 preconditions above hitting historical fixtures across four files.
`test_installed_distribution_migrates_v35_database_to_current[source|sdist]` fails identically
on dev for the seed reason. **Not a user-facing upgrade wall** -- the real boot path supplies the
seed (`config.py:7216, 7268`); the breakage is confined to constructors that open a DB directly.
Worth filing against the v48 author; not fixed here.

## Review-accuracy corrections (post-review, no code change)

Review came back merge-ready with four documentation Minors, all of the "don't overclaim"
class; fixed in a follow-up commit that changes no code. (1) The soft-delete leak is scoped as
INDEX-LAYER rather than user-visible in the migration header, these notes and design-doc
correction 3, after verifying all six consumers re-filter. (2) The lessons entry said "four
production consumers"; there are six -- the two in `rag_service.py` were missed. (3) The census
docstring claimed to be the only guard against a stale list; it asserts `required <= listed`,
so it covers the too-narrow direction only. The over-broad direction is covered by five other
assertions, measured by mutating the list to `content, deleted, usage_json` (5 red). (4) One
item was NOT actioned: `test-logs/ratchet-red-first.txt` is not untracked residue -- it is
TASK-21116's evidence artifact, committed to dev in `be0a16946` (PR #2014) despite the
`test-logs/` ignore rule. Deleting it here would remove another merged task's artifact inside
an unrelated commit, so it is left in place and raised with the coordinator instead. This
task's red-first proof is the untracked `test-logs/redfirst-base-fb0a9601e.txt`.
<!-- SECTION:NOTES:END -->
