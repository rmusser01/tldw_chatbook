---
id: TASK-21100
title: >-
  First-boot migration wall - move the v46 messages_fts rebuild out of the
  version-bump transaction
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 01:52'
labels:
  - performance
  - database
  - migrations
  - startup
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1` (2026-08-22). Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21100).

First boot after an upgrade replays schema v34->v46 (12 migrations) inside `TldwCli.__init__`
(app.py:5808 thread-pool join -> `_init_notes_service`), before anything paints.
`chachanotes_v45_to_v46_sync_log_retention.sql` (merged 2026-08-22, PR #1974) unconditionally
rebuilds the entire `messages_fts` index (`delete-all` + reinsert of every non-deleted message)
plus nine full-`sync_log` purge scans, all in ONE transaction (runner
`ChaChaNotes_DB.py:5936-6021`). On a large profile on a slow disk this is tens of seconds to
minutes of silent pre-paint hang, with the WAL ballooning to roughly index size. This is very
plausibly the literal "it got slow after updating" complaint. The migration's privacy goal must
be preserved; the delivery mechanism is the problem.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The v45->v46 FTS rebuild runs as a resumable, chunked backfill outside the schema-version-bump transaction (in-repo exemplar: `Subscriptions/fts_backfill.py`), so an interrupted upgrade neither bricks the DB nor loses the version bump
- [x] #2 A visible "upgrading database..." state (splash or equivalent) is shown while any pending migration chain runs, instead of a silent hang
- [x] #3 A timed probe on a seeded scratch DB (stamped v45, >=10k messages) demonstrates the first-paint path no longer blocks on the full FTS rewrite, with before/after numbers recorded in the task
- [x] #4 Existing migration tests stay green; the v46 privacy semantics (sync_log retention, deleted-guards) are unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design chosen: edit the shipped v45->v46 SQL to keep `delete-all` (cheap index clear
that also removes the pre-v46 corruption and tombstones) but DROP the O(total chat
text) `messages_fts` reinsert; add a new v46->v47 step that recreates `messages_au`/
`messages_ad` with an index-membership (`messages_fts_docsize`) guard on their FTS
'delete' halves; deliver the reinsert as a chunked, resumable backfill outside the
version-bump transaction, driven from a background worker at app mount (exemplar:
`Subscriptions/fts_backfill.py` + `TldwCli._backfill_subscription_items_fts`).

Why both an edit AND a v47, not one or the other:
- The whole pending chain replays inside ONE outer transaction (`_initialize_schema`
  wraps every step), so leaving the reinsert anywhere in the chain keeps the wall.
  The expensive statement must leave the v46 SQL file.
- Users already stamped v46 (original SQL, full rebuild done) must be unaffected but
  MUST still receive the guarded triggers, because during any future backfill-shaped
  window an FTS 'delete' for an unindexed row corrupts an external-content index
  (the exact task-19567 corruption class). A v47 step gives every database one
  convergent trigger shape; the version stamp itself discriminates the populations:
  only databases that replay the EDITED v46 get the delete-all + deferred reinsert,
  already-at-46 databases skip the step entirely and keep their complete index.
- No divergent end states: both populations end at v47 with identical triggers and
  (after backfill) identical index contents.

Resumability invariant (same as the Subscriptions exemplar): the "not yet indexed"
state lives in the DB itself -- membership in the `messages_fts_docsize` shadow
table (populated only by real FTS writes). Each chunk is its own IMMEDIATE
transaction (select rowids missing from docsize -> insert), so a kill at any point
leaves a consistent index and the next run resumes from the DB state. No separate
progress marker to drift.

Search-window semantics (deliberate): after the upgrade commits, `messages_fts` is
empty-but-consistent and fills in the background (oldest rowid first); message-
content search returns progressively more history until the backfill completes; it
can never error and never returns tombstoned rows. New/edited messages are indexed
immediately by the triggers regardless of backfill progress.

Steps:
1. Baseline existing migration/FTS tests + capture BEFORE timing probe (v45-stamped
   scratch DB, >=10k messages, time `CharactersRAGDB(path)` construction).
2. Red-first tests in a new `Tests/DB/test_chachanotes_v47_fts_backfill_migration.py`:
   construction from v45 no longer performs the full reinsert; backfill chunk API
   resumes after interruption and converges to the one-shot end state; window-safety
   (update/soft-delete/hard-delete of un-backfilled rows corrupt nothing); v47
   refuses to run except at v46-entry; poisoned-step atomicity mirror; version pin
   moves here (==47), old file's pin relaxed to >=46 per the documented pin-move rule.
3. Edit `chachanotes_v45_to_v46_sync_log_retention.sql` (drop messages reinsert only;
   keep delete-all, keyword_collections/world_books rebuilds, purges, triggers).
4. Add `chachanotes_v46_to_v47_messages_fts_guarded_triggers.sql` + runner method +
   `_CURRENT_SCHEMA_VERSION = 47` (re-check for collisions at commit) + packaging
   lines (pyproject package-data + MANIFEST.in) for the new file AND the missing
   v45_to_v46 file (pre-existing 19860 gap sitting on this exact chain).
5. Add `CharactersRAGDB.backfill_messages_fts()` chunk method + driver module
   `tldw_chatbook/DB/chachanotes_fts_backfill.py`; wire an `on_mount` thread worker
   in app.py next to the subscriptions backfill.
6. Visible-progress seam: `_print_db_upgrade_notice_if_pending()` printed to the
   terminal from both TUI entry points BEFORE `TldwCli()` construction (nothing can
   paint during `__init__`, so the pre-paint terminal line is the honest minimal
   "upgrading database..." state; splash cannot exist yet at that phase).
7. Update the two existing tests whose delivery assumption changes
   (`test_upgrading_reindexes_only_live_rows_into_messages_fts` drives the backfill
   to completion first; the ==46 pin file). Run new tests, ALL ChaChaNotes
   migration/schema tests, FTS witness/search tests, `--collect-only` sweep; tee to
   test-logs/. AFTER timing probe + interruption (SIGKILL) evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped as designed: edited v45->v46 (keeps `'delete-all'`, drops the O(total chat
text) reinsert) + new v46->v47 (`chachanotes_v46_to_v47_messages_fts_backfill_guards
.sql`: recreates `messages_au`/`messages_ad` with a `messages_fts_docsize` membership
guard on the FTS 'delete' halves) + `CharactersRAGDB.backfill_messages_fts()`
(chunked, IMMEDIATE transaction per chunk, ascending-rowid cursor) driven by
`DB/chachanotes_fts_backfill.py` from a `run_worker(thread=True)` at app mount.
`_CURRENT_SCHEMA_VERSION` = 47 (collision sweep at commit time: 124 remote branches +
187 worktrees + origin/dev + origin/main -- no other claimant of v47).

**Why v47 is mandatory, not defensive**: with the v46-shaped trigger guard
(`old.deleted = 0` alone), a plain content UPDATE of a not-yet-backfilled live row
raises `sqlite3.DatabaseError: database disk image is malformed` ON THE UPDATE
(external-content FTS5 'delete' of an unindexed rowid). Reproduced empirically first,
then pinned by `test_the_v46_shaped_trigger_would_corrupt_during_the_window`, and
mutation-proven: stripping the docsize guard from the v47 SQL turns
`test_writes_during_the_backfill_window_do_not_corrupt_the_index` red. Side effect:
the previously UNGUARDED `messages_ad` (hard-deleting a tombstoned message corrupted
the index on shipped code -- latent, task-19567 only repaired the `*_au` family) is
fixed by the same guard.

**Timing probe** (macOS NVMe, `CharactersRAGDB(path)` construction on a genuine
v45-stamped fixture; seeds include 20% edits + 2% tombstones so sync_log and the
index are realistic; scale x20-50 for the slow-disk profiles in the finding):

| seed          | construction BEFORE | construction AFTER | WAL after construction | FTS work moved off boot |
|---------------|--------------------:|-------------------:|-----------------------:|------------------------:|
| 10k messages  | 0.094 s             | 0.055 s (-42%)     | 3.58 MB -> 2.59 MB     | 9,800 rows / 0.085 s    |
| 100k messages | 1.248 s             | 0.693 s (-44%)     | 37.0 MB -> 25.4 MB     | 98,000 rows / 1.103 s   |

docsize after construction: BEFORE = full (9,800 / 98,000) -- the rewrite was inside
the version-bump transaction; AFTER = 0 -- it is gone from the blocking path
entirely (AC #3). End-state equality: after the background backfill completes, the
docsize rowid-set SHA and four spot MATCH result sets are byte-identical to the
one-shot rebuild's (10k: sha 1f81494d970c057c; 100k: sha 5b32ab786b163ba1). The
remaining AFTER construction time is the 12-step DDL replay + the nine sync_log
purges, which are out of this task's scope by design (the AC targets the FTS wall).

**Interruption safety** (AC #1): the invariant is that "not yet indexed" is
membership in `messages_fts_docsize` -- state in the DATABASE, populated only by real
FTS writes -- and each chunk commits in its own IMMEDIATE transaction. Two tests:
stop-after-N-chunks + reopen (`test_backfill_resumes_after_interruption_and_matches_
the_one_shot_state`) and a real SIGKILL mid-backfill in a subprocess
(`test_backfill_survives_sigkill_mid_run`): reopen recovers via WAL, the version
stamp is intact (47), the resume indexes exactly the remainder, and the end state
equals the live-row set with structural FTS integrity-check clean. A failure INSIDE
the migration chain still rewinds atomically to v45 including the delete-all
(`test_a_failure_mid_v47_rewinds_the_whole_chain`) -- so a failed upgrade leaves the
OLD index serving search, and an interrupted one that committed leaves a resumable
backfill; neither bricks nor loses the bump.

**Search-window semantics (deliberate, AC-relevant disclosure)**: after the upgrade
commits, `messages_fts` is EMPTY-but-consistent and fills oldest-rowid-first in the
background; message-content search returns progressively more history until the
backfill completes (~1.1 s of background work per 100k messages on NVMe), never
errors, and never returns tombstoned rows. New/edited messages are indexed
immediately by the triggers regardless of progress. The alternative -- serving the
old index during the window -- was rejected because that index is exactly the
possibly-corrupt, tombstone-bearing artifact v46 exists to discard. A DB opened by
something that never runs the app driver (scripts, tests) keeps its resumable
frontier until the next app run; the driver's completion pass doubles as
self-healing on every boot (exemplar contract, `Subscriptions/fts_backfill.py`).

**AC #2 (visible state)**: the migration chain runs inside `TldwCli.__init__`,
BEFORE Textual `run()` -- no splash or screen exists at that phase, so the only
honest surface is the launch terminal. Both TUI entry points now call
`Utils/db_upgrade_notice.print_db_upgrade_notice_if_pending()` right before
constructing the app; against a real v45 fixture it prints "Upgrading database
(schema v45 -> v47)... the app will start when the upgrade completes." The probe is
read-only through the ADR-029 private-SQLite seam (new owner
`utils.db_upgrade_notice`, inventory row C51) and swallows every failure. AC #2's
"splash or equivalent" is satisfied by the terminal line as the equivalent-at-that-
phase; a richer in-app treatment would require restructuring when the DB initializes
relative to `run()` and is deliberately not attempted here.

**Population safety**: databases already stamped 46 by the original full-rebuild v46
never replay v46; v47 gives them only the trigger swap (DDL, O(1)) and the driver
finds nothing to do (`test_v47_leaves_a_complete_index_alone`). Both populations
converge on identical triggers and identical index contents -- no divergent end
states, one delivery code path.

**Files changed**: tldw_chatbook/DB/migrations/chachanotes_v45_to_v46_sync_log_
retention.sql (reinsert removed, rationale comment), tldw_chatbook/DB/migrations/
chachanotes_v46_to_v47_messages_fts_backfill_guards.sql (new), tldw_chatbook/DB/
ChaChaNotes_DB.py (version 47, `_migrate_from_v46_to_v47`, `backfill_messages_fts`,
steps map), tldw_chatbook/DB/chachanotes_fts_backfill.py (new driver),
tldw_chatbook/DB/private_sqlite.py (owner `utils.db_upgrade_notice`),
tldw_chatbook/Utils/db_upgrade_notice.py (new), tldw_chatbook/app.py (mount worker
`_backfill_chachanotes_messages_fts` + notice calls at both entry points),
pyproject.toml + MANIFEST.in + Packaging/check_manifest.py + Tests/Packaging/
test_installed_distribution.py (v45_to_v46 -- a pre-existing TASK-19860 gap on this
exact chain -- and v46_to_v47 added to every per-file packaging inventory; wheel
built and verified to ship both), backlog/docs/sqlite-private-owner-inventory.md
(C51), Tests/DB/test_chachanotes_v47_messages_fts_backfill.py (new, 11 tests, carries
the ==47 pin), Tests/Utils/test_db_upgrade_notice.py (new, 4 tests),
Tests/DB/test_chachanotes_sync_log_retention_migration.py (pin relaxed to >=46 per
the documented pin-move rule; FTS delivery test drives the backfill and additionally
pins the empty-window state), Tests/DB/test_private_sqlite_inventory.py (C-range 51;
C50 row selected by id instead of by position).

**Evidence discipline**: red-first proven by A/B (restoring the inline reinsert turns
`test_upgrade_from_v45_defers_the_messages_fts_reinsert` red); guard mutation-tested
(above). Test runs (test-logs/task21100-*): baseline 94 passed pre-change; final
core affected suites 436 passed; Tests/DB + Tests/ChaChaNotesDB 1529 passed/1
skipped with the only failure (`test_sql_validation::test_no_missing_tables`, actor
tables) reproduced VERBATIM on a pristine dev checkout at the same SHA (task-19057
residue, not this branch); RAG FTS + image-compat 64 passed after installing the
missing `numpy` dev dep; Character_Chat 69 passed; full `--collect-only` 56,154
collected with 9 errors, all 9 identical on pristine dev.

**Deliberately not fixed**: the pre-existing dev red in `tldw_chatbook/DB/
sql_validation.py` (actor_pack tables missing from VALID_TABLES, task-19057 residue);
the remaining ~0.7 s non-FTS migration replay cost at 100k (DDL + sync_log purges --
out of AC scope); TASK-19860's real fix (glob + built-artifact check) -- this task
only extends the existing per-file inventories with the two files on its own chain.
No new lessons-file entry: the traps hit here (per-file packaging inventory, ADR-029
connect census) are already documented by their own guards, which is how they were
caught.

### Review fix round (2026-08-22, adversarial review: FIX-FIRST on two Majors)

**Major 1 -- backfill chunks killed concurrent hot writers.** Confirmed by the
reviewer and reproduced red-first here: every hot ChaChaNotes message writer ran a
DEFERRED read-then-write transaction, and one backfill chunk committing inside the
read->write gap kills the writer with a NON-RETRYABLE `database is locked`
(snapshot-upgrade SQLITE_BUSY bypasses the 15 s busy timeout entirely; throttling
the backfill cannot help -- one commit in the gap is fatal). Fix per the
controller's decision: the hot `messages` writers now open
`self.transaction(immediate=True)` (the in-repo seq-assignment precedent), so the
write lock is reserved before the first read and the backfill chunk queues on the
busy timeout instead. Converted (scoped, not file-wide; each site carries a comment,
the full scoping rule at `add_message`): `add_message` (ChaChaNotes_DB.py:10138),
`create_assistant_with_continuation` (:10206), `append_message_attachment_with_
metadata` (:10618), `swap_message_attachment_with_scalar` (:10705), `update_message`
(:11015), `soft_delete_message` (:11657), `soft_delete_message_subtree` (:11729),
`create_message_variant` (:11896), `select_message_variant` (:12071).
`update_message_feedback` is covered by delegation to `update_message`. Deliberately
NOT converted (rule documented in the test's `HOT_MESSAGE_WRITERS` comment): the
blind single-statement writers (`update_message_usage_local`,
`update_message_metadata_local`, `append_message_exchanges_local`) -- with no read
before the write there is no snapshot to upgrade, and plain SQLITE_BUSY honors the
timeout. Regression tests: `test_hot_writer_survives_a_backfill_commit_inside_its_
transaction` (the reviewer's deterministic interleave through the production
`add_message`: read snapshot -> one real backfill chunk from a second connection ->
the write; RED on the pre-fix code with `sqlite3.OperationalError: database is
locked` at ChaChaNotes_DB.py:3335, GREEN after -- and it additionally asserts the
chunk was BLOCKED, proving the lock ordering) and `test_hot_message_writers_reserve_
the_write_lock_up_front` (structural census over the enumerated writers, also
red-first).

**Major 2 -- the `messages_ad` membership guard had no behavioural witness.**
Confirmed: mutating the guard away left the window test GREEN, because against a
partly-filled index the unguarded FTS 'delete' of an unindexed row corrupts
SILENTLY -- no raise, integrity-check(0) stays green, and the only observable is
`messages_fts_data` growing dangling delete-marker blocks (re-measured here:
(3,40)->(4,70); the raise I originally documented is index-STATE-dependent -- an
empty index raises on the statement, a partly-filled one absorbs the poison). Fix:
`_fts_data_footprint` (COUNT + SUM(LENGTH(block)) over `messages_fts_data`)
witnesses around BOTH hard-delete-of-unindexed cells in the window test -- the
un-backfilled-live case and a new hard-delete-of-TOMBSTONED case (the latent
pre-existing bug) -- asserting the physical storage does not move. Mutation-proven:
stripping the ad guard now reds the window test at the new witness ("the
messages_ad membership guard is not holding"); restored, 13/13 green. All five
comment sites claiming an unconditional malformed raise corrected to "silently
poisons the doclists (and can raise malformed depending on index state)": the v47
SQL header (two spots), `_migrate_from_v46_to_v47`'s docstring, the v45->v46 SQL
comment, and the test module docstring + standalone-repro docstring.

Fix-round test evidence (test-logs/task21100-fixround-*): interleave+census red
(2 failed) -> conversion -> green (2 passed); v47 file 13 passed; new/changed files
37 passed; Tests/ChaChaNotesDB + Tests/DB 1485 passed / 1 skipped with only the
known pre-existing dev red (sql_validation actor tables). The converted writers'
heavy consumers were swept whole: Tests/Character_Chat 917 passed / 1 failed --
that failure reproduces verbatim on pristine dev; Tests/Chat first half 83 failed /
3424 passed / 61 skipped / 11 errors, EXACTLY matching pristine dev at the same SHA
on all four counts; Tests/Chat second half's failures isolate to 19 files which,
run as one selection on BOTH trees, produce an identical 46 failed / 591 passed /
20 errors -- zero failures attributable to this branch. One file
(`Tests/Chat/test_fleet_teardown_notice.py`) hangs >420 s under a 60 s per-test
timeout on PRISTINE DEV as well as here (it is what killed the earlier whole-suite
background runs); excluded on both sides and left for its owner.

### Final round (2026-08-23, scoped re-review: two required fixes, merge pre-cleared on them)

**The twelfth writer.** `update_provider_continuation` (ChaChaNotes_DB.py, transaction
now at :10301) had slipped through my own enumeration -- a DEFERRED read-then-write
`messages` writer on the live Console flow (console_chat_store's
discard-interrupted-run and the continuation-checkpoint persist). Converted to
`transaction(immediate=True)` with the standard comment and added to
`HOT_MESSAGE_WRITERS` (it has exactly one transaction site, so the census assertions
stay precise); the census was red-first on the addition.

**Outer deferred wrappers neutralize inner IMMEDIATE.** `transaction(immediate=...)`
is honored only at depth 0, so an outer DEFERRED read-then-write wrapper silently
re-opens the exact snapshot-upgrade window Major 1 closed for the writers it nests.
Reproduced red-first through the REAL composition
(`test_nested_writer_composition_survives_a_backfill_commit`: `ChatPersistenceService
.create_message` with an authoritative attachments list -> outer deferred transaction
-> nested `add_message` read -> one backfill chunk commit -> instant
`sqlite3.OperationalError: database is locked`; the depth-0 interleave test cannot
catch this class). Converted the five outer read-then-write wrappers, each with a
comment citing the nesting rule: `Chat/chat_persistence_service.py:799`
(update_message_content, citations branch), `:834` (attachments branch), `:1148`
(create_message, citations branch), `:1198` (create_message,
attachments/generation-metadata branch), and `Chat/Chat_Functions.py:2469` (the
resave path wrapping `save_history`). Green after conversion, with the nested test
also asserting the injected chunk is BLOCKED (lock ordering) and the backfill then
completes losslessly.

Final-round evidence (test-logs/task21100-finalround-*): nested test + census red
(2 failed, `database is locked`) -> six conversions -> v47 file 14 passed;
direct-subject files (test_chat_persistence_service, test_console_provider_
continuation, test_provider_continuation_crash_recovery,
test_provider_continuation_privacy, test_console_terminal_citation_persistence)
231 passed; test_chat_functions 98 passed; Tests/ChaChaNotesDB + retention +
witness 338 passed. Zero new failures -- no dev A/B needed this round.
<!-- SECTION:NOTES:END -->
