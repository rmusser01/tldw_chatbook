---
id: TASK-16840
title: 'Replace the ChaChaNotes rollback registry with bootstrap-under-patched-schema-version fixtures'
status: Done
assignee: ['@claude']
created_date: '2026-08-16'
labels:
  - test-health
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the TASK-15765 review (PR #1695, F3): a knowledge-free alternative to the whole
`Tests/ChaChaNotesDB/schema_rollback.py` registry already exists in this repo —
`Tests/DB/test_chachanotes_note_folders_migration.py:31-38` bootstraps a genuinely
vN-shaped DB by patching `_CURRENT_SCHEMA_VERSION` to N and running the **real**
migration chain. The review verified for v16/v17/v34 that this yields true historical
schemas (sync triggers present, zero future tables/columns) and replays to current with
full object parity — with **zero hand-maintained per-version knowledge**, no ratchet, no
sweep, immune by construction to the v20..v27 trigger-loss class the registry's sweep
exists to catch.

The registry's costs are compounding, as predicted: at dev `ee741cf10` it has already
grown hand-written entries for v38 and v39 (schema is now 39,
`DB/ChaChaNotes_DB.py:247`), each a new chance for the class of error the guard only
partially sees. The review's F1 proved the parity sweep is **blind to column loss**
(a seeded `DROP COLUMN` mutation left all 22 replay targets green while four replayed to
a DB permanently missing a production column — columns are not sqlite_master rows), F2
documented three comments falsely describing the fixtures as historical, and F4 noted
column-order divergence for v16..v29 targets.

Migrate the three registry consumers to the bootstrap-under-patched-version primitive and
delete the registry + ratchet + sweep, or — if the registry is deliberately kept —
close F1 (per-table column-**set** comparison in `_schema_objects`, sets not tuples per
F4) and fix the F2 comments. Replacement is the durable end state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Migration-fixture tests obtain vN-shaped DBs without any hand-maintained per-version rollback knowledge (or, explicitly declined, with F1/F2/F4 closed instead)
- [x] #2 The trigger-loss and column-loss error classes are both impossible-by-construction or guarded with a mutation-tested oracle
- [x] #3 All current consumers (`test_chachanotes_db.py` v17, local-marks, dictionary-backfill fixtures) stay green and still pin what they pinned
- [x] #4 A version bump no longer demands a new rollback entry (no ratchet debt), or the remaining debt is documented at the registry
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe the bootstrap-under-patched-version primitive for v16/v17/v34: shape
   (sync triggers present, future artifacts absent), replay-to-current parity
   vs a fresh bootstrap, and timing vs the registry path. (DONE — parity
   IDENTICAL at all three; bootstrap-at-vN is FASTER than
   bootstrap-current+rollback. One drift found: the v4 base schema bakes in
   `conversation_local_marks`, so the marks fixture must drop its OWN
   migration-under-test artifacts — local knowledge, not registry knowledge.)
2. Baseline the five affected suites vs the branch base to a file.
3. Add `Tests/ChaChaNotesDB/historical_bootstrap.py` with a
   `chachanotes_db_at_version(path, version)` contextmanager (patch
   `_CURRENT_SCHEMA_VERSION`, bootstrap via the real chain, yield the open DB).
4. Convert the three registry consumers (marks v16, chachanotes v17,
   dictionary v34) to bootstrap-at-vN; adapt/strengthen the AC2 precondition
   assertions to the now-genuine vN shape (v17: sync triggers PRESENT and
   system_prompt-free, where the registry fixture had to assert them absent;
   v34: attachment metadata genuinely written with no trigger/index machinery).
   Point the note-folders exemplar's `_seed_v35` at the shared helper.
5. Replace `test_schema_rollback.py` with a bootstrap-at-vN -> replay ->
   parity sweep (keep the (type,name)+column-set oracle depth); delete
   `schema_rollback.py` (registry) and the completeness ratchet.
6. Mutation-test (Edit-based restores): seed MUT-A (emptied migration step)
   and MUT-B (column dropped mid-chain) and record honestly WHICH guard reddens
   (parity-vs-fresh is expected blind to same-chain mutations — both sides run
   the mutated chain; the consumers/feature tests are the catching oracle) plus
   a stamp-defect mutation that reddens the sweep itself.
7. Measure suite delta (bootstrap-per-fixture vs registry), state numbers.
8. ruff on touched files; full affected-suite run; lessons entry extension
   (close the 15730 -> 15765 -> 16197 -> 16840 arc with the final shape);
   Implementation Notes; Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the registry (F3's "durable end state" branch). Baseline at branch
base `cef56efaf`: 138 passed across the five affected suites.

**What died:** `Tests/ChaChaNotesDB/schema_rollback.py` (the per-version
removal registry, incl. its fresh v38/v39 entries) and the completeness
ratchet in `test_schema_rollback.py`. **What survived, transformed:** the
sweep, reborn in `Tests/ChaChaNotesDB/test_historical_bootstrap.py` —
bootstrap a genuinely vN DB under a patched `_CURRENT_SCHEMA_VERSION`
(new shared primitive `Tests/ChaChaNotesDB/historical_bootstrap.py::
chachanotes_db_at_version`, the idiom promoted from
`test_chachanotes_note_folders_migration.py`, whose `_seed_v35` now uses
it), reopen so the production chain replays to current, compare with a
straight fresh bootstrap at the review's oracle depth ((type,name) +
per-table column SETS) — now over v4..v38 (35 versions) instead of the
registry's v16..v38 floor.

**Converted consumers (all still pin what they pinned, several stronger):**
- marks v16 (`Tests/Chat/test_conversation_local_marks_service.py`):
  bootstrap-at-16. Found drift: the "v4" base schema bakes in
  `conversation_local_marks`, so the fixture drops its OWN
  migration-under-test artifacts (table+index) before replay —
  single-migration knowledge no future bump can invalidate. Mutation-proved:
  emptying V16→V17 reds exactly this fixture (and NOT the fresh-schema test,
  which the base's baked copy keeps green — this fixture is that migration's
  only guard).
- chachanotes v17 (`Tests/ChaChaNotesDB/test_chachanotes_db.py`): STRONGER
  preconditions — the four `conversations_sync_*` triggers now assert
  PRESENT in their real pre-V18, system_prompt-free form (the registry
  fixture had to assert them absent), so replay exercises the migration's
  genuine redefine-live-triggers path.
- dictionary v34 (`Tests/Character_Chat/test_dictionary_attachment_index.py`):
  STRONGER — seeds attachment metadata while the derived tables/triggers
  genuinely do not exist (asserted), i.e. the migration's actual premise;
  the registry fixture seeded at CURRENT version through live triggers and
  dropped the result.

**AC2 (mutation evidence, Edit-based restores, all restored):**
- Trigger-loss + column-loss REGISTRY-fidelity classes: dead with the
  registry (no hand-written removal exists to be wrong).
- MUT-A (emptied V35→V36 SQL, bump kept): sweep 35/35 GREEN, 9 note-folder
  consumer tests RED by name. MUT-A' (emptied V16→V17): exactly the
  converted marks fixture RED. MUT-B (`DROP COLUMN messages.usage_json`
  seeded into V37→V38): sweep GREEN, 7 usage_json consumer tests RED
  ("table messages has no column named usage_json").
  Honest refutation of the filing's expectation: a parity oracle derived
  from the chain is the IDENTITY on deterministic chain mutations — the old
  sweep caught these shapes only because the registry was a divergent second
  copy. Consumer/feature tests are the artifact-correctness oracle; the
  sweep's documented job is the genuine upgrade matrix (resume from every
  vN, stamp/dispatch wiring, stop-resume vs straight-through parity).
- MUT-D (unwired `migration_steps[38]` — the ratchet's old target): all 35
  sweep cases red with "Migration path undefined" — the chain self-detects;
  AC4 needs no bespoke guard.

**Perf (probe + suite):** bootstrap-at-vN ~80-130ms vs registry's
bootstrap-current ~220-255ms + rollback + replay — per-fixture cost DOWN;
the registry was never a perf win. Five-suite set: 138 tests/17.56s
baseline → 149 tests/22.70s cold, 11.93s warm (the +11 tests are the 12
extra sweep versions minus the ratchet); 35-case sweep alone 8.31s cold /
5.36s warm. No caching mitigation needed.

**Verification:** final run 149 passed; `--collect-only` over
Tests/ChaChaNotesDB + Tests/DB + Tests/Chat + Tests/Character_Chat: 7,855
collected, zero errors; ruff check/format clean on touched files. Lessons
entry extended in `backlog/docs/lessons-testing-evidence.md` (final shape
of the 15730→15765→16197→16840 arc). Production code untouched (diff is
Tests/ + backlog/ only).
<!-- SECTION:NOTES:END -->
