---
id: TASK-16840
title: 'Replace the ChaChaNotes rollback registry with bootstrap-under-patched-schema-version fixtures'
status: To Do
assignee: []
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
- [ ] #1 Migration-fixture tests obtain vN-shaped DBs without any hand-maintained per-version rollback knowledge (or, explicitly declined, with F1/F2/F4 closed instead)
- [ ] #2 The trigger-loss and column-loss error classes are both impossible-by-construction or guarded with a mutation-tested oracle
- [ ] #3 All current consumers (`test_chachanotes_db.py` v17, local-marks, dictionary-backfill fixtures) stay green and still pin what they pinned
- [ ] #4 A version bump no longer demands a new rollback entry (no ratchet debt), or the remaining debt is documented at the registry
<!-- AC:END -->
