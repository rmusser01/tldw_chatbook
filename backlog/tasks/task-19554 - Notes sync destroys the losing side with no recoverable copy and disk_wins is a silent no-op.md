---
id: TASK-19554
title: >-
  Notes sync destroys the losing side with no recoverable copy, and disk_wins is
  a silent no-op
status: To Do
assignee: []
created_date: '2026-08-21 20:04'
labels:
  - notes
  - sync
  - data-loss
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 3 (data layer & schema integrity) —
its **F4**, CONFIRMED as LIVE DATA LOSS; the companion `DISK_WINS` bug also
converges with Lane 6's "the app asserts outcomes it did not produce" theme.
Re-verified at this branch base.

**Part A — unrecoverable overwrite.** `NEWER_WINS` conflict resolution
(`tldw_chatbook/Notes/sync_engine.py:1034`) overwrites the losing side
wholesale with **no recoverable copy**, unattended, on a 300-second cycle.
The `SyncConflict` object *carries* both `db_content` and `disk_content`, but
`_record_conflict` (`sync_engine.py:416-436`) persists only **hashes** —
`db_content_hash`, `disk_content_hash` — and the `sync_conflicts` table has no
content columns at all. There is no `.bak`, no history table, no way to get the
discarded version back.

Aggravating: the Library UI deliberately excludes the `ASK` strategy, so the
only strategies offered to the user are the destructive ones.

**Part B — `disk_wins` is a lie.** `DISK_WINS` appears **exactly once** in
`sync_engine.py` — line 43, its own enum definition. There is no branch that
implements it. Selecting "disk wins" in the UI records the conflict and applies
**nothing**, and the run reports as synced. The user believes they chose the
disk copy; the database copy silently remains.

These are filed together because they share a locus and a fix conversation: the
conflict path needs to preserve what it discards, and it needs to actually do
what the selected strategy says.

## Acceptance Criteria

- [ ] No conflict resolution strategy destroys content without a recoverable
      copy of the losing side (persisted content, a sidecar file, or a history
      row — chosen for durability, not cleverness)
- [ ] `_record_conflict` persists enough to reconstruct the discarded version,
      not just its hash
- [ ] `DISK_WINS` either applies the disk copy as its name states, or is
      removed from the enum and from every UI surface that offers it — it must
      not remain selectable while doing nothing
- [ ] A sync run that applied no changes never reports as though it resolved
      the conflict
- [ ] Tests cover each offered strategy end-to-end: the winning side is
      applied, the losing side is recoverable, and the reported outcome matches
      what actually happened
- [ ] The decision on whether to re-offer `ASK` in the Library UI is made
      explicitly and recorded, given that excluding it currently leaves only
      destructive options
- [ ] `Docs/User_Guide/` states plainly what each strategy does to the losing
      copy
