---
id: TASK-19554
title: >-
  Notes sync destroys the losing side with no recoverable copy, and disk_wins is
  a silent no-op
status: Done
assignee:
  - '@claude'
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

- [x] No conflict resolution strategy destroys content without a recoverable
      copy of the losing side (persisted content, a sidecar file, or a history
      row — chosen for durability, not cleverness)
- [x] `_record_conflict` persists enough to reconstruct the discarded version,
      not just its hash
- [x] `DISK_WINS` either applies the disk copy as its name states, or is
      removed from the enum and from every UI surface that offers it — it must
      not remain selectable while doing nothing
- [x] A sync run that applied no changes never reports as though it resolved
      the conflict
- [x] Tests cover each offered strategy end-to-end: the winning side is
      applied, the losing side is recoverable, and the reported outcome matches
      what actually happened
- [x] The decision on whether to re-offer `ASK` in the Library UI is made
      explicitly and recorded, given that excluding it currently leaves only
      destructive options
- [x] `Docs/User_Guide/` states plainly what each strategy does to the losing
      copy

## Implementation Plan

1. **Born-red first.** `Tests/Notes/test_sync_conflict_preservation.py` — one
   pin per defect (NEWER_WINS loses the disk copy; `disk_wins` applies
   nothing), plus mutation controls (preserving the *winner* must not pass),
   run at base and recorded.
2. **Preservation, two surfaces, fail-closed.**
   - Primary, user-visible: a byte-exact sidecar written through the existing
     `PinnedSyncRoot` boundary next to the note file,
     `<name>.conflict-<UTC>-<side>.bak`. Recovery is a rename.
   - Second copy: `sync_conflicts` gains `losing_side` / `losing_content` /
     `preserved_file_path` (schema v43→v44, new-style atomic step per
     task-19553). Written only when a side is actually discarded, so the
     content shadow is bounded by real destruction events rather than by
     conflict *detection*.
   - If the sidecar cannot be written, the overwrite does **not** happen: the
     conflict is left unresolved and an error is recorded.
3. **Sidecars must not feed back.** Filter the sidecar marker out of
   `_scan_directory` so a preserved copy can never be re-ingested as a note
   (belt and braces on top of the non-scanned `.bak` suffix).
4. **Implement every offered strategy** on `both_changed`, in all three
   directions: `DB_WINS`, `DISK_WINS`, `NEWER_WINS`. Add conflict detection to
   `_sync_disk_to_db`, which had none and overwrote note bodies silently.
5. **Honest reporting.** `SyncConflict` gains `resolution` / `applied` /
   `preserved_path`; the row's `resolution`+`resolved_at` are stamped with what
   actually happened; the Library sync panel counts applied vs unresolved and
   names the sidecar.
6. Docs: `Docs/User_Guide/library/notes.md` +
   `Docs/Features/notes_bidirectional_sync.md` state per strategy what happens
   to the losing copy and where it goes.
7. Record the `ASK` re-offer question as an owner call, do not change it.

## Implementation Notes

Both defects fixed at the engine, plus a third of the same family the audit
did not name (see "Also found"). Two born-red pins, run and recorded at base
before any code changed:

* `test_newer_wins_db_newer_preserves_the_losing_disk_copy` →
  `assert 'Modified on disk' in []` — the overwritten file's text existed
  nowhere afterwards.
* `test_disk_wins_actually_applies_the_disk_copy` →
  `assert 'Modified in database' == 'Modified on disk'` — the note kept its
  database body.

13 of 13 red at base, 13 of 13 green after.

### Preservation design (and what was rejected)

**Shipped: sidecar first, DB row second, fail-closed.**

Before any resolution overwrites a side, that side's text is written verbatim
to `<file name>.conflict-<UTC>-<db|disk>.bak` next to the note file, through
the existing `PinnedSyncRoot` boundary (so a swapped root or a symlinked path
fails closed here exactly as it does for a note write). Then — and only then —
the winner is applied and the conflict row is stamped with `losing_side`,
`losing_content`, `preserved_file_path`, `resolution`, `resolved_at`.

*Why the sidecar is primary.* It is the only surface a user can already reach
with no new UI: it appears in the folder they are already syncing, sorts
directly beside the file it came from, says in its name which side and when,
and recovering it is a `mv`. It is deliberately byte-exact — no header, no diff
markers — because anything prepended turns recovery from a rename into an edit.

*Why the DB row as well* (AC #2 asks `_record_conflict` to persist enough to
reconstruct, not just a hash). The sidecar lives in a folder the user cleans
up, moves, or points a different tool at; the row survives that. The row is
also the only copy reachable programmatically. Cost is bounded by writing it
**only when a side is actually discarded**, never on mere detection: an `ask`
run, or a strategy that declines to apply, stores no content at all. That
matters because `sync_log` is already an unpruned full-content shadow of
`notes` and a second unconditional one would be a real regression.

*What surfaces it.* Three things, and no more — a conflict-review UI is an
owner question, not shipped:
1. The Library sync panel's activity log names the sidecar file
   ("Replaced copy saved as note.md.conflict-…-disk.bak").
2. `NotesSyncService.get_conflicts_for_session()` now returns the three
   columns, and a new `get_preserved_conflict_copies(limit)` lists every
   preserved copy across all sessions newest-first — the entry point that does
   NOT need a session id, because after an unattended 300-second auto-sync a
   user knows only that a note changed under them.
3. `Docs/User_Guide/library/notes.md` tells the user the files exist, what they
   are named, how to restore one, and that sync ignores them.

*Rejected — DB columns only.* This is the "preserved copy nothing can reach"
the brief warns about: nothing in the app renders `sync_conflicts`, so the
recovery story would have been "write a SQL query". It also stores content
where the user cannot see that anything was saved.

*Rejected — sidecar only.* Fails AC #2 literally, and ties the only copy to a
directory the user is actively managing (and, in the `disk_to_db` direction,
one they may treat as a scratch import folder).

*Rejected — a `.bak` of the whole file, or a notes-history table.* A history
table is the right long-term answer for note versioning generally, but it is a
much larger change than this conflict path and would be a schema and UI
programme of its own; scoping it here would have delayed a live P0.

*Fail-closed is the load-bearing part.* If the sidecar cannot be written the
overwrite does not happen: the conflict is left open, an error is recorded, and
both copies survive. Note that a read-only or otherwise unwritable root does
NOT stop `disk_wins`/`newer_wins` from destroying the note in the database —
that path only needs the DB — so without this rule preservation would fail
exactly where destruction still succeeded.

*Feedback loop closed.* Sidecars are dropped from every scan by the
`.conflict-` marker (not merely by the `.bak` suffix being outside the default
`['.md', '.txt']` set), so a preserved copy can never be ingested as a note,
which would otherwise turn each conflict into a duplicate note and then into
another file. Pinned by
`test_sidecars_are_never_re_ingested_as_notes`.

### `DISK_WINS`: implemented, not removed

Implemented. Removing it would have broken persisted user preferences
(`sync_conflict_resolution = "disk_wins"`), the `sync_sessions` CHECK
constraint that already lists it, and the engine's public enum — while
deleting a policy that is genuinely wanted (the file on disk is the
editor-of-record for anyone using an external editor). It now applies the disk
copy to the note, preserving the note body first.

`_resolve_both_changed` is one implementation shared by all three directions;
the direction only restricts which side may be written. A policy whose winner
cannot be written in the chosen direction (e.g. "Disk wins" during a
Library → Disk push) applies nothing and leaves the conflict OPEN — it does not
claim to have settled anything.

### Also found and fixed (same defect family)

* **`DB_WINS` was equally a no-op in bidirectional** — the `both_changed`
  branch there tested only `NEWER_WINS`. The audit named `DISK_WINS` because
  it has no branch anywhere; `DB_WINS` had one in `db_to_disk` only.
* **`NEWER_WINS` was a no-op in `db_to_disk`** even when the note was the newer
  side.
* **`disk_to_db` had no conflict detection at all.** "The file changed since
  the baseline" was treated as sufficient reason to overwrite the note, without
  ever asking whether the note had changed too — silent loss with not even a
  conflict row. It now detects, preserves, and resolves like the other two
  directions.
* **The panel called every recorded conflict "resolved"** (AC #4). It now
  counts `SyncConflict.applied`, names the preserved copy, and says
  "N conflicts left unresolved — both copies kept as they are" otherwise.
  The existing UI test's fake was tightened from the bare string `"c-1"` to a
  real `SyncConflict`, because a string cannot distinguish the two.

`deleted_on_disk` conflicts were left behaving exactly as before (auto-unlink
in `disk_to_db`, file recreation under `DB_WINS` in bidirectional, nothing
otherwise) — nothing is destroyed on that path, so it is not a data-loss
question. Only the bookkeeping changed, so the run cannot over-claim. Whether
bidirectional should also unlink under `disk_wins` is an open product question
(below).

### Schema

`_CURRENT_SCHEMA_VERSION` 43 → 44, `_migrate_from_v43_to_v44` +
`migrations/chachanotes_v43_to_v44_sync_conflict_preservation.sql`
(three `ADD COLUMN`s, DDL-only, rowcount-guarded version bump in the runner).
New-style per task-19553: entry-version guard, statements run through
`_execute_migration_statements` inside the step's transaction, so a half-applied
run is re-enterable (its already-applied `ADD COLUMN` skip) and a failure rolls
back. No new indexes, so the index census literal is unchanged.

The repo's one EXACT current-schema-version pin moved from
`Tests/DB/test_chachanotes_message_exchanges.py::test_schema_version_is_43`
(which owns v42→v43 only) to the newest migration's own file. Older migration
files now assert `>= their own version`, so a future bump touches the file that
caused it instead of an unrelated older one.

### Owner questions

1. **Should the Library panel re-offer `ASK`?** Not changed here — recording it
   as asked, per the task. The argument for: with `ASK` excluded, every policy
   the panel offers is one that overwrites something, on a 300-second unattended
   timer. The argument against, and why it is not a one-line revert: `ASK` is
   only meaningful if something *asks*, and nothing in the Library prompts on a
   conflict. Re-offering it today would produce a policy that records conflicts
   and never resolves them, with no surface to resolve them from — the same
   "selectable but does nothing" shape this task just removed from `DISK_WINS`.
   Making `ASK` real means building a conflict-review surface over
   `get_preserved_conflict_copies()` / `get_conflicts_for_session()`. That is a
   product decision and a separate task. What ships now is the mitigation that
   does not need it: the destructive policies are no longer destructive without
   a recoverable copy.
2. **Should bidirectional `disk_wins` unlink a note whose file was deleted?**
   `disk_to_db` already does this for every non-`ASK` policy; bidirectional does
   nothing. Aligning them is a behaviour change ("the file is gone" could mean
   the user deleted it, or that a sync partner has not created it yet), so it is
   left alone and reported honestly as unresolved rather than changed
   unilaterally.

### Files

* `tldw_chatbook/Notes/sync_engine.py` — preservation, strategy dispatch,
  sidecar write/filter, outcome bookkeeping.
* `tldw_chatbook/Notes/sync_service.py` — preservation columns in the session
  reader, new `get_preserved_conflict_copies`.
* `tldw_chatbook/DB/ChaChaNotes_DB.py` +
  `tldw_chatbook/DB/migrations/chachanotes_v43_to_v44_sync_conflict_preservation.sql`
* `tldw_chatbook/UI/Screens/library_screen.py` — honest conflict reporting.
* `Tests/Notes/test_sync_conflict_preservation.py` (new, 13 tests),
  `Tests/DB/test_chachanotes_sync_conflict_preservation_migration.py` (new,
  5 tests), `Tests/UI/test_library_shell.py` (+1 test, 1 fake tightened),
  `Tests/Notes/test_sync_engine.py` (hollow fixture repaired — see the lesson),
  `Tests/DB/test_chachanotes_message_exchanges.py`,
  `Tests/DB/test_chachanotes_console_project_context_migration.py`.
* `Docs/Features/notes_bidirectional_sync.md`,
  `Docs/User_Guide/library/notes.md`,
  `backlog/docs/lessons-testing-evidence.md`.
