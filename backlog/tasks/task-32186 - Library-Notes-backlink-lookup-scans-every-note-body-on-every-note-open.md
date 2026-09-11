---
id: TASK-32186
title: 'Library Notes: backlink lookup scans every note body on every note open'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-09 12:30'
updated_date: '2026-09-11 17:21'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Qodo review of PR #2552 (task-32145, "Linked from" in Note
Info). The backlink lookup answers "which notes link to this one" with a
leading-wildcard `LIKE` over the unindexed `notes.content` column
(`CharactersRAGDB.get_notes_linking_to`), and the Info panel starts that
query every time a note opens. No index can serve `%(note://<id>)%`, and the
`ORDER BY title` means the row limit does not bound the work — every active
note body is read and sorted before the first 51 rows come back.

On the vault sizes this program has tested (tens to low hundreds of notes)
the query is not measurable next to the note load it deliberately runs
beside, and it is off the critical path: its own worker, in its own thread,
and a failure leaves only the Info panel's "Linked from" line unanswered. It
becomes a real cost at vault scale, and the fix is a different shape from the
feature — a persisted source→target link relation, written where wikilinks
are already parsed (`note_import_plan_models.rewrite_wikilinks` for imports,
plus the save path for hand-typed links), read by an indexed lookup on the
target id. That is a schema migration and a backfill, not a query rewrite,
which is why it is not part of the feature PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening a note in a vault of several thousand notes fills "Linked
  from" without reading every note body, measured against the current
  full-scan query on the same corpus.
- [x] #2 The rows shown are the same rows the current containment query
  returns — the exact `(note://<id>)` link form, soft-deleted notes and the
  target itself excluded, ordered by title — for imported and hand-typed
  links alike.
- [x] #3 Links created, changed, and removed by an edit, an import, and a
  deletion are all reflected the next time the linked-to note's Info panel is
  opened.
- [x] #4 An existing database picks up its backlinks without the user
  re-importing anything.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: time get_notes_linking_to over 1k/3k/10k-note corpora (the LIKE scan is linear in corpus size).
2. Schema v72 -> v73: note_links(source_note_id, target_note_id) with an index on (target_note_id, source_note_id); migration backfills from existing note bodies.
3. One extractor (the (note://<id>) tail, which survives the [[t|title]](note://id) syntax change) + one maintenance helper, called from every note-content writer: _add_note_with_cursor / _update_note_with_cursor in ChaChaNotes_DB and _insert_note / _update_note in note_import_executor.
4. get_notes_linking_to joins note_links to notes (deleted = 0, id != target, ORDER BY title) so soft-deleted linkers drop out and restore brings them back.
5. RED->GREEN tests: migration + backfill on a v72 DB, edit/import/delete/restore round trips, and a query-plan budget proving the corpus is not scanned.
6. Re-measure, update VALID_TABLES + index census, guide note, derived-artifact checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Schema **v72 -> v73** adds `note_links(source_note_id, target_note_id)` —
composite primary key, `source_note_id` a FK to `notes` with ON DELETE
CASCADE, plus `idx_note_links_target(target_note_id, source_note_id)` — and
backfills it from the bodies an existing database already holds (AC #4).
`get_notes_linking_to` joins the relation to `notes`, so its cost is the
number of notes that link HERE rather than the size of the vault.

Where the relation is maintained: one extractor
(`extract_note_link_targets`, the `(note://<id>)` tail — the half that
survived task-32129's `[[target|title]]` change) and one writer helper
(`CharactersRAGDB.replace_note_links`), called from all four writers of
`notes.content`: `_add_note_with_cursor`, `_update_note_with_cursor`, and the
import target's `_insert_note`/`_update_note`, which write their own SQL for
version and client-id reasons and would otherwise have left imported
wikilinks — the links the feature exists for — invisible.

Decisions worth knowing:

* **No sync columns, no sync-log triggers.** Derived data, rebuildable from
  `notes.content` (the migration backfills it exactly that way). Replicating
  the edges would replicate the same fact twice and invite disagreement.
* **`target_note_id` is not a foreign key.** An import inserts sources and
  targets in one transaction in no guaranteed order, and a body may link to a
  note that no longer exists; an immediate FK check would reject the first and
  drop the second. Reads join to `notes`, so a dangling target is invisible.
* **Soft delete needs no trigger.** A deleted linker drops out through the
  join and returns with its restore, because its edges are never removed.
* **The backfill runs in Python** (SQLite has no regex), streamed on a second
  cursor in batches of 500 so a large vault is never fully in memory.
* **Ceiling:** a note id containing `)` is not representable in the link form
  (the extractor stops at the first one). Ids are UUIDs or validated opaque
  import ids, so this is unreachable in practice; documented at the regex.

Test scaffold: `historical_bootstrap` now creates a bare `note_links` for a
pre-v73 target, because fixtures seed historical databases through TODAY's
`add_note` (production never pairs a v72 database with v73 code — migration
runs in `__init__`). The migration test drops that scaffold before reopening,
so the step is genuinely observed creating and backfilling.

RED -> GREEN: `Tests/DB/test_chachanotes_v73_note_links_migration.py` (3, all
RED on "no such table: note_links"),
`Tests/Notes/test_note_backlink_query.py::
test_the_lookup_does_not_scan_the_note_corpus` (RED: no `_BACKLINK_SOURCES_SQL`),
`Tests/Notes/test_note_import_executor.py::
test_imported_bodies_maintain_the_note_link_relation` (RED: no such table).
The edit and Trash-restore round trips in `test_note_backlink_query.py` pass
before and after by design — they pin that the new implementation keeps
behaviour the scan already had. GREEN 13/13.

The budget is a query plan, not a wall clock (a timing assertion is a flake on
shared CI): the lookup must search `idx_note_links_target` and must never
`SCAN notes`.

Measured (`scratchpad/wave3-caps/backlinks-table/bench-{before,after}.txt`),
throwaway vaults of 1,000 / 3,000 / 10,000 notes, median of 20:

| corpus | before | after |
|---|---|---|
| 1,000 | 0.85 ms | 0.08 ms |
| 3,000 | 3.07 ms | 0.08 ms |
| 10,000 | 8.79 ms | 0.08 ms |

Before grows linearly with the vault; after is flat, bounded by inbound links.
On the brief's seeded profile (51 notes, a 35 KB hub, 40 linkers) it is
0.200 ms -> 0.102 ms — both far below anything a reader can see, which is what
this task's own Description predicted and why the fix is about vault scale.

Live: the dev-built profile was copied and opened with this branch; it
migrated v72 -> v73 on open and Info ▸ Linked from read "Linked from (40)"
with the Slip rows listed, no re-import
(`scratchpad/wave3-caps/backlinks-table/live-info-linked-from-40.txt`).

Also fixed here, as originally assigned: **task-32458**, the app-killing
`NoMatches` from this worker's paint. It was briefly reassigned to the
test-health group and reverted here, then handed back when their
controller-seam guard was found to skip work the seam owns; see that task's
notes for why the guard belongs in `apply_session_state`.

Two regressions the full run caught, both fixed here and worth knowing:
`sqlite3.Cursor.execute` returns the cursor itself, so the link writes added
inside the importer's `_update_note` overwrote the `result.rowcount` its return
value read (18 tests; lesson filed); and three migration tests asserted the
post-open schema version as the literal `72` — "the version current when I was
written" rather than "the chain completed" — which every bump breaks, now the
constant.

Modified: `tldw_chatbook/DB/ChaChaNotes_DB.py`,
`tldw_chatbook/DB/migrations/chachanotes_v72_to_v73_note_links.sql` (new),
`tldw_chatbook/DB/sql_validation.py`,
`tldw_chatbook/Notes/note_import_executor.py`,
`tldw_chatbook/Widgets/Library/library_notes_canvas.py`,
`Tests/ChaChaNotesDB/historical_bootstrap.py`,
`Tests/ChaChaNotesDB/test_index_census.py`,
`Tests/DB/test_chachanotes_v73_note_links_migration.py` (new),
`Tests/Notes/test_note_backlink_query.py`,
`Tests/Notes/test_note_import_executor.py`,
`Tests/DB/test_conversation_archive.py`,
`Tests/DB/test_chachanotes_v72_voice_trace_provenance_migration.py`,
`Tests/UI/test_library_notes_riders_backlinks.py`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
