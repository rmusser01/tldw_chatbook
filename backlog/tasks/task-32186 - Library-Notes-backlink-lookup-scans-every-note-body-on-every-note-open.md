---
id: TASK-32186
title: 'Library Notes: backlink lookup scans every note body on every note open'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-09 12:30'
updated_date: '2026-09-11 16:47'
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
- [ ] #1 Opening a note in a vault of several thousand notes fills "Linked
  from" without reading every note body, measured against the current
  full-scan query on the same corpus.
- [ ] #2 The rows shown are the same rows the current containment query
  returns — the exact `(note://<id>)` link form, soft-deleted notes and the
  target itself excluded, ordered by title — for imported and hand-typed
  links alike.
- [ ] #3 Links created, changed, and removed by an edit, an import, and a
  deletion are all reflected the next time the linked-to note's Info panel is
  opened.
- [ ] #4 An existing database picks up its backlinks without the user
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
