---
id: TASK-32263
title: >-
  Library Notes Import once rewrites wikilinks into note-uuid identifiers inside
  the user's own prose -- design decision to revisit
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 16:12'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
  - design-decision
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed as a design decision to revisit; task-32129 specified the rewrite, so this is not a defect against the shipped spec.

The importer rewrites `[[wikilinks]]` to `note://df8a9c4b-...`, putting 36-character machine identifiers inside sentences the user wrote. No other tool -- Obsidian included -- can read them back, and Export Markdown then produces a file that is no longer portable, which is the opposite of the local-first promise the same screen makes in copy.

Alternative worth deciding on: keep `[[wikilinks]]` verbatim in the body and resolve them at render time against the imported batch. Same navigation, none of the lock-in, and it removes the UUID scar from the user's prose.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user decision is recorded on rewrite-at-import versus resolve-at-render
- [x] #2 If revised: an imported note's body round-trips through Export Markdown with its links intact and readable by Obsidian
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the ruling: display-text links, [[Title]](note://uuid), rendered as the title in Preview.\n2. rewrite_wikilinks emits the wikilink form so the importer's own parser round-trips it and Obsidian reads the link; the (note://id) tail keeps get_notes_linking_to working.\n3. WIKILINK_SCAN swallows an existing (note://id) tail so a re-import cannot accumulate them.\n4. Preview renders the stored form as the title.\n5. RED/GREEN tests on the rewrite, the round-trip and the Preview transform.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DECISION (recorded per AC#1, taken by the user): revise -- rewrite at import, but as a display-text link. A resolved `[[wikilink]]` is now stored as `[[target|title]](note://<id>)`; Preview renders it as its display text.

Why that one form. It is the only spelling that satisfies all four constraints at once: the reader sees the linked notes TITLE rather than a 36-character identifier; `note_import_parsers._wikilinks` reads the stored body back, so an export/re-import round-trips; the exported file is still a working Obsidian wikilink; and `ChaChaNotes_DB.get_notes_linking_to` matches on the literal `(note://<id>)` token, which the tail preserves -- dropping it would have emptied every notes Linked from panel. Resolve-at-render was rejected on that last point: the backlink query is a containment search over stored content, and there is nowhere else the relation lives.

Two details the decision forced. The alias spelling is used rather than replacing the target, because the target is what resolves the link in Obsidian and two notes can share a title; the authors own alias always wins over the note title. And `WIKILINK_SCAN` now swallows a `(note://...)` tail it already wrote, so importing an exported note replaces the identifier instead of appending a second one.

Files: `tldw_chatbook/Notes/note_import_plan_models.py` (grammar, `rewrite_wikilinks`, new `render_note_links`), `tldw_chatbook/Notes/note_import_executor.py` (link keys carry the title), `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (Preview), `Tests/Notes/test_note_import_obsidian.py`, `Tests/Notes/test_note_backlink_query.py`, `Tests/Notes/test_note_import_executor.py`, `Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
