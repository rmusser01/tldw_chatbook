---
id: TASK-32262
title: >-
  Library Notes Import once review fidelity: dropped frontmatter keys, a no-
  change claim above a changed diff, generic copy for Obsidian file types
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
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four review-surface fidelity gaps found on the 71-file vault import:

- non-tag frontmatter keys (`mood: ok`) are dropped silently, with no review row and no receipt line;
- "Unchanged repeat" rows print `Content: no change` directly above a diff showing a changed line -- the classification compares source-to-source while the diff compares stored-note-to-raw-source;
- the repeat-import folder-collision panel opens already showing "That folder name already exists" against a field the user has not touched, with no radio pre-selected;
- `.canvas`, `.png` and `.pdf` all get the generic "This file type is not supported" while `.obsidian` and `.trash` get vault-aware copy; an Obsidian canvas deserves "Obsidian canvas - not a note".

The Obsidian work (task-32129) is otherwise the best-verified part of the screen -- every prior Obsidian complaint is answered and the vault is byte-identical by shasum through Check, Import and an in-app edit -- which is what makes these four visible.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontmatter keys that are not imported are reported in the review or the receipt
- [x] #2 A row classified as no change does not display a diff showing a change: both use one comparison basis
- [x] #3 The collision panel does not paint an error against an untouched field, and a default is pre-selected
- [x] #4 Obsidian-specific file types get vault-aware copy rather than the generic unsupported string
- [x] #5 Covered by tests for the no-change/diff agreement and for the dropped-key report
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Report non-tag frontmatter keys the import drops (parser records them, review row states them).\n2. Show the existing-note diff only on a row whose action writes content, so a no-change row never carries a changed diff.\n3. Pre-select the safe collision default and stop pre-arming the rename error.\n4. Vault-aware unsupported copy for .canvas and the other non-note types.\n5. RED/GREEN tests per item.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Four fidelity gaps, each traced before it was fixed.

1. Dropped frontmatter properties. Obsidian mode strips the leading YAML block and keeps only `title`, `tags` and `aliases`; everything else went nowhere and was reported nowhere. The parser now records the property names it drops on the payload (`ParsedNotePayload.unimported_frontmatter_keys`) and the review row states them ("… · not imported: status"). A frontmatter-only note reports nothing, because its block stays in the stored body.

2. "Content: no change." above a changed diff. PROVEN cause: the classification compares this sources payload fingerprint to the last IMPORT of it (`note_import_planner._classify_parsed_source`), while the row diff compares the STORED NOTE to the raw source (`library_note_import_controller._bounded_note_diff`). Those are different questions -- and for any Obsidian note the stored body carries rewritten links the file does not, so an unchanged repeat showed a diff by construction. The diff is now shown on the one row it describes: the row whose action replaces an existing notes content. Every other row keeps a diff-free effect line.

3. Pre-armed collision panel. `show_review` set `collision_rename_error` to "That folder name already exists." for an unresolved collision -- an error painted against a field nobody had touched -- and pre-filled the field with the name that was already known to collide. The controller now resolves a detected collision to the non-destructive default (a unique sibling) before the review opens, the reason line says where the notes will go, the rename field starts empty with its placeholder, and the error belongs to typed input again. If the planner refuses that default, the review still opens unresolved and approval is still blocked on it.

4. Generic unsupported copy. One sentence covered a canvas, a PNG and a PDF. A small extension->reason-code table gives the vault-aware copy `.obsidian`/`.trash` already had: "Obsidian canvas — not a note.", "Image — not a note. Add it in Library ▸ Media.", and so on. Anything not in the table keeps the honest generic sentence.

Verified on the 71-file vault through the real discovery/parser/planner/executor chain rendered at 235x52 and 100x30 (`wave3-caps/import-review/10-review-page1-*.txt`): the Daily notes read "not imported: created" / "not imported: mood", the canvas/PNG/PDF rows are named, and no row carries a diff.

Files: `tldw_chatbook/Notes/note_import_parsers.py`, `tldw_chatbook/Notes/note_import_plan_models.py`, `tldw_chatbook/Library/library_note_import_state.py`, `tldw_chatbook/UI/Library_Modules/library_note_import_controller.py`, `tldw_chatbook/Widgets/Library/library_note_import_canvas.py`, tests in `Tests/Notes/test_note_import_obsidian.py`, `Tests/Library/test_library_note_import_state.py`, `Tests/UI/Library_Modules/test_library_note_import_controller.py`, `Docs/User_Guide/library/notes.md`.
**Fix round 1 (review findings 2, 3, 4).** AC#2's second clause was ticked but
not implemented: suppressing the diff hid the by-construction defect without
removing it, and on the one row that does show a diff every line carrying a
link still read as changed. Both sides now reduce to the bare `[[target]]`
spelling first (`note_import_plan_models.wikilink_only`), so an unchanged
source produces an empty diff and a real change shows alone. Separately, the
collapsed-run summary reached `CollapsibleTitle`, whose `Content.from_text`
defaults to markup ON -- a vault folder named `[@click=app.quit]` became a live
action link and lost its own name; `Content()` turns that off, matching every
other Static on this canvas. And `.git` is now skipped at the walker for every
source, ungated by Obsidian mode or depth, so both platform adapters inherit
it: the git-backed 71-file vault reviews as 67 sources on one page instead of
174 across five.
<!-- SECTION:NOTES:END -->
