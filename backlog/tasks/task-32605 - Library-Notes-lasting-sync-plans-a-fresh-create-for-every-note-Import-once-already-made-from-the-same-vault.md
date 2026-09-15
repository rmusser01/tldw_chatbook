---
id: TASK-32605
title: >-
  Library Notes: lasting sync plans a fresh create for every note Import once
  already made from the same vault
status: To Do
assignee: []
created_date: '2026-09-15 06:37'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D1, P1, persona Alex / researcher, Obsidian workflow. B's headline finding.

What happened. Import once on the power vault created 54 notes (B cap 25, rail count 10 -> 64, sqlite confirms). A second Import once of the same folder correctly reorganises into 'Unchanged repeat (54)', every row defaulted to Skip and annotated 'Existing note: Monday (version 1). Folder placement: no change' (B cap 28). Minutes later, Add from files -> Keep a folder synced on the SAME folder -> Check changes returns '54 safe · 0 need attention · 4 skipped · 0 folder moves' with every row reading 'Create a Library note' (B cap 32). Activating would leave 108 notes -- the whole vault duplicated. The dry run is the contract that screen exists to state, so the declared effect is the finding; B did not activate.

This is a seam between two individually-correct wave-4 fixes: 32541 (#2676) taught Import once repeat detection, 32535 (#2679) gave the sync review Import once's Obsidian pass -- but not its receipt ledger.

Cause, PROVEN at the planner. The two paths do not share a note identity. The sync runtime mints a binding for every discovered file with a synthetic note id, sha256('note\0<root_id>\0<relative_path>') (Notes/notes_sync_runtime.py:951-958), and observes only that id; a note Import once created carries a different id and is invisible, so _plan_unbound sees file_exists and not note_exists and returns CREATE_NOTE / file_discovered (Notes/notes_sync_reconciler.py:437-467). Import once's dedup lives in a different planner, pinned by Tests/Notes/test_note_import_planner.py::test_a_two_row_csv_imported_twice_classifies_as_unchanged_repeat. No test pins the observed sync behaviour. The reconciler already has the vocabulary for the join -- an unbound binding with both sides present raises duplicate_authority (notes_sync_reconciler.py:443-449) -- it is simply never reached because the note is never found.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Keeping a folder synced on a folder Import once has already imported recognises the existing notes instead of planning a create for each
- [ ] #2 The sync review offers the same choice Import once offers on a repeat -- link to the existing note, skip, or create new -- with the existing note named on the row
- [ ] #3 Following both documented Obsidian paths on one vault cannot produce two notes per file
- [ ] #4 An end-to-end test covers import-then-sync on one folder and pins the recognition, on the production planner rather than a fake
<!-- AC:END -->
