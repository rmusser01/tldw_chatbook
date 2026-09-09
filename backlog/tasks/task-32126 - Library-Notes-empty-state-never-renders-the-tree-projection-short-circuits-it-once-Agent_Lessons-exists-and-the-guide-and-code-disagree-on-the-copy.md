---
id: TASK-32126
title: >-
  Library Notes empty state never renders: the tree projection short-circuits it
  once Agent_Lessons exists, and the guide and code disagree on the copy
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 15:56'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - onboarding
  - docs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN: the canvas yields the empty Static only when `tree_projection is None`, and a brand-new profile always has a tree because the `Agent_Lessons` folder is seeded (documented in notes.md). Result: Notes (0) shows one row, '▸ Agent_Lessons', and nothing else. The guide promises 'No notes yet. Create one to see it here.'; `_EMPTY_NOTES_COPY` in library_notes_state.py says 'No notes yet. Create your first note.'; neither renders. A first-time user's first question is what Agent_Lessons is and whether they made it. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A profile with zero notes shows an empty-state line with a call to action even when seeded folders exist
- [x] #2 Agent_Lessons carries a one-line gloss in place, or is hidden until it contains a note
- [x] #3 The guide and the code use the same empty-state copy
- [x] #4 Covered by a test on a zero-note projection that contains the seeded folder
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix _compose_tree_rows to render #library-notes-empty above the tree whenever the library holds zero notes (list_state.empty_kind == source-empty), independent of tree_projection.rows.\n2. Add AGENT_LESSONS_FOLDER_GLOSS to Notes/agent_lessons.py and gloss the Agent_Lessons folder row's label while the library is empty.\n3. Write failing widget tests in Tests/Widgets/Library/test_library_notes_canvas.py, then implement to green.\n4. Fix Docs/User_Guide/library/notes.md empty-state copy to match the code constant; document the gloss.\n5. Live-verify on the fresh profile.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed: _compose_tree_rows() in library_notes_canvas.py yielded #library-notes-empty only when tree_projection.rows was empty; a seeded Agent_Lessons folder always gives it a row, so a zero-note library never showed the empty state. Now yields the Static above the tree whenever list_state.empty_kind == 'source-empty' (the library holds zero notes), independent of the projection. Added AGENT_LESSONS_FOLDER_GLOSS to Notes/agent_lessons.py and gloss the Agent_Lessons folder row's label with it while the library is empty (a per-folder emptiness check is not available for a collapsed row; scoped to the exact zero-notes scenario this task covers -- see the ponytail comment at the gloss site). Fixed the guide's stale copy (Docs/User_Guide/library/notes.md) to match _EMPTY_NOTES_COPY exactly, and documented the gloss. Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py, tldw_chatbook/Notes/agent_lessons.py, Docs/User_Guide/library/notes.md, Tests/Widgets/Library/test_library_notes_canvas.py (2 new tests, TDD RED->GREEN). Live-verified on the fresh onboarding profile: Notes (0) shows 'No notes yet. Create your first note.' above '▸ Agent_Lessons — where Console a...' (gloss truncated by pane width; full text pinned in the unit test).

Addendum (PR #2538 review, Qodo finding 5): the new source-empty branch rendered the empty banner without checking whether the tree projection's root "folders"/"placements" slices were still loading. A fresh Notes visit computes empty_kind == "source-empty" before the bulk source-count lookup resolves (_local_source_counts.get("notes") is still None), while _request_library_notes_tree_initial_load's synchronous canvas sync already paints a loading pager row for each root slice -- so a user who does have notes could briefly see "No notes yet" above the loading rows. Guarded the branch with `not any(row.loading for row in projection.rows)`, using the LibraryNotesTreeRow.loading flag the tree builder already sets. Left the projection's older "if not projection.rows: yield empty" branch (pre-dating this task) alone: build_paged_library_notes_tree always returns at least one loading pager row while a slice is loading, so that branch's rows-is-empty condition can't coincide with an in-flight load in practice. Test: added test_source_empty_with_loading_root_slices_suppresses_empty_line (RED before the fix, GREEN after). Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py, Tests/Widgets/Library/test_library_notes_canvas.py.
<!-- SECTION:NOTES:END -->
