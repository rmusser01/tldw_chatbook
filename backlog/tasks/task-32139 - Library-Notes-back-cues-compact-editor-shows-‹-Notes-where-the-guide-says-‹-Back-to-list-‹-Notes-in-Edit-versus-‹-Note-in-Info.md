---
id: TASK-32139
title: >-
  Library Notes back cues: compact editor shows ‹ Notes where the guide says ‹
  Back to list; ‹ Notes in Edit versus ‹ Note in Info
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:58'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - copy
  - docs
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evidence assessor at 60x24: the editor's back cue is '‹ Notes', not the documented '‹ Back to list'. Design assessor at 235x52: Edit and Preview show '‹ Notes', Info shows '‹ Note'. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One back-cue wording across Edit, Preview and Info and across sizes
- [x] #2 The guide matches
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Unified the two editor back-cue buttons (#library-note-back in Edit/Preview, #library-note-context-back in Info -- previously '‹ Notes' vs '‹ Note') to one shared label, computed at both compose time and in the apply_session_state sync pass: '‹ Back to list' when state.compact, else '‹ Notes', matching the guide's own documented wording. Out of scope (different, untouched buttons): the note-loading/retry view's back button and the New-note-view's back button both already said '‹ Notes' unconditionally and were not part of this bug report. Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py, Docs/User_Guide/library/notes.md. Tests: Tests/UI/test_library_notes_wave_editor_keys.py (2 new tests, one per size). Live-verified: 235x52 shows '‹ Notes' in both Edit and Info; 60x24 shows '‹ Back to list' in Edit.

Fix round 1: deferred, no action taken (reviewer-accepted, one sentence in the guide): the New-note view's own back button and the note-loading/retry view's back button still read '‹ Notes' unconditionally at compact width (never '‹ Back to list') -- both are different, untouched buttons (confirmed out of scope in the original pass above too), and the guide's "on compact terminals use `‹ Back to list`" sentence is technically only true for the editor's two back cues this task actually unified.
<!-- SECTION:NOTES:END -->
