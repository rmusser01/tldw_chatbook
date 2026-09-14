---
id: TASK-32553
title: >-
  Library Notes: three names for "go back" — ‹ Notes, Back to navigator, Back to
  Notes
status: In Progress
assignee: []
created_date: '2026-09-13 06:48'
updated_date: '2026-09-14 18:32'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor A, everyone. Task-32139 aligned ‹ Notes / ‹ Note inside the editor; the other two surfaces still use their own words.

**What happened.** `‹ Notes` (editor, A 05), `Back to navigator` (Session Git panel, A 83), `Back to Notes` (Add from files chooser and import stepper, A 25; B 24). Captures: A 05, 25, 83; B 24.

**Cause.** PROVEN copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One back-cue grammar is used by the editor, the Session Git panel and the import stepper
- [ ] #2 notes.md and file-notes.md name the cue and are stamped
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live (done: import stepper reads 'Back to Notes'; Session Git panel and Folder-files work pane read 'Back to navigator').
2. RED pin for one back-cue grammar across the editor, the import stepper, Add from files, the sync-roots list, the Session Git panel and the Folder-files work pane.
3. Fix: one back_cue_label(destination) helper in library_shell_state.py (no widget-to-widget import cycle); _library_note_back_label routes through it so the editor keeps its compact wording; the six sites become '‹ Notes' / '‹ Files'.
4. GREEN + live capture of the three cues + notes.md / file-notes.md stamps.
<!-- SECTION:PLAN:END -->
