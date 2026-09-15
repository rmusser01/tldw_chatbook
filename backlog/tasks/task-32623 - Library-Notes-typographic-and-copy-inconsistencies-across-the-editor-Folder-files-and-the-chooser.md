---
id: TASK-32623
title: >-
  Library Notes: typographic and copy inconsistencies across the editor, Folder
  files and the chooser
status: To Do
assignee: []
created_date: '2026-09-15 06:43'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D15 and assessor A heuristic 2, every persona. A cleanup pass, filed as one task because each item is a one-line fix in a different file.

- 'Unavailable - server sync-folder capability not installed' uses a hyphen where the rest of the screen uses an em dash (B cap 29).
- Word counts disagree between two panes describing the same note: the editor footer reads 5,453 words and the Info pane reads 5454 (B caps 11 and 16). Residual of task-32538, which fixed a much larger error (404 for 5,407).
- The Folder-files footer reads with a leading space, a double space, and an instruction about a future state: 'typing in field | esc notes |  after esc: / focus search' (B cap 34).
- The Folder-files empty state offers 'Use file_notes' as a button label -- an internal config key shown to a first-timer (A cap 26). This is the single item that cost heuristic 2 its fourth point.
- The 'ctrl+end end of note' footer chip stays visible while focus is on a button, where the key does nothing (A section 11).
- The Synced placement badge is repeated on all 54 rows of a folder whose own row already says Sync managed (A section 11).
- Mode-row buttons shift horizontally when Discard new note appears and disappears, moving click targets under the cursor (A caps 04 to 05).

Cause PROVEN by capture for every item.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One dash convention across the surface
- [ ] #2 Two panes describing the same note report the same word count
- [ ] #3 Footer strings carry no stray whitespace and describe the current state, not a future one
- [ ] #4 No button label is an internal config key
- [ ] #5 A footer chip is shown only where its key does something, and controls do not move under the pointer as state changes
<!-- AC:END -->
