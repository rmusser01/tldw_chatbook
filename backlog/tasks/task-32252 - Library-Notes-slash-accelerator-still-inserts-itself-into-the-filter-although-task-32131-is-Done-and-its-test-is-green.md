---
id: TASK-32252
title: >-
  Library Notes slash accelerator still inserts itself into the filter
  although task-32131 is Done and its test is green
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three clean repros, by D and by the reconciling parent, from a state where **no control on the canvas is focused** (`R/caps/58`, `62`): pressing `/` focuses the notes filter and types `/` into it, so the query is wrong from the first character. The footer advertises `/ find note` and the guide lists it as "Focus the note filter".

Task-32131 is Done and `test_slash_focuses_the_notes_filter_without_inserting_itself` is green. The gap between what the test pins and what the terminal does is the theme of this run, and the rule for this one must be stated before it is fixed again: 32131's landed fix made `LibraryRailSearchInput`'s slash-swallow opt-in and passed `swallow_slash_on_focus=False` for the notes filter, because filter content can legitimately contain `/` (`Work/Q3`). Its green test drives the *unfocused-filter* case through the screen-level handler in a mounted harness. The live repro starts from a different state -- the canvas with nothing focused at all -- which the test never constructs, so the character still reaches the Input after focus is granted. Any fix must keep `Work/Q3` typeable, and the new test must be shown red on the live behaviour before it goes green.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pressing `/` when no control on the Notes canvas is focused focuses the filter and leaves its content unchanged
- [ ] #2 `/` typed into the already-focused filter still lands as a literal character, so `Work/Q3` stays typeable (the task-32131 ruling holds)
- [ ] #3 Covered by a new test that reproduces the live starting state (no focused control on the canvas) and is demonstrated failing before the fix; the test record states why `test_slash_focuses_the_notes_filter_without_inserting_itself` passes today
<!-- AC:END -->
