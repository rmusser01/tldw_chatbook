---
id: TASK-32252
title: >-
  Library Notes slash accelerator still inserts itself into the filter
  although task-32131 is Done and its test is green
status: Done
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
- [x] #1 Pressing `/` when no control on the Notes canvas is focused focuses the filter and leaves its content unchanged
- [x] #2 `/` typed into the already-focused filter still lands as a literal character, so `Work/Q3` stays typeable (the task-32131 ruling holds)
- [x] #3 Covered by a new test that reproduces the live starting state (no focused control on the canvas) and is demonstrated failing before the fix; the test record states why `test_slash_focuses_the_notes_filter_without_inserting_itself` passes today
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the live starting state: a Notes canvas with nothing focused and an empty filter.
2. If it reproduces, trace and fix; if not, record why the report read as it did.
3. Ship the regression pin the report's own state requires, with the `Work/Q3` half kept.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
**NOT REPRODUCED at dev 4a14b3f36f.** No code change; the task ships the regression pin AC#3 asks for, written against the live starting state the report describes.

What was tried, live on a seeded profile at 235x52, and in the harness:

- Notes list showing, focus on the rail row just pressed, `/` then `zz` -> filter focused, value `zz`. No leak.
- Notes list showing, `screen.set_focus(None)` (a genuinely unfocused canvas, which is the state `R/caps/61` shows), `/` -> filter focused, value `""`. No leak. This is the new pin.
- After Escape from the filter, `/` then `qq` -> the RAIL search box takes the keys (`#2590`'s landed behaviour), not the notes filter.
- A click on a non-focusable Static in the canvas does not blur the focused widget, so it cannot construct the state either.

The mechanism the captures are consistent with: the notes filter was ALREADY focused. `R/caps/61` reads as unfocused because a Textual `Input` shows its placeholder whenever its value is empty, focused or not, and a plain-text `capture-pane` cannot show the focus border colour — the two states are indistinguishable in that capture. With the filter focused, `/` types literally, which is exactly the task-32131 ruling this task's own AC#2 preserves so `Work/Q3` stays typeable. The reviewer then read "the filter is focused and contains `/`" as "`/` focused it and typed itself".

Both halves of the behaviour are now pinned from the unfocused state in one test, so a regression in either direction fails.

Files: `Tests/UI/test_library_notes_wave_editor_keys.py`.
<!-- SECTION:NOTES:END -->
