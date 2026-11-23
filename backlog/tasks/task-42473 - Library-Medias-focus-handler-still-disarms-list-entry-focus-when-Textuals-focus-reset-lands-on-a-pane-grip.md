---
id: TASK-42473
title: "Library Media's focus handler still disarms list-entry focus when Textual's focus reset lands on a pane grip"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- library
- focus
- hardening
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_rail_switch_keeps_reader_and_note.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42422. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** TASK-42422 fixed a Notes-to-resident-Media press that left Library focus on `#library-browse-library-grip`. A queued recompose removed the focused control, Textual's focus reset parked focus on the Library pane grip, and the screen's `DescendantFocus` handler disarmed the pending Media list-entry focus, because only Notes treats a grip landing as the app's own (Notes is exempt since TASK-41129). TASK-42422's guard keeps the entry's own landings off that path, so it left the handler alone (`UI/Screens/library_screen.py:10409`, `on_descendant_focus`). Any other route by which Textual's reset lands on a grip inside Media's entry window would still disarm the entry and leave focus on the grip, which is the defect class of TASK-41504. No failure has been seen since TASK-42422's fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A grip landing from Textual's focus reset inside Media's list-entry window does not disarm the entry, as for Notes; a test drives such a landing (for example by removing the focused control mid-window) and fails on the current handler
- [ ] #2 A user's own move to a grip (a click or a key) still disarms the entry, for Media and Notes alike; a test shows it
- [ ] #3 The rail-switch first-row tests pass in repeated runs under parallel load (at least 10 runs, or a soak recorded in the notes)
<!-- AC:END -->
