---
id: TASK-38741
title: "Make the 40x20 Library Notes source-choices keyboard test fit the UI lane budget"
status: To Do
assignee: []
created_date: 2026-11-04 03:09
labels:
- follow-up
- tests
- ci
- library
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-38695 put the File Notes workspace tests on the UI PR lane node by node, but left `Tests/UI/test_library_file_notes_workspace.py::test_library_notes_source_choices_render_and_switch_by_keyboard[size0]` off it: the test passes but takes 31-34 s locally (serial, `--timeout=180`), over the census's 30 s per-entry budget. At 40x20 it presses Tab up to 240 times to reach the source choices, so its cost is the keyboard walk, not the behaviour it checks. Until it fits, the 40x20 keyboard path for switching note sources is gated by no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The [size0] case still proves the source choices render and can be switched by keyboard alone at 40x20
- [ ] #2 It runs in under 20 s serially locally, measured over 3 runs and recorded in the notes
- [ ] #3 It is on the UI PR lane, appended at the census tail
<!-- AC:END -->
