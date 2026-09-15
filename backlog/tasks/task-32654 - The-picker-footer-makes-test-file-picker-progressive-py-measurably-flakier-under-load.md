---
id: TASK-32654
title: >-
  The picker footer makes test_file_picker_progressive.py measurably flakier under load
status: To Do
assignee: []
created_date: '2026-09-15 17:05'
labels:
  - library
  - picker
  - tests
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from task-32606, re-review 1. Added fragility, measured -- not a
pre-existing flake (task-32606's first report said otherwise and is corrected
in its notes).

MEASUREMENT. Under a fixed 7-file prefix, with the last three pairs run
SIMULTANEOUSLY so both sides saw the same machine load (load average 16-20,
~45 sibling pytest processes):

  branch: 9 runs, 4 reds -- test_sort_controls_preserve_highlight_and_file_filter[FileOpen] x3,
                            test_click_from_old_painted_row_cannot_select_replacement x1
  dev:    7 runs, 0 reds

Removing `yield Footer()` from `base_dialog.compose` in a scratch copy made
that prefix green. The file is `47 passed` in isolation on both sides.

MECHANISM (plausible, benign). `Footer.bindings_changed` schedules a
`recompose` of ~8 `FooterKey` widgets on every active-bindings change -- i.e.
on every focus move in every vendored picker -- which adds message-loop work
to a file whose assertions are `wait_until(..., seconds=4)`. No coordinate or
layout assertion fails; only the timing ones.

NOT THE FIX: raising the bound, or a retry. A timing assertion that needs a
bigger number to pass is the finding. Either make the waits event-driven
(await the listing's own settled signal rather than a wall clock), or stop
the footer recomposing on focus moves that do not change the visible chip
set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_file_picker_progressive.py is as reliable under load with the picker footer as without it
- [ ] #2 The measurement above is re-run on both sides and the branch red count is 0
- [ ] #3 No wait_until bound in that file was raised to achieve it
<!-- AC:END -->
