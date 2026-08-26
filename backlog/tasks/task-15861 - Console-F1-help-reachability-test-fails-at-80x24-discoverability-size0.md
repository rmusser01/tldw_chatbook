---
id: TASK-15861
title: Console F1 help reachability test fails at 80x24 (discoverability size0)
status: To Do
assignee: []
created_date: '2026-08-13 13:47'
labels:
  - console
  - tests
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_console_fleet_discoverability.py::test_console_f1_help_is_
scrollable_and_reachable_at_realistic_sizes[size0]` fails: at 80x24 the
rendered F1 help panel never shows the "Status markers:" line even after the
test scrolls — the assertion `'Status markers:' in <rendered text>` fails
while the 160x40 sibling (`[size1]`) passes.

This failure is PRE-EXISTING and unowned: during fleet PR 3a-2 (2026-08-13)
it was verified to fail on a pristine worktree of pristine origin/dev (branch
base `ec7db3c4c`, throwaway worktree created and removed for exactly this
check — see the PR 3a-2 ledger, Task 4 report). No fleet branch introduced
it; it is filed here so the red stops being rediscovered and re-diagnosed by
every branch that runs the file.

Diagnosis is part of the task: decide whether the test's scroll/geometry
probing is wrong at small sizes or whether the help panel genuinely clips
content unreachably at 80x24 (which would be a real accessibility defect —
the test exists precisely because an earlier fold hid the Agents section).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The [size0] parameterization passes for the right reason: either the help content is genuinely reachable by scrolling at 80x24, or the test's probing is shown to be at fault and fixed without weakening the reachability guarantee
- [ ] #2 The [size1] sibling and the rest of the discoverability file stay green
- [ ] #3 The diagnosis (test defect vs. real clipping) is recorded in Implementation Notes
<!-- AC:END -->
