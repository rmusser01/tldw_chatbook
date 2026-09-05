---
id: TASK-31629
title: Environment panel — deferred-minor review findings batch
status: To Do
assignee: []
created_date: '2026-09-04 23:10'
labels:
  - console
  - inspector
  - cleanup
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console Inspect rail Environment redesign (TASK-31450) went through
several whole-branch review rounds. The must-fix findings shipped on that
branch; this task collects the minors that were consciously deferred so they
are not lost. Each is small and independent — one PR can carry the batch, and
any single item can be dropped without blocking the rest.

They are recorded as one task rather than twelve because none of them is
worth its own review cycle, and because filing them separately would bury the
list they came from.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The Changes row annotates untracked files ("N untracked") so a tree whose changes are all untracked no longer reads as clean
- [ ] #2 Backlog task ids sort numerically, not lexically, in the Tasks list (`task-9` before `task-31450`)
- [ ] #3 The `"env-changes-review"` row id is an exported constant like every other row id, not a string literal repeated across module and screen
- [ ] #4 A failed `open_url` on the PR row tells the user something instead of only writing a log line
- [ ] #5 A `gh` invocation that fails with ENOENT is distinguished from one that fails for any other reason, so "not installed" and "broke" are not the same row
- [ ] #6 AC progress is counted only within a task file's acceptance-criteria section, so a checkbox elsewhere in the file cannot inflate it
- [ ] #7 Drilling into a fleet row clears the rail's auto-open dismissal consistently with the other paths that clear it
- [ ] #8 `relative_age` renders a sub-minute age as something better than "0m ago"
- [ ] #9 `serialize_console_rail_preferences` and the `ConsoleInspectorRail` class both carry docstrings describing what they own
- [ ] #10 The `_on_app_focus` test asserts the production path is reached rather than that a shadowing handler is absent
- [ ] #11 The rail's `UnknownModeError` catch is narrowed or justified in place
<!-- AC:END -->
