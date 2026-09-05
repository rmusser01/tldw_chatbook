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
- [ ] #12 `ENV_SUMMARY_BUDGET` (and the 34-column widget-test pin) match the rail's REAL content width — measured 30 columns at 80x24, 36 at 200x50, so the "Environment" title still ellipsizes to "Environm…" at the smallest size; a budget of ≤16 (or a width-aware budget) fits the full title, and the test must pin a width the smallest supported terminal actually produces (final re-review, live-measured 2026-09-04)
- [ ] #13 The Tasks section header summary gets the same budget treatment — a branch-linked summary like "task-31450 · In Progress" (24 cols) + 3-col toggle leaves 3 cols for the 5-col "Tasks" title at 80x24, reproducing the F1 squeeze; reachable on any feat/task-NNNNN branch (final re-review)
- [ ] #14 The F2 negative-control docstring in `Tests/UI/test_console_environment_wiring.py` stops claiming a click path it does not exercise (Textual focuses focus_on_click widgets on MouseDown before forwarding, so the test proves only the programmatic re-projection path)
- [ ] #15 `TasksEnvState.scanning` + its projection branch + `test_scanning_placeholder` are either wired to real behavior or deleted — provably dead code since the bounded-read optimization (owner call recorded at arc close)
<!-- AC:END -->
