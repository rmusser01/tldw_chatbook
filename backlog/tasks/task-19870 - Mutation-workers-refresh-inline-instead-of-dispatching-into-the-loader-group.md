---
id: TASK-19870
title: Mutation workers refresh inline instead of dispatching into the loader group
status: Done
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-29 05:46'
labels:
  - workers
  - concurrency
  - scheduling
  - watchlists
dependencies:
  - TASK-19559
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: surfaced by **TASK-19559**'s reviewer while verifying that task's
worker-group census.

**Historical filing context (2026-08-22):** this defect did not yet reproduce
on `dev` when the task was filed because the named worker groups —
`schedules-load-tasks` and `wc_notifications` — were being introduced by
TASK-19559, whose first review had returned do-not-ship. At `3605bd52d` every
schedules worker was `exclusive=True` with no group at all, the larger defect
TASK-19559 existed to fix. This follow-up was recorded before that merge so the
review observation would not be lost. TASK-19559 has since merged and is Done,
so its dependency and implementation precondition were satisfied before this
task began.

The shape: once a loader has its own worker group, a mutation worker that wants
the list refreshed afterwards should dispatch the refresh **into that group**,
so the group's exclusivity governs it. Instead these mutation workers `await`
the loader inline, inside their own worker:

- `UI/Screens/scheduling/schedules_workbench.py` — the delete / save / run /
  update / bulk-delete / bulk-toggle workers each `await self.load_tasks()`
  inline (`:496`, `:579`, `:666`, `:693`, `:913`, `:1002` at `3605bd52d`),
  while the standalone refresh calls dispatch into `schedules-load-tasks`
- the watchlists notification mark-read and dismiss handlers do the same
  outside `wc_notifications`
- the watchlists briefing generate, cast, and audio-synthesis workers likewise
  await `_load_briefings()` outside `wl-briefings-load`

An implementation-time audit of every mutation-triggered loader in both
screens found these eleven affected paths. Other mutation refreshes already
dispatch into their loader group directly or call a loader decorated with the
correct group and therefore remain out of scope.

The consequence is that an inline refresh is invisible to the group that is
supposed to serialize refreshes. A mutation-triggered refresh and a
user-triggered refresh can interleave over the same list, and the group's
cancel-the-previous guarantee does not apply to the one that was never
dispatched into it — the exact class of list corruption TASK-19559's reviewer
found in the Study pane (blocker R2), reached by a different route.

Low severity: both paths render from the same query, so the visible outcome is
a redundant or briefly out-of-order repaint rather than a lost write.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A refresh triggered by a mutation worker is subject to the same worker
      group as a refresh triggered directly by the user
- [x] #2 Two refreshes that overlap — one from a mutation, one from a user action —
      cannot interleave their writes to the same list
- [x] #3 A test drives a mutation and a concurrent user refresh and asserts the
      resulting list contents, and is mutation-checked (restoring the inline
      `await` makes it red)
- [x] #4 The schedules workbench, watchlists notification handlers, and watchlists
      briefing generation/cast/audio handlers are covered
- [x] #5 TASK-19559's worker-group guard is extended to notice an inline loader
      call inside a worker body, or the reason it cannot is recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a path-scoped AST guard that rejects inline awaits of the three affected loaders and pins all eleven mutation owners to their dispatch helper.
2. Write a mounted schedules overlap regression, then add one schedules refresh helper and route all direct and mutation refreshes through schedules-load-tasks.
3. Write mounted notification and artifact overlap regressions, then add the Watchlists notification and briefing refresh helpers and route all relevant call sites through their existing loader groups.
4. Run the targeted UI and architecture gate, Ruff lint/format, mutation-check the guard, complete task documentation, and review the diff before PR creation.

ADR required: no
ADR path: N/A
Reason: the task enforces TASK-19559's existing worker-group contract without changing an architectural boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added screen-local refresh helpers for schedules, notifications, and artifacts,
then routed all eleven audited mutation completions and the corresponding user
refreshes through their existing exclusive loader groups. Extended the worker
inventory with a path-scoped AST guard and mounted overlap regressions for all
three surfaces; generation-selection and detached cast/audio behavior remain
covered by focused dispatch tests.

Modified production and test files are the two affected screens plus the four
targeted test modules. The exact Python 3.11.13 gate collected 291 tests: 285
passed, three failures reproduced unchanged on `origin/dev` (the unrelated
Console wiring inventory violation and two Watchlists shell behaviors), and
three localhost feed-server tests were sandbox-blocked but passed 3/3 outside
the sandbox. A fresh task-owned gate passed 7/7 under Python 3.11.13 with the
following exact node inventory:

```bash
python -m pytest \
  Tests/Architecture/test_worker_exclusive_group_inventory.py::test_mutation_refreshes_dispatch_through_loader_group \
  Tests/UI/test_schedules_workbench.py::test_delete_mutation_refresh_cannot_repaint_after_newer_user_refresh \
  Tests/UI/test_watchlists_destination_shell.py::test_notification_mutation_refresh_cannot_overwrite_newer_pane_refresh \
  Tests/Watchlists/test_watchlists_artifacts_pane.py::test_generation_refresh_cannot_overwrite_newer_artifacts_refresh \
  Tests/Watchlists/test_watchlists_artifacts_pane.py::test_generation_refresh_selects_the_generated_briefing \
  Tests/Watchlists/test_watchlists_artifacts_pane.py::test_detached_cast_does_not_request_artifacts_refresh \
  Tests/Watchlists/test_watchlists_artifacts_pane.py::test_detached_audio_does_not_request_artifacts_refresh
```

Mutation checks proved both the structural guard and schedules race went red
after restoring one raw inline await, then the restored guard/race passed 2/2.

Ruff lint passed all six modified Python files. Ruff format passed five files;
`Tests/Watchlists/test_watchlists_artifacts_pane.py` retains the same unrelated
whole-file drift as `origin/dev`, while the task-added lines 530–659 pass a
range-scoped format check. Both working-tree and branch `git diff --check`
commands passed. Review of `git diff origin/dev...HEAD` found no task-owned
correctness issue. No full-suite run was requested, and no new general lesson
or documentation change was warranted.

ADR required: no
ADR path: N/A
Reason: this implements TASK-19559's existing worker-group contract without
changing storage, ownership, service, security, dependency, or long-lived UX
boundaries.
<!-- SECTION:NOTES:END -->

## Notes

Dependency satisfied: TASK-19559 is Done and merged, and its loader groups are
present. Historically, this task was intentionally held until that merge
because the pre-19559 code had the larger problem of no groups at all; applying
this follow-up before then would have sat on top of that defect incoherently.
