---
id: TASK-19870
title: Mutation workers refresh inline instead of dispatching into the loader group
status: In Progress
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-29 04:13'
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

**Precondition, stated up front: this does not reproduce on `dev` today.** The
worker groups it names — `schedules-load-tasks` and `wc_notifications` — are
introduced by TASK-19559, which is still in flight (its first review returned
do-not-ship). At `3605bd52d` every schedules worker is `exclusive=True` with no
group at all, which is the larger defect TASK-19559 exists to fix. This task is
the follow-up that becomes live the moment that one lands, and it is filed now
so the observation is not lost between a fix round and a merge.

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
- [ ] #1 A refresh triggered by a mutation worker is subject to the same worker
      group as a refresh triggered directly by the user
- [ ] #2 Two refreshes that overlap — one from a mutation, one from a user action —
      cannot interleave their writes to the same list
- [ ] #3 A test drives a mutation and a concurrent user refresh and asserts the
      resulting list contents, and is mutation-checked (restoring the inline
      `await` makes it red)
- [ ] #4 The schedules workbench, watchlists notification handlers, and watchlists
      briefing generation/cast/audio handlers are covered
- [ ] #5 TASK-19559's worker-group guard is extended to notice an inline loader
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

## Notes

Do not start this before TASK-19559 merges — the group names this task refers
to do not exist until then, and the pre-19559 code has a bigger problem (no
groups at all) that this fix would sit on top of incoherently.
