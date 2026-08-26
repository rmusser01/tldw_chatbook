---
id: TASK-19870
title: >-
  Mutation workers refresh inline instead of dispatching into the loader group
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - workers
  - concurrency
  - scheduling
  - watchlists
priority: low
dependencies:
  - TASK-19559
---

## Description

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

The consequence is that an inline refresh is invisible to the group that is
supposed to serialize refreshes. A mutation-triggered refresh and a
user-triggered refresh can interleave over the same list, and the group's
cancel-the-previous guarantee does not apply to the one that was never
dispatched into it — the exact class of list corruption TASK-19559's reviewer
found in the Study pane (blocker R2), reached by a different route.

Low severity: both paths render from the same query, so the visible outcome is
a redundant or briefly out-of-order repaint rather than a lost write.

## Acceptance Criteria

- [ ] A refresh triggered by a mutation worker is subject to the same worker
      group as a refresh triggered directly by the user
- [ ] Two refreshes that overlap — one from a mutation, one from a user action —
      cannot interleave their writes to the same list
- [ ] A test drives a mutation and a concurrent user refresh and asserts the
      resulting list contents, and is mutation-checked (restoring the inline
      `await` makes it red)
- [ ] The schedules workbench and the watchlists notification handlers are both
      covered
- [ ] TASK-19559's worker-group guard is extended to notice an inline loader
      call inside a worker body, or the reason it cannot is recorded

## Notes

Do not start this before TASK-19559 merges — the group names this task refers
to do not exist until then, and the pre-19559 code has a bigger problem (no
groups at all) that this fix would sit on top of incoherently.
