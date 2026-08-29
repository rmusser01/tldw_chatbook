# Mutation Loader Group Dispatch Design

Date: 2026-08-28
Status: Approved
ADR required: no
ADR path: N/A
Reason: this enforces the worker-group contract established by TASK-19559 and
does not change storage, ownership, service, security, dependency, or long-lived
application boundaries.
Backlog:
[TASK-19870](../../../backlog/tasks/task-19870%20-%20Mutation-workers-refresh-inline-instead-of-dispatching-into-the-loader-group.md)

## Summary

Route mutation-triggered list refreshes through the same exclusive Textual
worker groups as user-triggered refreshes. Add one screen-local dispatch seam
for each affected loader, replace every direct and mutation call site with that
seam, and structurally forbid inline awaits of those loaders in production.

The expanded audit covers eleven mutation paths: six schedule mutations, two
notification mutations, and briefing generation, script casting, and audio
synthesis. Other mutation-triggered loaders in the two screens already use a
grouped dispatch or a grouped `@work` method and need no change.

## Verified Problem

TASK-19559 gave each standalone loader a distinct exclusive worker group, but
some mutation workers still call the underlying async loader inline. An inline
call is invisible to Textual's `WorkerManager`, so it is not cancelled when a
new refresh enters the loader group. The inline and grouped calls may then both
write the same screen-owned list, allowing an older result to repaint after a
newer result.

Current `origin/dev` contains:

- six `await self.load_tasks()` calls inside schedule delete, save, run-now,
  enabled-state, bulk-delete, and bulk-toggle workers, outside
  `schedules-load-tasks`;
- two `await self._load_notifications()` calls inside mark-read and dismiss
  workers, outside `wc_notifications`;
- three `await self._load_briefings()` calls in the `finally` blocks of
  briefing generation, script casting, and audio synthesis, outside
  `wl-briefings-load`.

The remaining mutation refreshes audited in these screens are already safe:
they call `run_worker(..., group=<loader group>)`, call a grouped `@work`
method, or execute a narrower loader inside an outer worker already scheduled
into that loader's group.

## Goals

- Give mutation and user refreshes one scheduling seam per affected list.
- Preserve latest-request-wins behavior through the existing exclusive groups.
- Prevent overlapping refreshes from interleaving writes to one list.
- Preserve mutation success/failure handling and loader error presentation.
- Make a future inline await of an affected loader fail an architecture test.
- Cover schedules, notifications, and artifacts with production-shaped
  concurrency regressions.

## Non-Goals

- Changing mutation ordering, service APIs, persistence, or business rules.
- Converting every screen loader to a decorator or introducing a generic
  application-wide loader coordinator.
- Adding generation counters in parallel with Textual's worker cancellation.
- Refactoring mutation workers that already dispatch through the correct group.
- Broad changes to all worker groups or the TASK-19559 inventory.

## Dispatch Contract

Each screen owns a small synchronous helper that is the only production seam
for scheduling its affected loader:

- `SchedulesWorkbench._request_tasks_refresh()` schedules `load_tasks` with
  `exclusive=True` in `schedules-load-tasks`;
- `WatchlistsCollectionsScreen._request_notifications_refresh()` schedules
  `_load_notifications()` with `exclusive=True` in `wc_notifications`;
- `WatchlistsCollectionsScreen._request_briefings_refresh(...)` schedules
  `_load_briefings(...)` with `exclusive=True` in `wl-briefings-load` and
  forwards the optional briefing-selection identity.

User actions, lifecycle-triggered refreshes, and mutation completions call the
same helper. No caller awaits the raw loader. When a newer request is
dispatched, Textual cancels the older worker in that group before the newer
worker publishes its rows. Mutation workers remain in their existing mutation
groups; scheduling the follow-up refresh does not move or serialize the write
itself.

The helpers remain screen-local rather than introducing a generic abstraction.
There are only three loaders, their arguments differ, and the worker group
names are domain contracts that should remain visible beside the loader.

## Mutation Flow

For each affected mutation:

1. the existing mutation worker performs its service or database write;
2. existing notification and error handling runs unchanged;
3. the worker requests a refresh through the screen's dispatch helper;
4. the mutation worker completes without awaiting list acquisition or repaint;
5. the exclusive loader group decides which overlapping refresh is current;
6. only the surviving loader publishes list state.

The artifact helpers retain the current attachment guard before requesting a
refresh. Briefing generation passes its newly generated identity so the
surviving reload can preserve the intended selection.

## Error and Cancellation Behavior

Loaders keep their current error handling and empty/error-state presentation.
Mutation workers keep their current write errors and success toasts. The change
only transfers ownership of the refresh coroutine to Textual.

Cancellation of an older loader is expected control flow. It must not be
reported as a mutation failure or user-facing error. A mutation failure that
currently refreshes the list continues to request that refresh so the screen
reconciles with authoritative state.

## Structural Guard

Extend `Tests/Architecture/test_worker_exclusive_group_inventory.py` with a
targeted AST rule for the three affected production loader methods. In their
own screen modules, any `await self.load_tasks(...)`,
`await self._load_notifications(...)`, or
`await self._load_briefings(...)` is a violation. Calls inside tests are not
restricted.

This deliberately avoids whole-program call-graph inference. The exact risky
loaders and modules are explicit, reviewable inventory, while future mutation
paths in those modules are covered automatically.

## Verification

Use test-driven development. Before production changes, add regressions that
are red against the inline-await implementation.

Representative mounted Textual tests cover each loader group:

- schedules: overlap a mutation-completion refresh and user refresh with
  controllable service results; release them out of order and assert the newest
  task rows are rendered and the mutation occurred;
- notifications: perform mark-read or dismiss while a user refresh overlaps;
  assert the newest notification rows and the completed mutation;
- artifacts: complete generate, cast, or synthesize while a manual artifact
  refresh overlaps; assert only the newest briefing projection is committed
  and the mutation result is retained.

The architecture test is mutation-checked by restoring one inline await and
proving the guard fails. Targeted test modules, modified-file Ruff lint and
format checks, and `git diff --check` form the implementation gate. A full test
suite is not required unless separately requested.

## Rejected Alternatives

Decorating the raw loaders as `@work` methods was rejected because it changes
their call type, makes nested-worker waiting implicit, and expands the change
well beyond the defect.

Generation tokens were rejected because they duplicate the latest-request-wins
semantics already provided by exclusive Textual worker groups and introduce
new mutable state with no additional user value.

Duplicating `run_worker(...)` at every call site was rejected because it leaves
group names and options free to drift again; one dispatch helper per loader is
the smallest durable seam.
