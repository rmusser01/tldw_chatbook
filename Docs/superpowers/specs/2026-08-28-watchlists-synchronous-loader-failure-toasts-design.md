# Watchlists Synchronous Loader Failure Toasts Design

**Task:** TASK-2340 — Silent loader failures contradict TASK-1090's toast premise

**Status:** User-approved design; written specification under review

## Context

TASK-1090 deliberately allowed Watchlists background reads to log unexpected
failures at debug level only when users also received a failure toast. Current
`dev` still violates that bargain in two synchronous read paths:

- `WatchlistsCollectionsScreen._load_source_rows_for_tree` catches any service
  failure, writes a debug log, and returns an empty list.
- `WatchlistsCollectionsScreen.scoped_source_rows` does the same while resolving
  the Sources table's current scope.

In both cases the empty fallback is also the valid representation of “this scope
contains no sources.” The UI therefore cannot distinguish a failed read from an
honestly empty result.

TASK-2340's original audit list has drifted. `_resolve_breadcrumb_labels` now
reads only the already-published tree snapshot and has no exception handler;
`_refresh_feeds_region_for_scope` no longer exists. The repair must audit the
current code rather than recreate obsolete paths.

## Goals

- Surface each live synchronous source-loader failure through a fixed,
  markup-disabled error toast.
- Preserve the current safe empty-list fallback and debug exception logging.
- Keep exception messages, source names, URLs, watchlist names, and other
  untrusted values out of the toast.
- Add a structural contract that detects the next synchronous debug-swallow
  handler unless it either notifies in the same handler or carries an explicit,
  current exemption.
- Update TASK-2340 so its acceptance criteria describe the current code.

## Non-goals

- Do not convert synchronous compose-time reads into async workers.
- Do not add persistent inline recovery states.
- Do not promote expected background-read failures to warning logs.
- Do not change service, storage, backend, or navigation contracts.
- Do not repair unrelated layout-persistence or focus-recovery behavior.

## Chosen Design

### User-facing behavior

The two live exception handlers continue to log their full exception at debug
level for diagnostics and continue to return `[]`. Before returning, they call
the screen's existing `_notify_watchlists` seam with app-authored copy:

- `_load_source_rows_for_tree`: `Failed to load sources for this watchlist.`
- `scoped_source_rows`: `Failed to resolve sources for the selected scope.`

Both calls use `severity="error"` and `markup=False`. The messages contain no
dynamic data, so they neither disclose a service exception nor allow remote or
user-authored text to reach Rich markup parsing.

No new notification abstraction is introduced. `_notify_watchlists` already
degrades safely when the app or a test harness has no callable notifier and is
the established screen-local boundary for markup policy.

### Structural enforcement

`Tests/UI/test_watchlists_check_now_failure.py` gains an AST audit over
synchronous class methods on `WatchlistsCollectionsScreen`, including local
functions nested inside those methods. Each handler is identified by a stable
handler key: its qualified owner path plus the literal debug message it emits.
The scanner rejects duplicate keys so a second same-message handler cannot hide
behind an existing exemption.

For every `except` handler that calls `logger.debug(...)` or
`logger.opt(...).debug(...)`, the test requires one of:

1. a call to `self._notify_watchlists` with literal `severity="error"` and
   `markup=False` in that exact handler; or
2. that exact handler key appears in a test-local exemption mapping with a
   non-empty reason.

The scanner does not descend from one handler into nested exception handlers,
functions, lambdas, or classes when attributing its logger and notifier calls.
That boundary prevents a notifier in a child scope or sibling handler from
satisfying the handler under audit. A logger call qualifies only when its call
chain is rooted at the module's `logger` name and ends in `.debug`.

After notified handlers are removed, the discovered handler keys must equal the
exemption mapping's keys exactly. Missing entries fail as unhandled policy
violations; stale entries fail as obsolete exemptions. The current exemptions
are:

| Handler key | Reason |
| --- | --- |
| `_read_tree_data_snapshot.read_branch` + `Failed to load watchlists tree branch: {}.` | Failed branches are published in `TreeDataSnapshot.failures`; `_load_tree_data` emits one error toast per failure episode. |
| `_read_tree_data_snapshot` + `Failed to load Watchlists tree membership.` | The membership failure is published in the same snapshot failure set and paid for by `_load_tree_data`'s episode-level toast. |
| `_recompute_effective_layout` + `Workbench not mounted yet; layout applies on compose.` | This preserved broad layout-request fallback covers lookup, token creation, and workbench layout application; it is lifecycle/layout handling rather than a data loader and remains outside this policy. |
| `_schedule_layout_persist` + `Could not schedule preferred Watchlists layout persistence.` | This is preference-write scheduling, outside the synchronous loader policy. |
| `_persist_layout_worker` + `Failed to persist preferred Watchlists pane layout.` | This is a background preference writer, outside the synchronous loader policy. |
| `_persist_layout_worker` + `Could not acknowledge preferred Watchlists layout write.` | This is the preference writer's UI-thread acknowledgement path, outside the synchronous loader policy. |
| `_restore_focus_after_swap` + `No section tab to restore focus to after the swap.` | A missing rebuilt tab is a lifecycle fallback with no data-read failure to report. |

The two source loaders are intentionally absent from the exemptions. A future
synchronous debug-swallow handler fails the contract until its author either
adds a markup-safe error toast in the same handler or documents that precise
handler's reason for exclusion. Exact inventory equality keeps exemptions
reviewable and prevents an exempt method from silently accumulating unrelated
handlers.

### Regression coverage

Mounted Watchlists tests replace the relevant service read with a raising test
double and assert:

- the method still returns an empty list;
- exactly one fixed error message is emitted for the driven call;
- severity is `error`;
- `markup` is explicitly false; and
- neither the exception message nor any sentinel bracket-shaped content reaches
  the toast.

The structural test is also run against the pre-fix code during TDD. It must
fail by naming `_load_source_rows_for_tree` and `scoped_source_rows`, proving
that the contract detects the defect it exists to prevent.

## Current Audit Outcome

The original TASK-1090 synchronous list resolves as follows on current `dev`:

- `_load_source_rows_for_tree`: live silent failure; fix.
- `scoped_source_rows`: live silent failure; fix.
- `_resolve_breadcrumb_labels`: no longer catches failures; no change.
- `_refresh_feeds_region_for_scope`: removed; no change.

The broader AST inventory finds the seven exempt handlers listed above across
five method owners. None is a synchronous source loader that renders failure as
an honestly empty data region.

## Verification Scope

Verification is intentionally focused:

- the new mounted regressions;
- the synchronous-handler structural contract;
- the complete `Tests/UI/test_watchlists_check_now_failure.py` module;
- directly affected Watchlists screen tests if the mounted harness shares
  fixtures with another module;
- Ruff lint/format for modified Python files; and
- `git diff --check`.

No repository-wide sweep is planned unless explicitly requested.

## ADR Check

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a bounded correction to error reporting inside the existing
Watchlists screen. It changes no schema, ownership boundary, service contract,
runtime selection, dependency, or long-lived application structure.
