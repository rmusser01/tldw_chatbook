---
id: TASK-22201
title: >-
  Take workspace registry reads and the doubled row pipeline off the Console run tick
status: Done
assignee: ['@claude']
created_date: '2026-08-24'
labels:
  - performance
  - console
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22201).

PR #2034 (`a581f28e0`) reintroduced the exact hot-path shape TASK-21118 removed — on the
run tick instead of the keystroke path. `_sync_native_console_chat_ui`
(`UI/Screens/chat_screen.py:15206-15218`) builds `_build_console_workspace_context_state()`
three times per 0.2 s tick (twice via `_current_console_rail_state()`, once via
`_sync_console_workspace_context()`). Each build now reaches
`_console_browser_workspace_records()` twice (pre-existing browser labels + the NEW
`workspace_tree_projection(rows)` at `UI/Console_Modules/workspace.py:2708` -> `:1525-1535`
-> `:3865-3887`), and each call runs `ensure_default_workspace()` (SELECT + bindings probe
+ occasional DELETE write txn, `Workspaces/registry_service.py:572-609`, `:1173-1203`) plus
`list_workspaces()` — all synchronous on the event loop: roughly 45 extra queries/second
while a reply streams. The same builds also run the whole-row-set canonical-owner
reconciliation twice per build (`workspace.py:2628-2679` and again at `:1558-1563` inside
the tree projection): merge across five row groups, membership dict rebuild,
`_rows_with_latest_canonical_owner` + `_overlay_current_console_browser_markers` passes —
O(all conversations), pure Python, x3 per tick. The `state_changed` guard at
`chat_screen.py:11100` gates only the push, not the build. The keystroke path is still
clean (memo at `workspace.py:2726-2826` intact — verified live: 19 keys = 0 SQL at tip vs
48 at the pin).

## Acceptance Criteria

- [x] A run tick with unchanged registry state performs zero workspace-registry SQL (memoize `_console_browser_workspace_records` on the existing `mutation_generation`, the TASK-21118 pattern) — proven by the sqlite trace-callback probe from the review
- [x] The row merge / canonical-owner / overlay pipeline runs at most once per state build, and the build runs at most once per tick unless its inputs changed
- [x] `ensure_default_workspace()` (a write-capable repair) is not called from any display/build path — only from mutation paths
- [x] Per-tick build cost during streaming is measured before/after; tree/browser behavior is unchanged when the registry actually changes (existing tests green)

## Implementation Plan

1. Baseline measurement on the pristine worktree: a mounted-console probe (the
   TASK-21118 `_WorkspaceReadCounter` shape — `WorkspaceDB.connection`/`transaction`
   round-trips + registry read-method counts + `set_trace_callback` statement counts)
   across 50 direct `_sync_native_console_chat_ui()` ticks, plus wall-time and
   build/pipeline invocation counts.
2. Registry read memo, the TASK-21118 pattern extended: a generation-keyed
   read-through view over the app's `LocalWorkspaceRegistryService` serving the four
   display reads the Console context build performs (`get_active_workspace`,
   `list_workspaces`, `list_runtime_bindings`, `list_workspace_memberships`), cached
   on (service identity, `mutation_generation`), never caching on doubles without a
   real int generation and never caching a raised read.
   `_console_browser_workspace_records` is served from the same view's
   `list_workspaces` cache; `build_console_workspace_state` receives the view.
   ADR-028's from-disk binding-status recompute keeps running per build (only the
   SQL is cached).
3. Generation completeness for THIS memo's read set: today only workspace-record
   mutators bump. Add `_bump_mutation_generation()` to `save_runtime_binding`
   (covers `add_folder_binding`/`set_folder_binding_access`),
   `remove_runtime_binding`, `_delete_default_runtime_bindings` (on actual delete),
   and `link_membership`; update the property's contract docstring.
   `set_workspace_scope`/`set_change_review_enabled` write tables outside this read
   set and stay as-is.
4. `ensure_default_workspace` off the display path: delete the call from
   `_console_browser_workspace_records`. Repair ownership stays with the existing
   mutation/startup seams: app wiring (`app.py` `_wire_workspace_registry_services`),
   `archive_workspace`, `set_active_workspace`'s switch-to-Default repair, and
   `_set_active_workspace_for_console_session`'s global branch — the display path
   was never the only invoker.
5. Row pipeline once per build: `_with_console_conversation_browser_state` prepares
   ONE merged union (browser rows + workspace page-attempt rows), runs
   canonical-owner + overlay once over it, partitions the browser subset back out by
   display identity, and hands the prepared union to
   `workspace_tree_projection(rows, prepared_rows=...)`, which skips its own
   merge/canonical/overlay on the no-query path (the query path keeps its
   settled-lane pipeline — a different, small input set).
6. Build at most once per tick: a per-tick build cache object created at the top of
   `_sync_native_console_chat_ui`, revalidated at each of the three consumption
   points against a cheap volatile-input fingerprint (registry generation, current
   conversation id, store session tuples + active session id, run status, queued
   counts, canonical-membership revision, persisted-cache token, browser/lane
   queries); any fingerprint component failure or non-int generation means rebuild.
   Never consulted outside the tick, so every push path that builds after a
   mutation stays live. `_current_console_rail_state` and
   `_sync_console_workspace_context` accept an optional prebuilt state.
7. AC probes (red-first where feasible): new `Tests/UI/test_console_run_tick_workspace_reads.py`
   — (a) N settled ticks = 0 registry reads + 0 WorkspaceDB round-trips (+ trace-callback
   statement count 0); (b) counter-control test; (c) registry mutation (create/rename/
   set-active/binding change) followed by one tick refreshes browser records/tree; (d)
   pipeline runs ≤ once per build; (e) steady tick builds the state once.
8. Mutation tests: skip a generation bump → invalidation test reds; remove the memo →
   0-SQL probe reds.
9. Failure paths: registry write fails mid-transaction (generation NOT bumped on the
   raise paths — memo keeps serving the last committed truth, matching the DB);
   teardown-scoped except in `_sync_native_console_chat_ui` unchanged (the tick cache
   is a local, dies with the coroutine).
10. Targeted suites + `--collect-only` sweep, tee'd; `./scripts/preflight.sh`;
    re-measure per-tick cost; notes + Done.

## Implementation Notes

Three mechanisms, each the smallest one that closes its half of the finding:

1. **Generation-keyed display-read view** (`_ConsoleRegistryDisplayReads`,
   `UI/Console_Modules/workspace.py`): a read-through stand-in for the registry
   service serving the context build's four display reads
   (`get_active_workspace`, `list_workspaces`, `list_runtime_bindings`,
   `list_workspace_memberships`) from a cache revalidated against
   (service identity, `mutation_generation`) — the TASK-21118 memo pattern,
   extended. Raised reads are never cached; doubles without a real int
   generation stay live; ADR-028's from-disk binding-status recompute still
   runs per build (only the SQL is cached).
   `_console_browser_workspace_records` and `build_console_workspace_state`
   both read through it. **Generation contract widened** to cover the view's
   read set: `save_runtime_binding` (routes `add_folder_binding` /
   `set_folder_binding_access`), `remove_runtime_binding`,
   `_delete_default_runtime_bindings` (on actual delete), and
   `link_membership` now bump (`Workspaces/registry_service.py`);
   `set_workspace_scope`/`set_change_review_enabled` write tables outside the
   read set and deliberately do not.
2. **`ensure_default_workspace` off the display path**: removed from
   `_console_browser_workspace_records`. The repair's owners were already the
   mutation/startup seams — app wiring (`app.py`
   `_wire_workspace_registry_services`), `archive_workspace`,
   `set_active_workspace`'s switch-to-Default repair,
   `_set_active_workspace_for_console_session`'s global branch — the display
   path was never the only invoker, so nothing new had to take ownership.
3. **One pipeline pass + one build per tick**:
   `_with_console_conversation_browser_state` now merges browser + surviving
   page-attempt rows into ONE union, runs canonical-owner + overlay once, and
   partitions the browser subset back out by display identity (both passes are
   per-row, so this is exactly equivalent — see inline comment);
   `workspace_tree_projection(prepared_rows=...)` consumes the prepared union
   on the no-query path (stale page attempts are pruned BEFORE the union via
   the hoisted `_prune_stale_workspace_page_attempts`). The tick's builds are
   shared through `tick_workspace_build_scope` + `ConsoleTickWorkspaceBuilds`
   (task-15452's opt-in scope shape, plus asyncio-task identity): only the
   tick coroutine's own task reads the cache, each read revalidates a
   volatile-input fingerprint (registry generation, conversation id, store
   sessions/active, run status, queued counts, membership revision,
   persisted-cache token, lane generations, page-attempt shape), so the
   PR #660 freshness ruling is kept by mechanism — a session created mid-tick
   rebuilds — while interleaving handlers/workers always build live.

**A finding correction**: the review said 3 builds/tick; a stack probe measured
SIX (the inspector rows leg builds workspace state inside both rail-state
calls, the control bar, and the agent section). Explicitly threading state
through the three named call sites — the first implementation — left 4 of 6
builds; the task-scoped share catches all of them.

**Measured** (mounted-console harness, `Tests/UI/test_console_run_tick_workspace_reads.py`):
- Registry SQL, 5 settled ticks: **400 round-trips / 400 traced statements →
  0 / 0** (per tick: 16× ensure + 24× get_active + 24× list_workspaces +
  8× bindings + 8× memberships → all zero).
- Context builds per settled tick: **6 → 1** (gated at the
  `build_console_workspace_state` seam).
- 50-tick wall time (same venv, base worktree at `983aa5878` vs this branch):
  14.8–17.0 ms/tick → 13.0–14.3 ms/tick. Modest in this near-empty sandbox
  registry — the point of the fix is the eliminated event-loop SQL (incl. the
  occasional DELETE write txn) under real, contended databases (the 22200
  post-upgrade window), not sandbox wall time.

**Mutation-tested**: skipping the `rename_workspace` bump reds
`test_registry_mutation_reflects_in_the_next_tick`; deleting the view's cache
reds the 0-SQL probe (35 round-trips reappear); freezing the fingerprint reds
`test_tick_scope_rebuilds_when_registry_changes_mid_scope`.

**Failure paths walked**: every new bump sits after its transaction commits, so
a failed write rolls back AND leaves the generation unchanged — the memo keeps
serving the still-true pre-write state; raised reads are never cached, so a
persistently failing DB degrades per-build exactly as before. The tick scope's
`finally` clears the shared cache even when teardown kills the tick mid-await
(the scope sits inside the tick's teardown-scoped try), and the cache is
per-tick, so nothing outlives the coroutine.

**Tests**: new gate `Tests/UI/test_console_run_tick_workspace_reads.py`
(6 tests: 0-SQL probe w/ trace callback, counter control, mutation refresh,
pipeline once per build, one build per tick, mid-scope rebuild) + 1 new
registry-suite test for the binding/membership bumps + widened pure-reads
stability test. Targeted runs: Tests/Workspaces + workspace/rail/tray/tick UI
suites — 439 + 140 + 203 + 48 + 47 passed; `--collect-only` sweep of
Tests/UI + Tests/Workspaces: 16118 collected, 0 errors; preflight green after
regenerating the diagnostic inventory (reviewed: 1 removed debug — the deleted
ensure-repair log — and 1 added constant-string debug, no interpolation).
**12 pre-existing dev reds** in adjacent suites were each baselined RED on the
pristine pin `983aa5878` (test_console_new_workspace ×2, fleet-wake ×3,
session-settings ×2, stream-scrollback ×1, agent-rail ×1,
staged-evidence-strip ×3) — not caused by and not fixed by this change.

**Files**: `tldw_chatbook/UI/Console_Modules/workspace.py`,
`tldw_chatbook/UI/Screens/chat_screen.py`,
`tldw_chatbook/Workspaces/registry_service.py`,
`Tests/UI/test_console_run_tick_workspace_reads.py` (new),
`Tests/Workspaces/test_workspace_registry_service.py`,
`Docs/security/production-diagnostic-inventory.json` (regenerated).
