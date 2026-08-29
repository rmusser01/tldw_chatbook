# Console Context rail deferred-mount implementation plan

Outcome: candidate rejected. The measured implementation cleared the
first-interactive and full-ready thresholds but missed the Enter-to-worker input-p95 budget,
so no production deferral was retained. ADR-078 records the decision.

**Goal:** Reduce Console first-interactive mount time without caching screens,
moving layout, losing rail state, or delaying input.

**Architecture:** `ConsoleLeftRail` owns a construction-time deferred flag and
an idempotent async hydration method. `ChatScreen` mounts the final rail root
eagerly and schedules hydration before its existing post-refresh sync pass.
Pre-hydration rail syncs update backing state; DOM writes remain guarded.

ADR required: yes
ADR path: `backlog/decisions/078-defer-console-context-rail-content-until-first-refresh.md`
Reason: deferred composition changes the long-lived Console mount and lifecycle
contract, so the measured choice and rejected alternatives need a durable home.

## Task 1: Pin the deferred lifecycle with RED tests

1. Prove a deferred rail mounts only a lightweight placeholder before the first
   refresh and hydrates every section exactly once.
2. Prove workspace and section state synchronized before hydration is rendered
   by the hydrated tree.
3. Prove hydration is idempotent and harmless after unmount.
4. Prove a mounted `ChatScreen` paints an interactive composer before the rail
   content, then hydrates the rail and restores focus/state.

## Task 2: Implement narrow first-refresh hydration

1. Add `defer_content` and `hydrate_deferred_content()` to `ConsoleLeftRail`.
2. Keep the rail root's existing id, classes, sizing, visibility, and focus
   behavior; compose only one non-interactive placeholder while deferred.
3. Make pre-hydration workspace/section sync update the rail's backing state.
4. Construct the production rail deferred and schedule the screen hydration
   callback before ordinary post-refresh Console UI synchronization.
5. Fence hydration on mount state and tolerate navigation teardown.

## Task 3: Measure the real implementation

1. Extend the profiler with a real-production deferred variant that records
   first-interactive and post-hydration full-ready independently.
2. Run 30 balanced fresh-screen warm navigations against eager baseline and the
   real deferred implementation.
3. Retain only if first-interactive median improves at least 15%, full-ready
   median regresses no more than 5%, both input p95 values regress no more than
   10%, and pre-interaction widget work decreases.

## Task 4: Lifecycle and static verification

1. Run fresh-screen, rapid-switch, focus, restore, unmount, and interactive
   Console soak gates.
2. Run focused left-rail/workbench/shell tests, Ruff, `py_compile`, and diff
   checks.
3. Record the distributions, modified files, trade-offs, and independent review
   result in TASK-19505 before closing it.
