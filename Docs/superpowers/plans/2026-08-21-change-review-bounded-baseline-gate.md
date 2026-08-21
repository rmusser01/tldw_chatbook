# Change Review bounded baseline gate implementation plan

**Goal:** Keep optional Change Review observational and non-wedging while ensuring every potentially mutating tool waits for an established baseline, after existing preparation and permission decisions.

**Architecture:** `agent_runtime` owns generic call ordering and effective review verdicts. It gains one post-review/pre-dispatch hook receiving only calls that may proceed. The Console bridge supplies a Change Review hook that bypasses a fixed pure-runtime table and otherwise asks the app-owned root coordinator for one bounded all-roots wait. Timeout invalidation remains coordinator/tracker state; permission, stamps, refusals, audit, and invocation remain unchanged.

**Tech stack:** Python 3.11+, Textual, pytest, SQLite, fixed daemon workers.

ADR required: no
ADR path: `backlog/decisions/077-change-review-consent-and-asynchronous-finalization.md`
Reason: ADR-077 already specifies this cross-module interface and recovery policy.

## Task 1: Establish runtime ordering

1. Add RED `agent_runtime` tests proving preparation runs before review, review runs before the new dispatch gate, an explicit non-proceed call is omitted from the gate, and a preparation deferral reaches neither review nor gate.
2. Add a `before_tool_dispatch` dependency to `LoopDeps` and invoke it once with effective-`proceed` calls after review handling and before the per-call dispatch loop.
3. Factor effective verdict lookup so the gate filter and dispatch loop cannot diverge on call-id versus name fallback.
4. Preserve the existing raised-review behavior, then prove the new gate still runs before that fail-open policy dispatches calls.

## Task 2: Add the bounded Console baseline hook

1. Add RED bridge tests with a controllable reservation for the fixed pure bypass table: `find_tools`, `load_tools`, `skill_file`, `search_run_log`, `run_log_stats`, `run_log_slice`, `wait_agents`, and `check_agents` do not wait; spawn/install/script/message/provider/skill/unknown calls do.
2. Replace the current wrapper around `review_tool_calls` with `before_tool_dispatch`; use a three-second constant and wait once per remaining batch across the reservation's full root set.
3. On timeout, append one content-free alias-safe warning for the turn and let dispatch continue through its existing permission/invocation path.
4. Keep the legacy tracker compatibility path bounded with the same timeout while production coordinator ownership remains authoritative.

## Task 3: Make timeout invalidation durable

1. Add RED tracker tests proving unresolved roots become irrevocably errored after timeout and a late discovery/B result cannot restore their baseline or produce a normal diff.
2. Preserve completed roots in a partially ready handle and emit per-root tracking errors only for unresolved roots.
3. Ensure finalization of successful, failed, and cancelled runs publishes the timeout error through the existing atomic batch path.

## Task 4: Degrade survivor overlap honestly

1. Add barrier tests for a predecessor survivor holding a root lane while its successor waits for B and times out.
2. Add coordinator timeout handling that marks the current reservation invalid, marks the open predecessor survivor window invalid, and publishes tracking-error-only rows for both without claiming file counts.
3. Track a degraded epoch per canonical root. Registrations during the epoch receive bounded tracking-error tokens rather than starting a misleading baseline.
4. Retain the epoch until timed-out mutation work has finalized and every known survivor key has settled; then prove the next reservation establishes a fresh baseline and records changes normally.

## Task 5: Verify

1. Run focused runtime preparation/review/gate tests.
2. Run the complete Change Review coordinator/tracker/runtime suites and mounted Console three-turn regression.
3. Run Ruff on every changed Python file and `git diff --check`.
4. Re-review timeout races, pure bypass classification, degraded-epoch cleanup, and post-shutdown behavior before marking TASK-19503 Done.
