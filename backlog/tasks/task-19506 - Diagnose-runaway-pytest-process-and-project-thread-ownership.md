---
id: TASK-19506
title: Diagnose runaway pytest process and project thread ownership
status: Done
assignee:
  - '@codex'
created_date: '2026-08-21'
updated_date: '2026-08-21 20:15'
labels:
  - testing
  - performance
  - diagnostics
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Determine whether the observed long-running high-CPU pytest process represents a current application or fixture lifecycle defect before changing AgentService, fleet, or global test teardown behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The suspected phase-zero tests run with short per-test bounds, verbose current-test identity, RSS sampling, and classified project-owned thread names and counts
- [x] #2 Any reproduction is prefix-bisected and captures Python stacks plus producer/task state before termination
- [x] #3 Suspected fixture teardown tests assert project-owned thread counts return near the measured baseline
- [x] #4 An unbounded producer-signal wait is changed only when a deterministic test proves the producer can fail or exit without signaling
- [x] #5 A concrete defect receives an atomic fix task and RED test; otherwise this task closes with captured evidence and no speculative runtime changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add an opt-in pytest diagnostics plugin that records current-test identity, RSS, Python thread stacks, and classified production-owned tool/fleet thread deltas without changing the default suite.
2. Add RED unit tests for thread classification, survivor detection, and bounded cleanup behavior, then implement the smallest plugin.
3. Run the suspected AGENTS.md project-instruction, AgentService timeout, review-scope, and fleet lifecycle tests with -vv, a 30-second per-test timeout, periodic RSS samples, and strict post-teardown ownership checks.
4. Prefix-bisect and capture producer/task state only if the instrumented run reproduces a hang or survivor; otherwise retain the no-reproduction evidence and avoid runtime lifecycle changes.
5. Record the exact command, raw diagnostic report, conclusions, and task closeout.

ADR required: no
ADR path: N/A
Reason: this task adds opt-in test diagnostics and evidence only; it does not change runtime ownership, service contracts, or application lifecycle.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added an explicitly loaded pytest diagnostics plugin that continuously flushes current node/phase, RSS, total threads, classified `tool-*`/`fleet-*` threads, teardown survivors, and threshold-triggered Python stacks to JSONL. Strict ownership checks apply only when the plugin is requested; the default suite is unchanged.
- After rebasing onto dev at `5f720a404`, the suspected AGENTS.md/project-instruction, AgentService, review-scope, and fleet surface passed 380/380 tests in 78.06 seconds under `-vv --timeout=30`. The report records all 380 call outcomes as passed. RSS was 86.17 MiB initially, had an observed peak of 226.62 MiB, and ended at 226.16 MiB; sampled total threads peaked at five, sampled project-owned threads at one, and all 380 teardowns returned to the zero-owned-thread baseline within three seconds (longest explicit ownership settle: 2.25 seconds).
- No hang or survivor reproduced, so there was nothing honest to prefix-bisect and no producer/task state to repair. Review hardening added a stable-empty window, object-identity baselines, a final session inventory attributed to session finish, first-observation survivor attribution, phase-pinned stack labels, a terminal-record fence, explicit pytest outcomes/settle durations, and retained-stack path normalization. The stack path was independently exercised with a low-threshold control and retained setup/call snapshots. The current TASK-3316 replacement plus the repository's unbounded-background-signal AST guard passed 8/8 under a ten-second bound; no application wait was changed.
- Revalidated and terminated only stale app PID 79581 from the obsolete `rag-15810-hang` worktree (five days old, 604 cumulative CPU minutes, 233,632 KiB RSS). No files or worktrees were removed, and active pytest processes were left alone.
- Raw reports and the exact commands/conclusions are retained in `Docs/superpowers/qa/runaway-pytest-2026-08-21/`. No runtime AgentService, fleet, fixture, or global teardown change was justified.
- ADR required: no. This is opt-in diagnostic tooling and evidence, with no runtime ownership or lifecycle change.
<!-- SECTION:NOTES:END -->
