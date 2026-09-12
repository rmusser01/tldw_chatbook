---
id: TASK-32035
title: Enforce runtime-wide child admission with manual reserves
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 05:20'
updated_date: '2026-09-08 06:11'
labels:
  - agents
  - console
dependencies:
  - TASK-32019
  - TASK-32034
references:
  - backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Per-conversation live handles do not bound simultaneous child work across conversations, and a cancelled handle can free a logical slot while its worker still runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At most six child execution leases are occupied per Console runtime by default, with two slots reserved for manual work and the existing per-conversation cap also enforced.
- [x] #2 Reservations precede handles, rows, and threads; refused or failed starts do not consume spawn budget and have clear retryable copy.
- [x] #3 Spawn, finished-child continuation, inline mode, and surviving children use the same runtime owner, including owned operations still stopping after a terminal run status.
- [x] #4 Limits may be lowered without dropping owners; session or bridge replacement, pruning, and cancellation cannot reset occupied capacity.
- [x] #5 Deterministic cross-conversation races and retained-cleanup tests prove the actual bound and manual reserve.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR.
ADR path: backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md
Reason: implements the approved six-child runtime limit and two manual reserves using TASK-32034 ownership.
1. Add failing concurrent admission and retained-operation tests.
2. Reserve a child owner before conversation handles, rows, or threads across spawn, continuation, inline, and skill paths; unwind failures without consuming spawn allowance.
3. Keep leases until roots and owned operations settle, with live config changes and clear refusal copy.
4. Verify cross-conversation/manual reserves, existing fleet/continuation/bridge behavior, and update effective-setting documentation and review ledger.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-134 shared child admission: six child executions per Console runtime by default, with two slots reserved for manual work. Reservations precede conversation handles, run rows, and worker creation; spawn, continuation, inline/skill dispatch, and survivors share the same owner. Child leases survive terminal rows, pruning, and returned parent turns until root work and owned tool/model operations finish. Current configuration applies before admission without dropping existing owners.

Failure-path tests exposed a second counter in the pure loop: unnamed refused spawns still consumed its allowance after the service refunded them. Added the internal SpawnAdmissionRefusal type so pre-dispatch refusal and failed starts consume neither allowance; genuinely executed failed children retain their existing accounting. Thread construction/start and inline model-start failures unwind exactly once.

Files: Agents/execution_capacity.py, agent_service.py, agent_models.py, agent_runtime.py; new Tests/Chat/test_fleet_runtime_admission.py; extended ownership, boundary, and continuation tests; config comments, user guide, review ledger, and implementation plan. Two reindented multiline warnings preserve identical AST/levels/arguments/counts; only the agent-service diagnostic inventory digest changed. No new schema, dependency, billing, or diagnostic payload.

Verification: 451 targeted service/fleet/ownership/tool/bridge/lifecycle tests and 93 disjoint admission/continuation/pure-loop/boundary tests passed. New files pass Ruff lint/format, existing modified files add no lint findings, changed ranges formatted, scoped whitespace checks pass. Self-review complete. Existing unrelated TTS inventory drift and Console screen-size debt remain documented. No full suite or live-provider run. Logged a separate cold-start private-directory/run-log race candidate found during concurrency probing; it does not affect the admitted child bound.

ADR required: no new ADR; implements backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md. Automatic-chain reservations and wake budgets remain pending under TASK-32036/32037. Added the evidence-backed counter-refund lesson to backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
