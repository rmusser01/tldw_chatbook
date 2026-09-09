---
id: TASK-32119
title: Continue pause and recover bounded native goals
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 04:19'
updated_date: '2026-09-09 08:19'
labels:
  - agents
  - console
dependencies:
  - TASK-32118
references:
  - backlog/decisions/141-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need autonomous increments to continue within finite allowances and stop or recover without duplicating uncertain effects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated native increments share the original allowance and cannot exceed configured iteration, call, token or elapsed limits.
- [x] #2 Pause stops successor admission and Stop retains ownership until real execution cleanup; manual chat keeps its reserved capacity.
- [x] #3 Retryable pre-effect rejection uses bounded retries while denied permission, drift and unknown effects do not blindly retry.
- [x] #4 Quality review applies to one fresh checkpoint and cannot clear interrupted effects or unknown charges.
- [x] #5 A real process restart after acceptance dispatches no duplicate operation; even a clean checkpoint requires explicit Resume with original limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/141-native-console-goal-runs.md (Accepted)
Reason: implements the approved runtime ownership, durable continuation and distinct quality/recovery contracts; use existing ADR134/135 constraints without a second recovery owner.
1. Read TASK-32119, the Task4 brief and reviewed Task3 handoff, relevant ADRs and testing/live-verification lessons. Add targeted lifecycle, admission, exact review and real-operation backoff tests before implementation.
2. Extend the existing persisted-session hydration owner for stable goal identity and non-activating restoration; preserve actual history/provider state and same-conversation occupancy across UI aliases.
3. Implement event-driven bounded continuation after atomic checkpoints, persisted retry/deadline handling, pause-after-increment and cooperative Stop with physical ownership through cleanup. Preserve shared manual reserves and immutable allowances.
4. Project goal recovery from the one existing trusted runtime audit, retaining fleet behavior. Implement exact checkpoint/artifact quality review separately from typed interrupted-effect closure/resolution; unknown charges never become replay authority.
5. Add an isolated subprocess restart harness that terminates after acceptance and after a clean checkpoint, asserting zero duplicate provider/tool work and explicit Resume with original limits. Cover live authority/settings changes and conservative typed provider error classification.
6. Run the new modules and affected automatic-work/runtime regressions only, then scoped lint/format/whitespace checks. Self-review, document actual cancellation guarantees and API handoff, and commit the scoped slice before independent review. UI/live-model qualification remains Task5.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented runtime-owned bounded continuation, Pause/Stop/Resume, exact result review and separate conservative recovery closure. All iterations retain one immutable automatic-work allowance; goal and fleet share capacity/manual reserves. Stable non-activating conversation hydration and a single typed startup audit prevent duplicate execution. Schema v18 persists retry waits and supports settled stopped payload removal.

Independent review approved specification and quality after two fix rounds. Native step/model-turn/wall exits now pause immediately. Shutdown fences dispatch across awaited reads/preparation/acceptance, drains physical owners even when Stop persistence fails, and retains charges for acceptance already committed. Typed local pre-dispatch rejection is the only retry proof; generic remote failures remain conservative.

Verification: final broad targeted gate 164 passed; final amended scheduling/runtime/goal-and-fleet admission gate 78 passed (overlapping scopes, not additive). Both real child-process restart boundaries passed with zero replacement-runtime operations. Scoped Ruff/format and whitespace checks passed; no new legacy Ruff diagnostics. Existing Requests dependency warning remains. No full suite or live model call was run. Process restarts and controlled race orderings do not certify power-loss or atomic cross-thread cancellation.

Core changes: Agents goal service/models/accounting, Console coordinator/controller/runtime/gateway/hydration, automatic-work and goal DB owners, v18 migration, typed provider rejection contract and focused tests. Accepted ADR: [ADR-141](../decisions/141-native-console-goal-runs.md), preserving ADR-134/135. Plan: Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md. Commits c6f56b4d59, 07d281af2b, 7ae2282683, 35fd24b23c; documentation368b034506. UI notifications, mounted controls and live qualification belong to the following slice.
<!-- SECTION:NOTES:END -->
