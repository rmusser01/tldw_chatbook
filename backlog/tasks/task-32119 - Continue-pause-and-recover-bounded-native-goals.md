---
id: TASK-32119
title: Continue pause and recover bounded native goals
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:19'
updated_date: '2026-10-11 22:37'
labels:
  - agents
  - console
dependencies:
  - TASK-32118
references:
  - backlog/decisions/227-native-console-goal-runs.md
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
ADR path: backlog/decisions/227-native-console-goal-runs.md
Reason: accepted native goals integrated with current storage, runtime custody and authority.
Follow TASK-34797 and Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md; preserve historical evidence, test targeted current owners and record fresh receipts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented runtime-owned bounded continuation, Pause/Stop/Resume, exact result review and separate conservative recovery closure. All iterations retain one immutable automatic-work allowance; goal and fleet share capacity/manual reserves. Stable non-activating conversation hydration and a single typed startup audit prevent duplicate execution. Schema v18 persists retry waits and supports settled stopped payload removal.

Independent review approved specification and quality after two fix rounds. Native step/model-turn/wall exits now pause immediately. Shutdown fences dispatch across awaited reads/preparation/acceptance, drains physical owners even when Stop persistence fails, and retains charges for acceptance already committed. Typed local pre-dispatch rejection is the only retry proof; generic remote failures remain conservative.

Verification: final broad targeted gate 164 passed; final amended scheduling/runtime/goal-and-fleet admission gate 78 passed (overlapping scopes, not additive). Both real child-process restart boundaries passed with zero replacement-runtime operations. Scoped Ruff/format and whitespace checks passed; no new legacy Ruff diagnostics. Existing Requests dependency warning remains. No full suite or live model call was run. Process restarts and controlled race orderings do not certify power-loss or atomic cross-thread cancellation.

Core changes: Agents goal service/models/accounting, Console coordinator/controller/runtime/gateway/hydration, automatic-work and goal DB owners, v18 migration, typed provider rejection contract and focused tests. Accepted ADR: [ADR-227](../decisions/227-native-console-goal-runs.md), preserving ADR-134/135. Plan: Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md. Commits c6f56b4d59, 07d281af2b, 7ae2282683, 35fd24b23c; documentation368b034506. UI notifications, mounted controls and live qualification belong to the following slice.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior notes and counts describe historical source only; they do not certify the current integration.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior notes are historical source qualification only.
<!-- SECTION:NOTES:END -->
