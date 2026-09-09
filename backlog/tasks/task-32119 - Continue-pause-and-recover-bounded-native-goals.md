---
id: TASK-32119
title: Continue pause and recover bounded native goals
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:19'
updated_date: '2026-09-09 07:13'
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
- [ ] #1 Repeated native increments share the original allowance and cannot exceed configured iteration, call, token or elapsed limits.
- [ ] #2 Pause stops successor admission and Stop retains ownership until real execution cleanup; manual chat keeps its reserved capacity.
- [ ] #3 Retryable pre-effect rejection uses bounded retries while denied permission, drift and unknown effects do not blindly retry.
- [ ] #4 Quality review applies to one fresh checkpoint and cannot clear interrupted effects or unknown charges.
- [ ] #5 A real process restart after acceptance dispatches no duplicate operation; even a clean checkpoint requires explicit Resume with original limits.
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
