---
id: TASK-32118
title: Checkpoint goal progress and verify current artifacts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:18'
updated_date: '2026-09-09 06:23'
labels:
  - agents
  - console
dependencies:
  - TASK-32117
references:
  - backlog/decisions/141-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need retained progress and completion based on actual current evidence rather than unsupported model success reports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Goal reports are bounded and strictly validated; foreign evidence and model-provided authority or verification claims are refused.
- [ ] #2 Checkpoint, report, evidence, counters and attempt settlement commit atomically and reject conflicting repeated results.
- [ ] #3 No-progress and failure decisions preserve observed work; next-iteration requests retain the objective within finite memory limits.
- [ ] #4 Evidence survives original-output pruning through bounded private copies; aggregate payload capacity is reserved before work and settled history removal preserves accounting.
- [ ] #5 Automatic completion requires launch-bound verifiers and current checked artifacts; human review remains required unless explicitly disabled at launch. Failed checks or later edits invalidate proof.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/141-native-console-goal-runs.md (Accepted)
Reason: implements the approved private checkpoint, evidence freshness, bounded memory and durable retention contract.
1. Read the task3 brief, reviewed task2 result/observer contracts and relevant storage/testing lessons; add strict report and runtime-evidence behavior tests first.
2. Implement exact run-owned evidence resolution and bounded manifests/copies, deterministic progress and handoff, and atomic checkpoint/attempt settlement with durable result capacity reservations.
3. Add migration/retention changes only where the accepted contract requires them; protect uncertain work and preserve accounting tombstones on explicit settled-payload removal.
4. Verify real outgoing later-iteration requests, real CLI freshness and failure/rollback/replay cases with targeted tests, scoped lint and required owner checks. Self-review and commit before independent spec/quality review. Repetition and UI remain later tasks.
<!-- SECTION:PLAN:END -->
