---
id: TASK-32118
title: Checkpoint goal progress and verify current artifacts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:18'
updated_date: '2026-09-09 15:04'
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
- [x] #1 Goal reports are bounded and strictly validated; foreign evidence and model-provided authority or verification claims are refused.
- [x] #2 Checkpoint, report, evidence, counters and attempt settlement commit atomically and reject conflicting repeated results.
- [x] #3 No-progress and failure decisions preserve observed work; next-iteration requests retain the objective within finite memory limits.
- [x] #4 Evidence survives original-output pruning through bounded private copies; aggregate payload capacity is reserved before work and settled history removal preserves accounting.
- [x] #5 Automatic completion requires launch-bound verifiers and current checked artifacts; human review remains required unless explicitly disabled at launch. Failed checks or later edits invalidate proof.
- [ ] #6 Actual initial and later model requests include bounded exact selected verifier invocations, target and input references without relying on objective prose or fixture-only knowledge; existing memory, budget and evidence gates remain enforced.
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

Final whole-branch review fix wave (before new code): existing ADR-141 applies; no duplicate ADR. Reproduce the affected setup/selected-resource failures through isolated mounted/native requests, fix all findings in the shared final-review list while preserving owner boundaries, run focused amended regressions and scoped static checks, update user/qualification docs and commit. One independent scoped re-review follows. Root owns final AC/status/notes after approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict bounded reports, runtime-owned CLI/generic observations, exact verifier/input freshness, deterministic progress and fresh later-iteration requests. Schema v17 atomically persists checkpoints/evidence/counters and matching attempt settlement, with pre-admission payload reservations and protected accounting tombstones.

ADR required: yes
ADR path: backlog/decisions/141-native-console-goal-runs.md
Reason: implements the accepted private persistence and evidence contract; no additional ADR needed.

Independent spec and quality review approved implementation 810feb9dfe3 plus exact-selector fix 36bc4dafbe. Regressions cover real CLI execution, stale/newer contradictory results, multiple configurations of one script, legacy launch/checkpoint replay, actual second/third provider requests, rollback, pruning and capacity. Evidence: 60 final new-module passes; affected existing gate 525 passes/two stale schema expectations corrected in an 84-case focused gate; fix gate 107 passes/one fixture correction followed by 11 selection passes (108 unique affected cases; counts overlap). Scoped lint/format and differential checks have no new diagnostics; whitespace clean. Existing Requests dependency warning remains.

Updated design/plan/ADR clarification and real incident lessons. Main files: Agents/goal_models.py, goal_iteration.py, goal_run_service.py; DB/goal_runs.py, automatic_work.py and v17 migration; native observer/script seams; focused Agent/DB tests. Repetition, recovery mutations and UI remain the dependent slices.
<!-- SECTION:NOTES:END -->
