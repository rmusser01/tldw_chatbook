---
id: TASK-32118
title: Checkpoint goal progress and verify current artifacts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:18'
updated_date: '2026-10-11 22:37'
labels:
  - agents
  - console
dependencies:
  - TASK-32117
references:
  - backlog/decisions/227-native-console-goal-runs.md
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
- [x] #6 Actual initial and later model requests include bounded exact selected verifier invocations, target and input references without relying on objective prose or fixture-only knowledge; existing memory, budget and evidence gates remain enforced.
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
Implemented strict bounded reports, runtime-owned CLI/generic observations, exact verifier/input freshness, deterministic progress and fresh later-iteration requests. Schema v17 atomically persists checkpoints/evidence/counters and matching attempt settlement, with pre-admission payload reservations and protected accounting tombstones.

ADR required: yes
ADR path: backlog/decisions/227-native-console-goal-runs.md
Reason: implements the accepted private persistence and evidence contract; no additional ADR needed.

Independent spec and quality review approved implementation 810feb9dfe3 plus exact-selector fix 36bc4dafbe. Regressions cover real CLI execution, stale/newer contradictory results, multiple configurations of one script, legacy launch/checkpoint replay, actual second/third provider requests, rollback, pruning and capacity. Evidence: 60 final new-module passes; affected existing gate 525 passes/two stale schema expectations corrected in an 84-case focused gate; fix gate 107 passes/one fixture correction followed by 11 selection passes (108 unique affected cases; counts overlap). Scoped lint/format and differential checks have no new diagnostics; whitespace clean. Existing Requests dependency warning remains.

Updated design/plan/ADR clarification and real incident lessons. Main files: Agents/goal_models.py, goal_iteration.py, goal_run_service.py; DB/goal_runs.py, automatic_work.py and v17 migration; native observer/script seams; focused Agent/DB tests. Repetition, recovery mutations and UI remain the dependent slices.

Final whole-branch review fix 525c61a944 and independent scoped re-review at 052fd4b95f address all five final findings with no new issues. Canonical skill owners now project the exact selected verifier invocation, target and checked input into initial/later native requests under the 128 KiB launch ceiling, separate 16 KiB memory bound and existing request budget. The real CLI fixture derives its command from outgoing context.

Verification: 279 affected passes, 1 intentional live skip and 1 proven unchanged missing-run_id test failure. The 20 strengthened setup/source passes and 2 geometry/2 cleanup followups overlap. Scoped static checks pass with zero added legacy diagnostics. Deterministic real CLI execution still exits 7→0 across 2 increments and 5 calls with an unchanged verifier and the inspected corrected file/diff. Three separate baseline diagnostic failures and unsuccessful historical local-model trials remain disclosed; no full suite or further model calls.

ADR required: yes. Existing accepted backlog/decisions/227-native-console-goal-runs.md was clarified before this fix; no new stored fields, schema or execution/permission owner. Durable final review, exact evidence and limits: Docs/superpowers/reviews/2026-09-09-goal-runs-final-review.md and Docs/superpowers/reviews/2026-09-09-goal-runs-qualification.md, with QA under Docs/superpowers/qa/native-goals/task5/.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior notes and counts describe historical source only; they do not certify the current integration.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior notes are historical source qualification only.
<!-- SECTION:NOTES:END -->
