---
id: TASK-32118
title: Checkpoint goal progress and verify current artifacts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 04:18'
updated_date: '2026-10-12 01:49'
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

Current-dev integration (2026-10-11): implemented on a3c47c6c72 through complete code head 8bf7c02e51, with live ADR backlog/decisions/227-native-console-goal-runs.md. Schema 24 and exact recovery preserve current v23 owners; native Goal submission uses runtime custody, shared finite allowance roots and current admitted filesystem execution. Unstarted Pause/Resume now safely dispatches; uncertain accepted work remains fenced. Source history above remains historical.

Current qualification: 36 mounted Goal tests; 81 startup/hydration/review/wrap-up checks; 58 corrected native recovery/scheduling/source cases; 46 authority/storage/evidence checks; 75 existing budget/hooks checks; 16 capacity/real executor checks; current custody/profile rerun 6 passed. Final complete-head receipt and source hashes: pr2563-complete-head-receipt.json and pr2563-complete-head-verification.txt in /Users/macbook-dev/.codex/visualizations/2026/10/10/01a1267e-26ee-7513-9c34-58321c2fef6a/pr-review-wave2. The 59-test storage phase receipt was integrated working-tree evidence, not standalone phase-commit evidence.

New Python files pass Ruff and format; modified shared files introduce no Ruff findings over the pinned base. Screen ratchet passes; pre-existing controller/store excess shrinks and unrelated inherited governance failures are recorded without raising caps. Actual production CSS mounted setup/review was inspected at 80x24 and 160x44. External model trial remains explicitly unqualified/skipped; no full test sweep or cloud spend. Original source F9 location is historical; current canonical Settings is F4. Full integration independent review/latest-base publication remains tracked separately by TASK34797 AC5. See Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md and its task-1 report for exact evidence and limitations.

Resource qualification reopened: the complete-head 62-case receipt passed but reported FD growth 785; earlier mounted/native lanes also warned. Exact database/worker-owner diagnosis is pending while source stays frozen at 8bf7c02e51 for independent review. Do not interpret the preceding closeout as a completed resource-lifetime qualification. Task1 report records this DONE_WITH_CONCERNS status.

Combined independent-review fix 790d456f21: canonical allowance-root descendants and unknown accounting now block checkpoint settlement and late result approval; native Goal service/coordinator/UI/provider/tool callbacks retire only their newly acquired handles through current operation-owned boundaries. Fresh/borrowed real CLI and mounted paths, cancellation until physical completion, in-memory/foreign controls and maintenance drains pass. Supporting targeted lanes: 80 Goal runtime, 96 current budget/hooks/SQLite/maintenance, 36 mounted UI; no resource warnings. The exact post-finalizer fixed probe has zero AgentRuns/Chat/Workspace connections for both actual native paths. Original warning receipts and the explicitly labeled repeated baseline remain historical; fixed receipts use distinct names. ADR227 governs; source IDs/notes retained. New exact committed-head qualification is recorded in pr2563-fixed-head-receipt.json and pr2563-fixed-head-verification.txt, with full commands/limitations in the task-1 report. Status remains In Progress pending root scoped independent re-review; TASK34797 AC5 remains unchecked. No full sweep or live external-model trial.

Current integration closeout: independent complete-head and P1 fix review passed; publication-contract Task2 and its one scoped fix re-review passed with zero blockers. Latest dev7b7193d647 rebase preserved canonical native continuation ownership;25 affected continuation/startup tests passed, with unchanged1033/1033 startup-headroom warning. Actual current preflight passed all derived-artifact guards. Earlier79 byte-bound goal cases,80 runtime/96 owner/36 mounted lanes,37 affected UI and final14 setup cases retain distinct receipts; counts overlap. Goal worker cleanup has zero retained owned DB handles. Current F4, schema24/exact recovery and ADR227 govern; historical source failures and baseline size/style debt stay labelled. No full suite, external model quality or physical terminal UAT was claimed. Root publication/required hosted checks/merge remain separate gates.
<!-- SECTION:NOTES:END -->
