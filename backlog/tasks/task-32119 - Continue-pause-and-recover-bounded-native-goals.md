---
id: TASK-32119
title: Continue pause and recover bounded native goals
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 04:19'
updated_date: '2026-10-12 01:49'
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

Current-dev integration (2026-10-11): implemented on a3c47c6c72 through complete code head 8bf7c02e51, with live ADR backlog/decisions/227-native-console-goal-runs.md. Schema 24 and exact recovery preserve current v23 owners; native Goal submission uses runtime custody, shared finite allowance roots and current admitted filesystem execution. Unstarted Pause/Resume now safely dispatches; uncertain accepted work remains fenced. Source history above remains historical.

Current qualification: 36 mounted Goal tests; 81 startup/hydration/review/wrap-up checks; 58 corrected native recovery/scheduling/source cases; 46 authority/storage/evidence checks; 75 existing budget/hooks checks; 16 capacity/real executor checks; current custody/profile rerun 6 passed. Final complete-head receipt and source hashes: pr2563-complete-head-receipt.json and pr2563-complete-head-verification.txt in /Users/macbook-dev/.codex/visualizations/2026/10/10/01a1267e-26ee-7513-9c34-58321c2fef6a/pr-review-wave2. The 59-test storage phase receipt was integrated working-tree evidence, not standalone phase-commit evidence.

New Python files pass Ruff and format; modified shared files introduce no Ruff findings over the pinned base. Screen ratchet passes; pre-existing controller/store excess shrinks and unrelated inherited governance failures are recorded without raising caps. Actual production CSS mounted setup/review was inspected at 80x24 and 160x44. External model trial remains explicitly unqualified/skipped; no full test sweep or cloud spend. Original source F9 location is historical; current canonical Settings is F4. Full integration independent review/latest-base publication remains tracked separately by TASK34797 AC5. See Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md and its task-1 report for exact evidence and limitations.

Resource qualification reopened: the complete-head 62-case receipt passed but reported FD growth 785; earlier mounted/native lanes also warned. Exact database/worker-owner diagnosis is pending while source stays frozen at 8bf7c02e51 for independent review. Do not interpret the preceding closeout as a completed resource-lifetime qualification. Task1 report records this DONE_WITH_CONCERNS status.

Combined independent-review fix 790d456f21: canonical allowance-root descendants and unknown accounting now block checkpoint settlement and late result approval; native Goal service/coordinator/UI/provider/tool callbacks retire only their newly acquired handles through current operation-owned boundaries. Fresh/borrowed real CLI and mounted paths, cancellation until physical completion, in-memory/foreign controls and maintenance drains pass. Supporting targeted lanes: 80 Goal runtime, 96 current budget/hooks/SQLite/maintenance, 36 mounted UI; no resource warnings. The exact post-finalizer fixed probe has zero AgentRuns/Chat/Workspace connections for both actual native paths. Original warning receipts and the explicitly labeled repeated baseline remain historical; fixed receipts use distinct names. ADR227 governs; source IDs/notes retained. New exact committed-head qualification is recorded in pr2563-fixed-head-receipt.json and pr2563-fixed-head-verification.txt, with full commands/limitations in the task-1 report. Status remains In Progress pending root scoped independent re-review; TASK34797 AC5 remains unchecked. No full sweep or live external-model trial.

Current integration closeout: independent complete-head and P1 fix review passed; publication-contract Task2 and its one scoped fix re-review passed with zero blockers. Latest dev7b7193d647 rebase preserved canonical native continuation ownership;25 affected continuation/startup tests passed, with unchanged1033/1033 startup-headroom warning. Actual current preflight passed all derived-artifact guards. Earlier79 byte-bound goal cases,80 runtime/96 owner/36 mounted lanes,37 affected UI and final14 setup cases retain distinct receipts; counts overlap. Goal worker cleanup has zero retained owned DB handles. Current F4, schema24/exact recovery and ADR227 govern; historical source failures and baseline size/style debt stay labelled. No full suite, external model quality or physical terminal UAT was claimed. Root publication/required hosted checks/merge remain separate gates.
<!-- SECTION:NOTES:END -->
