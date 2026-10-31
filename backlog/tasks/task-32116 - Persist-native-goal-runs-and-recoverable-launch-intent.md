---
id: TASK-32116
title: Persist native goal runs and recoverable launch intent
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 04:16'
updated_date: '2026-10-12 01:49'
labels:
  - agents
  - console
dependencies: []
references:
  - backlog/decisions/227-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need a durable objective and a single recoverable conversation when starting or reopening autonomous work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identical repeated Start delivery resolves to the same goal, conversation identity and allowance; conflicting payloads are refused.
- [x] #2 Interrupted conversation and workspace provisioning can be reconciled without duplicate history or premature model/tool execution.
- [x] #3 Goal requests reject invalid types, oversized UTF-8 payloads and unbounded limits; stored goal policy and scope remain immutable.
- [x] #4 Reopening real SQLite preserves goal state and the existing automatic-work history; migration retains legacy fleet semantics.
- [x] #5 Goal and chain creation roll back together on failure; concurrent writers respect revision and launch identity constraints.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/227-native-console-goal-runs.md
Reason: implements accepted goal design with current schema, runtime custody, shared allowance roots, filesystem and exact backup ownership.
1. Follow Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md under TASK-34797; preserve original notes as historical source evidence.
2. Integrate only goal-specific changes into current owners; add schema23-to24 and exact recovery qualification.
3. Verify targeted native/runtime/storage/UI boundaries plus real CLI/restart and mounted unstarted Pause/Resume.
4. Record fresh receipts and limitations; retain incomplete checks until actual qualification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict immutable goal requests, AgentRunsDB v16 launch/chain persistence, and recoverable exact-identity conversation/workspace provisioning. Duplicate Start is idempotent; conflicting launches and retargeted bindings are refused; cross-store retries retain committed history. ADR: backlog/decisions/227-native-console-goal-runs.md. Implementation: 1b8ff9288a. Targeted verification: 165 + 39 passed (204 total); scoped lint/format and diff checks passed, unchanged legacy lint and existing RequestsDependencyWarning documented. Independent spec and quality review approved with no blocking findings. No dispatch, repetition or UI is claimed in this task; those remain dependent tasks. Detailed evidence: .superpowers/sdd/2026-09-08-gnhf-inspired-goal-runs/task-1-report.md.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior notes and counts describe historical source only; they do not certify the current integration.

Current-dev integration (2026-10-11): implemented on a3c47c6c72 through complete code head 8bf7c02e51, with live ADR backlog/decisions/227-native-console-goal-runs.md. Schema 24 and exact recovery preserve current v23 owners; native Goal submission uses runtime custody, shared finite allowance roots and current admitted filesystem execution. Unstarted Pause/Resume now safely dispatches; uncertain accepted work remains fenced. Source history above remains historical.

Current qualification: 36 mounted Goal tests; 81 startup/hydration/review/wrap-up checks; 58 corrected native recovery/scheduling/source cases; 46 authority/storage/evidence checks; 75 existing budget/hooks checks; 16 capacity/real executor checks; current custody/profile rerun 6 passed. Final complete-head receipt and source hashes: pr2563-complete-head-receipt.json and pr2563-complete-head-verification.txt in /Users/macbook-dev/.codex/visualizations/2026/10/10/01a1267e-26ee-7513-9c34-58321c2fef6a/pr-review-wave2. The 59-test storage phase receipt was integrated working-tree evidence, not standalone phase-commit evidence.

New Python files pass Ruff and format; modified shared files introduce no Ruff findings over the pinned base. Screen ratchet passes; pre-existing controller/store excess shrinks and unrelated inherited governance failures are recorded without raising caps. Actual production CSS mounted setup/review was inspected at 80x24 and 160x44. External model trial remains explicitly unqualified/skipped; no full test sweep or cloud spend. Original source F9 location is historical; current canonical Settings is F4. Full integration independent review/latest-base publication remains tracked separately by TASK34797 AC5. See Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md and its task-1 report for exact evidence and limitations.

Resource qualification reopened: the complete-head 62-case receipt passed but reported FD growth 785; earlier mounted/native lanes also warned. Exact database/worker-owner diagnosis is pending while source stays frozen at 8bf7c02e51 for independent review. Do not interpret the preceding closeout as a completed resource-lifetime qualification. Task1 report records this DONE_WITH_CONCERNS status.

Combined independent-review fix 790d456f21: canonical allowance-root descendants and unknown accounting now block checkpoint settlement and late result approval; native Goal service/coordinator/UI/provider/tool callbacks retire only their newly acquired handles through current operation-owned boundaries. Fresh/borrowed real CLI and mounted paths, cancellation until physical completion, in-memory/foreign controls and maintenance drains pass. Supporting targeted lanes: 80 Goal runtime, 96 current budget/hooks/SQLite/maintenance, 36 mounted UI; no resource warnings. The exact post-finalizer fixed probe has zero AgentRuns/Chat/Workspace connections for both actual native paths. Original warning receipts and the explicitly labeled repeated baseline remain historical; fixed receipts use distinct names. ADR227 governs; source IDs/notes retained. New exact committed-head qualification is recorded in pr2563-fixed-head-receipt.json and pr2563-fixed-head-verification.txt, with full commands/limitations in the task-1 report. Status remains In Progress pending root scoped independent re-review; TASK34797 AC5 remains unchecked. No full sweep or live external-model trial.

Current integration closeout: independent complete-head and P1 fix review passed; publication-contract Task2 and its one scoped fix re-review passed with zero blockers. Latest dev7b7193d647 rebase preserved canonical native continuation ownership;25 affected continuation/startup tests passed, with unchanged1033/1033 startup-headroom warning. Actual current preflight passed all derived-artifact guards. Earlier79 byte-bound goal cases,80 runtime/96 owner/36 mounted lanes,37 affected UI and final14 setup cases retain distinct receipts; counts overlap. Goal worker cleanup has zero retained owned DB handles. Current F4, schema24/exact recovery and ADR227 govern; historical source failures and baseline size/style debt stay labelled. No full suite, external model quality or physical terminal UAT was claimed. Root publication/required hosted checks/merge remain separate gates.
<!-- SECTION:NOTES:END -->
