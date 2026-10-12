---
id: TASK-32117
title: Execute one bounded native goal iteration with CLI evidence
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:17'
updated_date: '2026-10-12 00:45'
labels:
  - agents
  - console
dependencies:
  - TASK-32116
references:
  - backlog/decisions/227-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need goal iterations to use the existing Console execution path with reliable scope, accounting and observed command results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Only one durably accepted goal attempt may dispatch; invalid or failed authorization causes zero provider/helper/tool calls.
- [x] #2 Goal enablement and finite iteration limits remain independent of fleet settings while sharing runtime ownership and accounting.
- [x] #3 Selected tool scope restricts catalog, runtime, discovered and restored calls at actual dispatch, preserving authorized existing CLI execution.
- [x] #4 Actual script exit, timeout, identity and output evidence survives display formatting and cannot be forged by tool text.
- [x] #5 Fresh goal requests omit prior settled iteration history and preserve provider continuation within the current iteration.
- [x] #6 One runtime startup audit governs goal and fleet coordinators without revoking live owners when a view or service is attached.
- [x] #7 Selected additional read-only source bindings provide a bounded usable native read path; unselected siblings and all writes outside the primary writable binding remain refused under existing permissions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/227-native-console-goal-runs.md
Reason: native goal integration with current custody, schema24 and shared allowance roots.
Follow TASK-34797 and Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md; preserve historical notes; qualify current owners with targeted native, storage, backup and UI verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented one durably admitted native goal iteration through the existing Console/controller/agent and CLI owners. Goal policy and scope remain separate from fleet enablement while sharing accounting/capacity/recovery. Typed script/MCP outcomes survive formatting; exact selected roots and provider/MCP references are revalidated. Initial goal history is fresh; accepted work remains pending the next task's atomic checkpoint.

Implementation commits: 0ad46264124ffd5bbb0e4621aeb03615d00f6e44 and review fix 8597af3df0bc1e4f33ec33cf069e7bd7fb5f3a9d. Independent spec and quality review approved after fixing script/install refusal metadata, MCP/worker exception classification and cleanup ownership, and live disablement at durable acceptance.

Validation: final fix scope 146 passed; 5 manual regressions passed; 45 overlapping final script checks passed. Earlier broad run had 378 passes and one independently reproduced baseline missing-run_id review-hook failure. Existing requests warning remains. Scoped test lint/format and whitespace checks pass; no new differential production lint. Counts overlap and are not a unique aggregate. MCP tests prove local awaited-worker cleanup, not remote command termination. Live model qualification, checkpointing and UI remain subsequent tasks.

ADR required: yes. Existing accepted backlog/decisions/227-native-console-goal-runs.md governs this runtime/authority boundary; no duplicate ADR. Detailed evidence and review artifacts are in the plan's local SDD workspace.

Final whole-branch review fix 525c61a944 and independent scoped re-review at 052fd4b95f address all five final findings with no new issues. Selected source bindings now reach actual permission-gated fs_read/fs_list; native tests cover model-visible reads, primary-only writes, unselected siblings, instruction isolation and changed registry/filesystem authority.

Verification: 279 affected passes, 1 intentional live skip and 1 proven unchanged missing-run_id test failure. The 20 strengthened setup/source passes and 2 geometry/2 cleanup followups overlap. Scoped static checks pass with zero added legacy diagnostics. Deterministic real CLI execution still exits 7→0 across 2 increments and 5 calls with an unchanged verifier and the inspected corrected file/diff. Three separate baseline diagnostic failures and unsuccessful historical local-model trials remain disclosed; no full suite or further model calls.

ADR required: yes. Existing accepted backlog/decisions/227-native-console-goal-runs.md was clarified before this fix; no new stored fields, schema or execution/permission owner. Durable final review, exact evidence and limits: Docs/superpowers/reviews/2026-09-09-goal-runs-final-review.md and Docs/superpowers/reviews/2026-09-09-goal-runs-qualification.md, with QA under Docs/superpowers/qa/native-goals/task5/.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior notes and counts describe historical source only; they do not certify the current integration.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior evidence remains historical.

Current-dev integration (2026-10-11): implemented on a3c47c6c72 through complete code head 8bf7c02e51, with live ADR backlog/decisions/227-native-console-goal-runs.md. Schema 24 and exact recovery preserve current v23 owners; native Goal submission uses runtime custody, shared finite allowance roots and current admitted filesystem execution. Unstarted Pause/Resume now safely dispatches; uncertain accepted work remains fenced. Source history above remains historical.

Current qualification: 36 mounted Goal tests; 81 startup/hydration/review/wrap-up checks; 58 corrected native recovery/scheduling/source cases; 46 authority/storage/evidence checks; 75 existing budget/hooks checks; 16 capacity/real executor checks; current custody/profile rerun 6 passed. Final complete-head receipt and source hashes: pr2563-complete-head-receipt.json and pr2563-complete-head-verification.txt in /Users/macbook-dev/.codex/visualizations/2026/10/10/01a1267e-26ee-7513-9c34-58321c2fef6a/pr-review-wave2. The 59-test storage phase receipt was integrated working-tree evidence, not standalone phase-commit evidence.

New Python files pass Ruff and format; modified shared files introduce no Ruff findings over the pinned base. Screen ratchet passes; pre-existing controller/store excess shrinks and unrelated inherited governance failures are recorded without raising caps. Actual production CSS mounted setup/review was inspected at 80x24 and 160x44. External model trial remains explicitly unqualified/skipped; no full test sweep or cloud spend. Original source F9 location is historical; current canonical Settings is F4. Full integration independent review/latest-base publication remains tracked separately by TASK34797 AC5. See Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md and its task-1 report for exact evidence and limitations.

Resource qualification reopened: the complete-head 62-case receipt passed but reported FD growth 785; earlier mounted/native lanes also warned. Exact database/worker-owner diagnosis is pending while source stays frozen at 8bf7c02e51 for independent review. Do not interpret the preceding closeout as a completed resource-lifetime qualification. Task1 report records this DONE_WITH_CONCERNS status.

Combined independent-review fix 790d456f21: canonical allowance-root descendants and unknown accounting now block checkpoint settlement and late result approval; native Goal service/coordinator/UI/provider/tool callbacks retire only their newly acquired handles through current operation-owned boundaries. Fresh/borrowed real CLI and mounted paths, cancellation until physical completion, in-memory/foreign controls and maintenance drains pass. Supporting targeted lanes: 80 Goal runtime, 96 current budget/hooks/SQLite/maintenance, 36 mounted UI; no resource warnings. The exact post-finalizer fixed probe has zero AgentRuns/Chat/Workspace connections for both actual native paths. Original warning receipts and the explicitly labeled repeated baseline remain historical; fixed receipts use distinct names. ADR227 governs; source IDs/notes retained. New exact committed-head qualification is recorded in pr2563-fixed-head-receipt.json and pr2563-fixed-head-verification.txt, with full commands/limitations in the task-1 report. Status remains In Progress pending root scoped independent re-review; TASK34797 AC5 remains unchecked. No full sweep or live external-model trial.
<!-- SECTION:NOTES:END -->
