---
id: TASK-34796
title: Start native goal setup from the Console composer
status: Done
assignee:
  - '@codex'
created_date: '2026-10-11 22:38'
updated_date: '2026-10-12 01:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the accepted native-goal composer entry with current Console command and draft custody so users can review goal setup without accidental dispatch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing /goal with a task description opens native goal setup with the complete description prefilled, including multiline text; bare /goal opens the ordinary setup form.
- [x] #2 The slash popup discovers and completes /goal, and command dispatch consumes it without sending the command as an ordinary model message; existing literal/paste behavior is preserved.
- [x] #3 Goal enablement, selected workspace/provider resources, finite limits and explicit launch review remain authoritative; no goal work dispatches before Start confirmation.
- [x] #4 Cancellation and setup refusal preserve the composer draft, and confirmed launch persists the described objective through the existing goal service.
- [x] #5 Targeted composer/setup regressions and user documentation cover the new entry point.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/227-native-console-goal-runs.md
Reason: current-dev integration amendment to the accepted native-goal decision.
Follow Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md; preserve historical source evidence and qualify actual current owners.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Historical source TASK-32194 implementation record (not a claim of current-dev verification):

Added /goal [task description] to the existing Console command registry, popup and dispatcher. The complete description prepopulates existing native goal setup; bare /goal retains ordinary setup. Existing provider/resource selection, enablement, finite policy and explicit Review/Start remain authoritative. Cancel and refusal preserve the draft, and recognized /goal commands never become ordinary chat sends. No new storage or runtime ownership.

Verification: 118 affected composer/parser/popup/goal setup/navigation tests passed, followed by 38 overlapping final tests after mechanical lint cleanup. Six-file Ruff and format pass; the existing ChatScreen owner has 150 baseline/current diagnostics with no additions. Whitespace checks pass. Independent scoped review approved with no actionable findings. The initial two mounted RED failures reproduced missing popup/dispatch support; an intermediate test-fixture error was corrected by inspecting the saved goal through service.get rather than its body-free history entry. Existing RequestsDependencyWarning remains. No full-suite sweep or live model call.

ADR required: no new ADR. Existing backlog/decisions/141-native-console-goal-runs.md governs the reused launch boundary. User documentation is updated in Docs/User_Guide/console/agent-runs-and-tools.md. Exact gate output, independent review and limits are retained in Docs/superpowers/qa/native-goals/composer/README.md and linked from the original qualification report. The requested single draft PR against dev includes all session changes; full-branch prerequisite integration remains separate from this bounded entry-point approval.

Identifier reconciliation: initially filed as TASK-32193; the pre-PR scan found a concurrent committed shared-search task using that number, so this new task was renumbered to TASK-32194 before its first commit. The independent review retains the former ID as history; code and test evidence are unchanged. ADR-141 also conflicts with a later Library retirement ADR; our goal ADR was introduced first in 77bc58dc17 (2026-09-08), while the Library ADR first appears in 5caa634c2d (2026-09-09). Preserve the earlier goal ADR and disclose the remaining cross-branch conflict for integration.

Current integration ruling: preserve the earlier source goal ADR-141 provenance (77bc58dc17, 2026-09-08) and later Library ADR-141 history without rewriting foreign branches; the globally reserved live decision is backlog/decisions/227-native-console-goal-runs.md.

Current-dev integration (2026-10-11): implemented on a3c47c6c72 through complete code head 8bf7c02e51, with live ADR backlog/decisions/227-native-console-goal-runs.md. Schema 24 and exact recovery preserve current v23 owners; native Goal submission uses runtime custody, shared finite allowance roots and current admitted filesystem execution. Unstarted Pause/Resume now safely dispatches; uncertain accepted work remains fenced. Source history above remains historical.

Current qualification: 36 mounted Goal tests; 81 startup/hydration/review/wrap-up checks; 58 corrected native recovery/scheduling/source cases; 46 authority/storage/evidence checks; 75 existing budget/hooks checks; 16 capacity/real executor checks; current custody/profile rerun 6 passed. Final complete-head receipt and source hashes: pr2563-complete-head-receipt.json and pr2563-complete-head-verification.txt in /Users/macbook-dev/.codex/visualizations/2026/10/10/01a1267e-26ee-7513-9c34-58321c2fef6a/pr-review-wave2. The 59-test storage phase receipt was integrated working-tree evidence, not standalone phase-commit evidence.

New Python files pass Ruff and format; modified shared files introduce no Ruff findings over the pinned base. Screen ratchet passes; pre-existing controller/store excess shrinks and unrelated inherited governance failures are recorded without raising caps. Actual production CSS mounted setup/review was inspected at 80x24 and 160x44. External model trial remains explicitly unqualified/skipped; no full test sweep or cloud spend. Original source F9 location is historical; current canonical Settings is F4. Full integration independent review/latest-base publication remains tracked separately by TASK34797 AC5. See Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md and its task-1 report for exact evidence and limitations.

Resource qualification reopened: the complete-head 62-case receipt passed but reported FD growth 785; earlier mounted/native lanes also warned. Exact database/worker-owner diagnosis is pending while source stays frozen at 8bf7c02e51 for independent review. Do not interpret the preceding closeout as a completed resource-lifetime qualification. Task1 report records this DONE_WITH_CONCERNS status.

Combined independent-review fix 790d456f21: canonical allowance-root descendants and unknown accounting now block checkpoint settlement and late result approval; native Goal service/coordinator/UI/provider/tool callbacks retire only their newly acquired handles through current operation-owned boundaries. Fresh/borrowed real CLI and mounted paths, cancellation until physical completion, in-memory/foreign controls and maintenance drains pass. Supporting targeted lanes: 80 Goal runtime, 96 current budget/hooks/SQLite/maintenance, 36 mounted UI; no resource warnings. The exact post-finalizer fixed probe has zero AgentRuns/Chat/Workspace connections for both actual native paths. Original warning receipts and the explicitly labeled repeated baseline remain historical; fixed receipts use distinct names. ADR227 governs; source IDs/notes retained. New exact committed-head qualification is recorded in pr2563-fixed-head-receipt.json and pr2563-fixed-head-verification.txt, with full commands/limitations in the task-1 report. Status remains In Progress pending root scoped independent re-review; TASK34797 AC5 remains unchecked. No full sweep or live external-model trial.

Current integration closeout: independent complete-head and P1 fix review passed; publication-contract Task2 and its one scoped fix re-review passed with zero blockers. Latest dev7b7193d647 rebase preserved canonical native continuation ownership;25 affected continuation/startup tests passed, with unchanged1033/1033 startup-headroom warning. Actual current preflight passed all derived-artifact guards. Earlier79 byte-bound goal cases,80 runtime/96 owner/36 mounted lanes,37 affected UI and final14 setup cases retain distinct receipts; counts overlap. Goal worker cleanup has zero retained owned DB handles. Current F4, schema24/exact recovery and ADR227 govern; historical source failures and baseline size/style debt stay labelled. No full suite, external model quality or physical terminal UAT was claimed. Root publication/required hosted checks/merge remain separate gates.
<!-- SECTION:NOTES:END -->
