---
id: TASK-32120
title: Expose native goal controls and qualify CLI workflow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:19'
updated_date: '2026-10-12 00:45'
labels:
  - agents
  - console
dependencies:
  - TASK-32119
references:
  - backlog/decisions/227-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need to start and inspect autonomous goals in Console and see a real failed check become a verified corrected result.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real Console controls start, pause, stop, resume and review the exact goal while preserving ordinary drafts and navigation lifetime.
- [x] #2 Canonical F4 Settings expose independent goal enablement and finite policy; setup states actual executor authority and verification scope.
- [x] #3 A real controller and agent path runs a trusted CLI verifier, corrects an editable fixture and passes the unchanged check across at least two increments.
- [x] #4 Controls and evidence remain usable at 80x24 and the ordinary terminal size; displayed actions and footer hints are implemented.
- [x] #5 Targeted integration, migration, privacy and diagnostics checks cover the delivered feature; documented live-provider evidence or an explicitly recorded unavailable prerequisite remains honest.
- [x] #6 Every launch field shown at confirmation matches the submitted immutable request across awaited validation, and validation failure restores usable controls without stale selections.
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
Implemented Console goal controls under accepted ADR-227 (backlog/decisions/227-native-console-goal-runs.md): immutable setup from canonical owners, app-runtime-owned launch and restart Resume, exact checkpoint/run/evidence controls, bounded Older/Newer history and settled payload removal. Canonical F9 Console behavior saves independent finite [agents] goal policy atomically while preserving ordinary drafts and fleet settings.

The real controller/agent/SQLite/CLI fixture failed its trusted check (exit7), made an authorized fs_edit and passed the unchanged verifier (exit0) across two increments/five model calls. Production-style modal captures and mounted actions cover80x24/160x44. The configured local Qwen endpoint made two bounded attempts but did not call tools or correct the file; malformed/no-progress reports were refused. This is documented negative live evidence, not a successful model demonstration.

Independent Task5 specification and quality review approved implementation dd943d03143 and fix e98f6c68aa after allfour restart/catalog/summary/history findings were addressed. One nonblocking stale-validation form-state finding is carried into the final whole-branch fix wave and remains explicitly tracked in Docs/superpowers/reviews/2026-09-09-goal-runs-qualification.md.

Evidence:98 affected passes/1opt-in live skip,10selected controls passes,6setup passes; fix gate84passes and6strengthened followups. Counts overlap. Scoped Ruff/format and whitespace pass, with no added lint tuples in touched legacy owners. Architecture/privacy/migration gate97passed/3proven pre-goal diagnostic failures/1historical-commit skip; global diagnostics remain red. No full suite. Commands, actual traces, rendered controls, limitations and baseline proofs are retained in Docs/superpowers/qa/native-goals/task5/README.md and the qualification report. Main owners: UI/Console_Modules/goals.py, Console goal widgets/settings helper, Chat/console_goal_runs.py, existing runtime/skill/settings/command owners and targeted Tests/UI,Tests/Chat.

The requested successful live corrected artifact was not obtained from the configured model; the deterministic provider demonstrates the real CLI correction while the actual live failures remain preserved. Whole-branch review/integration reconciliation remains separate from this task-scoped approval. No new ADR beyond ADR-227; applicable implementation decisions and limits are linked in the qualification report.

Final whole-branch review fix 525c61a944 and independent scoped re-review at 052fd4b95f address all five final findings with no new issues. All launch fields are frozen before awaited validation, final Start persists the displayed request and rejected/stale validation restores only current enabled controls. Mounted all-field and cleanup regressions verify the service result.

Verification: 279 affected passes, 1 intentional live skip and 1 proven unchanged missing-run_id test failure. The 20 strengthened setup/source passes and 2 geometry/2 cleanup followups overlap. Scoped static checks pass with zero added legacy diagnostics. Deterministic real CLI execution still exits 7→0 across 2 increments and 5 calls with an unchanged verifier and the inspected corrected file/diff. Three separate baseline diagnostic failures and unsuccessful historical local-model trials remain disclosed; no full suite or further model calls.

ADR required: yes. Existing accepted backlog/decisions/227-native-console-goal-runs.md was clarified before this fix; no new stored fields, schema or execution/permission owner. Durable final review, exact evidence and limits: Docs/superpowers/reviews/2026-09-09-goal-runs-final-review.md and Docs/superpowers/reviews/2026-09-09-goal-runs-qualification.md, with QA under Docs/superpowers/qa/native-goals/task5/.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior notes and counts describe historical source only; they do not certify the current integration.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior notes are historical source qualification only.

Current-dev integration (2026-10-11): implemented on a3c47c6c72 through complete code head 8bf7c02e51, with live ADR backlog/decisions/227-native-console-goal-runs.md. Schema 24 and exact recovery preserve current v23 owners; native Goal submission uses runtime custody, shared finite allowance roots and current admitted filesystem execution. Unstarted Pause/Resume now safely dispatches; uncertain accepted work remains fenced. Source history above remains historical.

Current qualification: 36 mounted Goal tests; 81 startup/hydration/review/wrap-up checks; 58 corrected native recovery/scheduling/source cases; 46 authority/storage/evidence checks; 75 existing budget/hooks checks; 16 capacity/real executor checks; current custody/profile rerun 6 passed. Final complete-head receipt and source hashes: pr2563-complete-head-receipt.json and pr2563-complete-head-verification.txt in /Users/macbook-dev/.codex/visualizations/2026/10/10/01a1267e-26ee-7513-9c34-58321c2fef6a/pr-review-wave2. The 59-test storage phase receipt was integrated working-tree evidence, not standalone phase-commit evidence.

New Python files pass Ruff and format; modified shared files introduce no Ruff findings over the pinned base. Screen ratchet passes; pre-existing controller/store excess shrinks and unrelated inherited governance failures are recorded without raising caps. Actual production CSS mounted setup/review was inspected at 80x24 and 160x44. External model trial remains explicitly unqualified/skipped; no full test sweep or cloud spend. Original source F9 location is historical; current canonical Settings is F4. Full integration independent review/latest-base publication remains tracked separately by TASK34797 AC5. See Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md and its task-1 report for exact evidence and limitations.

Resource qualification reopened: the complete-head 62-case receipt passed but reported FD growth 785; earlier mounted/native lanes also warned. Exact database/worker-owner diagnosis is pending while source stays frozen at 8bf7c02e51 for independent review. Do not interpret the preceding closeout as a completed resource-lifetime qualification. Task1 report records this DONE_WITH_CONCERNS status.

Combined independent-review fix 790d456f21: canonical allowance-root descendants and unknown accounting now block checkpoint settlement and late result approval; native Goal service/coordinator/UI/provider/tool callbacks retire only their newly acquired handles through current operation-owned boundaries. Fresh/borrowed real CLI and mounted paths, cancellation until physical completion, in-memory/foreign controls and maintenance drains pass. Supporting targeted lanes: 80 Goal runtime, 96 current budget/hooks/SQLite/maintenance, 36 mounted UI; no resource warnings. The exact post-finalizer fixed probe has zero AgentRuns/Chat/Workspace connections for both actual native paths. Original warning receipts and the explicitly labeled repeated baseline remain historical; fixed receipts use distinct names. ADR227 governs; source IDs/notes retained. New exact committed-head qualification is recorded in pr2563-fixed-head-receipt.json and pr2563-fixed-head-verification.txt, with full commands/limitations in the task-1 report. Status remains In Progress pending root scoped independent re-review; TASK34797 AC5 remains unchecked. No full sweep or live external-model trial.
<!-- SECTION:NOTES:END -->
