---
id: TASK-33432
title: Progress-triggered bounded supervisor wakes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-29 18:11'
updated_date: '2026-09-29 20:13'
labels:
  - agents
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let committed child progress request a supervisor turn through the existing automatic-work scheduler and shared finite budgets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Progress intake uses metadata-only IDs and a distinct claim source; claiming never consumes a report and duplicate notices cannot repeatedly wake the supervisor.
- [x] #2 Progress and completion wakes share existing coalescing, fairness, manual priority, slots, chain generations, wall and spending limits; busy supervisors are not interrupted.
- [x] #3 Preacceptance abort releases claims; accepted or uncertain attempts never replay automatically after restart; a wake asks for a fresh message read.
- [x] #4 Targeted mixed-source, duplicate, exhaustion, collection-race, restart and unchanged-completion checks pass; backup schema covers new persistent state.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend existing AgentRuns wake attempt and claim schema to identify completion and progress sources, with migration and installed recovery catalog parity. 2. Pin live-progress claim scope, shared generation budgets, duplicate protection, preacceptance rollback and restart nonreplay. 3. Add metadata-only committed enqueue intake and pending-ID revalidation to the existing scheduler, retaining its coalescing, manual priority and slots. 4. Wake with bounded report IDs and request a fresh read; keep progress claims and completion receipts distinct. 5. Run targeted ledger/scheduler/integration checks, static analysis and independent review. ADR required: yes; ADR path: backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md; reason: new automatic wake source sharing existing finite authority.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented progress-triggered supervisor wakes through the existing ConsoleFleetWakeCoordinator and AgentRuns automatic-work ledger. Committed enqueue observers publish message/source IDs only; native-owner and current pending-ID checks precede admission. Notices ask for a fresh read_agent_messages call without copying bodies or granting consent.

Progress claims remain distinct from terminal survivor claims and share the same coalescing, conversation fairness, manual reserve/priority, slots and three-generation chain budget. Pure progress never stamps child completion. Preacceptance collection or close releases the prepared claims and generation; accepted/uncertain work keeps the existing review fence. Finite claim workers are shielded and settled before close cleanup decides whether a refund is proven, and progress registry owners are read through locked snapshots.

AgentRuns schema v23 adds attempt cause/message IDs and exact-source progress claims. Frozen v18/v21/v22 recovery catalogs and the linear 18→21→22→23 migration remain qualified; the restricted migration authorizer admits only SQLite's exact named quick_check needed by the new checked column. Temporary report/source claims stay process-local while existing lifecycle/budget metadata stays durable. Same-process Save retains those claims; a fresh restart fences an old completed temporary progress/mixed chain whose stored empty ID list has lost that claim set. Saved reports remain manually readable, with no new cross-database promotion protocol.

Evidence: initial progress claim tests failed without the new API; postcommit close/dispose barriers reproduced leaked prepared attempts; completed temporary Save/restart reproduced an unfenced active chain. Final mixed targeted run passed 64 checks in 230.05s (/private/tmp/tldw-progress-final-review-output.log). A final physical-custody and accepted-cancellation run passed 5 checks in 18.85s (/private/tmp/tldw-progress-custody-final-output.log). Core wake/ledger/new tests pass Ruff and formatting; shared bridge/controller added-line lint is clean; git diff --check passes. Existing config-aware completion harnesses now use the supported bootstrap profile. Fairness evidence compares exact durable accepted_at ordering and both provider notices once, because asynchronous provider preparation arrival is not admission order; only observed healthy fixture polling deadlines were extended, with production limits unchanged.

Modified the wake ledger/models/schema/recovery policy, metadata-only bridge/controller hooks, wake coordinator, focused DB/Chat tests, Console agent guide and lessons-testing-evidence. ADR required: yes; implemented backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md, including its temporary claim recovery clarification. Task remains In Progress with criteria unchecked pending root independent review.

Independent review repairs preserve causal work through native Save. Progress intake now resolves the exact source conversation from existing AgentRuns lineage in an owned metadata-only worker, so new saved manual chains receive their own bucket while old temporary survivors retain their claims and budgets. Removing a stale completion retains unified pending membership when progress remains, preventing a second pending chain from being stranded. Source aliases share one active delivery per native session. Only an exact accepted AGENT_WAKE token may carry its frozen source conversation into ledger/run metadata; manual preparation with no token or a foreign coordinator token refuses on both provider paths. The current native saved conversation remains the owner for Canvas, chat data, scratch, policy and approvals. A real Canvas authority regression reproduced the generic-ID leak before that separation. ADR199 now documents this boundary, and lessons-testing-evidence records the Save/source incident.

Final review qualification: 72 focused ledger/schema/progress/existing completion scheduling/recovery/real provider dispatch checks passed in 142.45s (/private/tmp/tldw-progress-review-final2-output.log and matching .log). The stronger Save/plain+agent/Canvas rerun passed 2 checks in 16.30s (/private/tmp/tldw-progress-canvas-fixed-output.log). Independent read-only review passed the 8 causal Save, Canvas, stale-completion membership, shared native slot and manual/foreign-token refusal cases in 41.94s with no remaining actionable TASK33432 findings. Ruff passes the core/new modules; changed wake/controller body ranges are formatted; added-line bridge/controller lint and git diff --check are clean. Existing unrelated shared-file formatting/lint debt was left unchanged. Status and criteria remain review-pending for root closeout.

Final root disposition, September 29: independent re-review approved all remaining source intake, shared membership, native alias admission and Canvas authority repairs; eight focused cases passed in 41.94s. The final combined wake/ledger/schema/completion selection passed 72 cases in 142.45s. Changed-code static/format and whitespace checks passed. All four acceptance criteria are satisfied under ADR-199; the separate native lifecycle contention qualification belongs to TASK-33431. Targeted evidence only; no full suite or live-provider claim.
<!-- SECTION:NOTES:END -->
