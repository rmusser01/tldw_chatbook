---
id: TASK-33432
title: Progress-triggered bounded supervisor wakes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-29 18:11'
updated_date: '2026-10-02 02:23'
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
- [x] #5 Incoming Console wake and run-control regressions observe actual asynchronous admission and publication; production deadlines, controls and authority gates remain unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the existing progress/completion shared claim, generation and budget boundaries. 2. Trace accepted automatic wake attempts through incoming compaction failure, retry latch and exact preflight-block copy ownership. 3. Qualify real automatic compaction, failed-spend disclosure, nonrebilling, exact recovery copy and hook refusal/refund paths. 4. Check startup census, diagnostic guards, changed-code static analysis and independent review. ADR required: no; direct integration under backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md and backlog/decisions/052-console-conversation-memory-and-compaction-policy.md. Reason: no new service, data or authority boundary.
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

October 1 CI/integration closeout: populated the existing real mixed claim-plan regression and registered idx_automatic_progress_claims_attempt. Independent rebase review reproduced upstream hook admission returning before proven-preacceptance cleanup, leaving a prepared attempt and reserved generation. The seven-line controller repair marks only an exact authorized live AGENT_WAKE token with acceptance not started; existing cleanup aborts/refunds it. The actual plain/agent gateway regression proves manual/copied-token refusal, no provider call before refusal, one retry after clearing the hook and no duplicate replay. Six authority/retry checks pass 43.72s and three existing readiness/acceptance/completion guards pass 11.51s; independent reviewer passes both new paths plus three guards and approves. Affected 145-case schema/messaging and 147-case lifecycle selections pass; overlapping selections are not summed. Final 75-file changed-code and diagnostic guards pass. Existing ADR-199 AC3 covers this repair; no new ADR. Criteria satisfied; Done. PR #2918 is rebased on dev 31d4f9b764 and awaits its published-head remote checks.

Final dev 84247cb843 qualification: the stopped combined selection recorded 138 passed/7 failed in1147.92s; it is not claimed green. Independent observation located the held fixture's five-second expiry during valid preparation: gateway dispatch at 9.578s, first yield 4.6ms later, normal durable commit/cleanup and an unchanged controller/store/gateway path. Only first-chunk/wake-entry fixture preparation gets 15s; control actions keep 5s and wake paint/ledger/timer settles keep 8s. Coalesced UI sync does not acknowledge paint, so the unchanged-completion regression now observes actual Running and terminal glyph publication without interaction or idle polling.

Fresh final affected selection passes 13 cases in 177.22s (four mounted wake cases, both exact hook-refund gateways, CSS budget/bundle and UI census 1031/1033); independent review passes all five originally failing controls in 209.78s and approves both test corrections. A private frozen-terminal-publication control reaches streaming/ledger completion then fails at the exact settled-glyph assertion; a private never-first-chunk control fails at its bounded 15s precondition. The disabled-delivery-hook-only probe passed due to a coalesced tail and is not accepted as necessary-hook evidence. These selections overlap with earlier runs and are not summed.

Final 78-file changed-code fatal/added-line checks, ten new-file Ruff/format checks, owned test-range formatting and whitespace pass. Last predicate formatting preserves the tested AST. Modified only the two mounted test harnesses, task/review/plan/ledger records and the measured testing lesson; production gates, deadlines, budgets, APIs and authority remain unchanged. ADR required: no; direct test-evidence qualification under ADR-199 AC4 and new AC5. All criteria satisfied; Done. Fresh PR-head CI remains required after publishing the rebase; no full suite, live provider, merge or inherited-size/static/resource blanket claim.

Reopened for incoming compaction changes on dev 27e718f01d; accepted compaction failure/nonreplay and exact hook-refund qualification is pending.

Final incoming compaction integration: independent source review finds no actionable wake/compaction overlap. Acceptance precedes compaction, so accepted failures retain generation and nonreplay custody; exact recovery-copy ownership and durable-parent lineage remain intact. Ten real compaction/refund/nonreplay cases pass in 82.63s (/private/tmp/review-pr2918-compaction-wake-rebase.log), including both hook-refund provider paths. Mounted saved reopen/close passes 2 in 18.15s. Startup/CSS/guide guards pass 4 in 51.29s: imports 679/686, UI-ready 1032/1033, original CSS and module pins unchanged. No production timing or authority change.

The preserving Chat 73→74→75 composition and exact recovery gates are qualified under TASK-33431. Final 79-file changed-code/new-file/owned-range static and whitespace checks pass; audited diagnostic inventory verifies 629 owners and 15 sink files. Independent schema review approves 39 overlapping checks. ADR required: no new ADR; existing ADR-199/052. Docs-only dev 922440b93e is the following rebase target and changes no runtime file. All five criteria remain satisfied; Done. Original 13-task burn-down is complete; fresh PR-head remote checks remain required, with no full suite, live provider, Windows or merge claim.
<!-- SECTION:NOTES:END -->
