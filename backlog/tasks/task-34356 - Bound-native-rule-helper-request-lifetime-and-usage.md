---
id: TASK-34356
title: Bound native rule helper request lifetime and usage
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:54'
updated_date: '2026-10-13 02:08'
labels: []
dependencies:
  - TASK-34355
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/228-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 3. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Four app-wide and one per-Chat helper slots remain held through physical provider settlement.
- [ ] #2 Native helpers obey deadlines and zero transport retries without changing existing auxiliary behavior.
- [ ] #3 Late cancelled output is rejected and original-owner usage is charged once.
- [ ] #4 Cancelled off-loop preparation retains its Chat and app capacity until the physical preparer settles and never dispatches a late provider request.
- [ ] #5 Explicit DeepSeek reasoning effort reaches ordinary and native helper requests unchanged; omitted settings retain defaults and unsupported API mode refuses before network.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: implements accepted runtime and provider boundary. Follow Task 3: watched failing physical-lease and provider tests, app-owned capacity and usage-once lifecycle, native-only gateway timeout/output/no-retry changes, targeted ordinary auxiliary regressions and static verification.

Current-dev PR3010 integration: follow Docs/superpowers/plans/2026-10-11-pr3010-current-dev-integration.md on the actual post-prior-merge base. ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: preserve the approved local storage/recovery, helper custody and shared native/hook acceptance; source ADR219 remapped after the older claimant. Reproduce reviewed failures before fixes; qualify actual next schema, six voice residues, exact private/index/table catalogs, preparation-through-transport custody, lazy first-use, current Goal/Personal Context owners and separately labelled real-provider/native UI evidence. Historical source qualification above is retained; reopened acceptance awaits current integrated verification and independent review.

2026-10-12 released Task1 on private START13b204cb638ca891175610f5e687a80def71c21e. ADR required: yes (existing). ADR path: backlog/decisions/228-console-learned-response-rules.md; preserves ADR227 Goal and current Personal Context/native ownership. Implement complete91path source workflow by current symbol. Qualify actual schema80-to81/four catalog variants/all predecessors, six voice residues, seven tables/three query/index pins/private import purpose, preparation-through-transport physical custody and original-owner usage, current Goal/hook/queue atomic receipts, lazy first-send/global-before-Chat, manager/draft/Stop/currentness and finite native worker retirement. Preserve all bounds and source history. Use targeted RED/GREEN; measure bounded critical cases for existing required job capacity, preserving inherited targets/order/caps/processes with executable membership negatives. Separate actual local-model/external-hook/native paint from scripted evidence. Root owns actual-base reconciliation, independent reviews, hosted proof and task closure; all AC remain open.

2026-10-12 Task3 at separate Rootleaf START1ed1da223c8754356aa1270975e38a4d11d7f0bc, preserving frozen Goal3a. Ruling: complete current required publication contracts and documented optional DeepSeek parameter plumbing under the standing all-issues-before-merge instruction. Cost if wrong: one bounded implementation/review and affected qualification cycle; stale modal results, altered provider defaults or falsely renewed diagnostic pins. ADR required: no new ADR. Existing ADR173/228/062 (successor063)/064 govern; amend064 for documented distinct none and honest unimplemented Responses refusal. Before code retain genuine gone-subtree/queued-action, timestamp and actual primary/leased-helper request RED. Guard view attachment and captured reload scope/currentness without suppressing store errors or weakening CAS; use canonical UTC only for new builder writes. Forward explicitly selected/configured optional DeepSeek none/low/high/max through existing immutable resolution and strict adapter validation; omitted remains omitted. Refuse unsupported Responses before effect, no Responses migration/default nonthinking/alias expansion. Review every final changed diagnostic statement and sink topology against current0ac pin98dbeef before canonical inventory generation. Qualify affected lifecycle/saved-revision/custody/HTTP/refusal/usage/closure/schema/UTC and final required Derived scripts; preserve all15 existing critical selectors, processes, profiles and caps and measure any bounded added selection. No source/native/provider acceptance waiver, paid/model/GUI/user-profile/keyring/children/ref changes. Task status/open AC/history retained. New task3 receipts/report/handoff and one scoped SPEC/QUALITY review; root owns publication/live qualification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-228 helper custody through physical synchronous-worker finally or genuine async transport cleanup, retaining four app/one Chat reservations after cancelled awaiters. Native calls clamp output to 4096/current model cap, transport to remaining at most 30 seconds, and retries to zero; ordinary auxiliary defaults remain intact. Captured late usage is charged once to the original source; unknown usage remains unknown. App shutdown seals admission while reporting pending physical cleanup. Held-thread, retained-task cancellation, late async usage and slow usage persistence tests pass; Black/Ruff/mypy pass. Broad targeted gateway/runtime runs pass after excluding only 22 cases reproduced against unchanged owners; baseline limitations are recorded in the execution ledger.

2026-10-12 current integration (historical source evidence above remains unchanged): Helper capacity now begins before off-loop preparation and stays owned through preparation, queued transport, active HTTP and cleanup. Revocation prevents queued adapters from starting; late actual usage charges the original owner once. Native SQL cancellation retains its physical handle until operation-owned cleanup. Inherited deadlines/caps remain unchanged. Existing ADR228 governs, with ADR097 live-revision GC and ADR227 Goal constraints applied. Current report: E/pr3010-current-integration/task-1-report.md. All tasks stay In Progress and acceptance remains open pending independent review, actual-base/hosted qualification and unmet external/model checks.

Task3 implementation phase: attachment/subtree, generation/scope and editor-digest guards retain public manager lifecycle and dirty edits; new builder stamps use canonical UTC with historic offset parsing retained. DeepSeek explicit/configured none/low/high/max traverses existing resolution, editor rebasing, prepared primary/leased helper and both compatibility kwargs builders; unset stays omitted and unsupported Responses refuses before transport. Genuine RED receipts task3-01 (mixed test typo retained), task3-05 owner RED and task3-07 dispatch RED; stable task3-04 103 pass, task3-06 662 pass and task3-08 74 pass. task3-02/03 remain mixed diagnostics, not final qualification. All paid count remains seven. Reviewed final diagnostic statements against 98dbeef/current0ac: five verbatim moves from store to branch/stream owners, seven fixed-text response_rule warnings, no new interpolation or sink topology. Canonical inventory generated only after that review. Five bounded critical selectors appended after original fifteen; exact inherited argv/order/process/markers/caps/aggregate retained, executable omission/comment negatives extended. Local combined addition/membership measurement 23.78s (runner30.65s) against observed parent677s/1200s, not a hosted guarantee. Current-head required guards and commit binding follow; external provider/native/Goal/inherited limits remain open, no Done.
<!-- SECTION:NOTES:END -->
