---
id: TASK-34805
title: Own response rule learning and assessment in Console runtime
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:55'
updated_date: '2026-10-12 23:14'
labels: []
dependencies:
  - TASK-34359
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/228-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 7. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Eligible settled responses are checked before completion release and next queue drain.
- [ ] #2 Stop, foreground input and source changes fence late activation or correction.
- [ ] #3 App-owned state survives view remount and public editor testing never autoactivates changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: app-owned learning, assessment and completion/queue lifecycle. Follow Task 7 RED-GREEN: real generation boundary checks before completion publication/drain, current source and binding CAS, foreground/Stop cancellation including after generation, fresh manual repair source, editor validation and disposable view projections; targeted runtime/shutdown/ownership checks.

Current-dev PR3010 integration: follow Docs/superpowers/plans/2026-10-11-pr3010-current-dev-integration.md on the actual post-prior-merge base. ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: preserve the approved local storage/recovery, helper custody and shared native/hook acceptance; source ADR219 remapped after the older claimant. Reproduce reviewed failures before fixes; qualify actual next schema, six voice residues, exact private/index/table catalogs, preparation-through-transport custody, lazy first-use, current Goal/Personal Context owners and separately labelled real-provider/native UI evidence. Historical source qualification above is retained; reopened acceptance awaits current integrated verification and independent review.

2026-10-12 released Task1 on private START13b204cb638ca891175610f5e687a80def71c21e. ADR required: yes (existing). ADR path: backlog/decisions/228-console-learned-response-rules.md; preserves ADR227 Goal and current Personal Context/native ownership. Implement complete91path source workflow by current symbol. Qualify actual schema80-to81/four catalog variants/all predecessors, six voice residues, seven tables/three query/index pins/private import purpose, preparation-through-transport physical custody and original-owner usage, current Goal/hook/queue atomic receipts, lazy first-send/global-before-Chat, manager/draft/Stop/currentness and finite native worker retirement. Preserve all bounds and source history. Use targeted RED/GREEN; measure bounded critical cases for existing required job capacity, preserving inherited targets/order/caps/processes with executable membership negatives. Separate actual local-model/external-hook/native paint from scripted evidence. Root owns actual-base reconciliation, independent reviews, hosted proof and task closure; all AC remain open.

2026-10-12 current Goal ruling (ADR227/228): assess only the exact coordinator-issued, accepted Goal iteration handoff pinned through native parent identity and message version. A genuine live GoalIterationAuthorization is required; arbitrary SYSTEM rows, copied metadata and shape-compatible objects confer no rule authority. Recheck Goal/binding/source/Stop currentness. Completed Goal runs expose no existing same-native-run repair seam: preserve the assessment and explicitly refuse automatic repair rather than admitting an ordinary queued turn. No second native run, Goal attempt, machine receipt, allowance or repeated tool work; keep record_outcome identity guard and inherited absolute limits unchanged. Phase12 exposed the genuine source_changed assessment defect; qualify passing, violating and revoked/fake input cases. This enforces inherited ownership and leaves settled-Goal automatic repair unsupported, documented for independent review. ADR required: yes (existing amendment), backlog/decisions/228-console-learned-response-rules.md.

2026-10-12 connected trace lifetime ruling: phase24 reproduced startup GC removing newly admitted user/assistant semantic revisions before provider call reservation, leaving an accepted repair retained for recovery. Apply existing ADR097 decision16 and ADR228: preserve the live provenance owned by current durable dispatch checkpoints through call reservation, without new storage, a retention owner, timing suppression or weaker GC authorization. Map: durable turn acceptance -> canonical user/assistant revisions -> captured request provenance -> ConsoleTraceBoundaryFactory reservation; TraceGarbageCollector mark/sweep must respect the current accepted owner during the gap. Prove actual ordinary-send and native-repair survival with real collector/admission/reservation, and orphan/revoked/current-version/epoch-change reclamation or refusal controls. Costs: a wrong missing root deletes live provenance; an overbroad root leaks orphan retention. Keep existing graph epochs, native handles, deletion grants, privacy/fork/purge and caps. ADR required: yes, existing ADR097/228 amendment; no new ADR ID.

2026-10-12 phase37 genuine native SQLite RED: learning activates its tested rule but silently ignores the existing maintenance gate refusing initial native repair, returning reason tested. Preserve the committed rule and all queue gates; propagate the False admission result through the existing initial_repair_unavailable projection. Existing ADR228 and current honest-refusal AC apply; no authority, allowance or queue contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed app-owned learning, completed-answer assessment, source/binding CAS, editor Test reuse and immutable revisions, physical helper accounting, disposable projections, completion holds, bounded repair and real AgentService budget transfer. ADR-228 owns the architecture. Verified 212 targeted tests, five independently reproduced unsupported external-command exclusions and one inherited expected failure. New runtime, builder/store/model/resource seams and Console controller/store/queue/runtime/cost/status owners were updated; black changed ranges, scoped ruff and package mypy pass. Self-review found and fixed nested generation holds and lost hook-free budgets with actual AgentService RED-GREEN witnesses.

2026-10-12 current integration (historical source evidence above remains unchanged): Integrated lazy first-send and global-before-Console composition, current task/source fencing, retained-work repair, Stop and physical helper/native SQL retirement. Genuine maintenance admission RED returned tested despite refused repair; the existing initial_repair_unavailable projection now preserves active binding and honestly reports refusal. The automatic GC loop invalidates only stale-epoch pending results while reusing unchanged-epoch deferred compaction. Existing ADR228 governs, with ADR097 live-revision GC and ADR227 Goal constraints applied. Current report: E/pr3010-current-integration/task-1-report.md. All tasks stay In Progress and acceptance remains open pending independent review, actual-base/hosted qualification and unmet external/model checks.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Source TASK-34360 at codex/wave2-source-pr-3010 (73823fde3a4e) was created on 2026-10-04. The older current-dev task keeps 34360; this preserved younger response-rules task is TASK-34805. Original implementation history remains above; references and dependency identities follow this remap.
