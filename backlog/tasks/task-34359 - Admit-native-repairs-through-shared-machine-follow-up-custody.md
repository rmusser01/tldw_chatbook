---
id: TASK-34359
title: Admit native repairs through shared machine follow-up custody
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:55'
updated_date: '2026-10-12 23:14'
labels: []
dependencies:
  - TASK-34358
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/232-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 6. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native and genuine hook feedback produce one atomic follow-up and shared receipt.
- [ ] #2 Stale, forged, duplicate or historical assessments cannot authorize repairs.
- [ ] #3 Two native and three shared continuation limits, foreground priority and permission gates hold.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/232-console-learned-response-rules.md. Reason: native/hook shared machine admission and atomic receipt custody. Follow Task 6 RED-GREEN: current assessment lookup, real Stop composition, retained counters/deduplication, same-cursor receipt and fingerprint, rollback/foreground/veto/recovery tests; retain old hook protocol and permissions.

Current-dev PR3010 integration: follow Docs/superpowers/plans/2026-10-11-pr3010-current-dev-integration.md on the actual post-prior-merge base. ADR required: yes. ADR path: backlog/decisions/232-console-learned-response-rules.md. Reason: preserve the approved local storage/recovery, helper custody and shared native/hook acceptance; source ADR219 remapped after the older claimant. Reproduce reviewed failures before fixes; qualify actual next schema, six voice residues, exact private/index/table catalogs, preparation-through-transport custody, lazy first-use, current Goal/Personal Context owners and separately labelled real-provider/native UI evidence. Historical source qualification above is retained; reopened acceptance awaits current integrated verification and independent review.

2026-10-12 released Task1 on private START13b204cb638ca891175610f5e687a80def71c21e. ADR required: yes (existing). ADR path: backlog/decisions/232-console-learned-response-rules.md; preserves ADR227 Goal and current Personal Context/native ownership. Implement complete91path source workflow by current symbol. Qualify actual schema80-to81/four catalog variants/all predecessors, six voice residues, seven tables/three query/index pins/private import purpose, preparation-through-transport physical custody and original-owner usage, current Goal/hook/queue atomic receipts, lazy first-send/global-before-Chat, manager/draft/Stop/currentness and finite native worker retirement. Preserve all bounds and source history. Use targeted RED/GREEN; measure bounded critical cases for existing required job capacity, preserving inherited targets/order/caps/processes with executable membership negatives. Separate actual local-model/external-hook/native paint from scripted evidence. Root owns actual-base reconciliation, independent reviews, hosted proof and task closure; all AC remain open.

2026-10-12 current Goal ruling (ADR227/232): assess only the exact coordinator-issued, accepted Goal iteration handoff pinned through native parent identity and message version. A genuine live GoalIterationAuthorization is required; arbitrary SYSTEM rows, copied metadata and shape-compatible objects confer no rule authority. Recheck Goal/binding/source/Stop currentness. Completed Goal runs expose no existing same-native-run repair seam: preserve the assessment and explicitly refuse automatic repair rather than admitting an ordinary queued turn. No second native run, Goal attempt, machine receipt, allowance or repeated tool work; keep record_outcome identity guard and inherited absolute limits unchanged. Phase12 exposed the genuine source_changed assessment defect; qualify passing, violating and revoked/fake input cases. This enforces inherited ownership and leaves settled-Goal automatic repair unsupported, documented for independent review. ADR required: yes (existing amendment), backlog/decisions/232-console-learned-response-rules.md.

2026-10-12 connected trace lifetime ruling: phase24 reproduced startup GC removing newly admitted user/assistant semantic revisions before provider call reservation, leaving an accepted repair retained for recovery. Apply existing ADR097 decision16 and ADR232: preserve the live provenance owned by current durable dispatch checkpoints through call reservation, without new storage, a retention owner, timing suppression or weaker GC authorization. Map: durable turn acceptance -> canonical user/assistant revisions -> captured request provenance -> ConsoleTraceBoundaryFactory reservation; TraceGarbageCollector mark/sweep must respect the current accepted owner during the gap. Prove actual ordinary-send and native-repair survival with real collector/admission/reservation, and orphan/revoked/current-version/epoch-change reclamation or refusal controls. Costs: a wrong missing root deletes live provenance; an overbroad root leaks orphan retention. Keep existing graph epochs, native handles, deletion grants, privacy/fork/purge and caps. ADR required: yes, existing ADR097/232 amendment; no new ADR ID.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Native references resolve current runtime-owned assessments and pinned rules, compose only owned Stop results, preserve hook receipts and share one queue custody. Source versions and strict host counters join the acceptance fingerprint; shared receipt writes on the message/checkpoint cursor. Duplicate callbacks, vetoes, required barriers, foreground input and limits refuse repair. ADR-232. RED-GREEN includes hybrid chain identity and missing genuine hook provenance. 25 focused cases; 38 existing queue cases; 5 supported hook/agent budget controls; 54 passing DB/UI owner cases (9 exact baseline DB failures excluded). External Stop tests also fail on unchanged source in this Windows environment; no assertion/guard weakened. black/ruff/mypy pass.

2026-10-12 current integration (historical source evidence above remains unchanged): Native and genuine external Stop-hook corrections share the existing receipt, causal limits, deadline and queue admission. Settled Goal repair explicitly refuses because no same-run continuation can preserve the existing accepted Goal attempt and native_run_id. No record_outcome guard or allowance changed. Current checkpoint roots and canonical mutation epochs preserve admitted semantic revisions until real trace reservation; revoked/orphan revisions are reclaimed by real GC. Existing ADR232 governs, with ADR097 live-revision GC and ADR227 Goal constraints applied. Current report: E/pr3010-current-integration/task-1-report.md. All tasks stay In Progress and acceptance remains open pending independent review, actual-base/hosted qualification and unmet external/model checks.
<!-- SECTION:NOTES:END -->
