---
id: TASK-34358
title: Learn and calibrate native response rule candidates
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:54'
updated_date: '2026-10-12 23:14'
labels: []
dependencies:
  - TASK-34357
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/228-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 5. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Learning discriminates the recorded and paraphrased violation from acceptable controls within three attempts and 120 seconds.
- [ ] #2 Synthetic responses cannot fabricate work success and unrelated action guidance remains inactive.
- [ ] #3 Editor tests retain the exact edited candidate and reuse validation only with matching evidence and protocol.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: implements approved calibration and inactive editor candidate service. Follow Task 5 RED-GREEN: bounded drafts and four fixed-evidence controls, hidden host expectations, conservative feedback scope, exact edited-candidate testing and evidence-bound reuse; targeted tests and static verification.

Current-dev PR3010 integration: follow Docs/superpowers/plans/2026-10-11-pr3010-current-dev-integration.md on the actual post-prior-merge base. ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: preserve the approved local storage/recovery, helper custody and shared native/hook acceptance; source ADR219 remapped after the older claimant. Reproduce reviewed failures before fixes; qualify actual next schema, six voice residues, exact private/index/table catalogs, preparation-through-transport custody, lazy first-use, current Goal/Personal Context owners and separately labelled real-provider/native UI evidence. Historical source qualification above is retained; reopened acceptance awaits current integrated verification and independent review.

2026-10-12 released Task1 on private START13b204cb638ca891175610f5e687a80def71c21e. ADR required: yes (existing). ADR path: backlog/decisions/228-console-learned-response-rules.md; preserves ADR227 Goal and current Personal Context/native ownership. Implement complete91path source workflow by current symbol. Qualify actual schema80-to81/four catalog variants/all predecessors, six voice residues, seven tables/three query/index pins/private import purpose, preparation-through-transport physical custody and original-owner usage, current Goal/hook/queue atomic receipts, lazy first-send/global-before-Chat, manager/draft/Stop/currentness and finite native worker retirement. Preserve all bounds and source history. Use targeted RED/GREEN; measure bounded critical cases for existing required job capacity, preserving inherited targets/order/caps/processes with executable membership negatives. Separate actual local-model/external-hook/native paint from scripted evidence. Root owns actual-base reconciliation, independent reviews, hosted proof and task closure; all AC remain open.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bounded native drafting returns inactive candidates; four opaque host-labelled controls share original evidence and use the public evaluator. No prompt replay or executable hook. Exact edited definitions remain unchanged. Feedback-only reuse binds original input, fixture fingerprints and protocol; missing provenance requires retesting. Conservative host action/target checks and original-task wrapper reject additional-action guidance. ADR-228. RED missing builder -> GREEN 21 focused cases; 81 combined model/storage/semantic cases; black/ruff/mypy pass. Scripted classifications qualify orchestration, not actual model quality.

2026-10-12 current integration (historical source evidence above remains unchanged): Integrated draft, independent calibration, tested activation and correction learning under current helper and native owners. Actual offline Qwen2.5-0.5b failed learning (three draft calls) and all seven semantic controls; invalid schema remains couldnt_verify. No prompt/parser loosening, synthetic output replacement or quality-pass claim was made. Existing ADR228 governs, with ADR097 live-revision GC and ADR227 Goal constraints applied. Current report: E/pr3010-current-integration/task-1-report.md. All tasks stay In Progress and acceptance remains open pending independent review, actual-base/hosted qualification and unmet external/model checks.
<!-- SECTION:NOTES:END -->
