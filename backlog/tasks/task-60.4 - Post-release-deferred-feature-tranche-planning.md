---
id: TASK-60.4
title: Post-release deferred feature tranche planning
status: Done
labels:
- roadmap
- planning
- post-release
priority: medium
parent_task_id: TASK-60
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Convert verified residual risks from Phase 3-6 closeout into staged implementation tranches after the actual-use audit establishes what is currently broken.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP runtime launch, write sync promotion, Workspaces/Library depth, citation/snippet carry-through, and optional dependency/package polish each have a staged follow-up tranche.
- [x] #2 Each tranche references evidence from the actual-use audit before implementation work is prioritized.
- [x] #3 Feature planning does not mask P0/P1 usability breakage discovered by the audit.
- [x] #4 The product maturity roadmap distinguishes verified shipped behavior from deferred future work and newly discovered broken behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression coverage requiring the deferred tranche plan, public roadmap gate wording, tracked follow-up tasks, and completed post-release validation status.
2. Create follow-up tranche tasks for ACP runtime launch, write sync promotion, Workspaces/Library depth, citation/snippet carry-through, and optional dependency/package polish.
3. Write the deferred tranche plan from `TASK-60.3` actual-use audit evidence and explicitly preserve the P0/P1 usability gate.
4. Update the product maturity tracker and public roadmap so verified shipped behavior stays distinct from deferred future work.
5. Close `TASK-60.4` and parent `TASK-60` only after focused documentation regressions pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/superpowers/plans/2026-05-22-post-release-deferred-feature-tranches.md` to stage the five deferred feature tranches from `TASK-60.3` actual-use audit evidence.
- Created follow-up tranche tasks `TASK-60.4.1` through `TASK-60.4.5` for ACP runtime launch, write sync promotion, Workspaces/Library depth, citation/snippet carry-through, and optional dependency/package polish.
- Updated `Docs/superpowers/trackers/product-maturity-roadmap.md` and `Docs/Product_Roadmap.md` so verified shipped behavior, recoverably blocked behavior, and deferred future work stay distinct.
- Added regression coverage for the tranche plan, follow-up task titles, roadmap gates, and completed post-release validation status.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed deferred feature tranche planning. The remaining high-value future work is now staged, evidence-linked, and gated behind unresolved P0/P1 usability defects instead of being treated as already shipped behavior.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
