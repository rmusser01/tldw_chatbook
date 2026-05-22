---
id: TASK-60.3
title: Post-release cross-screen workflow validation
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-22 05:53'
labels:
  - ux
  - hci
  - qa
  - workflows
dependencies: []
parent_task_id: TASK-60
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate end-to-end product workflows across destinations using the actual app, with special focus on handoffs into Console as the agentic control surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home active work can open details and route relevant work into Console or shows a clear blocked state.
- [x] #2 Library Search/RAG can produce evidence or a recoverable blocked state, then hand off context into Console where appropriate.
- [x] #3 Artifacts and Chatbooks can be opened, resumed, or clearly blocked from Home/Artifacts/Console without dead controls.
- [x] #4 Personas, Skills, MCP, ACP, Watchlists, Schedules, and Workflows handoffs into Console are verified or explicitly classified as blocked future work with user-visible recovery copy.
- [x] #5 At least five power-user repeated workflows are timed qualitatively for friction, shortcuts, state persistence, and recovery paths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Refresh current dev state and existing post-release QA evidence so the workflow audit starts from real merged behavior.
2. Run actual app/CDP or mounted-pilot validation for the required cross-screen workflows: Home to Console, Library RAG to Console, Artifacts/Chatbooks resume, and destination handoffs into Console.
3. Capture evidence files under Docs/superpowers/qa/product-maturity/post-release-ux-hci/ with screenshots only when a visible UI state changes or needs user approval.
4. Classify each workflow as complete, recoverably blocked, or defective using P0/P1/P2/P3 severity and create follow-up Backlog tasks for unresolved P0/P1 findings.
5. Run focused regression tests for verified paths and update TASK-60.3 with implementation notes, final summary, AC status, and residual risks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added consolidated cross-screen workflow evidence at `Docs/superpowers/qa/product-maturity/post-release-ux-hci/2026-05-22-cross-screen-workflow-validation.md`.
- Verified Home, Library/Search-RAG, Artifacts/Chatbooks, Personas, Skills, ACP, Watchlists, Schedules, Workflows, and Console source-readiness seams with focused mounted Textual regressions.
- Updated the post-release QA index and product-maturity tracker so `TASK-60.3` is recorded as verified while MCP/ACP runtime depth, Workspaces/Library depth, write sync, and citation/snippet carry-through stay in `TASK-60.4`.
- No visible UI changed in this slice, so no new screenshot approval was required; prior actual screenshot evidence remains linked in the screen evidence index.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed post-release cross-screen workflow validation. Required handoffs are verified or honestly classified as recoverable/future service-depth work with user-visible copy, and no unresolved P0/P1 workflow findings remain for `TASK-60.3`.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
