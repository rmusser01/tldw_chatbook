---
id: TASK-60.4.3
title: Post-release Workspaces and Library depth tranche
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 11:15'
labels:
  - post-release
  - workspaces
  - library
  - ux
dependencies: []
parent_task_id: TASK-60.4
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deepen the Workspaces and Library model after the audit confirms the current local surfaces are usable but workspace membership, Collections membership, Import/Export depth, and cross-workspace source authority need product depth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace and Library depth requirements reference TASK-60.3 actual-use audit evidence.
- [x] #2 Workspace switching never hides Library or Notes items; it only changes context eligibility, authority labels, and handoff availability.
- [x] #3 Collections membership, deeper Import/Export, and workspace source authority are staged without regressing global search/view access.
- [x] #4 QA verifies cross-workspace view/edit/search behavior and Console-context restrictions with actual app use before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read TASK-60.3 evidence and current workspace/display-state seams.
2. Add failing pure and mounted regressions proving Library remains globally visible while workspace context only gates Console/RAG/agent use.
3. Add a Library Workspaces-depth display state and mount it inside the existing Library workbench without changing source query filters.
4. Surface staged Collections membership, Import/Export depth, and workspace source-authority labels as read-only next actions.
5. Run focused tests, capture actual textual-web/CDP screenshots for QA, then update task notes and roadmap evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a Library Workspaces depth display state that keeps Library and Notes globally visible while calculating active-workspace Console/RAG staging eligibility.
- Mounted Workspaces mode inside the destination-native Library workbench with distinct source, scope, and handoff columns plus action-first inspector copy.
- Added no-source recovery that exposes `Import sources`, removes misleading local-source-snapshot copy, and explains why handoff is blocked.
- Added mounted and pure regressions for cross-workspace visibility, source authority labels, staged Collections and Import/Export copy, and empty-state recovery.
- Captured actual textual-web/CDP evidence at `Docs/superpowers/qa/product-maturity/post-release-ux-hci/actual-screenshots/2026-05-22-task-60-4-3-library-workspaces-empty-polish.png`; user approved the rendered screen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Workspaces and Library depth tranche. Library Workspaces now communicates that browsing/searching remains global while Console/RAG/agent staging is limited by active-workspace eligibility, with visible blocked-state recovery and approved actual-screen QA evidence.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] Acceptance criteria checked.
- [x] Implementation notes added.
- [x] Focused automated verification passed: 40 passed, 8 warnings.
- [x] Static verification passed: `py_compile` and `git diff --check`.
- [x] Actual textual-web/CDP screenshot captured and approved.
- [x] QA evidence linked from the post-release QA index and product-maturity roadmap.
<!-- DOD:END -->
