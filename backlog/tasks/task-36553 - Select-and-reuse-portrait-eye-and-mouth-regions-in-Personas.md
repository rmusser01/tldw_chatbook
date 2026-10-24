---
id: TASK-36553
title: Select and reuse portrait eye and mouth regions in Personas
status: To Do
created_date: 2026-10-24 14:24
dependencies:
- TASK-36552
labels:
- console
- characters
- portrait-compositing
references:
- backlog/decisions/242-console-radio-portrait-performance.md
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-24-portrait-compositing-implementation.md
- backlog/decisions/243-source-bound-portrait-compositing.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users mark expression regions once on their prepared portrait and inspect them precisely before spending generation requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The existing Living portrait editor offers three labeled source-space regions, contain-mapped two-corner mouse selection, keyboard-accessible numeric adjustment and detailed mask preview; letterboxing and terminal-cell aspect do not shift saved coordinates.
- [ ] #2 Users explicitly confirm an open-eyes/closed-mouth prepared base before saving a recipe; unsupported bases retain the existing ready-made six-frame import workflow.
- [ ] #3 Save regions reuses unchanged canonical pixel identity/settings after reopen without provider calls or revision churn from equivalent PNG encoding; exact source/profile/editor/revision checks preserve active artwork and Cancel discards only unsaved edits.
- [ ] #4 Saving changed regions retires the exact draft sample and atomically installs a rest-only/zero-component snapshot for the new recipe, without relabeling old frames as a complete manual set; the applied DB performance remains unchanged.
- [ ] #5 Focused controls, narrow layouts, Static/Reduce motion, shared image budgets and token-backed styling work with production CSS; mounted coordinate/keyboard/stale-save/reopen tests reject late preview/save work.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
