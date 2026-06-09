---
id: TASK-89.9
title: Compact Library source column width
status: Done
labels:
- library
- ux
- layout
priority: medium
parent_task_id: TASK-89
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce dead space in the Library left source/navigation column so the content hub uses screen width for the primary detail area while preserving readable source actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library source browser column is visibly narrower than the detail column at supported desktop Textual sizes.
- [x] #2 Source action labels remain readable and keyboard/click reachable after narrowing.
- [x] #3 The detail area receives the reclaimed width without breaking the inspector or workbench pane alignment.
- [x] #4 Focused regression coverage verifies the source/detail/inspector width relationship.
- [x] #5 Actual rendered CDP screenshot is captured and approved before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a visual layout contract refinement inside an existing Library workbench. It does not change storage, schema, sync policy, data ownership, service contracts, provider/runtime boundaries, or security posture.

1. Add a focused geometry regression for wide Library viewports proving the source browser stays compact while the detail pane receives the reclaimed width.
2. Update source TCSS tokens/rules so `#library-source-browser` uses a compact content-fit width and no longer shares the inspector flex sizing.
3. Rerun focused visual/layout and Library content-hub regressions.
4. Capture a rendered CDP screenshot of the Library screen for approval before marking the task Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a wide-viewport geometry regression proving the Library source browser stays content-fit while the detail pane receives the reclaimed width.
- Changed the Library source browser from flexible `2fr` sizing to a compact 31-cell width in both the source TCSS token/rule and the Python-composed pane style, with matching min/max constraints so it cannot expand into dead space.
- Preserved the existing taller source action buttons and verified labels remain readable and reachable inside the compact pane.
- Captured and received approval for the rendered CDP screenshot at `Docs/superpowers/qa/product-maturity/screen-qa/library/source-column-compact-cdp-2026-06-09.png`.
- ADR check: no ADR required because this is a visual layout refinement inside an existing destination contract, with no storage, schema, sync, service contract, or ownership-boundary changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Compacted the Library source/navigation column so it fits its content instead of consuming dead space, reallocating width to the primary detail area while preserving source action readability and workbench alignment.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
