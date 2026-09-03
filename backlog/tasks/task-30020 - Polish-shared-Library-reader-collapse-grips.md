---
id: TASK-30020
title: Polish shared Library reader collapse grips
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 04:54'
updated_date: '2026-09-03 13:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the shared Library reader collapse targets spatially distinct and visually calm so users can identify both pane controls without distracting selected-state fill.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The furthest-left Library grip paints two same-direction arrows at approximately 35% and 65% of its height, while the Items grip paints one arrow at its vertical midpoint.
- [x] #2 Expanded and collapsed direction, accessible names and tooltips, keyboard and pointer activation, independent collapse behavior, and five-column geometry remain unchanged.
- [x] #3 Focused, hovered, and pressed grips keep a neutral background without stripes, reverse video, or filled accent treatment; focus remains clearly visible through a theme-aware accent outline or equivalent accent treatment without changing geometry.
- [x] #4 Media, Collections, Conversations, Notes, Prompts, and Skills inherit the same shared grip behavior and styling.
- [x] #5 Targeted mounted paint/compositor coverage and live 160×50, 120×35, 100×30, and 80×24 walkthroughs verify arrow placement, focus styling, containment, and the absence of overflow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add mounted red tests for the approved Library/Items arrow topology and calm focus/active paint across supported terminal heights.
2. Implement height-aware arrow placement in the shared LibraryAdaptiveReaderPaneGrip while preserving its labels, activation contract, and five-column geometry.
3. Centralize neutral rest/hover/pressed and theme-aware focus styling in the shared reader TCSS, then regenerate the canonical CSS bundles.
4. Run targeted shared-shell and Media regressions plus CSS bundle validation.
5. Capture and inspect production-shaped 160×50, 120×35, 100×30, and 80×24 live renders for every approved visual and containment requirement.

ADR required: no
ADR path: N/A
Reason: This is a small visual refinement of the shared shell already governed by ADR-086; it changes no storage, service, runtime, or application-structure boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added height-aware shared grip rendering: the Library control paints proportional
  dual arrows while the Items control remains centered, without changing labels,
  activation, independent collapse behavior, or five-column layout.
- Centralized neutral interaction styling for all six readers. Focus uses
  theme-aware top/bottom endcaps so arrows remain visible without a filled or
  striped selected state.
- Regenerated the canonical CSS bundles and retained production-shaped 160×50,
  120×35, 100×30, and 80×24 PNG/SVG evidence plus its geometry ledger.
- Addressed all four Qodo findings: documented the render return contract,
  centralized the arrow ratio while keeping unit expectations independent, split
  environment bootstrap from ordered runtime imports, and selected a portable
  temporary workspace without import-time filesystem mutation.
- Verified 66 focused shell/Media/capture tests, six production-shaped cross-reader
  resize cases, Ruff, Python compilation, CSS bundle reproduction, and the live
  four-size capture utility. The portability regression was observed failing before
  the bootstrap correction and passing afterward.
- ADR required: no. ADR-086 already governs the shared adaptive reader shell, and
  this refinement changes no architectural boundary.
<!-- SECTION:NOTES:END -->
