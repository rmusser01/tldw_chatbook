---
id: TASK-2090
title: 'Roleplay: re-anchor the preview conversation affordance (F-039)'
status: Done
assignee: []
created_date: '2026-08-03 17:25'
updated_date: '2026-08-04 11:07'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Preview conversation' bar is stranded at bottom center, detached from the canvas it belongs to; the screen's payoff is two clicks and one expand away. Evidence: roleplay-170x50.png. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preview affordance is visually attached to the character/canvas it previews,Tests/snapshot updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing layout test first: at 170x50 the preview pane sits immediately above the center detail stack (attached to the canvas), not stranded at the work-area bottom; expand/collapse glyph on the toggle. 2. personas_screen compose_content: move PersonasPreviewPane above #personas-detail-stack inside the work area. 3. Preview pane: toggle label gains a collapse glyph (expanded/collapsed state). 4. Visual verification via SVG text-grid capture at 170x50; run preview + layout + parity suites + ruff. ADR required: no - layout placement of an existing widget; no structure/contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved PersonasPreviewPane above #personas-detail-stack inside the work area so the preview toggle anchors the top of the center canvas (flush against the detail stack's top edge, pinned by layout test) instead of a strip stranded at the work-area bottom; toggle gained an expand/collapse glyph (▸/▾) synced across click and the expand() API. Files: tldw_chatbook/UI/Screens/personas_screen.py (compose order), tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py (glyph); test in Tests/UI/test_personas_library_toolbar_layout.py. Visual verification: headless SVG text-grid capture at 170x50 shows the expanded preview heading the center column. Verified: gate 462 passed (layout, preview, full workbench, visual parity); ruff clean. ADR: not required (widget placement).
<!-- SECTION:NOTES:END -->
