---
id: TASK-66
title: Compact Console chat tab close control
status: Done
assignee:
- '@codex'
labels:
- console
- ux
- ui
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the Console chat tab close affordance renders as a compact x control with a small click buffer instead of inheriting the full chat tab button width.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console chat tab close control renders as a compact x-sized button with small horizontal buffer.
- [x] #2 Close control is visibly narrower than the chat tab label and remains clickable.
- [x] #3 A mounted UI regression verifies the rendered close control width does not regress to the Textual default tab-sized button.
- [x] #4 A real Textual-web screenshot is captured for Console approval.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a mounted Console regression that measures the rendered close-tab button width relative to the chat tab button.
2. Update the chat tab close control styling or widget setup so Textual's default Button min-width no longer controls the close affordance.
3. Run focused Console/tab tests and diff checks.
4. Capture an actual Textual-web Console screenshot showing the compact x control.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Changed the Console chat tab close label from the multiplication glyph to ASCII `x`.
- Constrained the close button width/min-width/max-width to three columns at widget construction time so Textual's default Button minimum cannot expand it to tab width.
- Updated source chat-tab TCSS and rebuilt `tldw_cli_modular.tcss` from the source partial.
- Added a mounted regression asserting the close control is ASCII `x`, compact, and narrower than the chat tab.
- Captured Textual-web evidence at `Docs/superpowers/qa/console-ui/2026-05-21-console-tab-close-compact.png`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
