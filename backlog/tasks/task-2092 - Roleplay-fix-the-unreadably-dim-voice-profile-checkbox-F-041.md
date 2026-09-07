---
id: TASK-2092
title: 'Roleplay: fix the unreadably dim voice-profile checkbox (F-041)'
status: Done
assignee: []
created_date: '2026-08-03 17:25'
updated_date: '2026-08-04 11:49'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Disabled 'Include assigned voice profile' checkbox is so dim it reads as a dark gap in the Inspector. Evidence: roleplay-170x50.png. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Disabled checkbox remains legible as a disabled control,Tests/snapshot updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing style test first: disabled inspector checkbox computes full opacity, ds disabled label color, and a raised-surface glyph box (bundle-loaded app). 2. PersonasInspectorPane DEFAULT_CSS: scoped :disabled rules for #personas-export-include-tts (opacity 100%, label $ds-text-disabled, glyph box $ds-surface-raised with visible glyph color) - widget-level CSS, so no bundle regeneration needed (check_bundle_sync untouched). 3. Verify via computed styles + SVG capture. ADR required: no - presentational CSS only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scoped :disabled rules in PersonasInspectorPane DEFAULT_CSS for #personas-export-include-tts: opacity 100% (Textual base rule dims to 0.7), label color via the theme's text-disabled derivation (matches the ds-text-disabled convention - widget DEFAULT_CSS cannot reference bundle-scoped ds-* names, noted in the CSS comment), glyph box on $surface with the text glyph color. No component TCSS touched, so no bundle regeneration; check_bundle_sync.py still passes. Verification: new computed-style test (opacity/alpha/component background) plus a render probe showing the glyph box + label painted ('▐X▌ Include assigned voice profile'); pane suite 29 passed; ruff clean. ADR: not required (presentational CSS).
<!-- SECTION:NOTES:END -->
