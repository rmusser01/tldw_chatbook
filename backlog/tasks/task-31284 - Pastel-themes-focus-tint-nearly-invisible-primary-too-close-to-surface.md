---
id: TASK-31284
title: 'Pastel themes: focus tint nearly invisible (primary too close to surface)'
status: Done
assignee: []
created_date: '2026-09-04 19:03'
updated_date: '2026-09-04 19:09'
labels:
  - themes
  - a11y
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-31264 replaced the frozen slate $ds-focus-bg with the generated primary-at-30% tint. Measured fallout: 10 themes whose primary sits near their surface luminance (pastel_dreams 1.07x, sweet_sorbet 1.02x, kawaii_candy, bunny_fluff, neon_sunset_drive, spring_meadowburst, +4) get a focus background shift under 1.15x — the bold+underline cues in Button:focus carry them, but the non-obscuring focus contract's own mechanism (a visible bg shift, TASK-345) is effectively nullified there, the same failure mode that motivated the original literal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each affected theme's resolved focus tint shifts its composite at least 1.25x against the theme's surface while keeping text on the composite at 4.5:1 or better
- [x] #2 A test pins the focus-tint floor for all registered themes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add all-themes focus-tint test: resolved block-cursor-blurred-background composited over surface must shift >=1.25x with text >=4.5:1 on the composite (red = worklist)\n2. Solver: adjust tint base lightness (30% alpha kept) per failing theme; 8-digit-hex dict override block-cursor-blurred-background\n3. Verify gate green + live spot-check one pastel theme
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
10 weak-tint themes + 3 unreadable-text-on-tint themes fixed by overriding block-cursor-blurred-background per theme (8-digit hex, 30% alpha kept, tint base walked in lightness until composite shifts >=1.30x vs surface AND text >=4.55:1 on the composite). Side effect (intended): Textual's blurred block cursor in the same themes gains the same visibility. New gate test_resolved_focus_tint_is_visible_and_readable_on_every_theme pins a 1.25x shift floor + AA text for all registered themes. Live-verified pastel_dreams: focused button paints #fddacd vs white surface, matching the computed composite.
<!-- SECTION:NOTES:END -->
