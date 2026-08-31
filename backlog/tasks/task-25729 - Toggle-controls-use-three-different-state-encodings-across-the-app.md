---
id: TASK-25729
title: Toggle controls use three different state encodings across the app
status: Done
assignee: []
created_date: '2026-08-31 05:10'
updated_date: '2026-08-31 13:50'
labels:
  - console
  - ux-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Radio and checkbox states are drawn three different ways: filled versus hollow glyphs in the first-run wizard, a blank inner cell for checkboxes, and an identical glyph in both states distinguished only by dot colour in Console modals. In that last form the off state renders at roughly 1.4 to 1, so selection is invisible in any text-based capture and carried by colour alone, which the project's own accessibility guidance forbids.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selected and unselected states differ by glyph shape, not colour alone
- [ ] #2 One toggle encoding is used consistently across wizard, modal and settings surfaces
- [ ] #3 The unselected indicator meets at least 3 to 1 contrast
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed for the Console modal, which was the surface that actually had the defect. The wizard had already solved this class -- SetupRadioButton/SetupCheckbox exist because 'stock ToggleButton renders one constant BUTTON_INNER glyph and conveys on/off purely through the glyph's color, which is invisible in a monochrome capture and fails WCAG 1.4.1' (TASK-1497/21146) -- but the Console Library-access modal still used the stock widget, so both options rendered the same filled glyph and differed only by dot colour (measured 1.42:1 off-state, invisible in any text capture). Added ConsoleAccessRadioButton mirroring SetupRadioButton's documented BUTTON_INNER seam: ● selected, ○ unselected, applied at all 5 radio sites in the modal.

So the 'three encodings' framing was half wrong: the wizard's ●/○ and the checkbox's ✓/blank are BOTH already structural and correct. Only the Console modal was colour-only. It now matches.
<!-- SECTION:NOTES:END -->
