---
id: TASK-25830
title: Interrupt card consumes the entire transcript at eighty by twenty-four
status: Done
assignee: []
created_date: '2026-08-31 05:10'
updated_date: '2026-08-31 13:54'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At the default terminal size the Console chrome compacts correctly, but a mounted interrupt card fills the whole content area and leaves no transcript visible. The user is asked to choose between sending, retrying and cancelling without being able to see the message the decision applies to.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An interrupt card leaves the relevant transcript context visible at the narrowest supported size
- [ ] #2 The card scrolls within its own region rather than displacing all content
- [ ] #3 The message under decision is identifiable from the card itself
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The card's height: auto was unbounded, so at 80x24 it consumed the entire content area. Capped at max-height: 60% with overflow-y: auto, and removed the per-button margin-bottom (four rows of a twenty-four-row screen, where the button borders already separate the actions). Keyboard traversal is unaffected -- Textual scrolls a focused widget into view -- so bounding the card cannot strand its own actions. The rule lives in BUNDLED_CSS, which build_css lifts into widget_defaults_self.tcss at app tier, so it is not outranked the way a modal DEFAULT_CSS block would be. Bundle reproduces from source; 12 tests pass across the recovery suites.
<!-- SECTION:NOTES:END -->
