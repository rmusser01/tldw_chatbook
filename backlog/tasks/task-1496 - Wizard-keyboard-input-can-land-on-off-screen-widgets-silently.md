---
id: TASK-1496
title: Wizard keyboard input can land on off-screen widgets silently
status: Done
assignee: []
created_date: '2026-07-31 00:22'
updated_date: '2026-07-31 01:04'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT: with a local server discovered, Tab from the Provider RadioSet focused the below-fold Use-this-server button; a typed API key went into the void and Protect-keys silently never activated. Focus must scroll targets into view and follow visual order (key input should precede the detected-server button or order made visually logical). Related: TASK-1267.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Focusing any wizard widget scrolls it into view
- [ ] #2 Tab order on the Provider step reaches the key input before any discovery affordance or matches visual order
- [ ] #3 Typing a key after radio selection + Tab is captured (Pilot regression test)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Provider compose reordered: key input+actions precede detected-server banner/button in DOM and visually, so Tab order matches sight order; step region scrolling makes focused widgets scroll into view. Live-verified: typed key renders as dots, Protect-keys activates.
<!-- SECTION:NOTES:END -->
