---
id: TASK-1496
title: Wizard keyboard input can land on off-screen widgets silently
status: To Do
assignee: []
created_date: '2026-07-31 00:22'
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
