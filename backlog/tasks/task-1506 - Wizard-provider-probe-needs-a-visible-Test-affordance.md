---
id: TASK-1506
title: Wizard provider probe needs a visible Test affordance
status: Done
assignee: []
created_date: '2026-07-31 00:22'
updated_date: '2026-07-31 02:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT: the live key/endpoint probe only fires on Enter inside the key field — undiscoverable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Test button (or equivalent visible affordance) triggers the probe
- [ ] #2 Probe states (testing/ok/could-not-verify) render adjacent to it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test button shares the key input's row (1495 row budget unchanged), fires the same tokened probe as Enter-in-field; injected-probe test.
<!-- SECTION:NOTES:END -->
