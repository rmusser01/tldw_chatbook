---
id: TASK-833
title: Migrate remaining consumers off the Console rail-section shim
status: To Do
assignee: []
created_date: '2026-07-26 22:08'
labels:
  - tech-debt
  - cleanup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #940 moved the rail section header into Widgets/destination_rail.py and left Widgets/Console/console_rail_section.py as an identity-alias shim so no consumer had to change. Four consumers still import through it: Widgets/Home/home_rail.py, Widgets/Library/library_rail.py, UI/Screens/home_screen.py, UI/Screens/library_screen.py. Until they move, the claim that the widget was extracted out of Console's private namespace holds only for new callers. The migration is a textual import swap with no behaviour change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All four consumers import DestinationRailSectionHeader from Widgets/destination_rail.py,console_rail_section.py is either deleted or carries an explicit deprecation note with a removal horizon,No behaviour or rendering change in Home, Library, Console, or Personas,Existing rail tests for all four consumers pass unchanged
<!-- AC:END -->
