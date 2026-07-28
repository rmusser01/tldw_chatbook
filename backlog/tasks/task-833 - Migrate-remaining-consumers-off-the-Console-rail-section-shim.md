---
id: TASK-833
title: Migrate remaining consumers off the Console rail-section shim
status: Done
assignee: []
created_date: '2026-07-26 22:08'
updated_date: '2026-07-27 20:33'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All consumers import DestinationRailSectionHeader / RAIL_SECTION_TOGGLE_PREFIX from Widgets/destination_rail.py directly; Widgets/Console/console_rail_section.py is deleted.

The task named four consumers; chat_screen.py was a fifth. With nothing importing it the shim is deleted rather than deprecated (AC #1 allows either) -- a deprecation note with no consumers is just a file to trip over later. test_console_section_header_is_the_shared_widget goes with it: it existed only to assert the alias resolved, and there is no alias left.

Pure rename, no behaviour change. 407 passed across the rail, Console, Home and Library suites; the one failure is a pre-existing pending_study_initial_section AttributeError, verified identical on clean dev.
<!-- SECTION:NOTES:END -->
