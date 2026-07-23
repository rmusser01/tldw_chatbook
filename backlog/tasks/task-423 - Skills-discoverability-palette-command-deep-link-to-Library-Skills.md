---
id: TASK-423
title: Skills discoverability - palette command deep link to Library Skills
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:19'
updated_date: '2026-07-21 19:55'
labels:
  - skills
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review (verified live). Skills has no nav tab (by design) and no command palette hit: typing 'skills' into Ctrl+P surfaces only a fuzzy 'Switch to Library' match. The only path is knowing to look inside Library's Browse rail. The legacy 'skills' route already resolves to Library so plumbing exists; there is no palette entry or deep link to the Skills rail row. NNG heuristic 6 (recognition rather than recall).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A command palette entry matching 'skills' navigates to Library with the Skills rail row selected,Legacy 'skills' route deep-links land on the Skills row rather than generic Library,Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Gap analysis: the deep-link half already existed - NavigateToScreen('skills') resolves to Library and app._LEGACY_ROUTE_LIBRARY_NAV_CONTEXT['skills'] lands the Skills rail row, covered by an existing test_screen_navigation deep-link test (AC 2 verified, no code needed). The real gap was the palette: typing 'skills' only fuzzy-matched the generic Library command via alias terms, whose action posts plain NavigateToScreen('library') - generic Library, and nothing in the label said why it matched. Added TabNavigationProvider.LIBRARY_SUBROUTE_COMMANDS ((route, command text, help) tuples, currently one entry) rendered by search() as a labeled 'Tab Navigation: Library — Skills' hit matching the command text, help text, and the bare route term; its action posts NavigateToScreen('skills') which rides the existing legacy-context map onto the Skills row. route_for_tab('skills') passes through unaliased. Updated the one-command-per-destination pins (13 destination commands + sub-route deep links = 14; 'no standalone Skills command' assertion now asserts the deep-link label instead - the old pin encoded exactly the decision this task reverses). Palette suites 76 passed, screen navigation 56 passed.
<!-- SECTION:NOTES:END -->
