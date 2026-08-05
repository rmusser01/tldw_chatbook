---
id: TASK-716
title: Library workspace panel loses scroll and disclosure state and explains itself only via tooltips
status: Done
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - library
  - workspaces
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The workspace management surface lives at the bottom of the Library rail's collapsed Details disclosure. After Create local workspace the rail recomposes: scroll returns to top, the disclosure re-collapses, and the Active row that would confirm the action is hidden. The disabled Use in Console button explains its blocked state only via tooltip, and clicking it gives zero feedback. Finding M5; captures cap-20-25.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Disclosure open state and rail scroll position survive a recompose triggered by workspace actions
- [x] #2 After creating a workspace the updated Active row (or an equivalent confirmation) is visible without re-navigating
- [x] #3 A disabled Use in Console press surfaces its reason visibly (not tooltip-only)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Keep Use in Console pressable-but-blocked so the handler's explanatory warning is reachable; dim via a blocked class.
2. Preserve #library-rail scroll across the create-workspace recompose (capture + double-deferred forced restore).
3. Update pinned disabled assertions; add press-explains + scroll-preservation tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two mechanisms found: (a) the blocked Use in Console button was disabled, and a disabled Textual Button never emits Pressed, so the handler's existing explanatory notify (the block reason) was unreachable - the control read as dead with tooltip-only explanation. It now stays pressable with .library-source-action-blocked dim styling; the handler explains inline. (b) create_local_workspace's whole-screen refresh(recompose=True) replaced #library-rail and reset its scroll, hiding the Details disclosure and the updated Active row; _preserve_library_rail_scroll captures scroll_y and restores it after two refresh hops with scroll_to(force=True) (the fresh rail's scroll bounds are not computed on the first hop). UAT-report correction: the "disclosure re-collapsed" claim was unverifiable from the captures - the disclosure open state persists via rail preferences; the loss was the scroll position. _assert_policy_recovery_copy gained pressable_when_blocked (Library opts in; Watchlists/Skills keep disabled semantics). Suites: depth 11 green; destination shells 103 green; test_library_shell failures verified pre-existing/flaky at branch base (same test passes-then-fails on identical consecutive runs at base).
<!-- SECTION:NOTES:END -->
