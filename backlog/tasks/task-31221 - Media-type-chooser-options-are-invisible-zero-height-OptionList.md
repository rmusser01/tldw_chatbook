---
id: TASK-31221
title: Media type chooser - options are invisible (zero-height OptionList)
status: Done
assignee: []
created_date: '2026-09-03 22:30'
updated_date: '2026-09-04 00:31'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique P1: choices.styles.height = min(8, max(1, len(options))) ignores OptionList's 2-row default chrome, so the common 2-option case renders an empty bordered band and selection is blind (verified in code; the Console popup rule documents the exact cost).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every type option is visible for any option count
- [x] #2 The highlighted option is visually indicated
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #2358. Root cause was the app-global '*:focus { outline: solid }' fallback painting OVER the option rows (no geometry cost) - third widget bitten after TASK-1160/TASK-2300; fixed the sanctioned way (outline:none in the library module beside TASK-2300's rationale; highlighted-row recolour carries focus, satisfying AC2). Pinned with PAINTED-TEXT assertions on the production harness - every region assertion measured correct while the paint was covered. Live-verified.
<!-- SECTION:NOTES:END -->
