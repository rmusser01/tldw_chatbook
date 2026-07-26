---
id: TASK-831
title: Retire the obsolete Personas rail tooltip patch
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
personas_screen.py _sync_personas_rail_tooltips() exists only to overwrite Console's hard-coded 'Open Context rail' tooltip after compose, because ConsoleRailHandle had fixed Console-specific tooltip strings. PR #940 extracted a shared DestinationRailHandle that takes an open_tooltip parameter, which makes the patch redundant. Personas' two handles pass no badge, so they can use the shared base directly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Personas' rail handles construct DestinationRailHandle with explicit open_tooltip values,_sync_personas_rail_tooltips() and its call sites are deleted,Personas' rail handle tooltips read as Library and Inspector rather than Console's Context wording,Existing Personas workbench tests pass unchanged
<!-- AC:END -->
