---
id: TASK-831
title: Retire the obsolete Personas rail tooltip patch
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
personas_screen.py _sync_personas_rail_tooltips() exists only to overwrite Console's hard-coded 'Open Context rail' tooltip after compose, because ConsoleRailHandle had fixed Console-specific tooltip strings. PR #940 extracted a shared DestinationRailHandle that takes an open_tooltip parameter, which makes the patch redundant. Personas' two handles pass no badge, so they can use the shared base directly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Personas' rail handles construct DestinationRailHandle with explicit open_tooltip values,_sync_personas_rail_tooltips() and its call sites are deleted,Personas' rail handle tooltips read as Library and Inspector rather than Console's Context wording,Existing Personas workbench tests pass unchanged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Personas' two rail handles now construct DestinationRailHandle with explicit open_tooltip values ('Open Library rail' / 'Open Inspector rail'), and _sync_personas_rail_tooltips() plus its call site in on_mount are deleted.

The patch existed only because ConsoleRailHandle hard-coded Console's 'Open Context rail' wording, so Personas had to overwrite it after compose. PR #940's shared base takes open_tooltip, which makes the overwrite redundant; Personas passes no badge, so it uses the base directly rather than the Console subclass.

Verified live rather than by construction: driving to the Personas screen and reading the mounted buttons gives tooltip='Open Library rail' and tooltip='Open Inspector rail'. 230 passed across the Personas workbench and destination-rail suites; the single failure, TestImportExport::test_import_failure_shows_recovery_copy, is pre-existing and reproduces identically on a clean dev checkout.
<!-- SECTION:NOTES:END -->
